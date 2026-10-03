//! The backend ladder: [`Backend`], its `archmage` tier mapping, and parsing.

use core::fmt;
use core::str::FromStr;

// The capability tokens back `probes_on_host`, the std-only runtime probe.
#[cfg(feature = "std")]
use archmage::{
    NeonAesToken, NeonToken, ScalarToken, SimdToken, Wasm128Token, X64V1Token, X64V2Token,
    X64V3GfniCryptoToken, X64V3Token,
};
#[cfg(all(feature = "std", feature = "avx512"))]
use archmage::{X64V4Token, X64V4xToken};

/// Architectural minimum SVE vector length: 128 bits, the byte-granule
/// floor. The live width is a per-thread query
/// ([`Backend::lane_bytes_on_host`]).
const SVE_MIN_LANE_BYTES: usize = 16;

/// The SIMD capability ladder, one variant per `archmage` tier the ecosystem
/// has kernels for.
///
/// Variants are named after the [`archmage`](https://github.com/imazen/archmage)
/// tier they prove and declared in that tier's dispatch-priority order
/// (strongest first), mirroring `archmage-macros/src/tiers.rs` at the pinned
/// rev. The declaration order therefore *is* the ladder: it carries
/// capability, and it cannot drift from `archmage`'s source of truth without
/// the ordering test failing.
///
/// [`Ord`] follows declaration order with **weaker sorting greater**, matching
/// `fff`'s convention so the downgrade-only override check ports unchanged:
/// `requested >= detected` reads "requested is at most as strong as
/// detected". `min()` over a summoning set is therefore the strongest tier.
///
/// Cross-arch variants never share a host: a wrong-arch token summons `None`,
/// so detection over the full ladder is unambiguous.
///
/// The three scalable AArch64 tiers are local rows: `archmage`'s registry
/// carries no SVE tokens, so they bind the tokens in
/// `simdispatch::arch::aarch64` and carry local dispatch priorities until
/// upstream lands registry tiers; the ordering canary marks them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Backend {
    /// x86 AVX-512 with GFNI and vector crypto (`X64V4xToken`, priority 50).
    /// Detection requires the `avx512` feature.
    V4x,
    /// x86 AVX-512F/BW/CD/DQ/VL (`X64V4Token`, priority 40).
    /// Detection requires the `avx512` feature; this token does not prove GFNI.
    V4,
    /// x86 AVX2 + GFNI + crypto, 32-byte GFNI field multiply
    /// (`X64V3GfniCryptoToken`, `tiers.rs` priority 37).
    V3GfniCrypto,
    /// x86 AVX2, 32-byte split-nibble shuffle (`X64V3Token`, priority 30).
    V3,
    /// x86 SSE4.2, 16-byte split-nibble shuffle (`X64V2Token`, priority 20).
    V2,
    /// x86 SSE2 baseline (`X64V1Token`, priority 10). Always summons on
    /// x86_64; the x86 floor below which only `Scalar` remains.
    V1,
    /// AArch64 SVE2 with the AES extension: the scalable PMULL128 tier
    /// (`Sve2AesToken`, local priority 48). Detection requires the `sve`
    /// feature on an eligible nightly; neither base SVE2 nor NEON AES
    /// proves this extension.
    Sve2Aes,
    /// AArch64 SVE2 (`Sve2Token`, local priority 46): EOR3, TBL2, and the
    /// SVE2 integer operations. Detection requires the `sve` feature on an
    /// eligible nightly; this tier proves neither the AES and bitperm
    /// extensions nor NEON AES.
    Sve2,
    /// AArch64 SVE baseline (`SveToken`, local priority 45). Detection
    /// requires the `sve` feature on an eligible nightly; `lane_bytes()`
    /// reports the architectural minimum, not the live vector length.
    Sve,
    /// AArch64 NEON + AES, 16-byte NEON (`NeonAesToken`, priority 30). The
    /// `aes` feature proves PMULL, so this is the PMULL tier (R4).
    NeonAes,
    /// AArch64 NEON baseline (`NeonToken`, priority 20).
    Neon,
    /// WebAssembly `simd128` (`Wasm128Token`, priority 20).
    Wasm128,
    /// Portable fallback, always present (`ScalarToken`, priority 0).
    Scalar,
}

impl Backend {
    /// Every backend in detection-preference order — the shipped ladder. A
    /// `&'static` slice (not an array) so adding a tier is not a breaking
    /// change to the type.
    pub const ALL: &'static [Backend] = &[
        Backend::V4x,
        Backend::V4,
        Backend::V3GfniCrypto,
        Backend::V3,
        Backend::V2,
        Backend::V1,
        Backend::Sve2Aes,
        Backend::Sve2,
        Backend::Sve,
        Backend::NeonAes,
        Backend::Neon,
        Backend::Wasm128,
        Backend::Scalar,
    ];

    /// Short stable identifier, also the value accepted by the `SIMD_BACKEND`
    /// override and the [`Backend::from_name`] / [`FromStr`] parse set.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Backend::V4x => "v4x",
            Backend::V4 => "v4",
            Backend::V3GfniCrypto => "v3_gfni_crypto",
            Backend::V3 => "v3",
            Backend::V2 => "v2",
            Backend::V1 => "v1",
            Backend::Sve2Aes => "sve2_aes",
            Backend::Sve2 => "sve2",
            Backend::Sve => "sve",
            Backend::NeonAes => "neon_aes",
            Backend::Neon => "neon",
            Backend::Wasm128 => "wasm128",
            Backend::Scalar => "scalar",
        }
    }

    /// Parse a backend name, as accepted by the `SIMD_BACKEND` override.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Backend> {
        Some(match name {
            "v4x" => Backend::V4x,
            "v4" => Backend::V4,
            "v3_gfni_crypto" => Backend::V3GfniCrypto,
            "v3" => Backend::V3,
            "v2" => Backend::V2,
            "v1" => Backend::V1,
            "sve2_aes" => Backend::Sve2Aes,
            "sve2" => Backend::Sve2,
            "sve" => Backend::Sve,
            "neon_aes" => Backend::NeonAes,
            "neon" => Backend::Neon,
            "wasm128" => Backend::Wasm128,
            "scalar" => Backend::Scalar,
            _ => return None,
        })
    }

    /// Architectural vector width in bytes per tier. Consumers that derive
    /// buffer geometry from the resolved backend import this; they never
    /// re-derive it. The scalable AArch64 tiers report the architectural
    /// minimum — 16 bytes — not the live vector length; the current width
    /// is [`lane_bytes_on_host`](Self::lane_bytes_on_host).
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        match self {
            Backend::V4x | Backend::V4 => 64,
            Backend::V3GfniCrypto | Backend::V3 => 32,
            Backend::V2
            | Backend::V1
            | Backend::Sve2Aes
            | Backend::Sve2
            | Backend::Sve
            | Backend::NeonAes
            | Backend::Neon
            | Backend::Wasm128 => 16,
            Backend::Scalar => 8,
        }
    }

    /// The vector width in bytes this tier presents on the current host and
    /// thread.
    ///
    /// Fixed-width tiers equal [`lane_bytes`](Self::lane_bytes). The
    /// scalable AArch64 tiers report the calling thread's current SVE
    /// vector length — a multiple of the 16-byte architectural minimum that
    /// may be larger — but only after an SVE proof summons; without one
    /// (builds without the compiled surface, non-AArch64 hosts, SVE-less
    /// hosts, or `no_std` builds without detection) they report that
    /// minimum rather than executing SVE. Vector length is per-thread
    /// state: callers hold no expectation that two queries or two threads
    /// agree, and must not cache the value in process-wide state.
    #[must_use]
    pub fn lane_bytes_on_host(self) -> usize {
        match self {
            #[cfg(simdispatch_sve)]
            Backend::Sve2Aes | Backend::Sve2 | Backend::Sve => {
                crate::arch::aarch64::SveToken::summon().map_or(
                    SVE_MIN_LANE_BYTES,
                    crate::arch::aarch64::SveToken::vector_bytes,
                )
            }
            #[cfg(not(simdispatch_sve))]
            Backend::Sve2Aes | Backend::Sve2 | Backend::Sve => SVE_MIN_LANE_BYTES,
            _ => self.lane_bytes(),
        }
    }

    /// Whether this tier's token summons on the current host — the one probe
    /// in the stack: `archmage::SimdToken::summon` for registry tokens,
    /// archmage's public `is_aarch64_feature_available!` macro for the
    /// local SVE tokens. A tier is "on the host" exactly when this is true.
    ///
    /// Runtime summoning needs the `std` feature; without it the crate
    /// reports [`Backend::Scalar`] and never probes.
    #[cfg(feature = "std")]
    #[must_use]
    pub(crate) fn probes_on_host(self) -> bool {
        match self {
            #[cfg(feature = "avx512")]
            Backend::V4x => X64V4xToken::summon().is_some(),
            #[cfg(not(feature = "avx512"))]
            Backend::V4x => false,
            #[cfg(feature = "avx512")]
            Backend::V4 => X64V4Token::summon().is_some(),
            #[cfg(not(feature = "avx512"))]
            Backend::V4 => false,
            Backend::V3GfniCrypto => X64V3GfniCryptoToken::summon().is_some(),
            Backend::V3 => X64V3Token::summon().is_some(),
            Backend::V2 => X64V2Token::summon().is_some(),
            Backend::V1 => X64V1Token::summon().is_some(),
            #[cfg(simdispatch_sve)]
            Backend::Sve2Aes => crate::arch::aarch64::Sve2AesToken::summon().is_some(),
            #[cfg(not(simdispatch_sve))]
            Backend::Sve2Aes => false,
            #[cfg(simdispatch_sve)]
            Backend::Sve2 => crate::arch::aarch64::Sve2Token::summon().is_some(),
            #[cfg(not(simdispatch_sve))]
            Backend::Sve2 => false,
            #[cfg(simdispatch_sve)]
            Backend::Sve => crate::arch::aarch64::SveToken::summon().is_some(),
            #[cfg(not(simdispatch_sve))]
            Backend::Sve => false,
            Backend::NeonAes => NeonAesToken::summon().is_some(),
            Backend::Neon => NeonToken::summon().is_some(),
            Backend::Wasm128 => Wasm128Token::summon().is_some(),
            // ScalarToken always summons; kept as a summon call so every
            // variant binds the token that proves it, uniformly.
            Backend::Scalar => ScalarToken::summon().is_some(),
        }
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Error returned when a [`Backend`] name is not recognized.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ParseBackendError;

impl fmt::Display for ParseBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("unknown SIMD backend name")
    }
}

impl core::error::Error for ParseBackendError {}

impl FromStr for Backend {
    type Err = ParseBackendError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        Backend::from_name(name).ok_or(ParseBackendError)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mirrors the `archmage` tier priorities at the pinned rev
    /// (`archmage-macros/src/tiers.rs`). This table is the ordering canary:
    /// it must be regenerated on every `archmage` rev bump, and the ladder
    /// must be reconciled with upstream rather than reordered locally.
    const TIERS_RS_MIRROR: &[(Backend, u32)] = &[
        (Backend::V4x, 50),
        (Backend::V4, 40),
        (Backend::V3GfniCrypto, 37),
        (Backend::V3, 30),
        (Backend::V2, 20),
        (Backend::V1, 10),
        // Local rows, not `tiers.rs` entries: the registry carries no SVE
        // tokens, so the scalable tiers bind `simdispatch::arch::aarch64`
        // tokens and local dispatch priorities between the x86 and NEON
        // families. An upstream SVE tier lands at its archmage priority and
        // these rows are reconciled then.
        (Backend::Sve2Aes, 48),
        (Backend::Sve2, 46),
        (Backend::Sve, 45),
        (Backend::NeonAes, 30),
        (Backend::Neon, 20),
        (Backend::Wasm128, 20),
        (Backend::Scalar, 0),
    ];

    #[test]
    fn all_pins_variant_order_to_tiers_rs() {
        // The mirror and the ladder must line up exactly (same length, same
        // order) or this fails — adding a tier without updating the mirror is
        // a build error of the canary.
        let mut mirrored_order = [Backend::Scalar; 13];
        for (i, &(backend, _)) in TIERS_RS_MIRROR.iter().enumerate() {
            mirrored_order[i] = backend;
        }
        assert_eq!(Backend::ALL, mirrored_order.as_slice());
    }

    #[test]
    fn order_encodes_capability_within_family() {
        // Stronger sorts less; weaker sorts greater.
        assert!(Backend::V4x < Backend::V4);
        assert!(Backend::V4 < Backend::V3GfniCrypto);
        assert!(Backend::V3GfniCrypto < Backend::V3);
        assert!(Backend::V3 < Backend::V2);
        assert!(Backend::V2 < Backend::V1);
        assert!(Backend::NeonAes < Backend::Neon);
        assert!(Backend::Wasm128 < Backend::Scalar);
        assert!(Backend::Neon < Backend::Wasm128);
        // The downgrade check reads `requested >= detected` — weaker-or-equal.
        assert!(Backend::V3GfniCrypto < Backend::Scalar);
        // The shipped ladder concatenates family ladders: x86 above ARM.
        assert!(Backend::V1 < Backend::NeonAes);
        // The scalable family sits between the x86 and NEON ladders.
        assert!(Backend::V1 < Backend::Sve2Aes);
        assert!(Backend::Sve2Aes < Backend::Sve2);
        assert!(Backend::Sve2 < Backend::Sve);
        assert!(Backend::Sve < Backend::NeonAes);
    }

    #[test]
    fn lane_bytes_are_architectural() {
        assert_eq!(Backend::V4x.lane_bytes(), 64);
        assert_eq!(Backend::V4.lane_bytes(), 64);
        assert_eq!(Backend::V3GfniCrypto.lane_bytes(), 32);
        assert_eq!(Backend::V3.lane_bytes(), 32);
        assert_eq!(Backend::V2.lane_bytes(), 16);
        assert_eq!(Backend::V1.lane_bytes(), 16);
        // The scalable minimum, not the live vector length.
        assert_eq!(Backend::Sve2Aes.lane_bytes(), 16);
        assert_eq!(Backend::Sve2.lane_bytes(), 16);
        assert_eq!(Backend::Sve.lane_bytes(), 16);
        assert_eq!(Backend::NeonAes.lane_bytes(), 16);
        assert_eq!(Backend::Neon.lane_bytes(), 16);
        assert_eq!(Backend::Wasm128.lane_bytes(), 16);
        assert_eq!(Backend::Scalar.lane_bytes(), 8);
    }

    #[test]
    fn lane_bytes_on_host_is_at_least_the_architectural_width() {
        for &backend in Backend::ALL {
            assert!(backend.lane_bytes_on_host() >= backend.lane_bytes());
        }
    }

    #[test]
    #[cfg(not(simdispatch_sve))]
    fn scalable_tiers_report_the_minimum_width_without_the_compiled_surface() {
        // Without the SVE surface there is no proof to query, so the
        // scalable tiers cannot execute SVE and report the minimum.
        assert_eq!(Backend::Sve2Aes.lane_bytes_on_host(), 16);
        assert_eq!(Backend::Sve2.lane_bytes_on_host(), 16);
        assert_eq!(Backend::Sve.lane_bytes_on_host(), 16);
    }

    #[test]
    fn names_roundtrip() {
        for &backend in Backend::ALL {
            assert_eq!(Backend::from_name(backend.name()), Some(backend));
            assert_eq!(backend.name().parse::<Backend>(), Ok(backend));
        }
    }

    /// Stack-only `core::fmt::Write` sink: this crate is `#![no_std]` (no
    /// `alloc`), so `to_string()` is unavailable even in unit tests. Longest
    /// backend name is "v3_gfni_crypto" (13 bytes).
    struct FmtBuf {
        bytes: [u8; 24],
        len: usize,
    }

    impl FmtBuf {
        fn new() -> Self {
            FmtBuf {
                bytes: [0; 24],
                len: 0,
            }
        }
    }

    impl core::fmt::Write for FmtBuf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            for byte in s.bytes() {
                self.bytes[self.len] = byte;
                self.len += 1;
            }
            Ok(())
        }
    }

    #[test]
    fn display_matches_name() {
        for &backend in Backend::ALL {
            let mut buf = FmtBuf::new();
            use core::fmt::Write;
            write!(&mut buf, "{backend}").unwrap();
            let rendered = core::str::from_utf8(&buf.bytes[..buf.len]).unwrap();
            assert_eq!(rendered, backend.name());
        }
    }

    #[test]
    fn unknown_and_historical_names_are_rejected() {
        // `avx2` was fff's semantic label; tiers are named after the archmage
        // tier they prove, so the historical name is not a valid identifier.
        assert_eq!(Backend::from_name("avx2"), None);
        assert_eq!(Backend::from_name("gfni"), None);
        assert_eq!(Backend::from_name("pmull"), None);
        assert_eq!(Backend::from_name("neon_aesx"), None);
        // The architecture feature string is hyphenated; the override name
        // is not.
        assert_eq!(Backend::from_name("sve2-aes"), None);
        assert_eq!("bogus".parse::<Backend>(), Err(ParseBackendError));
    }

    // Double-check compile-time-proof tier summons where the platform forces
    // it. Probing is the std-only runtime path, so these are std-gated too.
    #[test]
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    fn v1_always_summons_on_x86_64() {
        assert!(Backend::V1.probes_on_host());
    }

    #[test]
    #[cfg(all(feature = "std", not(feature = "avx512")))]
    fn avx512_tiers_do_not_summon_without_avx512() {
        assert!(!Backend::V4x.probes_on_host());
        assert!(!Backend::V4.probes_on_host());
    }

    #[test]
    #[cfg(all(feature = "std", not(simdispatch_sve)))]
    fn sve_tiers_do_not_summon_without_the_compiled_surface() {
        assert!(!Backend::Sve2Aes.probes_on_host());
        assert!(!Backend::Sve2.probes_on_host());
        assert!(!Backend::Sve.probes_on_host());
    }

    #[test]
    #[cfg(all(feature = "std", feature = "avx512"))]
    fn avx512_probes_match_archmage_tokens() {
        assert_eq!(
            Backend::V4x.probes_on_host(),
            X64V4xToken::summon().is_some()
        );
        assert_eq!(Backend::V4.probes_on_host(), X64V4Token::summon().is_some());
    }

    #[test]
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    fn neon_always_summons_on_aarch64() {
        assert!(Backend::Neon.probes_on_host());
    }
}
