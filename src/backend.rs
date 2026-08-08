//! The backend ladder: [`Backend`], its `archmage` tier mapping, and parsing.

use core::fmt;
use core::str::FromStr;

// The capability tokens back `probes_on_host`, the std-only runtime probe.
#[cfg(feature = "std")]
use archmage::{
    NeonAesToken, NeonToken, ScalarToken, SimdToken, Wasm128Token, X64V1Token, X64V2Token,
    X64V3GfniCryptoToken, X64V3Token,
};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Backend {
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
    /// change to the type, per the `fff/.plans` decision.
    pub const ALL: &'static [Backend] = &[
        Backend::V3GfniCrypto,
        Backend::V3,
        Backend::V2,
        Backend::V1,
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
            Backend::V3GfniCrypto => "v3_gfni_crypto",
            Backend::V3 => "v3",
            Backend::V2 => "v2",
            Backend::V1 => "v1",
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
            "v3_gfni_crypto" => Backend::V3GfniCrypto,
            "v3" => Backend::V3,
            "v2" => Backend::V2,
            "v1" => Backend::V1,
            "neon_aes" => Backend::NeonAes,
            "neon" => Backend::Neon,
            "wasm128" => Backend::Wasm128,
            "scalar" => Backend::Scalar,
            _ => return None,
        })
    }

    /// Architectural vector width in bytes per tier. Consumers that derive
    /// buffer geometry from the resolved backend import this; they never
    /// re-derive it.
    #[must_use]
    pub const fn lane_bytes(self) -> usize {
        match self {
            Backend::V3GfniCrypto | Backend::V3 => 32,
            Backend::V2 | Backend::V1 | Backend::NeonAes | Backend::Neon | Backend::Wasm128 => 16,
            Backend::Scalar => 8,
        }
    }

    /// Whether this tier's `archmage` token summons on the current host —
    /// the one probe in the stack (`archmage::SimdToken::summon`). A tier is
    /// "on the host" exactly when this is true.
    ///
    /// Runtime summoning needs the `std` feature; without it the crate
    /// reports [`Backend::Scalar`] and never probes.
    #[cfg(feature = "std")]
    #[must_use]
    pub(crate) fn probes_on_host(self) -> bool {
        match self {
            Backend::V3GfniCrypto => X64V3GfniCryptoToken::summon().is_some(),
            Backend::V3 => X64V3Token::summon().is_some(),
            Backend::V2 => X64V2Token::summon().is_some(),
            Backend::V1 => X64V1Token::summon().is_some(),
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
        (Backend::V3GfniCrypto, 37),
        (Backend::V3, 30),
        (Backend::V2, 20),
        (Backend::V1, 10),
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
        let mut mirrored_order = [Backend::Scalar; 8];
        for (i, &(backend, _)) in TIERS_RS_MIRROR.iter().enumerate() {
            mirrored_order[i] = backend;
        }
        assert_eq!(Backend::ALL, mirrored_order.as_slice());
    }

    #[test]
    fn order_encodes_capability_within_family() {
        // Stronger sorts less; weaker sorts greater.
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
    }

    #[test]
    fn lane_bytes_are_architectural() {
        assert_eq!(Backend::V3GfniCrypto.lane_bytes(), 32);
        assert_eq!(Backend::V3.lane_bytes(), 32);
        assert_eq!(Backend::V2.lane_bytes(), 16);
        assert_eq!(Backend::V1.lane_bytes(), 16);
        assert_eq!(Backend::NeonAes.lane_bytes(), 16);
        assert_eq!(Backend::Neon.lane_bytes(), 16);
        assert_eq!(Backend::Wasm128.lane_bytes(), 16);
        assert_eq!(Backend::Scalar.lane_bytes(), 8);
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
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    fn neon_always_summons_on_aarch64() {
        assert!(Backend::Neon.probes_on_host());
    }
}
