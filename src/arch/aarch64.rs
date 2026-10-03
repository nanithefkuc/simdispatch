//! AArch64 scalable-vector capability tokens and the raw intrinsic surface.
//!
//! The module exists only where the `sve` feature, a little-endian AArch64
//! target, and an eligible nightly compiler coincide: `build.rs` defines the
//! `simdispatch_sve` cfg that gates the declaration, and the crate root
//! carries the single `#![feature(stdarch_aarch64_sve)]` gate. On every
//! other build the module is absent and the scalable
//! [`Backend`](crate::Backend) tiers never probe.
//!
//! # Capability tokens
//!
//! [`SveToken`], [`Sve2Token`], and [`Sve2AesToken`] are zero-sized proofs
//! with no public constructor: `summon` mints one only after archmage's
//! public `is_aarch64_feature_available!` macro accepts the exact feature
//! string, keeping host probing single-sourced in `archmage`. Stronger
//! tokens extract already-proven weaker capabilities without probing again
//! (`Sve2AesToken::sve2`, `Sve2Token::sve`); there is no inverse. A token
//! proves instruction availability only: `Sve2Token` proves neither the
//! optional AES and bitperm extensions nor NEON AES, and no SVE token pins
//! a vector length.
//!
//! # Width
//!
//! [`SveToken::vector_bytes`] reports the calling thread's current SVE
//! vector length — a multiple of the 16-byte architectural minimum that
//! may be larger. Vector length is per-thread state: two queries, or two
//! threads, hold no agreement, and the value must not be cached in
//! process-wide state. [`Backend::lane_bytes_on_host`](crate::Backend)
//! routes through this proof; the const
//! [`lane_bytes`](crate::Backend::lane_bytes) answers remain the
//! architectural minimum.
//!
//! # Raw intrinsics
//!
//! The glob re-export below is the raw `core::arch::aarch64` surface for
//! kernel-owning crates, so an SVE kernel file imports one path. The
//! re-export names unstable items: a consumer that *names* them (a type,
//! an intrinsic, a re-export of its own) adds its own crate-root gate
//! `#![cfg_attr(all(target_arch = "aarch64", target_endian = "little"),
//! feature(stdarch_aarch64_sve))]`; calling only the safe token-gated
//! wrappers needs no gate.
//!
//! # Boundaries
//!
//! Every intrinsic — including the width query — is individually
//! `#[target_feature]`-gated, and a function whose signature moves a
//! scalable value by value must itself enable the feature. The safe
//! boundary is therefore slices, pointers, and scalars: scalable values
//! are born and die inside target-feature functions, and generic standard
//! helpers instantiated over scalable types inherit the non-`+sve` context
//! and fail the same way. The width wrapper and its `unsafe`
//! `svcntb_enabled` sibling carry their proofs in place; the crate's residue
//! ledger in `AGENTS.md` enumerates it.

pub use core::arch::aarch64::*;

/// Proof that FEAT_SVE is present on this host.
///
/// Minted by [`summon`](Self::summon) after archmage's
/// `is_aarch64_feature_available!("sve")` accepts the probe. The probe runs
/// whatever detection `archmage` itself carries: this crate's `std` feature
/// gates the resolve pipeline, not the token, and the dependency's own
/// default features stay on regardless.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SveToken {
    _private: (),
}

impl SveToken {
    /// Probe FEAT_SVE through archmage's public macro — the single host
    /// probe in the stack — returning a proof, or `None` off a host (or
    /// build) without SVE.
    pub fn summon() -> Option<Self> {
        archmage::is_aarch64_feature_available!("sve").then(|| SveToken { _private: () })
    }

    /// The calling thread's current SVE vector length in bytes: a multiple
    /// of the 16-byte architectural minimum, up to the 256-byte maximum.
    ///
    /// The token proves FEAT_SVE for the process; the vector length is
    /// per-thread state and may differ between queries and threads, so
    /// callers observing it across a prepared boundary uphold their own
    /// fixed-width contract.
    #[must_use]
    pub fn vector_bytes(self) -> usize {
        svcntb_sve(self)
    }
}

/// Proof that FEAT_SVE2 is present on this host.
///
/// Minted by [`summon`](Self::summon) after archmage's
/// `is_aarch64_feature_available!("sve2")` accepts the probe. This token
/// proves neither the optional AES and bitperm extensions nor NEON AES.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sve2Token {
    _private: (),
}

impl Sve2Token {
    /// Probe FEAT_SVE2 through archmage's public macro, returning a proof,
    /// or `None` on a host (or build) with base SVE at most.
    pub fn summon() -> Option<Self> {
        archmage::is_aarch64_feature_available!("sve2").then(|| Sve2Token { _private: () })
    }

    /// Extract the base-SVE capability this token already proves, without
    /// probing again: FEAT_SVE2 implies FEAT_SVE.
    #[must_use]
    pub fn sve(self) -> SveToken {
        SveToken { _private: () }
    }

    /// The calling thread's current SVE vector length in bytes, through the
    /// base-SVE proof this token already carries
    /// ([`SveToken::vector_bytes`]).
    #[must_use]
    pub fn vector_bytes(self) -> usize {
        self.sve().vector_bytes()
    }
}

/// Proof that the SVE2 AES extension is present on this host.
///
/// Minted by [`summon`](Self::summon) after archmage's
/// `is_aarch64_feature_available!("sve2-aes")` accepts the probe. This is
/// the scalable PMULL128 tier; neither [`Sve2Token`] nor base SVE proves
/// it, and this token proves neither the bitperm extension nor NEON AES.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Sve2AesToken {
    _private: (),
}

impl Sve2AesToken {
    /// Probe the SVE2 AES extension through archmage's public macro,
    /// returning a proof, or `None` on a host (or build) without it.
    pub fn summon() -> Option<Self> {
        archmage::is_aarch64_feature_available!("sve2-aes").then(|| Sve2AesToken { _private: () })
    }

    /// Extract the base-SVE2 capability this token already proves, without
    /// probing again: the AES extension implies FEAT_SVE2.
    #[must_use]
    pub fn sve2(self) -> Sve2Token {
        Sve2Token { _private: () }
    }

    /// The calling thread's current SVE vector length in bytes, through the
    /// base-SVE proof behind the base-SVE2 proof this token already carries
    /// ([`SveToken::vector_bytes`]).
    #[must_use]
    pub fn vector_bytes(self) -> usize {
        self.sve2().sve().vector_bytes()
    }
}

// SINCE: the caller holds `SveToken`, which `summon` minted only after
//        archmage's `is_aarch64_feature_available!("sve")` accepted the
//        probe (compile-time fast path included), so FEAT_SVE is present
//        for this process.
// THUS:  entering the `#[target_feature(enable = "sve")]` sibling is sound;
//        its body is the single `CNTB` query, which reads no memory, writes
//        no state, and needs no ABI accommodation beyond the enabled
//        feature.
#[allow(unsafe_code)]
fn svcntb_sve(_proof: SveToken) -> usize {
    // SINCE: this sibling is private to the proof-gated wrapper above and
    // enables SVE, the feature required by the memory-free width query.
    // THUS: its only call executes with the caller's proven capability.
    #[allow(unsafe_code)]
    #[target_feature(enable = "sve")]
    unsafe fn svcntb_enabled() -> usize {
        core::arch::aarch64::svcntb() as usize
    }
    unsafe { svcntb_enabled() }
}

#[cfg(test)]
mod tests {
    use super::SveToken;

    #[test]
    fn vector_bytes_is_a_granule_multiple() {
        // Meaningful only on an SVE host; off one, summon reports None and
        // there is no width to observe.
        if let Some(token) = SveToken::summon() {
            let bytes = token.vector_bytes();
            assert!(bytes >= 16 && bytes % 16 == 0, "vector bytes {bytes}");
        }
    }
}
