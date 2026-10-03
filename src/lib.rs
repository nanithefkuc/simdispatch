//! # simdispatch — SIMD backend selection for the FEC stack
//!
//! [`simdispatch`](crate) is the Level 0 crate: the single source for SIMD
//! capability selection across the FEC ecosystem. Everything else that ships
//! vector kernels — `fff`, `cafft`, and every future consumer — gets its
//! backend from here.
//!
//! Capability proof comes from [`archmage`](https://github.com/imazen/archmage)
//! capability tokens: a tier is "on the host" exactly when its token
//! [`summon`](archmage::SimdToken::summon)s. There is one probe in the whole
//! stack, and it lives here.
//!
//! ## The two primitives
//!
//! - [`Backend`] — the ladder. Variants are named after the `archmage` tier
//!   they prove and ordered by that tier's dispatch priority (strongest
//!   first), so the order carries capability and cannot drift from
//!   `archmage`'s source of truth.
//! - [`Selection`] — a consumer declares the backends it implements
//!   ([`Selection::supports`]) and resolves the choice for the current host:
//!   detected via `summon`, narrowed to the supported set, then adjusted by
//!   the **downgrade-only** `SIMD_BACKEND` override. Refusing to upgrade is a
//!   soundness property: running vector code the CPU cannot execute is
//!   undefined behaviour, not a preference.
//!
//! [`backend()`] reports the process-wide resolution of the full ladder
//! ([`Backend::ALL`]); a consumer with a narrower kernel set builds its own
//! [`Selection`] instead.
//!
//! ## `SIMD_BACKEND`
//!
//! The one stack-wide override, replacing the per-crate `FFF_BACKEND` /
//! `CAFFT_BACKEND` as crates migrate. Accepted values are
//! [`Backend::name()`] values (`scalar`, `v1`, `v2`, `v3`, `v3_gfni_crypto`, `v4`, `v4x`,
//! `neon`, `neon_aes`, `wasm128`, `sve2_aes`, `sve2`, `sve`). It is **downgrade-only**: a
//! request for a backend the host cannot run, or one stronger than what
//! detection found, is ignored. The `v4` and `v4x` tiers are available with
//! the `avx512` feature; the `sve2_aes`, `sve2`, and `sve` tiers additionally
//! need the `sve` feature on an eligible nightly AArch64 build — elsewhere
//! they stay recognized names that never summon.
//! `SIMD_BACKEND=scalar` forces the whole stack to portable code —
//! the escape hatch operators and differential testing need.
//!
//! ## `no_std`
//!
//! The crate is `#![no_std]`. Without the `std` feature there is no
//! environment and no runtime detection, so [`backend()`] and any
//! [`Selection::resolve`] report [`Backend::Scalar`] unconditionally —
//! matching how the field kernels behave on a std-less build.
//!
//! ## Non-scope
//!
//! The default build ships no kernels, no intrinsics, no dispatch tables,
//! and no `unsafe`; it answers which consumer `#[target_feature]` function
//! is legal to call and never touches kernel bodies. The off-by-default
//! `sve` feature is the one scoped exception: on an eligible nightly
//! AArch64 build it adds the scalable-vector capability tokens and the raw
//! `core::arch::aarch64` re-export at `simdispatch::arch::aarch64`, with
//! the `unsafe` residue enumerated in the crate's residue ledger.

#![no_std]
#![deny(missing_docs)]
#![deny(unsafe_code)]
// The single library-feature gate for the SVE intrinsic surface. build.rs
// defines `simdispatch_sve` only when the `sve` feature meets a
// little-endian AArch64 target and an eligible nightly compiler, so this
// attribute never reaches a build that cannot carry it.
#![cfg_attr(simdispatch_sve, feature(stdarch_aarch64_sve))]

#[cfg(feature = "std")]
extern crate std;

#[cfg(simdispatch_sve)]
pub mod arch;
mod backend;
mod selection;
pub use backend::{Backend, ParseBackendError};
pub use selection::Selection;

#[cfg(feature = "internals")]
pub use selection::internals;

#[cfg(feature = "std")]
use std::sync::LazyLock;

/// The process-wide resolution of the full ladder [`Backend::ALL`], narrowed
/// to nothing, then adjusted by the `SIMD_BACKEND` override.
#[cfg(feature = "std")]
static BACKEND: LazyLock<Backend> = LazyLock::new(|| {
    Selection::new("SIMD_BACKEND")
        .supports(Backend::ALL)
        .resolve()
});

/// The backend the full ecosystem ladder resolves to on this host, detected
/// once per process.
///
/// This is the *un-narrowed* answer: the strongest [`Backend`] a host that
/// implemented every tier could use. A consumer that implements only a subset
/// must resolve its own [`Selection`] rather than reuse this value.
#[cfg(feature = "std")]
#[must_use]
pub fn backend() -> Backend {
    *BACKEND
}

/// The backend the full ecosystem ladder resolves to on this host.
///
/// Without `std` there is no runtime detection and no environment override,
/// so this reports [`Backend::Scalar`] unconditionally.
#[cfg(not(feature = "std"))]
#[must_use]
pub fn backend() -> Backend {
    Backend::Scalar
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_reports_a_member_of_the_ladder() {
        let backend = backend();
        assert!(
            Backend::ALL.contains(&backend),
            "backend() returned {backend:?}, not in the ladder"
        );
    }

    #[cfg(not(feature = "std"))]
    #[test]
    fn no_std_reports_scalar() {
        assert_eq!(backend(), Backend::Scalar);
        let selected = Selection::new("SIMD_BACKEND")
            .supports(Backend::ALL)
            .resolve();
        assert_eq!(selected, Backend::Scalar);
    }

    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    #[test]
    fn x86_full_ladder_floor_is_v1_without_a_scalar_override() {
        // V1 (SSE2) always summons on x86_64, so the resolved full ladder must
        // be an x86 tier — unless `SIMD_BACKEND=scalar` forced the whole
        // process to portable code, which is exactly the override's job.
        let backend = backend();
        if std::env::var("SIMD_BACKEND")
            .ok()
            .is_some_and(|v| v == "scalar")
        {
            assert_eq!(backend, Backend::Scalar);
        } else {
            assert!(
                matches!(
                    backend,
                    Backend::V4x
                        | Backend::V4
                        | Backend::V3GfniCrypto
                        | Backend::V3
                        | Backend::V2
                        | Backend::V1
                ),
                "full ladder resolved to {backend:?} on x86_64"
            );
        }
    }

    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    #[test]
    fn aarch64_full_ladder_floor_is_neon_without_a_scalar_override() {
        let backend = backend();
        if std::env::var("SIMD_BACKEND")
            .ok()
            .is_some_and(|v| v == "scalar")
        {
            assert_eq!(backend, Backend::Scalar);
        } else {
            assert!(
                matches!(
                    backend,
                    Backend::Sve2Aes
                        | Backend::Sve2
                        | Backend::Sve
                        | Backend::NeonAes
                        | Backend::Neon
                ),
                "full ladder resolved to {backend:?} on aarch64"
            );
        }
    }
}
