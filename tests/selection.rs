//! Host-true selection invariants.
//!
//! Every assertion here holds on any host regardless of what it detects, and
//! in both `std` and `no_std` builds. Environment mutation is deliberately
//! *not* done in this file — that is `tests/env_override.rs`, a separate
//! process so the two never race on `std::env`.

use simdispatch::{Backend, Selection};

fn resolve_supported(supported: &'static [Backend]) -> Backend {
    Selection::new("SIMD_BACKEND").supports(supported).resolve()
}

fn resolve_full_ladder() -> Backend {
    Selection::new("SIMD_BACKEND")
        .supports(Backend::ALL)
        .resolve()
}

#[test]
fn full_ladder_resolves_to_a_member() {
    let backend = resolve_full_ladder();
    assert!(Backend::ALL.contains(&backend));
}

#[test]
fn empty_supported_set_resolves_to_scalar() {
    assert_eq!(resolve_supported(&[]), Backend::Scalar);
}

#[test]
fn scalar_only_set_resolves_to_scalar() {
    assert_eq!(resolve_supported(&[Backend::Scalar]), Backend::Scalar);
}

#[test]
fn subset_result_is_supported_or_scalar() {
    // A consumer's resolved backend is always one of the backends it
    // implements, or Scalar when none of them can run. Never a tier the
    // consumer did not declare.
    let subsets: &[&[Backend]] = &[
        &[Backend::V4x, Backend::V4, Backend::V3GfniCrypto],
        &[Backend::V4x, Backend::V3GfniCrypto, Backend::Scalar],
        &[Backend::V2, Backend::V1],
        &[Backend::V3GfniCrypto, Backend::V3],
        &[Backend::NeonAes, Backend::Neon],
        &[Backend::V2, Backend::Neon, Backend::Scalar],
        &[Backend::V3, Backend::Scalar],
    ];
    for &set in subsets {
        let resolved = resolve_supported(set);
        assert!(
            set.contains(&resolved) || resolved == Backend::Scalar,
            "resolved {resolved:?} for supported {set:?}"
        );
    }
}

#[test]
#[cfg(all(feature = "std", target_arch = "x86_64"))]
fn avx512_selection_respects_archmage_proofs_and_supported_set() {
    use archmage::{SimdToken, X64V4Token, X64V4xToken};

    let has_v4x = X64V4xToken::summon().is_some();
    let has_v4 = X64V4Token::summon().is_some();
    let selected = resolve_supported(&[Backend::V4x, Backend::V4, Backend::V3GfniCrypto]);
    let requested = std::env::var("SIMD_BACKEND").ok();
    if cfg!(feature = "avx512") && std::env::var_os("SIMDISPATCH_REQUIRE_V4X").is_some() {
        assert!(has_v4x, "V4x token did not summon on the required host");
        assert_eq!(selected, Backend::V4x);
        assert!(has_v4, "V4 token did not summon on the required host");
    }
    if !cfg!(feature = "avx512") || !has_v4x {
        assert_ne!(selected, Backend::V4x);
    } else if requested.is_none() {
        assert_eq!(selected, Backend::V4x);
    }
    if !cfg!(feature = "avx512") || !has_v4 {
        assert_ne!(selected, Backend::V4);
    } else if !has_v4x && requested.is_none() {
        assert_eq!(selected, Backend::V4);
    }
    if cfg!(feature = "avx512") && has_v4 && requested.is_none() {
        assert_eq!(
            resolve_supported(&[Backend::V4, Backend::V3GfniCrypto]),
            Backend::V4
        );
    }
    if requested.as_deref() == Some("v4") && cfg!(feature = "avx512") && has_v4 {
        assert_eq!(selected, Backend::V4);
    }
    assert_ne!(
        resolve_supported(&[Backend::V4, Backend::V3GfniCrypto]),
        Backend::V4x
    );
    assert_ne!(resolve_supported(&[Backend::V3GfniCrypto]), Backend::V4x);
}

#[test]
fn same_supported_set_agrees_across_resolutions() {
    // The cross-crate agreement invariant: resolution is a pure function of
    // (supported set, host, override), so re-resolving the same set — as two
    // consumers would — always yields the same backend. Re-probing per
    // consumer cannot diverge (the cafft rot this crate exists to delete).
    let set: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
    let first = resolve_supported(set);
    let second = resolve_supported(set);
    assert_eq!(first, second);
}

#[test]
#[cfg(feature = "internals")]
fn internals_resolve_with_is_stable_across_calls() {
    // Same deterministic core, directly — verifies the injectable pathway the
    // production resolve() is built on is deterministic too.
    let on_host = |b: Backend| matches!(b, Backend::V3GfniCrypto | Backend::V3 | Backend::V2);
    let set: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
    let a = simdispatch::internals::resolve_with(set, on_host, None);
    let b = simdispatch::internals::resolve_with(set, on_host, None);
    assert_eq!(a, b);
    assert_eq!(a, Backend::V3);
}
