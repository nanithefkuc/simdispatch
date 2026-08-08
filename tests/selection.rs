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
