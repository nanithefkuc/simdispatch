//! `SIMD_BACKEND` override behavior, exercised against the *real*
//! environment variable.
//!
//! This file runs in its own process (cargo compiles each integration-test
//! file separately) and its single [`#[test]`] is the only mutator of the
//! variable, so the Rust-2024 `unsafe` on [`set_var`](std::env::set_var) /
//! [`remove_var`](std::env::remove_var) cannot race a concurrent `env::var`
//! reader — there is none in this process.
//!
//! The test first normalizes the variable away so an ambient
//! `SIMD_BACKEND` from the outer environment (e.g. the ladder sweep) cannot
//! skew the assertions: the point is the overrides this test itself sets.

use simdispatch::{Backend, Selection};

fn resolve_all() -> Backend {
    Selection::new("SIMD_BACKEND")
        .supports(Backend::ALL)
        .resolve()
}

#[test]
fn simd_backend_override_is_downgrade_only() {
    // Normalize: start from no override so the assertions are about the
    // overrides this test sets, not an ambient one.
    // SAFETY: single-threaded test process; no concurrent env::var readers.
    unsafe { std::env::remove_var("SIMD_BACKEND") };

    // Baseline: host detection with no override.
    let baseline = resolve_all();

    // Unknown values are ignored: resolution is exactly the baseline.
    // SAFETY: single-threaded test process; no concurrent env::var readers.
    unsafe { std::env::set_var("SIMD_BACKEND", "not_a_backend") };
    assert_eq!(resolve_all(), baseline);

    // `scalar` forces scalar on every host — the escape hatch.
    // SAFETY: single-threaded test process; no concurrent env::var readers.
    unsafe { std::env::set_var("SIMD_BACKEND", "scalar") };
    assert_eq!(resolve_all(), Backend::Scalar);

    // Host-true downgrade proof, only on the arch that can assert it
    // deterministically. The `feature = "std"` guard keeps this out of no_std
    // runs, where resolution reports Scalar and no env is read.
    #[cfg(all(target_arch = "x86_64", feature = "std"))]
    {
        // V1 (SSE2) always summons on x86_64 and is the weakest x86 tier, so
        // a request for it resolves exactly to V1: a real, honored downgrade.
        // SAFETY: single-threaded test process; no concurrent env::var read.
        unsafe { std::env::set_var("SIMD_BACKEND", "v1") };
        assert_eq!(resolve_all(), Backend::V1);
    }

    // Clean up so nothing else in this process sees a stale value.
    // SAFETY: single-threaded test process; no concurrent env::var readers.
    unsafe { std::env::remove_var("SIMD_BACKEND") };
}
