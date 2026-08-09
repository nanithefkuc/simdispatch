//! The selection primitive: a consumer declares the backends it implements
//! and resolves the choice for the current host.

use crate::backend::Backend;

/// A resolve pipeline: host detection via `archmage` `summon()`, narrowed to
/// a consumer's supported set, then adjusted by a **downgrade-only**
/// environment override.
///
/// Build one per consumer, feeding it the [`Backend`] set that consumer
/// actually implements:
///
/// ```
/// use simdispatch::{Backend, Selection};
///
/// // The kernel set this crate ships (example).
/// const KERNELS: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
///
/// let chosen = Selection::new("SIMD_BACKEND").supports(KERNELS).resolve();
/// ```
///
/// `resolve()` is deterministic for a fixed host and supported set — it is a
/// pure function of the `summon()` results and the environment override, so
/// two consumers resolving the same set on the same host always agree.
#[derive(Clone, Copy)]
pub struct Selection {
    // Read only by the std resolve path; without std there is no environment
    // override to consult, so the field is an API-shape placeholder there.
    #[cfg_attr(not(feature = "std"), allow(dead_code))]
    override_var: &'static str,
    supported: &'static [Backend],
}

impl Selection {
    /// Start a selection, reading the override from the named environment
    /// variable. The ecosystem override is `SIMD_BACKEND`; the parameter
    /// exists so the resolution pipeline is testable in isolation.
    #[must_use]
    pub const fn new(override_var: &'static str) -> Self {
        Selection {
            override_var,
            supported: &[],
        }
    }

    /// Declare the backends this consumer implements. Order is irrelevant:
    /// capability order comes from the [`Backend`] enum, host proof from
    /// `summon()`.
    #[must_use]
    pub const fn supports(mut self, supported: &'static [Backend]) -> Self {
        self.supported = supported;
        self
    }

    /// Resolve the backend for this host and environment.
    ///
    /// 1. **Detect**: the strongest supported tier whose `archmage` token
    ///    summons on this host. If none does — or the supported set is empty
    ///    — this is [`Backend::Scalar`].
    /// 2. **Override, downgrade-only**: accept the override only when it
    ///    names a supported backend **that itself summons on this host**
    ///    (`archmage::SimdToken::summon`, the same arch-rooted probe as
    ///    detection) and is at most as strong as the detected one; otherwise
    ///    keep the detected value. A request for a backend the host cannot
    ///    run — a different arch (`neon` on x86) or a tier the host lacks
    ///    (`v2` on a V1-only host) — does not summon and is ignored.
    ///
    /// Without the `std` feature there is no environment and no runtime
    /// detection, so this reports [`Backend::Scalar`] unconditionally.
    #[must_use]
    pub fn resolve(self) -> Backend {
        #[cfg(feature = "std")]
        {
            let requested = std::env::var(self.override_var)
                .ok()
                .and_then(|value| Backend::from_name(value.trim()));
            let on_host = Backend::probes_on_host;
            let detected = detect(self.supported, on_host);
            apply_override(self.supported, detected, requested, on_host)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = self;
            Backend::Scalar
        }
    }
}

/// The strongest supported tier for which `on_host` is true, else
/// [`Backend::Scalar`].
///
/// [`Ord`](core::cmp::Ord) declares weaker backends greater, so `min()` over
/// the summoning subset selects the strongest tier the host and the supported
/// set agree on.
///
/// Only the capability pipeline (selection, and `internals` experiments) uses
/// this; without `std` the crate reports [`Backend::Scalar`] and never
/// detects.
#[cfg(any(feature = "std", feature = "internals"))]
fn detect(supported: &[Backend], on_host: impl Fn(Backend) -> bool) -> Backend {
    supported
        .iter()
        .copied()
        .filter(|&backend| on_host(backend))
        .min()
        .unwrap_or(Backend::Scalar)
}

/// Apply the downgrade-only override.
///
/// `requested` is accepted only when it is in `supported`, **summons on this
/// host** (`on_host`), and is at most as strong as `detected`; otherwise
/// `detected` stands. Anything else — unknown, unsupported, not summonable
/// on this host, or stronger — is ignored.
///
/// The `on_host(requested)` gate is the arch root (R10): the request is
/// re-probed with the same `summon()` used for detection, so a backend the
/// host cannot run is refused regardless of where `>=` places it in the
/// ladder. That covers a different arch (`neon` on x86 — the total `Ord`
/// would otherwise accept it, since ARM tiers sort weaker than x86) and a
/// tier the host lacks (`v2` on a V1-only host — same family, still
/// unrunable). `Scalar` always summons, so `SIMD_BACKEND=scalar` still
/// forces scalar on every host.
///
/// `>=` (weaker-or-equal, downgrade-only) is normally implied by
/// membership + summoning + `detect` being the strongest summoning tier, so
/// it reads as the explicit downgrade guard rather than the sole soundness
/// check — a backstop against a caller
/// passing a probe inconsistent with `detect`'s.
///
/// Refusing upgrades is a soundness property, not a preference: running
/// vector code the CPU cannot execute is undefined behaviour.
#[cfg(any(feature = "std", feature = "internals"))]
fn apply_override(
    supported: &[Backend],
    detected: Backend,
    requested: Option<Backend>,
    on_host: impl Fn(Backend) -> bool,
) -> Backend {
    match requested {
        Some(requested)
            if supported.contains(&requested) && on_host(requested) && requested >= detected =>
        {
            requested
        }
        _ => detected,
    }
}

/// Unstable internals for benchmarking and downstream experimentation.
///
/// Nothing behind this module is a compatibility promise; the public surface
/// is [`Selection`](crate::Selection). The host probe is injected here so the
/// resolve pipeline can be exercised against a simulated host without
/// affecting real detection.
#[cfg(feature = "internals")]
pub mod internals {
    use crate::backend::Backend;

    /// The deterministic resolve core behind
    /// [`Selection::resolve`](crate::Selection::resolve), with the host probe
    /// and override injected directly.
    pub fn resolve_with(
        supported: &[Backend],
        on_host: impl Fn(Backend) -> bool,
        requested: Option<Backend>,
    ) -> Backend {
        apply_override(supported, detect(supported, &on_host), requested, &on_host)
    }

    /// The downgrade-only override decision, standalone.
    ///
    /// `on_host(requested)` must describe the same host as the probe used to
    /// produce `detected`.
    pub fn apply_override(
        supported: &[Backend],
        detected: Backend,
        requested: Option<Backend>,
        on_host: impl Fn(Backend) -> bool,
    ) -> Backend {
        super::apply_override(supported, detected, requested, on_host)
    }

    /// Strongest supported tier accepted by `on_host`, else [`Backend::Scalar`].
    pub fn detect(supported: &[Backend], on_host: impl Fn(Backend) -> bool) -> Backend {
        super::detect(supported, on_host)
    }
}

// Without `std` (or `internals`) the pipeline under test does not exist:
// `resolve()` is a Scalar stub, and `no_std_reports_scalar` in lib.rs is the
// assertion for that build.
#[cfg(test)]
#[cfg(any(feature = "std", feature = "internals"))]
mod tests {
    use super::*;
    use crate::backend::Backend;

    /// Simulated x86 host: every x86 tier summons.
    fn x86_host() -> impl Fn(Backend) -> bool {
        |b: Backend| {
            matches!(
                b,
                Backend::V3GfniCrypto | Backend::V3 | Backend::V2 | Backend::V1
            )
        }
    }

    /// Simulated strongest host in the shipped ladder: every tier summons
    /// (including the always-present [`Backend::Scalar`]).
    fn full_host() -> impl Fn(Backend) -> bool {
        |_b: Backend| true
    }

    /// Simulated x86 floor: only the SSE2 baseline and the portable fallback
    /// summon — a host without SSE4.2/AVX2, so `V2`/`V3` are unavailable.
    fn v1_only_host() -> impl Fn(Backend) -> bool {
        |b: Backend| matches!(b, Backend::V1 | Backend::Scalar)
    }

    #[test]
    fn detect_picks_strongest_supported_summoning() {
        assert_eq!(detect(&[Backend::V2, Backend::V1], x86_host()), Backend::V2);
        // Order of the supported argument is irrelevant.
        assert_eq!(
            detect(&[Backend::V1, Backend::V2, Backend::V3], x86_host()),
            Backend::V3
        );
        assert_eq!(detect(Backend::ALL, full_host()), Backend::V3GfniCrypto);
    }

    #[test]
    fn detect_falls_back_to_scalar_when_nothing_supported_summons() {
        assert_eq!(detect(&[], x86_host()), Backend::Scalar);
        // Only ARM tiers supported on an x86 host: nothing summons.
        assert_eq!(
            detect(&[Backend::NeonAes, Backend::Neon], x86_host()),
            Backend::Scalar
        );
        assert_eq!(detect(&[Backend::Scalar], x86_host()), Backend::Scalar);
    }

    #[test]
    fn narrowing_omits_host_best_resolves_to_strongest_supported() {
        // The cafft rot, as a regression: a supported set that omits the
        // host's best tier must resolve to the strongest *supported* tier,
        // not silently fall through past it.
        assert_eq!(
            detect(&[Backend::V3, Backend::V2, Backend::V1], x86_host()),
            Backend::V3
        );
        assert_eq!(
            detect(&[Backend::V1, Backend::Neon], x86_host()),
            Backend::V1
        );
        // Full ladder on a host whose top tier is V3: V3, not V2/V1.
        let v3_host = |b: Backend| matches!(b, Backend::V3 | Backend::V2 | Backend::V1);
        assert_eq!(detect(Backend::ALL, v3_host), Backend::V3);
    }

    #[test]
    fn scalar_override_forces_scalar() {
        // strongest host; Scalar always summons so it is accepted as a
        // downgrade on any of them.
        let detected = detect(Backend::ALL, full_host());
        assert_eq!(detected, Backend::V3GfniCrypto);
        assert_eq!(
            apply_override(Backend::ALL, detected, Some(Backend::Scalar), full_host()),
            Backend::Scalar
        );
    }

    #[test]
    fn override_stronger_than_detected_is_ignored() {
        const SET: &[Backend] = &[Backend::V3, Backend::V2, Backend::Scalar];
        // With a permissive probe (`|_| true`, everything summons), the
        // downgrade-only `>=` guard alone refuses the V3-on-V2 upgrade.
        assert_eq!(
            apply_override(SET, Backend::V2, Some(Backend::V3), |_| true),
            Backend::V2
        );
        // And when no vector tier summons at all, an upgrade is refused even
        // though a permissive probe claims the host could run it.
        assert_eq!(
            apply_override(SET, Backend::Scalar, Some(Backend::V3), |_| true),
            Backend::Scalar
        );
    }

    #[test]
    fn override_within_family_downgrades() {
        const SET: &[Backend] = &[
            Backend::V3GfniCrypto,
            Backend::V3,
            Backend::V2,
            Backend::V1,
            Backend::Scalar,
        ];
        let detected = Backend::V3GfniCrypto;
        assert_eq!(
            apply_override(SET, detected, Some(Backend::V2), full_host()),
            Backend::V2
        );
        assert_eq!(
            apply_override(SET, detected, Some(Backend::Scalar), full_host()),
            Backend::Scalar
        );
        // Same-strength request is a no-op downgrade.
        assert_eq!(
            apply_override(SET, detected, Some(Backend::V3GfniCrypto), full_host()),
            Backend::V3GfniCrypto
        );
    }

    #[test]
    fn cross_family_override_is_refused() {
        const SET: &[Backend] = &[
            Backend::V3,
            Backend::V2,
            Backend::V1,
            Backend::NeonAes,
            Backend::Neon,
            Backend::Scalar,
        ];
        let detected = detect(SET, x86_host());
        assert_eq!(detected, Backend::V3);
        // `neon` sorts weaker than the x86 tiers, so `Neon >= V3` alone would
        // accept it; the arch-rooted `on_host(requested)` probe refuses it
        // because `NeonToken` does not summon on an x86 host.
        assert_eq!(
            apply_override(SET, detected, Some(Backend::Neon), x86_host()),
            Backend::V3
        );
        assert_eq!(
            apply_override(SET, detected, Some(Backend::NeonAes), x86_host()),
            Backend::V3
        );
    }

    #[test]
    fn unavailable_within_family_request_is_refused() {
        // The same arch-rooted probe also refuses a same-family request the
        // host lacks: `v2` (SSE4.2) on a V1-only (SSE2) host does not summon,
        // so the override is ignored rather than resolving a backend that
        // would SIGILL the CPU.
        const SET: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
        let detected = detect(SET, v1_only_host());
        assert_eq!(detected, Backend::V1);
        assert_eq!(
            apply_override(SET, detected, Some(Backend::V2), v1_only_host()),
            Backend::V1
        );
        // Scalar is still accepted on the same host.
        assert_eq!(
            apply_override(SET, detected, Some(Backend::Scalar), v1_only_host()),
            Backend::Scalar
        );
    }

    #[test]
    fn override_not_in_supported_is_ignored() {
        const SET: &[Backend] = &[Backend::V3, Backend::V1, Backend::Scalar];
        // V2 is not in the supported set, so it is refused even though a
        // permissive probe claims the host could run it.
        assert_eq!(
            apply_override(SET, Backend::V3, Some(Backend::V2), |_| true),
            Backend::V3
        );
    }

    #[test]
    fn missing_or_unknown_override_leaves_detected() {
        const SET: &[Backend] = &[Backend::V3, Backend::Scalar];
        assert_eq!(
            apply_override(SET, Backend::V3, None, |_| true),
            Backend::V3
        );
        // An unknown env value parses to None before apply_override sees it.
        assert_eq!(Backend::from_name("turbo"), None);
    }

    // Detection is override-independent: these assert host-floor invariants
    // that hold on the real platform regardless of any `SIMD_BACKEND`.
    #[test]
    #[cfg(all(feature = "std", target_arch = "x86_64"))]
    fn x86_detection_floor_is_v1() {
        // V1 (SSE2) is the x86_64 baseline and always summons, so detection
        // over the full ladder never falls below an x86 tier.
        let detected = detect(Backend::ALL, Backend::probes_on_host);
        assert!(
            matches!(
                detected,
                Backend::V3GfniCrypto | Backend::V3 | Backend::V2 | Backend::V1
            ),
            "detected {detected:?} on x86_64"
        );
    }

    #[test]
    #[cfg(all(feature = "std", target_arch = "aarch64"))]
    fn aarch64_detection_floor_is_neon() {
        let detected = detect(Backend::ALL, Backend::probes_on_host);
        assert!(
            matches!(detected, Backend::NeonAes | Backend::Neon),
            "detected {detected:?} on aarch64"
        );
    }

    #[test]
    fn supported_membership_uses_equality_not_strength() {
        // An absent weaker tier is refused even though the probe claims the
        // host could run it and `>=` would pass: membership in the consumer's
        // supported set is its own gate.
        const SET: &[Backend] = &[Backend::V3GfniCrypto, Backend::Scalar];
        let detected = Backend::V3GfniCrypto;
        assert_eq!(
            apply_override(SET, detected, Some(Backend::V2), |b| b == Backend::V2),
            Backend::V3GfniCrypto
        );
    }
}
