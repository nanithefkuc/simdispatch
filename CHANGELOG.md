# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Name reservation and crate scaffold (P0), renamed from `simdet` to
  `simdispatch`. The crate and its git repository now use the permanent name.
- `archmage` dependency pinned to `de519319b5670d93f71dada4c49cdfd83c0fc0ec`
  — the tip of `imazen/archmage#66` (`X64V3GfniCryptoToken`, AVX2 + GFNI
  without AVX-512) on the author's fork, pending the upstream merge. Re-pin
  procedure in `.plans/05-conventions.md`.
- Crate documentation set per the ecosystem ground rule: `AGENTS.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, the AI-authorship warning header on
  `README.md`, and `LICENSE`.
- The Level 0 design for the backend selection degrade ladder, in `.plans/`:
  `Backend` enum ordered by `archmage` tier priority, the single `resolve()`
  pipeline backing it, and the `SIMD_BACKEND` downgrade-only override.

### Added (P1 — the ladder in `simdispatch`)

- `Backend`: the initial degrade-ladder variants
  (`V3GfniCrypto` / `V3` / `V2` / `V1` / `NeonAes` / `Neon` / `Wasm128` /
  `Scalar`), named after the `archmage` tier they prove and ordered by that
  tier's dispatch priority. Weak `Ord` sorts greater (the fff convention, so
  `requested >= detected` downgrade checks port unchanged). `V4x` is deferred
  until a validated 512-bit kernel exists.
- `Backend::ALL: &'static [Backend]`; `name` / `from_name` / `Display` /
  `FromStr` / `ParseBackendError`; `lane_bytes()`; the arch-family mapping.
- `Selection`: declare the supported set with `supports(&[...])`, resolve for
  the host with `resolve()` — detection via the single `archmage` `summon()`
  probe, narrowed to the supported set, then adjusted by the downgrade-only
  `SIMD_BACKEND` override.
- The process-wide `static BACKEND` and `backend()` over the full ladder.
- The `SIMD_BACKEND` environment override (downgrade-only: a request is
  honored only when it is in the supported set, itself summons on this host —
  the request is re-probed with `archmage` `summon()` — and is at most as
  strong as the detected tier; otherwise ignored, never faked).
  `SIMD_BACKEND=scalar` forces the whole stack to portable code. A request
  for a backend the host cannot run — a different arch (`neon` on x86) or a
  tier the host lacks (`v2` on a V1-only host) — is refused (R10).
- Cargo features: `std` (default; runtime detection + override + `backend()`)
  and `internals` (unstable selection internals with an injectable host
  probe, per R8).
- Tests: ordering pinned to the `archmage` `tiers.rs` mirror, narrowing
  (including the cafft-rot regression), downgrade-only override, cross-family
  refusal, host-floor invariants (cfg-gated), and the real-environment
  `SIMD_BACKEND` override test.

### Changed

- Nothing in the public API yet: version stays `0.0.0` until the first
  consumer (P2) migrates, per the plan.

## [0.0.0] — 2026-07-31

### Added

- Name reservation on crates.io and the `simdet` repository head. No library
  code.

[0.0.0]: https://github.com/nanithefkuc/simdispatch/commits/main
