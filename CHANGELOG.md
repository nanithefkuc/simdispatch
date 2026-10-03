# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and releases follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `just sve-runtime`: QEMU execution at multiple vector lengths with
  direct scalar comparisons for SVE/SVE2/SVE2-AES intrinsics, paired
  carryless-product layout, and actual capability/override assertions.

- CI runs the crate validation gate, checks the minimum supported Rust
  toolchain, and exercises the scalar environment override on a hosted runner.
- QEMU runtime CI executes the SVE/SVE2/SVE2-AES correctness matrix at
  VL 128/256/512, including real downgrade overrides and capability-negative
  configurations, with an explicitly provisioned AArch64 cross-toolchain.
- **`sve` feature (off by default):** the AArch64 scalable-vector capability
  surface — `Backend::Sve2Aes` / `Sve2` / `Sve` (override names `sve2_aes`,
  `sve2`, `sve`) ordered between the x86 and NEON ladders, the zero-sized
  proof tokens `simdispatch::arch::aarch64::{SveToken, Sve2Token,
  Sve2AesToken}` minted through archmage's
  `is_aarch64_feature_available!`, and the raw `core::arch::aarch64`
  re-export. The surface compiles only where `build.rs` confirms a
  little-endian AArch64 target and a nightly at or above the verified floor
  (release 1.101.0-nightly, commit date 2026-09-27 — the `nightly-2026-09-28`
  distribution); on stable and everywhere else the feature is inert with a
  visible build warning and the three names never summon. **Breaking:**
  adding the tiers to `Backend::ALL` changes full-ladder resolution on SVE
  hosts built with the feature — consumers dispatch on
  `Selection::supports` with their implemented tiers.
- `Backend::lane_bytes_on_host()`: the tier's width on the current host and
  thread. Fixed-width tiers equal `lane_bytes()`; the scalable tiers report
  the thread's current SVE vector length after a proof summons (the
  architectural minimum otherwise), where the const `lane_bytes()` stays the
  16-byte minimum.

## [0.2.0] — 2026-09-28

### Added

- `V4x` and `V4` detection via their Archmage tokens when the `avx512`
  feature is enabled, including their override names and lane widths. Without
  the feature or a summoning token, selection degrades to the strongest
  supported lower tier. V4 does not prove GFNI; V4x requires additional
  extensions beyond AVX-512F/BW/GFNI.

### Changed

- **Breaking:** Adding `V4x` and `V4` to `Backend::ALL` changes the
  full-ladder resolution on capable hosts. Consumers using `backend()` for
  kernel dispatch must switch to `Selection::supports` with their implemented
  tiers, and enable `avx512` only when they implement V4 or V4x.

## [0.1.0] — 2026-09-09

### Added

- Name reservation and crate scaffold, renamed from `simdet` to
  `simdispatch`. The crate and its git repository now use the permanent name.
- `archmage` dependency pinned exactly at `=0.9.29` from crates.io — the
  first release carrying `X64V3GfniCryptoToken` (AVX2 + GFNI without
  AVX-512, `imazen/archmage#66`, merged upstream).
- Crate documentation set per the ecosystem ground rule: `AGENTS.md`,
  `CHANGELOG.md`, `CONTRIBUTING.md`, the AI-authorship warning header on
  `README.md`, and `LICENSE`.
- The backend selection degrade ladder design: `Backend` enum ordered by
  `archmage` tier priority, the single `resolve()` pipeline backing it, and
  the `SIMD_BACKEND` downgrade-only override.
- `Backend`: the initial degrade-ladder variants
  (`V3GfniCrypto` / `V3` / `V2` / `V1` / `NeonAes` / `Neon` / `Wasm128` /
  `Scalar`), named after the `archmage` tier they prove and ordered by that
  tier's dispatch priority. Weak `Ord` sorts greater (the fff convention, so
  `requested >= detected` downgrade checks port unchanged).
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

## [0.0.0] — 2026-07-31

### Added

- Name reservation on crates.io and the `simdet` repository head. No library
  code.

[0.0.0]: https://github.com/nanithefkuc/simdispatch/commits/main
[0.1.0]: https://github.com/nanithefkuc/simdispatch/compare/b7c921f...f78b369
[0.2.0]: https://github.com/nanithefkuc/simdispatch/compare/f78b369...main
