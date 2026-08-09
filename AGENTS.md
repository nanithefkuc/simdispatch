# AGENTS.md — simdispatch

Authority for the `simdispatch` crate: the Level 0 SIMD capability selection
backend that every SIMD-accelerated crate in the FEC stack consumes. If a rule
lives here, it outranks a habit inherited from `fff`, `cafft`, or anywhere
else — this crate is the single source for backend detection and ordering.

## What this is

`simdispatch` answers one question: **which SIMD backend is it legal to call on
this host?** It takes `archmage` capability tokens, adds a canonical backend
naming/ordering layer, a per-crate supported-set cap, and the downgrade-only
`SIMD_BACKEND` environment override, and produces a resolved `Backend` value.
Consumers keep every kernel (`#[target_feature]` today, `#[arcane]` after the
archmage adoption); this crate never touches kernel bodies.

## Ground rules (do not break)

1. **`archmage`'s ordering is our ordering.** `Backend` variant order
   reproduces `archmage`'s tier dispatch priorities exactly
   (`archmage-macros/src/tiers.rs`, backed by `token-registry.toml`).
   Reordering a variant, or choosing a priority that contradicts `tiers.rs`,
   is a behavioral and safety change. A new backend appears at its archmage
   priority position — never appended to the end as a shortcut.
2. **Detection is single-source: `summon()`.** The only host probe in the
   stack is `archmage`'s token `summon()`. No crate re-implements CPUID or
   `std::is_*_feature_detected!` on top of it. This crate ships the one
   `resolve()` pipeline; `fff`'s `detect()`, `cafft`'s `cap()` and
   `supported_on_host()`, and per-crate re-probing are the defect class this
   crate exists to delete.
3. **`SIMD_BACKEND` is downgrade-only.** The override accepts a backend only
   if the host can run it and it is at most what detection found. Refusing to
   upgrade is a soundness property: running vector code the CPU cannot execute
   is undefined behaviour, not a preference. There is exactly one override for
   the whole stack; per-crate overrides (`FFF_BACKEND`, `CAFFT_BACKEND`, …)
   are deleted as crates migrate.
4. **No kernels, no intrinsics, no `unsafe`.** `#![deny(unsafe_code)]` at the
   root. This crate composes proofs, it does not compute bytes.
5. **One runtime dependency: `archmage`, pinned by rev until
   `imazen/archmage#66` merges.** Bootstrapping on a fork rev is temporary;
   re-pin to the upstream merge commit once that pull request lands. No other
   runtime dependency may be added without the umbrella exception process.
6. **`internals` is unstable by contract.** The `internals` feature exposes
   `pub(crate)` items for benchmarking and downstream experiments; nothing
   behind it is a compatibility promise.

## What consumers must not do (this crate owns backend selection)

- Do **not** re-derive capability in a consumer (`cap()`, `supported_on_host`,
  a per-crate env override). Selection is single-source here (umbrella
  `AGENTS.md`, "Backend selection is single-source").
- Do **not** hardcode an ordered ladder in a consumer. A consumer declares the
  *set* of backends it implements (`Selection::supports(&[…])`); order and
  host proof come from this crate + `archmage`.
- Do **not** name a backend by its historical `fff`/`cafft` semantic label
  (`Gfni`, `Avx2`, `Ssse3`, `Pmull`) in new code. Variants are named after the
  `archmage` tier they prove, so the ladder cannot drift from the source of
  truth.

## Cross-crate invariants

- **Ordering feeds comparison, which feeds soundness.** Variant order encodes
  capability and the downgrade check (`requested >= detected`). Missorting a
  pair lets an override "upgrade" to a weaker backend mislabeled stronger —
  the check then refuses a legitimate downgrade or, worse, accepts an
  incapable one. A test asserts variant order against a table mirroring
  `tiers.rs`; it must be updated on every `archmage` rev bump.
- **A backend carries its proof.** Each `Backend` variant binds the `archmage`
  token that proves it; `summon()` succeeding for that token is the
  justification a consumer's `#[target_feature]` call is sound. A kernel added
  under a feature set not covered by the variant's token invalidates that
  proof — gate it down, don't widen the token silently.
- **`v4`/`v4x` require the `archmage` `avx512` cargo feature.** These tiers do
  not summon (do not even compile their dispatch) unless that feature is on.
  Consumers that implement them enable it; the ladder otherwise tops out at
  `v3_gfni_crypto`.
- **`lane_bytes()` is architectural and lives here.** 64/32/16/8 per tier;
  consumers deriving buffer geometry from a backend import it, never re-derive.

## Numbers

Capability ordering is canonical (from `archmage`), not measured, so
`BENCHMARKS.md`-style records are not expected here. But any *policy* number
(this crate has none of consequence today) follows the umbrella rule: carried
in `BENCHMARKS.md`-style docs, never in doc comments.

## Working here

```sh
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
# Selection is resolved once per process; sweep the ladder explicitly:
SIMD_BACKEND=scalar cargo test --all-features
```

Edition 2024, MSRV 1.89. The crate is `#![no_std]`; `std` enters internally
only for `archmage`'s runtime detection (default features).
