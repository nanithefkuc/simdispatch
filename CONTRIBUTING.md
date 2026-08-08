# Contributing

## Ground rules for this crate

`simdispatch` is the single source for backend detection and ordering in the
FEC stack. Two rules follow and are not negotiable:

1. **Selection is sound by construction.** Ordering comes from `archmage`'s
   tier priorities, never from a cherry-picked ladder. Detection is
   `Token::summon()`, never a re-implemented probe. The downgrade-only
   `SIMD_BACKEND` override is the only switch.
2. **No `unsafe`, ever.** `#![deny(unsafe_code)]` at the root. This crate
   composes proofs; it does not compute bytes. Consumer kernel bodies are not
   this crate's business.

## Running the tests

```sh
cargo test --all-features
cargo test --no-default-features   # no-std: resolution reports Scalar
```

Selection resolves once per process, so one run covers only the host's best
backend. Sweep the ladder explicitly:

```sh
SIMD_BACKEND=scalar cargo test --all-features
```

`SIMD_BACKEND` is downgrade-only: a request for a backend the host cannot
execute is ignored, not faked — put `std::process::exit` checks (or a similar
assert) behind any CI job that must prove a specific backend was selected.

## Adding a backend

A backend is an `archmage` tier the ecosystem has a kernel for; it is added at
that tier's `archmage` priority position, and never renames or reorders an
existing one. Each addition needs:

- The variant, bound to its `archmage` token (via the tier's `summon()`).
- `lane_bytes()`, `name()`, `from_name()` entries.
- A test asserting the variant occupies its `tiers.rs` priority position
  against the mirrored order table.
- A selection test for the failure mode that rotted `cafft`: a consumer
  declaring a supported set that omits a host-detected variant still resolves
  to the strongest *supported* tier.

## When the `archmage` pin moves

`imazen/archmage#66` merging means re-pinning from the fork rev to the
upstream merge commit. Do it in one sitting with the consuming crates:

```sh
git -C simdispatch checkout -b chore/archmage-pin
# Cargo.toml: editor — point archmage.git at imazen/archmage, rev at the merge sha
cargo update -p archmage
cargo test --all-features
cargo clippy --all-targets --all-features -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

Confirm the priority table still matches `archmage-macros/src/tiers.rs` at
the new rev, and bump every crate that consumes this crate's `Backend` set in
the same sitting (umbrella rule: renames and restructuring land per-crate and
per-repo, dependents updated together).

## Before opening a PR

```sh
cargo fmt --all -- --check
cargo clippy --all-features --all-targets -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --all-features --no-deps
```

The crate denies `missing_docs` and `unsafe_code`, so new public items need
doc comments and no unsafe may appear. MSRV is 1.89 and is checked in CI; do
not reach for newer standard-library APIs without raising it deliberately.

Commit messages follow the ecosystem rule (umbrella `AGENTS.md`): subject at
most ~10 words, no implementation detail (that goes in the PR and
`CHANGELOG.md`), and no references to planning artifacts (`.plans/` milestone
tags and the like).
