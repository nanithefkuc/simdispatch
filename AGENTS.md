# AGENTS.md — simdispatch

Authority for the `simdispatch` crate: the Level 0 SIMD capability selection
backend that every SIMD-accelerated crate in the FEC stack consumes. If a rule
lives here, it outranks a habit inherited from `fgf`, `butterfly-fft`, or
anywhere else in the stack — this crate is the single source for backend
detection and ordering.

## What this is

`simdispatch` answers one question: **which SIMD backend is it legal to call on
this host?** It takes `archmage` capability tokens, adds a canonical backend
naming/ordering layer, a per-crate supported-set cap, and the downgrade-only
`SIMD_BACKEND` environment override, and produces a resolved `Backend` value.
Consumers keep every kernel (`#[target_feature]` today, `#[arcane]` after the
archmage adoption); this crate never touches kernel bodies.

## Tooling

`just validate` is the pull-request gate. The shared recipe surface is
documented once in the umbrella's root `AGENTS.md`; only what is specific to
this crate is written here.

- **`TIERS` is empty.** This crate *resolves* backends, it does not dispatch
  over them — there are no kernels, so re-running the suite once per tier would
  measure nothing new. `just test-tiers` says so and runs the host
  configuration only. Ladder behaviour (narrowing, the downgrade-only
  override, cross-family refusal, and the real `SIMD_BACKEND` environment case)
  is asserted inside the tests themselves.
- **`MIRI` is empty.** Ground rule 4 holds for the default build, and the
  `sve` residue below is Miri-dead: the nightly-gated SVE surface compiles
  only on little-endian aarch64 under an eligible nightly, a configuration
  Miri's interpreter does not carry (no `stdarch_aarch64_sve` support), so
  `just unsafe-check` reports and skips. That is a configuration boundary,
  not a gap to fill.
- **No bench targets.** There is no `benches/`; capability ordering is
  canonical rather than measured, so `just bench` and `just perf-bench` have
  nothing to run. `COV_IGNORE` is empty — every line counts toward the 95%
  gate.
- `justfile` is a byte-identical vendored copy and is never edited here; the
  umbrella's `just drift` check fails on a modified copy. Crate-specific values
  and any recipe unique to `simdispatch` belong in `crate.just`.
- **`just sve-check` verifies the SVE surface by codegen.** A `lib` example
  (`examples/sve_intrinsics.rs`, gated on `sve`) name-guards the raw
  `core::arch::aarch64` re-export and exercises the SVE2 EOR3/PMUL/TBL2 and
  SVE2-AES PMULLB/PMULLT-pair intrinsics through the exact token boundaries.
  The recipe builds it for `aarch64-unknown-linux-gnu` on the pinned
  `nightly-2026-09-28` (release, so real codegen), then runs clippy and
  rustdoc over the same target, then a host all-features check proving the
  example stays inert where `build.rs` declines the surface. The canary cfg
   `sve_canary_required` is declared by `build.rs` and armed only in that
   recipe's cross-build `RUSTFLAGS`: the example carries a `compile_error!` when the
  cfg is armed but `simdispatch_sve` is absent, so a vacuous pass (wrong
  toolchain, wrong target, surface silently not compiled) is impossible. It
   is compile verification, not a runtime proof — no SVE host is involved.
- **`just sve-runtime` executes the intrinsic verifier under QEMU.** It
  requires `qemu-aarch64`, `aarch64-linux-gnu-gcc`, an AArch64 glibc sysroot,
  `jq`, `timeout`, and the pinned nightly target. Optional positional
  arguments override the QEMU executable, sysroot, and linker. The recipe
  builds `tests/sve_runtime.rs` once at baseline AArch64, then asserts actual
  token availability, backend resolution, and vector width in separate
  processes at VL 128/256/512. Direct intrinsic comparisons exercise tails,
  whole-vector table geometry, bounded small-table loads, and paired
  carryless-product layout; sentinel guards detect out-of-payload stores.
  Forced `sve2`/`sve` runs assert the real override result. `neoverse-v1`
  and `max,sve=off` assert absent capabilities and upgrade refusal. An
  isolated SVE2-without-AES configuration is not part of this recipe.
  Missing tools, absent required surface, expectation mismatches, and
  timeouts fail; no runtime case passes through a skip.
  CI runs this recipe on `ubuntu-24.04` with distro `qemu-user`,
  `gcc-aarch64-linux-gnu`, `libc6-dev-arm64-cross`, and `jq`, records the
  installed tool versions, and uses the pinned nightly AArch64 target.
  The job supplies correctness evidence only, including wider-VL execution;
  no timing result is a benchmark under B7.

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
   `resolve()` pipeline; `fgf`'s historical `detect()`, `butterfly-fft`'s
   historical `cap()` and `supported_on_host()`, and per-crate re-probing are
   the defect class this crate exists to delete.
3. **`SIMD_BACKEND` is downgrade-only.** The override accepts a backend only
   if the host can run it and it is at most what detection found. Refusing to
   upgrade is a soundness property: running vector code the CPU cannot execute
   is undefined behaviour, not a preference. There is exactly one override for
   whole stack; per-crate overrides (`FFF_BACKEND`, `BUTTERFLY_FFT_BACKEND`, …)
   are deleted as crates migrate.
4. **No kernels, no intrinsics, no `unsafe` — one scoped opt-in exception.**
   `#![deny(unsafe_code)]` at the root, so every `unsafe` item carries its own
   `#[allow(unsafe_code)]`. The off-by-default `sve` feature adds the one
   exception: the SVE/SVE2/SVE2-AES capability tokens in
   `src/arch/aarch64.rs` and the raw `core::arch::aarch64` re-export, gated
   by the `simdispatch_sve` cfg that `build.rs` defines only for a
   little-endian aarch64 target on an eligible nightly. The exception is
   capability plumbing, not kernels: no `#[target_feature]` compute path
   ships here, and the residue ledger below enumerates the surviving
   `unsafe` in full.
5. **One runtime dependency: `archmage`, pinned exactly from crates.io.** The
   pin is `=0.9.29` — the first release carrying `X64V3GfniCryptoToken`
   (`imazen/archmage#66`, merged upstream). No floating range: tier ordering
   feeds the downgrade check, so a bump is a deliberate re-pin that re-runs
   the ordering test. No other runtime dependency may be added without the
   umbrella exception process.
6. **`internals` is unstable by contract.** The `internals` feature exposes
   `pub(crate)` items for benchmarking and downstream experiments; nothing
   behind it is a compatibility promise.

## Testing

- Tests: ordering pinned to the `archmage` `tiers.rs` mirror, narrowing
  (including the butterfly-fft regression), downgrade-only override,
  cross-family refusal, host-floor invariants (cfg-gated), and the real
  environment `SIMD_BACKEND` override test.
- Do **not** re-derive capability in a consumer (`cap()`, `supported_on_host`,
  a per-crate env override). Selection is single-source here (umbrella
  `AGENTS.md`, "Backend selection is single-source").
- Do **not** hardcode an ordered ladder in a consumer. A consumer declares the
  *set* of backends it implements (`Selection::supports(&[…])`); order and
  host proof come from this crate + `archmage`.
- Do **not** name a backend by its historical `fff`/`butterfly-fft` semantic
  label (`Gfni`, `Avx2`, `Ssse3`, `Pmull`) in new code. Variants are named after
  the `archmage` tier they prove, so the ladder cannot drift from the source of
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
  Consumers that implement `v4` or `v4x` enable `simdispatch/avx512`; the ladder
  otherwise tops out at `v3_gfni_crypto`.
- **`lane_bytes()` is architectural and lives here.** 64/32/16/8 per tier;
  consumers deriving buffer geometry from a backend import it, never re-derive.
- **The scalable tiers are local rows.** `sve2_aes` / `sve2` / `sve` bind the
  tokens in `simdispatch::arch::aarch64` (behind the off-default `sve`
  feature), not registry tokens; their priorities are local until `archmage`
  lands SVE tiers, and the ordering canary marks them as such.

## Unsafe residue ledger

The default build is `#![deny(unsafe_code)]` with zero exceptions. The
off-by-default `sve` feature carries exactly two `unsafe` items, both in
`src/arch/aarch64.rs`:

1. `svcntb_sve(_proof: SveToken)` — the per-item
   `#[allow(unsafe_code)]` wrapper entering the `#[target_feature(enable =
   "sve")]` sibling. Residue class 2 of the umbrella's kernel-residue
   enumeration (K5): the `SveToken` parameter in the signature is the
   SINCE–THUS proof carried in place above the item, so a caller cannot
   reach the sibling without a proof `summon` minted.
2. the nested `unsafe fn svcntb_enabled()` inside it — the single `CNTB`
   query, the only way to read the thread's current SVE vector length; no
   safe wrapper for it exists on this toolchain floor.

Nothing else in this crate, feature on or off, contains `unsafe`. A new
`unsafe` item requires a ledger entry here and a SINCE–THUS proof at the
item (P12).

The compile-verification example `examples/sve_intrinsics.rs` is a separate
`lib` crate with its own `#![deny(unsafe_code)]` root and exactly two
`unsafe` items, both per-item `#[allow(unsafe_code)]` with SINCE–THUS proofs
in place; they exist only where `simdispatch_sve` is defined and are
exercised by `just sve-check`, never shipped as kernels:

3. `sve2_eor3_pmul_tbl2(_proof: Sve2Token, ..)` — the `#[target_feature(enable
   = "sve,sve2")]` frame calling `sveor3_u8`, `svpmul_u8`, `svtbl2_u8` and
   the guarded `svld1_u8`/`svst1_u8` blocks; residue class 2 (K5): the token
   parameter is the proof, and each load/store predicate is
   `svwhilelt_b8_u64(0, 32)` over a `[u8; 32]` buffer, so active lanes stay
   inside the array.
4. `sve2_aes_pmull_pair(_proof: Sve2AesToken, ..)` — the
   `#[target_feature(enable = "sve,sve2,sve2-aes")]` frame calling
   `svpmullb_pair_u8` / `svpmullt_pair_u8` with the same token-proof and
   `svwhilelt_b8_u64(0, 16)`-bounded load/store structure.

The runtime integration test `tests/sve_runtime.rs` is a separate crate
with its own `#![deny(unsafe_code)]` root. Its seven safe wrappers and
seven nested target-feature siblings each have a per-item exception and
SINCE–THUS proof. They compile only under `simdispatch_sve` and are test
operations, never shipped kernels:

5. `xor_assign_sve` and its `body`: `SveToken` proves the SVE entry;
   equal-length slices and `whilelt` bound every byte load and store.
6. `tbl_full_sve` and its `body`: `SveToken` proves entry; the table spans
   the calling thread's complete VL, while indices and stores are bounded
   by their equal-length slices.
7. `replicate_table_sve` and its `body`: `SveToken` proves entry;
   `svld1rq_u8` reads the referenced 16-byte table and replicates it, while
   `whilelt` bounds output stores.
8. `eor3_sve2` and its `body`: `Sve2Token` proves SVE/SVE2 entry;
   equal-length slices and `whilelt` bound the three inputs and output.
9. `pmul_sve2` and its `body`: `Sve2Token` proves SVE/SVE2 entry;
   equal-length slices and `whilelt` bound byte-polynomial-product access.
10. `tbl2_sve2` and its `body`: `Sve2Token` proves SVE/SVE2 entry; both
    tables span the complete VL, with predicated index loads and stores.
11. `pmull_pair_sve2aes` and its `body`: `Sve2AesToken` proves entry to
    SVE/SVE2/SVE2-AES. Typed `u64` slices have equal, even lane counts;
    the even `svcntd` stride preserves pair boundaries and `whilelt`
    bounds loads and stores, including partial-vector tails.

## Numbers

Capability ordering is canonical (from `archmage`), not measured, so
`BENCHMARKS.md`-style records are not expected here. But any *policy* number
(this crate has none of consequence today) follows the umbrella rule: carried
in `BENCHMARKS.md`-style docs, never in doc comments.

The umbrella's **B7 SVE2 development contract** governs consumers of the
scalable tiers: 128-bit VL is the sole optimization target, and QEMU CI
provides correctness evidence at VL 128/256/512. Wider-VL execution evidence
comes only from QEMU. SVE2/SVE2-AES performance numbers and speedup claims
are prohibited until the hardware restriction is explicitly lifted under
B7. Capability ordering is not a throughput ranking, and width-dependent
memory safety remains required at every supported live VL.

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
