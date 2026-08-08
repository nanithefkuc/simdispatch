> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# simdispatch — SIMD backend selection

Level 0 crate: the single source for SIMD capability selection in the FEC
stack. Everything else that ships vector kernels — `fff`, `cafft`, and every
future consumer — gets its backend from here.

Capability proof comes from [`archmage`](https://github.com/imazen/archmage)
capability tokens: a tier is "on the host" exactly when its token
`summon()`s. There is one probe in the whole stack, and it lives here. No
kernels, no intrinsics, no `unsafe`.

## What you get

- **`Backend`** — the ladder. Variants are named after the `archmage` tier
  they prove (`V3GfniCrypto`, `V3`, `V2`, `V1`, `NeonAes`, `Neon`, `Wasm128`,
  `Scalar`) and declared in that tier's dispatch-priority order (strongest
  first), so the order carries capability and cannot drift from `archmage`'s
  source of truth. `Backend::ALL`, `name`/`from_name`/`Display`/`FromStr`,
  `lane_bytes()`.
- **`Selection`** — a consumer declares the backends it implements
  (`Selection::supports(&[...])`) and resolves the choice for the current
  host: detected via `summon()`, narrowed to the supported set, then adjusted
  by the **downgrade-only** `SIMD_BACKEND` override. `Selection::resolve()`
  is a pure function of the host and the override, so two consumers resolving
  the same set always agree — the invariant cafft's re-probe broke.
- **`simdispatch::backend()`** — the process-wide resolution of the full
  ladder, detected once per process.

```rust
use simdispatch::{Backend, Selection};

// The kernel set this crate ships (example).
const KERNELS: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
let chosen = Selection::new("SIMD_BACKEND").supports(KERNELS).resolve();
```

## `SIMD_BACKEND`

The one stack-wide override, replacing the per-crate `FFF_BACKEND` /
`CAFFT_BACKEND` as crates migrate. Accepted values are `Backend::name()`
values (`scalar`, `v1`, `v2`, `v3`, `v3_gfni_crypto`, `neon`, `neon_aes`,
`wasm128`).

It is **downgrade-only**. A request is honored only when it names a backend
the host can run — the request is re-probed with `archmage` `summon()`, the
same arch-rooted probe detection uses, so `neon` on x86 or a tier the host
lacks is ignored — it is at most as strong as what detection found (an
upgrade is refused), and it is in the consumer's supported set. Anything else
is ignored, never faked. Refusing to upgrade is a soundness property: running
vector code the CPU cannot execute is undefined behaviour.
`SIMD_BACKEND=scalar` forces the whole stack to portable code — the escape
hatch operators and differential-testing need.

Where a CI job must prove it ran a specific backend, assert the reported
backend inside the process rather than trusting the request to be honored.

## Features

- `std` (default) — runtime detection (`summon()`) + the `SIMD_BACKEND`
  override + the process-wide `backend()`. Without it the crate is
  `#![no_std]` and reports `Backend::Scalar` unconditionally.
- `internals` — unstable selection internals for benchmarking and
  downstream experiments. Nothing behind it is a compatibility promise.

## Dependency

```toml
[dependencies]
archmage = { git = "https://github.com/nanithefkuc/archmage", rev = "de519319b5670d93f71dada4c49cdfd83c0fc0ec" }
```

Pinned to the tip of [`imazen/archmage#66`](https://github.com/imazen/archmage/pull/66)
(`X64V3GfniCryptoToken` — AVX2 + GFNI without AVX-512) on the author's fork
until the PR merges upstream. Once merged, re-pin to
`https://github.com/imazen/archmage` at the merge commit; see
`.plans/04-conventions.md` for the bump procedure. A fork rev is a temporary
address; every crate below `simdispatch` consumes this pin, so the swap
happens here first, in the same sitting as the dependent updates.

## License

MIT. See [LICENSE](LICENSE).
