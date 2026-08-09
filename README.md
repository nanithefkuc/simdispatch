> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# simdispatch — SIMD backend selection

`simdispatch` is the single source of SIMD capability selection shared across
SIMD-accelerated crates. It resolves which vector backend the current host can
run, once per process, and exposes that choice to every consumer that ships
vector kernels. It contains no kernels, no intrinsics, and no `unsafe`.

Capability proof comes from [`archmage`](https://github.com/imazen/archmage)
capability tokens: a tier is "on the host" exactly when its token `summon()`s.
There is one probe in the whole stack, and it lives here.

- **`Backend`** — the capability ladder. Each variant is named after the
  `archmage` tier it proves and declared in that tier's dispatch-priority
  order (strongest first), so the order carries capability and cannot drift
  from `archmage`'s source of truth. Provides `Backend::ALL`,
  `name`/`from_name`/`Display`/`FromStr`, and `lane_bytes()`.
- **`Selection`** — a consumer declares the backends it implements
  (`Selection::supports(&[...])`) and resolves the choice for the current
  host: detected via `summon()`, narrowed to the supported set, then adjusted
  by the downgrade-only `SIMD_BACKEND` override. `Selection::resolve()` is a
  pure function of the host and the override, so two consumers resolving the
  same set always agree.
- **`simdispatch::backend()`** — the process-wide resolution of the full
  ladder, detected once per process.

```rust
use simdispatch::{Backend, Selection};

// The kernel set a consumer ships (example).
const KERNELS: &[Backend] = &[Backend::V3, Backend::V2, Backend::V1, Backend::Scalar];
let chosen = Selection::new("SIMD_BACKEND").supports(KERNELS).resolve();
```

## Usage

The MSRV is Rust 1.89.

`simdispatch` is distributed through git only; it is not published to
[crates.io](https://crates.io).

```toml
[dependencies]
simdispatch = { git = "https://github.com/nanithefkuc/simdispatch" }
```

Portable `no_std` builds drop runtime detection and report
`Backend::Scalar` unconditionally:

```toml
[dependencies]
simdispatch = { git = "https://github.com/nanithefkuc/simdispatch", default-features = false }
```

### Features

| Feature | Result |
| --- | --- |
| `std` (default) | runtime detection (`summon()`), the `SIMD_BACKEND` override, and process-wide `backend()` |
| `--no-default-features` | `#![no_std]`, reports `Backend::Scalar` unconditionally, never probes |
| `internals` | unstable selection internals for benchmarking and downstream experiments; no compatibility promise |

## Backends

Each backend variant maps to one `archmage` tier and a lane width. `name()`
values are the identifiers accepted by `SIMD_BACKEND`.

| Backend | `name` | Target | Lane bytes |
| --- | --- | --- | --- |
| `V3GfniCrypto` | `v3_gfni_crypto` | x86 AVX2 + GFNI + crypto | 32 |
| `V3` | `v3` | x86 AVX2 split-nibble shuffle | 32 |
| `V2` | `v2` | x86 SSE4.2 split-nibble shuffle | 16 |
| `V1` | `v1` | x86 SSE2 baseline | 16 |
| `NeonAes` | `neon_aes` | AArch64 NEON + AES (proves PMULL) | 16 |
| `Neon` | `neon` | AArch64 NEON baseline | 16 |
| `Wasm128` | `wasm128` | WebAssembly `simd128` | 16 |
| `Scalar` | `scalar` | portable fallback, always present | 8 |

## `SIMD_BACKEND`

`SIMD_BACKEND=v3_gfni_crypto|v3|v2|v1|neon_aes|neon|wasm128|scalar` requests a
backend at process startup.

It is **downgrade-only**. A request is honored only when it names a backend
the host can run — the request is re-probed with the same `archmage`
`summon()` detection uses, so `neon` on x86 or a tier the host lacks is
ignored — it is at most as strong as what detection found (an upgrade is
refused), and it is in the consumer's supported set. Anything else is ignored,
never faked. Refusing to upgrade is a soundness property: running vector code
the CPU cannot execute is undefined behaviour. `SIMD_BACKEND=scalar` forces the
whole stack to portable code.

Where a CI job must prove it ran a specific backend, assert the reported
backend inside the process rather than trusting the request to be honored.

## Building

`simdispatch` builds on stable Rust (edition 2024, MSRV 1.89) with no extra
tooling or target-feature flags — the backend is selected at runtime:

```sh
cargo build                        # default: std detection
cargo build --no-default-features  # portable no_std, always Scalar
cargo test --all-features
```

## Dependency

`simdispatch` depends on [`archmage`](https://github.com/imazen/archmage) for
its capability tokens, pinned to a fork rev of
[`imazen/archmage#66`](https://github.com/imazen/archmage/pull/66)
(`X64V3GfniCryptoToken` — AVX2 + GFNI without AVX-512) until the PR merges
upstream:

```toml
archmage = { git = "https://github.com/nanithefkuc/archmage", rev = "de519319b5670d93f71dada4c49cdfd83c0fc0ec" }
```

## License

MIT. See [LICENSE](LICENSE).
