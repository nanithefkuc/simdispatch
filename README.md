> [!WARNING]
> This library was made with the help of AI. While the library has tests
to check for regressions, things may break. Audit the code yourself, or with
your own agent before using.

# simdispatch — SIMD backend selection

`simdispatch` is the single source of SIMD capability selection shared across
SIMD-accelerated crates. It resolves which vector backend the current host can
run, once per process, and exposes that choice to every consumer that ships
vector kernels. The default build contains no kernels, no intrinsics, and no
`unsafe`; the off-by-default `sve` feature is the one scoped exception,
documented below.

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
| `avx512` | enables `archmage/avx512` and V4/V4x detection on capable x86 hosts |
| `sve` | off-default AArch64 SVE/SVE2/SVE2-AES capability surface; nightly-only, see [SVE](#sve) |
| `--no-default-features` | `#![no_std]`, reports `Backend::Scalar` unconditionally, never probes |
| `internals` | unstable selection internals for benchmarking and downstream experiments; no compatibility promise |

## Backends

Each backend variant maps to one `archmage` tier and a lane width. `name()`
values are the identifiers accepted by `SIMD_BACKEND`.

| Backend | `name` | Target | Lane bytes |
| --- | --- | --- | --- |
| `V4x` | `v4x` | x86 AVX-512 + GFNI + vector crypto (`avx512` feature) | 64 |
| `V4` | `v4` | x86 AVX-512F/BW/CD/DQ/VL, without a GFNI guarantee (`avx512` feature) | 64 |
| `V3GfniCrypto` | `v3_gfni_crypto` | x86 AVX2 + GFNI + crypto | 32 |
| `V3` | `v3` | x86 AVX2 split-nibble shuffle | 32 |
| `V2` | `v2` | x86 SSE4.2 split-nibble shuffle | 16 |
| `V1` | `v1` | x86 SSE2 baseline | 16 |
| `Sve2Aes` | `sve2_aes` | AArch64 SVE2 + AES extension (`sve` feature, eligible nightly) | 16 (minimum) |
| `Sve2` | `sve2` | AArch64 SVE2 (`sve` feature, eligible nightly) | 16 (minimum) |
| `Sve` | `sve` | AArch64 SVE baseline (`sve` feature, eligible nightly) | 16 (minimum) |
| `NeonAes` | `neon_aes` | AArch64 NEON + AES (proves PMULL) | 16 |
| `Neon` | `neon` | AArch64 NEON baseline | 16 |
| `Wasm128` | `wasm128` | WebAssembly `simd128` | 16 |
| `Scalar` | `scalar` | portable fallback, always present | 8 |

## `SIMD_BACKEND`

`SIMD_BACKEND=v4x|v4|v3_gfni_crypto|v3|v2|v1|sve2_aes|sve2|sve|neon_aes|neon|wasm128|scalar`
requests a backend at process startup.

Without `avx512`, `v4` and `v4x` remain recognized names but never summon,
even on an AVX-512 host. With the feature enabled, V4 is selectable when V4x
is unavailable. V4 does not prove GFNI: consumers using GFNI instructions
must not declare V4 for those kernels. V4x proves GFNI but also requires
VPOPCNTDQ, IFMA, VBMI, VBMI2, BITALG, VNNI, VPCLMULQDQ and VAES beyond
the V4 feature set. A host with V4 and GFNI but without those extensions
can select V4 for a non-GFNI kernel or a lower GFNI-capable tier.

Without `sve`, `sve2_aes`, `sve2`, and `sve` remain recognized names but
never summon, even on an SVE host.

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

## SVE

The off-by-default `sve` feature adds the AArch64 scalable-vector tiers
`Sve2Aes` / `Sve2` / `Sve` (override names `sve2_aes`, `sve2`, `sve`) and the
capability surface behind them. It is deliberately conservative about
toolchains: the intrinsic surface needs the unstable
`stdarch_aarch64_sve` library feature, so `build.rs` compiles it only when
the `sve` feature is requested **and** the target is little-endian AArch64
**and** the compiler is a nightly at or above the verified floor — release
1.101.0-nightly with a commit date of 2026-09-27 or later, i.e. the
`nightly-2026-09-28` distribution and newer. Anywhere else — stable
compilers, other targets, older nightlies — the surface is inert (the
feature compiles as a no-op with a visible build warning), the three names
stay recognized but never summon, and the crate remains stable-buildable.

Three zero-sized proof types carry detection:
`simdispatch::arch::aarch64::{SveToken, Sve2Token, Sve2AesToken}`, minted by
`summon()` after archmage's `is_aarch64_feature_available!` accepts the exact
feature string (`"sve"`, `"sve2"`, `"sve2-aes"`). Stronger tokens extract
already-proven weaker capabilities (`Sve2AesToken::sve2`, `Sve2Token::sve`);
there is no inverse, and `Sve2Token` proves neither the optional AES and
bitperm extensions nor NEON AES. The module also re-exports the raw
`core::arch::aarch64` intrinsic surface; naming any unstable item from it
requires the consumer's own crate-root feature gate.

The scalable widths are two different answers. `lane_bytes()` is constant and
reports the architectural minimum (16 bytes), which is insufficient for
full-vector buffer allocation or loop strides when the live width is larger.
`lane_bytes_on_host()` and `SveToken::vector_bytes()` report the
calling thread's current SVE vector length, a per-thread multiple of that
minimum that may be larger and must not be cached in process-wide state.

The feature is the crate's one `unsafe` exception: the width query enters a
`#[target_feature(enable = "sve")]` sibling behind a proof-carrying wrapper,
enumerated in the crate's residue ledger. Everything else — kernels,
dispatch — stays in consumers.

## Building

`just sve-runtime` cross-builds and executes the intrinsic verifier under
QEMU at VL 128/256/512, checking actual tokens, selected backends, and
scalar equality. It requires the pinned nightly and AArch64 Rust target,
`qemu-aarch64`, `aarch64-linux-gnu-gcc`, an AArch64 glibc sysroot, `jq`,
and `timeout`. The default sysroot is `/usr/aarch64-linux-gnu`; optional
positional arguments override the QEMU executable, sysroot, and linker.
The runner also checks real downgrade overrides and SVE-only/no-SVE
configurations. Isolated SVE2-without-AES coverage remains outside this
runtime matrix. Missing prerequisites or mismatched expectations fail.

`simdispatch` builds on stable Rust (edition 2024, MSRV 1.89) with no extra
tooling or target-feature flags — the backend is selected at runtime. The
`sve` feature is the one exception: it compiles its surface only on an
eligible nightly (see [SVE](#sve)); everywhere else it is inert.

```sh
cargo build                        # default: std detection
cargo build --no-default-features  # portable no_std, always Scalar
cargo test --all-features
```

## Dependency

`simdispatch` depends on [`archmage`](https://github.com/imazen/archmage) for
its capability tokens. Pinned exactly to the crates.io release carrying
`X64V3GfniCryptoToken` — AVX2 + GFNI without AVX-512
([`imazen/archmage#66`](https://github.com/imazen/archmage/pull/66), merged
upstream):

```toml
archmage = "=0.9.29"
```

## License

MIT. See [LICENSE](LICENSE).
