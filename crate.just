# Crate-specific values for the shared `justfile`, which is vendored verbatim
# from the ecosystem's canonical copy and must not be edited here. Recipes
# unique to simdispatch belong in this file and are documented in `AGENTS.md`.

CRATE := 'simdispatch'
MSRV := '1.89'

# Backend tiers this crate dispatches over, strongest first. `just test-tiers`
# and `just cover` re-run once per tier, because dispatch resolves one backend
# per process. Empty means the crate owns no dispatch surface.
TIERS := ''

# `cargo miri test` argument sets, one per line. Empty when the crate declares
# no unsafe surface, in which case `just unsafe-check` reports and skips.
MIRI := ''

# Coverage exclusions for code that cannot execute on a CI host. Empty means
# every line of the crate counts.
COV_IGNORE := ''

# Pinned nightly above the build.rs SVE floor, cross-checked from CI. The
# canary cfg is declared (check-cfg) and armed by RUSTFLAGS only for the
# AArch64 steps, where the `simdispatch_sve` cfg is expected to exist: the
# example's `compile_error!` turns an absent SVE surface into a hard
# failure instead of a vacuous pass.
SVE_NIGHTLY := 'nightly-2026-09-28'
SVE_TARGET := 'aarch64-unknown-linux-gnu'
SVE_FLAGS := '--cfg=sve_canary_required --check-cfg=cfg(sve_canary_required) -D warnings'

# Compile-only verification of the SVE capability surface on the pinned
# nightly, cross-compiled for AArch64: full codegen of the `sve_intrinsics`
# example lib (release), clippy and rustdoc over the same target, then a
# host all-features check proving the example stays inert where build.rs
# declines the surface. Nothing runs: this is codegen verification, not a
# runtime proof.
[group('test')]
sve-check:
    RUSTFLAGS='{{SVE_FLAGS}}' cargo +{{SVE_NIGHTLY}} build --release --features sve \
        --target {{SVE_TARGET}} --example sve_intrinsics
    RUSTFLAGS='{{SVE_FLAGS}}' cargo +{{SVE_NIGHTLY}} clippy --target {{SVE_TARGET}} \
        --all-targets --features sve -- -D warnings
    RUSTFLAGS='{{SVE_FLAGS}}' RUSTDOCFLAGS='-D warnings' \
        cargo +{{SVE_NIGHTLY}} doc --target {{SVE_TARGET}} --features sve --no-deps
    cargo check --all-features --all-targets

# Runtime verification of the SVE capability surface under QEMU user-mode
# emulation, the runtime complement to the compile-only `sve-check`. The test
# executable is cross-built once (retrieved from cargo's JSON messages, never
# a hardcoded path), then launched per configuration with the full
# SVE_TEST_* contract: `max` CPUs pin sve-max-vq 1/2/4 with every token
# present and every operation family exercised, forced runs assert the real
# downgrade-only SIMD_BACKEND override against live detection, and negative
# runs assert an SVE-only CPU and a no-SVE CPU both refuse the sve2_aes
# upgrade request. Every launch is `timeout`-bounded: a hung predication
# loop is a failure, not a stall. The tested QEMU CPU configurations do not
# isolate SVE2 without SVE2-AES; that negative case is not claimed here.
[group('test')]
sve-runtime QEMU="qemu-aarch64" SYSROOT="/usr/aarch64-linux-gnu" LINKER="aarch64-linux-gnu-gcc":
    #!/usr/bin/env bash
    set -euo pipefail
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER='{{LINKER}}'
    '{{QEMU}}' --version
    EXE=$(RUSTFLAGS='{{SVE_FLAGS}}' cargo +{{SVE_NIGHTLY}} test --no-run --release \
        --features sve --target {{SVE_TARGET}} --test sve_runtime \
        --message-format=json-render-diagnostics \
        | jq -r 'select(.reason == "compiler-artifact"
                     and .target.name == "sve_runtime"
                     and .executable != null) | .executable')
    test -n "$EXE" || { echo 'sve_runtime executable not produced' >&2; exit 1; }
    echo "test executable: $EXE"

    # label vl sve sve2 sve2_aes backend ops cpu [override assignments...]
    run() {
        local label=$1 vl=$2 s=$3 s2=$4 s2a=$5 backend=$6 ops=$7 cpu=$8
        shift 8
        echo "== case $label (cpu $cpu)"
        env -u SIMD_BACKEND "$@" \
            "SVE_TEST_CASE=$label" "SVE_TEST_VL_BYTES=$vl" \
            "SVE_TEST_SVE=$s" "SVE_TEST_SVE2=$s2" "SVE_TEST_SVE2_AES=$s2a" \
            "SVE_TEST_BACKEND=$backend" "SVE_TEST_OPS=$ops" \
            timeout 120 '{{QEMU}}' -L '{{SYSROOT}}' \
            -cpu "$cpu" "$EXE" --test-threads=1 --nocapture
    }

    for vq in 1 2 4; do
        vl=$((vq * 16))
        cpu="max,sve-max-vq=$vq,sve-default-vector-length=$vl"
        run "max-vq$vq" "$vl" 1 1 1 sve2_aes base,sve2,sve2_aes "$cpu"
        run "max-vq$vq-forced-sve2" "$vl" 1 1 1 sve2 base,sve2,sve2_aes "$cpu" \
            SIMD_BACKEND=sve2
        run "max-vq$vq-forced-sve" "$vl" 1 1 1 sve base "$cpu" SIMD_BACKEND=sve
    done
    # SVE without SVE2: the sve2_aes override request must be refused and
    # resolution stay at the detected top tier.
    run neoverse-v1-sve-only 32 1 0 0 sve base neoverse-v1 SIMD_BACKEND=sve2_aes
    # No SVE at all: detection falls to NEON+AES and the upgrade is refused.
    run max-sve-off 16 0 0 0 neon_aes none max,sve=off SIMD_BACKEND=sve2_aes
