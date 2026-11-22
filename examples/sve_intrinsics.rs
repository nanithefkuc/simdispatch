//! Compile-only verification of the SVE capability surface: a separate
//! `lib` example that name-guards the raw `core::arch::aarch64` re-export
//! and exercises the SVE2 and SVE2-AES intrinsics through the exact
//! [`Sve2Token`] / [`Sve2AesToken`] boundaries.
//!
//! Nothing here is a kernel, field arithmetic, or a runtime proof: the
//! target is codegen for an AArch64 host this machine may not be, so the
//! crate is never run. Scalable values are born and die inside
//! `#[target_feature]` functions; the public boundary is capability tokens,
//! fixed arrays, and slices, per the boundary contract of
//! `simdispatch::arch::aarch64`.
//!
//! On every build where `build.rs` does not define `simdispatch_sve` the
//! module below is absent and this crate compiles to nothing. The
//! `sve_canary_required` cfg — set only by the recipe that expects an
//! eligible AArch64 build — turns that silence into a hard failure.
//!
//! [`Sve2Token`]: simdispatch::arch::aarch64::Sve2Token
//! [`Sve2AesToken`]: simdispatch::arch::aarch64::Sve2AesToken

#![deny(unsafe_code)]
#![cfg_attr(simdispatch_sve, feature(stdarch_aarch64_sve))]

// The pinned-nightly cross-build this example exists for sets
// `sve_canary_required` via RUSTFLAGS. If that cfg is armed while
// `build.rs` declined to define `simdispatch_sve` — wrong target, wrong
// channel, or a toolchain below the floor — the SVE surface is absent and
// every later step would verify nothing. Fail instead.
#[cfg(all(sve_canary_required, not(simdispatch_sve)))]
compile_error!(
    "sve_canary_required is set but the simdispatch_sve cfg is absent: \
     the SVE surface did not compile, so this verification run is vacuous"
);

/// SVE2 three-way XOR, carryless PMUL, and two-vector TBL2 in one chain.
///
/// Every predicate is `svwhilelt_b8_u64(0, 32)` over `[u8; 32]` buffers:
/// active lanes address only bytes the arrays own, and no all-true
/// predicate is formed against a buffer smaller than the vector length.
///
/// # Safety
///
/// The caller must hold the passed `Sve2Token`, which `summon` minted only
/// after FEAT_SVE2 was proven present for this process.
#[allow(unsafe_code)]
#[cfg(simdispatch_sve)]
#[target_feature(enable = "sve,sve2")]
pub unsafe fn sve2_eor3_pmul_tbl2(
    _proof: simdispatch::arch::aarch64::Sve2Token,
    a: &[u8; 32],
    b: &[u8; 32],
    c: &[u8; 32],
    indices: &[u8; 32],
    out: &mut [u8; 32],
) {
    use simdispatch::arch::aarch64::*;
    // SINCE: this item enables "sve,sve2", covering every intrinsic named
    //        below, and the caller's `Sve2Token` proves FEAT_SVE2 on the
    //        host, so entering this function is sound.
    // THUS:  the `#[target_feature]` calls need no further proof, and the
    //        only memory accesses are the guarded `svld1_u8` / `svst1_u8`
    //        blocks, each bounded by `svwhilelt_b8_u64(0, 32)` over a
    //        `[u8; 32]` buffer, so every active lane is inside the array.
    let pg = svwhilelt_b8_u64(0, 32);
    let va = unsafe { svld1_u8(pg, a.as_ptr()) };
    let vb = unsafe { svld1_u8(pg, b.as_ptr()) };
    let vc = unsafe { svld1_u8(pg, c.as_ptr()) };
    let vi = unsafe { svld1_u8(pg, indices.as_ptr()) };
    let r = sveor3_u8(va, vb, vc);
    let r = svpmul_u8(r, va);
    let table = svcreate2_u8(r, vb);
    let r = svtbl2_u8(table, vi);
    unsafe { svst1_u8(pg, out.as_mut_ptr(), r) };
}

/// SVE2-AES polynomial multiply long bottom/top pair intrinsics.
///
/// `svpmullb_pair_u8` and `svpmullt_pair_u8` are the carryless-multiply
/// pair this tier exists for; the predicate is `svwhilelt_b8_u64(0, 16)`
/// over `[u8; 16]` buffers, so no lane reads or writes past the arrays.
///
/// # Safety
///
/// The caller must hold the passed `Sve2AesToken`, which `summon` minted
/// only after the SVE2 AES extension was proven present for this process.
#[allow(unsafe_code)]
#[cfg(simdispatch_sve)]
#[target_feature(enable = "sve,sve2,sve2-aes")]
pub unsafe fn sve2_aes_pmull_pair(
    _proof: simdispatch::arch::aarch64::Sve2AesToken,
    a: &[u8; 16],
    b: &[u8; 16],
    out: &mut [u8; 16],
) {
    use simdispatch::arch::aarch64::*;
    // SINCE: this item enables "sve,sve2,sve2-aes", covering every
    //        intrinsic named below, and the caller's `Sve2AesToken` proves
    //        the extension on the host, so entering this function is sound.
    // THUS:  the `#[target_feature]` calls need no further proof, and the
    //        only memory accesses are the guarded `svld1_u8` / `svst1_u8`
    //        blocks, each bounded by `svwhilelt_b8_u64(0, 16)` over a
    //        `[u8; 16]` buffer, so every active lane is inside the array.
    let pg = svwhilelt_b8_u64(0, 16);
    let va = unsafe { svld1_u8(pg, a.as_ptr()) };
    let vb = unsafe { svld1_u8(pg, b.as_ptr()) };
    let lo = svpmullb_pair_u8(va, vb);
    let hi = svpmullt_pair_u8(va, vb);
    let r = svadd_u8_x(pg, lo, hi);
    unsafe { svst1_u8(pg, out.as_mut_ptr(), r) };
}
