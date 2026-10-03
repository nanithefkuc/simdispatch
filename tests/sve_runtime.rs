//! Runtime verification of the SVE capability surface on a real or emulated
//! AArch64 host, unlike the compile-only `sve_intrinsics` example.
//!
//! The crate is launched per configuration by the `sve-runtime` recipe: each
//! invocation fixes one QEMU CPU model and one set of expectations through
//! `SVE_TEST_*` environment variables (plus `SIMD_BACKEND` for override
//! cases). Every expectation is an assertion — a missing or malformed
//! variable fails the run, an unavailable required capability fails the run,
//! and a skipped configuration cannot pass. Where the compiled surface is
//! absent (non-AArch64 hosts, stable toolchains, builds where `build.rs`
//! declines the gate) this file compiles to nothing.
//!
//! Each operation drives real intrinsics over guarded buffers whose lengths
//! straddle the live vector length and compares the result against an
//! independent scalar reference, so a pass certifies the operation actually
//! executed with the expected layout — not merely that the process survived.
//!
//! # Environment contract
//!
//! | Variable | Meaning |
//! | --- | --- |
//! | `SVE_TEST_CASE` | label used in assertion messages |
//! | `SVE_TEST_VL_BYTES` | expected `SveToken::vector_bytes`, a multiple of 16 in 16..=256 |
//! | `SVE_TEST_SVE`, `SVE_TEST_SVE2`, `SVE_TEST_SVE2_AES` | expected token availability, `0` or `1` |
//! | `SVE_TEST_BACKEND` | expected resolved backend name after the `SIMD_BACKEND` override policy |
//! | `SVE_TEST_OPS` | comma list of required families from `base,sve2,sve2_aes`, or `none` |

#![deny(unsafe_code)]
#![cfg_attr(simdispatch_sve, feature(stdarch_aarch64_sve))]

// Armed only by the recipe's cross-build RUSTFLAGS: if the canary cfg is set
// while `build.rs` declined the SVE surface, every test below would verify
// nothing. Fail instead of passing vacuously.
#[cfg(all(sve_canary_required, not(simdispatch_sve)))]
compile_error!(
    "sve_canary_required is armed but build.rs declined the SVE surface; \
                the sve-runtime recipe requires an eligible nightly AArch64 build"
);

#[cfg(simdispatch_sve)]
mod runtime {
    use simdispatch::arch::aarch64::{Sve2AesToken, Sve2Token, SveToken};
    use simdispatch::{Backend, Selection};
    use std::env;

    /// Sentinel bytes surrounding every guarded buffer.
    const GUARD: usize = 32;

    /// A pseudorandom stream: deterministic, dependency-free.
    struct XorShift(u64);

    impl XorShift {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.0 = x;
            x
        }

        fn fill_u8(&mut self, buf: &mut [u8]) {
            for chunk in buf.chunks_mut(8) {
                let v = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&v[..chunk.len()]);
            }
        }
    }

    /// One recipe-fixed run configuration, parsed strictly from the
    /// environment. A missing or malformed variable is a failure, never a
    /// skip.
    struct Case {
        label: String,
        vl: usize,
        sve: bool,
        sve2: bool,
        sve2_aes: bool,
        backend: Backend,
        ops: Vec<&'static str>,
    }

    fn required(name: &str) -> String {
        env::var(name).unwrap_or_else(|_| {
            panic!("{name} is not set: the sve-runtime recipe must launch every case with the full SVE_TEST_* contract")
        })
    }

    fn boolean(name: &str) -> bool {
        match required(name).as_str() {
            "1" => true,
            "0" => false,
            other => panic!("{name} must be 0 or 1, got {other:?}"),
        }
    }

    fn case() -> Case {
        let vl: usize = required("SVE_TEST_VL_BYTES")
            .parse()
            .unwrap_or_else(|e| panic!("SVE_TEST_VL_BYTES is not a number: {e}"));
        assert!(
            vl.is_multiple_of(16) && (16..=256).contains(&vl),
            "SVE_TEST_VL_BYTES must be a multiple of 16 in 16..=256, got {vl}"
        );
        let backend = required("SVE_TEST_BACKEND");
        let ops_raw = required("SVE_TEST_OPS");
        let ops: Vec<&'static str> = if ops_raw == "none" {
            Vec::new()
        } else {
            ops_raw
                .split(',')
                .map(|op| match op {
                    "base" => "base",
                    "sve2" => "sve2",
                    "sve2_aes" => "sve2_aes",
                    other => panic!("unknown SVE_TEST_OPS entry {other:?}"),
                })
                .collect()
        };
        for (i, op) in ops.iter().enumerate() {
            assert!(
                !ops[..i].contains(op),
                "duplicate SVE_TEST_OPS entry {op:?}"
            );
        }
        Case {
            label: required("SVE_TEST_CASE"),
            vl,
            sve: boolean("SVE_TEST_SVE"),
            sve2: boolean("SVE_TEST_SVE2"),
            sve2_aes: boolean("SVE_TEST_SVE2_AES"),
            backend: Backend::from_name(&backend)
                .unwrap_or_else(|| panic!("unknown SVE_TEST_BACKEND {backend:?}")),
            ops,
        }
    }

    /// Carryless 8x8 -> 16 product in GF(2)[x].
    fn clmul8(a: u8, b: u8) -> u16 {
        let mut r = 0u16;
        for i in 0..8 {
            if b >> i & 1 == 1 {
                r ^= (a as u16) << i;
            }
        }
        r
    }

    /// Carryless 64x64 -> 128 product in GF(2)[x].
    fn clmul64(a: u64, b: u64) -> u128 {
        let mut r = 0u128;
        for i in 0..64 {
            if b >> i & 1 == 1 {
                r ^= (a as u128) << i;
            }
        }
        r
    }

    /// A heap buffer with intact sentinel guards on both sides of the
    /// payload window.
    struct Guarded {
        buf: Vec<u8>,
        offset: usize,
        len: usize,
    }

    impl Guarded {
        fn filled(seed: u64, len: usize) -> Self {
            let mut g = Guarded {
                buf: vec![0xA5; len + 2 * GUARD],
                offset: GUARD,
                len,
            };
            XorShift(seed | 1).fill_u8(g.payload_mut());
            g
        }

        fn zeroed(len: usize) -> Self {
            Guarded {
                buf: vec![0xA5; len + 2 * GUARD],
                offset: GUARD,
                len,
            }
        }

        fn payload(&self) -> &[u8] {
            &self.buf[self.offset..self.offset + self.len]
        }

        fn payload_mut(&mut self) -> &mut [u8] {
            &mut self.buf[self.offset..self.offset + self.len]
        }

        /// Panics unless every guard byte still holds its sentinel: proof
        /// that no store changed bytes outside the payload.
        fn assert_intact(&self, what: &str) {
            for (i, byte) in self.buf[..self.offset]
                .iter()
                .chain(self.buf[self.offset + self.len..].iter())
                .enumerate()
            {
                assert_eq!(*byte, 0xA5, "{what}: guard byte {i} clobbered");
            }
        }
    }

    fn live_vl() -> usize {
        SveToken::summon()
            .expect("live_vl requires an SVE proof")
            .vector_bytes()
    }

    /// Lengths straddling every boundary: below/above the vector length,
    /// exact multiples, and ragged tails.
    fn lengths(vl: usize) -> Vec<usize> {
        let mut lens = vec![0, 1, 2, 15, 16, 17, 31, 32, 33, 64, 65];
        lens.extend([vl - 1, vl, vl + 1, 2 * vl, 3 * vl + 5]);
        lens.sort_unstable();
        lens.dedup();
        lens
    }

    // SINCE: the caller holds `SveToken`, which `summon` minted only after
    //        archmage's `is_aarch64_feature_available!("sve")` accepted the
    //        probe, so FEAT_SVE is present for this process.
    // THUS:  entering the `#[target_feature(enable = "sve")]` sibling is
    //        sound; the sibling's predicated accesses stay within the
    //        equal-length slices the wrapper validated.
    #[allow(unsafe_code)]
    fn xor_assign_sve(dst: &mut [u8], _proof: SveToken, src: &[u8]) {
        assert_eq!(dst.len(), src.len());
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve"; every ld1b/st1b lane is predicated by
        //        whilelt over the validated length.
        // THUS:  each memory lane addresses an in-bounds byte of the
        //        caller's slices.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve")]
        unsafe fn body(dst: *mut u8, src: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let all = svptrue_b8();
                let mut i = 0usize;
                loop {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    if !svptest_any(all, pg) {
                        break;
                    }
                    let a = svld1_u8(pg, src.add(i));
                    let b = svld1_u8(pg, dst.add(i));
                    svst1_u8(pg, dst.add(i), sveor_u8_x(pg, a, b));
                    i += svcntb() as usize;
                }
            }
        }
        unsafe { body(dst.as_mut_ptr(), src.as_ptr(), dst.len()) }
    }

    // SINCE: the caller holds `SveToken` as above, and `table` spans
    //        exactly the live vector length, so the all-true table load
    //        reads only owned bytes at every vector length.
    // THUS:  entering the sve sibling is sound; the TBL inputs are fully
    //        owned and the predicated index load and result store stay
    //        within the validated slices.
    #[allow(unsafe_code)]
    fn tbl_full_sve(out: &mut [u8], _proof: SveToken, table: &[u8], indices: &[u8]) {
        assert_eq!(table.len(), live_vl(), "table must span the live vector");
        assert_eq!(indices.len(), out.len());
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve"; the wrapper asserted the table length
        //        equals svcntb(), making the all-true table load exact.
        // THUS:  no lane overreads the table or crosses the index/output
        //        bounds.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve")]
        unsafe fn body(out: *mut u8, table: *const u8, indices: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let all = svptrue_b8();
                let t = svld1_u8(all, table);
                let mut i = 0usize;
                loop {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    if !svptest_any(all, pg) {
                        break;
                    }
                    let idx = svld1_u8(pg, indices.add(i));
                    svst1_u8(pg, out.add(i), svtbl_u8(t, idx));
                    i += svcntb() as usize;
                }
            }
        }
        unsafe {
            body(
                out.as_mut_ptr(),
                table.as_ptr(),
                indices.as_ptr(),
                out.len(),
            )
        }
    }

    // SINCE: the caller holds `SveToken` as above, and `table` is exactly
    //        16 bytes: LD1RQ reads one quadword and replicates it, so the
    //        load is bounded by construction at every vector length.
    // THUS:  entering the sve sibling is sound; the 16-byte table is never
    //        overread even where the live vector length exceeds it.
    #[allow(unsafe_code)]
    fn replicate_table_sve(out: &mut [u8], _proof: SveToken, table: &[u8; 16]) {
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve"; svld1rq_u8 reads exactly the 16 bytes
        //        the fixed-size reference owns.
        // THUS:  the replicated table load is in-bounds and the predicated
        //        store stays within `out`.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve")]
        unsafe fn body(out: *mut u8, table: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let t = svld1rq_u8(svptrue_b8(), table);
                let all = svptrue_b8();
                let mut i = 0usize;
                loop {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    if !svptest_any(all, pg) {
                        break;
                    }
                    svst1_u8(pg, out.add(i), t);
                    i += svcntb() as usize;
                }
            }
        }
        unsafe { body(out.as_mut_ptr(), table.as_ptr(), out.len()) }
    }

    // SINCE: the caller holds `Sve2Token`, which `summon` minted only after
    //        FEAT_SVE2 was proven present, so the `sve,sve2` sibling is
    //        executable; the wrapper asserts equal slice lengths.
    // THUS:  the whilelt-bounded EOR3 loads and stores touch only owned bytes.
    #[allow(unsafe_code)]
    fn eor3_sve2(out: &mut [u8], _proof: Sve2Token, a: &[u8], b: &[u8], c: &[u8]) {
        assert_eq!(b.len(), a.len());
        assert_eq!(c.len(), a.len());
        assert_eq!(out.len(), a.len());
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve,sve2", covering sveor3_u8 and the base
        //        loads and stores predicated over the validated slices.
        // THUS:  every intrinsic call is feature-legal and every lane is
        //        in-bounds.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve,sve2")]
        unsafe fn body(out: *mut u8, a: *const u8, b: *const u8, c: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let mut i = 0;
                while i < len {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    let va = svld1_u8(pg, a.add(i));
                    let vb = svld1_u8(pg, b.add(i));
                    let vc = svld1_u8(pg, c.add(i));
                    svst1_u8(pg, out.add(i), sveor3_u8(va, vb, vc));
                    i += svcntb() as usize;
                }
            }
        }
        unsafe {
            body(
                out.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                c.as_ptr(),
                a.len(),
            )
        }
    }

    // SINCE: the caller holds `Sve2Token` as above; svpmul_u8 needs only
    //        FEAT_SVE2 and its inputs are owned equal-length slices.
    // THUS:  entering the sibling is sound and predication bounds memory;
    //        the truncated byte product is checked against an independent
    //        shift/xor reference by the caller.
    #[allow(unsafe_code)]
    fn pmul_sve2(out: &mut [u8], _proof: Sve2Token, a: &[u8], b: &[u8]) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), out.len());
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve,sve2", covering svpmul_u8.
        // THUS:  every lane stays inside the validated slices.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve,sve2")]
        unsafe fn body(out: *mut u8, a: *const u8, b: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let all = svptrue_b8();
                let mut i = 0usize;
                loop {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    if !svptest_any(all, pg) {
                        break;
                    }
                    let va = svld1_u8(pg, a.add(i));
                    let vb = svld1_u8(pg, b.add(i));
                    svst1_u8(pg, out.add(i), svpmul_u8(va, vb));
                    i += svcntb() as usize;
                }
            }
        }
        unsafe { body(out.as_mut_ptr(), a.as_ptr(), b.as_ptr(), a.len()) }
    }

    // SINCE: the caller holds `Sve2Token` as above; `table0` and `table1`
    //        each span exactly the live vector length, so TBL2's
    //        concatenated two-vector table is fully owned.
    // THUS:  entering the sibling is sound; every table lane and every
    //        predicated index/store lane is in-bounds.
    #[allow(unsafe_code)]
    fn tbl2_sve2(out: &mut [u8], _proof: Sve2Token, table0: &[u8], table1: &[u8], indices: &[u8]) {
        let vl = live_vl();
        assert_eq!(table0.len(), vl);
        assert_eq!(table1.len(), vl);
        assert_eq!(indices.len(), out.len());
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve,sve2", covering svtbl2_u8 and svcreate2_u8
        //        over the two validated full-vector table slices.
        // THUS:  the all-true table loads read exactly owned bytes.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve,sve2")]
        unsafe fn body(out: *mut u8, t0: *const u8, t1: *const u8, indices: *const u8, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let all = svptrue_b8();
                let v0 = svld1_u8(all, t0);
                let v1 = svld1_u8(all, t1);
                let table = svcreate2_u8(v0, v1);
                let mut i = 0usize;
                loop {
                    let pg = svwhilelt_b8_u64(i as u64, len as u64);
                    if !svptest_any(all, pg) {
                        break;
                    }
                    let idx = svld1_u8(pg, indices.add(i));
                    svst1_u8(pg, out.add(i), svtbl2_u8(table, idx));
                    i += svcntb() as usize;
                }
            }
        }
        unsafe {
            body(
                out.as_mut_ptr(),
                table0.as_ptr(),
                table1.as_ptr(),
                indices.as_ptr(),
                out.len(),
            )
        }
    }

    // SINCE: the caller holds `Sve2AesToken`, which `summon` minted only
    //        after the SVE2 AES extension was proven present, so the
    //        `sve,sve2,sve2-aes` sibling is executable; the wrapper asserts
    //        an even lane count, and since svcntd() is even (the vector
    //        length is a multiple of 16 bytes) every iteration and tail
    //        starts on a pair edge, so no partial PMULL pair is formed.
    // THUS:  each PMULLB/PMULLT pair reads two owned lanes and writes two
    //        owned lanes, and the paired 64x64->128 layout is checked
    //        against an independent shift/xor reference by the caller.
    #[allow(unsafe_code)]
    fn pmull_pair_sve2aes(
        lo: &mut [u64],
        hi: &mut [u64],
        _proof: Sve2AesToken,
        a: &[u64],
        b: &[u64],
    ) {
        assert_eq!(a.len(), b.len());
        assert_eq!(a.len(), lo.len());
        assert_eq!(a.len(), hi.len());
        assert!(
            a.len().is_multiple_of(2),
            "paired products need whole 64-bit lane pairs, got {}",
            a.len()
        );
        // SINCE: this sibling is private to the proof-gated wrapper above
        //        and enables "sve,sve2,sve2-aes", covering
        //        svpmullb_pair_u64 / svpmullt_pair_u64; the stride of one
        //        svcntd() per iteration keeps every pair aligned.
        // THUS:  no partial segment loads unproven lanes and the predicated
        //        stores stay within the validated slices.
        #[allow(unsafe_code)]
        #[target_feature(enable = "sve,sve2,sve2-aes")]
        unsafe fn body(lo: *mut u64, hi: *mut u64, a: *const u64, b: *const u64, len: usize) {
            use simdispatch::arch::aarch64::*;
            unsafe {
                let mut i = 0usize;
                while i < len {
                    let pg = svwhilelt_b64_u64(i as u64, len as u64);
                    let va = svld1_u64(pg, a.add(i));
                    let vb = svld1_u64(pg, b.add(i));
                    svst1_u64(pg, lo.add(i), svpmullb_pair_u64(va, vb));
                    svst1_u64(pg, hi.add(i), svpmullt_pair_u64(va, vb));
                    i += svcntd() as usize;
                }
            }
        }
        unsafe {
            body(
                lo.as_mut_ptr(),
                hi.as_mut_ptr(),
                a.as_ptr(),
                b.as_ptr(),
                a.len(),
            )
        }
    }

    /// Sentinel words surrounding a natively typed `u64`-lane buffer. The
    /// payload needs no reinterpretation, so the SVE2-AES lane tests never
    /// depend on a byte allocation's alignment.
    struct Guarded64 {
        buf: Vec<u64>,
        offset: usize,
        len: usize,
    }

    /// Guard words on each side of a [`Guarded64`] payload.
    const GUARD64: usize = 4;

    impl Guarded64 {
        fn filled(seed: u64, len: usize) -> Self {
            let mut g = Guarded64 {
                buf: vec![0xA5A5_A5A5_A5A5_A5A5; len + 2 * GUARD64],
                offset: GUARD64,
                len,
            };
            let mut rng = XorShift(seed | 1);
            for w in g.payload_mut() {
                *w = rng.next_u64();
            }
            g
        }

        fn zeroed(len: usize) -> Self {
            Guarded64 {
                buf: vec![0xA5A5_A5A5_A5A5_A5A5; len + 2 * GUARD64],
                offset: GUARD64,
                len,
            }
        }

        fn payload(&self) -> &[u64] {
            &self.buf[self.offset..self.offset + self.len]
        }

        fn payload_mut(&mut self) -> &mut [u64] {
            &mut self.buf[self.offset..self.offset + self.len]
        }

        /// Panics unless every guard word still holds its sentinel.
        fn assert_intact(&self, what: &str) {
            for (i, word) in self.buf[..self.offset]
                .iter()
                .chain(self.buf[self.offset + self.len..].iter())
                .enumerate()
            {
                assert_eq!(
                    *word, 0xA5A5_A5A5_A5A5_A5A5,
                    "{what}: guard word {i} clobbered"
                );
            }
        }
    }

    #[test]
    fn sve_runtime_verifier() {
        let c = case();
        println!("case: {}", c.label);

        // -- capability ------------------------------------------------
        let sve = SveToken::summon();
        let sve2 = Sve2Token::summon();
        let sve2_aes = Sve2AesToken::summon();
        println!(
            "tokens: sve={} sve2={} sve2_aes={}",
            sve.is_some(),
            sve2.is_some(),
            sve2_aes.is_some()
        );
        assert_eq!(sve.is_some(), c.sve, "{}: SVE token mismatch", c.label);
        assert_eq!(sve2.is_some(), c.sve2, "{}: SVE2 token mismatch", c.label);
        assert_eq!(
            sve2_aes.is_some(),
            c.sve2_aes,
            "{}: SVE2-AES token mismatch",
            c.label
        );

        // -- widths ----------------------------------------------------
        for tier in [Backend::Sve2Aes, Backend::Sve2, Backend::Sve] {
            assert_eq!(tier.lane_bytes(), 16, "{}: const lane_bytes", c.label);
        }
        if let Some(token) = sve {
            let vl = token.vector_bytes();
            println!("measured vector_bytes: {vl}");
            assert_eq!(vl, c.vl, "{}: live vector length", c.label);
            for tier in [Backend::Sve2Aes, Backend::Sve2, Backend::Sve] {
                assert_eq!(
                    tier.lane_bytes_on_host(),
                    c.vl,
                    "{}: {tier:?} live width",
                    c.label
                );
            }
        } else {
            for tier in [Backend::Sve2Aes, Backend::Sve2, Backend::Sve] {
                assert_eq!(
                    tier.lane_bytes_on_host(),
                    16,
                    "{}: {tier:?} must report the minimum without a proof",
                    c.label
                );
            }
        }

        // -- backend resolution and the real SIMD_BACKEND override -----
        let resolved = Selection::new("SIMD_BACKEND")
            .supports(Backend::ALL)
            .resolve();
        let expected = c.backend;
        assert_eq!(
            resolved, expected,
            "{}: resolved backend after override policy",
            c.label
        );
        assert_eq!(
            simdispatch::backend(),
            expected,
            "{}: process-wide backend",
            c.label
        );
        println!("resolved backend: {}", resolved.name());

        // -- required-operation families -------------------------------
        for family in ["base", "sve2", "sve2_aes"] {
            if !c.ops.contains(&family) {
                continue;
            }
            let present = match family {
                "base" => sve.is_some(),
                "sve2" => sve2.is_some(),
                _ => sve2_aes.is_some(),
            };
            assert!(
                present,
                "{}: SVE_TEST_OPS requires {family} but its token is absent",
                c.label
            );
        }

        // -- base SVE operations ----------------------------------------
        if let Some(token) = sve {
            let vl = token.vector_bytes();
            let lens = lengths(vl);

            for len in &lens {
                let mut dst = Guarded::filled(0x9E37_79B9_7F4A_7C15 ^ *len as u64, *len);
                let src = Guarded::filled(0xDEAD_BEEF_CAFE_F00D ^ *len as u64, *len);
                let old: Vec<u8> = dst.payload().to_vec();
                xor_assign_sve(dst.payload_mut(), token, src.payload());
                dst.assert_intact("xor_assign");
                src.assert_intact("xor_assign source");
                for (j, want) in old
                    .iter()
                    .zip(src.payload())
                    .map(|(d, s)| d ^ s)
                    .enumerate()
                {
                    assert_eq!(dst.payload()[j], want, "xor_assign len {len} byte {j}");
                }
            }
            println!("base: predicated XOR verified over {} lengths", lens.len());

            // full-vector TBL: the table spans the live VL; indices mix
            // in-range values with out-of-range ones (architectural 0).
            let table = Guarded::filled(0x5555_AAAA_5555_AAAA, vl);
            let oor = vl + 3; // fits a byte lane only below VL 253
            for len in &lens {
                let mut indices = Guarded::filled(0x3333_CCCC_3333_CCCC ^ *len as u64, *len);
                for (i, j) in indices.payload_mut().iter_mut().enumerate() {
                    let edges = [0, 15, vl - 1, oor.min(255), 255];
                    *j = if i < edges.len() {
                        edges[i] as u8
                    } else {
                        ((*j).wrapping_mul(31) as usize % vl) as u8
                    };
                }
                let mut out = Guarded::zeroed(*len);
                tbl_full_sve(out.payload_mut(), token, table.payload(), indices.payload());
                out.assert_intact("tbl_full");
                indices.assert_intact("tbl_full indices");
                table.assert_intact("tbl_full table");
                for j in 0..*len {
                    let idx = indices.payload()[j] as usize;
                    let want = if idx < vl { table.payload()[idx] } else { 0 };
                    assert_eq!(out.payload()[j], want, "tbl_full len {len} byte {j}");
                }
            }
            println!("base: full-vector TBL verified (in-range and zeroing indices)");

            // bounded 16-byte replicate load at every VL
            let mut table16 = [0u8; 16];
            XorShift(0x0F0F_0F0F_0F0F_0F0F).fill_u8(&mut table16);
            for len in &lens {
                let mut out = Guarded::zeroed(*len);
                replicate_table_sve(out.payload_mut(), token, &table16);
                out.assert_intact("replicate_table");
                for (j, out_byte) in out.payload().iter().enumerate() {
                    assert_eq!(
                        *out_byte,
                        table16[j % 16],
                        "replicate_table len {len} byte {j}"
                    );
                }
            }
            println!("base: ld1rq bounded table replication verified");
        }

        // -- SVE2 operations --------------------------------------------
        if let Some(token) = sve2 {
            let vl = token.sve().vector_bytes();
            let lens = lengths(vl);

            // EOR3 across full-vector boundaries and predicated tails.
            for &len in &lens {
                let a = Guarded::filled(0x1111_2222_3333_4444, len);
                let b = Guarded::filled(0x5555_6666_7777_8888, len);
                let d = Guarded::filled(0x9999_AAAA_BBBB_CCCC, len);
                let mut out = Guarded::zeroed(len);
                eor3_sve2(
                    out.payload_mut(),
                    token,
                    a.payload(),
                    b.payload(),
                    d.payload(),
                );
                a.assert_intact("eor3");
                b.assert_intact("eor3");
                d.assert_intact("eor3");
                out.assert_intact("eor3");
                for j in 0..len {
                    assert_eq!(
                        out.payload()[j],
                        a.payload()[j] ^ b.payload()[j] ^ d.payload()[j],
                        "eor3 len {len} byte {j}"
                    );
                }
            }
            println!("sve2: EOR3 parity verified over {} lengths", lens.len());

            // PMUL truncated byte product over straddling lengths
            for len in &lens {
                let a = Guarded::filled(0xAAAA_BEEE_0000_1111 ^ *len as u64, *len);
                let b = Guarded::filled(0xBBBB_F00D_2222_3333 ^ *len as u64, *len);
                let mut out = Guarded::zeroed(*len);
                pmul_sve2(out.payload_mut(), token, a.payload(), b.payload());
                out.assert_intact("pmul");
                for j in 0..*len {
                    let want = clmul8(a.payload()[j], b.payload()[j]) as u8;
                    assert_eq!(out.payload()[j], want, "pmul len {len} byte {j}");
                }
            }
            println!(
                "sve2: PMUL verified against clmul8 over {} lengths",
                lens.len()
            );

            // TBL2 over the concatenated two-vector table
            let t0 = Guarded::filled(0xC0DE_C0DE_C0DE_C0DE, vl);
            let t1 = Guarded::filled(0xFEED_FEED_FEED_FEED, vl);
            let bound = (2 * vl + 3).min(256); // byte lanes cap at 255
            for len in &lens {
                let mut indices = Guarded::filled(0xD00D_D00D_5555_6666 ^ *len as u64, *len);
                for (i, j) in indices.payload_mut().iter_mut().enumerate() {
                    let edges = [0, vl - 1, vl, 2 * vl - 1, 2 * vl, 255];
                    *j = if i < edges.len() {
                        edges[i].min(255) as u8
                    } else {
                        ((*j).wrapping_mul(7) as usize % bound) as u8
                    };
                }
                let mut out = Guarded::zeroed(*len);
                tbl2_sve2(
                    out.payload_mut(),
                    token,
                    t0.payload(),
                    t1.payload(),
                    indices.payload(),
                );
                out.assert_intact("tbl2");
                for j in 0..*len {
                    let idx = indices.payload()[j] as usize;
                    let want = if idx < vl {
                        t0.payload()[idx]
                    } else if idx < 2 * vl {
                        t1.payload()[idx - vl]
                    } else {
                        0
                    };
                    assert_eq!(out.payload()[j], want, "tbl2 len {len} byte {j}");
                }
            }
            println!("sve2: TBL2 two-vector table geometry verified");
        }

        // -- SVE2-AES paired carryless products -------------------------
        if let Some(token) = sve2_aes {
            let lanes64 = token.sve2().sve().vector_bytes() / 8;
            // even lane counts only: each PMULL pair consumes two lanes
            let mut lens = vec![0, 2, 4, lanes64, lanes64 + 2, 2 * lanes64, 3 * lanes64 + 2];
            lens.sort_unstable();
            lens.dedup();
            for &len in &lens {
                let a = Guarded64::filled(0x600D_600D_600D_600D ^ len as u64, len);
                let b = Guarded64::filled(0xBAD0_BAD0_BAD0_BAD0 ^ len as u64, len);
                let mut lo = Guarded64::zeroed(len);
                let mut hi = Guarded64::zeroed(len);
                pmull_pair_sve2aes(
                    lo.payload_mut(),
                    hi.payload_mut(),
                    token,
                    a.payload(),
                    b.payload(),
                );
                lo.assert_intact("pmullb_pair");
                hi.assert_intact("pmullt_pair");
                for s in 0..len / 2 {
                    let got_b =
                        lo.payload()[2 * s] as u128 | (lo.payload()[2 * s + 1] as u128) << 64;
                    assert_eq!(
                        got_b,
                        clmul64(a.payload()[2 * s], b.payload()[2 * s]),
                        "pmullb_pair len {len} segment {s}: bottom-half 64x64->128 layout"
                    );
                    let got_t =
                        hi.payload()[2 * s] as u128 | (hi.payload()[2 * s + 1] as u128) << 64;
                    assert_eq!(
                        got_t,
                        clmul64(a.payload()[2 * s + 1], b.payload()[2 * s + 1]),
                        "pmullt_pair len {len} segment {s}: top-half 64x64->128 layout"
                    );
                }
            }
            println!(
                "sve2_aes: PMULLB/PMULLT pair layout verified at {lanes64} u64 lanes per vector over {} even lengths",
                lens.len()
            );
        }

        println!("case {}: verified", c.label);
    }
}
