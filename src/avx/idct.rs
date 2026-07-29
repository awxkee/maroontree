/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#![allow(clippy::needless_range_loop)]
use crate::idct::IdctDequant;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
struct I32x8(__m256i);

#[inline]
#[target_feature(enable = "avx2")]
fn splat(v: i32) -> __m256i {
    _mm256_set1_epi32(v)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: *const i32) -> I32x8 {
    unsafe { I32x8(_mm256_loadu_si256(src.cast::<__m256i>())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8_partial(src: *const i32, valid: usize) -> I32x8 {
    match valid {
        8 => load_i32x8(src),
        4 => {
            let lo = unsafe { _mm_loadu_si128(src.cast()) };
            I32x8(_mm256_inserti128_si256::<0>(_mm256_setzero_si256(), lo))
        }
        _ => unreachable!("inverse AVX2 groups contain four or eight lanes"),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8(dst: *mut i32, v: I32x8) {
    unsafe { _mm256_storeu_si256(dst.cast::<__m256i>(), v.0) };
}

impl I32x8 {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn add(self, rhs: I32x8) -> I32x8 {
        I32x8(_mm256_add_epi32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn sub(self, rhs: I32x8) -> I32x8 {
        I32x8(_mm256_sub_epi32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn muli(self, k: i32) -> I32x8 {
        I32x8(_mm256_mullo_epi32(self.0, splat(k)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn rsh<const SH: i32>(self, add: i32) -> I32x8 {
        I32x8(_mm256_srai_epi32::<SH>(_mm256_add_epi32(
            self.0,
            splat(add),
        )))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn clip(self, min: __m256i, max: __m256i) -> I32x8 {
        I32x8(_mm256_min_epi32(_mm256_max_epi32(self.0, min), max))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn neg(self) -> I32x8 {
        I32x8(_mm256_sub_epi32(_mm256_setzero_si256(), self.0))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8(c: &mut [I32x8; 8]) {
    let r0 = c[0].0;
    let r1 = c[1].0;
    let r2 = c[2].0;
    let r3 = c[3].0;
    let r4 = c[4].0;
    let r5 = c[5].0;
    let r6 = c[6].0;
    let r7 = c[7].0;

    let t0 = _mm256_unpacklo_epi32(r0, r1);
    let t1 = _mm256_unpackhi_epi32(r0, r1);
    let t2 = _mm256_unpacklo_epi32(r2, r3);
    let t3 = _mm256_unpackhi_epi32(r2, r3);
    let t4 = _mm256_unpacklo_epi32(r4, r5);
    let t5 = _mm256_unpackhi_epi32(r4, r5);
    let t6 = _mm256_unpacklo_epi32(r6, r7);
    let t7 = _mm256_unpackhi_epi32(r6, r7);

    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    c[0] = I32x8(_mm256_permute2x128_si256::<0x20>(u0, u4));
    c[1] = I32x8(_mm256_permute2x128_si256::<0x20>(u1, u5));
    c[2] = I32x8(_mm256_permute2x128_si256::<0x20>(u2, u6));
    c[3] = I32x8(_mm256_permute2x128_si256::<0x20>(u3, u7));
    c[4] = I32x8(_mm256_permute2x128_si256::<0x31>(u0, u4));
    c[5] = I32x8(_mm256_permute2x128_si256::<0x31>(u1, u5));
    c[6] = I32x8(_mm256_permute2x128_si256::<0x31>(u2, u6));
    c[7] = I32x8(_mm256_permute2x128_si256::<0x31>(u3, u7));
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_store_8x8(dst: *mut i32, stride: usize, tile: &mut [I32x8; 8]) {
    transpose_8x8(tile);
    for i in 0..8usize {
        store_i32x8(unsafe { dst.add(i * stride) }, tile[i]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_i32(r0: __m128i, r1: __m128i, r2: __m128i, r3: __m128i) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(r0, r1);
    let t1 = _mm_unpackhi_epi32(r0, r1);
    let t2 = _mm_unpacklo_epi32(r2, r3);
    let t3 = _mm_unpackhi_epi32(r2, r3);
    [
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    ]
}

/// Transpose four frequency vectors into four- or eight-row scratch. AVX2
/// lanes are split into 128-bit halves so W=4 and H=4 tails never need scalar
/// lane extraction or write beyond the transform.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_store_4x8(dst: *mut i32, stride: usize, tile: &[I32x8; 4], valid: usize) {
    debug_assert!(valid == 4 || valid == 8);
    let lo = transpose_4x4_i32(
        _mm256_castsi256_si128(tile[0].0),
        _mm256_castsi256_si128(tile[1].0),
        _mm256_castsi256_si128(tile[2].0),
        _mm256_castsi256_si128(tile[3].0),
    );
    for (row, value) in lo.into_iter().enumerate() {
        unsafe { _mm_storeu_si128(dst.add(row * stride).cast(), value) };
    }
    if valid == 8 {
        let hi = transpose_4x4_i32(
            _mm256_extracti128_si256::<1>(tile[0].0),
            _mm256_extracti128_si256::<1>(tile[1].0),
            _mm256_extracti128_si256::<1>(tile[2].0),
            _mm256_extracti128_si256::<1>(tile[3].0),
        );
        for (row, value) in hi.into_iter().enumerate() {
            unsafe { _mm_storeu_si128(dst.add((row + 4) * stride).cast(), value) };
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_transposed_rows_i32x8<const N: usize>(dst: *mut i32, y: usize, rows: &[I32x8; N]) {
    debug_assert!(N.is_multiple_of(8));
    let stride = N;
    let mut x = 0usize;
    while x < N {
        let mut tile = [
            rows[x],
            rows[x + 1],
            rows[x + 2],
            rows[x + 3],
            rows[x + 4],
            rows[x + 5],
            rows[x + 6],
            rows[x + 7],
        ];
        transpose_store_8x8(unsafe { dst.add(y * N + x) }, stride, &mut tile);
        x += 8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant8<const DQ1: bool>(lvl: __m256i, q: __m256i, cf_max: __m256i) -> __m256i {
    let absl = _mm256_abs_epi32(lvl);
    // Only the low 24 product bits are observable, so a full-lane i32
    // multiply is exact even when the mathematical product exceeds i32.
    let masked = _mm256_and_si256(_mm256_mullo_epi32(absl, q), splat(0x00ff_ffff));
    let masked = if DQ1 {
        _mm256_srli_epi32::<1>(masked)
    } else {
        masked
    };
    let sign = _mm256_srai_epi32::<31>(lvl);
    let neg_one = _mm256_and_si256(sign, splat(1));
    let cap = _mm256_add_epi32(cf_max, neg_one);
    let mag = _mm256_min_epi32(masked, cap);
    _mm256_sign_epi32(mag, lvl)
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant_q8<const QM: bool>(dequant: &IdctDequant, rc: usize) -> __m256i {
    let ac = splat(dequant.ac_q);
    let base = if rc == 0 {
        _mm256_blend_epi32::<0b0000_0001>(ac, splat(dequant.dc_q))
    } else {
        ac
    };
    if !QM {
        return base;
    }
    let qm = dequant.qm.expect("QM dequant path requires a matrix");
    debug_assert!(rc + 8 <= qm.len());
    let weights8 = unsafe { _mm_loadl_epi64(qm.as_ptr().add(rc).cast::<__m128i>()) };
    let weights = _mm256_cvtepu8_epi32(weights8);
    _mm256_srli_epi32::<5>(_mm256_add_epi32(
        _mm256_mullo_epi32(base, weights),
        splat(16),
    ))
}

#[target_feature(enable = "avx2")]
fn dequant_levels<const N: usize, const DQ1: bool>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    if dequant.qm.is_some() {
        dequant_levels_impl::<N, DQ1, true>(levels, dequant)
    } else {
        dequant_levels_impl::<N, DQ1, false>(levels, dequant)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dequant_levels_impl<const N: usize, const DQ1: bool, const QM: bool>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    let mut coeff_u = MaybeUninit::<[i32; N]>::uninit();
    let coeff_ptr = coeff_u.as_mut_ptr() as *mut i32;
    let cfm = splat(dequant.cf_max);
    let (level_chunks, level_tail) = levels.as_chunks::<8>();
    debug_assert!(level_tail.is_empty());
    for (chunk_index, level) in level_chunks.iter().enumerate() {
        let rc = chunk_index * 8;
        let level = unsafe { _mm256_loadu_si256(level.as_ptr().cast::<__m256i>()) };
        let coeff = dequant8::<DQ1>(level, dequant_q8::<QM>(dequant, rc), cfm);
        unsafe {
            _mm256_storeu_si256(coeff_ptr.add(rc).cast::<__m256i>(), coeff);
        }
    }
    unsafe { coeff_u.assume_init() }
}

#[derive(Clone, Copy)]
struct I16x16(__m256i);

impl I16x16 {
    #[inline]
    #[target_feature(enable = "avx2")]
    fn qadd(self, rhs: Self) -> Self {
        Self(_mm256_adds_epi16(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn qsub(self, rhs: Self) -> Self {
        Self(_mm256_subs_epi16(self.0, rhs.0))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn pack_i32x16(lo: __m256i, hi: __m256i) -> I16x16 {
    // vpackssdw is lane-local; swap the middle 64-bit chunks back into
    // sequential [lo0..lo7, hi0..hi7] order.
    I16x16(_mm256_permute4x64_epi64::<0xd8>(_mm256_packs_epi32(lo, hi)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn rot2_rsh_s16<const SH: i32>(a: I16x16, ka: i16, b: I16x16, kb: i16) -> I16x16 {
    let ab_lo = _mm256_unpacklo_epi16(a.0, b.0);
    let ab_hi = _mm256_unpackhi_epi16(a.0, b.0);
    let coeff = _mm256_set1_epi32(i32::from(ka as u16) | (i32::from(kb) << 16));
    let bias = splat(1 << (SH - 1));
    let lo = _mm256_srai_epi32::<SH>(_mm256_add_epi32(_mm256_madd_epi16(ab_lo, coeff), bias));
    let hi = _mm256_srai_epi32::<SH>(_mm256_add_epi32(_mm256_madd_epi16(ab_hi, coeff), bias));
    // unpacklo/hi are lane-local, so packing lo with hi restores lanes 0..15.
    I16x16(_mm256_packs_epi32(lo, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_shift_s16<const SH: i32>(v: I16x16) -> I16x16 {
    // mulhrs(v, 2^(15-SH)) is exactly (v + 2^(SH-1)) >> SH and
    // avoids the signed-16 overflow of an explicit bias add.
    I16x16(_mm256_mulhrs_epi16(v.0, _mm256_set1_epi16(1 << (15 - SH))))
}

#[inline]
#[target_feature(enable = "avx2")]
fn prescale_s16(v: I16x16) -> I16x16 {
    I16x16(_mm256_mulhrs_epi16(v.0, _mm256_set1_epi16(181 << 7)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mid_shift_s16(v: I16x16, shift: i32) -> I16x16 {
    match shift {
        1 => round_shift_s16::<1>(v),
        2 => round_shift_s16::<2>(v),
        _ => unreachable!("unsupported s16 inverse mid-shift"),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_s16(v: &mut [I16x16; 8]) {
    let r0 = _mm256_castsi256_si128(v[0].0);
    let r1 = _mm256_castsi256_si128(v[1].0);
    let r2 = _mm256_castsi256_si128(v[2].0);
    let r3 = _mm256_castsi256_si128(v[3].0);
    let r4 = _mm256_castsi256_si128(v[4].0);
    let r5 = _mm256_castsi256_si128(v[5].0);
    let r6 = _mm256_castsi256_si128(v[6].0);
    let r7 = _mm256_castsi256_si128(v[7].0);

    let a0 = _mm_unpacklo_epi16(r0, r1);
    let a1 = _mm_unpacklo_epi16(r2, r3);
    let a2 = _mm_unpacklo_epi16(r4, r5);
    let a3 = _mm_unpacklo_epi16(r6, r7);
    let a4 = _mm_unpackhi_epi16(r0, r1);
    let a5 = _mm_unpackhi_epi16(r2, r3);
    let a6 = _mm_unpackhi_epi16(r4, r5);
    let a7 = _mm_unpackhi_epi16(r6, r7);

    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpacklo_epi32(a2, a3);
    let b2 = _mm_unpacklo_epi32(a4, a5);
    let b3 = _mm_unpacklo_epi32(a6, a7);
    let b4 = _mm_unpackhi_epi32(a0, a1);
    let b5 = _mm_unpackhi_epi32(a2, a3);
    let b6 = _mm_unpackhi_epi32(a4, a5);
    let b7 = _mm_unpackhi_epi32(a6, a7);
    let rows = [
        _mm_unpacklo_epi64(b0, b1),
        _mm_unpackhi_epi64(b0, b1),
        _mm_unpacklo_epi64(b4, b5),
        _mm_unpackhi_epi64(b4, b5),
        _mm_unpacklo_epi64(b2, b3),
        _mm_unpackhi_epi64(b2, b3),
        _mm_unpacklo_epi64(b6, b7),
        _mm_unpackhi_epi64(b6, b7),
    ];
    let zero = _mm256_setzero_si256();
    for (dst, row) in v.iter_mut().zip(rows) {
        *dst = I16x16(_mm256_inserti128_si256::<0>(zero, row));
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_16x16_s16(v: &mut [I16x16; 16]) {
    let mut a = [v[0].0; 16];
    for i in (0..16).step_by(2) {
        a[i / 2] = _mm256_unpacklo_epi16(v[i].0, v[i + 1].0);
        a[i / 2 + 8] = _mm256_unpackhi_epi16(v[i].0, v[i + 1].0);
    }
    let mut b = [a[0]; 16];
    for i in (0..16).step_by(2) {
        b[i / 2] = _mm256_unpacklo_epi32(a[i], a[i + 1]);
        b[i / 2 + 8] = _mm256_unpackhi_epi32(a[i], a[i + 1]);
    }
    let mut c = [b[0]; 16];
    for i in (0..16).step_by(2) {
        c[i / 2] = _mm256_unpacklo_epi64(b[i], b[i + 1]);
        c[i / 2 + 8] = _mm256_unpackhi_epi64(b[i], b[i + 1]);
    }
    let lo = |x: __m256i, y: __m256i| _mm256_permute2x128_si256::<0x20>(x, y);
    let hi = |x: __m256i, y: __m256i| _mm256_permute2x128_si256::<0x31>(x, y);
    v[0] = I16x16(lo(c[0], c[1]));
    v[1] = I16x16(lo(c[8], c[9]));
    v[2] = I16x16(lo(c[4], c[5]));
    v[3] = I16x16(lo(c[12], c[13]));
    v[8] = I16x16(hi(c[0], c[1]));
    v[9] = I16x16(hi(c[8], c[9]));
    v[10] = I16x16(hi(c[4], c[5]));
    v[11] = I16x16(hi(c[12], c[13]));
    v[4] = I16x16(lo(c[2], c[3]));
    v[5] = I16x16(lo(c[10], c[11]));
    v[6] = I16x16(lo(c[6], c[7]));
    v[7] = I16x16(lo(c[14], c[15]));
    v[12] = I16x16(hi(c[2], c[3]));
    v[13] = I16x16(hi(c[10], c[11]));
    v[14] = I16x16(hi(c[6], c[7]));
    v[15] = I16x16(hi(c[14], c[15]));
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn dequant_levels_s16_avx2<const DQ1: bool>(
    levels: &[i32],
    coeff: &mut [MaybeUninit<i16>],
    dequant: &IdctDequant,
) {
    if dequant.qm.is_some() {
        dequant_levels_s16_avx2_impl::<DQ1, true>(levels, coeff, dequant);
    } else {
        dequant_levels_s16_avx2_impl::<DQ1, false>(levels, coeff, dequant);
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn dequant_levels_s16_avx2_impl<const DQ1: bool, const QM: bool>(
    levels: &[i32],
    coeff: &mut [MaybeUninit<i16>],
    dequant: &IdctDequant,
) {
    debug_assert_eq!(dequant.cf_max, i16::MAX as i32);
    debug_assert_eq!(levels.len(), coeff.len());
    let cfm = splat(dequant.cf_max);
    let (level_chunks, level_tail) = levels.as_chunks::<16>();
    let (coeff_chunks, coeff_tail) = coeff.as_chunks_mut::<16>();
    debug_assert!(level_tail.is_empty());
    debug_assert!(coeff_tail.is_empty());
    for (chunk_index, (level, dst)) in level_chunks.iter().zip(coeff_chunks.iter_mut()).enumerate()
    {
        let rc = chunk_index * 16;
        let lo = unsafe { _mm256_loadu_si256(level[..8].as_ptr().cast()) };
        let hi = unsafe { _mm256_loadu_si256(level[8..].as_ptr().cast()) };
        let lo = dequant8::<DQ1>(lo, dequant_q8::<QM>(dequant, rc), cfm);
        let hi = dequant8::<DQ1>(hi, dequant_q8::<QM>(dequant, rc + 8), cfm);
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast(), pack_i32x16(lo, hi).0) };
    }
}

// Keep the register-only transforms as macros: Rust cannot combine target_feature
// with inline(always), and an out-of-line call here spills the live YMM state.
macro_rules! inv_dct8_v_s16 {
    ($values:expr) => {{
        let c: &mut [I16x16; 8] = $values;
        let (e0, e1, e2, e3) = (c[0], c[2], c[4], c[6]);
        let d0 = rot2_rsh_s16::<8>(e0, 181, e2, 181);
        let d1 = rot2_rsh_s16::<8>(e0, 181, e2, -181);
        let d2 = rot2_rsh_s16::<12>(e1, 1567, e3, 312).qsub(e3);
        let d3 = rot2_rsh_s16::<12>(e1, -312, e3, 1567).qadd(e1);
        let p0 = d0.qadd(d3);
        let p2 = d1.qadd(d2);
        let p4 = d1.qsub(d2);
        let p6 = d0.qsub(d3);
        let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
        let t4a = rot2_rsh_s16::<12>(in1, 799, in7, 79).qsub(in7);
        let t5a0 = rot2_rsh_s16::<11>(in5, 1703, in3, -1138);
        let t6a0 = rot2_rsh_s16::<11>(in5, 1138, in3, 1703);
        let t7a = rot2_rsh_s16::<12>(in1, -79, in7, 799).qadd(in1);
        let t4 = t4a.qadd(t5a0);
        let t5a = t4a.qsub(t5a0);
        let t7 = t7a.qadd(t6a0);
        let t6a = t7a.qsub(t6a0);
        let t5 = rot2_rsh_s16::<8>(t6a, 181, t5a, -181);
        let t6 = rot2_rsh_s16::<8>(t6a, 181, t5a, 181);
        c[0] = p0.qadd(t7);
        c[1] = p2.qadd(t6);
        c[2] = p4.qadd(t5);
        c[3] = p6.qadd(t4);
        c[4] = p6.qsub(t4);
        c[5] = p4.qsub(t5);
        c[6] = p2.qsub(t6);
        c[7] = p0.qsub(t7);
    }};
}

macro_rules! inv_dct16_v_s16 {
    ($values:expr) => {{
        let c: &mut [I16x16; 16] = $values;
        let mut e: [I16x16; 8] = std::array::from_fn(|i| c[2 * i]);
        inv_dct8_v_s16!(&mut e);
        let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
        let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);
        let t8a = rot2_rsh_s16::<12>(in1, 401, in15, 20).qsub(in15);
        let t9a = rot2_rsh_s16::<11>(in9, 1583, in7, -1299);
        let t10a = rot2_rsh_s16::<12>(in5, 1931, in11, 484).qsub(in11);
        let t11a = rot2_rsh_s16::<12>(in13, -176, in3, -1189).qadd(in13);
        let t12a = rot2_rsh_s16::<12>(in13, 1189, in3, -176).qadd(in3);
        let t13a = rot2_rsh_s16::<12>(in5, -484, in11, 1931).qadd(in5);
        let t14a = rot2_rsh_s16::<11>(in9, 1299, in7, 1583);
        let t15a = rot2_rsh_s16::<12>(in1, -20, in15, 401).qadd(in1);
        let t8 = t8a.qadd(t9a);
        let t9 = t8a.qsub(t9a);
        let t10 = t11a.qsub(t10a);
        let t11 = t11a.qadd(t10a);
        let t12 = t12a.qadd(t13a);
        let t13 = t12a.qsub(t13a);
        let t14 = t15a.qsub(t14a);
        let t15 = t15a.qadd(t14a);
        let u9a = rot2_rsh_s16::<12>(t14, 1567, t9, 312).qsub(t9);
        let u14a = rot2_rsh_s16::<12>(t14, -312, t9, 1567).qadd(t14);
        let u10a = rot2_rsh_s16::<12>(t13, 312, t10, -1567).qsub(t13);
        let u13a = rot2_rsh_s16::<12>(t13, 1567, t10, 312).qsub(t10);
        let v8a = t8.qadd(t11);
        let v9 = u9a.qadd(u10a);
        let v10 = u9a.qsub(u10a);
        let v11a = t8.qsub(t11);
        let v12a = t15.qsub(t12);
        let v13 = u14a.qsub(u13a);
        let v14 = u14a.qadd(u13a);
        let v15a = t15.qadd(t12);
        let w10a = rot2_rsh_s16::<8>(v13, 181, v10, -181);
        let w13a = rot2_rsh_s16::<8>(v13, 181, v10, 181);
        let w11 = rot2_rsh_s16::<8>(v12a, 181, v11a, -181);
        let w12 = rot2_rsh_s16::<8>(v12a, 181, v11a, 181);
        c[0] = e[0].qadd(v15a);
        c[1] = e[1].qadd(v14);
        c[2] = e[2].qadd(w13a);
        c[3] = e[3].qadd(w12);
        c[4] = e[4].qadd(w11);
        c[5] = e[5].qadd(w10a);
        c[6] = e[6].qadd(v9);
        c[7] = e[7].qadd(v8a);
        c[8] = e[7].qsub(v8a);
        c[9] = e[6].qsub(v9);
        c[10] = e[5].qsub(w10a);
        c[11] = e[4].qsub(w11);
        c[12] = e[3].qsub(w12);
        c[13] = e[2].qsub(w13a);
        c[14] = e[1].qsub(v14);
        c[15] = e[0].qsub(v15a);
    }};
}

macro_rules! inv_dct32_v_s16 {
    ($values:expr) => {{
        let c: &mut [I16x16; 32] = $values;
    macro_rules! unrotate5 {
        ($a:literal, $b:literal, $d:literal, $e:literal, $f:literal) => {{
            let carry = c[$a];
            c[$a] = c[$b];
            c[$b] = c[$d];
            c[$d] = c[$e];
            c[$e] = c[$f];
            c[$f] = carry;
        }};
    }
    unrotate5!(1, 2, 4, 8, 16);
    unrotate5!(3, 6, 12, 24, 17);
    unrotate5!(5, 10, 20, 9, 18);
    unrotate5!(7, 14, 28, 25, 19);
    unrotate5!(11, 22, 13, 26, 21);
    unrotate5!(15, 30, 29, 27, 23);

    let (even, w) = c.split_at_mut(16);
    let even: &mut [I16x16; 16] = even.try_into().unwrap();
    let w: &mut [I16x16; 16] = w.try_into().unwrap();
    inv_dct16_v_s16!(even);
    w.swap(1, 8);
    w.swap(2, 4);
    w.swap(3, 12);
    w.swap(5, 10);
    w.swap(7, 14);
    w.swap(11, 13);

    let (a, b) = (w[0], w[15]);
    w[0] = rot2_rsh_s16::<12>(a, 201, b, 5).qsub(b);
    w[15] = rot2_rsh_s16::<12>(a, -5, b, 201).qadd(a);
    let (a, b) = (w[1], w[14]);
    w[1] = rot2_rsh_s16::<12>(a, -1061, b, -2751).qadd(a);
    w[14] = rot2_rsh_s16::<12>(a, 2751, b, -1061).qadd(b);
    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh_s16::<12>(a, 1751, b, 393).qsub(b);
    w[13] = rot2_rsh_s16::<12>(a, -393, b, 1751).qadd(a);
    let (a, b) = (w[3], w[12]);
    w[3] = rot2_rsh_s16::<12>(a, -239, b, -1380).qadd(a);
    w[12] = rot2_rsh_s16::<12>(a, 1380, b, -239).qadd(b);
    let (a, b) = (w[4], w[11]);
    w[4] = rot2_rsh_s16::<12>(a, 995, b, 123).qsub(b);
    w[11] = rot2_rsh_s16::<12>(a, -123, b, 995).qadd(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh_s16::<12>(a, -583, b, -2106).qadd(a);
    w[10] = rot2_rsh_s16::<12>(a, 2106, b, -583).qadd(b);
    let (a, b) = (w[6], w[9]);
    w[6] = rot2_rsh_s16::<11>(a, 1220, b, -1645);
    w[9] = rot2_rsh_s16::<11>(a, 1645, b, 1220);
    let (a, b) = (w[7], w[8]);
    w[7] = rot2_rsh_s16::<12>(a, -44, b, -601).qadd(a);
    w[8] = rot2_rsh_s16::<12>(a, 601, b, -44).qadd(b);

    for &(i, j, reverse) in &[
        (0usize, 1usize, false),
        (2, 3, true),
        (4, 5, false),
        (6, 7, true),
        (8, 9, false),
        (10, 11, true),
        (12, 13, false),
        (14, 15, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = b.qsub(a);
            w[j] = b.qadd(a);
        } else {
            w[i] = a.qadd(b);
            w[j] = a.qsub(b);
        }
    }

    let (a, b) = (w[1], w[14]);
    w[1] = rot2_rsh_s16::<12>(b, 799, a, 79).qsub(a);
    w[14] = rot2_rsh_s16::<12>(b, -79, a, 799).qadd(b);
    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh_s16::<12>(b, 79, a, -799).qsub(b);
    w[13] = rot2_rsh_s16::<12>(b, 799, a, 79).qsub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh_s16::<11>(b, 1703, a, -1138);
    w[10] = rot2_rsh_s16::<11>(b, 1138, a, 1703);
    let (a, b) = (w[6], w[9]);
    w[6] = rot2_rsh_s16::<11>(b, -1138, a, -1703);
    w[9] = rot2_rsh_s16::<11>(b, 1703, a, -1138);

    for &(i, j, reverse) in &[
        (0usize, 3usize, false),
        (1, 2, false),
        (4, 7, true),
        (5, 6, true),
        (8, 11, false),
        (9, 10, false),
        (12, 15, true),
        (13, 14, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = b.qsub(a);
            w[j] = b.qadd(a);
        } else {
            w[i] = a.qadd(b);
            w[j] = a.qsub(b);
        }
    }

    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh_s16::<12>(b, 1567, a, 312).qsub(a);
    w[13] = rot2_rsh_s16::<12>(b, -312, a, 1567).qadd(b);
    let (a, b) = (w[3], w[12]);
    w[3] = rot2_rsh_s16::<12>(b, 1567, a, 312).qsub(a);
    w[12] = rot2_rsh_s16::<12>(b, -312, a, 1567).qadd(b);
    let (a, b) = (w[4], w[11]);
    w[4] = rot2_rsh_s16::<12>(b, 312, a, -1567).qsub(b);
    w[11] = rot2_rsh_s16::<12>(b, 1567, a, 312).qsub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh_s16::<12>(b, 312, a, -1567).qsub(b);
    w[10] = rot2_rsh_s16::<12>(b, 1567, a, 312).qsub(a);

    for &(i, j, reverse) in &[
        (0usize, 7usize, false),
        (1, 6, false),
        (2, 5, false),
        (3, 4, false),
        (8, 15, true),
        (9, 14, true),
        (10, 13, true),
        (11, 12, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = b.qsub(a);
            w[j] = b.qadd(a);
        } else {
            w[i] = a.qadd(b);
            w[j] = a.qsub(b);
        }
    }

    for &(i, j) in &[(4usize, 11usize), (5, 10), (6, 9), (7, 8)] {
        let (a, b) = (w[i], w[j]);
        w[i] = rot2_rsh_s16::<8>(b, 181, a, -181);
        w[j] = rot2_rsh_s16::<8>(b, 181, a, 181);
    }
    for i in 0..16 {
        let e = even[i];
        let o = w[15 - i];
        even[i] = e.qadd(o);
        w[15 - i] = e.qsub(o);
    }
    }};
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct8_v(c: &mut [I32x8; 8], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    // --- even half: inv_dct4 on c[0], c[2], c[4], c[6] ---
    let (e0, e1, e2, e3) = (c[0], c[2], c[4], c[6]);
    let d0 = e0.add(e2).muli(181).rsh::<8>(128); // ((in0+in2)*181+128)>>8
    let d1 = e0.sub(e2).muli(181).rsh::<8>(128); // ((in0-in2)*181+128)>>8
    let d2 = e1.muli(1567).sub(e3.muli(-312)).rsh::<12>(2048).sub(e3); // *1567 - *(3784-4096)
    let d3 = e1.muli(-312).add(e3.muli(1567)).rsh::<12>(2048).add(e1);
    let p0 = clip(d0.add(d3)); // even outputs (scalar c[0],c[2],c[4],c[6])
    let p2 = clip(d1.add(d2));
    let p4 = clip(d1.sub(d2));
    let p6 = clip(d0.sub(d3));

    // --- odd half ---
    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    // t4a = ((in1*799 - in7*(4017-4096) + 2048)>>12) - in7   ; (4017-4096) = -79
    let t4a = in1.muli(799).sub(in7.muli(-79)).rsh::<12>(2048).sub(in7);
    let t5a0 = in5.muli(1703).sub(in3.muli(1138)).rsh::<11>(1024);
    let t6a0 = in5.muli(1138).add(in3.muli(1703)).rsh::<11>(1024);
    let t7a = in1.muli(-79).add(in7.muli(799)).rsh::<12>(2048).add(in1);

    let t4 = clip(t4a.add(t5a0));
    let t5a = clip(t4a.sub(t5a0));
    let t7 = clip(t7a.add(t6a0));
    let t6a = clip(t7a.sub(t6a0));

    let t5 = t6a.sub(t5a).muli(181).rsh::<8>(128); // ((t6a-t5a)*181+128)>>8
    let t6 = t6a.add(t5a).muli(181).rsh::<8>(128); // ((t6a+t5a)*181+128)>>8

    // --- combine (scalar t0..t3 are the even outputs p0,p2,p4,p6) ---
    c[0] = clip(p0.add(t7));
    c[1] = clip(p2.add(t6));
    c[2] = clip(p4.add(t5));
    c[3] = clip(p6.add(t4));
    c[4] = clip(p6.sub(t4));
    c[5] = clip(p4.sub(t5));
    c[6] = clip(p2.sub(t6));
    c[7] = clip(p0.sub(t7));
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct4_v(c: &mut [I32x8; 4], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let t0 = in0.add(in2).muli(181).rsh::<8>(128);
    let t1 = in0.sub(in2).muli(181).rsh::<8>(128);
    let t2 = in1.muli(1567).sub(in3.muli(-312)).rsh::<12>(2048).sub(in3);
    let t3 = in1.muli(-312).add(in3.muli(1567)).rsh::<12>(2048).add(in1);
    c[0] = clip(t0.add(t3));
    c[1] = clip(t1.add(t2));
    c[2] = clip(t1.sub(t2));
    c[3] = clip(t0.sub(t3));
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_adst4_v(c: &mut [I32x8; 4], _min: i32, _max: i32) {
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    c[0] = in0
        .muli(1321)
        .add(in2.muli(-293))
        .add(in3.muli(-1614))
        .add(in1.muli(-752))
        .rsh::<12>(2048)
        .add(in2)
        .add(in3)
        .add(in1);
    c[1] = in0
        .muli(-1614)
        .sub(in2.muli(1321))
        .sub(in3.muli(-293))
        .add(in1.muli(-752))
        .rsh::<12>(2048)
        .add(in0)
        .sub(in3)
        .add(in1);
    c[2] = in0.sub(in2).add(in3).muli(209).rsh::<8>(128);
    c[3] = in0
        .muli(-293)
        .add(in2.muli(-1614))
        .sub(in3.muli(1321))
        .sub(in1.muli(-752))
        .rsh::<12>(2048)
        .add(in0)
        .add(in2)
        .sub(in1);
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_adst8_v(c: &mut [I32x8; 8], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);
    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let (in4, in5, in6, in7) = (c[4], c[5], c[6], c[7]);
    let t0a = in7.muli(-20).add(in0.muli(401)).rsh::<12>(2048).add(in7);
    let t1a = in7.muli(401).sub(in0.muli(-20)).rsh::<12>(2048).sub(in0);
    let t2a = in5.muli(-484).add(in2.muli(1931)).rsh::<12>(2048).add(in5);
    let t3a = in5.muli(1931).sub(in2.muli(-484)).rsh::<12>(2048).sub(in2);
    let t4a = in3.muli(1299).add(in4.muli(1583)).rsh::<11>(1024);
    let t5a = in3.muli(1583).sub(in4.muli(1299)).rsh::<11>(1024);
    let t6a = in1.muli(1189).add(in6.muli(-176)).rsh::<12>(2048).add(in6);
    let t7a = in1.muli(-176).sub(in6.muli(1189)).rsh::<12>(2048).add(in1);
    let t0 = clip(t0a.add(t4a));
    let t1 = clip(t1a.add(t5a));
    let mut t2 = clip(t2a.add(t6a));
    let mut t3 = clip(t3a.add(t7a));
    let t4 = clip(t0a.sub(t4a));
    let t5 = clip(t1a.sub(t5a));
    let mut t6 = clip(t2a.sub(t6a));
    let mut t7 = clip(t3a.sub(t7a));
    let t4a = t4.muli(-312).add(t5.muli(1567)).rsh::<12>(2048).add(t4);
    let t5a = t4.muli(1567).sub(t5.muli(-312)).rsh::<12>(2048).sub(t5);
    let t6a = t7.muli(-312).sub(t6.muli(1567)).rsh::<12>(2048).add(t7);
    let t7a = t7.muli(1567).add(t6.muli(-312)).rsh::<12>(2048).add(t6);
    c[0] = clip(t0.add(t2));
    c[7] = clip(t1.add(t3)).neg();
    t2 = clip(t0.sub(t2));
    t3 = clip(t1.sub(t3));
    c[1] = clip(t4a.add(t6a)).neg();
    c[6] = clip(t5a.add(t7a));
    t6 = clip(t4a.sub(t6a));
    t7 = clip(t5a.sub(t7a));
    c[3] = t2.add(t3).muli(181).rsh::<8>(128).neg();
    c[4] = t2.sub(t3).muli(181).rsh::<8>(128);
    c[2] = t6.add(t7).muli(181).rsh::<8>(128);
    c[5] = t6.sub(t7).muli(181).rsh::<8>(128).neg();
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_dct16_v(c: &mut [I32x8; 16], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    let mut e: [I32x8; 8] = std::array::from_fn(|i| c[2 * i]);
    inv_dct8_v(&mut e, min, max);

    // odd inputs (read before any write-back to c)
    let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
    let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);

    // stage 1 ; (4076-4096)=-20, (3612-4096)=-484, (3920-4096)=-176
    let t8a = in1.muli(401).sub(in15.muli(-20)).rsh::<12>(2048).sub(in15);
    let t9a = in9.muli(1583).sub(in7.muli(1299)).rsh::<11>(1024);
    let t10a = in5
        .muli(1931)
        .sub(in11.muli(-484))
        .rsh::<12>(2048)
        .sub(in11);
    let t11a = in13
        .muli(-176)
        .sub(in3.muli(1189))
        .rsh::<12>(2048)
        .add(in13);
    let t12a = in13.muli(1189).add(in3.muli(-176)).rsh::<12>(2048).add(in3);
    let t13a = in5.muli(-484).add(in11.muli(1931)).rsh::<12>(2048).add(in5);
    let t14a = in9.muli(1299).add(in7.muli(1583)).rsh::<11>(1024);
    let t15a = in1.muli(-20).add(in15.muli(401)).rsh::<12>(2048).add(in1);

    // stage 2 (butterflies)
    let t8 = clip(t8a.add(t9a));
    let t9 = clip(t8a.sub(t9a));
    let t10 = clip(t11a.sub(t10a));
    let t11 = clip(t11a.add(t10a));
    let t12 = clip(t12a.add(t13a));
    let t13 = clip(t12a.sub(t13a));
    let t14 = clip(t15a.sub(t14a));
    let t15 = clip(t15a.add(t14a));

    // stage 3 (rotations) ; (3784-4096)=-312
    // t10a' = ((-(t13*(-312) + t10*1567) + 2048)>>12) - t13 = ((t13*312 - t10*1567 + 2048)>>12) - t13
    let u9a = t14.muli(1567).sub(t9.muli(-312)).rsh::<12>(2048).sub(t9);
    let u14a = t14.muli(-312).add(t9.muli(1567)).rsh::<12>(2048).add(t14);
    let u10a = t13.muli(312).sub(t10.muli(1567)).rsh::<12>(2048).sub(t13);
    let u13a = t13.muli(1567).sub(t10.muli(-312)).rsh::<12>(2048).sub(t10);

    // stage 4 (butterflies)
    let v8a = clip(t8.add(t11));
    let v9 = clip(u9a.add(u10a));
    let v10 = clip(u9a.sub(u10a));
    let v11a = clip(t8.sub(t11));
    let v12a = clip(t15.sub(t12));
    let v13 = clip(u14a.sub(u13a));
    let v14 = clip(u14a.add(u13a));
    let v15a = clip(t15.add(t12));

    // stage 5 (181/256 rotations)
    let w10a = v13.sub(v10).muli(181).rsh::<8>(128);
    let w13a = v13.add(v10).muli(181).rsh::<8>(128);
    let w11 = v12a.sub(v11a).muli(181).rsh::<8>(128);
    let w12 = v12a.add(v11a).muli(181).rsh::<8>(128);

    // combine with even outputs e[0..8] (= scalar t0..t7)
    c[0] = clip(e[0].add(v15a));
    c[1] = clip(e[1].add(v14));
    c[2] = clip(e[2].add(w13a));
    c[3] = clip(e[3].add(w12));
    c[4] = clip(e[4].add(w11));
    c[5] = clip(e[5].add(w10a));
    c[6] = clip(e[6].add(v9));
    c[7] = clip(e[7].add(v8a));
    c[8] = clip(e[7].sub(v8a));
    c[9] = clip(e[6].sub(v9));
    c[10] = clip(e[5].sub(w10a));
    c[11] = clip(e[4].sub(w11));
    c[12] = clip(e[3].sub(w12));
    c[13] = clip(e[2].sub(w13a));
    c[14] = clip(e[1].sub(v14));
    c[15] = clip(e[0].sub(v15a));
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv_adst16_v(c: &mut [I32x8; 16], min: i32, max: i32) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);

    let (in0, in1, in2, in3) = (c[0], c[1], c[2], c[3]);
    let (in4, in5, in6, in7) = (c[4], c[5], c[6], c[7]);
    let (in8, in9, in10, in11) = (c[8], c[9], c[10], c[11]);
    let (in12, in13, in14, in15) = (c[12], c[13], c[14], c[15]);

    let mut t0 = in15
        .muli(4091 - 4096)
        .add(in0.muli(201))
        .rsh::<12>(2048)
        .add(in15);
    let mut t1 = in15
        .muli(201)
        .sub(in0.muli(4091 - 4096))
        .rsh::<12>(2048)
        .sub(in0);
    let mut t2 = in13
        .muli(3973 - 4096)
        .add(in2.muli(995))
        .rsh::<12>(2048)
        .add(in13);
    let mut t3 = in13
        .muli(995)
        .sub(in2.muli(3973 - 4096))
        .rsh::<12>(2048)
        .sub(in2);
    let mut t4 = in11
        .muli(3703 - 4096)
        .add(in4.muli(1751))
        .rsh::<12>(2048)
        .add(in11);
    let mut t5 = in11
        .muli(1751)
        .sub(in4.muli(3703 - 4096))
        .rsh::<12>(2048)
        .sub(in4);
    let mut t6 = in9.muli(1645).add(in6.muli(1220)).rsh::<11>(1024);
    let mut t7 = in9.muli(1220).sub(in6.muli(1645)).rsh::<11>(1024);
    let mut t8 = in7
        .muli(2751)
        .add(in8.muli(3035 - 4096))
        .rsh::<12>(2048)
        .add(in8);
    let mut t9 = in7
        .muli(3035 - 4096)
        .sub(in8.muli(2751))
        .rsh::<12>(2048)
        .add(in7);
    let mut t10 = in5
        .muli(2106)
        .add(in10.muli(3513 - 4096))
        .rsh::<12>(2048)
        .add(in10);
    let mut t11 = in5
        .muli(3513 - 4096)
        .sub(in10.muli(2106))
        .rsh::<12>(2048)
        .add(in5);
    let mut t12 = in3
        .muli(1380)
        .add(in12.muli(3857 - 4096))
        .rsh::<12>(2048)
        .add(in12);
    let mut t13 = in3
        .muli(3857 - 4096)
        .sub(in12.muli(1380))
        .rsh::<12>(2048)
        .add(in3);
    let mut t14 = in1
        .muli(601)
        .add(in14.muli(4052 - 4096))
        .rsh::<12>(2048)
        .add(in14);
    let mut t15 = in1
        .muli(4052 - 4096)
        .sub(in14.muli(601))
        .rsh::<12>(2048)
        .add(in1);

    let t0a = clip(t0.add(t8));
    let t1a = clip(t1.add(t9));
    let t2a = clip(t2.add(t10));
    let t3a = clip(t3.add(t11));
    let t4a = clip(t4.add(t12));
    let t5a = clip(t5.add(t13));
    let t6a = clip(t6.add(t14));
    let t7a = clip(t7.add(t15));
    let mut t8a = clip(t0.sub(t8));
    let mut t9a = clip(t1.sub(t9));
    let mut t10a = clip(t2.sub(t10));
    let mut t11a = clip(t3.sub(t11));
    let mut t12a = clip(t4.sub(t12));
    let mut t13a = clip(t5.sub(t13));
    let mut t14a = clip(t6.sub(t14));
    let mut t15a = clip(t7.sub(t15));

    t8 = t8a
        .muli(4017 - 4096)
        .add(t9a.muli(799))
        .rsh::<12>(2048)
        .add(t8a);
    t9 = t8a
        .muli(799)
        .sub(t9a.muli(4017 - 4096))
        .rsh::<12>(2048)
        .sub(t9a);
    t10 = t10a
        .muli(2276)
        .add(t11a.muli(3406 - 4096))
        .rsh::<12>(2048)
        .add(t11a);
    t11 = t10a
        .muli(3406 - 4096)
        .sub(t11a.muli(2276))
        .rsh::<12>(2048)
        .add(t10a);
    t12 = t13a
        .muli(4017 - 4096)
        .sub(t12a.muli(799))
        .rsh::<12>(2048)
        .add(t13a);
    t13 = t13a
        .muli(799)
        .add(t12a.muli(4017 - 4096))
        .rsh::<12>(2048)
        .add(t12a);
    t14 = t15a
        .muli(2276)
        .sub(t14a.muli(3406 - 4096))
        .rsh::<12>(2048)
        .sub(t14a);
    t15 = t15a
        .muli(3406 - 4096)
        .add(t14a.muli(2276))
        .rsh::<12>(2048)
        .add(t15a);

    t0 = clip(t0a.add(t4a));
    t1 = clip(t1a.add(t5a));
    t2 = clip(t2a.add(t6a));
    t3 = clip(t3a.add(t7a));
    t4 = clip(t0a.sub(t4a));
    t5 = clip(t1a.sub(t5a));
    t6 = clip(t2a.sub(t6a));
    t7 = clip(t3a.sub(t7a));
    t8a = clip(t8.add(t12));
    t9a = clip(t9.add(t13));
    t10a = clip(t10.add(t14));
    t11a = clip(t11.add(t15));
    t12a = clip(t8.sub(t12));
    t13a = clip(t9.sub(t13));
    t14a = clip(t10.sub(t14));
    t15a = clip(t11.sub(t15));

    let t4b = t4
        .muli(3784 - 4096)
        .add(t5.muli(1567))
        .rsh::<12>(2048)
        .add(t4);
    let t5b = t4
        .muli(1567)
        .sub(t5.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t5);
    let t6b = t7
        .muli(3784 - 4096)
        .sub(t6.muli(1567))
        .rsh::<12>(2048)
        .add(t7);
    let t7b = t7
        .muli(1567)
        .add(t6.muli(3784 - 4096))
        .rsh::<12>(2048)
        .add(t6);
    t12 = t12a
        .muli(3784 - 4096)
        .add(t13a.muli(1567))
        .rsh::<12>(2048)
        .add(t12a);
    t13 = t12a
        .muli(1567)
        .sub(t13a.muli(3784 - 4096))
        .rsh::<12>(2048)
        .sub(t13a);
    t14 = t15a
        .muli(3784 - 4096)
        .sub(t14a.muli(1567))
        .rsh::<12>(2048)
        .add(t15a);
    t15 = t15a
        .muli(1567)
        .add(t14a.muli(3784 - 4096))
        .rsh::<12>(2048)
        .add(t14a);

    c[0] = clip(t0.add(t2));
    c[15] = clip(t1.add(t3)).neg();
    let t2b = clip(t0.sub(t2));
    let t3b = clip(t1.sub(t3));
    c[3] = clip(t4b.add(t6b)).neg();
    c[12] = clip(t5b.add(t7b));
    t6 = clip(t4b.sub(t6b));
    t7 = clip(t5b.sub(t7b));
    c[1] = clip(t8a.add(t10a)).neg();
    c[14] = clip(t9a.add(t11a));
    t10 = clip(t8a.sub(t10a));
    t11 = clip(t9a.sub(t11a));
    c[2] = clip(t12.add(t14));
    c[13] = clip(t13.add(t15)).neg();
    let t14b = clip(t12.sub(t14));
    let t15b = clip(t13.sub(t15));
    c[7] = t2b.add(t3b).muli(181).rsh::<8>(128).neg();
    c[8] = t2b.sub(t3b).muli(181).rsh::<8>(128);
    c[4] = t6.add(t7).muli(181).rsh::<8>(128);
    c[11] = t6.sub(t7).muli(181).rsh::<8>(128).neg();
    c[6] = t10.add(t11).muli(181).rsh::<8>(128);
    c[9] = t10.sub(t11).muli(181).rsh::<8>(128).neg();
    c[5] = t14b.add(t15b).muli(181).rsh::<8>(128).neg();
    c[10] = t14b.sub(t15b).muli(181).rsh::<8>(128);
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_dequant16_i32x8<const QM: bool>(
    levels: &[i32; 256],
    x: usize,
    y: usize,
    dequant: &IdctDequant,
) -> I32x8 {
    let rc = x * 16 + y;
    let lvl = unsafe { _mm256_loadu_si256(levels.as_ptr().add(rc).cast::<__m256i>()) };
    I32x8(dequant8::<false>(
        lvl,
        dequant_q8::<QM>(dequant, rc),
        splat(dequant.cf_max),
    ))
}

#[target_feature(enable = "avx2")]
fn inv16x16_mixed_dequant_avx2<const ROW_ADST: bool, const COL_ADST: bool>(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    if dequant.qm.is_some() {
        inv16x16_mixed_dequant_avx2_impl::<ROW_ADST, COL_ADST, true>(levels, dequant)
    } else {
        inv16x16_mixed_dequant_avx2_impl::<ROW_ADST, COL_ADST, false>(levels, dequant)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inv16x16_mixed_dequant_avx2_impl<const ROW_ADST: bool, const COL_ADST: bool, const QM: bool>(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);

    // First inverse dimension consumes quantized levels directly. This fuses
    // dequantization with the first transform pass and stores a real transposed
    // scratch: scratch[y_frequency * 16 + x_spatial].
    let mut scratch_u = MaybeUninit::<[i32; 256]>::uninit();
    for y in (0..16usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|x| load_dequant16_i32x8::<QM>(levels, x, y, dequant));
        if ROW_ADST {
            inv_adst16_v(&mut rows, dequant.rmin, dequant.rmax);
        } else {
            inv_dct16_v(&mut rows, dequant.rmin, dequant.rmax);
        }
        for row in rows.iter_mut() {
            *row = row.rsh::<2>(2).clip(cmn, cmx);
        }
        store_transposed_rows_i32x8::<16>(scratch_u.as_mut_ptr().cast(), y, &rows);
    }
    let scratch = unsafe { scratch_u.assume_init() };

    let mut out = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(8) {
        let mut cols: [I32x8; 16] =
            std::array::from_fn(|y| load_i32x8(unsafe { scratch.as_ptr().add(y * 16 + x) }));
        if COL_ADST {
            inv_adst16_v(&mut cols, dequant.cmin, dequant.cmax);
        } else {
            inv_dct16_v(&mut cols, dequant.cmin, dequant.cmax);
        }
        for y in 0..16usize {
            let r = cols[y].rsh::<4>(8);
            store_i32x8(unsafe { (out.as_mut_ptr() as *mut i32).add(y * 16 + x) }, r);
        }
    }
    unsafe { out.assume_init() }
}

const INV_DCT: u8 = 0;
const INV_ADST: u8 = 1;
const INV_IDENTITY: u8 = 2;

#[inline]
#[target_feature(enable = "avx2")]
fn identity_v(v: I32x8, len: usize) -> I32x8 {
    match len {
        4 => v.add(v.muli(1697).rsh::<12>(2048)),
        8 => v.add(v),
        16 => v.add(v).add(v.muli(1697).rsh::<11>(1024)),
        _ => unreachable!("unsupported inverse identity length"),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply4<const KIND: u8>(c: &mut [I32x8; 4], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct4_v(c, min, max),
        INV_ADST => inv_adst4_v(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 4)),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply8<const KIND: u8>(c: &mut [I32x8; 8], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct8_v(c, min, max),
        INV_ADST => inv_adst8_v(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 8)),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn apply16<const KIND: u8>(c: &mut [I32x8; 16], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct16_v(c, min, max),
        INV_ADST => inv_adst16_v(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 16)),
        _ => unreachable!(),
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn inv_dct32_x8_staged(
    src: *const i32,
    src_stride: usize,
    dst: *mut i32,
    dst_stride: usize,
    min: i32,
    max: i32,
) {
    let mn = splat(min);
    let mx = splat(max);
    let clip = |v: I32x8| v.clip(mn, mx);
    let load = |position: usize| load_i32x8(unsafe { src.add(position * src_stride) });

    let mut w: [I32x8; 16] = std::array::from_fn(|i| load(2 * i));
    inv_dct16_v(&mut w, min, max);
    let mut even_store = MaybeUninit::<[i32; 128]>::uninit();
    let even_ptr = even_store.as_mut_ptr() as *mut i32;
    for (i, value) in w.iter().copied().enumerate() {
        store_i32x8(unsafe { even_ptr.add(i * 8) }, value);
    }

    // Odd stage 1.
    let odd = |i: usize| load(2 * i + 1);
    let (a, b) = (odd(0), odd(15));
    w[0] = a.muli(201).sub(b.muli(-5)).rsh::<12>(2048).sub(b);
    w[15] = a.muli(-5).add(b.muli(201)).rsh::<12>(2048).add(a);
    let (a, b) = (odd(8), odd(7));
    w[1] = a.muli(-1061).sub(b.muli(2751)).rsh::<12>(2048).add(a);
    w[14] = a.muli(2751).add(b.muli(-1061)).rsh::<12>(2048).add(b);
    let (a, b) = (odd(4), odd(11));
    w[2] = a.muli(1751).sub(b.muli(-393)).rsh::<12>(2048).sub(b);
    w[13] = a.muli(-393).add(b.muli(1751)).rsh::<12>(2048).add(a);
    let (a, b) = (odd(12), odd(3));
    w[3] = a.muli(-239).sub(b.muli(1380)).rsh::<12>(2048).add(a);
    w[12] = a.muli(1380).add(b.muli(-239)).rsh::<12>(2048).add(b);
    let (a, b) = (odd(2), odd(13));
    w[4] = a.muli(995).sub(b.muli(-123)).rsh::<12>(2048).sub(b);
    w[11] = a.muli(-123).add(b.muli(995)).rsh::<12>(2048).add(a);
    let (a, b) = (odd(10), odd(5));
    w[5] = a.muli(-583).sub(b.muli(2106)).rsh::<12>(2048).add(a);
    w[10] = a.muli(2106).add(b.muli(-583)).rsh::<12>(2048).add(b);
    let (a, b) = (odd(6), odd(9));
    w[6] = a.muli(1220).sub(b.muli(1645)).rsh::<11>(1024);
    w[9] = a.muli(1645).add(b.muli(1220)).rsh::<11>(1024);
    let (a, b) = (odd(14), odd(1));
    w[7] = a.muli(-44).sub(b.muli(601)).rsh::<12>(2048).add(a);
    w[8] = a.muli(601).add(b.muli(-44)).rsh::<12>(2048).add(b);

    // Odd stage 2.
    for &(i, j, reverse) in &[
        (0, 1, false),
        (2, 3, true),
        (4, 5, false),
        (6, 7, true),
        (8, 9, false),
        (10, 11, true),
        (12, 13, false),
        (14, 15, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = clip(b.sub(a));
            w[j] = clip(b.add(a));
        } else {
            w[i] = clip(a.add(b));
            w[j] = clip(a.sub(b));
        }
    }

    // Odd stage 3 rotations.
    let (a, b) = (w[1], w[14]);
    w[1] = b.muli(799).sub(a.muli(-79)).rsh::<12>(2048).sub(a);
    w[14] = b.muli(-79).add(a.muli(799)).rsh::<12>(2048).add(b);
    let (a, b) = (w[2], w[13]);
    w[2] = b.muli(-79).add(a.muli(799)).neg().rsh::<12>(2048).sub(b);
    w[13] = b.muli(799).sub(a.muli(-79)).rsh::<12>(2048).sub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = b.muli(1703).sub(a.muli(1138)).rsh::<11>(1024);
    w[10] = b.muli(1138).add(a.muli(1703)).rsh::<11>(1024);
    let (a, b) = (w[6], w[9]);
    w[6] = b.muli(1138).add(a.muli(1703)).neg().rsh::<11>(1024);
    w[9] = b.muli(1703).sub(a.muli(1138)).rsh::<11>(1024);

    // Odd stage 4.
    for &(i, j, reverse) in &[
        (0, 3, false),
        (1, 2, false),
        (4, 7, true),
        (5, 6, true),
        (8, 11, false),
        (9, 10, false),
        (12, 15, true),
        (13, 14, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = clip(b.sub(a));
            w[j] = clip(b.add(a));
        } else {
            w[i] = clip(a.add(b));
            w[j] = clip(a.sub(b));
        }
    }

    // Odd stage 5 rotations.
    let (a, b) = (w[2], w[13]);
    w[2] = b.muli(1567).sub(a.muli(-312)).rsh::<12>(2048).sub(a);
    w[13] = b.muli(-312).add(a.muli(1567)).rsh::<12>(2048).add(b);
    let (a, b) = (w[3], w[12]);
    w[3] = b.muli(1567).sub(a.muli(-312)).rsh::<12>(2048).sub(a);
    w[12] = b.muli(-312).add(a.muli(1567)).rsh::<12>(2048).add(b);
    let (a, b) = (w[4], w[11]);
    w[4] = b.muli(-312).add(a.muli(1567)).neg().rsh::<12>(2048).sub(b);
    w[11] = b.muli(1567).sub(a.muli(-312)).rsh::<12>(2048).sub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = b.muli(-312).add(a.muli(1567)).neg().rsh::<12>(2048).sub(b);
    w[10] = b.muli(1567).sub(a.muli(-312)).rsh::<12>(2048).sub(a);

    // Odd stage 6.
    for &(i, j, reverse) in &[
        (0, 7, false),
        (1, 6, false),
        (2, 5, false),
        (3, 4, false),
        (8, 15, true),
        (9, 14, true),
        (10, 13, true),
        (11, 12, true),
    ] {
        let (a, b) = (w[i], w[j]);
        if reverse {
            w[i] = clip(b.sub(a));
            w[j] = clip(b.add(a));
        } else {
            w[i] = clip(a.add(b));
            w[j] = clip(a.sub(b));
        }
    }

    // Odd stage 7.
    for &(i, j) in &[(4, 11), (5, 10), (6, 9), (7, 8)] {
        let (a, b) = (w[i], w[j]);
        w[i] = b.sub(a).muli(181).rsh::<8>(128);
        w[j] = b.add(a).muli(181).rsh::<8>(128);
    }

    for i in 0..16 {
        let e = load_i32x8(unsafe { even_ptr.add(i * 8) });
        let o = w[15 - i];
        store_i32x8(unsafe { dst.add(i * dst_stride) }, clip(e.add(o)));
        store_i32x8(unsafe { dst.add((31 - i) * dst_stride) }, clip(e.sub(o)));
    }
}

macro_rules! define_inverse_first_pass_avx2 {
    ($name:ident, $len:literal, $apply:ident) => {
        #[inline(never)]
        #[target_feature(enable = "avx2")]
        fn $name<const ROW: u8, const PRESCALE: bool, const MID_SHIFT: i32>(
            coeff: *const i32,
            scratch: *mut i32,
            y0: usize,
            valid: usize,
            dequant: &IdctDequant,
            w: usize,
            h: usize,
        ) {
            let mut values: [I32x8; $len] = std::array::from_fn(|x| {
                let mut z = load_i32x8_partial(unsafe { coeff.add(x * h + y0) }, valid);
                if PRESCALE {
                    z = z.muli(181).rsh::<8>(128);
                }
                z
            });
            $apply::<ROW>(&mut values, dequant.rmin, dequant.rmax);
            let cmn = splat(dequant.cmin);
            let cmx = splat(dequant.cmax);
            for value in values.iter_mut() {
                *value = match MID_SHIFT {
                    0 => value.clip(cmn, cmx),
                    1 => value.rsh::<1>(1).clip(cmn, cmx),
                    2 => value.rsh::<2>(2).clip(cmn, cmx),
                    _ => unreachable!(),
                };
            }
            if valid == 8 && w >= 8 {
                for x in (0..$len).step_by(8) {
                    let mut tile: [I32x8; 8] = std::array::from_fn(|i| values[x + i]);
                    transpose_store_8x8(unsafe { scratch.add(y0 * w + x) }, w, &mut tile);
                }
            } else {
                for x in (0..$len).step_by(4) {
                    let tile: [I32x8; 4] = std::array::from_fn(|i| values[x + i]);
                    transpose_store_4x8(unsafe { scratch.add(y0 * w + x) }, w, &tile, valid);
                }
            }
        }
    };
}

define_inverse_first_pass_avx2!(inverse_first_pass4_avx2, 4, apply4);
define_inverse_first_pass_avx2!(inverse_first_pass8_avx2, 8, apply8);
define_inverse_first_pass_avx2!(inverse_first_pass16_avx2, 16, apply16);

#[inline(never)]
#[target_feature(enable = "avx2")]
fn inverse_first_pass32_avx2<const PRESCALE: bool, const MID_SHIFT: i32>(
    coeff: *const i32,
    scratch: *mut i32,
    y0: usize,
    valid: usize,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    debug_assert_eq!(w, 32);
    debug_assert_eq!(valid, 8);
    let mut input_store = MaybeUninit::<[i32; 256]>::uninit();
    let input_ptr = input_store.as_mut_ptr() as *mut i32;
    for x in 0..32 {
        let mut z = load_i32x8_partial(unsafe { coeff.add(x * h + y0) }, valid);
        if PRESCALE {
            z = z.muli(181).rsh::<8>(128);
        }
        store_i32x8(unsafe { input_ptr.add(x * 8) }, z);
    }

    let mut pass_store = MaybeUninit::<[i32; 256]>::uninit();
    let pass_ptr = pass_store.as_mut_ptr() as *mut i32;
    inv_dct32_x8_staged(input_ptr, 8, pass_ptr, 8, dequant.rmin, dequant.rmax);
    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);
    for x in (0..32).step_by(8) {
        let mut tile: [I32x8; 8] = std::array::from_fn(|i| {
            let z = load_i32x8(unsafe { pass_ptr.add((x + i) * 8) });
            match MID_SHIFT {
                0 => z.clip(cmn, cmx),
                1 => z.rsh::<1>(1).clip(cmn, cmx),
                2 => z.rsh::<2>(2).clip(cmn, cmx),
                _ => unreachable!(),
            }
        });
        transpose_store_8x8(unsafe { scratch.add(y0 * w + x) }, w, &mut tile);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inverse_first_pass_avx2<const ROW: u8, const PRESCALE: bool, const MID_SHIFT: i32>(
    coeff: *const i32,
    scratch: *mut i32,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    for y0 in (0..h).step_by(8) {
        let valid = (h - y0).min(8);
        match w {
            4 => inverse_first_pass4_avx2::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, valid, dequant, w, h,
            ),
            8 => inverse_first_pass8_avx2::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, valid, dequant, w, h,
            ),
            16 => inverse_first_pass16_avx2::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, valid, dequant, w, h,
            ),
            32 => {
                debug_assert_eq!(ROW, INV_DCT);
                inverse_first_pass32_avx2::<PRESCALE, MID_SHIFT>(
                    coeff, scratch, y0, valid, dequant, w, h,
                );
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_inverse_second_pass_avx2<const LEN: usize>(
    values: &[I32x8; LEN],
    out: *mut i32,
    x0: usize,
    valid: usize,
    w: usize,
) {
    for y in 0..LEN {
        let value = values[y].rsh::<4>(8);
        if valid == 8 {
            store_i32x8(unsafe { out.add(y * w + x0) }, value);
        } else {
            debug_assert_eq!(valid, 4);
            unsafe {
                _mm_storeu_si128(out.add(y * w + x0).cast(), _mm256_castsi256_si128(value.0))
            };
        }
    }
}

macro_rules! define_inverse_second_pass_avx2 {
    ($name:ident, $len:literal, $apply:ident) => {
        #[inline(never)]
        #[target_feature(enable = "avx2")]
        fn $name<const COL: u8>(
            scratch: *const i32,
            out: *mut i32,
            x0: usize,
            valid: usize,
            dequant: &IdctDequant,
            w: usize,
            h: usize,
        ) {
            debug_assert_eq!(h, $len);
            let mut values: [I32x8; $len] = std::array::from_fn(|y| {
                load_i32x8_partial(unsafe { scratch.add(y * w + x0) }, valid)
            });
            $apply::<COL>(&mut values, dequant.cmin, dequant.cmax);
            store_inverse_second_pass_avx2::<$len>(&values, out, x0, valid, w);
        }
    };
}

define_inverse_second_pass_avx2!(inverse_second_pass4_avx2, 4, apply4);
define_inverse_second_pass_avx2!(inverse_second_pass8_avx2, 8, apply8);
define_inverse_second_pass_avx2!(inverse_second_pass16_avx2, 16, apply16);

#[inline(never)]
#[target_feature(enable = "avx2")]
fn inverse_second_pass32_avx2(
    scratch: *const i32,
    out: *mut i32,
    x0: usize,
    valid: usize,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    debug_assert_eq!(h, 32);
    let mut input_store = MaybeUninit::<[i32; 256]>::uninit();
    let input_ptr = input_store.as_mut_ptr() as *mut i32;
    for y in 0..32 {
        let z = load_i32x8_partial(unsafe { scratch.add(y * w + x0) }, valid);
        store_i32x8(unsafe { input_ptr.add(y * 8) }, z);
    }
    let mut pass_store = MaybeUninit::<[i32; 256]>::uninit();
    let pass_ptr = pass_store.as_mut_ptr() as *mut i32;
    inv_dct32_x8_staged(input_ptr, 8, pass_ptr, 8, dequant.cmin, dequant.cmax);
    for y in 0..32 {
        let value = load_i32x8(unsafe { pass_ptr.add(y * 8) }).rsh::<4>(8);
        if valid == 8 {
            store_i32x8(unsafe { out.add(y * w + x0) }, value);
        } else {
            debug_assert_eq!(valid, 4);
            unsafe {
                _mm_storeu_si128(out.add(y * w + x0).cast(), _mm256_castsi256_si128(value.0))
            };
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inverse_second_pass_avx2<const COL: u8>(
    scratch: *const i32,
    out: *mut i32,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    for x0 in (0..w).step_by(8) {
        let valid = (w - x0).min(8);
        match h {
            4 => inverse_second_pass4_avx2::<COL>(scratch, out, x0, valid, dequant, w, h),
            8 => inverse_second_pass8_avx2::<COL>(scratch, out, x0, valid, dequant, w, h),
            16 => inverse_second_pass16_avx2::<COL>(scratch, out, x0, valid, dequant, w, h),
            32 => {
                debug_assert_eq!(COL, INV_DCT);
                inverse_second_pass32_avx2(scratch, out, x0, valid, dequant, w, h);
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn inverse_avx2<
    const N: usize,
    const W: usize,
    const H: usize,
    const ROW: u8,
    const COL: u8,
    const PRESCALE: bool,
    const MID_SHIFT: i32,
    const DQ1: bool,
>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    let coeff = dequant_levels::<N, DQ1>(levels, dequant);
    let mut scratch_store = MaybeUninit::<[i32; N]>::uninit();
    let scratch_ptr = scratch_store.as_mut_ptr() as *mut i32;
    inverse_first_pass_avx2::<ROW, PRESCALE, MID_SHIFT>(coeff.as_ptr(), scratch_ptr, dequant, W, H);
    let mut out = MaybeUninit::<[i32; N]>::uninit();
    inverse_second_pass_avx2::<COL>(scratch_ptr, out.as_mut_ptr().cast(), dequant, W, H);
    unsafe { out.assume_init() }
}

macro_rules! inverse_avx2_entry {
    ($name:ident, $n:literal, $w:literal, $h:literal, $row:expr, $col:expr, $pre:expr, $mid:literal, $dq1:expr) => {
        #[target_feature(enable = "avx2")]
        pub(crate) fn $name(levels: &[i32; $n], dequant: &IdctDequant) -> [i32; $n] {
            inverse_avx2::<$n, $w, $h, $row, $col, $pre, $mid, $dq1>(levels, dequant)
        }
    };
}

macro_rules! inverse_dct_s16_avx2_entry {
    (
        $name:ident, $n:literal, $w:literal, $h:literal,
        $pre:literal, $mid:literal, $dq1:literal
    ) => {
        #[target_feature(enable = "avx2")]
        pub(crate) fn $name(levels: &[i32; $n], dequant: &IdctDequant) -> [i32; $n] {
            if can_use_s16_inverse(dequant) {
                return inverse_dct_s16_avx2::<$n, $w, $h, $dq1, $pre, $mid>(levels, dequant);
            }
            inverse_avx2::<$n, $w, $h, INV_DCT, INV_DCT, $pre, $mid, $dq1>(levels, dequant)
        }
    };
}

inverse_avx2_entry!(
    idct_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_DCT,
    INV_DCT,
    false,
    0,
    false
);
inverse_avx2_entry!(
    idct_dequant_4x8_avx2,
    32,
    4,
    8,
    INV_DCT,
    INV_DCT,
    true,
    0,
    false
);
inverse_avx2_entry!(
    idct_dequant_8x4_avx2,
    32,
    8,
    4,
    INV_DCT,
    INV_DCT,
    true,
    0,
    false
);
inverse_avx2_entry!(
    idct_dequant_4x16_avx2,
    64,
    4,
    16,
    INV_DCT,
    INV_DCT,
    false,
    1,
    false
);
inverse_avx2_entry!(
    idct_dequant_16x4_avx2,
    64,
    16,
    4,
    INV_DCT,
    INV_DCT,
    false,
    1,
    false
);
inverse_dct_s16_avx2_entry!(idct_dequant_8x16_avx2, 128, 8, 16, true, 1, false);
inverse_dct_s16_avx2_entry!(idct_dequant_16x8_avx2, 128, 16, 8, true, 1, false);
inverse_dct_s16_avx2_entry!(idct_dequant_16x32_avx2, 512, 16, 32, true, 1, true);
inverse_dct_s16_avx2_entry!(idct_dequant_32x16_avx2, 512, 32, 16, true, 1, true);

inverse_avx2_entry!(
    iadst_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_ADST,
    INV_ADST,
    false,
    0,
    false
);
inverse_avx2_entry!(
    iadstdct_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_DCT,
    INV_ADST,
    false,
    0,
    false
);
inverse_avx2_entry!(
    idctadst_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_ADST,
    INV_DCT,
    false,
    0,
    false
);
inverse_avx2_entry!(
    iadst_dequant_4x8_avx2,
    32,
    4,
    8,
    INV_ADST,
    INV_ADST,
    true,
    0,
    false
);
inverse_avx2_entry!(
    iadstdct_dequant_4x8_avx2,
    32,
    4,
    8,
    INV_DCT,
    INV_ADST,
    true,
    0,
    false
);
inverse_avx2_entry!(
    idctadst_dequant_4x8_avx2,
    32,
    4,
    8,
    INV_ADST,
    INV_DCT,
    true,
    0,
    false
);
inverse_avx2_entry!(
    iadst_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_ADST,
    INV_ADST,
    false,
    1,
    false
);
inverse_avx2_entry!(
    iadstdct_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_DCT,
    INV_ADST,
    false,
    1,
    false
);
inverse_avx2_entry!(
    idctadst_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_ADST,
    INV_DCT,
    false,
    1,
    false
);
inverse_avx2_entry!(
    iadst_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_ADST,
    INV_ADST,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iadstdct_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_DCT,
    INV_ADST,
    true,
    1,
    false
);
inverse_avx2_entry!(
    idctadst_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_ADST,
    INV_DCT,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iadst_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_ADST,
    INV_ADST,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iadstdct_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_DCT,
    INV_ADST,
    true,
    1,
    false
);
inverse_avx2_entry!(
    idctadst_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_ADST,
    INV_DCT,
    true,
    1,
    false
);

inverse_avx2_entry!(
    ivdct_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_IDENTITY,
    INV_DCT,
    false,
    0,
    false
);
inverse_avx2_entry!(
    ihdct_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_DCT,
    INV_IDENTITY,
    false,
    0,
    false
);
inverse_avx2_entry!(
    ivdct_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_IDENTITY,
    INV_DCT,
    false,
    1,
    false
);
inverse_avx2_entry!(
    ihdct_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_DCT,
    INV_IDENTITY,
    false,
    1,
    false
);
inverse_avx2_entry!(
    ivdct_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_IDENTITY,
    INV_DCT,
    true,
    1,
    false
);
inverse_avx2_entry!(
    ihdct_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_DCT,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_avx2_entry!(
    ivdct_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_IDENTITY,
    INV_DCT,
    true,
    1,
    false
);
inverse_avx2_entry!(
    ihdct_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_DCT,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iidentity_dequant_4x4_avx2,
    16,
    4,
    4,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    0,
    false
);
inverse_avx2_entry!(
    iidentity_dequant_8x8_avx2,
    64,
    8,
    8,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    1,
    false
);
inverse_avx2_entry!(
    iidtx_dequant_8x16_avx2,
    128,
    8,
    16,
    INV_IDENTITY,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iidtx_dequant_16x8_avx2,
    128,
    16,
    8,
    INV_IDENTITY,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_avx2_entry!(
    iidentity_dequant_16x16_avx2,
    256,
    16,
    16,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    2,
    false
);

#[target_feature(enable = "avx2")]
pub(crate) fn iadstdct_dequant_16x16_avx2(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_mixed_dequant_avx2::<false, true>(levels, dequant)
}

#[target_feature(enable = "avx2")]
pub(crate) fn idctadst_dequant_16x16_avx2(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_mixed_dequant_avx2::<true, false>(levels, dequant)
}

#[target_feature(enable = "avx2")]
pub(crate) fn iadst_dequant_16x16_avx2(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    inv16x16_mixed_dequant_avx2::<true, true>(levels, dequant)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_widened_s16(dst: *mut i32, value: I16x16) {
    unsafe {
        _mm256_storeu_si256(
            dst.cast(),
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(value.0)),
        );
        _mm256_storeu_si256(
            dst.add(8).cast(),
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(value.0)),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_s16x16(src: *const i16, lanes: usize) -> I16x16 {
    match lanes {
        8 => {
            let lo = unsafe { _mm_loadu_si128(src.cast()) };
            I16x16(_mm256_inserti128_si256::<0>(_mm256_setzero_si256(), lo))
        }
        16 => I16x16(unsafe { _mm256_loadu_si256(src.cast()) }),
        _ => unreachable!("unsupported i16 lane count"),
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_widened_s16_lanes(dst: *mut i32, value: I16x16, lanes: usize) {
    unsafe {
        _mm256_storeu_si256(
            dst.cast(),
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(value.0)),
        );
        if lanes == 16 {
            _mm256_storeu_si256(
                dst.add(8).cast(),
                _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(value.0)),
            );
        }
    }
}

#[target_feature(enable = "avx2")]
fn inverse_8x8_s16_avx2(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    let mut coeff = [MaybeUninit::<i16>::uninit(); 64];
    dequant_levels_s16_avx2::<false>(levels, &mut coeff, dequant);
    let zero = _mm256_setzero_si256();
    let mut v: [I16x16; 8] = std::array::from_fn(|x| {
        let row = unsafe { _mm_loadu_si128(coeff.as_ptr().add(x * 8).cast()) };
        I16x16(_mm256_inserti128_si256::<0>(zero, row))
    });
    inv_dct8_v_s16!(&mut v);
    for value in &mut v {
        *value = round_shift_s16::<1>(*value);
    }
    transpose_8x8_s16(&mut v);
    inv_dct8_v_s16!(&mut v);
    let mut out = MaybeUninit::<[i32; 64]>::uninit();
    let dst = out.as_mut_ptr() as *mut i32;
    for (y, value) in v.iter().copied().enumerate() {
        let value = round_shift_s16::<4>(value);
        unsafe {
            _mm256_storeu_si256(
                dst.add(y * 8).cast(),
                _mm256_cvtepi16_epi32(_mm256_castsi256_si128(value.0)),
            );
        }
    }
    unsafe { out.assume_init() }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn first_pass8_s16_avx2(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    debug_assert_eq!(h, 16);
    let mut v: [I16x16; 8] = std::array::from_fn(|x| {
        let value = I16x16(unsafe { _mm256_loadu_si256(coeff.add(x * h + y0).cast()) });
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct8_v_s16!(&mut v);
    let zero = _mm256_setzero_si256();
    for half in 0..2 {
        let mut tile: [I16x16; 8] = std::array::from_fn(|i| {
            let value = mid_shift_s16(v[i], mid_shift).0;
            let row = if half == 0 {
                _mm256_castsi256_si128(value)
            } else {
                _mm256_extracti128_si256::<1>(value)
            };
            I16x16(_mm256_inserti128_si256::<0>(zero, row))
        });
        transpose_8x8_s16(&mut tile);
        for (i, value) in tile.iter().copied().enumerate() {
            unsafe {
                _mm_storeu_si128(
                    scratch.add((y0 + half * 8 + i) * 8).cast(),
                    _mm256_castsi256_si128(value.0),
                );
            }
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn first_pass16_s16_avx2(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    let lanes = (h - y0).min(16);
    let mut v: [I16x16; 16] = std::array::from_fn(|x| {
        let value = load_s16x16(unsafe { coeff.add(x * h + y0) }, lanes);
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct16_v_s16!(&mut v);
    for value in &mut v {
        *value = mid_shift_s16(*value, mid_shift);
    }
    transpose_16x16_s16(&mut v);
    for (i, value) in v.iter().copied().take(lanes).enumerate() {
        unsafe { _mm256_storeu_si256(scratch.add((y0 + i) * 16).cast(), value.0) };
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn first_pass32_s16_avx2(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    let mut v: [I16x16; 32] = std::array::from_fn(|x| {
        let value = I16x16(unsafe { _mm256_loadu_si256(coeff.add(x * h + y0).cast()) });
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct32_v_s16!(&mut v);
    for x0 in (0..32).step_by(16) {
        let mut tile: [I16x16; 16] = std::array::from_fn(|i| mid_shift_s16(v[x0 + i], mid_shift));
        transpose_16x16_s16(&mut tile);
        for (i, value) in tile.iter().copied().enumerate() {
            unsafe {
                _mm256_storeu_si256(scratch.add((y0 + i) * 32 + x0).cast(), value.0);
            }
        }
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn second_pass8_s16_avx2(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let mut v: [I16x16; 8] = std::array::from_fn(|y| {
        I16x16(unsafe { _mm256_loadu_si256(scratch.add(y * w + x0).cast()) })
    });
    inv_dct8_v_s16!(&mut v);
    for (y, value) in v.iter().copied().enumerate() {
        store_widened_s16(unsafe { out.add(y * w + x0) }, round_shift_s16::<4>(value));
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn second_pass16_s16_avx2(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let lanes = (w - x0).min(16);
    let mut v: [I16x16; 16] =
        std::array::from_fn(|y| load_s16x16(unsafe { scratch.add(y * w + x0) }, lanes));
    inv_dct16_v_s16!(&mut v);
    for (y, value) in v.iter().copied().enumerate() {
        store_widened_s16_lanes(
            unsafe { out.add(y * w + x0) },
            round_shift_s16::<4>(value),
            lanes,
        );
    }
}

#[inline(never)]
#[target_feature(enable = "avx2")]
fn second_pass32_s16_avx2(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let mut v: [I16x16; 32] = std::array::from_fn(|y| {
        I16x16(unsafe { _mm256_loadu_si256(scratch.add(y * w + x0).cast()) })
    });
    inv_dct32_v_s16!(&mut v);
    for (y, value) in v.iter().copied().enumerate() {
        store_widened_s16(unsafe { out.add(y * w + x0) }, round_shift_s16::<4>(value));
    }
}

#[target_feature(enable = "avx2")]
fn inverse_dct_s16_avx2<
    const N: usize,
    const W: usize,
    const H: usize,
    const DQ1: bool,
    const PRESCALE: bool,
    const MID_SHIFT: i32,
>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    debug_assert_eq!(N, W * H);
    let mut coeff = [MaybeUninit::<i16>::uninit(); N];
    dequant_levels_s16_avx2::<DQ1>(levels, &mut coeff, dequant);

    let mut scratch = MaybeUninit::<[i16; N]>::uninit();
    let scratch_ptr = scratch.as_mut_ptr() as *mut i16;
    for y0 in (0..H).step_by(16) {
        match W {
            8 => first_pass8_s16_avx2(
                coeff.as_ptr().cast(),
                scratch_ptr,
                y0,
                H,
                PRESCALE,
                MID_SHIFT,
            ),
            16 => first_pass16_s16_avx2(
                coeff.as_ptr().cast(),
                scratch_ptr,
                y0,
                H,
                PRESCALE,
                MID_SHIFT,
            ),
            32 => first_pass32_s16_avx2(
                coeff.as_ptr().cast(),
                scratch_ptr,
                y0,
                H,
                PRESCALE,
                MID_SHIFT,
            ),
            _ => unreachable!("unsupported s16 inverse row length"),
        }
    }

    let mut out = MaybeUninit::<[i32; N]>::uninit();
    let out_ptr = out.as_mut_ptr() as *mut i32;
    for x0 in (0..W).step_by(16) {
        match H {
            8 => second_pass8_s16_avx2(scratch_ptr, out_ptr, x0, W),
            16 => second_pass16_s16_avx2(scratch_ptr, out_ptr, x0, W),
            32 => second_pass32_s16_avx2(scratch_ptr, out_ptr, x0, W),
            _ => unreachable!("unsupported s16 inverse column length"),
        }
    }
    unsafe { out.assume_init() }
}

#[inline]
fn can_use_s16_inverse(dequant: &IdctDequant) -> bool {
    dequant.cf_max == i16::MAX as i32
        && dequant.rmin == i16::MIN as i32
        && dequant.rmax == i16::MAX as i32
        && dequant.cmin == i16::MIN as i32
        && dequant.cmax == i16::MAX as i32
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_8x8_avx2(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    if can_use_s16_inverse(dequant) {
        return inverse_8x8_s16_avx2(levels, dequant);
    }
    let coeff = dequant_levels::<64, false>(levels, dequant);
    let mut v: [I32x8; 8] =
        std::array::from_fn(|x| load_i32x8(unsafe { coeff.as_ptr().add(x * 8) }));

    inv_dct8_v(&mut v, dequant.rmin, dequant.rmax);

    let cmn = splat(dequant.cmin);
    let cmx = splat(dequant.cmax);
    for vv in v.iter_mut() {
        *vv = vv.rsh::<1>(1).clip(cmn, cmx);
    }

    transpose_8x8(&mut v);
    inv_dct8_v(&mut v, dequant.cmin, dequant.cmax);

    let mut out = MaybeUninit::<[i32; 64]>::uninit();
    for (y, vv) in v.iter().copied().enumerate() {
        let r = vv.rsh::<4>(8);
        store_i32x8(unsafe { (out.as_mut_ptr() as *mut i32).add(y * 8) }, r);
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_16x16_avx2(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    if can_use_s16_inverse(dequant) {
        return inverse_dct_s16_avx2::<256, 16, 16, false, false, 2>(levels, dequant);
    }
    inv16x16_mixed_dequant_avx2::<false, false>(levels, dequant)
}

#[target_feature(enable = "avx2")]
pub(crate) fn idct_dequant_32x32_avx2(levels: &[i32; 1024], dequant: &IdctDequant) -> [i32; 1024] {
    if can_use_s16_inverse(dequant) {
        return inverse_dct_s16_avx2::<1024, 32, 32, true, false, 2>(levels, dequant);
    }
    inverse_avx2::<1024, 32, 32, INV_DCT, INV_DCT, false, 2, true>(levels, dequant)
}

#[cfg(test)]
mod dequant_parity_tests {
    use super::*;

    fn scalar(level: i32, q: i32, cf_max: i32, dq1: bool) -> i32 {
        let product = u64::from(level.unsigned_abs()) * q as u64;
        let magnitude =
            (((product & 0xff_ffff) >> u32::from(dq1)) as i32).min(cf_max + i32::from(level < 0));
        if level < 0 { -magnitude } else { magnitude }
    }

    #[target_feature(enable = "avx2")]
    fn vector<const DQ1: bool>(levels: [i32; 8], quants: [i32; 8], cf_max: i32) -> [i32; 8] {
        let levels = unsafe { _mm256_loadu_si256(levels.as_ptr().cast()) };
        let quants = unsafe { _mm256_loadu_si256(quants.as_ptr().cast()) };
        let result = dequant8::<DQ1>(levels, quants, splat(cf_max));
        let mut out = [0; 8];
        unsafe { _mm256_storeu_si256(out.as_mut_ptr().cast(), result) };
        out
    }

    #[test]
    fn packed_dequant_matches_scalar_for_both_shifts_and_all_bit_depth_caps() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        let levels = [i32::MIN, -0x0123_4567, -65536, -1, 0, 1, 65536, i32::MAX];
        let quants = [1, 13, 255, 256, 4095, 32767, 65535, 233_000];
        for cf_max in [32767, 131071, 2_097_151] {
            let got0 = unsafe { vector::<false>(levels, quants, cf_max) };
            let got1 = unsafe { vector::<true>(levels, quants, cf_max) };
            let want0 = std::array::from_fn(|i| scalar(levels[i], quants[i], cf_max, false));
            let want1 = std::array::from_fn(|i| scalar(levels[i], quants[i], cf_max, true));
            assert_eq!(got0, want0, "dq_shift=0 cf_max={cf_max}");
            assert_eq!(got1, want1, "dq_shift=1 cf_max={cf_max}");
        }
    }

    #[test]
    fn s16_dispatch_requires_the_complete_8bit_clip_regime() {
        let eight_bit_qm = IdctDequant {
            dc_q: 77,
            ac_q: 91,
            rmin: i16::MIN as i32,
            rmax: i16::MAX as i32,
            cmin: i16::MIN as i32,
            cmax: i16::MAX as i32,
            cf_max: i16::MAX as i32,
            qm: Some(&[32]),
        };
        assert!(can_use_s16_inverse(&eight_bit_qm));

        for highbd in [
            IdctDequant {
                rmin: i16::MIN as i32 - 1,
                ..eight_bit_qm
            },
            IdctDequant {
                rmax: (1 << 17) - 1,
                ..eight_bit_qm
            },
            IdctDequant {
                cmin: i16::MIN as i32 - 1,
                ..eight_bit_qm
            },
            IdctDequant {
                cmax: (1 << 15) + 1,
                ..eight_bit_qm
            },
            IdctDequant {
                cf_max: (1 << 17) - 1,
                ..eight_bit_qm
            },
        ] {
            assert!(!can_use_s16_inverse(&highbd));
        }
    }
}
