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

use crate::dct::{
    SQRT2, WC4_0, WC4_1, WC8_0, WC8_1, WC8_2, WC8_3, WC16_0, WC16_1, WC16_2, WC16_3, WC16_4,
    WC16_5, WC16_6, WC16_7, WC32,
};
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
fn load_i32x8(ptr: *const i32) -> I32x8 {
    unsafe { I32x8(_mm256_loadu_si256(ptr.cast::<__m256i>())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8(ptr: *mut i32, v: I32x8) {
    unsafe { _mm256_storeu_si256(ptr.cast::<__m256i>(), v.0) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn sar_epi64_16(v: __m256i) -> __m256i {
    let sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
    _mm256_or_si256(_mm256_srli_epi64::<16>(v), _mm256_slli_epi64::<48>(sign))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_q16_epi32(v: __m256i, c: __m256i) -> __m256i {
    let even = sar_epi64_16(_mm256_mul_epi32(v, c));
    let odd_v = _mm256_srli_epi64::<32>(v);
    let odd_c = _mm256_srli_epi64::<32>(c);
    let odd = _mm256_slli_epi64::<32>(sar_epi64_16(_mm256_mul_epi32(odd_v, odd_c)));
    _mm256_blend_epi32::<0b1010_1010>(even, odd)
}

#[inline]
fn quant_flat<const N: usize>(coeffs: &[i32; N], dc_q: i32, ac_q: i32, out: &mut [i32; N]) {
    let mq = |a: i32, b: i32| {
        let prod = (a as i64) * (b as i64);
        let mag = prod.unsigned_abs();
        if mag < 65536 {
            return 0;
        }
        let lvl = ((mag + 32768) >> 16) as i32;
        if prod >= 0 { lvl } else { -lvl }
    };
    out[0] = mq(coeffs[0], dc_q);
    for i in 1..N {
        out[i] = mq(coeffs[i], ac_q);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn quant_prod_i64(prod: __m256i) -> __m256i {
    let zero = _mm256_setzero_si256();
    let sign = _mm256_cmpgt_epi64(zero, prod);
    let mag = _mm256_sub_epi64(_mm256_xor_si256(prod, sign), sign);
    let active = _mm256_cmpgt_epi64(mag, _mm256_set1_epi64x(65535));
    let lvl = _mm256_srli_epi64::<16>(_mm256_add_epi64(mag, _mm256_set1_epi64x(32768)));
    let neg = _mm256_sub_epi64(zero, lvl);
    let signed = _mm256_blendv_epi8(lvl, neg, sign);
    _mm256_and_si256(signed, active)
}

#[inline]
#[target_feature(enable = "avx2")]
fn quant_q16_epi32(v: __m256i, q: __m256i) -> __m256i {
    let even = quant_prod_i64(_mm256_mul_epi32(v, q));
    let odd_v = _mm256_srli_epi64::<32>(v);
    let odd_q = _mm256_srli_epi64::<32>(q);
    let odd = _mm256_slli_epi64::<32>(quant_prod_i64(_mm256_mul_epi32(odd_v, odd_q)));
    _mm256_blend_epi32::<0b1010_1010>(even, odd)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_quant_target_i32x8(
    cf: *mut i32,
    tf: *mut f32,
    coeff: I32x8,
    base: usize,
    dc_q: i32,
    ac_q: i32,
) {
    let ac = _mm256_set1_epi32(ac_q);
    let q = if base == 0 {
        _mm256_blend_epi32::<0b0000_0001>(ac, _mm256_set1_epi32(dc_q))
    } else {
        ac
    };
    let levels = quant_q16_epi32(coeff.0, q);
    let target = _mm256_mul_ps(
        _mm256_mul_ps(_mm256_cvtepi32_ps(coeff.0), _mm256_cvtepi32_ps(q)),
        _mm256_set1_ps(1.0 / 65536.0),
    );
    unsafe {
        _mm256_storeu_si256(cf.add(base).cast::<__m256i>(), levels);
        _mm256_storeu_ps(tf.add(base), target);
    }
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
    fn muls_q16(self, coeff: i32) -> I32x8 {
        I32x8(mul_q16_epi32(self.0, splat(coeff)))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn fma_sqrt2(self, rhs: I32x8) -> I32x8 {
        self.muls_q16(SQRT2).add(rhs)
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn shr<const N: i32>(self) -> I32x8 {
        I32x8(_mm256_srai_epi32::<N>(self.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn shl<const N: i32>(self) -> I32x8 {
        I32x8(_mm256_slli_epi32::<N>(self.0))
    }

    #[inline]
    #[target_feature(enable = "avx2")]
    fn shr_round<const N: i32>(self) -> I32x8 {
        I32x8(_mm256_srai_epi32::<N>(_mm256_add_epi32(
            self.0,
            splat(1 << (N - 1)),
        )))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_i32(c: &mut [I32x8; 8]) {
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
fn transpose_store_8x8_i32(dst: *mut i32, stride: usize, tile: &mut [I32x8; 8]) {
    transpose_8x8_i32(tile);
    for i in 0..8usize {
        store_i32x8(unsafe { dst.add(i * stride) }, tile[i]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_transposed_cols_i32x8<const N: usize>(dst: *mut i32, x: usize, c: &[I32x8; N]) {
    debug_assert!(N.is_multiple_of(8));
    let stride = N;
    let mut v = 0usize;
    while v < N {
        let mut tile = [
            c[v],
            c[v + 1],
            c[v + 2],
            c[v + 3],
            c[v + 4],
            c[v + 5],
            c[v + 6],
            c[v + 7],
        ];
        transpose_store_8x8_i32(unsafe { dst.add(x * N + v) }, stride, &mut tile);
        v += 8;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct1d_4_v_i32(c: &mut [I32x8; 4]) {
    let t0 = c[0].add(c[3]);
    let t1 = c[1].add(c[2]);
    let sum = t0.add(t1);
    let diff = t0.sub(t1);

    let t2 = c[0].sub(c[3]).muls_q16(WC4_0);
    let t3 = c[1].sub(c[2]).muls_q16(WC4_1);
    let t2p = t2.add(t3);
    let t3p = t2.sub(t3);
    let t2pp = t2p.fma_sqrt2(t3p);

    c[0] = sum;
    c[1] = t2pp;
    c[2] = diff;
    c[3] = t3p;
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct1d_8_v_i32(c: &mut [I32x8; 8]) {
    let mut evens = [
        c[0].add(c[7]),
        c[1].add(c[6]),
        c[2].add(c[5]),
        c[3].add(c[4]),
    ];
    dct1d_4_v_i32(&mut evens);

    let mut odds = [
        c[0].sub(c[7]).muls_q16(WC8_0),
        c[1].sub(c[6]).muls_q16(WC8_1),
        c[2].sub(c[5]).muls_q16(WC8_2),
        c[3].sub(c[4]).muls_q16(WC8_3),
    ];
    dct1d_4_v_i32(&mut odds);

    odds[0] = odds[0].fma_sqrt2(odds[1]);
    odds[1] = odds[1].add(odds[2]);
    odds[2] = odds[2].add(odds[3]);

    c[0] = evens[0];
    c[1] = odds[0];
    c[2] = evens[1];
    c[3] = odds[1];
    c[4] = evens[2];
    c[5] = odds[2];
    c[6] = evens[3];
    c[7] = odds[3];
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct1d_16_v_i32(c: &mut [I32x8; 16]) {
    let mut evens = [
        c[0].add(c[15]),
        c[1].add(c[14]),
        c[2].add(c[13]),
        c[3].add(c[12]),
        c[4].add(c[11]),
        c[5].add(c[10]),
        c[6].add(c[9]),
        c[7].add(c[8]),
    ];
    let mut odds = [
        c[0].sub(c[15]).muls_q16(WC16_0),
        c[1].sub(c[14]).muls_q16(WC16_1),
        c[2].sub(c[13]).muls_q16(WC16_2),
        c[3].sub(c[12]).muls_q16(WC16_3),
        c[4].sub(c[11]).muls_q16(WC16_4),
        c[5].sub(c[10]).muls_q16(WC16_5),
        c[6].sub(c[9]).muls_q16(WC16_6),
        c[7].sub(c[8]).muls_q16(WC16_7),
    ];

    dct1d_8_v_i32(&mut evens);
    dct1d_8_v_i32(&mut odds);

    odds[0] = odds[0].fma_sqrt2(odds[1]);
    odds[1] = odds[1].add(odds[2]);
    odds[2] = odds[2].add(odds[3]);
    odds[3] = odds[3].add(odds[4]);
    odds[4] = odds[4].add(odds[5]);
    odds[5] = odds[5].add(odds[6]);
    odds[6] = odds[6].add(odds[7]);

    c[0] = evens[0];
    c[1] = odds[0];
    c[2] = evens[1];
    c[3] = odds[1];
    c[4] = evens[2];
    c[5] = odds[2];
    c[6] = evens[3];
    c[7] = odds[3];
    c[8] = evens[4];
    c[9] = odds[4];
    c[10] = evens[5];
    c[11] = odds[5];
    c[12] = evens[6];
    c[13] = odds[6];
    c[14] = evens[7];
    c[15] = odds[7];
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct1d_32_v_i32(c: &mut [I32x8; 32]) {
    let mut evens = std::array::from_fn::<I32x8, 16, _>(|i| c[i].add(c[31 - i]));
    let mut odds = std::array::from_fn::<I32x8, 16, _>(|i| c[i].sub(c[31 - i]));

    dct1d_16_v_i32(&mut evens);
    for i in 0..16 {
        odds[i] = odds[i].muls_q16(WC32[i]);
    }
    dct1d_16_v_i32(&mut odds);

    odds[0] = odds[0].fma_sqrt2(odds[1]);
    odds[1] = odds[1].add(odds[2]);
    odds[2] = odds[2].add(odds[3]);
    odds[3] = odds[3].add(odds[4]);
    odds[4] = odds[4].add(odds[5]);
    odds[5] = odds[5].add(odds[6]);
    odds[6] = odds[6].add(odds[7]);
    odds[7] = odds[7].add(odds[8]);
    odds[8] = odds[8].add(odds[9]);
    odds[9] = odds[9].add(odds[10]);
    odds[10] = odds[10].add(odds[11]);
    odds[11] = odds[11].add(odds[12]);
    odds[12] = odds[12].add(odds[13]);
    odds[13] = odds[13].add(odds[14]);
    odds[14] = odds[14].add(odds[15]);

    for i in 0..16 {
        c[2 * i] = evens[i];
        c[2 * i + 1] = odds[i];
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load8_i32(ptr: &[i32], stride: usize) -> [I32x8; 8] {
    std::array::from_fn(|y| load_i32x8(unsafe { ptr.as_ptr().add(y * stride) }))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load16_i32(ptr: &[i32], stride: usize) -> [I32x8; 16] {
    std::array::from_fn(|y| load_i32x8(unsafe { ptr.as_ptr().add(y * stride) }))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load32_i32(ptr: &[i32], stride: usize) -> [I32x8; 32] {
    std::array::from_fn(|y| load_i32x8(unsafe { ptr.as_ptr().add(y * stride) }))
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct8x8_avx2_quant_t(
    input: &[i32; 64],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 64], [f32; 64]) {
    let mut cols = load8_i32(input, 8);
    dct1d_8_v_i32(&mut cols);
    transpose_8x8_i32(&mut cols);
    dct1d_8_v_i32(&mut cols);

    let mut cf = MaybeUninit::<[i32; 64]>::uninit();
    let mut tf = MaybeUninit::<[f32; 64]>::uninit();
    for (k, col) in cols.iter().copied().enumerate() {
        store_quant_target_i32x8(
            cf.as_mut_ptr().cast(),
            tf.as_mut_ptr().cast(),
            col,
            k * 8,
            dc_q,
            ac_q,
        );
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct16x16_avx2_coeffs(input: &[i32; 256]) -> [i32; 256] {
    // Stage 1: vertical DCT-16 in 8-column groups. Store a real transposed
    // scratch: tmp[x * 16 + vertical_frequency]. The second pass can then
    // use contiguous loads instead of scalar strided reconstruction.
    let mut tmp_u = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(8) {
        let mut cols = load16_i32(&input[x..], 16);
        dct1d_16_v_i32(&mut cols);
        store_transposed_cols_i32x8::<16>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    // Stage 2: horizontal DCT-16 in 8 vertical-frequency lanes.
    let mut out = MaybeUninit::<[i32; 256]>::uninit();
    for y in (0..16usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 16 + y) }));
        dct1d_16_v_i32(&mut rows);
        for u in 0..16usize {
            unsafe {
                store_i32x8(
                    (out.as_mut_ptr() as *mut i32).add(u * 16 + y),
                    rows[u].shr::<1>(),
                );
            }
        }
    }
    unsafe { out.assume_init() }
}

#[inline]
#[target_feature(enable = "avx2")]
fn sar_epi64_12(v: __m256i) -> __m256i {
    let sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v);
    _mm256_or_si256(_mm256_srli_epi64::<12>(v), _mm256_slli_epi64::<52>(sign))
}

#[inline]
#[target_feature(enable = "avx2")]
fn adst16_round_shift_pair(even: __m256i, odd: __m256i) -> __m256i {
    let even32 = sar_epi64_12(even);
    let odd32 = _mm256_slli_epi64::<32>(sar_epi64_12(odd));
    _mm256_blend_epi32::<0b1010_1010>(even32, odd32)
}

#[inline]
#[target_feature(enable = "avx2")]
fn adst1d_16_v_i32(c: &mut [I32x8; 16]) {
    let mut out = [I32x8(_mm256_setzero_si256()); 16];
    for o in 0..16usize {
        let mut even = _mm256_set1_epi64x(2048);
        let mut odd = _mm256_set1_epi64x(2048);
        for j in 0..16usize {
            let k = _mm256_set1_epi32(crate::dct::ADST16_FWD_Q12[o][j] as i32);
            let v = c[j].0;
            even = _mm256_add_epi64(even, _mm256_mul_epi32(v, k));
            odd = _mm256_add_epi64(odd, _mm256_mul_epi32(_mm256_srli_epi64::<32>(v), k));
        }
        out[o] = I32x8(adst16_round_shift_pair(even, odd));
    }
    *c = out;
}

#[inline]
#[target_feature(enable = "avx2")]
fn tx16x16_adst_dct_avx2_quant_t<const COL_ADST: bool, const ROW_ADST: bool>(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp_u = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(8) {
        let mut cols = load16_i32(&input[x..], 16);
        if COL_ADST {
            adst1d_16_v_i32(&mut cols);
        } else {
            dct1d_16_v_i32(&mut cols);
        }
        store_transposed_cols_i32x8::<16>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 256]>::uninit();
    let mut tf = MaybeUninit::<[f32; 256]>::uninit();
    for y in (0..16usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 16 + y) }));
        if ROW_ADST {
            adst1d_16_v_i32(&mut rows);
        } else {
            dct1d_16_v_i32(&mut rows);
        }
        for u in 0..16usize {
            store_quant_target_i32x8(
                cf.as_mut_ptr().cast(),
                tf.as_mut_ptr().cast(),
                rows[u].shr::<1>(),
                u * 16 + y,
                dc_q,
                ac_q,
            );
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn adst16x16_avx2_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_avx2_quant_t::<true, true>(input, dc_q, ac_q)
}

#[target_feature(enable = "avx2")]
pub(crate) fn adstdct16x16_avx2_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_avx2_quant_t::<true, false>(input, dc_q, ac_q)
}

#[target_feature(enable = "avx2")]
pub(crate) fn dctadst16x16_avx2_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_avx2_quant_t::<false, true>(input, dc_q, ac_q)
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct16x16_avx2_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp_u = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(8) {
        let mut cols = load16_i32(&input[x..], 16);
        dct1d_16_v_i32(&mut cols);
        store_transposed_cols_i32x8::<16>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 256]>::uninit();
    let mut tf = MaybeUninit::<[f32; 256]>::uninit();
    for y in (0..16usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 16 + y) }));
        dct1d_16_v_i32(&mut rows);
        for u in 0..16usize {
            store_quant_target_i32x8(
                cf.as_mut_ptr().cast(),
                tf.as_mut_ptr().cast(),
                rows[u].shr::<1>(),
                u * 16 + y,
                dc_q,
                ac_q,
            );
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct16x16_avx2_i32(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    let coeffs = dct16x16_avx2_coeffs(input);
    quant_flat(&coeffs, dc_q, ac_q, input);
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct32x32_avx2_coeffs(input: &[i32; 1024]) -> [i32; 1024] {
    // Stage 1: vertical DCT-32 in 8-column groups. Store a true transposed
    // scratch: tmp[x * 32 + vertical_frequency].
    let mut tmp_u = MaybeUninit::<[i32; 1024]>::uninit();
    for x in (0..32usize).step_by(8) {
        let mut cols = load32_i32(&input[x..], 32);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v_i32(&mut cols);
        store_transposed_cols_i32x8::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    // Stage 2: horizontal DCT-32 with contiguous loads from the transposed scratch.
    let mut out = MaybeUninit::<[i32; 1024]>::uninit();
    for y in (0..32usize).step_by(8) {
        let mut rows: [I32x8; 32] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 32 + y) }));
        dct1d_32_v_i32(&mut rows);
        for u in 0..32usize {
            unsafe {
                store_i32x8(
                    (out.as_mut_ptr() as *mut i32).add(u * 32 + y),
                    rows[u].shr_round::<8>(),
                );
            }
        }
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct32x32_avx2_quant_t(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    let mut tmp_u = MaybeUninit::<[i32; 1024]>::uninit();
    for x in (0..32usize).step_by(8) {
        let mut cols = load32_i32(&input[x..], 32);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v_i32(&mut cols);
        store_transposed_cols_i32x8::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 1024]>::uninit();
    let mut tf = MaybeUninit::<[f32; 1024]>::uninit();
    for y in (0..32usize).step_by(8) {
        let mut rows: [I32x8; 32] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 32 + y) }));
        dct1d_32_v_i32(&mut rows);
        for u in 0..32usize {
            store_quant_target_i32x8(
                cf.as_mut_ptr().cast(),
                tf.as_mut_ptr().cast(),
                rows[u].shr_round::<8>(),
                u * 32 + y,
                dc_q,
                ac_q,
            );
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct32x32_avx2_i32(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    let coeffs = dct32x32_avx2_coeffs(input);
    quant_flat(&coeffs, dc_q, ac_q, input);
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct8x16_avx2_coeffs(input: &[i32; 128]) -> [i32; 128] {
    let mut rows = load16_i32(input, 8);
    dct1d_16_v_i32(&mut rows);

    let mut a: [I32x8; 8] = rows[..8].try_into().unwrap();
    let mut b: [I32x8; 8] = rows[8..16].try_into().unwrap();
    transpose_8x8_i32(&mut a);
    transpose_8x8_i32(&mut b);
    dct1d_8_v_i32(&mut a);
    dct1d_8_v_i32(&mut b);

    let mut out = MaybeUninit::<[i32; 128]>::uninit();
    for fx in 0..8usize {
        unsafe {
            let dst = (out.as_mut_ptr() as *mut i32).add(fx * 16);
            store_i32x8(dst, a[fx].muls_q16(46341));
            store_i32x8(dst.add(8), b[fx].muls_q16(46341));
        }
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct8x16_avx2_quant_t(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
    let mut rows = load16_i32(input, 8);
    dct1d_16_v_i32(&mut rows);

    let mut a: [I32x8; 8] = rows[..8].try_into().unwrap();
    let mut b: [I32x8; 8] = rows[8..16].try_into().unwrap();
    transpose_8x8_i32(&mut a);
    transpose_8x8_i32(&mut b);
    dct1d_8_v_i32(&mut a);
    dct1d_8_v_i32(&mut b);

    let mut cf = MaybeUninit::<[i32; 128]>::uninit();
    let mut tf = MaybeUninit::<[f32; 128]>::uninit();
    for fx in 0..8usize {
        store_quant_target_i32x8(
            cf.as_mut_ptr().cast(),
            tf.as_mut_ptr().cast(),
            a[fx].muls_q16(46341),
            fx * 16,
            dc_q,
            ac_q,
        );
        store_quant_target_i32x8(
            cf.as_mut_ptr().cast(),
            tf.as_mut_ptr().cast(),
            b[fx].muls_q16(46341),
            fx * 16 + 8,
            dc_q,
            ac_q,
        );
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct8x16_avx2_i32(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x16_avx2_coeffs(input);
    quant_flat(&coeffs, dc_q, ac_q, input);
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct16x32_avx2_quant_t(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    // TX_16X32 scalar path does the 32-point vertical pass first, with a
    // temporary << 6 scale, then the 16-point horizontal pass and rounded >> 6.
    // Keep the same order so the Q16 truncation points stay byte-identical.
    let mut tmp_u = MaybeUninit::<[i32; 512]>::uninit();
    for x in (0..16usize).step_by(8) {
        let mut cols = load32_i32(&input[x..], 16);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v_i32(&mut cols);
        // tmp[x * 32 + fy], i.e. contiguous vertical-frequency lanes for
        // the second pass.
        store_transposed_cols_i32x8::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 512]>::uninit();
    let mut tf = MaybeUninit::<[f32; 512]>::uninit();
    for fy in (0..32usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|x| load_i32x8(unsafe { tmp.as_ptr().add(x * 32 + fy) }));
        dct1d_16_v_i32(&mut rows);
        for fx in 0..16usize {
            store_quant_target_i32x8(
                cf.as_mut_ptr().cast(),
                tf.as_mut_ptr().cast(),
                rows[fx].shr_round::<6>(),
                fx * 32 + fy,
                dc_q,
                ac_q,
            );
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn dct32x16_avx2_quant_t(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    // TX_32X16 scalar path does the 32-point horizontal pass first. Load 8 rows
    // at a time, transpose four 8x8 tiles so each vector holds one x-frequency
    // candidate across eight rows, then store a normal row-major scratch for
    // the vertical 16-point pass.
    let mut tmp_u = MaybeUninit::<[i32; 512]>::uninit();
    let tmp_ptr = tmp_u.as_mut_ptr().cast::<i32>();
    for y in (0..16usize).step_by(8) {
        let zero = I32x8(_mm256_setzero_si256());
        let mut cols = [zero; 32];
        for x in (0..32usize).step_by(8) {
            let mut tile = [
                load_i32x8(unsafe { input.as_ptr().add(y * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 1) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 2) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 3) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 4) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 5) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 6) * 32 + x) }),
                load_i32x8(unsafe { input.as_ptr().add((y + 7) * 32 + x) }),
            ];
            transpose_8x8_i32(&mut tile);
            for i in 0..8usize {
                cols[x + i] = tile[i].shl::<6>();
            }
        }
        dct1d_32_v_i32(&mut cols);

        // Store row-major scratch: tmp[row * 32 + fx]. Each 8x8 tile is
        // transposed back from frequency vectors with row lanes into row vectors
        // with frequency lanes.
        for fx in (0..32usize).step_by(8) {
            let mut tile = [
                cols[fx],
                cols[fx + 1],
                cols[fx + 2],
                cols[fx + 3],
                cols[fx + 4],
                cols[fx + 5],
                cols[fx + 6],
                cols[fx + 7],
            ];
            transpose_store_8x8_i32(unsafe { tmp_ptr.add(y * 32 + fx) }, 32, &mut tile);
        }
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 512]>::uninit();
    let mut tf = MaybeUninit::<[f32; 512]>::uninit();
    for fx in (0..32usize).step_by(8) {
        let mut rows: [I32x8; 16] =
            std::array::from_fn(|y| load_i32x8(unsafe { tmp.as_ptr().add(y * 32 + fx) }));
        dct1d_16_v_i32(&mut rows);

        for fy in (0..16usize).step_by(8) {
            let mut tile = [
                rows[fy].shr_round::<6>(),
                rows[fy + 1].shr_round::<6>(),
                rows[fy + 2].shr_round::<6>(),
                rows[fy + 3].shr_round::<6>(),
                rows[fy + 4].shr_round::<6>(),
                rows[fy + 5].shr_round::<6>(),
                rows[fy + 6].shr_round::<6>(),
                rows[fy + 7].shr_round::<6>(),
            ];
            transpose_8x8_i32(&mut tile);
            for i in 0..8usize {
                store_quant_target_i32x8(
                    cf.as_mut_ptr().cast(),
                    tf.as_mut_ptr().cast(),
                    tile[i],
                    (fx + i) * 16 + fy,
                    dc_q,
                    ac_q,
                );
            }
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[cfg(test)]
mod avx2_vs_scalar {
    use super::*;
    use crate::dct::{
        dct8x16_coeffs, dct8x16_i32_scalar, dct16x16_coeffs, dct16x16_scalar, dct32x32_coeffs,
        dct32x32_scalar,
    };

    const QUANT_PAIRS: &[(i32, i32)] = &[(65536, 65536), (65536, 46341), (32768, 32768)];

    fn lcg(state: &mut u32) -> i32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*state >> 16) as i32 & 0x3ff) - 512
    }

    fn fill_lcg(buf: &mut [i32], seed: u32, mask: i32) {
        let mut s = seed;
        for v in buf.iter_mut() {
            *v = (lcg(&mut s) & mask) - ((mask + 1) >> 1);
        }
    }

    #[test]
    fn dct16x16_coeffs_and_levels_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in [0x1234_5678, 0xdead_beef] {
            let mut input = [0i32; 256];
            fill_lcg(&mut input, seed, 0x1ff);
            assert_eq!(
                dct16x16_coeffs(&input),
                unsafe { dct16x16_avx2_coeffs(&input) },
                "16x16 coeff mismatch seed={seed:#x}"
            );
            for &(dc_q, ac_q) in QUANT_PAIRS {
                let mut s = input;
                let mut a = input;
                dct16x16_scalar(&mut s, dc_q, ac_q);
                unsafe { dct16x16_avx2_i32(&mut a, dc_q, ac_q) };
                assert_eq!(
                    s, a,
                    "16x16 level mismatch seed={seed:#x} dc_q={dc_q} ac_q={ac_q}"
                );
            }
        }
    }

    #[test]
    fn dct32x32_coeffs_and_levels_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in [0x1234_5678, 0xdead_beef] {
            let mut input = [0i32; 1024];
            fill_lcg(&mut input, seed, 0x7f);
            assert_eq!(
                dct32x32_coeffs(&input),
                unsafe { dct32x32_avx2_coeffs(&input) },
                "32x32 coeff mismatch seed={seed:#x}"
            );
            for &(dc_q, ac_q) in QUANT_PAIRS {
                let mut s = input;
                let mut a = input;
                dct32x32_scalar(&mut s, dc_q, ac_q);
                unsafe { dct32x32_avx2_i32(&mut a, dc_q, ac_q) };
                assert_eq!(
                    s, a,
                    "32x32 level mismatch seed={seed:#x} dc_q={dc_q} ac_q={ac_q}"
                );
            }
        }
    }

    #[test]
    fn dct8x16_coeffs_and_levels_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in [0x1234_5678, 0xdead_beef] {
            let mut input = [0i32; 128];
            fill_lcg(&mut input, seed, 0x1ff);
            assert_eq!(
                dct8x16_coeffs(&input),
                unsafe { dct8x16_avx2_coeffs(&input) },
                "8x16 coeff mismatch seed={seed:#x}"
            );
            for &(dc_q, ac_q) in QUANT_PAIRS {
                let mut s = input;
                let mut a = input;
                dct8x16_i32_scalar(&mut s, dc_q, ac_q);
                unsafe { dct8x16_avx2_i32(&mut a, dc_q, ac_q) };
                assert_eq!(
                    s, a,
                    "8x16 level mismatch seed={seed:#x} dc_q={dc_q} ac_q={ac_q}"
                );
            }
        }
    }
}
