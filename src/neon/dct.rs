/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
use std::arch::aarch64::*;
use std::mem::MaybeUninit;

#[derive(Clone, Copy)]
struct I32x8 {
    lo: int32x4_t,
    hi: int32x4_t,
}

#[derive(Clone, Copy)]
struct I32x4(int32x4_t);

impl I32x4 {
    #[inline]
    #[target_feature(enable = "neon")]
    fn add(self, rhs: I32x4) -> I32x4 {
        I32x4(vaddq_s32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn sub(self, rhs: I32x4) -> I32x4 {
        I32x4(vsubq_s32(self.0, rhs.0))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn muls_q16(self, coeff: i32) -> I32x4 {
        let c = vdup_n_s32(coeff);
        I32x4(vcombine_s32(
            vshrn_n_s64(vmull_s32(vget_low_s32(self.0), c), 16),
            vshrn_n_s64(vmull_s32(vget_high_s32(self.0), c), 16),
        ))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn fma_sqrt2(self, b: I32x4) -> I32x4 {
        self.muls_q16(SQRT2).add(b)
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn shr<const N: i32>(self) -> I32x4 {
        I32x4(vshrq_n_s32(self.0, N))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn shl<const N: i32>(self) -> I32x4 {
        I32x4(vshlq_n_s32(self.0, N))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn shr_round<const N: i32>(self) -> I32x4 {
        I32x4(vrshrq_n_s32(self.0, N))
    }
}

impl I32x8 {
    #[inline]
    #[target_feature(enable = "neon")]
    fn add(self, rhs: I32x8) -> I32x8 {
        I32x8 {
            lo: vaddq_s32(self.lo, rhs.lo),
            hi: vaddq_s32(self.hi, rhs.hi),
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn sub(self, rhs: I32x8) -> I32x8 {
        I32x8 {
            lo: vsubq_s32(self.lo, rhs.lo),
            hi: vsubq_s32(self.hi, rhs.hi),
        }
    }

    /// Multiply all lanes by a Q0.16 coefficient, returning truncated Q0.16 result.
    /// Uses 64-bit intermediate to avoid i32 overflow.
    /// Safe for inputs up to ~15 bits and coefficients up to ~10x (WC32[15]).
    #[inline]
    #[target_feature(enable = "neon")]
    fn muls_q16(self, coeff: i32) -> I32x8 {
        let c = vdup_n_s32(coeff);
        I32x8 {
            lo: vcombine_s32(
                vshrn_n_s64::<16>(vmull_s32(vget_low_s32(self.lo), c)),
                vshrn_n_s64::<16>(vmull_s32(vget_high_s32(self.lo), c)),
            ),
            hi: vcombine_s32(
                vshrn_n_s64::<16>(vmull_s32(vget_low_s32(self.hi), c)),
                vshrn_n_s64::<16>(vmull_s32(vget_high_s32(self.hi), c)),
            ),
        }
    }

    /// fma: self * SQRT2_Q16 + b  (matches scalar fmla_sqrt2)
    #[inline]
    #[target_feature(enable = "neon")]
    fn fma_sqrt2(self, b: I32x8) -> I32x8 {
        self.muls_q16(SQRT2).add(b)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_i32(
    r0: int32x4_t,
    r1: int32x4_t,
    r2: int32x4_t,
    r3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v0 = vtrn1q_s32(r0, r1);
    let v1 = vtrn2q_s32(r0, r1);
    let v2 = vtrn1q_s32(r2, r3);
    let v3 = vtrn2q_s32(r2, r3);
    let c0 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(v0),
        vreinterpretq_s64_s32(v2),
    ));
    let c1 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(v1),
        vreinterpretq_s64_s32(v3),
    ));
    let c2 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(v0),
        vreinterpretq_s64_s32(v2),
    ));
    let c3 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(v1),
        vreinterpretq_s64_s32(v3),
    ));
    (c0, c1, c2, c3)
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_8x8_i32(c: &mut [I32x8; 8]) {
    // Transpose the "lo" 4×4 block and the "hi" 4×4 block independently,
    // then recombine — matching the f32 version exactly.
    let (a0, a1, a2, a3) = transpose_4x4_i32(c[0].lo, c[1].lo, c[2].lo, c[3].lo);
    let (b0, b1, b2, b3) = transpose_4x4_i32(c[0].hi, c[1].hi, c[2].hi, c[3].hi);
    let (cc0, cc1, cc2, cc3) = transpose_4x4_i32(c[4].lo, c[5].lo, c[6].lo, c[7].lo);
    let (d0, d1, d2, d3) = transpose_4x4_i32(c[4].hi, c[5].hi, c[6].hi, c[7].hi);

    c[0] = I32x8 { lo: a0, hi: cc0 };
    c[1] = I32x8 { lo: a1, hi: cc1 };
    c[2] = I32x8 { lo: a2, hi: cc2 };
    c[3] = I32x8 { lo: a3, hi: cc3 };
    c[4] = I32x8 { lo: b0, hi: d0 };
    c[5] = I32x8 { lo: b1, hi: d1 };
    c[6] = I32x8 { lo: b2, hi: d2 };
    c[7] = I32x8 { lo: b3, hi: d3 };
}

#[inline]
#[target_feature(enable = "neon")]
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
#[target_feature(enable = "neon")]
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

    // Post-butterfly odd-half combine — mirrors scalar exactly
    odds[0] = odds[0].fma_sqrt2(odds[1]); // odds[0]*SQRT2 + odds[1]
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
#[target_feature(enable = "neon")]
fn dct1d_4_v4_i32(c: &mut [I32x4; 4]) {
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
#[target_feature(enable = "neon")]
fn dct1d_8_v4_i32(c: &mut [I32x4; 8]) {
    let mut evens = [
        c[0].add(c[7]),
        c[1].add(c[6]),
        c[2].add(c[5]),
        c[3].add(c[4]),
    ];
    dct1d_4_v4_i32(&mut evens);

    let mut odds = [
        c[0].sub(c[7]).muls_q16(WC8_0),
        c[1].sub(c[6]).muls_q16(WC8_1),
        c[2].sub(c[5]).muls_q16(WC8_2),
        c[3].sub(c[4]).muls_q16(WC8_3),
    ];
    dct1d_4_v4_i32(&mut odds);

    // Post-butterfly odd-half combine — mirrors scalar exactly
    odds[0] = odds[0].fma_sqrt2(odds[1]); // odds[0]*SQRT2 + odds[1]
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
#[target_feature(enable = "neon")]
fn dct1d_16_v4_i32(c: &mut [I32x4; 16]) {
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

    dct1d_8_v4_i32(&mut evens);
    dct1d_8_v4_i32(&mut odds);

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
#[target_feature(enable = "neon")]
fn dct1d_32_v4_i32(c: &mut [I32x4; 32]) {
    let mut evens = std::array::from_fn::<I32x4, 16, _>(|i| c[i].add(c[31 - i]));
    let mut odds = std::array::from_fn::<I32x4, 16, _>(|i| c[i].sub(c[31 - i]));

    dct1d_16_v4_i32(&mut evens);

    for i in 0..16 {
        odds[i] = odds[i].muls_q16(WC32[i]);
    }
    dct1d_16_v4_i32(&mut odds);

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
#[target_feature(enable = "neon")]
fn load_n_i32x4<const N: usize>(ptr: &[i32], stride: usize) -> [I32x4; N] {
    unsafe {
        std::array::from_fn(|y| {
            let p = ptr.get_unchecked(y * stride..).as_ptr();
            I32x4(vld1q_s32(p))
        })
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(ptr: *const i32) -> I32x4 {
    unsafe { I32x4(vld1q_s32(ptr)) }
}

#[inline]
#[target_feature(enable = "neon")]
unsafe fn store_i32x4(dst: *mut i32, v: I32x4) {
    unsafe {
        vst1q_s32(dst, v.0);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_store_4x4_i32(dst: *mut i32, stride: usize, tile: &mut [I32x4; 4]) {
    let (r0, r1, r2, r3) = transpose_4x4_i32(tile[0].0, tile[1].0, tile[2].0, tile[3].0);
    unsafe {
        vst1q_s32(dst, r0);
        vst1q_s32(dst.add(stride), r1);
        vst1q_s32(dst.add(2 * stride), r2);
        vst1q_s32(dst.add(3 * stride), r3);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_transposed_cols_i32x4<const N: usize>(dst: *mut i32, x: usize, c: &[I32x4; N]) {
    debug_assert!(N.is_multiple_of(4));
    let stride = N;
    let mut v = 0usize;
    while v < N {
        let mut tile = [c[v], c[v + 1], c[v + 2], c[v + 3]];
        transpose_store_4x4_i32(unsafe { dst.add(x * N + v) }, stride, &mut tile);
        v += 4;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load8_i32(ptr: &[i32], stride: usize) -> [I32x8; 8] {
    unsafe {
        let row = |y: usize| {
            let p = ptr.get_unchecked(y * stride..);
            I32x8 {
                lo: vld1q_s32(p.as_ptr()),
                hi: vld1q_s32(p.get_unchecked(4..).as_ptr()),
            }
        };
        std::array::from_fn(row)
    }
}

/// Vectorized mul_q16: (data * coeff_vec) >> 16, lane-wise.
#[inline]
#[target_feature(enable = "neon")]
fn mul_q16_vec(data: int32x4_t, coeff: int32x4_t) -> int32x4_t {
    vcombine_s32(
        vshrn_n_s64::<16>(vmull_s32(vget_low_s32(data), vget_low_s32(coeff))),
        vshrn_n_s64::<16>(vmull_s32(vget_high_s32(data), vget_high_s32(coeff))),
    )
}

#[inline]
fn quant_flat<const N: usize>(coeffs: &[i32; N], dc_q: i32, ac_q: i32, out: &mut [i32; N]) {
    // Round-to-nearest (magnitude-symmetric) so the quant error is zero-mean,
    // matching the scalar `quant_q16`. A bare `>> 16` truncates toward -inf and
    // the bias accumulates into a dark dot at the block's top-left corner.
    let mq = |a: i32, b: i32| {
        let prod = (a as i64) * (b as i64);
        let mag = prod.unsigned_abs();
        if mag < 32768 {
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
#[target_feature(enable = "neon")]
fn quant_q16_half_i64(prod: int64x2_t) -> int32x2_t {
    let zero64 = vdupq_n_s64(0);
    let mag = vabsq_s64(prod);
    let active = vcgtq_s64(mag, vdupq_n_s64(32767));
    let lvl = vshrq_n_s64::<16>(vaddq_s64(mag, vdupq_n_s64(32768)));
    let neg = vnegq_s64(lvl);
    let signed = vbslq_s64(vcltq_s64(prod, zero64), neg, lvl);
    vmovn_s64(vbslq_s64(active, signed, zero64))
}

#[inline]
#[target_feature(enable = "neon")]
fn quant_q16_vec_i32(v: int32x4_t, q: int32x4_t) -> int32x4_t {
    vcombine_s32(
        quant_q16_half_i64(vmull_s32(vget_low_s32(v), vget_low_s32(q))),
        quant_q16_half_i64(vmull_s32(vget_high_s32(v), vget_high_s32(q))),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn store_quant_target_i32x4(
    cf: *mut i32,
    tf: *mut f32,
    coeff: I32x4,
    base: usize,
    dc_q: i32,
    ac_q: i32,
) {
    let q = if base == 0 {
        vsetq_lane_s32::<0>(dc_q, vdupq_n_s32(ac_q))
    } else {
        vdupq_n_s32(ac_q)
    };
    let levels = quant_q16_vec_i32(coeff.0, q);
    let target = vmulq_f32(
        vmulq_f32(vcvtq_f32_s32(coeff.0), vcvtq_f32_s32(q)),
        vdupq_n_f32(1.0 / 65536.0),
    );
    unsafe {
        vst1q_s32(cf.add(base), levels);
        vst1q_f32(tf.add(base), target);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x8_neon_quant_t(
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
        store_quant_target_i32x4(
            cf.as_mut_ptr().cast(),
            tf.as_mut_ptr().cast(),
            I32x4(col.lo),
            k * 8,
            dc_q,
            ac_q,
        );
        store_quant_target_i32x4(
            cf.as_mut_ptr().cast(),
            tf.as_mut_ptr().cast(),
            I32x4(col.hi),
            k * 8 + 4,
            dc_q,
            ac_q,
        );
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn adst1d_16_v4_i32(c: &mut [I32x4; 16]) {
    let mut out = [I32x4(vdupq_n_s32(0)); 16];
    for o in 0..16usize {
        let mut lo = vdupq_n_s64(2048);
        let mut hi = vdupq_n_s64(2048);
        for j in 0..16usize {
            let k = vdup_n_s32(crate::dct::ADST16_FWD_Q12[o][j] as i32);
            lo = vmlal_s32(lo, vget_low_s32(c[j].0), k);
            hi = vmlal_s32(hi, vget_high_s32(c[j].0), k);
        }
        out[o] = I32x4(vcombine_s32(vshrn_n_s64::<12>(lo), vshrn_n_s64::<12>(hi)));
    }
    *c = out;
}

#[inline]
#[target_feature(enable = "neon")]
fn tx16x16_adst_dct_neon_quant_t<const COL_ADST: bool, const ROW_ADST: bool>(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp_u = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(4) {
        let mut cols = load_n_i32x4::<16>(&input[x..], 16);
        if COL_ADST {
            adst1d_16_v4_i32(&mut cols);
        } else {
            dct1d_16_v4_i32(&mut cols);
        }
        store_transposed_cols_i32x4::<16>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 256]>::uninit();
    let mut tf = MaybeUninit::<[f32; 256]>::uninit();
    for y in (0..16usize).step_by(4) {
        let mut rows: [I32x4; 16] =
            std::array::from_fn(|x| load_i32x4(unsafe { tmp.as_ptr().add(x * 16 + y) }));
        if ROW_ADST {
            adst1d_16_v4_i32(&mut rows);
        } else {
            dct1d_16_v4_i32(&mut rows);
        }
        for u in 0..16usize {
            store_quant_target_i32x4(
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

#[target_feature(enable = "neon")]
pub(crate) fn adst16x16_neon_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_neon_quant_t::<true, true>(input, dc_q, ac_q)
}

#[target_feature(enable = "neon")]
pub(crate) fn adstdct16x16_neon_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_neon_quant_t::<true, false>(input, dc_q, ac_q)
}

#[target_feature(enable = "neon")]
pub(crate) fn dctadst16x16_neon_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    tx16x16_adst_dct_neon_quant_t::<false, true>(input, dc_q, ac_q)
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x16_neon_quant_t(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp_u = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(4) {
        let mut cols = load_n_i32x4::<16>(&input[x..], 16);
        dct1d_16_v4_i32(&mut cols);
        store_transposed_cols_i32x4::<16>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 256]>::uninit();
    let mut tf = MaybeUninit::<[f32; 256]>::uninit();
    for y in (0..16usize).step_by(4) {
        let mut rows: [I32x4; 16] =
            std::array::from_fn(|x| load_i32x4(unsafe { tmp.as_ptr().add(x * 16 + y) }));
        dct1d_16_v4_i32(&mut rows);
        for u in 0..16usize {
            store_quant_target_i32x4(
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

#[target_feature(enable = "neon")]
pub(crate) fn dct32x32_neon_coeffs(input: &[i32; 1024]) -> [i32; 1024] {
    // Stage 1: vertical DCT-32 in four-column groups. Store a true transposed
    // scratch: tmp[x * 32 + vertical_frequency].
    let mut tmp_u = MaybeUninit::<[i32; 1024]>::uninit();
    for x in (0..32usize).step_by(4) {
        let mut cols = load_n_i32x4::<32>(&input[x..], 32);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v4_i32(&mut cols);
        store_transposed_cols_i32x4::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    // Stage 2: horizontal DCT-32 with contiguous loads from transposed scratch.
    let mut out = MaybeUninit::<[i32; 1024]>::uninit();
    for y in (0..32usize).step_by(4) {
        let mut rows: [I32x4; 32] =
            std::array::from_fn(|x| load_i32x4(unsafe { tmp.as_ptr().add(x * 32 + y) }));
        dct1d_32_v4_i32(&mut rows);
        for u in 0..32usize {
            let n = rows[u].shr_round::<8>();
            unsafe {
                store_i32x4((out.as_mut_ptr() as *mut i32).add(u * 32 + y), n);
            }
        }
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct32x32_neon_quant_t(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    let mut tmp_u = MaybeUninit::<[i32; 1024]>::uninit();
    for x in (0..32usize).step_by(4) {
        let mut cols = load_n_i32x4::<32>(&input[x..], 32);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v4_i32(&mut cols);
        store_transposed_cols_i32x4::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 1024]>::uninit();
    let mut tf = MaybeUninit::<[f32; 1024]>::uninit();
    for y in (0..32usize).step_by(4) {
        let mut rows: [I32x4; 32] =
            std::array::from_fn(|x| load_i32x4(unsafe { tmp.as_ptr().add(x * 32 + y) }));
        dct1d_32_v4_i32(&mut rows);
        for u in 0..32usize {
            store_quant_target_i32x4(
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

#[target_feature(enable = "neon")]
pub(crate) fn dct32x32_neon_i32(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    let coeffs = dct32x32_neon_coeffs(input);
    quant_flat(&coeffs, dc_q, ac_q, input);
}

/// Forward 2-D integer 8x16 DCT (NEON) -> normalized coefficients
/// `out[fx*16 + fy]` (horiz freq fx 0..8, vert freq fy 0..16; DC at index 0),
/// pre-quantization. Mirrors scalar `dct8x16_coeffs`.
#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon_coeffs(input: &[i32; 128]) -> [i32; 128] {
    // Stage 1: vertical DCT-16 in two 4-column groups.  This avoids the old
    // [I32x8; 16] shape, which is 32 physical NEON registers before any
    // butterfly temporaries.  Store row-major scratch: tmp[fy * 8 + x].
    let mut tmp_u = MaybeUninit::<[i32; 128]>::uninit();
    for x in (0..8usize).step_by(4) {
        let mut cols: [I32x4; 16] =
            std::array::from_fn(|y| load_i32x4(unsafe { input.as_ptr().add(y * 8 + x) }));
        dct1d_16_v4_i32(&mut cols);
        for fy in 0..16usize {
            unsafe {
                store_i32x4((tmp_u.as_mut_ptr() as *mut i32).add(fy * 8 + x), cols[fy]);
            }
        }
    }
    let tmp = unsafe { tmp_u.assume_init() };

    // Stage 2: horizontal DCT-8 in 4-frequency groups.  Load four contiguous
    // scratch rows, transpose two 4x4 tiles normally, then run an 8-point DCT
    // with one int32x4_t register set.
    let nrm = vdupq_n_s32(46341);
    let mut out = MaybeUninit::<[i32; 128]>::uninit();
    for y in (0..16usize).step_by(4) {
        let r0 = unsafe { vld1q_s32(tmp.as_ptr().add(y * 8)) };
        let r1 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 1) * 8)) };
        let r2 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 2) * 8)) };
        let r3 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 3) * 8)) };
        let (c0, c1, c2, c3) = transpose_4x4_i32(r0, r1, r2, r3);

        let r4 = unsafe { vld1q_s32(tmp.as_ptr().add(y * 8 + 4)) };
        let r5 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 1) * 8 + 4)) };
        let r6 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 2) * 8 + 4)) };
        let r7 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 3) * 8 + 4)) };
        let (c4, c5, c6, c7) = transpose_4x4_i32(r4, r5, r6, r7);

        let mut rows = [
            I32x4(c0),
            I32x4(c1),
            I32x4(c2),
            I32x4(c3),
            I32x4(c4),
            I32x4(c5),
            I32x4(c6),
            I32x4(c7),
        ];
        dct1d_8_v4_i32(&mut rows);
        for fx in 0..8usize {
            unsafe {
                vst1q_s32(
                    (out.as_mut_ptr() as *mut i32).add(fx * 16 + y),
                    mul_q16_vec(rows[fx].0, nrm),
                );
            }
        }
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon_quant_t(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
    let mut tmp_u = MaybeUninit::<[i32; 128]>::uninit();
    for x in (0..8usize).step_by(4) {
        let mut cols: [I32x4; 16] =
            std::array::from_fn(|y| load_i32x4(unsafe { input.as_ptr().add(y * 8 + x) }));
        dct1d_16_v4_i32(&mut cols);
        for fy in 0..16usize {
            unsafe {
                store_i32x4((tmp_u.as_mut_ptr() as *mut i32).add(fy * 8 + x), cols[fy]);
            }
        }
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let nrm = vdupq_n_s32(46341);
    let mut cf = MaybeUninit::<[i32; 128]>::uninit();
    let mut tf = MaybeUninit::<[f32; 128]>::uninit();
    for y in (0..16usize).step_by(4) {
        let r0 = unsafe { vld1q_s32(tmp.as_ptr().add(y * 8)) };
        let r1 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 1) * 8)) };
        let r2 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 2) * 8)) };
        let r3 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 3) * 8)) };
        let (c0, c1, c2, c3) = transpose_4x4_i32(r0, r1, r2, r3);

        let r4 = unsafe { vld1q_s32(tmp.as_ptr().add(y * 8 + 4)) };
        let r5 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 1) * 8 + 4)) };
        let r6 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 2) * 8 + 4)) };
        let r7 = unsafe { vld1q_s32(tmp.as_ptr().add((y + 3) * 8 + 4)) };
        let (c4, c5, c6, c7) = transpose_4x4_i32(r4, r5, r6, r7);

        let mut rows = [
            I32x4(c0),
            I32x4(c1),
            I32x4(c2),
            I32x4(c3),
            I32x4(c4),
            I32x4(c5),
            I32x4(c6),
            I32x4(c7),
        ];
        dct1d_8_v4_i32(&mut rows);
        for fx in 0..8usize {
            store_quant_target_i32x4(
                cf.as_mut_ptr().cast(),
                tf.as_mut_ptr().cast(),
                I32x4(mul_q16_vec(rows[fx].0, nrm)),
                fx * 16 + y,
                dc_q,
                ac_q,
            );
        }
    }
    unsafe { (cf.assume_init(), tf.assume_init()) }
}

#[allow(unused)]
#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon_i32(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x16_neon_coeffs(input);
    quant_flat(&coeffs, dc_q, ac_q, input);
}

#[target_feature(enable = "neon")]
pub(crate) fn dct16x32_neon_quant_t(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    // Match scalar TX_16X32 ordering: vertical DCT-32 with input << 6,
    // horizontal DCT-16, then rounded >> 6 before quant/target generation.
    let mut tmp_u = MaybeUninit::<[i32; 512]>::uninit();
    for x in (0..16usize).step_by(4) {
        let mut cols = load_n_i32x4::<32>(&input[x..], 16);
        for c in cols.iter_mut() {
            *c = c.shl::<6>();
        }
        dct1d_32_v4_i32(&mut cols);
        store_transposed_cols_i32x4::<32>(tmp_u.as_mut_ptr().cast(), x, &cols);
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 512]>::uninit();
    let mut tf = MaybeUninit::<[f32; 512]>::uninit();
    for fy in (0..32usize).step_by(4) {
        let mut rows: [I32x4; 16] =
            std::array::from_fn(|x| load_i32x4(unsafe { tmp.as_ptr().add(x * 32 + fy) }));
        dct1d_16_v4_i32(&mut rows);
        for fx in 0..16usize {
            store_quant_target_i32x4(
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

#[target_feature(enable = "neon")]
pub(crate) fn dct32x16_neon_quant_t(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    // Match scalar TX_32X16 ordering: horizontal DCT-32 first. Four rows are
    // processed in one int32x4_t lane set, then stored as normal row-major
    // scratch for the vertical DCT-16 pass.
    let mut tmp_u = MaybeUninit::<[i32; 512]>::uninit();
    let tmp_ptr = tmp_u.as_mut_ptr().cast::<i32>();
    for y in (0..16usize).step_by(4) {
        let z = I32x4(vdupq_n_s32(0));
        let mut cols = [z; 32];
        for x in (0..32usize).step_by(4) {
            let r0 = unsafe { vld1q_s32(input.as_ptr().add(y * 32 + x)) };
            let r1 = unsafe { vld1q_s32(input.as_ptr().add((y + 1) * 32 + x)) };
            let r2 = unsafe { vld1q_s32(input.as_ptr().add((y + 2) * 32 + x)) };
            let r3 = unsafe { vld1q_s32(input.as_ptr().add((y + 3) * 32 + x)) };
            let (c0, c1, c2, c3) = transpose_4x4_i32(r0, r1, r2, r3);
            cols[x] = I32x4(c0).shl::<6>();
            cols[x + 1] = I32x4(c1).shl::<6>();
            cols[x + 2] = I32x4(c2).shl::<6>();
            cols[x + 3] = I32x4(c3).shl::<6>();
        }
        dct1d_32_v4_i32(&mut cols);

        for fx in (0..32usize).step_by(4) {
            let mut tile = [cols[fx], cols[fx + 1], cols[fx + 2], cols[fx + 3]];
            transpose_store_4x4_i32(unsafe { tmp_ptr.add(y * 32 + fx) }, 32, &mut tile);
        }
    }
    let tmp = unsafe { tmp_u.assume_init() };

    let mut cf = MaybeUninit::<[i32; 512]>::uninit();
    let mut tf = MaybeUninit::<[f32; 512]>::uninit();
    for fx in (0..32usize).step_by(4) {
        let mut rows: [I32x4; 16] =
            std::array::from_fn(|y| load_i32x4(unsafe { tmp.as_ptr().add(y * 32 + fx) }));
        dct1d_16_v4_i32(&mut rows);

        for fy in (0..16usize).step_by(4) {
            let mut tile = [
                rows[fy].shr_round::<6>(),
                rows[fy + 1].shr_round::<6>(),
                rows[fy + 2].shr_round::<6>(),
                rows[fy + 3].shr_round::<6>(),
            ];
            let (c0, c1, c2, c3) = transpose_4x4_i32(tile[0].0, tile[1].0, tile[2].0, tile[3].0);
            tile = [I32x4(c0), I32x4(c1), I32x4(c2), I32x4(c3)];
            for i in 0..4usize {
                store_quant_target_i32x4(
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
mod neon_vs_scalar {
    use crate::dct::{dct8x16_i32_scalar, dct32x32_scalar};
    use crate::neon::{dct8x16_neon_i32, dct32x32_neon_i32};

    /// Simple 32-bit LCG for deterministic pseudo-random inputs in -512..=511
    /// (well within the safe range for WC32[15] ≈ 10×). NOTE: a real spread of
    /// values is required so these tests actually exercise the transform LAYOUT
    /// (a flat/constant block is symmetric and hides transpose/orientation bugs).
    fn lcg(state: &mut u32) -> i32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*state >> 16) as i32 & 0x3FF) - 512
    }

    fn fill_lcg(buf: &mut [i32], seed: u32) {
        let mut s = seed;
        for v in buf.iter_mut() {
            *v = lcg(&mut s);
        }
    }

    /// Ramp: values −128, −127, …, wrapping at 256.
    fn fill_ramp(buf: &mut [i32]) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i % 256) as i32 - 128;
        }
    }

    /// Alternating ±64 to avoid overflow through deep butterfly chains.
    fn fill_alt(buf: &mut [i32]) {
        for (i, v) in buf.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 64 } else { -64 };
        }
    }

    // (dc_q, ac_q) pairs in Q0.16: 1.0 = 65536, √2/2 ≈ 46341.
    const QUANT_PAIRS: &[(i32, i32)] = &[
        (65536, 65536), // identity
        (65536, 46341), // DC full-scale, AC √2/2
        (32768, 32768), // both halved
    ];

    fn run_32x32(input: [i32; 1024], dc_q: i32, ac_q: i32) -> ([i32; 1024], [i32; 1024]) {
        let mut scalar = input;
        dct32x32_scalar(&mut scalar, dc_q, ac_q);
        let mut neon = input;
        unsafe { dct32x32_neon_i32(&mut neon, dc_q, ac_q) };
        (scalar, neon)
    }

    #[test]
    fn test_32x32_zeros() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_32x32([0i32; 1024], dc_q, ac_q);
            assert_eq!(s, n, "32x32 zeros dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_32x32_constant() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            // Small constant to stay within range through WC32[15] ≈ 10× amplification.
            let (s, n) = run_32x32([8i32; 1024], dc_q, ac_q);
            assert_eq!(s, n, "32x32 constant dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_32x32_ramp() {
        let mut input = [0i32; 1024];
        // Narrower range for 32×32 due to deeper butterfly accumulation.
        for (i, v) in input.iter_mut().enumerate() {
            *v = (i % 128) as i32 - 64;
        }
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_32x32(input, dc_q, ac_q);
            let first = s.iter().zip(n.iter()).position(|(a, b)| a != b);
            assert_eq!(
                s, n,
                "32x32 ramp dc_q={dc_q} ac_q={ac_q}: mismatch at {first:?}"
            );
        }
    }

    #[test]
    fn test_32x32_random_seed0() {
        let mut input = [0i32; 1024];
        // ±127 range to avoid overflow through 32-point WC32 cascade
        let mut s = 0xDEAD_BEEFu32;
        for v in input.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (s >> 16) as i32 % 128 - 64;
        }
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (sc, n) = run_32x32(input, dc_q, ac_q);
            let first = sc.iter().zip(n.iter()).position(|(a, b)| a != b);
            assert_eq!(
                sc, n,
                "32x32 rand(DEADBEEF) dc_q={dc_q} ac_q={ac_q}: mismatch at {first:?}"
            );
        }
    }

    #[test]
    fn test_32x32_random_seed1() {
        let mut input = [0i32; 1024];
        let mut s = 0x1234_5678u32;
        for v in input.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (s >> 16) as i32 % 128 - 64;
        }
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (sc, n) = run_32x32(input, dc_q, ac_q);
            assert_eq!(sc, n, "32x32 rand(12345678) dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_32x32_high_amplitude_parity() {
        // The forward DCT-32 now pre-shifts the residual by B=6 headroom bits, so
        // the cascade runs at far higher magnitudes than the ±127 used above. This
        // guards scalar<->NEON parity (and absence of i32 overflow) at the worst
        // case: full bd=12 residual range (±4095) in adversarial patterns.
        let amp = 4095i32;
        let mk = |f: &dyn Fn(usize, usize) -> i32| {
            let mut input = [0i32; 1024];
            for y in 0..32 {
                for x in 0..32 {
                    input[y * 32 + x] = f(x, y);
                }
            }
            input
        };
        let checker = mk(&|x, y| if (x + y) & 1 == 0 { amp } else { -amp });
        let vstripe = mk(&|x, _| if x & 1 == 0 { amp } else { -amp });
        let mut rnd = [0i32; 1024];
        let mut s = 0xABCD_1234u32;
        for v in rnd.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (s >> 16) as i32 % (2 * amp + 1) - amp;
        }
        for input in [checker, vstripe, rnd] {
            for &(dc_q, ac_q) in QUANT_PAIRS {
                let (sc, n) = run_32x32(input, dc_q, ac_q);
                assert_eq!(sc, n, "32x32 high-amplitude dc_q={dc_q} ac_q={ac_q}");
            }
        }
    }

    fn run_8x16(input: [i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [i32; 128]) {
        let mut scalar = input;
        dct8x16_i32_scalar(&mut scalar, dc_q, ac_q);
        let mut neon = input;
        unsafe { dct8x16_neon_i32(&mut neon, dc_q, ac_q) };
        (scalar, neon)
    }

    #[test]
    fn test_8x16_zeros() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16([0i32; 128], dc_q, ac_q);
            assert_eq!(s, n, "8x16 zeros dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x16_constant() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16([64i32; 128], dc_q, ac_q);
            assert_eq!(s, n, "8x16 constant dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x16_ramp() {
        let mut input = [0i32; 128];
        fill_ramp(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16(input, dc_q, ac_q);
            assert_eq!(s, n, "8x16 ramp dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x16_alternating() {
        let mut input = [0i32; 128];
        fill_alt(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16(input, dc_q, ac_q);
            assert_eq!(s, n, "8x16 alternating dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x16_random_seed0() {
        let mut input = [0i32; 128];
        fill_lcg(&mut input, 0xDEAD_BEEF);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16(input, dc_q, ac_q);
            let first = s.iter().zip(n.iter()).position(|(a, b)| a != b);
            assert_eq!(
                s, n,
                "8x16 rand(DEADBEEF) dc_q={dc_q} ac_q={ac_q}: mismatch at {first:?}"
            );
        }
    }

    #[test]
    fn test_8x16_random_seed1() {
        let mut input = [0i32; 128];
        fill_lcg(&mut input, 0x1234_5678);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x16(input, dc_q, ac_q);
            assert_eq!(s, n, "8x16 rand(12345678) dc_q={dc_q} ac_q={ac_q}");
        }
    }
}
