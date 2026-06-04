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

use crate::dct::{
    SQRT2, WC4_0, WC4_1, WC8_0, WC8_1, WC8_2, WC8_3, WC16_0, WC16_1, WC16_2, WC16_3, WC16_4,
    WC16_5, WC16_6, WC16_7, WC32,
};
use std::arch::aarch64::*;

// ── Vector type: two int32x4_t lanes = 8 × i32 ───────────────────────────────

#[derive(Clone, Copy)]
struct I32x8 {
    lo: int32x4_t, // lanes 0-3
    hi: int32x4_t, // lanes 4-7
}

impl I32x8 {
    #[inline]
    #[target_feature(enable = "neon")]
    fn zero() -> Self {
        Self {
            lo: vdupq_n_s32(0),
            hi: vdupq_n_s32(0),
        }
    }

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
                vshrn_n_s64(vmull_s32(vget_low_s32(self.lo), c), 16),
                vshrn_n_s64(vmull_s32(vget_high_s32(self.lo), c), 16),
            ),
            hi: vcombine_s32(
                vshrn_n_s64(vmull_s32(vget_low_s32(self.hi), c), 16),
                vshrn_n_s64(vmull_s32(vget_high_s32(self.hi), c), 16),
            ),
        }
    }

    /// fma: self * SQRT2_Q16 + b  (matches scalar fmla_sqrt2)
    #[inline]
    #[target_feature(enable = "neon")]
    fn fma_sqrt2(self, b: I32x8) -> I32x8 {
        self.muls_q16(SQRT2).add(b)
    }

    /// Arithmetic right shift by N (normalize / quantize step).
    #[inline]
    #[target_feature(enable = "neon")]
    fn shr<const N: i32>(self) -> I32x8 {
        I32x8 {
            lo: vshrq_n_s32(self.lo, N),
            hi: vshrq_n_s32(self.hi, N),
        }
    }
}

// ── 4×4 and 8×8 transpose (identical structure to f32 version) ───────────────

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

// ── 1-D DCT kernels ───────────────────────────────────────────────────────────

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

// ── Load / store helpers ──────────────────────────────────────────────────────

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
        std::array::from_fn(|i| row(i))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load16_i32(ptr: &[i32], stride: usize) -> [I32x8; 16] {
    unsafe {
        let row = |y: usize| {
            let p = ptr.get_unchecked(y * stride..);
            I32x8 {
                lo: vld1q_s32(p.as_ptr()),
                hi: vld1q_s32(p.get_unchecked(4..).as_ptr()),
            }
        };
        std::array::from_fn(|i| row(i))
    }
}

/// Shift right by N and store 8 rows × 8 columns into a flat [i32; 64].
#[inline]
#[target_feature(enable = "neon")]
fn shr_and_store<const N: i32>(cols: &[I32x8; 8], out: &mut [i32; 64]) {
    for (k, col) in cols.iter().enumerate() {
        unsafe {
            vst1q_s32(out[k * 8..].as_mut_ptr(), vshrq_n_s32(col.lo, N));
            vst1q_s32(out[k * 8 + 4..].as_mut_ptr(), vshrq_n_s32(col.hi, N));
        }
    }
}

/// Quantize (shift + mul_q16) and store into [i32; 64].
/// DC at (0,0) uses dc_q, everything else uses ac_q.
#[inline]
#[target_feature(enable = "neon")]
fn quant_and_store(cols: &[I32x8; 8], dc_q: i32, ac_q: i32, out: &mut [i32; 64]) {
    let q_ac_mul = vdupq_n_s32(ac_q);
    let q_dc_mul = vsetq_lane_s32::<0>(dc_q, q_ac_mul);

    for k in 0..8 {
        let col = cols[k];
        let norm_lo = col.lo;
        let norm_hi = col.hi;

        // Q0.16 multiply each lane by its quantizer
        let q_lo = mul_q16_vec(norm_lo, if k == 0 { q_dc_mul } else { q_ac_mul });
        let q_hi = mul_q16_vec(norm_hi, q_ac_mul);

        unsafe {
            vst1q_s32(out[k * 8..].as_mut_ptr(), q_lo);
            vst1q_s32(out[k * 8 + 4..].as_mut_ptr(), q_hi);
        }
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

#[target_feature(enable = "neon")]
pub(crate) fn dct8x8_neon_i32(input: &mut [i32; 64], dc_q: i32, ac_q: i32) {
    let mut cols = load8_i32(input.as_slice(), 8);
    dct1d_8_v_i32(&mut cols);
    transpose_8x8_i32(&mut cols);
    dct1d_8_v_i32(&mut cols);
    transpose_8x8_i32(&mut cols);
    quant_and_store(&cols, dc_q, ac_q, input);
}
/// 16×16 DCT with quantization. Normalizes by 1/256 (>> 8).
#[target_feature(enable = "neon")]
pub(crate) fn dct16x16_neon_i32(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    unsafe {
        // Column-wise DCT-16 on left half (cols 0-7) and right half (cols 8-15)
        let mut c_l = load16_i32(input.as_slice(), 16);
        let mut c_r = load16_i32(&input[8..], 16);

        dct1d_16_v_i32(&mut c_l);
        dct1d_16_v_i32(&mut c_r);

        // Split into row-frequency groups 0..8 and 8..16, then transpose each
        let mut top_l: [I32x8; 8] = c_l[..8].try_into().unwrap();
        let mut bot_l: [I32x8; 8] = c_l[8..16].try_into().unwrap();
        let mut top_r: [I32x8; 8] = c_r[0..8].try_into().unwrap();
        let mut bot_r: [I32x8; 8] = c_r[8..16].try_into().unwrap();

        transpose_8x8_i32(&mut top_l);
        transpose_8x8_i32(&mut bot_l);
        transpose_8x8_i32(&mut top_r);
        transpose_8x8_i32(&mut bot_r);

        let mut d_a = [I32x8::zero(); 16];
        let mut d_b = [I32x8::zero(); 16];
        d_a[0..8].copy_from_slice(&top_l);
        d_a[8..16].copy_from_slice(&top_r);
        d_b[0..8].copy_from_slice(&bot_l);
        d_b[8..16].copy_from_slice(&bot_r);

        // Row-wise DCT-16
        dct1d_16_v_i32(&mut d_a);
        dct1d_16_v_i32(&mut d_b);

        // FIX 2: build q_dc_mul without a dangling temporary pointer.
        // vsetq_lane_s32 replaces lane 0 of an all-ac_q vector with dc_q.
        let q_ac_mul = vdupq_n_s32(ac_q);
        let q_dc_mul = vsetq_lane_s32::<0>(dc_q, q_ac_mul);

        for u in 0..16usize {
            let va = &d_a[u];
            let vb = &d_b[u];

            let norm = |v: int32x4_t| v;

            let q_a_lo = mul_q16_vec(norm(va.lo), if u == 0 { q_dc_mul } else { q_ac_mul });
            let q_a_hi = mul_q16_vec(norm(va.hi), q_ac_mul);
            let q_b_lo = mul_q16_vec(norm(vb.lo), q_ac_mul);
            let q_b_hi = mul_q16_vec(norm(vb.hi), q_ac_mul);

            let base = &mut input[u * 16..];
            vst1q_s32(base.as_mut_ptr().cast(), q_a_lo);
            vst1q_s32(base[4..].as_mut_ptr().cast(), q_a_hi);
            vst1q_s32(base[8..].as_mut_ptr().cast(), q_b_lo);
            vst1q_s32(base[12..].as_mut_ptr().cast(), q_b_hi);
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn wc32_vecs() -> [int32x4_t; 4] {
    unsafe {
        [
            vld1q_s32(WC32[0..4].as_ptr()),   // k=0..3
            vld1q_s32(WC32[4..8].as_ptr()),   // k=4..7
            vld1q_s32(WC32[8..12].as_ptr()),  // k=8..11
            vld1q_s32(WC32[12..16].as_ptr()), // k=12..15
        ]
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dct1d_32_v_i32(c: &mut [I32x8; 32]) {
    let mut evens = std::array::from_fn::<I32x8, 16, _>(|i| c[i].add(c[31 - i]));
    let mut odds = std::array::from_fn::<I32x8, 16, _>(|i| c[i].sub(c[31 - i]));

    // ── Even half: recurse with DCT-16 ───────────────────────────────────────
    dct1d_16_v_i32(&mut evens);

    // ── Odd half: scale by WC32, then DCT-16 ─────────────────────────────────
    // Scale lane-by-lane using the scalar WC32 coefficients.
    // Each I32x8 holds the same logical element across 8 independent columns,
    // so the coefficient is scalar (same for all 8 lanes).
    for i in 0..16 {
        odds[i] = odds[i].muls_q16(WC32[i]);
    }
    dct1d_16_v_i32(&mut odds);

    // ── Post-butterfly odd-half combine chain ─────────────────────────────────
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

    // ── Interleave even/odd outputs ───────────────────────────────────────────
    for i in 0..16 {
        c[2 * i] = evens[i];
        c[2 * i + 1] = odds[i];
    }
}

// ── Load helpers for 32-wide rows/columns ────────────────────────────────────

/// Load 32 rows × 8 cols (left or right half) with given column stride.
#[inline]
#[target_feature(enable = "neon")]
fn load32_i32(ptr: &[i32], stride: usize) -> [I32x8; 32] {
    unsafe {
        std::array::from_fn(|y| {
            let p = &ptr[y * stride..];
            I32x8 {
                lo: vld1q_s32(p.as_ptr()),
                hi: vld1q_s32(p[4..].as_ptr()),
            }
        })
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_32x8_to_8x32(c: &mut [I32x8; 32]) {
    let mut q0: [I32x8; 8] = c[0..8].try_into().unwrap();
    let mut q1: [I32x8; 8] = c[8..16].try_into().unwrap();
    let mut q2: [I32x8; 8] = c[16..24].try_into().unwrap();
    let mut q3: [I32x8; 8] = c[24..32].try_into().unwrap();

    transpose_8x8_i32(&mut q0);
    transpose_8x8_i32(&mut q1);
    transpose_8x8_i32(&mut q2);
    transpose_8x8_i32(&mut q3);

    c[0..8].copy_from_slice(&q0);
    c[8..16].copy_from_slice(&q1);
    c[16..24].copy_from_slice(&q2);
    c[24..32].copy_from_slice(&q3);
}

#[target_feature(enable = "neon")]
pub(crate) fn dct32x32_neon_i32(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    let mut tmp = [0i32; 1024];

    // ── Pass 1: column-wise DCT-32 ────────────────────────────────────────────
    // Process 4 groups of 8 columns (col_start = 0, 8, 16, 24).
    // load32_i32 reads 32 rows × 8 cols at stride=32 from input[col_start..].
    // Bounds: max index = col_start + 31*32 + 7 = col_start + 999 ≤ 24+999 = 1023 ✓
    for group in 0..4usize {
        let col_start = group * 8;
        let mut cols = load32_i32(&input[col_start..], 32);
        dct1d_32_v_i32(&mut cols);
        for v in 0..32usize {
            let base = &mut tmp[v * 32 + col_start..];
            unsafe {
                vst1q_s32(base.as_mut_ptr(), cols[v].lo);
                vst1q_s32(base[4..].as_mut_ptr(), cols[v].hi);
            }
        }
    }

    // ── Pass 2: row-wise DCT-32 + quantize ───────────────────────────────────
    // Process 4 groups of 8 rows (row_start = 0, 8, 16, 24).
    //
    // FIX: use load8_i32 (stride=32) instead of load32_i32 to read only the
    // 8 rows in this group.  load8_i32 reads 8 rows × 8 cols at stride=32.
    // Bounds for each segment: max index = row_start*32 + col_offset + 7*32 + 7
    //   = row_start*32 + col_offset + 231
    //   worst case (group=3, col_offset=24): 768 + 24 + 231 = 1023 ✓
    //
    // After load8_i32 + transpose_8x8, seg[u].lane[y] = tmp[(row_start+y)*32 + col_offset+u]
    // for u=0..8, y=0..8, which is what we want for the row DCT.
    let q_ac = vdupq_n_s32(ac_q);
    let q_dc = vsetq_lane_s32::<0>(dc_q, q_ac);

    for group in 0..4usize {
        let row_start = group * 8;
        let base_off = row_start * 32;

        let mut seg_a = load8_i32(&tmp[base_off..], 32); // cols  0- 7
        let mut seg_b = load8_i32(&tmp[base_off + 8..], 32); // cols  8-15
        let mut seg_c = load8_i32(&tmp[base_off + 16..], 32); // cols 16-23
        let mut seg_d = load8_i32(&tmp[base_off + 24..], 32); // cols 24-31

        // Transpose: seg[u].lane[y] = tmp[(row_start+y)*32 + col_offset+u]
        transpose_8x8_i32(&mut seg_a);
        transpose_8x8_i32(&mut seg_b);
        transpose_8x8_i32(&mut seg_c);
        transpose_8x8_i32(&mut seg_d);

        // Reassemble: rows[col_freq].lane[y] = tmp[(row_start+y)*32 + col_freq]
        let mut rows = [I32x8::zero(); 32];
        rows[0..8].copy_from_slice(&seg_a);
        rows[8..16].copy_from_slice(&seg_b);
        rows[16..24].copy_from_slice(&seg_c);
        rows[24..32].copy_from_slice(&seg_d);

        dct1d_32_v_i32(&mut rows);

        // rows[u].lane[y] = DCT32(row row_start+y)[u]
        // Store to input[u * 32 + row_start + y]
        // DC: u==0 and group==0 (row_start==0), lane 0 only
        for u in 0..32usize {
            let (ql, qh) = if u == 0 && group == 0 {
                (mul_q16_vec(rows[u].lo, q_dc), mul_q16_vec(rows[u].hi, q_ac))
            } else {
                (mul_q16_vec(rows[u].lo, q_ac), mul_q16_vec(rows[u].hi, q_ac))
            };
            unsafe {
                let base = &mut input[u * 32 + row_start..];
                vst1q_s32(base.as_mut_ptr(), ql);
                vst1q_s32(base[4..].as_mut_ptr(), qh);
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn dct8x16_neon_i32(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    // Load left half (cols 0-7) and right half (cols 8-15), stride 16.
    let mut left: [I32x8; 8] = std::array::from_fn(|row| {
        let p = &input[row * 16..];
        unsafe {
            I32x8 {
                lo: vld1q_s32(p.as_ptr()),
                hi: vld1q_s32(p[4..].as_ptr()),
            }
        }
    });
    let mut right: [I32x8; 8] = std::array::from_fn(|row| {
        let p = &input[row * 16 + 8..];
        unsafe {
            I32x8 {
                lo: vld1q_s32(p.as_ptr()),
                hi: vld1q_s32(p[4..].as_ptr()),
            }
        }
    });

    transpose_8x8_i32(&mut left);
    transpose_8x8_i32(&mut right);

    let mut c = [I32x8::zero(); 16];
    c[..8].copy_from_slice(&left);
    c[8..16].copy_from_slice(&right);
    dct1d_16_v_i32(&mut c);

    let mut cl: [I32x8; 8] = c[0..8].try_into().unwrap();
    let mut cr: [I32x8; 8] = c[8..16].try_into().unwrap();
    transpose_8x8_i32(&mut cl);
    transpose_8x8_i32(&mut cr);

    dct1d_8_v_i32(&mut cl);
    dct1d_8_v_i32(&mut cr);

    // Quantize helpers.
    let q_ac = vdupq_n_s32(ac_q);
    let q_dc = vsetq_lane_s32::<0>(dc_q, q_ac);

    // Store: output[v*16 + u], v = DCT-8 freq (0..7), u = DCT-16 freq (0..15).
    // For v=0: DC at lane 0 of cl[0].lo.
    for v in 0usize..8 {
        let norm = |x: int32x4_t| x;

        let (ql_lo, ql_hi, qr_lo, qr_hi) = if v == 0 {
            (
                mul_q16_vec(norm(cl[0].lo), q_dc),
                mul_q16_vec(norm(cl[0].hi), q_ac),
                mul_q16_vec(norm(cr[0].lo), q_ac),
                mul_q16_vec(norm(cr[0].hi), q_ac),
            )
        } else {
            (
                mul_q16_vec(norm(cl[v].lo), q_ac),
                mul_q16_vec(norm(cl[v].hi), q_ac),
                mul_q16_vec(norm(cr[v].lo), q_ac),
                mul_q16_vec(norm(cr[v].hi), q_ac),
            )
        };

        unsafe {
            let base = &mut input[v * 16..];
            vst1q_s32(base.as_mut_ptr(), ql_lo);
            vst1q_s32(base[4..].as_mut_ptr(), ql_hi);
            vst1q_s32(base[8..].as_mut_ptr(), qr_lo);
            vst1q_s32(base[12..].as_mut_ptr(), qr_hi);
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon_vs_scalar {
    use super::*;
    use crate::dct::{dct8x8_scalar, dct8x16_i32_scalar, dct16x16_scalar, dct32x32_scalar};
    use crate::neon::{dct8x8_neon_i32, dct8x16_neon_i32, dct16x16_neon_i32, dct32x32_neon_i32};

    // ── helpers ───────────────────────────────────────────────────────────

    /// Simple 32-bit LCG for deterministic pseudo-random inputs.
    /// Returns values in ±511 to stay well within the safe range for WC32[15] ≈ 10×.
    fn lcg(state: &mut u32) -> i32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (((*state >> 16) as i32 & 0x3FF) - 256).min(-2048).max(2048)
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

    // ── dct8x8 ────────────────────────────────────────────────────────────

    fn run_8x8(input: [i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [i32; 64]) {
        let mut scalar = input;
        dct8x8_scalar(&mut scalar, dc_q, ac_q);
        let mut neon = input;
        // SAFETY: aarch64 target guarantees NEON availability at runtime.
        unsafe { dct8x8_neon_i32(&mut neon, dc_q, ac_q) };
        (scalar, neon)
    }

    #[test]
    fn test_8x8_zeros() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8([0i32; 64], dc_q, ac_q);
            assert_eq!(s, n, "8x8 zeros dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x8_constant() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8([64i32; 64], dc_q, ac_q);
            assert_eq!(s, n, "8x8 constant dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x8_ramp() {
        let mut input = [0i32; 64];
        fill_ramp(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8(input, dc_q, ac_q);
            assert_eq!(s, n, "8x8 ramp dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x8_alternating() {
        let mut input = [0i32; 64];
        fill_alt(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8(input, dc_q, ac_q);
            assert_eq!(s, n, "8x8 alternating dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_8x8_random_seed0() {
        let mut input = [0i32; 64];
        fill_lcg(&mut input, 0xDEAD_BEEF);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8(input, dc_q, ac_q);
            let first = s.iter().zip(n.iter()).position(|(a, b)| a != b);
            assert_eq!(
                s, n,
                "8x8 rand(DEADBEEF) dc_q={dc_q} ac_q={ac_q}: mismatch at {first:?}"
            );
        }
    }

    #[test]
    fn test_8x8_random_seed1() {
        let mut input = [0i32; 64];
        fill_lcg(&mut input, 0x1234_5678);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_8x8(input, dc_q, ac_q);
            assert_eq!(s, n, "8x8 rand(12345678) dc_q={dc_q} ac_q={ac_q}");
        }
    }

    // ── dct16x16 ──────────────────────────────────────────────────────────

    fn run_16x16(input: [i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [i32; 256]) {
        let mut scalar = input;
        dct16x16_scalar(&mut scalar, dc_q, ac_q);
        let mut neon = input;
        unsafe { dct16x16_neon_i32(&mut neon, dc_q, ac_q) };
        (scalar, neon)
    }

    #[test]
    fn test_16x16_zeros() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_16x16([0i32; 256], dc_q, ac_q);
            assert_eq!(s, n, "16x16 zeros dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_16x16_constant() {
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_16x16([32i32; 256], dc_q, ac_q);
            assert_eq!(s, n, "16x16 constant dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_16x16_ramp() {
        let mut input = [0i32; 256];
        fill_ramp(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_16x16(input, dc_q, ac_q);
            assert_eq!(s, n, "16x16 ramp dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_16x16_alternating() {
        let mut input = [0i32; 256];
        fill_alt(&mut input);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_16x16(input, dc_q, ac_q);
            assert_eq!(s, n, "16x16 alternating dc_q={dc_q} ac_q={ac_q}");
        }
    }

    #[test]
    fn test_16x16_random_seed1() {
        let mut input = [0i32; 256];
        fill_lcg(&mut input, 0x1234_5678);
        for &(dc_q, ac_q) in QUANT_PAIRS {
            let (s, n) = run_16x16(input, dc_q, ac_q);
            assert_eq!(s, n, "16x16 rand(12345678) dc_q={dc_q} ac_q={ac_q}");
        }
    }

    // ── dct32x32 ──────────────────────────────────────────────────────────

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

    // ── dct8x16 ───────────────────────────────────────────────────────────
    //
    // NEON and scalar apply operations in the same order (DCT-16 on rows,
    // DCT-8 on columns) and produce bit-exact results.

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
