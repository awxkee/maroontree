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
use crate::idct::IdctDequant;
use std::arch::aarch64::*;
use std::mem::MaybeUninit;

/// Eight `i32` lanes, split into a low/high `int32x4_t` pair (same layout as the
/// forward DCT's `I32x8`). Each value is one of the 8 spatial positions of a row
/// of 8 independent 1-D transforms (lane = which transform).
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
    fn muli(self, k: i32) -> I32x4 {
        I32x4(vmulq_s32(self.0, vdupq_n_s32(k)))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn rsh<const SH: i32>(self, add: i32) -> I32x4 {
        I32x4(vshrq_n_s32(vaddq_s32(self.0, vdupq_n_s32(add)), SH))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn clip(self, min: int32x4_t, max: int32x4_t) -> I32x4 {
        I32x4(vminq_s32(vmaxq_s32(self.0, min), max))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn neg(self) -> I32x4 {
        I32x4(vnegq_s32(self.0))
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

    /// Lane-wise multiply by the i32 constant `k` (low 32 bits, like scalar `*`).
    #[inline]
    #[target_feature(enable = "neon")]
    fn muli(self, k: i32) -> I32x8 {
        let c = vdupq_n_s32(k);
        I32x8 {
            lo: vmulq_s32(self.lo, c),
            hi: vmulq_s32(self.hi, c),
        }
    }

    /// `(self + add) >> SH`, rounding bias `add` folded in; arithmetic shift.
    #[inline]
    #[target_feature(enable = "neon")]
    fn rsh<const SH: i32>(self, add: i32) -> I32x8 {
        let a = vdupq_n_s32(add);
        I32x8 {
            lo: vshrq_n_s32(vaddq_s32(self.lo, a), SH),
            hi: vshrq_n_s32(vaddq_s32(self.hi, a), SH),
        }
    }

    /// Clamp every lane to `[min, max]` (matches scalar `clip`).
    #[inline]
    #[target_feature(enable = "neon")]
    fn clip(self, min: int32x4_t, max: int32x4_t) -> I32x8 {
        I32x8 {
            lo: vminq_s32(vmaxq_s32(self.lo, min), max),
            hi: vminq_s32(vmaxq_s32(self.hi, min), max),
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(
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

/// Transpose the 8x8 of `c` (lane <-> vector index), swapping the row and column
/// dimensions between the two passes.
#[inline]
#[target_feature(enable = "neon")]
fn transpose_8x8(c: &mut [I32x8; 8]) {
    let (a0, a1, a2, a3) = transpose_4x4(c[0].lo, c[1].lo, c[2].lo, c[3].lo);
    let (b0, b1, b2, b3) = transpose_4x4(c[0].hi, c[1].hi, c[2].hi, c[3].hi);
    let (e0, e1, e2, e3) = transpose_4x4(c[4].lo, c[5].lo, c[6].lo, c[7].lo);
    let (f0, f1, f2, f3) = transpose_4x4(c[4].hi, c[5].hi, c[6].hi, c[7].hi);
    c[0] = I32x8 { lo: a0, hi: e0 };
    c[1] = I32x8 { lo: a1, hi: e1 };
    c[2] = I32x8 { lo: a2, hi: e2 };
    c[3] = I32x8 { lo: a3, hi: e3 };
    c[4] = I32x8 { lo: b0, hi: f0 };
    c[5] = I32x8 { lo: b1, hi: f1 };
    c[6] = I32x8 { lo: b2, hi: f2 };
    c[7] = I32x8 { lo: b3, hi: f3 };
}

/// Lane-parallel inverse DCT-8 across the 8 vectors of `c`, clipping to
/// `[min, max]` exactly where the scalar `inv_dct8_1d` (and its nested
/// `inv_dct4_1d`) clip. dav1d `inv_dct8_1d_internal_c`, tx64=0.
#[inline]
#[target_feature(enable = "neon")]
fn inv_dct8_v(c: &mut [I32x8; 8], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
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

#[derive(Clone, Copy)]
struct I16x8v(int16x8_t);

#[derive(Clone, Copy)]
struct W32(int32x4_t, int32x4_t);

impl I16x8v {
    #[inline]
    #[target_feature(enable = "neon")]
    fn qadd(self, r: I16x8v) -> I16x8v {
        I16x8v(vqaddq_s16(self.0, r.0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn qsub(self, r: I16x8v) -> I16x8v {
        I16x8v(vqsubq_s16(self.0, r.0))
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn wadd(self, r: I16x8v) -> W32 {
        W32(
            vaddl_s16(vget_low_s16(self.0), vget_low_s16(r.0)),
            vaddl_high_s16(self.0, r.0),
        )
    }
    #[inline]
    #[target_feature(enable = "neon")]
    fn wsub(self, r: I16x8v) -> W32 {
        W32(
            vsubl_s16(vget_low_s16(self.0), vget_low_s16(r.0)),
            vsubl_high_s16(self.0, r.0),
        )
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn wmul_rsh<const SH: i32>(w: W32, k: i32) -> I16x8v {
    let c = vdupq_n_s32(k);
    I16x8v(vcombine_s16(
        vrshrn_n_s32::<SH>(vmulq_s32(w.0, c)),
        vrshrn_n_s32::<SH>(vmulq_s32(w.1, c)),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
fn rot2_rsh<const SH: i32>(a: I16x8v, ka: i16, b: I16x8v, kb: i16) -> I16x8v {
    let lo = vmlal_n_s16(vmull_n_s16(vget_low_s16(a.0), ka), vget_low_s16(b.0), kb);
    let hi = vmlal_high_n_s16(vmull_high_n_s16(a.0, ka), b.0, kb);
    I16x8v(vcombine_s16(vrshrn_n_s32::<SH>(lo), vrshrn_n_s32::<SH>(hi)))
}

macro_rules! inv_dct8_v_x8_s16 {
    ($values:expr) => {{
        let c: &mut [I16x8v; 8] = $values;
        let (e0, e1, e2, e3) = (c[0], c[2], c[4], c[6]);
        let d0 = wmul_rsh::<8>(e0.wadd(e2), 181);
        let d1 = wmul_rsh::<8>(e0.wsub(e2), 181);
        let d2 = rot2_rsh::<12>(e1, 1567, e3, 312).qsub(e3);
        let d3 = rot2_rsh::<12>(e1, -312, e3, 1567).qadd(e1);
        let p0 = d0.qadd(d3);
        let p2 = d1.qadd(d2);
        let p4 = d1.qsub(d2);
        let p6 = d0.qsub(d3);
        let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
        let t4a = rot2_rsh::<12>(in1, 799, in7, 79).qsub(in7);
        let t5a0 = rot2_rsh::<11>(in5, 1703, in3, -1138);
        let t6a0 = rot2_rsh::<11>(in5, 1138, in3, 1703);
        let t7a = rot2_rsh::<12>(in1, -79, in7, 799).qadd(in1);
        let t4 = t4a.qadd(t5a0);
        let t5a = t4a.qsub(t5a0);
        let t7 = t7a.qadd(t6a0);
        let t6a = t7a.qsub(t6a0);
        let t5 = wmul_rsh::<8>(t6a.wsub(t5a), 181);
        let t6 = wmul_rsh::<8>(t6a.wadd(t5a), 181);
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

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct8_v_x4(c: &mut [I32x4; 8], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);

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
#[target_feature(enable = "neon")]
fn inv_dct4_v_x4(c: &mut [I32x4; 4], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);
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
#[target_feature(enable = "neon")]
fn inv_adst4_v_x4(c: &mut [I32x4; 4], _min: i32, _max: i32) {
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
#[target_feature(enable = "neon")]
fn inv_adst8_v_x4(c: &mut [I32x4; 8], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);
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

macro_rules! inv_dct16_v_x8_s16 {
    ($values:expr) => {{
        let c: &mut [I16x8v; 16] = $values;
        let mut e: [I16x8v; 8] = std::array::from_fn(|i| c[2 * i]);
        inv_dct8_v_x8_s16!(&mut e);

        let (in1, in3, in5, in7) = (c[1], c[3], c[5], c[7]);
        let (in9, in11, in13, in15) = (c[9], c[11], c[13], c[15]);

        let t8a = rot2_rsh::<12>(in1, 401, in15, 20).qsub(in15);
        let t9a = rot2_rsh::<11>(in9, 1583, in7, -1299);
        let t10a = rot2_rsh::<12>(in5, 1931, in11, 484).qsub(in11);
        let t11a = rot2_rsh::<12>(in13, -176, in3, -1189).qadd(in13);
        let t12a = rot2_rsh::<12>(in13, 1189, in3, -176).qadd(in3);
        let t13a = rot2_rsh::<12>(in5, -484, in11, 1931).qadd(in5);
        let t14a = rot2_rsh::<11>(in9, 1299, in7, 1583);
        let t15a = rot2_rsh::<12>(in1, -20, in15, 401).qadd(in1);

        let t8 = t8a.qadd(t9a);
        let t9 = t8a.qsub(t9a);
        let t10 = t11a.qsub(t10a);
        let t11 = t11a.qadd(t10a);
        let t12 = t12a.qadd(t13a);
        let t13 = t12a.qsub(t13a);
        let t14 = t15a.qsub(t14a);
        let t15 = t15a.qadd(t14a);

        let u9a = rot2_rsh::<12>(t14, 1567, t9, 312).qsub(t9);
        let u14a = rot2_rsh::<12>(t14, -312, t9, 1567).qadd(t14);
        let u10a = rot2_rsh::<12>(t13, 312, t10, -1567).qsub(t13);
        let u13a = rot2_rsh::<12>(t13, 1567, t10, 312).qsub(t10);

        let v8a = t8.qadd(t11);
        let v9 = u9a.qadd(u10a);
        let v10 = u9a.qsub(u10a);
        let v11a = t8.qsub(t11);
        let v12a = t15.qsub(t12);
        let v13 = u14a.qsub(u13a);
        let v14 = u14a.qadd(u13a);
        let v15a = t15.qadd(t12);

        let w10a = wmul_rsh::<8>(v13.wsub(v10), 181);
        let w13a = wmul_rsh::<8>(v13.wadd(v10), 181);
        let w11 = wmul_rsh::<8>(v12a.wsub(v11a), 181);
        let w12 = wmul_rsh::<8>(v12a.wadd(v11a), 181);

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

/// s16 counterpart of [`inv_dct32_v_x4`], 8 columns per call.
macro_rules! inv_dct32_v_x8_s16 {
    ($values:expr) => {{
        let c: &mut [I16x8v; 32] = $values;
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
    let even: &mut [I16x8v; 16] = even.try_into().unwrap();
    let w: &mut [I16x8v; 16] = w.try_into().unwrap();
    inv_dct16_v_x8_s16!(even);

    w.swap(1, 8);
    w.swap(2, 4);
    w.swap(3, 12);
    w.swap(5, 10);
    w.swap(7, 14);
    w.swap(11, 13);

    // Odd stage 1.
    let (a, b) = (w[0], w[15]);
    w[0] = rot2_rsh::<12>(a, 201, b, 5).qsub(b);
    w[15] = rot2_rsh::<12>(a, -5, b, 201).qadd(a);
    let (a, b) = (w[1], w[14]);
    w[1] = rot2_rsh::<12>(a, -1061, b, -2751).qadd(a);
    w[14] = rot2_rsh::<12>(a, 2751, b, -1061).qadd(b);
    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh::<12>(a, 1751, b, 393).qsub(b);
    w[13] = rot2_rsh::<12>(a, -393, b, 1751).qadd(a);
    let (a, b) = (w[3], w[12]);
    w[3] = rot2_rsh::<12>(a, -239, b, -1380).qadd(a);
    w[12] = rot2_rsh::<12>(a, 1380, b, -239).qadd(b);
    let (a, b) = (w[4], w[11]);
    w[4] = rot2_rsh::<12>(a, 995, b, 123).qsub(b);
    w[11] = rot2_rsh::<12>(a, -123, b, 995).qadd(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh::<12>(a, -583, b, -2106).qadd(a);
    w[10] = rot2_rsh::<12>(a, 2106, b, -583).qadd(b);
    let (a, b) = (w[6], w[9]);
    w[6] = rot2_rsh::<11>(a, 1220, b, -1645);
    w[9] = rot2_rsh::<11>(a, 1645, b, 1220);
    let (a, b) = (w[7], w[8]);
    w[7] = rot2_rsh::<12>(a, -44, b, -601).qadd(a);
    w[8] = rot2_rsh::<12>(a, 601, b, -44).qadd(b);

    // Odd stage 2.
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

    // Odd stage 3 rotations.
    let (a, b) = (w[1], w[14]);
    w[1] = rot2_rsh::<12>(b, 799, a, 79).qsub(a);
    w[14] = rot2_rsh::<12>(b, -79, a, 799).qadd(b);
    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh::<12>(b, 79, a, -799).qsub(b);
    w[13] = rot2_rsh::<12>(b, 799, a, 79).qsub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh::<11>(b, 1703, a, -1138);
    w[10] = rot2_rsh::<11>(b, 1138, a, 1703);
    let (a, b) = (w[6], w[9]);
    w[6] = rot2_rsh::<11>(b, -1138, a, -1703);
    w[9] = rot2_rsh::<11>(b, 1703, a, -1138);

    // Odd stage 4.
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

    // Odd stage 5 rotations.
    let (a, b) = (w[2], w[13]);
    w[2] = rot2_rsh::<12>(b, 1567, a, 312).qsub(a);
    w[13] = rot2_rsh::<12>(b, -312, a, 1567).qadd(b);
    let (a, b) = (w[3], w[12]);
    w[3] = rot2_rsh::<12>(b, 1567, a, 312).qsub(a);
    w[12] = rot2_rsh::<12>(b, -312, a, 1567).qadd(b);
    let (a, b) = (w[4], w[11]);
    w[4] = rot2_rsh::<12>(b, 312, a, -1567).qsub(b);
    w[11] = rot2_rsh::<12>(b, 1567, a, 312).qsub(a);
    let (a, b) = (w[5], w[10]);
    w[5] = rot2_rsh::<12>(b, 312, a, -1567).qsub(b);
    w[10] = rot2_rsh::<12>(b, 1567, a, 312).qsub(a);

    // Odd stage 6.
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

    // Odd stage 7.
    for &(i, j) in &[(4usize, 11usize), (5, 10), (6, 9), (7, 8)] {
        let (a, b) = (w[i], w[j]);
        w[i] = wmul_rsh::<8>(b.wsub(a), 181);
        w[j] = wmul_rsh::<8>(b.wadd(a), 181);
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
#[target_feature(enable = "neon")]
fn inv_dct16_v_x4(c: &mut [I32x4; 16], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);

    let mut e: [I32x4; 8] = std::array::from_fn(|i| c[2 * i]);
    inv_dct8_v_x4(&mut e, min, max);

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
#[target_feature(enable = "neon")]
fn inv_adst16_v_x4(c: &mut [I32x4; 16], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);

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
fn transpose_store_4x4(dst: *mut i32, stride: usize, tile: &mut [I32x4; 4]) {
    let (r0, r1, r2, r3) = transpose_4x4(tile[0].0, tile[1].0, tile[2].0, tile[3].0);
    unsafe {
        vst1q_s32(dst, r0);
        vst1q_s32(dst.add(stride), r1);
        vst1q_s32(dst.add(2 * stride), r2);
        vst1q_s32(dst.add(3 * stride), r3);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_transposed_rows_i32x4<const N: usize>(dst: *mut i32, y: usize, rows: &[I32x4; N]) {
    debug_assert!(N.is_multiple_of(4));
    let stride = N;
    let mut x = 0usize;
    while x < N {
        let mut tile = [rows[x], rows[x + 1], rows[x + 2], rows[x + 3]];
        transpose_store_4x4(unsafe { dst.add(y * N + x) }, stride, &mut tile);
        x += 4;
    }
}

/// Dequant 4 levels: `coeff = sign(lvl) * min((|lvl|*q) & 0xff_ffff, cf_max + (lvl<0))`.
/// `lvl==0` falls out to 0 naturally. Matches the scalar dequant loop bit-for-bit.
#[inline]
#[target_feature(enable = "neon")]
fn dequant4<const DQ1: bool>(lvl: int32x4_t, q: int32x4_t, cf_max: int32x4_t) -> int32x4_t {
    let absl = vabsq_s32(lvl);
    // Only the low 24 product bits are observable. `vmulq_s32` therefore
    // matches the scalar u64 product exactly and avoids two widening multiplies.
    let masked = vandq_s32(vmulq_s32(absl, q), vdupq_n_s32(0x00ff_ffff));
    let masked = if DQ1 {
        vshrq_n_s32::<1>(masked)
    } else {
        masked
    };
    let sign = vshrq_n_s32::<31>(lvl);
    let cap = vaddq_s32(cf_max, vandq_s32(sign, vdupq_n_s32(1)));
    let mag = vminq_s32(masked, cap);
    vsubq_s32(veorq_s32(mag, sign), sign)
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant_q4<const QM: bool>(dequant: &IdctDequant, rc: usize) -> int32x4_t {
    let ac = vdupq_n_s32(dequant.ac_q);
    let base = if rc == 0 {
        vsetq_lane_s32(dequant.dc_q, ac, 0)
    } else {
        ac
    };
    if !QM {
        return base;
    }
    let qm = dequant.qm.expect("QM dequant path requires a matrix");
    debug_assert!(rc + 4 <= qm.len());
    let weights16 = unsafe {
        vmovl_u8(vreinterpret_u8_u32(vld1_lane_u32::<0>(
            qm.as_ptr().add(rc).cast::<u32>(),
            vdup_n_u32(0),
        )))
    };
    let weights = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(weights16)));
    vshrq_n_s32::<5>(vaddq_s32(vmulq_s32(base, weights), vdupq_n_s32(16)))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_dequant16_i32x4<const QM: bool>(
    levels: &[i32; 256],
    x: usize,
    y: usize,
    dequant: &IdctDequant,
) -> I32x4 {
    let rc = x * 16 + y;
    let lvl = unsafe { vld1q_s32(levels.as_ptr().add(rc)) };
    I32x4(dequant4::<false>(
        lvl,
        dequant_q4::<QM>(dequant, rc),
        vdupq_n_s32(dequant.cf_max),
    ))
}

#[target_feature(enable = "neon")]
fn inv16x16_mixed_dequant_neon<const ROW_ADST: bool, const COL_ADST: bool>(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    if dequant.qm.is_some() {
        inv16x16_mixed_dequant_neon_impl::<ROW_ADST, COL_ADST, true>(levels, dequant)
    } else {
        inv16x16_mixed_dequant_neon_impl::<ROW_ADST, COL_ADST, false>(levels, dequant)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inv16x16_mixed_dequant_neon_impl<const ROW_ADST: bool, const COL_ADST: bool, const QM: bool>(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    let cmn = vdupq_n_s32(dequant.cmin);
    let cmx = vdupq_n_s32(dequant.cmax);

    // Fused dequant + first inverse dimension. Store a real transposed scratch:
    // scratch[y_frequency * 16 + x_spatial].
    let mut scratch_u = MaybeUninit::<[i32; 256]>::uninit();
    for y in (0..16usize).step_by(4) {
        let mut rows: [I32x4; 16] =
            std::array::from_fn(|x| load_dequant16_i32x4::<QM>(levels, x, y, dequant));
        if ROW_ADST {
            inv_adst16_v_x4(&mut rows, dequant.rmin, dequant.rmax);
        } else {
            inv_dct16_v_x4(&mut rows, dequant.rmin, dequant.rmax);
        }
        for row in rows.iter_mut() {
            *row = row.rsh::<2>(2).clip(cmn, cmx);
        }
        store_transposed_rows_i32x4::<16>(scratch_u.as_mut_ptr().cast(), y, &rows);
    }
    let scratch = unsafe { scratch_u.assume_init() };

    let mut out = MaybeUninit::<[i32; 256]>::uninit();
    for x in (0..16usize).step_by(4) {
        let mut cols: [I32x4; 16] =
            std::array::from_fn(|y| load_i32x4(unsafe { scratch.as_ptr().add(y * 16 + x) }));
        if COL_ADST {
            inv_adst16_v_x4(&mut cols, dequant.cmin, dequant.cmax);
        } else {
            inv_dct16_v_x4(&mut cols, dequant.cmin, dequant.cmax);
        }
        for y in 0..16usize {
            let r = cols[y].rsh::<4>(8);
            unsafe { store_i32x4((out.as_mut_ptr() as *mut i32).add(y * 16 + x), r) };
        }
    }
    unsafe { out.assume_init() }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_8x8_s16(v: &mut [I16x8v; 8]) {
    let mut a: [int16x8_t; 8] = std::array::from_fn(|i| v[i].0);
    // stage 1: 16-bit
    let mut b: [int16x8_t; 8] = [a[0]; 8];
    for i in 0..4 {
        b[2 * i] = vtrn1q_s16(a[2 * i], a[2 * i + 1]);
        b[2 * i + 1] = vtrn2q_s16(a[2 * i], a[2 * i + 1]);
    }
    // stage 2: 32-bit
    for i in 0..2 {
        let (o, q) = (4 * i, 4 * i);
        a[q] = vreinterpretq_s16_s32(vtrn1q_s32(
            vreinterpretq_s32_s16(b[o]),
            vreinterpretq_s32_s16(b[o + 2]),
        ));
        a[q + 2] = vreinterpretq_s16_s32(vtrn2q_s32(
            vreinterpretq_s32_s16(b[o]),
            vreinterpretq_s32_s16(b[o + 2]),
        ));
        a[q + 1] = vreinterpretq_s16_s32(vtrn1q_s32(
            vreinterpretq_s32_s16(b[o + 1]),
            vreinterpretq_s32_s16(b[o + 3]),
        ));
        a[q + 3] = vreinterpretq_s16_s32(vtrn2q_s32(
            vreinterpretq_s32_s16(b[o + 1]),
            vreinterpretq_s32_s16(b[o + 3]),
        ));
    }
    // stage 3: 64-bit
    for i in 0..4 {
        v[i] = I16x8v(vreinterpretq_s16_s64(vtrn1q_s64(
            vreinterpretq_s64_s16(a[i]),
            vreinterpretq_s64_s16(a[i + 4]),
        )));
        v[i + 4] = I16x8v(vreinterpretq_s16_s64(vtrn2q_s64(
            vreinterpretq_s64_s16(a[i]),
            vreinterpretq_s64_s16(a[i + 4]),
        )));
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn dequant_levels_s16_neon<const DQ1: bool>(
    levels: &[i32],
    coeff: &mut [MaybeUninit<i16>],
    dequant: &IdctDequant,
) {
    if dequant.qm.is_some() {
        dequant_levels_s16_neon_impl::<DQ1, true>(levels, coeff, dequant);
    } else {
        dequant_levels_s16_neon_impl::<DQ1, false>(levels, coeff, dequant);
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn dequant_levels_s16_neon_impl<const DQ1: bool, const QM: bool>(
    levels: &[i32],
    coeff: &mut [MaybeUninit<i16>],
    dequant: &IdctDequant,
) {
    debug_assert_eq!(dequant.cf_max, i16::MAX as i32);
    debug_assert_eq!(levels.len(), coeff.len());
    let cfm = vdupq_n_s32(dequant.cf_max);
    let (level_chunks, level_tail) = levels.as_chunks::<8>();
    let (coeff_chunks, coeff_tail) = coeff.as_chunks_mut::<8>();
    debug_assert!(level_tail.is_empty());
    debug_assert!(coeff_tail.is_empty());
    for (chunk_index, (level, dst)) in level_chunks.iter().zip(coeff_chunks.iter_mut()).enumerate()
    {
        let rc = chunk_index * 8;
        let lo = unsafe { vld1q_s32(level[..4].as_ptr()) };
        let hi = unsafe { vld1q_s32(level[4..].as_ptr()) };
        let lo = dequant4::<DQ1>(lo, dequant_q4::<QM>(dequant, rc), cfm);
        let hi = dequant4::<DQ1>(hi, dequant_q4::<QM>(dequant, rc + 4), cfm);
        unsafe {
            vst1q_s16(
                dst.as_mut_ptr().cast(),
                vcombine_s16(vmovn_s32(lo), vmovn_s32(hi)),
            )
        };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn prescale_s16(v: I16x8v) -> I16x8v {
    I16x8v(vcombine_s16(
        vrshrn_n_s32::<8>(vmull_n_s16(vget_low_s16(v.0), 181)),
        vrshrn_n_s32::<8>(vmull_high_n_s16(v.0, 181)),
    ))
}

#[inline]
#[target_feature(enable = "neon")]
fn mid_shift_s16(v: I16x8v, shift: i32) -> I16x8v {
    match shift {
        1 => I16x8v(vrshrq_n_s16::<1>(v.0)),
        2 => I16x8v(vrshrq_n_s16::<2>(v.0)),
        _ => unreachable!("unsupported s16 inverse mid-shift"),
    }
}

/// One 8-row strip of an 8-wide s16 row pass, transposed into the i16 scratch.
/// The caller supplies the coefficient stride so this also covers 8x16.
#[inline(never)]
#[target_feature(enable = "neon")]
fn first_pass8_s16(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    let mut v: [I16x8v; 8] = std::array::from_fn(|x| {
        let value = I16x8v(unsafe { vld1q_s16(coeff.add(x * h + y0)) });
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct8_v_x8_s16!(&mut v);
    let mut tile = v.map(|value| mid_shift_s16(value, mid_shift));
    transpose_8x8_s16(&mut tile);
    for (i, value) in tile.iter().enumerate() {
        unsafe { vst1q_s16(scratch.add((y0 + i) * 8), value.0) };
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn first_pass32_s16(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    let mut v: [I16x8v; 32] = std::array::from_fn(|x| {
        let value = I16x8v(unsafe { vld1q_s16(coeff.add(x * h + y0)) });
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct32_v_x8_s16!(&mut v);
    for x0 in (0..32).step_by(8) {
        let mut tile: [I16x8v; 8] = std::array::from_fn(|i| mid_shift_s16(v[x0 + i], mid_shift));
        transpose_8x8_s16(&mut tile);
        for (i, t) in tile.iter().enumerate() {
            unsafe { vst1q_s16(scratch.add((y0 + i) * 32 + x0), t.0) };
        }
    }
}

/// One 8-column strip of a 32-high s16 column pass, widening to i32 on store.
/// `#[inline(never)]` for the same reason as [`first_pass32_s16`].
#[inline(never)]
#[target_feature(enable = "neon")]
fn second_pass32_s16(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let mut v: [I16x8v; 32] =
        std::array::from_fn(|y| I16x8v(unsafe { vld1q_s16(scratch.add(y * w + x0)) }));
    inv_dct32_v_x8_s16!(&mut v);
    for y in 0..32 {
        let r = vrshrq_n_s16::<4>(v[y].0);
        unsafe {
            vst1q_s32(out.add(y * w + x0), vmovl_s16(vget_low_s16(r)));
            vst1q_s32(out.add(y * w + x0 + 4), vmovl_high_s16(r));
        }
    }
}

/// One 8-row strip of a 16-wide s16 row pass. The dynamic coefficient stride
/// shares this kernel between 16x8, 16x16, and 16x32.
#[inline(never)]
#[target_feature(enable = "neon")]
fn first_pass16_s16(
    coeff: *const i16,
    scratch: *mut i16,
    y0: usize,
    h: usize,
    prescale: bool,
    mid_shift: i32,
) {
    let mut v: [I16x8v; 16] = std::array::from_fn(|x| {
        let value = I16x8v(unsafe { vld1q_s16(coeff.add(x * h + y0)) });
        if prescale { prescale_s16(value) } else { value }
    });
    inv_dct16_v_x8_s16!(&mut v);
    for x0 in (0..16).step_by(8) {
        let mut tile: [I16x8v; 8] = std::array::from_fn(|i| mid_shift_s16(v[x0 + i], mid_shift));
        transpose_8x8_s16(&mut tile);
        for (i, t) in tile.iter().enumerate() {
            unsafe { vst1q_s16(scratch.add((y0 + i) * 16 + x0), t.0) };
        }
    }
}

/// One 8-column strip of a 16-high s16 column pass, widening to i32 on store.
#[inline(never)]
#[target_feature(enable = "neon")]
fn second_pass16_s16(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let mut v: [I16x8v; 16] =
        std::array::from_fn(|y| I16x8v(unsafe { vld1q_s16(scratch.add(y * w + x0)) }));
    inv_dct16_v_x8_s16!(&mut v);
    for y in 0..16 {
        let r = vrshrq_n_s16::<4>(v[y].0);
        unsafe {
            vst1q_s32(out.add(y * w + x0), vmovl_s16(vget_low_s16(r)));
            vst1q_s32(out.add(y * w + x0 + 4), vmovl_high_s16(r));
        }
    }
}

#[inline(never)]
#[target_feature(enable = "neon")]
fn second_pass8_s16(scratch: *const i16, out: *mut i32, x0: usize, w: usize) {
    let mut v: [I16x8v; 8] =
        std::array::from_fn(|y| I16x8v(unsafe { vld1q_s16(scratch.add(y * w + x0)) }));
    inv_dct8_v_x8_s16!(&mut v);
    for (y, vv) in v.iter().enumerate() {
        let r = vrshrq_n_s16::<4>(vv.0);
        unsafe {
            vst1q_s32(out.add(y * w + x0), vmovl_s16(vget_low_s16(r)));
            vst1q_s32(out.add(y * w + x0 + 4), vmovl_high_s16(r));
        }
    }
}

/// Pure-DCT inverse for the 8-bit clip regime. Dequantization and both transform
/// passes stay in i16; the output is widened once after the final shift.
#[target_feature(enable = "neon")]
fn inverse_dct_s16<
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
    dequant_levels_s16_neon::<DQ1>(levels, &mut coeff, dequant);

    let mut scratch_u = MaybeUninit::<[i16; N]>::uninit();
    let scratch = scratch_u.as_mut_ptr() as *mut i16;
    for y0 in (0..H).step_by(8) {
        match W {
            8 => first_pass8_s16(coeff.as_ptr().cast(), scratch, y0, H, PRESCALE, MID_SHIFT),
            16 => first_pass16_s16(coeff.as_ptr().cast(), scratch, y0, H, PRESCALE, MID_SHIFT),
            32 => first_pass32_s16(coeff.as_ptr().cast(), scratch, y0, H, PRESCALE, MID_SHIFT),
            _ => unreachable!("unsupported s16 inverse row length"),
        }
    }

    let mut out_u = MaybeUninit::<[i32; N]>::uninit();
    let out = out_u.as_mut_ptr() as *mut i32;
    for x0 in (0..W).step_by(8) {
        match H {
            8 => second_pass8_s16(scratch, out, x0, W),
            16 => second_pass16_s16(scratch, out, x0, W),
            32 => second_pass32_s16(scratch, out, x0, W),
            _ => unreachable!("unsupported s16 inverse column length"),
        }
    }
    unsafe { out_u.assume_init() }
}

#[inline]
fn can_use_s16_inverse(dequant: &IdctDequant) -> bool {
    dequant.cf_max == i16::MAX as i32
        && dequant.rmin == i16::MIN as i32
        && dequant.rmax == i16::MAX as i32
        && dequant.cmin == i16::MIN as i32
        && dequant.cmax == i16::MAX as i32
}

const INV_DCT: u8 = 0;
const INV_ADST: u8 = 1;
const INV_IDENTITY: u8 = 2;

#[inline]
#[target_feature(enable = "neon")]
fn identity_v(v: I32x4, len: usize) -> I32x4 {
    match len {
        4 => v.add(v.muli(1697).rsh::<12>(2048)),
        8 => v.add(v),
        16 => v.add(v).add(v.muli(1697).rsh::<11>(1024)),
        _ => unreachable!("unsupported inverse identity length"),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn apply4<const KIND: u8>(c: &mut [I32x4; 4], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct4_v_x4(c, min, max),
        INV_ADST => inv_adst4_v_x4(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 4)),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn apply8<const KIND: u8>(c: &mut [I32x4; 8], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct8_v_x4(c, min, max),
        INV_ADST => inv_adst8_v_x4(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 8)),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn apply16<const KIND: u8>(c: &mut [I32x4; 16], min: i32, max: i32) {
    match KIND {
        INV_DCT => inv_dct16_v_x4(c, min, max),
        INV_ADST => inv_adst16_v_x4(c, min, max),
        INV_IDENTITY => c.iter_mut().for_each(|v| *v = identity_v(*v, 16)),
        _ => unreachable!(),
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct32_v_x4(c: &mut [I32x4; 32], min: i32, max: i32) {
    // In-place inverse perfect shuffle: interleaved coefficients -> packed
    // even and odd halves.
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
    let even: &mut [I32x4; 16] = even.try_into().unwrap();
    let w: &mut [I32x4; 16] = w.try_into().unwrap();
    inv_dct16_v_x4(even, min, max);
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
    let clip = |v: I32x4| v.clip(mn, mx);

    // The odd input permutation is four-bit reversal. Once permuted, every
    // stage-1 rotation overwrites only the pair it consumed.
    w.swap(1, 8);
    w.swap(2, 4);
    w.swap(3, 12);
    w.swap(5, 10);
    w.swap(7, 14);
    w.swap(11, 13);

    // Odd stage 1.
    let (a, b) = (w[0], w[15]);
    w[0] = a.muli(201).sub(b.muli(-5)).rsh::<12>(2048).sub(b);
    w[15] = a.muli(-5).add(b.muli(201)).rsh::<12>(2048).add(a);
    let (a, b) = (w[1], w[14]);
    w[1] = a.muli(-1061).sub(b.muli(2751)).rsh::<12>(2048).add(a);
    w[14] = a.muli(2751).add(b.muli(-1061)).rsh::<12>(2048).add(b);
    let (a, b) = (w[2], w[13]);
    w[2] = a.muli(1751).sub(b.muli(-393)).rsh::<12>(2048).sub(b);
    w[13] = a.muli(-393).add(b.muli(1751)).rsh::<12>(2048).add(a);
    let (a, b) = (w[3], w[12]);
    w[3] = a.muli(-239).sub(b.muli(1380)).rsh::<12>(2048).add(a);
    w[12] = a.muli(1380).add(b.muli(-239)).rsh::<12>(2048).add(b);
    let (a, b) = (w[4], w[11]);
    w[4] = a.muli(995).sub(b.muli(-123)).rsh::<12>(2048).sub(b);
    w[11] = a.muli(-123).add(b.muli(995)).rsh::<12>(2048).add(a);
    let (a, b) = (w[5], w[10]);
    w[5] = a.muli(-583).sub(b.muli(2106)).rsh::<12>(2048).add(a);
    w[10] = a.muli(2106).add(b.muli(-583)).rsh::<12>(2048).add(b);
    let (a, b) = (w[6], w[9]);
    w[6] = a.muli(1220).sub(b.muli(1645)).rsh::<11>(1024);
    w[9] = a.muli(1645).add(b.muli(1220)).rsh::<11>(1024);
    let (a, b) = (w[7], w[8]);
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
        let e = even[i];
        let o = w[15 - i];
        even[i] = clip(e.add(o));
        w[15 - i] = clip(e.sub(o));
    }
}

#[target_feature(enable = "neon")]
fn dequant_levels_neon<const N: usize, const DQ1: bool>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    if dequant.qm.is_some() {
        dequant_levels_neon_impl::<N, DQ1, true>(levels, dequant)
    } else {
        dequant_levels_neon_impl::<N, DQ1, false>(levels, dequant)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dequant_levels_neon_impl<const N: usize, const DQ1: bool, const QM: bool>(
    levels: &[i32; N],
    dequant: &IdctDequant,
) -> [i32; N] {
    let mut coeff = MaybeUninit::<[i32; N]>::uninit();
    let cfm = vdupq_n_s32(dequant.cf_max);
    let (level_chunks, level_tail) = levels.as_chunks::<4>();
    debug_assert!(level_tail.is_empty());
    for (chunk_index, level) in level_chunks.iter().enumerate() {
        let rc = chunk_index * 4;
        let level = unsafe { vld1q_s32(level.as_ptr()) };
        let value = dequant4::<DQ1>(level, dequant_q4::<QM>(dequant, rc), cfm);
        unsafe { vst1q_s32((coeff.as_mut_ptr() as *mut i32).add(rc), value) };
    }
    unsafe { coeff.assume_init() }
}

macro_rules! define_inverse_first_pass_neon {
    ($name:ident, $len:literal, $apply:ident) => {
        #[inline(never)]
        #[target_feature(enable = "neon")]
        fn $name<const ROW: u8, const PRESCALE: bool, const MID_SHIFT: i32>(
            coeff: *const i32,
            scratch: *mut i32,
            y0: usize,
            dequant: &IdctDequant,
            w: usize,
            h: usize,
        ) {
            let mut v: [I32x4; $len] = std::array::from_fn(|x| {
                let mut z = I32x4(unsafe { vld1q_s32(coeff.add(x * h + y0)) });
                if PRESCALE {
                    z = z.muli(181).rsh::<8>(128);
                }
                z
            });
            $apply::<ROW>(&mut v, dequant.rmin, dequant.rmax);
            let cmn = vdupq_n_s32(dequant.cmin);
            let cmx = vdupq_n_s32(dequant.cmax);
            for x in (0..$len).step_by(4) {
                let mut tile: [I32x4; 4] = std::array::from_fn(|i| {
                    let z = v[x + i];
                    match MID_SHIFT {
                        0 => z.clip(cmn, cmx),
                        1 => z.rsh::<1>(1).clip(cmn, cmx),
                        2 => z.rsh::<2>(2).clip(cmn, cmx),
                        _ => unreachable!(),
                    }
                });
                transpose_store_4x4(unsafe { scratch.add(y0 * w + x) }, w, &mut tile);
            }
        }
    };
}

define_inverse_first_pass_neon!(inverse_first_pass4_neon, 4, apply4);
define_inverse_first_pass_neon!(inverse_first_pass8_neon, 8, apply8);
define_inverse_first_pass_neon!(inverse_first_pass16_neon, 16, apply16);

#[inline(never)]
#[target_feature(enable = "neon")]
fn inverse_first_pass32_neon<const PRESCALE: bool, const MID_SHIFT: i32>(
    coeff: *const i32,
    scratch: *mut i32,
    y0: usize,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    debug_assert_eq!(w, 32);
    let mut v: [I32x4; 32] = std::array::from_fn(|x| {
        let mut z = I32x4(unsafe { vld1q_s32(coeff.add(x * h + y0)) });
        if PRESCALE {
            z = z.muli(181).rsh::<8>(128);
        }
        z
    });
    inv_dct32_v_x4(&mut v, dequant.rmin, dequant.rmax);
    let cmn = vdupq_n_s32(dequant.cmin);
    let cmx = vdupq_n_s32(dequant.cmax);
    for x in (0..32).step_by(4) {
        let mut tile: [I32x4; 4] = std::array::from_fn(|i| {
            let z = v[x + i];
            match MID_SHIFT {
                0 => z.clip(cmn, cmx),
                1 => z.rsh::<1>(1).clip(cmn, cmx),
                2 => z.rsh::<2>(2).clip(cmn, cmx),
                _ => unreachable!(),
            }
        });
        transpose_store_4x4(unsafe { scratch.add(y0 * w + x) }, w, &mut tile);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inverse_first_pass_neon<const ROW: u8, const PRESCALE: bool, const MID_SHIFT: i32>(
    coeff: *const i32,
    scratch: *mut i32,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    for y0 in (0..h).step_by(4) {
        match w {
            4 => inverse_first_pass4_neon::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, dequant, w, h,
            ),
            8 => inverse_first_pass8_neon::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, dequant, w, h,
            ),
            16 => inverse_first_pass16_neon::<ROW, PRESCALE, MID_SHIFT>(
                coeff, scratch, y0, dequant, w, h,
            ),
            32 => {
                debug_assert_eq!(ROW, INV_DCT);
                inverse_first_pass32_neon::<PRESCALE, MID_SHIFT>(coeff, scratch, y0, dequant, w, h);
            }
            _ => unreachable!(),
        }
    }
}

macro_rules! define_inverse_second_pass_neon {
    ($name:ident, $len:literal, $apply:ident) => {
        #[inline(never)]
        #[target_feature(enable = "neon")]
        fn $name<const COL: u8>(
            scratch: *const i32,
            out: *mut i32,
            x0: usize,
            dequant: &IdctDequant,
            w: usize,
        ) {
            let mut v: [I32x4; $len] =
                std::array::from_fn(|y| I32x4(unsafe { vld1q_s32(scratch.add(y * w + x0)) }));
            $apply::<COL>(&mut v, dequant.cmin, dequant.cmax);
            for y in 0..$len {
                let r = v[y].rsh::<4>(8);
                unsafe { vst1q_s32(out.add(y * w + x0), r.0) };
            }
        }
    };
}

define_inverse_second_pass_neon!(inverse_second_pass4_neon, 4, apply4);
define_inverse_second_pass_neon!(inverse_second_pass8_neon, 8, apply8);
define_inverse_second_pass_neon!(inverse_second_pass16_neon, 16, apply16);

#[inline(never)]
#[target_feature(enable = "neon")]
fn inverse_second_pass32_neon(
    scratch: *const i32,
    out: *mut i32,
    x0: usize,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    debug_assert_eq!(h, 32);
    let mut v: [I32x4; 32] =
        std::array::from_fn(|y| I32x4(unsafe { vld1q_s32(scratch.add(y * w + x0)) }));
    inv_dct32_v_x4(&mut v, dequant.cmin, dequant.cmax);
    for y in 0..32 {
        let r = v[y].rsh::<4>(8);
        unsafe { vst1q_s32(out.add(y * w + x0), r.0) };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inverse_second_pass_neon<const COL: u8>(
    scratch: *const i32,
    out: *mut i32,
    dequant: &IdctDequant,
    w: usize,
    h: usize,
) {
    for x0 in (0..w).step_by(4) {
        match h {
            4 => inverse_second_pass4_neon::<COL>(scratch, out, x0, dequant, w),
            8 => inverse_second_pass8_neon::<COL>(scratch, out, x0, dequant, w),
            16 => inverse_second_pass16_neon::<COL>(scratch, out, x0, dequant, w),
            32 => {
                debug_assert_eq!(COL, INV_DCT);
                inverse_second_pass32_neon(scratch, out, x0, dequant, w, h);
            }
            _ => unreachable!(),
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn inverse_neon<
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
    let coeff = dequant_levels_neon::<N, DQ1>(levels, dequant);
    let mut scratch = MaybeUninit::<[i32; N]>::uninit();
    inverse_first_pass_neon::<ROW, PRESCALE, MID_SHIFT>(
        coeff.as_ptr(),
        scratch.as_mut_ptr().cast(),
        dequant,
        W,
        H,
    );
    let mut out = MaybeUninit::<[i32; N]>::uninit();
    inverse_second_pass_neon::<COL>(
        scratch.as_ptr().cast(),
        out.as_mut_ptr().cast(),
        dequant,
        W,
        H,
    );
    unsafe { out.assume_init() }
}

macro_rules! inverse_neon_entry {
    ($name:ident, $n:literal, $w:literal, $h:literal, $row:expr, $col:expr, $pre:expr, $mid:literal, $dq1:expr) => {
        #[target_feature(enable = "neon")]
        pub(crate) fn $name(levels: &[i32; $n], dequant: &IdctDequant) -> [i32; $n] {
            inverse_neon::<$n, $w, $h, $row, $col, $pre, $mid, $dq1>(levels, dequant)
        }
    };
}

macro_rules! inverse_dct_s16_neon_entry {
    (
        $name:ident, $n:literal, $w:literal, $h:literal,
        $pre:literal, $mid:literal, $dq1:literal
    ) => {
        #[target_feature(enable = "neon")]
        pub(crate) fn $name(levels: &[i32; $n], dequant: &IdctDequant) -> [i32; $n] {
            if can_use_s16_inverse(dequant) {
                return inverse_dct_s16::<$n, $w, $h, $dq1, $pre, $mid>(levels, dequant);
            }
            inverse_neon::<$n, $w, $h, INV_DCT, INV_DCT, $pre, $mid, $dq1>(levels, dequant)
        }
    };
}

inverse_neon_entry!(
    idct_dequant_4x4_neon,
    16,
    4,
    4,
    INV_DCT,
    INV_DCT,
    false,
    0,
    false
);
inverse_neon_entry!(
    idct_dequant_4x8_neon,
    32,
    4,
    8,
    INV_DCT,
    INV_DCT,
    true,
    0,
    false
);
inverse_neon_entry!(
    idct_dequant_8x4_neon,
    32,
    8,
    4,
    INV_DCT,
    INV_DCT,
    true,
    0,
    false
);
inverse_neon_entry!(
    idct_dequant_4x16_neon,
    64,
    4,
    16,
    INV_DCT,
    INV_DCT,
    false,
    1,
    false
);
inverse_neon_entry!(
    idct_dequant_16x4_neon,
    64,
    16,
    4,
    INV_DCT,
    INV_DCT,
    false,
    1,
    false
);
inverse_dct_s16_neon_entry!(idct_dequant_8x16_neon, 128, 8, 16, true, 1, false);
inverse_dct_s16_neon_entry!(idct_dequant_16x8_neon, 128, 16, 8, true, 1, false);
inverse_dct_s16_neon_entry!(idct_dequant_16x32_neon, 512, 16, 32, true, 1, true);
inverse_dct_s16_neon_entry!(idct_dequant_32x16_neon, 512, 32, 16, true, 1, true);

inverse_neon_entry!(
    iadst_dequant_4x4_neon,
    16,
    4,
    4,
    INV_ADST,
    INV_ADST,
    false,
    0,
    false
);
inverse_neon_entry!(
    iadstdct_dequant_4x4_neon,
    16,
    4,
    4,
    INV_DCT,
    INV_ADST,
    false,
    0,
    false
);
inverse_neon_entry!(
    idctadst_dequant_4x4_neon,
    16,
    4,
    4,
    INV_ADST,
    INV_DCT,
    false,
    0,
    false
);
inverse_neon_entry!(
    iadst_dequant_4x8_neon,
    32,
    4,
    8,
    INV_ADST,
    INV_ADST,
    true,
    0,
    false
);
inverse_neon_entry!(
    iadstdct_dequant_4x8_neon,
    32,
    4,
    8,
    INV_DCT,
    INV_ADST,
    true,
    0,
    false
);
inverse_neon_entry!(
    idctadst_dequant_4x8_neon,
    32,
    4,
    8,
    INV_ADST,
    INV_DCT,
    true,
    0,
    false
);
inverse_neon_entry!(
    iadst_dequant_8x8_neon,
    64,
    8,
    8,
    INV_ADST,
    INV_ADST,
    false,
    1,
    false
);
inverse_neon_entry!(
    iadstdct_dequant_8x8_neon,
    64,
    8,
    8,
    INV_DCT,
    INV_ADST,
    false,
    1,
    false
);
inverse_neon_entry!(
    idctadst_dequant_8x8_neon,
    64,
    8,
    8,
    INV_ADST,
    INV_DCT,
    false,
    1,
    false
);
inverse_neon_entry!(
    iadst_dequant_8x16_neon,
    128,
    8,
    16,
    INV_ADST,
    INV_ADST,
    true,
    1,
    false
);
inverse_neon_entry!(
    iadstdct_dequant_8x16_neon,
    128,
    8,
    16,
    INV_DCT,
    INV_ADST,
    true,
    1,
    false
);
inverse_neon_entry!(
    idctadst_dequant_8x16_neon,
    128,
    8,
    16,
    INV_ADST,
    INV_DCT,
    true,
    1,
    false
);
inverse_neon_entry!(
    iadst_dequant_16x8_neon,
    128,
    16,
    8,
    INV_ADST,
    INV_ADST,
    true,
    1,
    false
);
inverse_neon_entry!(
    iadstdct_dequant_16x8_neon,
    128,
    16,
    8,
    INV_DCT,
    INV_ADST,
    true,
    1,
    false
);
inverse_neon_entry!(
    idctadst_dequant_16x8_neon,
    128,
    16,
    8,
    INV_ADST,
    INV_DCT,
    true,
    1,
    false
);

inverse_neon_entry!(
    ivdct_dequant_4x4_neon,
    16,
    4,
    4,
    INV_IDENTITY,
    INV_DCT,
    false,
    0,
    false
);
inverse_neon_entry!(
    ihdct_dequant_4x4_neon,
    16,
    4,
    4,
    INV_DCT,
    INV_IDENTITY,
    false,
    0,
    false
);
inverse_neon_entry!(
    ivdct_dequant_8x8_neon,
    64,
    8,
    8,
    INV_IDENTITY,
    INV_DCT,
    false,
    1,
    false
);
inverse_neon_entry!(
    ihdct_dequant_8x8_neon,
    64,
    8,
    8,
    INV_DCT,
    INV_IDENTITY,
    false,
    1,
    false
);
inverse_neon_entry!(
    ivdct_dequant_8x16_neon,
    128,
    8,
    16,
    INV_IDENTITY,
    INV_DCT,
    true,
    1,
    false
);
inverse_neon_entry!(
    ihdct_dequant_8x16_neon,
    128,
    8,
    16,
    INV_DCT,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_neon_entry!(
    ivdct_dequant_16x8_neon,
    128,
    16,
    8,
    INV_IDENTITY,
    INV_DCT,
    true,
    1,
    false
);
inverse_neon_entry!(
    ihdct_dequant_16x8_neon,
    128,
    16,
    8,
    INV_DCT,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_neon_entry!(
    iidentity_dequant_4x4_neon,
    16,
    4,
    4,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    0,
    false
);
inverse_neon_entry!(
    iidentity_dequant_8x8_neon,
    64,
    8,
    8,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    1,
    false
);
inverse_neon_entry!(
    iidtx_dequant_8x16_neon,
    128,
    8,
    16,
    INV_IDENTITY,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_neon_entry!(
    iidtx_dequant_16x8_neon,
    128,
    16,
    8,
    INV_IDENTITY,
    INV_IDENTITY,
    true,
    1,
    false
);
inverse_neon_entry!(
    iidentity_dequant_16x16_neon,
    256,
    16,
    16,
    INV_IDENTITY,
    INV_IDENTITY,
    false,
    2,
    false
);

#[target_feature(enable = "neon")]
pub(crate) fn iadstdct_dequant_16x16_neon(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_mixed_dequant_neon::<false, true>(levels, dequant)
}

#[target_feature(enable = "neon")]
pub(crate) fn idctadst_dequant_16x16_neon(
    levels: &[i32; 256],
    dequant: &IdctDequant,
) -> [i32; 256] {
    inv16x16_mixed_dequant_neon::<true, false>(levels, dequant)
}

#[target_feature(enable = "neon")]
pub(crate) fn iadst_dequant_16x16_neon(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    inv16x16_mixed_dequant_neon::<true, true>(levels, dequant)
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_8x8_neon(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    if can_use_s16_inverse(dequant) {
        return inverse_dct_s16::<64, 8, 8, false, false, 1>(levels, dequant);
    }
    let (rmin, rmax, cmin, cmax) = (dequant.rmin, dequant.rmax, dequant.cmin, dequant.cmax);
    let coeff = dequant_levels_neon::<64, false>(levels, dequant);

    let load = |x: usize| unsafe {
        I32x8 {
            lo: vld1q_s32(coeff[x * 8..].as_ptr()),
            hi: vld1q_s32(coeff[x * 8 + 4..].as_ptr()),
        }
    };
    let mut v: [I32x8; 8] = std::array::from_fn(load);

    inv_dct8_v(&mut v, rmin, rmax);

    let cmn = vdupq_n_s32(cmin);
    let cmx = vdupq_n_s32(cmax);
    for vv in v.iter_mut() {
        *vv = vv.rsh::<1>(1).clip(cmn, cmx);
    }

    transpose_8x8(&mut v);
    inv_dct8_v(&mut v, cmin, cmax);

    let mut out = MaybeUninit::<[i32; 64]>::uninit();
    for (y, vv) in v.iter().enumerate() {
        let r = vv.rsh::<4>(8);
        unsafe {
            vst1q_s32((out.as_mut_ptr() as *mut i32).add(y * 8), r.lo);
            vst1q_s32((out.as_mut_ptr() as *mut i32).add(y * 8 + 4), r.hi);
        }
    }
    unsafe { out.assume_init() }
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_16x16_neon(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    if can_use_s16_inverse(dequant) {
        return inverse_dct_s16::<256, 16, 16, false, false, 2>(levels, dequant);
    }
    inv16x16_mixed_dequant_neon::<false, false>(levels, dequant)
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_32x32_neon(levels: &[i32; 1024], dequant: &IdctDequant) -> [i32; 1024] {
    // At bd == 8 every intermediate clamp in this transform is exactly the int16
    // range, so the whole thing runs in int16 with saturating arithmetic.
    if can_use_s16_inverse(dequant) {
        return inverse_dct_s16::<1024, 32, 32, true, false, 2>(levels, dequant);
    }
    inverse_neon::<1024, 32, 32, INV_DCT, INV_DCT, false, 2, true>(levels, dequant)
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

    #[target_feature(enable = "neon")]
    fn vector<const DQ1: bool>(levels: [i32; 4], quants: [i32; 4], cf_max: i32) -> [i32; 4] {
        let levels = unsafe { vld1q_s32(levels.as_ptr()) };
        let quants = unsafe { vld1q_s32(quants.as_ptr()) };
        let result = dequant4::<DQ1>(levels, quants, vdupq_n_s32(cf_max));
        let mut out = [0; 4];
        unsafe { vst1q_s32(out.as_mut_ptr(), result) };
        out
    }

    #[test]
    fn packed_dequant_matches_scalar_for_both_shifts_and_all_bit_depth_caps() {
        let level_sets = [
            [i32::MIN, -0x0123_4567, -65536, -1],
            [0, 1, 65536, i32::MAX],
        ];
        let quant_sets = [[1, 13, 255, 256], [4095, 32767, 65535, 233_000]];
        for cf_max in [32767, 131071, 2_097_151] {
            for (levels, quants) in level_sets.into_iter().zip(quant_sets) {
                let got0 = unsafe { vector::<false>(levels, quants, cf_max) };
                let got1 = unsafe { vector::<true>(levels, quants, cf_max) };
                let want0 = std::array::from_fn(|i| scalar(levels[i], quants[i], cf_max, false));
                let want1 = std::array::from_fn(|i| scalar(levels[i], quants[i], cf_max, true));
                assert_eq!(got0, want0, "dq_shift=0 cf_max={cf_max}");
                assert_eq!(got1, want1, "dq_shift=1 cf_max={cf_max}");
            }
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
                rmax: (1 << 17) - 1,
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

#[cfg(test)]
mod s16_real_data {
    // replay harness: MT_COEFF_REPLAY=<N>:<path>
    use super::*;

    fn replay_dim(raw: &[u8], dim: usize, nblocks: usize) {
        let (min, max) = (i16::MIN as i32, i16::MAX as i32);
        let mut diff = 0usize;
        let mut worst = 0i32;
        for b in 0..nblocks {
            let base = b * dim * dim * 4;
            let get = |r: usize, l: usize| {
                let o = base + (r * dim + l) * 4;
                i32::from_le_bytes(raw[o..o + 4].try_into().unwrap())
            };
            let mut want = vec![vec![0i32; dim]; dim];
            let mut got = vec![vec![0i32; dim]; dim];
            unsafe {
                for half in 0..(dim / 4) {
                    let b0 = half * 4;
                    if dim == 16 {
                        let mut cc: [I32x4; 16] = std::array::from_fn(|r| {
                            I32x4(vld1q_s32(
                                [get(r, b0), get(r, b0 + 1), get(r, b0 + 2), get(r, b0 + 3)]
                                    .as_ptr(),
                            ))
                        });
                        inv_dct16_v_x4(&mut cc, min, max);
                        for r in 0..16 {
                            let mut o = [0i32; 4];
                            vst1q_s32(o.as_mut_ptr(), cc[r].0);
                            for l in 0..4 {
                                want[r][b0 + l] = o[l];
                            }
                        }
                    } else {
                        let mut cc: [I32x4; 32] = std::array::from_fn(|r| {
                            I32x4(vld1q_s32(
                                [get(r, b0), get(r, b0 + 1), get(r, b0 + 2), get(r, b0 + 3)]
                                    .as_ptr(),
                            ))
                        });
                        inv_dct32_v_x4(&mut cc, min, max);
                        for r in 0..32 {
                            let mut o = [0i32; 4];
                            vst1q_s32(o.as_mut_ptr(), cc[r].0);
                            for l in 0..4 {
                                want[r][b0 + l] = o[l];
                            }
                        }
                    }
                }
                for half in 0..(dim / 8) {
                    let b0 = half * 8;
                    if dim == 16 {
                        let mut cc: [I16x8v; 16] = std::array::from_fn(|r| {
                            let row: [i16; 8] = std::array::from_fn(|l| get(r, b0 + l) as i16);
                            I16x8v(vld1q_s16(row.as_ptr()))
                        });
                        inv_dct16_v_x8_s16!(&mut cc);
                        for r in 0..16 {
                            let mut o = [0i16; 8];
                            vst1q_s16(o.as_mut_ptr(), cc[r].0);
                            for l in 0..8 {
                                got[r][b0 + l] = o[l] as i32;
                            }
                        }
                    } else {
                        let mut cc: [I16x8v; 32] = std::array::from_fn(|r| {
                            let row: [i16; 8] = std::array::from_fn(|l| get(r, b0 + l) as i16);
                            I16x8v(vld1q_s16(row.as_ptr()))
                        });
                        inv_dct32_v_x8_s16!(&mut cc);
                        for r in 0..32 {
                            let mut o = [0i16; 8];
                            vst1q_s16(o.as_mut_ptr(), cc[r].0);
                            for l in 0..8 {
                                got[r][b0 + l] = o[l] as i32;
                            }
                        }
                    }
                }
            }
            if got != want {
                diff += 1;
                for r in 0..dim {
                    for l in 0..dim {
                        worst = worst.max((got[r][l] - want[r][l]).abs());
                    }
                }
            }
        }
        println!(
            "REAL-DATA REPLAY dim={dim}: {diff} / {nblocks} blocks diverge; worst delta {worst}"
        );
        assert_eq!(diff, 0, "s16 diverges on real {dim}-point data");
    }

    /// Replay REAL dequantised 8x8 blocks (dumped via MT_COEFF_DUMP) through
    /// both lanes. Random full-range coefficients are NOT a legal DCT spectrum;
    /// only real data answers whether s16 saturation ever diverges in practice.
    #[test]
    fn s16_matches_i32_on_real_blocks() {
        let Ok(spec) = std::env::var("MT_COEFF_REPLAY") else {
            eprintln!("MT_COEFF_REPLAY unset — skipping");
            return;
        };
        let (nstr, path) = spec.split_once(':').unwrap();
        let dim: usize = nstr.parse().unwrap();
        let raw = std::fs::read(path).unwrap();
        let bytes_per = dim * dim * 4;
        let nblocks = raw.len() / bytes_per;
        if dim != 8 {
            replay_dim(&raw, dim, nblocks);
            return;
        }
        let (mut diff, mut checked) = (0usize, 0usize);
        let mut worst = 0i32;
        for b in 0..nblocks {
            let base = b * 256;
            let mut src = [[0i32; 8]; 8];
            for r in 0..8 {
                for l in 0..8 {
                    let o = base + (r * 8 + l) * 4;
                    src[r][l] = i32::from_le_bytes(raw[o..o + 4].try_into().unwrap());
                }
            }
            let (min, max) = (i16::MIN as i32, i16::MAX as i32);
            let mut want = [[0i32; 8]; 8];
            unsafe {
                for half in 0..2 {
                    let mut c: [I32x4; 8] = std::array::from_fn(|r| {
                        let b0 = half * 4;
                        I32x4(vld1q_s32(
                            [src[r][b0], src[r][b0 + 1], src[r][b0 + 2], src[r][b0 + 3]].as_ptr(),
                        ))
                    });
                    inv_dct8_v_x4(&mut c, min, max);
                    for r in 0..8 {
                        let mut o = [0i32; 4];
                        vst1q_s32(o.as_mut_ptr(), c[r].0);
                        for l in 0..4 {
                            want[r][half * 4 + l] = o[l];
                        }
                    }
                }
            }
            let mut got = [[0i32; 8]; 8];
            unsafe {
                let mut c: [I16x8v; 8] = std::array::from_fn(|r| {
                    let row: [i16; 8] = std::array::from_fn(|l| src[r][l] as i16);
                    I16x8v(vld1q_s16(row.as_ptr()))
                });
                inv_dct8_v_x8_s16!(&mut c);
                for r in 0..8 {
                    let mut o = [0i16; 8];
                    vst1q_s16(o.as_mut_ptr(), c[r].0);
                    for l in 0..8 {
                        got[r][l] = o[l] as i32;
                    }
                }
            }
            checked += 1;
            if got != want {
                diff += 1;
                for r in 0..8 {
                    for l in 0..8 {
                        worst = worst.max((got[r][l] - want[r][l]).abs());
                    }
                }
            }
        }
        println!("REAL-DATA REPLAY: {diff} / {checked} blocks diverge; worst lane delta {worst}");
        assert_eq!(diff, 0, "s16 diverges on real encoder data");
    }
}

#[cfg(test)]
mod s16_bench {
    use super::*;

    /// Kernel throughput only: inputs are converted to vector form ONCE,
    /// outside the timed region. (A previous version built the input vectors
    /// scalar-wise inside the loop and measured gather overhead instead.)
    #[test]
    fn bench_s16_vs_i32() {
        let Ok(spec) = std::env::var("MT_BENCH_REPLAY") else {
            eprintln!("MT_BENCH_REPLAY unset — skipping");
            return;
        };
        let (nstr, path) = spec.split_once(':').unwrap();
        let dim: usize = nstr.parse().unwrap();
        let raw = std::fs::read(path).unwrap();
        let per = dim * dim * 4;
        let nb = (raw.len() / per).min(3000);
        let (min, max) = (i16::MIN as i32, i16::MAX as i32);
        let get = |b: usize, r: usize, l: usize| {
            let o = b * per + (r * dim + l) * 4;
            i32::from_le_bytes(raw[o..o + 4].try_into().unwrap())
        };

        // ---- preconvert (untimed) ----
        let mut in32: Vec<Vec<I32x4>> = Vec::new();
        let mut in16: Vec<Vec<I16x8v>> = Vec::new();
        unsafe {
            for b in 0..nb {
                for half in 0..(dim / 4) {
                    let b0 = half * 4;
                    in32.push(
                        (0..dim)
                            .map(|r| {
                                I32x4(vld1q_s32(
                                    [
                                        get(b, r, b0),
                                        get(b, r, b0 + 1),
                                        get(b, r, b0 + 2),
                                        get(b, r, b0 + 3),
                                    ]
                                    .as_ptr(),
                                ))
                            })
                            .collect(),
                    );
                }
                for half in 0..(dim / 8) {
                    let b0 = half * 8;
                    in16.push(
                        (0..dim)
                            .map(|r| {
                                let row: [i16; 8] =
                                    std::array::from_fn(|l| get(b, r, b0 + l) as i16);
                                I16x8v(vld1q_s16(row.as_ptr()))
                            })
                            .collect(),
                    );
                }
            }
        }

        const REPS: usize = 60;
        let mut sink = 0i64;

        let t0 = std::time::Instant::now();
        for _ in 0..REPS {
            for src in in32.iter() {
                unsafe {
                    if dim == 32 {
                        let mut c: [I32x4; 32] =
                            std::array::from_fn(|r| std::hint::black_box(src[r]));
                        inv_dct32_v_x4(&mut c, min, max);
                        for v in c.iter() {
                            let mut o = [0i32; 4];
                            vst1q_s32(o.as_mut_ptr(), std::hint::black_box(*v).0);
                            sink += o[0] as i64;
                        }
                    } else {
                        let mut c: [I32x4; 16] =
                            std::array::from_fn(|r| std::hint::black_box(src[r]));
                        inv_dct16_v_x4(&mut c, min, max);
                        for v in c.iter() {
                            let mut o = [0i32; 4];
                            vst1q_s32(o.as_mut_ptr(), std::hint::black_box(*v).0);
                            sink += o[0] as i64;
                        }
                    }
                }
            }
        }
        let t_i32 = t0.elapsed();

        let t1 = std::time::Instant::now();
        for _ in 0..REPS {
            for src in in16.iter() {
                unsafe {
                    if dim == 32 {
                        let mut c: [I16x8v; 32] =
                            std::array::from_fn(|r| std::hint::black_box(src[r]));
                        inv_dct32_v_x8_s16!(&mut c);
                        for v in c.iter() {
                            let mut o = [0i16; 8];
                            vst1q_s16(o.as_mut_ptr(), std::hint::black_box(*v).0);
                            sink += o[0] as i64;
                        }
                    } else {
                        let mut c: [I16x8v; 16] =
                            std::array::from_fn(|r| std::hint::black_box(src[r]));
                        inv_dct16_v_x8_s16!(&mut c);
                        for v in c.iter() {
                            let mut o = [0i16; 8];
                            vst1q_s16(o.as_mut_ptr(), std::hint::black_box(*v).0);
                            sink += o[0] as i64;
                        }
                    }
                }
            }
        }
        let t_s16 = t1.elapsed();

        println!(
            "BENCH dim={dim}: {} cols/iter each | i32 {:?}  s16 {:?}  speedup {:.2}x  (sink {sink})",
            nb * dim,
            t_i32,
            t_s16,
            t_i32.as_secs_f64() / t_s16.as_secs_f64()
        );
    }
}
