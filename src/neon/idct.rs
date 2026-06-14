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

/// Transpose a 4x4 i32 tile (verbatim port of the forward DCT helper).
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

/// Dequant 4 levels: `coeff = sign(lvl) * min((|lvl|*q) & 0xff_ffff, cf_max + (lvl<0))`.
/// `lvl==0` falls out to 0 naturally. Matches the scalar dequant loop bit-for-bit.
#[inline]
#[target_feature(enable = "neon")]
fn dequant4(lvl: int32x4_t, q: int32x4_t, cf_max: int32x4_t) -> int32x4_t {
    let absl = vabsq_s32(lvl);
    // 64-bit widening |lvl|*q (both >= 0), then mask the low 24 bits.
    let mask = vdupq_n_s64(0xff_ffff);
    let plo = vandq_s64(vmull_s32(vget_low_s32(absl), vget_low_s32(q)), mask);
    let phi = vandq_s64(vmull_s32(vget_high_s32(absl), vget_high_s32(q)), mask);
    // Masked value <= 0xff_ffff, so the narrow to i32 is exact.
    let masked = vcombine_s32(vmovn_s64(plo), vmovn_s64(phi));
    // cap = cf_max + (lvl < 0 ? 1 : 0)
    let neg1 = vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(lvl), 31));
    let cap = vaddq_s32(cf_max, neg1);
    let mag = vminq_s32(masked, cap);
    // Apply sign of lvl: negative lanes take -mag.
    let neg = vnegq_s32(mag);
    let signmask = vcltq_s32(lvl, vdupq_n_s32(0));
    vbslq_s32(signmask, neg, mag)
}

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_8x8_neon(levels: &[i32; 64], dequant: &IdctDequant) -> [i32; 64] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let (dc_q, ac_q) = (dequant.dc_q, dequant.ac_q);

    let cfm = vdupq_n_s32(cf_max);
    let mut coeff = [0i32; 64];
    // first group of 4: lane 0 is DC.
    let q_dc = vsetq_lane_s32(dc_q, vdupq_n_s32(ac_q), 0);
    let q_ac = vdupq_n_s32(ac_q);
    unsafe {
        let l = vld1q_s32(levels.as_ptr());
        vst1q_s32(coeff.as_mut_ptr(), dequant4(l, q_dc, cfm));
        let mut i = 4;
        while i < 64 {
            let l = vld1q_s32(levels.as_ptr().add(i));
            vst1q_s32(coeff.as_mut_ptr().add(i), dequant4(l, q_ac, cfm));
            i += 4;
        }
    }

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
