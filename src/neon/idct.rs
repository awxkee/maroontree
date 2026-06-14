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

#[inline]
#[target_feature(enable = "neon")]
fn inv_dct16_v(c: &mut [I32x8; 16], min: i32, max: i32) {
    let mn = vdupq_n_s32(min);
    let mx = vdupq_n_s32(max);
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

#[target_feature(enable = "neon")]
pub(crate) fn idct_dequant_16x16_neon(levels: &[i32; 256], dequant: &IdctDequant) -> [i32; 256] {
    let (rmin, rmax, cmin, cmax, cf_max) = (
        dequant.rmin,
        dequant.rmax,
        dequant.cmin,
        dequant.cmax,
        dequant.cf_max,
    );
    let (dc_q, ac_q) = (dequant.dc_q, dequant.ac_q);

    let cfm = vdupq_n_s32(cf_max);
    let q_dc = vsetq_lane_s32(dc_q, vdupq_n_s32(ac_q), 0);
    let q_ac = vdupq_n_s32(ac_q);
    let mut coeff_u = MaybeUninit::<[i32; 256]>::uninit();
    unsafe {
        vst1q_s32(
            coeff_u.as_mut_ptr().cast(),
            dequant4(vld1q_s32(levels.as_ptr()), q_dc, cfm),
        );
        let mut i = 4;
        while i < 256 {
            let l = vld1q_s32(levels.as_ptr().add(i));
            let coeff_ptr = coeff_u.as_mut_ptr() as *mut i32;
            vst1q_s32(coeff_ptr.add(i), dequant4(l, q_ac, cfm));
            i += 4;
        }
    }

    let coeff = unsafe { coeff_u.assume_init() };

    let load_l = |x: usize| unsafe {
        I32x8 {
            lo: vld1q_s32(coeff.as_ptr().add(x * 16)),
            hi: vld1q_s32(coeff.as_ptr().add(x * 16 + 4)),
        }
    };
    let load_r = |x: usize| unsafe {
        I32x8 {
            lo: vld1q_s32(coeff.as_ptr().add(x * 16 + 8)),
            hi: vld1q_s32(coeff.as_ptr().add(x * 16 + 12)),
        }
    };
    let mut c_l: [I32x8; 16] = std::array::from_fn(load_l);
    let mut c_r: [I32x8; 16] = std::array::from_fn(load_r);

    inv_dct16_v(&mut c_l, rmin, rmax);
    inv_dct16_v(&mut c_r, rmin, rmax);

    let cmn = vdupq_n_s32(cmin);
    let cmx = vdupq_n_s32(cmax);
    for vv in c_l.iter_mut().chain(c_r.iter_mut()) {
        *vv = vv.rsh::<2>(2).clip(cmn, cmx);
    }

    let mut ll: [I32x8; 8] = c_l[0..8].try_into().unwrap();
    let mut lr: [I32x8; 8] = c_l[8..16].try_into().unwrap();
    let mut rl: [I32x8; 8] = c_r[0..8].try_into().unwrap();
    let mut rr: [I32x8; 8] = c_r[8..16].try_into().unwrap();
    transpose_8x8(&mut ll);
    transpose_8x8(&mut lr);
    transpose_8x8(&mut rl);
    transpose_8x8(&mut rr);
    // w_l: cols 0..8, rows 0..8 = ll, rows 8..16 = rl ; w_r: cols 8..16, lr / rr
    let mut w_l: [I32x8; 16] = std::array::from_fn(|i| if i < 8 { ll[i] } else { rl[i - 8] });
    let mut w_r: [I32x8; 16] = std::array::from_fn(|i| if i < 8 { lr[i] } else { rr[i - 8] });

    // --- col pass: inv_dct16 across y (vector index) per lane x; clip [cmin,cmax] ---
    inv_dct16_v(&mut w_l, cmin, cmax);
    inv_dct16_v(&mut w_r, cmin, cmax);

    // final: (t + 8) >> 4 (no clamp), store row-major out[y*16 + x]
    let mut out = MaybeUninit::<[i32; 256]>::uninit();
    for r in 0..16 {
        let lo = w_l[r].rsh::<4>(8); // cols 0..8
        let hi = w_r[r].rsh::<4>(8); // cols 8..16
        unsafe {
            let dst_ptr = out.as_mut_ptr() as *mut i32;
            vst1q_s32(dst_ptr.add(r * 16), lo.lo);
            vst1q_s32(dst_ptr.add(r * 16 + 4), lo.hi);
            vst1q_s32(dst_ptr.add(r * 16 + 8), hi.lo);
            vst1q_s32(dst_ptr.add(r * 16 + 12), hi.hi);
        }
    }
    unsafe { out.assume_init() }
}
