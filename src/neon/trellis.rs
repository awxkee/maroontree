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

use core::arch::aarch64::*;

use crate::cost::{COEF_RATE_LUT, coef_rate_bits, rate_cost};

#[target_feature(enable = "neon")]
pub(crate) fn trellis_dist_current_zero_scan_neon(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    let mut i = 0usize;
    let n = scan.len();
    while i + 4 <= n {
        let r0 = unsafe { *scan.get_unchecked(i) } as usize;
        let r1 = unsafe { *scan.get_unchecked(i + 1) } as usize;
        let r2 = unsafe { *scan.get_unchecked(i + 2) } as usize;
        let r3 = unsafe { *scan.get_unchecked(i + 3) } as usize;
        let tv = unsafe {
            [
                *tf.get_unchecked(r0),
                *tf.get_unchecked(r1),
                *tf.get_unchecked(r2),
                *tf.get_unchecked(r3),
            ]
        };
        let lv = unsafe {
            [
                *cf.get_unchecked(r0),
                *cf.get_unchecked(r1),
                *cf.get_unchecked(r2),
                *cf.get_unchecked(r3),
            ]
        };
        let qv = [
            if r0 == 0 { dc_q2 } else { ac_q2 },
            if r1 == 0 { dc_q2 } else { ac_q2 },
            if r2 == 0 { dc_q2 } else { ac_q2 },
            if r3 == 0 { dc_q2 } else { ac_q2 },
        ];
        let t = vabsq_f32(unsafe { vld1q_f32(tv.as_ptr()) });
        let l = vcvtq_f32_s32(vabsq_s32(unsafe { vld1q_s32(lv.as_ptr()) }));
        let q = unsafe { vld1q_f32(qv.as_ptr()) };
        let e = vsubq_f32(t, l);
        let cur = vmulq_f32(q, vmulq_f32(e, e));
        let zero = vmulq_f32(q, vmulq_f32(t, t));
        unsafe {
            vst1q_f32(dst_cur.as_mut_ptr().add(i), cur);
            vst1q_f32(dst_zero.as_mut_ptr().add(i), zero);
        }
        i += 4;
    }
    for ((out_cur, out_zero), &rc32) in dst_cur[i..n]
        .iter_mut()
        .zip(dst_zero[i..n].iter_mut())
        .zip(scan[i..n].iter())
    {
        let rc = rc32 as usize;
        let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
        let at = unsafe { (*tf.get_unchecked(rc)).abs() };
        let e = at - unsafe { (*cf.get_unchecked(rc)).unsigned_abs() } as f32;
        *out_cur = dq2 * (e * e);
        *out_zero = dq2 * (at * at);
    }
}

#[inline]
fn round_down_scalar_one(
    cf: &mut [i32],
    tf: &[f32],
    rc: usize,
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    let c = cf[rc];
    if c == 0 {
        return;
    }
    let l = c.unsigned_abs();
    let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
    let at = tf[rc].abs();
    let e_l = at - l as f32;
    let e_dn = at - (l - 1) as f32;
    let cost_l = dq2 * (e_l * e_l) + rate_cost(lambda, coef_rate_bits(l));
    let cost_dn = dq2 * (e_dn * e_dn) + rate_cost(lambda, coef_rate_bits(l - 1));
    if cost_dn < cost_l {
        cf[rc] = if c < 0 { -(l as i32 - 1) } else { l as i32 - 1 };
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn trellis_round_down_scan_neon(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    let mut i = 0usize;
    let n = scan.len();
    let one_i = vdupq_n_s32(1);
    let zero_i = vdupq_n_s32(0);
    let lambda_v = vdupq_n_f32(lambda);

    while i + 4 <= n {
        let r0 = unsafe { *scan.get_unchecked(i) } as usize;
        let r1 = unsafe { *scan.get_unchecked(i + 1) } as usize;
        let r2 = unsafe { *scan.get_unchecked(i + 2) } as usize;
        let r3 = unsafe { *scan.get_unchecked(i + 3) } as usize;

        let c0 = unsafe { *cf.get_unchecked(r0) };
        let c1 = unsafe { *cf.get_unchecked(r1) };
        let c2 = unsafe { *cf.get_unchecked(r2) };
        let c3 = unsafe { *cf.get_unchecked(r3) };
        let l0 = c0.unsigned_abs() as i32;
        let l1 = c1.unsigned_abs() as i32;
        let l2 = c2.unsigned_abs() as i32;
        let l3 = c3.unsigned_abs() as i32;

        // COEF_RATE_LUT covers 0..63. Preserve exact scalar Golomb-tail behavior
        // by falling back chunk-wise as soon as any lane leaves the LUT domain.
        if (l0 | l1 | l2 | l3) >= 64 {
            round_down_scalar_one(cf, tf, r0, dc_q2, ac_q2, lambda);
            round_down_scalar_one(cf, tf, r1, dc_q2, ac_q2, lambda);
            round_down_scalar_one(cf, tf, r2, dc_q2, ac_q2, lambda);
            round_down_scalar_one(cf, tf, r3, dc_q2, ac_q2, lambda);
            i += 4;
            continue;
        }

        let tv = unsafe {
            [
                *tf.get_unchecked(r0),
                *tf.get_unchecked(r1),
                *tf.get_unchecked(r2),
                *tf.get_unchecked(r3),
            ]
        };
        let lv = [l0, l1, l2, l3];
        let ldnv = [
            (l0 - 1).max(0),
            (l1 - 1).max(0),
            (l2 - 1).max(0),
            (l3 - 1).max(0),
        ];
        let qv = [
            if r0 == 0 { dc_q2 } else { ac_q2 },
            if r1 == 0 { dc_q2 } else { ac_q2 },
            if r2 == 0 { dc_q2 } else { ac_q2 },
            if r3 == 0 { dc_q2 } else { ac_q2 },
        ];
        let rate_l = unsafe {
            [
                *COEF_RATE_LUT.get_unchecked(l0 as usize),
                *COEF_RATE_LUT.get_unchecked(l1 as usize),
                *COEF_RATE_LUT.get_unchecked(l2 as usize),
                *COEF_RATE_LUT.get_unchecked(l3 as usize),
            ]
        };
        let rate_dn = unsafe {
            [
                *COEF_RATE_LUT.get_unchecked(ldnv[0] as usize),
                *COEF_RATE_LUT.get_unchecked(ldnv[1] as usize),
                *COEF_RATE_LUT.get_unchecked(ldnv[2] as usize),
                *COEF_RATE_LUT.get_unchecked(ldnv[3] as usize),
            ]
        };

        let at = vabsq_f32(unsafe { vld1q_f32(tv.as_ptr()) });
        let l = vcvtq_f32_s32(unsafe { vld1q_s32(lv.as_ptr()) });
        let l_dn = vcvtq_f32_s32(vmaxq_s32(
            vsubq_s32(unsafe { vld1q_s32(lv.as_ptr()) }, one_i),
            zero_i,
        ));
        let q = unsafe { vld1q_f32(qv.as_ptr()) };
        let e_l = vsubq_f32(at, l);
        let e_dn = vsubq_f32(at, l_dn);
        let cost_l = vfmaq_f32(
            vmulq_f32(lambda_v, unsafe { vld1q_f32(rate_l.as_ptr()) }),
            q,
            vmulq_f32(e_l, e_l),
        );
        let cost_dn = vfmaq_f32(
            vmulq_f32(lambda_v, unsafe { vld1q_f32(rate_dn.as_ptr()) }),
            q,
            vmulq_f32(e_dn, e_dn),
        );
        let better = vcltq_f32(cost_dn, cost_l);
        let mut mask = [0u32; 4];
        unsafe { vst1q_u32(mask.as_mut_ptr(), better) };

        if mask[0] != 0 && c0 != 0 {
            unsafe { *cf.get_unchecked_mut(r0) = if c0 < 0 { -(l0 - 1) } else { l0 - 1 } };
        }
        if mask[1] != 0 && c1 != 0 {
            unsafe { *cf.get_unchecked_mut(r1) = if c1 < 0 { -(l1 - 1) } else { l1 - 1 } };
        }
        if mask[2] != 0 && c2 != 0 {
            unsafe { *cf.get_unchecked_mut(r2) = if c2 < 0 { -(l2 - 1) } else { l2 - 1 } };
        }
        if mask[3] != 0 && c3 != 0 {
            unsafe { *cf.get_unchecked_mut(r3) = if c3 < 0 { -(l3 - 1) } else { l3 - 1 } };
        }
        i += 4;
    }

    for &rc32 in scan[i..n].iter() {
        round_down_scalar_one(cf, tf, rc32 as usize, dc_q2, ac_q2, lambda);
    }
}
