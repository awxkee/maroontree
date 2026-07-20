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

use std::arch::aarch64::*;

/// Nearest centroid for each pixel of a `w`x`h` UV block (k-means DIM = 2).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn uv_nearest_indices_neon(
    src_u: &[u16],
    src_v: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    centers: &[(i32, i32)],
    out: &mut [u8],
) {
    debug_assert!((1..=8).contains(&centers.len()));
    debug_assert_eq!(out.len(), w * h);
    // Splat the centroids once for the whole block rather than per 8 pixels.
    let mut cu = [vdupq_n_s16(0); 8];
    let mut cv = [vdupq_n_s16(0); 8];
    for (j, &(u, v)) in centers.iter().enumerate() {
        cu[j] = vdupq_n_s16(u as i16);
        cv[j] = vdupq_n_s16(v as i16);
    }
    let k = centers.len();
    assert!(k <= 8);
    for (y, orow) in out.chunks_exact_mut(w).enumerate() {
        let ru = &src_u[(cy + y) * stride + cx..][..w];
        let rv = &src_v[(cy + y) * stride + cx..][..w];
        let mut uc = ru.as_chunks::<8>().0.iter();
        let mut vc = rv.as_chunks::<8>().0.iter();
        let mut oc = orow.as_chunks_mut::<8>().0.iter_mut();
        for ((ub, vb), ob) in (&mut uc).zip(&mut vc).zip(&mut oc) {
            let u = unsafe { vreinterpretq_s16_u16(vld1q_u16(ub.as_ptr())) };
            let v = unsafe { vreinterpretq_s16_u16(vld1q_u16(vb.as_ptr())) };
            let mut best_lo = vdupq_n_s32(i32::MAX);
            let mut best_hi = vdupq_n_s32(i32::MAX);
            let mut idx_lo = vdupq_n_s32(0);
            let mut idx_hi = vdupq_n_s32(0);
            for j in 0..k {
                let du = vsubq_s16(u, cu[j]);
                let dv = vsubq_s16(v, cv[j]);
                let dlo = vaddq_s32(
                    vmull_s16(vget_low_s16(du), vget_low_s16(du)),
                    vmull_s16(vget_low_s16(dv), vget_low_s16(dv)),
                );
                let dhi = vaddq_s32(vmull_high_s16(du, du), vmull_high_s16(dv, dv));
                let jv = vdupq_n_s32(j as i32);
                // STRICT less-than in increasing index order reproduces the
                // scalar `(dist, index)` tuple order exactly.
                let m_lo = vcltq_s32(dlo, best_lo);
                let m_hi = vcltq_s32(dhi, best_hi);
                idx_lo = vbslq_s32(m_lo, jv, idx_lo);
                idx_hi = vbslq_s32(m_hi, jv, idx_hi);
                best_lo = vminq_s32(dlo, best_lo);
                best_hi = vminq_s32(dhi, best_hi);
            }
            // Narrow 2x i32x4 -> u8x8 and store straight into the output row.
            unsafe {
                vst1_u8(
                    ob.as_mut_ptr(),
                    vmovn_u16(vcombine_u16(
                        vmovn_u32(vreinterpretq_u32_s32(idx_lo)),
                        vmovn_u32(vreinterpretq_u32_s32(idx_hi)),
                    )),
                )
            };
        }
        for ((&uu, &vv), o) in ru
            .as_chunks::<8>()
            .1
            .iter()
            .zip(rv.as_chunks::<8>().1.iter())
            .zip(orow.as_chunks_mut::<8>().1.iter_mut())
        {
            let (uu, vv) = (uu as i32, vv as i32);
            let mut best = i32::MAX;
            let mut bi = 0u8;
            for (j, &(c0, c1)) in centers.iter().enumerate() {
                let (du, dv) = (uu - c0, vv - c1);
                let d = du * du + dv * dv;
                if d < best {
                    best = d;
                    bi = j as u8;
                }
            }
            *o = bi;
        }
    }
}

/// Nearest centroid for each pixel of a `w`x`h` luma block (k-means DIM = 1).
#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn luma_nearest_indices_neon(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    centers: &[i32],
    out: &mut [u8],
) {
    debug_assert!((1..=8).contains(&centers.len()));
    debug_assert_eq!(out.len(), w * h);
    let mut cs = [vdupq_n_s16(0); 8];
    for (j, &c) in centers.iter().enumerate() {
        cs[j] = vdupq_n_s16(c as i16);
    }
    let k = centers.len();
    assert!(k <= 8);
    for (y, orow) in out.chunks_exact_mut(w).enumerate() {
        let row = &src[(py + y) * stride + px..][..w];
        let mut rc = row.as_chunks::<8>().0.iter();
        let mut oc = orow.as_chunks_mut::<8>().0.iter_mut();
        for (rb, ob) in (&mut rc).zip(&mut oc) {
            let s = unsafe { vreinterpretq_s16_u16(vld1q_u16(rb.as_ptr())) };
            let mut best_lo = vdupq_n_s32(i32::MAX);
            let mut best_hi = vdupq_n_s32(i32::MAX);
            let mut idx_lo = vdupq_n_s32(0);
            let mut idx_hi = vdupq_n_s32(0);
            #[allow(clippy::needless_range_loop)]
            for j in 0..k {
                let d = vsubq_s16(s, cs[j]);
                let dlo = vmull_s16(vget_low_s16(d), vget_low_s16(d));
                let dhi = vmull_high_s16(d, d);
                let jv = vdupq_n_s32(j as i32);
                let m_lo = vcltq_s32(dlo, best_lo);
                let m_hi = vcltq_s32(dhi, best_hi);
                idx_lo = vbslq_s32(m_lo, jv, idx_lo);
                idx_hi = vbslq_s32(m_hi, jv, idx_hi);
                best_lo = vminq_s32(dlo, best_lo);
                best_hi = vminq_s32(dhi, best_hi);
            }
            unsafe {
                vst1_u8(
                    ob.as_mut_ptr(),
                    vmovn_u16(vcombine_u16(
                        vmovn_u32(vreinterpretq_u32_s32(idx_lo)),
                        vmovn_u32(vreinterpretq_u32_s32(idx_hi)),
                    )),
                )
            };
        }
        for (&sv, o) in row
            .as_chunks::<8>()
            .1
            .iter()
            .zip(orow.as_chunks_mut::<8>().1.iter_mut())
        {
            let sv = sv as i32;
            let mut best = i32::MAX;
            let mut bi = 0u8;
            for (j, &c) in centers.iter().enumerate() {
                let d = (sv - c) * (sv - c);
                if d < best {
                    best = d;
                    bi = j as u8;
                }
            }
            *o = bi;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kmeans::{luma_nearest_indices_scalar, uv_nearest_indices_scalar};

    #[test]
    fn matches_scalar_including_ties() {
        let (w, h, stride) = (13usize, 5usize, 32usize);
        let mut u = vec![0u16; stride * 16];
        let mut v = vec![0u16; stride * 16];
        for i in 0..u.len() {
            u[i] = ((i * 37) % 256) as u16;
            v[i] = ((i * 53) % 256) as u16;
        }
        // Centroids placed so equidistant ties occur (0 and 8 around 4).
        let centers = [(4i32, 4i32), (4, 4), (0, 8), (8, 0), (128, 128)];
        for k in 1..=centers.len() {
            let c = &centers[..k];
            let mut got = vec![0u8; w * h];
            unsafe { uv_nearest_indices_neon(&u, &v, stride, 3, 2, w, h, c, &mut got) };
            let mut want = vec![0u8; w * h];
            uv_nearest_indices_scalar(&u, &v, stride, 3, 2, w, h, c, &mut want);
            assert_eq!(got, want, "uv k={k}");
        }

        let lc = [4i32, 4, 0, 8, 200];
        for k in 1..=lc.len() {
            let c = &lc[..k];
            let mut got = vec![0u8; w * h];
            unsafe { luma_nearest_indices_neon(&u, stride, 3, 2, w, h, c, &mut got) };
            let mut want = vec![0u8; w * h];
            luma_nearest_indices_scalar(&u, stride, 3, 2, w, h, c, &mut want);
            assert_eq!(got, want, "luma k={k}");
        }
    }
}
