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

pub(crate) type UvNearestIndicesFn =
    fn(&[u16], &[u16], usize, usize, usize, usize, usize, &[(i32, i32)], &mut [u8]);

pub(crate) type LumaNearestIndicesFn =
    fn(&[u16], usize, usize, usize, usize, usize, &[i32], &mut [u8]);

#[allow(clippy::too_many_arguments)]
pub(crate) fn uv_nearest_indices_scalar(
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
    for y in 0..h {
        for x in 0..w {
            let u = src_u[(cy + y) * stride + cx + x] as i32;
            let v = src_v[(cy + y) * stride + cx + x] as i32;
            let mut best_dist = i64::MAX;
            let mut best_idx = 0;
            for (i, &(cu, cv)) in centers.iter().enumerate() {
                let du = i64::from(u - cu);
                let dv = i64::from(v - cv);
                let dist = du * du + dv * dv;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            out[y * w + x] = best_idx as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_nearest_indices_scalar(
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
    for y in 0..h {
        for x in 0..w {
            let sample = src[(py + y) * stride + px + x] as i32;
            let mut best_dist = i64::MAX;
            let mut best_idx = 0;
            for (i, &center) in centers.iter().enumerate() {
                let delta = i64::from(sample - center);
                let dist = delta * delta;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }
            out[y * w + x] = best_idx as u8;
        }
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[allow(clippy::too_many_arguments)]
fn uv_nearest_indices_neon_wrap(
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
    unsafe {
        crate::neon::uv_nearest_indices_neon(src_u, src_v, stride, cx, cy, w, h, centers, out)
    }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
#[allow(clippy::too_many_arguments)]
fn luma_nearest_indices_neon_wrap(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    centers: &[i32],
    out: &mut [u8],
) {
    unsafe { crate::neon::luma_nearest_indices_neon(src, stride, px, py, w, h, centers, out) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[allow(clippy::too_many_arguments)]
fn uv_nearest_indices_avx2_wrap(
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
    unsafe { crate::avx::uv_nearest_indices_avx2(src_u, src_v, stride, cx, cy, w, h, centers, out) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
#[allow(clippy::too_many_arguments)]
fn luma_nearest_indices_avx2_wrap(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    centers: &[i32],
    out: &mut [u8],
) {
    unsafe { crate::avx::luma_nearest_indices_avx2(src, stride, px, py, w, h, centers, out) }
}

#[derive(Clone, Copy)]
pub(crate) struct KmeansDispatch {
    pub(crate) uv_nearest_indices: UvNearestIndicesFn,
    pub(crate) luma_nearest_indices: LumaNearestIndicesFn,
}

impl KmeansDispatch {
    pub(crate) const fn scalar() -> Self {
        Self {
            uv_nearest_indices: uv_nearest_indices_scalar,
            luma_nearest_indices: luma_nearest_indices_scalar,
        }
    }

    pub(crate) fn selected() -> Self {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            Self {
                uv_nearest_indices: uv_nearest_indices_neon_wrap,
                luma_nearest_indices: luma_nearest_indices_neon_wrap,
            }
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                Self {
                    uv_nearest_indices: uv_nearest_indices_avx2_wrap,
                    luma_nearest_indices: luma_nearest_indices_avx2_wrap,
                }
            } else {
                Self::scalar()
            }
        }
        #[cfg(not(any(
            all(target_arch = "aarch64", feature = "neon"),
            all(target_arch = "x86_64", feature = "avx")
        )))]
        {
            Self::scalar()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selected_matches_scalar_including_ties_and_tails() {
        let (w, h, stride) = (29usize, 5usize, 48usize);
        let mut u = vec![0u16; stride * 16];
        let mut v = vec![0u16; stride * 16];
        for i in 0..u.len() {
            u[i] = ((i * 37) % 4096) as u16;
            v[i] = ((i * 53) % 4096) as u16;
        }
        u[2 * stride + 3] = 0;
        v[2 * stride + 3] = 4095;
        u[2 * stride + 4] = 4095;
        v[2 * stride + 4] = 0;
        let dispatch = KmeansDispatch::selected();
        let centers_uv = [
            (0, 4095),
            (4095, 0),
            (2048, 2048),
            (2048, 2048),
            (4095, 4095),
        ];
        let centers_y = [0, 4095, 2048, 2048, 1];
        for k in 1..=centers_uv.len() {
            let mut scalar = vec![0; w * h];
            let mut selected = vec![0; w * h];
            uv_nearest_indices_scalar(&u, &v, stride, 3, 2, w, h, &centers_uv[..k], &mut scalar);
            (dispatch.uv_nearest_indices)(
                &u,
                &v,
                stride,
                3,
                2,
                w,
                h,
                &centers_uv[..k],
                &mut selected,
            );
            assert_eq!(selected, scalar, "UV k={k}");

            scalar.fill(0);
            selected.fill(0);
            luma_nearest_indices_scalar(&u, stride, 3, 2, w, h, &centers_y[..k], &mut scalar);
            (dispatch.luma_nearest_indices)(&u, stride, 3, 2, w, h, &centers_y[..k], &mut selected);
            assert_eq!(selected, scalar, "luma k={k}");
        }
    }
}
