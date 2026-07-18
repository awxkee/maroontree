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

pub(crate) fn sad_f32(
    src: &[f32],
    sstride: usize,
    pred: &[f32],
    pstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            // Safety: neon is available; slices cover w x h at the given strides.
            return unsafe { crate::av2::neon::sad_f32_neon(src, sstride, pred, pstride, w, h) };
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { crate::av2::avx::sad_f32_avx2(src, sstride, pred, pstride, w, h) };
        }
    }
    sad_f32_scalar(src, sstride, pred, pstride, w, h)
}

pub(crate) fn sad_f32_scalar(
    src: &[f32],
    sstride: usize,
    pred: &[f32],
    pstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut total = 0i64;
    for r in 0..h {
        let sr = &src[r * sstride..r * sstride + w];
        let pr = &pred[r * pstride..r * pstride + w];
        for k in 0..w {
            total += ((sr[k] - pr[k]).round() as i32).unsigned_abs() as i64;
        }
    }
    total as u64
}

/// Geometry and scaling for a tightly packed residual destination.
#[derive(Clone, Copy)]
pub(crate) struct ResidualSpec {
    pub(crate) src_stride: usize,
    pub(crate) pred_stride: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) scale: f32,
}

/// Generate a scaled, tightly packed residual from independently-strided f32
/// source and prediction planes.
pub(crate) fn scaled_residual_f32(dst: &mut [f32], src: &[f32], pred: &[f32], spec: ResidualSpec) {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            unsafe { crate::av2::neon::scaled_residual_f32_neon(dst, src, pred, spec) };
            return;
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            unsafe { crate::av2::avx::scaled_residual_f32_avx2(dst, src, pred, spec) };
            return;
        }
    }
    scaled_residual_f32_scalar(dst, src, pred, spec);
}

pub(crate) fn scaled_residual_f32_scalar(
    dst: &mut [f32],
    src: &[f32],
    pred: &[f32],
    spec: ResidualSpec,
) {
    let ResidualSpec {
        src_stride,
        pred_stride,
        width,
        height,
        scale,
    } = spec;
    debug_assert!(dst.len() >= width * height);
    for y in 0..height {
        for x in 0..width {
            dst[y * width + x] = (src[y * src_stride + x] - pred[y * pred_stride + x]) * scale;
        }
    }
}

/// Copy an independently-strided f32 predictor into a packed buffer while
/// generating the corresponding packed, scaled residual.
pub(crate) fn copy_f32_prediction_and_scaled_residual(
    prediction_dst: &mut [f32],
    residual_dst: &mut [f32],
    src: &[f32],
    prediction: &[f32],
    spec: ResidualSpec,
) {
    let ResidualSpec {
        src_stride,
        pred_stride,
        width,
        height,
        scale,
    } = spec;
    debug_assert!(prediction_dst.len() >= width * height);
    debug_assert!(residual_dst.len() >= width * height);
    for y in 0..height {
        let dst = y * width;
        let src = &src[y * src_stride..][..width];
        let pred = &prediction[y * pred_stride..][..width];
        prediction_dst[dst..dst + width].copy_from_slice(pred);
        for ((residual, &source), &reference) in
            residual_dst[dst..dst + width].iter_mut().zip(src).zip(pred)
        {
            *residual = (source - reference) * scale;
        }
    }
}

/// Convert a packed or strided u16 motion-compensation predictor to packed f32
/// while generating its scaled residual against a strided source plane.
pub(crate) fn u16_prediction_and_scaled_residual_f32(
    prediction_dst: &mut [f32],
    residual_dst: &mut [f32],
    src: &[f32],
    prediction: &[u16],
    spec: ResidualSpec,
) {
    let ResidualSpec {
        src_stride,
        pred_stride,
        width,
        height,
        scale,
    } = spec;
    debug_assert!(prediction_dst.len() >= width * height);
    debug_assert!(residual_dst.len() >= width * height);
    for y in 0..height {
        let dst = y * width;
        let src = &src[y * src_stride..][..width];
        let pred = &prediction[y * pred_stride..][..width];
        for (((prediction_dst, residual), &source), &reference) in prediction_dst[dst..dst + width]
            .iter_mut()
            .zip(&mut residual_dst[dst..dst + width])
            .zip(src)
            .zip(pred)
        {
            let reference = reference as f32;
            *prediction_dst = reference;
            *residual = (source - reference) * scale;
        }
    }
}

/// Convert an f32 predictor to the integer form consumed by chroma inverse
/// transforms while generating its scaled residual against a strided source.
pub(crate) fn f32_prediction_and_scaled_residual_i32(
    prediction_dst: &mut [i32],
    residual_dst: &mut [f32],
    src: &[f32],
    prediction: &[f32],
    spec: ResidualSpec,
) {
    scaled_residual_f32(residual_dst, src, prediction, spec);
    prediction_f32_to_i32(
        prediction_dst,
        prediction,
        spec.pred_stride,
        spec.width,
        spec.height,
    );
}

/// Convert a strided f32 prediction rectangle to a packed integer buffer using
/// the encoder's existing positive-sample rounding convention.
pub(crate) fn prediction_f32_to_i32(
    dst: &mut [i32],
    prediction: &[f32],
    prediction_stride: usize,
    width: usize,
    height: usize,
) {
    debug_assert!(dst.len() >= width * height);
    for y in 0..height {
        for (dst, &prediction) in dst[y * width..][..width]
            .iter_mut()
            .zip(&prediction[y * prediction_stride..][..width])
        {
            *dst = (prediction + 0.5) as i32;
        }
    }
}

#[inline(always)]
pub(crate) fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
    let (e, f, g, h) = (a + c, a - c, b + d, b - d);
    [e + g, f + h, f - h, e - g]
}

pub(crate) fn satd_f32(
    src: &[f32],
    sstride: usize,
    pred: &[f32],
    pstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        if std::arch::is_aarch64_feature_detected!("neon") {
            return unsafe { crate::av2::neon::satd_f32_neon(src, sstride, pred, pstride, w, h) };
        }
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { crate::av2::avx::satd_f32_avx2(src, sstride, pred, pstride, w, h) };
        }
    }
    satd_f32_scalar(src, sstride, pred, pstride, w, h)
}

pub(crate) fn satd_f32_scalar(
    src: &[f32],
    sstride: usize,
    pred: &[f32],
    pstride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut total = 0u64;
    let mut ty = 0;
    while ty < h {
        let mut tx = 0;
        while tx < w {
            let mut m = [[0i32; 4]; 4];
            for r in 0..4 {
                let sr = &src[(ty + r) * sstride + tx..];
                let pr = &pred[(ty + r) * pstride + tx..];
                let d: [i32; 4] = std::array::from_fn(|k| (sr[k] - pr[k]).round() as i32);
                m[r] = had4(d[0], d[1], d[2], d[3]);
            }
            for (((&a, &b), &c), &d) in m[0].iter().zip(&m[1]).zip(&m[2]).zip(&m[3]) {
                let col = had4(a, b, c, d);
                for v in col {
                    total += v.unsigned_abs() as u64;
                }
            }
            tx += 4;
        }
        ty += 4;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(vals: &[i32], w: usize, h: usize) -> Vec<f32> {
        assert_eq!(vals.len(), w * h);
        vals.iter().map(|&v| v as f32).collect()
    }

    #[test]
    fn sad_matches_manual() {
        let s = buf(&[10, 20, 30, 40], 2, 2);
        let p = buf(&[12, 17, 30, 44], 2, 2);
        // |10-12|+|20-17|+|30-30|+|40-44| = 2+3+0+4 = 9
        assert_eq!(sad_f32(&s, 2, &p, 2, 2, 2), 9);
    }

    #[test]
    fn scaled_residual_dispatch_matches_scalar() {
        let (width, height) = (16, 8);
        let src_stride = 19;
        let pred_stride = 17;
        let src: Vec<f32> = (0..src_stride * height).map(|i| (i % 251) as f32).collect();
        let pred: Vec<f32> = (0..pred_stride * height)
            .map(|i| ((i * 7) % 251) as f32)
            .collect();
        let mut scalar = vec![0.0; width * height];
        let mut dispatched = vec![0.0; width * height];
        let spec = ResidualSpec {
            src_stride,
            pred_stride,
            width,
            height,
            scale: 0.125,
        };
        scaled_residual_f32_scalar(&mut scalar, &src, &pred, spec);
        scaled_residual_f32(&mut dispatched, &src, &pred, spec);
        assert_eq!(dispatched, scalar);
    }

    #[test]
    fn prediction_conversion_helpers_match_scalar_reference() {
        let spec = ResidualSpec {
            src_stride: 4,
            pred_stride: 3,
            width: 2,
            height: 2,
            scale: 0.5,
        };
        let src = [10.0, 20.0, 0.0, 0.0, 30.0, 40.0, 0.0, 0.0];
        let prediction_u16 = [8u16, 18, 0, 28, 38, 0];
        let mut prediction_f32 = [0.0; 4];
        let mut residual = [0.0; 4];
        u16_prediction_and_scaled_residual_f32(
            &mut prediction_f32,
            &mut residual,
            &src,
            &prediction_u16,
            spec,
        );
        assert_eq!(prediction_f32, [8.0, 18.0, 28.0, 38.0]);
        assert_eq!(residual, [1.0; 4]);

        let prediction = [8.25, 18.75, 0.0, 28.25, 38.75, 0.0];
        let mut prediction_i32 = [0; 4];
        f32_prediction_and_scaled_residual_i32(
            &mut prediction_i32,
            &mut residual,
            &src,
            &prediction,
            spec,
        );
        assert_eq!(prediction_i32, [8, 19, 28, 39]);
        assert_eq!(residual, [0.875, 0.625, 0.875, 0.625]);
    }

    #[test]
    fn satd_zero_on_equal() {
        let s = buf(&(0..16).collect::<Vec<_>>(), 4, 4);
        assert_eq!(satd_f32(&s, 4, &s, 4, 4, 4), 0);
    }

    #[test]
    fn satd_dc_only() {
        // A constant residual of +5 over 4x4: only the DC Hadamard coefficient is
        // nonzero. had4 has gain 2 per pass -> DC = 5 * 4 (spatial sum halves...) ;
        // rather than hard-code the scale, assert it equals the closed form:
        // DC coeff = sum(res) with the 4x4 Hadamard's (2*2) row/col gain = 16*... .
        let s = buf(&[100; 16], 4, 4);
        let p = buf(&[95; 16], 4, 4);
        // Only DC survives; magnitude = |sum(d)| * (hadamard gain). d=5 each, 16 of
        // them; had4 doubles per pass (x4 total), so DC = 5*16 ... but the energy is
        // concentrated in one coeff = 5 * 16 (spatial) * ... Assert via scalar ref.
        assert_eq!(
            satd_f32(&s, 4, &p, 4, 4, 4),
            satd_f32_scalar(&s, 4, &p, 4, 4, 4)
        );
        // A pure DC residual has all AC Hadamard coeffs zero, so SATD == |DC|.
        // DC = |Σd| with the transform's constant gain 16 (2 per 1-D pass, 4x4).
        assert_eq!(satd_f32_scalar(&s, 4, &p, 4, 4, 4), 5 * 16);
    }

    #[test]
    fn satd_ge_sad_scaled() {
        // Parseval: Σ|Hadamard| >= |DC| and total energy is preserved up to the
        // constant gain, so SATD should never be zero when SAD is nonzero.
        let s = buf(&[3, 9, 1, 7, 2, 8, 4, 6, 5, 0, 3, 9, 1, 2, 3, 4], 4, 4);
        let p = buf(&[0; 16], 4, 4);
        assert!(satd_f32(&s, 4, &p, 4, 4, 4) > 0);
        assert_eq!(
            sad_f32(&s, 4, &p, 4, 4, 4),
            3 + 9 + 1 + 7 + 2 + 8 + 4 + 6 + 5 + 3 + 9 + 1 + 2 + 3 + 4
        );
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_matches_scalar() {
        // Deterministic pseudo-random blocks at a few sizes.
        let mut seed = 0x1234_5678u32;
        let mut rng = || {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            ((seed >> 16) & 0xff) as i32
        };
        for &(w, h) in &[(4, 4), (8, 8), (16, 16), (32, 32), (8, 4), (4, 8)] {
            let sstride = w + 3;
            let pstride = w + 1;
            let src: Vec<f32> = (0..sstride * h).map(|_| rng() as f32).collect();
            let pred: Vec<f32> = (0..pstride * h).map(|_| rng() as f32).collect();
            assert_eq!(
                sad_f32_scalar(&src, sstride, &pred, pstride, w, h),
                unsafe { crate::av2::neon::sad_f32_neon(&src, sstride, &pred, pstride, w, h) },
                "sad {w}x{h}"
            );
            assert_eq!(
                satd_f32_scalar(&src, sstride, &pred, pstride, w, h),
                unsafe { crate::av2::neon::satd_f32_neon(&src, sstride, &pred, pstride, w, h) },
                "satd {w}x{h}"
            );
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_motion_metrics_match_scalar() {
        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let width = 32;
        let height = 16;
        let src_stride = width + 3;
        let pred_stride = width + 5;
        let src: Vec<f32> = (0..src_stride * height)
            .map(|i| ((i * 37 + i / src_stride * 11) & 1023) as f32)
            .collect();
        let pred: Vec<f32> = (0..pred_stride * height)
            .map(|i| ((i * 19 + i / pred_stride * 7) & 1023) as f32)
            .collect();
        assert_eq!(
            sad_f32_scalar(&src, src_stride, &pred, pred_stride, width, height),
            unsafe {
                crate::av2::avx::sad_f32_avx2(&src, src_stride, &pred, pred_stride, width, height)
            }
        );
        assert_eq!(
            satd_f32_scalar(&src, src_stride, &pred, pred_stride, width, height),
            unsafe {
                crate::av2::avx::satd_f32_avx2(&src, src_stride, &pred, pred_stride, width, height)
            }
        );
    }
}
