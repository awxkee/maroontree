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

pub(crate) fn luma_satd_scale(base_q_idx: u8, subsampled: bool) -> f32 {
    let t = crate::tuning::get();
    let base: f32 = t.satd_base;
    // 4:2:0/4:2:2 (and mono, passed as subsampled): flat base — the ramp
    // measured +1.22% on the 420 top grid (x_volcanic +7.3). The top-band
    // finer-partition preference is a 4:4:4 phenomenon.
    if subsampled {
        return base;
    }
    let top: f32 = t.satd_top;
    let (lo, hi) = (t.satd_knee_lo, t.satd_knee_hi);
    let qi = base_q_idx as f32;
    if qi <= lo {
        top
    } else if qi >= hi {
        base
    } else {
        top + (base - top) * (qi - lo) / (hi - lo).max(1.0)
    }
}

#[inline(always)]
fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
    let (e, f, g, h) = (a + c, a - c, b + d, b - d);
    [e + g, f + h, f - h, e - g]
}

/// Static luma partition SATD kernel.
///
/// `pred` and `residual` are packed `width * height` blocks. An empty `pred`
/// selects the constant `dc` predictor; an empty `residual` means prediction-
/// only distortion. This representation avoids a closure in the hot loop and
/// maps directly to the SIMD implementations.
#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_satd_scalar(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> u64 {
    debug_assert!(width * height <= 32 * 32);
    debug_assert!(width.is_multiple_of(4) && height.is_multiple_of(4));
    debug_assert!(pred.is_empty() || pred.len() >= width * height);
    debug_assert!(residual.is_empty() || residual.len() >= width * height);

    let mut error = [0i32; 32 * 32];
    for y in 0..height {
        let src_row = &src[(py + y) * stride + px..(py + y) * stride + px + width];
        for (x, &s) in src_row.iter().enumerate() {
            let i = y * width + x;
            let prediction = if pred.is_empty() { dc } else { pred[i] };
            let reconstruction = if residual.is_empty() {
                prediction
            } else {
                prediction + residual[i]
            };
            error[i] = i32::from(s) - reconstruction.clamp(0, max_value);
        }
    }

    let mut satd = 0u64;
    for ty in (0..height).step_by(4) {
        for tx in (0..width).step_by(4) {
            let mut rows = [[0i32; 4]; 4];
            for y in 0..4 {
                let d = &error[(ty + y) * width + tx..];
                rows[y] = had4(d[0], d[1], d[2], d[3]);
            }
            // Transposed access: `x` walks the columns of each row.
            #[allow(clippy::needless_range_loop)]
            for x in 0..4 {
                for value in had4(rows[0][x], rows[1][x], rows[2][x], rows[3][x]) {
                    satd += value.unsigned_abs() as u64;
                }
            }
        }
    }

    satd
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_sse<P: crate::intrapred::Pel>(
    src: &[P],
    stride: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    recon_at: impl Fn(usize) -> i32,
) -> f32 {
    let max_value = (1 << bit_depth) - 1;
    let mut sse = 0i64;
    for y in 0..height {
        let src_row = &src[(py + y) * stride + px..(py + y) * stride + px + width];
        for (x, &s) in src_row.iter().enumerate() {
            let i = y * width + x;
            let delta = (s.widen() - recon_at(i).clamp(0, max_value)) as i64;
            sse += delta * delta;
        }
    }
    sse as f32
}

#[inline]
pub(crate) fn rd_cost(distortion: f32, lambda: f32, rate: f32) -> f32 {
    distortion + lambda * rate
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_satd_is_zero_for_exact_reconstruction() {
        let src: Vec<u16> = (0..64).map(|i| i * 3).collect();
        let pred: Vec<i32> = src.iter().map(|&value| i32::from(value)).collect();
        assert_eq!(luma_satd_scalar(&src, 8, 0, 0, 8, 8, 255, &pred, 0, &[]), 0);
    }

    #[test]
    fn chroma_sse_matches_manual_error() {
        let src = vec![10, 20, 30, 40];
        assert_eq!(
            chroma_sse(&src, 2, 0, 0, 2, 2, 8, |i| [12, 17, 30, 44][i]),
            29.0
        );
    }
}
