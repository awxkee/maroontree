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

pub(crate) const LUMA_SATD_SCALE: f32 = 1.25;

#[inline(always)]
fn had4(a: i32, b: i32, c: i32, d: i32) -> [i32; 4] {
    let (e, f, g, h) = (a + c, a - c, b + d, b - d);
    [e + g, f + h, f - h, e - g]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn luma_satd(
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
    qstep: f32,
    recon_at: impl Fn(usize) -> i32,
) -> f32 {
    debug_assert!(width * height <= 32 * 32);
    debug_assert!(width.is_multiple_of(4) && height.is_multiple_of(4));
    let max_value = (1 << bit_depth) - 1;
    let mut residual = [0i32; 32 * 32];
    for y in 0..height {
        let src_row = &src[(py + y) * stride + px..(py + y) * stride + px + width];
        for (x, &s) in src_row.iter().enumerate() {
            let i = y * width + x;
            residual[i] = s - recon_at(i).clamp(0, max_value);
        }
    }

    let mut satd = 0u64;
    for ty in (0..height).step_by(4) {
        for tx in (0..width).step_by(4) {
            let mut rows = [[0i32; 4]; 4];
            for y in 0..4 {
                let d = &residual[(ty + y) * width + tx..];
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

    // The unnormalized 4x4 Hadamard has roughly four times SAD's coefficient
    // scale. qstep maps the L1-domain error onto the existing SSE/rate axis.
    satd as f32 * qstep.max(1.0) * 0.25 * LUMA_SATD_SCALE
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn chroma_sse(
    src: &[i32],
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
            let delta = (s - recon_at(i).clamp(0, max_value)) as i64;
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
        let src: Vec<i32> = (0..64).map(|i| i * 3).collect();
        assert_eq!(luma_satd(&src, 8, 0, 0, 8, 8, 8, 32.0, |i| src[i]), 0.0);
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
