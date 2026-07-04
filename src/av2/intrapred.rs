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

use crate::util::FastRound;

#[inline]
fn blk_size_log2(n: usize) -> i32 {
    match n {
        4 => 2,
        8 => 3,
        16 => 4,
        32 => 5,
        64 => 6,
        _ => (usize::BITS - 1 - n.leading_zeros()) as i32,
    }
}

#[inline]
fn abs_diff(a: i32, b: i32) -> i32 {
    if a > b { a - b } else { b - a }
}

/// avm `paeth_predictor_single`: nearest of {left, top, top_left} to
/// `top + left - top_left`, with the left/top/top_left tie order.
#[inline]
fn paeth_single(left: i32, top: i32, top_left: i32) -> i32 {
    let base = top + left - top_left;
    let p_left = abs_diff(base, left);
    let p_top = abs_diff(base, top);
    let p_top_left = abs_diff(base, top_left);
    if p_left <= p_top && p_left <= p_top_left {
        left
    } else if p_top <= p_top_left {
        top
    } else {
        top_left
    }
}

/// PAETH predictor for a `bs`x`bs` block → row-major `bs*bs` f32 samples.
pub(crate) fn paeth(bs: usize, above: &[i32], left: &[i32], corner: i32) -> Vec<f32> {
    let mut out = vec![0f32; bs * bs];
    for (r, &l) in left[..bs].iter().enumerate() {
        let row = &mut out[r * bs..r * bs + bs];
        for (dst, &top) in row[..bs].iter_mut().zip(above[..bs].iter()) {
            *dst = paeth_single(l, top, corner) as f32;
        }
    }
    out
}

#[rustfmt::skip]
static SM_WEIGHTS: [[i32; 64]; 3] = [
    [32, 8, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [32, 32, 16, 16, 8, 8, 4, 4, 2, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

#[inline]
fn smooth_scale(bs: usize) -> usize {
    let n_pel = bs * bs;
    (n_pel >= 64) as usize + (n_pel > 512) as usize
}

fn smooth_scalar(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    // AV2 highbd_smooth_predictor (avm_dsp/intrapred.c).
    let bl = left[bs];
    let tr = above[bs];
    let lg = blk_size_log2(bs);
    let scale = ((lg - 2) + (lg - 2) + 2) >> 2;
    let blend_max_log2 = blk_size_log2(32);
    let dr = |v: i32, bits: i32| (v + (1 << (bits - 1))) >> bits;
    let mut out = vec![0f32; bs * bs];
    for r in 0..bs {
        let s_top = 32 >> blk_size_log2(64).min(((r as i32) << 1) >> scale);
        let l = left[r];
        for c in 0..bs {
            let s_left = 32 >> blk_size_log2(64).min(((c as i32) << 1) >> scale);
            let top = above[c];
            let mut predv = bl + dr((top - bl) * (bs as i32 - 1 - r as i32), lg);
            let mut predh = tr + dr((l - tr) * (bs as i32 - 1 - c as i32), lg);
            predv += dr((top - predv) * s_top, blend_max_log2 + 1);
            predh += dr((l - predh) * s_left, blend_max_log2 + 1);
            out[r * bs + c] = dr(predv + predh, 1) as f32;
        }
    }
    out
}

/// AV2/AVM SMOOTH_V: vertical half of [`smooth`]. Bit-exact to dav2d
/// `ipred_smooth_v_c`. `left[bs]` = bottom-left.
fn smooth_v_scalar(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let log2 = blk_size_log2(bs);
    let rnd = (bs as i32) >> 1;
    let w = &SM_WEIGHTS[smooth_scale(bs)];
    let bottom = left[bs];
    let mut out = vec![0f32; bs * bs];
    for y in 0..bs {
        let (off, w_ver) = (bs as i32 - 1 - y as i32, w[y]);
        let row = &mut out[y * bs..y * bs + bs];
        for (dst, &above_x) in row.iter_mut().zip(&above[..bs]) {
            let pred = bottom + (((above_x - bottom) * off + rnd) >> log2);
            *dst = (pred + (((above_x - pred) * w_ver + 32) >> 6)) as f32;
        }
    }
    out
}

/// AV2/AVM SMOOTH_H: horizontal half of [`smooth`]. Bit-exact to dav2d
/// `ipred_smooth_h_c`. `above[bs]` = top-right.
fn smooth_h_scalar(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let log2 = blk_size_log2(bs);
    let rnd = (bs as i32) >> 1;
    let w = &SM_WEIGHTS[smooth_scale(bs)];
    let right = above[bs];
    let mut out = vec![0f32; bs * bs];
    for (y, &l) in left[..bs].iter().enumerate() {
        let diff = l - right;
        let row = &mut out[y * bs..y * bs + bs];
        for (x, dst) in row.iter_mut().enumerate() {
            let pred = right + ((diff * (bs as i32 - 1 - x as i32) + rnd) >> log2);
            *dst = (pred + (((l - pred) * w[x] + 32) >> 6)) as f32;
        }
    }
    out
}

/// Public SMOOTH dispatch: NEON (4-lane, MAC) on aarch64, scalar elsewhere. The
/// NEON kernel is bit-exact to [`smooth_scalar`] (validated lane-for-lane).
pub(crate) fn smooth(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    smooth_scalar(bs, above, left)
}

/// Public SMOOTH_V dispatch (see [`smooth`]).
pub(crate) fn smooth_v(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bs.is_multiple_of(4) {
            return unsafe { neon::smooth_v(bs, above, left) };
        }
    }
    smooth_v_scalar(bs, above, left)
}

/// Public SMOOTH_H dispatch (see [`smooth`]).
pub(crate) fn smooth_h(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bs.is_multiple_of(4) {
            return unsafe { neon::smooth_h(bs, above, left) };
        }
    }
    smooth_h_scalar(bs, above, left)
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::{SM_WEIGHTS, smooth_scale};
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    fn mla_n(acc: int32x4_t, v: int32x4_t, k: i32) -> int32x4_t {
        vmlaq_s32(acc, v, vdupq_n_s32(k))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn shr(v: int32x4_t, n: i32) -> int32x4_t {
        vshlq_s32(v, vdupq_n_s32(-n))
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn store(v: int32x4_t, dst: &mut [f32]) {
        unsafe {
            vst1q_f32(dst.as_mut_ptr(), vcvtq_f32_s32(v));
        }
    }

    #[inline]
    #[target_feature(enable = "neon")]
    fn xcoef(base: i32) -> int32x4_t {
        static IDX: [i32; 4] = [0, 1, 2, 3];
        unsafe { vsubq_s32(vdupq_n_s32(base), vld1q_s32(IDX.as_ptr())) }
    }

    #[target_feature(enable = "neon")]
    pub(crate) fn smooth_v(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
        let log2 = 31 - (bs as u32).leading_zeros() as i32;
        let rnd = bs as i32 >> 1;
        let w = &SM_WEIGHTS[smooth_scale(bs)];
        let bottom = left[bs];
        let mut out = vec![0f32; bs * bs];
        let (bb, r32, rndb) = (vdupq_n_s32(bottom), vdupq_n_s32(32), vdupq_n_s32(rnd));
        for y in 0..bs {
            let (off, w_ver) = (bs as i32 - 1 - y as i32, w[y]);
            let row = &mut out[y * bs..y * bs + bs];
            let mut x = 0;
            while x < bs {
                unsafe {
                    let av = vld1q_s32(above[x..].as_ptr());
                    let mut pred = vaddq_s32(bb, shr(mla_n(rndb, vsubq_s32(av, bb), off), log2));
                    pred = vaddq_s32(pred, shr(mla_n(r32, vsubq_s32(av, pred), w_ver), 6));
                    store(pred, &mut row[x..]);
                    x += 4;
                }
            }
        }
        out
    }

    #[target_feature(enable = "neon")]
    pub(crate) fn smooth_h(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
        let log2 = 31 - (bs as u32).leading_zeros() as i32;
        let rnd = bs as i32 >> 1;
        let w = &SM_WEIGHTS[smooth_scale(bs)];
        let right = above[bs];
        let mut out = vec![0f32; bs * bs];
        let (rb, r32, rndb) = (vdupq_n_s32(right), vdupq_n_s32(32), vdupq_n_s32(rnd));
        for (y, &l) in left[..bs].iter().enumerate() {
            let diff = l - right;
            let lb = vdupq_n_s32(l);
            let row = &mut out[y * bs..y * bs + bs];
            let mut x = 0;
            while x < bs {
                unsafe {
                    let wx = vld1q_s32(w[x..].as_ptr());
                    let xc = xcoef(bs as i32 - 1 - x as i32);
                    let mut pred = vaddq_s32(rb, shr(mla_n(rndb, xc, diff), log2));
                    pred = vaddq_s32(pred, shr(vmlaq_s32(r32, vsubq_s32(lb, pred), wx), 6));
                    store(pred, &mut row[x..]);
                }
                x += 4;
            }
        }
        out
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn build_refs(
    rec: &[f32],
    pw: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    have_above: bool,
    have_left: bool,
    tr_px: usize, // available top-right pixels (0 = none)
    bl_px: usize, // available bottom-left pixels (0 = none)
    neutral: f32,
) -> (Vec<i32>, Vec<i32>, i32) {
    // AVM initializes missing intra references to the mid-sample value for
    // the coded bit depth: 128 for 8-bit, 512 for 10-bit, 2048 for 12-bit.
    // Using the 8-bit constant for high-bit-depth streams makes the encoder
    // subtract too-small predictors. The decoder then adds those residuals
    // to its correct high-bit-depth predictor, producing bright rectangular
    // blocks along frame/tile/partition edges.
    let base: i32 = neutral.fast_round() as i32;
    let g = |y: usize, x: usize| (rec[y * pw + x] + 0.5) as i32;
    let mut above = vec![0i32; 2 * bs];
    let mut left = vec![0i32; 2 * bs];
    let n_top = if have_above { bs } else { 0 };
    let n_left = if have_left { bs } else { 0 };

    // NEED_ABOVE
    if n_top > 0 {
        for (c, dst) in above[0..bs].iter_mut().enumerate() {
            *dst = g(y0 - 1, x0 + c);
        }
        let tr = tr_px.min(bs);
        for c in 0..tr {
            above[bs + c] = g(y0 - 1, x0 + bs + c);
        }
        let repeat_val = if tr > 0 {
            above[bs + tr - 1]
        } else {
            above[bs - 1]
        };
        for s in above.iter_mut().take(2 * bs).skip(bs + tr) {
            *s = repeat_val;
        }
    } else if n_left > 0 {
        let v = g(y0, x0 - 1);
        for s in above.iter_mut() {
            *s = v;
        }
    } else {
        for s in above.iter_mut() {
            *s = base - 1;
        }
    }

    // NEED_LEFT
    if n_left > 0 {
        for (r, dst) in left[..bs].iter_mut().enumerate() {
            *dst = g(y0 + r, x0 - 1);
        }
        let bln = bl_px.min(bs);
        for (r, dst) in left[bs..bs + bln].iter_mut().enumerate() {
            *dst = g(y0 + bs + r, x0 - 1);
        }
        let fill_from = bs + bln;
        let repeat_val = if bln > 0 {
            left[bs + bln - 1]
        } else {
            left[bs - 1]
        };
        for s in left.iter_mut().take(2 * bs).skip(fill_from) {
            *s = repeat_val;
        }
    } else if n_top > 0 {
        let v = g(y0 - 1, x0);
        for s in left.iter_mut() {
            *s = v;
        }
    } else {
        for s in left.iter_mut() {
            *s = base + 1;
        }
    }

    let corner = if n_top > 0 && n_left > 0 {
        g(y0 - 1, x0 - 1)
    } else if n_top > 0 {
        g(y0 - 1, x0)
    } else if n_left > 0 {
        g(y0, x0 - 1)
    } else {
        base
    };

    (above, left, corner)
}
