//! Non-directional intra predictors (PAETH, SMOOTH) and reference-sample
//! construction, ported bit-exact from avm `avm_dsp/intrapred.c` and
//! `av2/common/reconintra.c` (`av2_build_intra_predictors_high_default`,
//! 8-bit / mrl_index 0 path). Only square luma blocks are used here.
//!
//! Reference layout: `above[0..bs)` = top row, `above[bs..2bs)` = top-right,
//! `left[0..bs)` = left column, `left[bs..2bs)` = bottom-left, plus a separate
//! `corner` (= avm `above_row[-1]`). PAETH reads `above[0..bs)`, `left[0..bs)`,
//! `corner`; SMOOTH additionally reads `above[bs]` (top-right "tr") and
//! `left[bs]` (bottom-left "bl").
//!
//! SMOOTH/SMOOTH_V/SMOOTH_H are expressed in dav2d's `ipred_smooth*` table-MAC
//! form (`SM_WEIGHTS` lookup + `pred += ((s - pred) * w + 32) >> 6`), which is
//! bit-identical to the previous computed-weight formulation for every square
//! size, and carries an aarch64 NEON path (4 columns/iteration, `vmlaq_s32`
//! fused multiply-accumulate) that is bit-exact to the scalar core.

/// `blk_size_log2[n]` = log2(n) for the power-of-two block dimensions avm uses.
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

/// dav2d `sm_weights[scale][i] = 32 >> min(6, (i<<2) >> scale)` (AV2/AVM SMOOTH
/// blend weights). For the square luma blocks used here `scale = (n_pel>=64) +
/// (n_pel>512)`. Equivalent to the previous `32 >> min(6, (i<<1) >> scale')`
/// formula for every square size, just expressed as dav2d's lookup table.
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

/// AV2/AVM SMOOTH: two-axis blend of a vertical gradient (toward bottom-left)
/// and a horizontal gradient (toward top-right), each refined by the
/// boundary-weighted MAC `pred += ((sample - pred) * w + 32) >> 6`. Bit-exact to
/// dav2d `ipred_smooth_c`. `above[bs]` = top-right, `left[bs]` = bottom-left.
fn smooth_scalar(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let log2 = blk_size_log2(bs);
    let rnd = (bs as i32) >> 1;
    let w = &SM_WEIGHTS[smooth_scale(bs)];
    let (right, bottom) = (above[bs], left[bs]);
    let mut out = vec![0f32; bs * bs];
    for (y, &l) in left[..bs].iter().enumerate() {
        let (diff_hor, off_ver, w_ver) = (l - right, bs as i32 - 1 - y as i32, w[y]);
        let row = &mut out[y * bs..y * bs + bs];
        for (x, (dst, &above_x)) in row.iter_mut().zip(&above[..bs]).enumerate() {
            let mut pred_ver = bottom + (((above_x - bottom) * off_ver + rnd) >> log2);
            let mut pred_hor = right + ((diff_hor * (bs as i32 - 1 - x as i32) + rnd) >> log2);
            pred_ver += ((above_x - pred_ver) * w_ver + 32) >> 6;
            pred_hor += ((l - pred_hor) * w[x] + 32) >> 6;
            *dst = ((pred_ver + pred_hor + 1) >> 1) as f32;
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
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bs.is_multiple_of(4) {
            return unsafe { neon::smooth(bs, above, left) };
        }
    }
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

/// NEON+MAC SMOOTH kernels. Each iteration predicts 4 columns with two fused
/// multiply-accumulates per axis: `vmlaq_s32(acc, v, vdupq_n_s32(k))` for the
/// scalar-weighted vertical gradient and top-blend, and `vmlaq_s32(acc, v, w)`
/// for the per-column left-weight blend, mirroring dav2d's `ipred_smooth*`
/// arithmetic exactly. Right-shifts use `vshlq_s32` by a negative count (signed
/// arithmetic shift, matching `>>` on `i32`).
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

    #[inline]
    #[target_feature(enable = "neon")]
    pub(crate) fn smooth(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
        let log2 = 31 - (bs as u32).leading_zeros() as i32;
        let rnd = bs as i32 >> 1;
        let w = &SM_WEIGHTS[smooth_scale(bs)];
        let (right, bottom) = (above[bs], left[bs]);
        let mut out = vec![0f32; bs * bs];
        let (rb, bb, r32, rndb) = (
            vdupq_n_s32(right),
            vdupq_n_s32(bottom),
            vdupq_n_s32(32),
            vdupq_n_s32(rnd),
        );
        for (y, &l) in left[..bs].iter().enumerate() {
            let (diff_hor, off_ver, w_ver) = (l - right, bs as i32 - 1 - y as i32, w[y]);
            let lb = vdupq_n_s32(l);
            let row = &mut out[y * bs..y * bs + bs];
            let mut x = 0;
            while x < bs {
                unsafe {
                    let av = vld1q_s32(above[x..].as_ptr());
                    let wx = vld1q_s32(w[x..].as_ptr());
                    let xc = xcoef(bs as i32 - 1 - x as i32);
                    // pred_ver = bottom + ((av - bottom) * off_ver + rnd) >> log2
                    let mut pv = vaddq_s32(bb, shr(mla_n(rndb, vsubq_s32(av, bb), off_ver), log2));
                    // pred_ver += ((av - pred_ver) * w_ver + 32) >> 6
                    pv = vaddq_s32(pv, shr(mla_n(r32, vsubq_s32(av, pv), w_ver), 6));
                    // pred_hor = right + (diff_hor * xc + rnd) >> log2
                    let mut ph = vaddq_s32(rb, shr(mla_n(rndb, xc, diff_hor), log2));
                    // pred_hor += ((l - pred_hor) * w[x] + 32) >> 6   (per-column weights)
                    ph = vaddq_s32(ph, shr(vmlaq_s32(r32, vsubq_s32(lb, ph), wx), 6));
                    // out = (pred_ver + pred_hor + 1) >> 1
                    store(
                        shr(vaddq_s32(vaddq_s32(pv, ph), vdupq_n_s32(1)), 1),
                        &mut row[x..],
                    );
                }
                x += 4;
            }
        }
        out
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
) -> (Vec<i32>, Vec<i32>, i32) {
    const BASE: i32 = 128; // 128 << (bd-8), bd=8
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
            *s = BASE - 1;
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
            *s = BASE + 1;
        }
    }

    let corner = if n_top > 0 && n_left > 0 {
        g(y0 - 1, x0 - 1)
    } else if n_top > 0 {
        g(y0 - 1, x0)
    } else if n_left > 0 {
        g(y0, x0 - 1)
    } else {
        BASE
    };

    (above, left, corner)
}
