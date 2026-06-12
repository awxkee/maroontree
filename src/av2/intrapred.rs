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

/// avm `divide_round(value, bits) = (value + (1<<(bits-1))) >> bits` with the
/// arithmetic right shift C uses for signed values.
#[inline]
fn divide_round(value: i32, bits: i32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
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

const BLEND_WEIGHT_MAX: i32 = 32;

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

/// AV2 SMOOTH predictor (the two-axis blend, not the simple AV1 version) for a
/// `bs`x`bs` block. `above[bs]` = top-right, `left[bs]` = bottom-left.
pub(crate) fn smooth(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let bl = left[bs];
    let tr = above[bs];
    let log2_h = blk_size_log2(bs);
    let log2_w = blk_size_log2(bs);
    // scale = ROUND_POWER_OF_TWO((log2(bh)-2 + log2(bw)-2), 2)
    let scale = ((log2_h - 2 + log2_w - 2) + 2) >> 2;
    let blend_max_log2 = blk_size_log2(BLEND_WEIGHT_MAX as usize); // 5
    let clamp_log2 = blk_size_log2((BLEND_WEIGHT_MAX << 1) as usize); // log2(64)=6
    let mut out = vec![0f32; bs * bs];
    for (r, &l) in left[..bs].iter().enumerate() {
        let s_top = BLEND_WEIGHT_MAX >> clamp_log2.min(((r as i32) << 1) >> scale);
        let row = &mut out[r * bs..r * bs + bs];
        for (c, (dst, &top)) in row[..bs].iter_mut().zip(above[..bs].iter()).enumerate() {
            let s_left = BLEND_WEIGHT_MAX >> clamp_log2.min(((c as i32) << 1) >> scale);
            let mut predv = bl + divide_round((top - bl) * (bs as i32 - 1 - r as i32), log2_h);
            let mut predh = tr + divide_round((l - tr) * (bs as i32 - 1 - c as i32), log2_w);
            predv += divide_round((top - predv) * s_top, blend_max_log2 + 1);
            predh += divide_round((l - predh) * s_left, blend_max_log2 + 1);
            *dst = divide_round(predv + predh, 1) as f32;
        }
    }
    out
}

/// AVM SMOOTH_V predictor: the vertical half of [`smooth`]. Each sample interpolates
/// from the top row (r=0) toward the bottom-left sample (r=bs-1), with the same
/// secondary top-blend as SMOOTH. `above[bs..]`/`left[bs]` carry top-right/bottom-left.
pub(crate) fn smooth_v(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let bl = left[bs];
    let log2_h = blk_size_log2(bs);
    let log2_w = blk_size_log2(bs);
    let scale = ((log2_h - 2 + log2_w - 2) + 2) >> 2;
    let blend_max_log2 = blk_size_log2(BLEND_WEIGHT_MAX as usize); // 5
    let clamp_log2 = blk_size_log2((BLEND_WEIGHT_MAX << 1) as usize); // 6
    let mut out = vec![0f32; bs * bs];
    for r in 0..bs {
        let s_top = BLEND_WEIGHT_MAX >> clamp_log2.min(((r as i32) << 1) >> scale);
        let row = &mut out[r * bs..r * bs + bs];
        for (&top, out) in above[..bs].iter().zip(row[..bs].iter_mut()) {
            let mut predv = bl + divide_round((top - bl) * (bs as i32 - 1 - r as i32), log2_h);
            predv += divide_round((top - predv) * s_top, blend_max_log2 + 1);
            *out = predv as f32;
        }
    }
    out
}

/// AVM SMOOTH_H predictor: the horizontal half of [`smooth`]. Each sample interpolates
/// from the left column (c=0) toward the top-right sample (c=bs-1), with the same
/// secondary left-blend as SMOOTH.
pub(crate) fn smooth_h(bs: usize, above: &[i32], left: &[i32]) -> Vec<f32> {
    let tr = above[bs];
    let log2_h = blk_size_log2(bs);
    let log2_w = blk_size_log2(bs);
    let scale = ((log2_h - 2 + log2_w - 2) + 2) >> 2;
    let blend_max_log2 = blk_size_log2(BLEND_WEIGHT_MAX as usize); // 5
    let clamp_log2 = blk_size_log2((BLEND_WEIGHT_MAX << 1) as usize); // 6
    let mut out = vec![0f32; bs * bs];
    for r in 0..bs {
        let l = left[r];
        let row = &mut out[r * bs..r * bs + bs];
        for (c, out) in row[..bs].iter_mut().enumerate() {
            let s_left = BLEND_WEIGHT_MAX >> clamp_log2.min(((c as i32) << 1) >> scale);
            let mut predh = tr + divide_round((l - tr) * (bs as i32 - 1 - c as i32), log2_w);
            predh += divide_round((l - predh) * s_left, blend_max_log2 + 1);
            *out = predh as f32;
        }
    }
    out
}

/// Build avm reference samples for a square `bs` luma block at `(y0,x0)` in the
/// reconstructed plane `rec` (stride `pw`). Returns `(above, left, corner)` with
/// `above`/`left` of length `2*bs`. Availability flags and pixel counts come
/// from the caller (frame/SB geometry). Mirrors
/// `av2_build_intra_predictors_high_default` for NEED_ABOVE|NEED_LEFT|
/// NEED_ABOVELEFT (+ optional top-right / bottom-left for SMOOTH).
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
