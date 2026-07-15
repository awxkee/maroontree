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
use crate::av2::cdfs_qctx::{CHROMA_SKIP_TX32_QC, SKIP_TX8_QC, SKIP_TX16_QC};
use crate::av2::coder::Coeff;
use crate::av2::lossless::levels_to_coeffs_4x4;
use crate::av2::wht::fwht4x4;
use std::sync::OnceLock;

#[allow(clippy::too_many_arguments)]
pub(crate) fn sb_tu_contexts(
    tus: &[Vec<Coeff>; 4],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    qc: usize,
    mc: i64,
    mr: i64,
) -> ([u32; 4], [usize; 4]) {
    let pos = [(0usize, 0usize), (0, 32), (32, 0), (32, 32)];
    let mut skip_cdfs = [0u32; 4];
    let mut dc_sign_ctxs = [0usize; 4];
    for i in 0..4 {
        let (ty, tx) = pos[i];
        let cy = (sb_y + ty) / 4;
        let cx = (sb_x + tx) / 4;
        let a = &above[cx..cx + 8];
        let l = &left[cy..cy + 8];
        let merge_a = a.iter().fold(0u8, |acc, &b| acc | b);
        let merge_l = l.iter().fold(0u8, |acc, &b| acc | b);
        let sctx = (((merge_a & 0x3F).min(4) + (merge_l & 0x3F).min(4)) as usize + 3) >> 1;
        let dcs: i32 = a.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>()
            + l.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>();
        let sgn = dcs - 8 - 8;
        skip_cdfs[i] = CHROMA_SKIP_TX32_QC[qc][sctx] as u32;
        dc_sign_ctxs[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
        // update context with this TU's res_ctx
        let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
        let res = if nz.is_empty() {
            0x40u8
        } else {
            let cul = (nz
                .iter()
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(7)) as u8;
            let dc = nz
                .iter()
                .find(|&&(s, _)| s == 0)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let dcbits = if dc > 0 {
                0x80
            } else if dc < 0 {
                0x00
            } else {
                0x40
            };
            (cul & 0x3F) | dcbits
        };
        // avm av2_set_entropy_contexts zeroes entropy context for out-of-frame
        // columns/rows of a partial-edge TU (in-frame → cul_level, rest → 0).
        // Replicate that here so DC-sign/skip contexts for the next TU match the
        // decoder. For SB-aligned/padded encodes mc/mr are multiples of 16 so the
        // clamp is a no-op and behavior is unchanged.
        let in_cols = (mc - cx as i64).clamp(0, 8) as usize;
        let in_rows = (mr - cy as i64).clamp(0, 8) as usize;
        let (above_in, above_pad) = above[cx..cx + 8].split_at_mut(in_cols);
        above_in.fill(res);
        above_pad.fill(0x40);
        let (left_in, left_pad) = left[cy..cy + 8].split_at_mut(in_rows);
        left_in.fill(res);
        left_pad.fill(0x40);
    }
    (skip_cdfs, dc_sign_ctxs)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn sb_tu_contexts_64x32(
    tus: &[Vec<Coeff>; 2],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    qc: usize,
    mc: i64,
    mr: i64,
) -> ([u32; 2], [usize; 2]) {
    let pos = [(0usize, 0usize), (0, 32)];
    let mut skip_cdfs = [0u32; 2];
    let mut dc_sign_ctxs = [0usize; 2];
    for i in 0..2 {
        let (ty, tx) = pos[i];
        let cy = (sb_y + ty) / 4;
        let cx = (sb_x + tx) / 4;
        let a = &above[cx..cx + 8];
        let l = &left[cy..cy + 8];
        let merge_a = a.iter().fold(0u8, |acc, &b| acc | b);
        let merge_l = l.iter().fold(0u8, |acc, &b| acc | b);
        let sctx = (((merge_a & 0x3F).min(4) + (merge_l & 0x3F).min(4)) as usize + 3) >> 1;
        let dcs: i32 = a.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>()
            + l.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>();
        let sgn = dcs - 8 - 8;
        skip_cdfs[i] = CHROMA_SKIP_TX32_QC[qc][sctx] as u32;
        dc_sign_ctxs[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
        let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
        let res = if nz.is_empty() {
            0x40u8
        } else {
            let cul = nz
                .iter()
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(7) as u8;
            let dc = nz
                .iter()
                .find(|&&(s, _)| s == 0)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let dcbits = if dc > 0 {
                0x80
            } else if dc < 0 {
                0x00
            } else {
                0x40
            };
            (cul & 0x3F) | dcbits
        };
        let in_cols = (mc - cx as i64).clamp(0, 8) as usize;
        let in_rows = (mr - cy as i64).clamp(0, 8) as usize;
        let (above_in, above_pad) = above[cx..cx + 8].split_at_mut(in_cols);
        above_in.fill(res);
        above_pad.fill(0x40);
        let (left_in, left_pad) = left[cy..cy + 8].split_at_mut(in_rows);
        left_in.fill(res);
        left_pad.fill(0x40);
    }
    (skip_cdfs, dc_sign_ctxs)
}

/// Position, quantizer and frame-edge state shared by luma transform-context
/// updates. The above/left buffers remain explicit mutable arguments so their borrow
/// is limited to one update call.
#[derive(Clone, Copy)]
pub(crate) struct TxbContextSpec {
    pub(crate) sb_y: usize,
    pub(crate) sb_x: usize,
    pub(crate) qc: usize,
    pub(crate) mi_cols: i64,
    pub(crate) mi_rows: i64,
    pub(crate) block_eq_tx: bool,
}

/// Per-TU skip / DC-sign contexts for an arbitrary list of TX_32X32 sub-TUs within
/// an SB (used by the 32X64 / 32X32 partition leaves). `pos` holds the SB-relative
/// pixel offsets in coding order; updates `above`/`left` coeff-context arrays with
/// the same edge-clamp as `sb_tu_contexts`.
pub(crate) fn sb_tu_contexts_pos(
    pos: &[(usize, usize)],
    tus: &[Vec<Coeff>],
    above: &mut [u8],
    left: &mut [u8],
    spec: &TxbContextSpec,
) -> (Vec<u32>, Vec<usize>) {
    let TxbContextSpec {
        sb_y,
        sb_x,
        qc,
        mi_cols: mc,
        mi_rows: mr,
        block_eq_tx,
    } = *spec;
    let mut skip_cdfs = vec![0u32; pos.len()];
    let mut dc_sign_ctxs = vec![0usize; pos.len()];
    for (i, &(ty, tx)) in pos.iter().enumerate() {
        let cy = (sb_y + ty) / 4;
        let cx = (sb_x + tx) / 4;
        let a = &above[cx..cx + 8];
        let l = &left[cy..cy + 8];
        let merge_a = a.iter().fold(0u8, |acc, &b| acc | b);
        let merge_l = l.iter().fold(0u8, |acc, &b| acc | b);
        let sctx = (((merge_a & 0x3F).min(4) + (merge_l & 0x3F).min(4)) as usize + 3) >> 1;
        // avm get_txb_ctx (plane 0): a single full-block transform (block == tx)
        // forces txb_skip_ctx = 0; the neighbor skip_contexts[top][left] formula
        // applies only when block != tx (e.g. the 32X64 stacked TUs).
        let sctx = if block_eq_tx { 0 } else { sctx };
        let dcs: i32 = a.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>()
            + l.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>();
        let sgn = dcs - 8 - 8;
        skip_cdfs[i] = CHROMA_SKIP_TX32_QC[qc][sctx] as u32;
        dc_sign_ctxs[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
        let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
        let res = if nz.is_empty() {
            0x40u8
        } else {
            let cul = nz
                .iter()
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(7) as u8;
            let dc = nz
                .iter()
                .find(|&&(s, _)| s == 0)
                .map(|&(_, l)| l)
                .unwrap_or(0);
            let dcbits = if dc > 0 {
                0x80
            } else if dc < 0 {
                0x00
            } else {
                0x40
            };
            (cul & 0x3F) | dcbits
        };
        let in_cols = (mc - cx as i64).clamp(0, 8) as usize;
        let in_rows = (mr - cy as i64).clamp(0, 8) as usize;
        let (above_in, above_pad) = above[cx..cx + 8].split_at_mut(in_cols);
        above_in.fill(res);
        above_pad.fill(0x40);
        let (left_in, left_pad) = left[cy..cy + 8].split_at_mut(in_rows);
        left_in.fill(res);
        left_pad.fill(0x40);
    }
    (skip_cdfs, dc_sign_ctxs)
}

/// Context for a single rectangular luma TU (16-tap family: TX_16X64 4×16 mi,
/// TX_64X16 16×4 mi, TX_16X16 4×4 mi). `wu`/`hu` are the tx width/height in mi units
/// (tx_size_wide_unit/high_unit). Skip ctx is 0 when `block_eq_tx` (block == tx, which
/// holds for all single-TX 16-family leaves); dc_sign ctx sums neighbor sign bits over
/// the tx units. Updates `wu` above + `hu` left entries with this TU's cul/DC byte.
/// Returns `(skip_cdf, dc_sign_ctx)`.
pub(crate) fn sb_tu_contexts_rect(
    tu: &[Coeff],
    above: &mut [u8],
    left: &mut [u8],
    spec: &TxbContextSpec,
    wu: usize,
    hu: usize,
) -> (u32, usize) {
    let TxbContextSpec {
        sb_y,
        sb_x,
        qc,
        mi_cols: mc,
        mi_rows: mr,
        block_eq_tx,
    } = *spec;
    let cy = sb_y / 4;
    let cx = sb_x / 4;
    let a = &above[cx..cx + wu];
    let l = &left[cy..cy + hu];
    let sctx = if block_eq_tx {
        0
    } else {
        let merge_a = a.iter().fold(0u8, |acc, &b| acc | b);
        let merge_l = l.iter().fold(0u8, |acc, &b| acc | b);
        (((merge_a & 0x3F).min(4) + (merge_l & 0x3F).min(4)) as usize + 3) >> 1
    };
    let dcs: i32 = a.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>()
        + l.iter().map(|&b| ((b & 0xC0) >> 6) as i32).sum::<i32>();
    // Neutral sign byte (0x40) contributes 1 each; subtract the neutral baseline.
    let sgn = dcs - (wu as i32) - (hu as i32);
    // txb_skip cdf is selected by the TX's `ctx` field. Most rect luma leaves
    // (16X32/32X16/16X64/64X16) are ctx=3 → CHROMA_SKIP_TX32_QC. The 8-family rect
    // leaves (8X16/8X32/16X8/32X8, min side = 2 mi) are ctx=2 → SKIP_TX16_QC. Using the
    // wrong class still decodes the skip *bit* as 0 but diverges the arithmetic range
    // state, desyncing the following eob.
    let skip_cdf = if wu.min(hu) == 2 && wu.max(hu) == 2 {
        // 8×8 corner (TX_8X8) is ctx=1 → SKIP_TX8_QC.
        SKIP_TX8_QC[qc][sctx] as u32
    } else if (wu.min(hu) == 2 && wu.max(hu) >= 4) || (wu == 4 && hu == 4) {
        // TX_16X16 and the 16-wide rectangular family use txs_ctx=2.
        SKIP_TX16_QC[qc][sctx] as u32
    } else {
        CHROMA_SKIP_TX32_QC[qc][sctx] as u32
    };
    let dc_sign_ctx = ((sgn != 0) as usize) + ((sgn > 0) as usize);
    let nz: Vec<Coeff> = tu.iter().cloned().filter(|&(_, l)| l != 0).collect();
    let res = if nz.is_empty() {
        0x40u8
    } else {
        let cul = nz
            .iter()
            .map(|&(_, l)| l.unsigned_abs())
            .sum::<u32>()
            .min(7) as u8;
        let dc = nz
            .iter()
            .find(|&&(s, _)| s == 0)
            .map(|&(_, l)| l)
            .unwrap_or(0);
        let dcbits = if dc > 0 {
            0x80
        } else if dc < 0 {
            0x00
        } else {
            0x40
        };
        (cul & 0x3F) | dcbits
    };
    let in_cols = (mc - cx as i64).clamp(0, wu as i64) as usize;
    let in_rows = (mr - cy as i64).clamp(0, hu as i64) as usize;
    let (above_in, above_pad) = above[cx..cx + wu].split_at_mut(in_cols);
    above_in.fill(res);
    above_pad.fill(0x40);
    let (left_in, left_pad) = left[cy..cy + hu].split_at_mut(in_rows);
    left_in.fill(res);
    left_pad.fill(0x40);
    (skip_cdf, dc_sign_ctx)
}
/// must be padded up to the SB grid; replicating the last row/column keeps the
/// boundary residual small. The decoder is told the padded size and the caller
/// crops the top-left `w`x`h` region back out.
pub(crate) fn pad_plane(src: &[f32], w: usize, h: usize, pw: usize, ph: usize) -> Vec<f32> {
    if pw == w && ph == h {
        return src.to_vec();
    }
    let mut out = vec![0f32; pw * ph];
    for (y, dst_row) in out.chunks_exact_mut(pw).enumerate() {
        let src_row = &src[y.min(h - 1) * w..][..w];
        let (dst_data, dst_pad) = dst_row.split_at_mut(w);
        dst_data.copy_from_slice(src_row);
        dst_pad.fill(src_row[w - 1]);
    }
    out
}
/// SB-aligned (multiple of 64) size for a given dimension.
pub(crate) fn sb_align(n: usize) -> usize {
    n.div_ceil(64) * 64
}

/// DC prediction for a `bw`-wide × `bh`-tall block (4:2:2 chroma is 32×64).
pub(crate) fn dc_pred(rec: &[f32], w: usize, y0: usize, x0: usize, bs: usize, neutral: f32) -> f32 {
    dc_pred_bounded(rec, w, y0, x0, bs, neutral, usize::MAX, usize::MAX)
}

/// AVM `highbd_dc_predictor_subsampled` (reconintra.h): the chroma DC base for a
/// CfL block wider or taller than 32 px subsamples its reference ring by 2 (in
/// each dimension larger than 32) and divides by the *subsampled* sample count.
/// AVM guards this on `uv_mode == UV_CFL_PRED && (txw > 32 || txh > 32)`, so it is
/// the correct DC base only for the whole-64 CfL chroma path — the standard
/// (full-reference) `dc_pred` stays correct everywhere else.
///
/// The divisor uses AVM `resolve_divisor_32`, which for a power-of-two count
/// (always 32 or 64 here) reduces exactly to round-half-up `(sum + count/2) /
/// count`; the result is clipped to the pixel range like AVM's
/// `clip_pixel_highbd`. Interior-only, matching the unbounded `dc_pred` this path
/// already relied on (the whole-64 SB reads a padded, SB-aligned plane).
pub(crate) fn dc_pred_cfl_subsampled(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    neutral: f32,
    bd: i32,
) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    let ss = if bs > 32 { 2 } else { 1 };
    let mut sum: i64 = 0;
    let mut count: i64 = 0;
    if ha {
        let base = (y0 - 1) * w + x0;
        let mut i = 0;
        while i < bs {
            sum += rec[base + i] as i64;
            count += 1;
            i += ss;
        }
    }
    if hl {
        let mut i = 0;
        while i < bs {
            sum += rec[(y0 + i) * w + x0 - 1] as i64;
            count += 1;
            i += ss;
        }
    }
    if count == 0 {
        return neutral;
    }
    let dc = (sum + count / 2) / count;
    let maxv = (1i64 << bd) - 1;
    dc.clamp(0, maxv) as f32
}

/// DC predictor that limits neighbor samples to the in-frame region and replicates the
/// last available sample beyond it, matching AVM (`build_intra_predictors` uses
/// `n_left_px = min(txh, yd+txh)` / `n_top_px = min(txw, xr+txw)` then extends). `fw`/`fh`
/// are the frame width/height; pass `usize::MAX` for an interior block (no clamping).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dc_pred_bounded(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    neutral: f32,
    fw: usize,
    fh: usize,
) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    // AVM's neighbor availability uses the 8-px-aligned (MI-grid) frame extent, not the
    // exact display dimension; align fw/fh up to a multiple of 8 before clamping.
    let fh_a = if fh == usize::MAX { fh } else { (fh + 7) & !7 };
    let fw_a = if fw == usize::MAX { fw } else { (fw + 7) & !7 };
    // available in-frame samples along each edge; the rest replicate the last available.
    let avail_left = if fh_a == usize::MAX {
        bs
    } else {
        (fh_a.saturating_sub(y0)).min(bs).max(1)
    };
    let avail_above = if fw_a == usize::MAX {
        bs
    } else {
        (fw_a.saturating_sub(x0)).min(bs).max(1)
    };
    let sa: i64 = if ha {
        (0..bs)
            .map(|i| {
                let xi = x0 + i.min(avail_above - 1);
                rec[(y0 - 1) * w + xi] as i64
            })
            .sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bs)
            .map(|i| {
                let yi = y0 + i.min(avail_left - 1);
                rec[yi * w + x0 - 1] as i64
            })
            .sum()
    } else {
        0
    };
    let b = bs as i64;
    let p = if ha && hl {
        (sa + sl + b) / (2 * b)
    } else if ha {
        (sa + b / 2) / b
    } else if hl {
        (sl + b / 2) / b
    } else {
        return neutral;
    };
    p as f32
}

#[allow(dead_code)]
pub(crate) fn dc_pred_unbounded(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    neutral: f32,
) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    let sa: i64 = if ha {
        (0..bs).map(|i| rec[(y0 - 1) * w + x0 + i] as i64).sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bs).map(|i| rec[(y0 + i) * w + x0 - 1] as i64).sum()
    } else {
        0
    };
    let b = bs as i64;
    let p = if ha && hl {
        (sa + sl + b) / (2 * b)
    } else if ha {
        (sa + b / 2) / b
    } else if hl {
        (sl + b / 2) / b
    } else {
        return neutral;
    };
    p as f32
}

#[inline]
pub(crate) fn rect_rows<T>(
    plane: &[T],
    stride: usize,
    y: usize,
    x: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = &[T]> {
    plane
        .chunks_exact(stride)
        .skip(y)
        .take(height)
        .map(move |row| &row[x..x + width])
}

#[inline]
pub(crate) fn rect_rows_mut<T>(
    plane: &mut [T],
    stride: usize,
    y: usize,
    x: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = &mut [T]> {
    plane
        .chunks_exact_mut(stride)
        .skip(y)
        .take(height)
        .map(move |row| &mut row[x..x + width])
}

/// Read-only rectangular view into a strided floating-point plane.
#[derive(Clone, Copy)]
pub(crate) struct PlaneRect<'a> {
    pub(crate) plane: &'a [f32],
    pub(crate) stride: usize,
    pub(crate) y: usize,
    pub(crate) x: usize,
}

#[inline]
pub(crate) fn rect_sse_f32(
    a: &PlaneRect<'_>,
    b: &PlaneRect<'_>,
    width: usize,
    height: usize,
) -> f32 {
    let f = resolve_pixel_sse_f32();
    rect_rows(a.plane, a.stride, a.y, a.x, width, height)
        .zip(rect_rows(b.plane, b.stride, b.y, b.x, width, height))
        .map(|(a_row, b_row)| unsafe { f(a_row, b_row) })
        .sum()
}

pub(crate) fn get_residual(
    plane: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    pred: f32,
) -> Vec<f32> {
    let mut r = vec![0f32; bs * bs];
    for (dst_row, src_row) in r
        .chunks_exact_mut(bs)
        .zip(rect_rows(plane, w, y0, x0, bs, bs))
    {
        for (dst, &src) in dst_row.iter_mut().zip(src_row) {
            *dst = src - pred;
        }
    }
    r
}
pub(crate) fn put_block(plane: &mut [f32], w: usize, y0: usize, x0: usize, bs: usize, rec: &[f32]) {
    for (dst_row, src_row) in rect_rows_mut(plane, w, y0, x0, bs, bs).zip(rec.chunks_exact(bs)) {
        dst_row.copy_from_slice(src_row);
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dc_pred_rect_subsampled(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    neutral: f32,
    bd: i32,
) -> f32 {
    let (ha, hl) = (y0 > 0, x0 > 0);
    let ss_hor = if bw > 32 { 2 } else { 1 };
    let ss_ver = if bh > 32 { 2 } else { 1 };
    let mut sum: i64 = 0;
    let mut count: i64 = 0;
    if ha {
        let mut i = 0;
        while i < bw {
            sum += rec[(y0 - 1) * w + x0 + i] as i64;
            count += 1;
            i += ss_hor;
        }
    }
    if hl {
        let mut i = 0;
        while i < bh {
            sum += rec[(y0 + i) * w + x0 - 1] as i64;
            count += 1;
            i += ss_ver;
        }
    }
    if count == 0 {
        return neutral;
    }
    let (scale, shift) = resolve_divisor_32(count as u32);
    let rounding: i64 = (1i64 << shift) >> 1;
    let p = ((sum * scale + rounding) >> shift).clamp(0, (1 << bd) - 1);
    p as f32
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn dc_pred_rect(
    rec: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    neutral: f32,
    bd: i32,
) -> f32 {
    dc_pred_rect_bounded(
        rec,
        &BoundedIntraRect {
            stride: w,
            y: y0,
            x: x0,
            width: bw,
            height: bh,
            frame_width: usize::MAX,
            frame_height: usize::MAX,
        },
        neutral,
        bd,
    )
}

/// Geometry and coded-plane bounds for a rectangular intra predictor.
#[derive(Clone, Copy)]
pub(crate) struct BoundedIntraRect {
    pub(crate) stride: usize,
    pub(crate) y: usize,
    pub(crate) x: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) frame_width: usize,
    pub(crate) frame_height: usize,
}

/// Rectangular DC predictor with decoder-style edge replication. The coded plane
/// bounds may be smaller than the SB-padded backing allocation.
pub(crate) fn dc_pred_rect_bounded(
    rec: &[f32],
    rect: &BoundedIntraRect,
    neutral: f32,
    bd: i32,
) -> f32 {
    let BoundedIntraRect {
        stride: w,
        y: y0,
        x: x0,
        width: bw,
        height: bh,
        frame_width: fw,
        frame_height: fh,
    } = *rect;
    let (ha, hl) = (y0 > 0, x0 > 0);
    // Bounds are already expressed in this plane's coded sample grid. In
    // subsampled chroma that grid need not itself be a multiple of eight.
    let avail_above = if fw == usize::MAX {
        bw
    } else {
        fw.saturating_sub(x0).min(bw).max(1)
    };
    let avail_left = if fh == usize::MAX {
        bh
    } else {
        fh.saturating_sub(y0).min(bh).max(1)
    };
    let sa: i64 = if ha {
        (0..bw)
            .map(|i| rec[(y0 - 1) * w + x0 + i.min(avail_above - 1)] as i64)
            .sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bh)
            .map(|i| rec[(y0 + i.min(avail_left - 1)) * w + x0 - 1] as i64)
            .sum()
    } else {
        0
    };
    // avm `highbd_dc_predictor` (reconintra.h) averages the `count` reference.
    let (count, sum) = match (ha, hl) {
        (true, true) => (bw + bh, sa + sl),
        (true, false) => (bw, sa),
        (false, true) => (bh, sl),
        (false, false) => return neutral,
    };
    let (scale, shift) = resolve_divisor_32(count as u32);
    let rounding: i64 = (1i64 << shift) >> 1;
    let p = ((sum * scale + rounding) >> shift).clamp(0, (1 << bd) - 1);
    p as f32
}

/// avm `div_lut` (warped_motion.h): reciprocals at `DIV_LUT_PREC_BITS`=9 precision.
static DIV_LUT: [u16; 129] = [
    512, 508, 504, 500, 496, 493, 489, 485, 482, 478, 475, 471, 468, 465, 462, 458, 455, 452, 449,
    446, 443, 440, 437, 434, 431, 428, 426, 423, 420, 417, 415, 412, 410, 407, 405, 402, 400, 397,
    395, 392, 390, 388, 386, 383, 381, 379, 377, 374, 372, 370, 368, 366, 364, 362, 360, 358, 356,
    354, 352, 350, 349, 347, 345, 343, 341, 340, 338, 336, 334, 333, 331, 329, 328, 326, 324, 323,
    321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295,
    294, 293, 291, 290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 277, 275, 274, 273, 272,
    271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256,
];

/// avm `resolve_divisor_32` (warped_motion.h): decomposes D so 1/D ≈ scale/2^shift.
pub(crate) fn resolve_divisor_32(d: u32) -> (i64, u32) {
    let mut shift = 31 - d.leading_zeros(); // get_msb(D) = floor(log2 D)
    let e = d - (1u32 << shift); // D with the MSB cleared
    let f = if shift > 7 {
        // ROUND_POWER_OF_TWO(e, shift - 7)
        let s = shift - 7;
        ((e + (1u32 << (s - 1))) >> s) as usize
    } else {
        (e << (7 - shift)) as usize
    };
    shift += 9; // DIV_LUT_PREC_BITS
    (DIV_LUT[f] as i64, shift)
}
/// Residual of a `bw`-wide × `bh`-tall block, row-major (`r[yy*bw + xx]`).
pub(crate) fn get_residual_rect(
    plane: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    pred: f32,
) -> Vec<f32> {
    let mut r = vec![0f32; bw * bh];
    for (dst_row, src_row) in r
        .chunks_exact_mut(bw)
        .zip(rect_rows(plane, w, y0, x0, bw, bh))
    {
        for (dst, &src) in dst_row.iter_mut().zip(src_row) {
            *dst = src - pred;
        }
    }
    r
}
/// Write a `bw`-wide × `bh`-tall reconstructed block back into `plane`.
pub(crate) fn put_block_rect(
    plane: &mut [f32],
    w: usize,
    y0: usize,
    x0: usize,
    bw: usize,
    bh: usize,
    rec: &[f32],
) {
    for (dst_row, src_row) in rect_rows_mut(plane, w, y0, x0, bw, bh).zip(rec.chunks_exact(bw)) {
        dst_row.copy_from_slice(src_row);
    }
}

#[inline(always)]
pub(crate) fn pixel_to_i32(v: f32) -> i32 {
    // Reconstructed AV2 sample planes are already clipped to the valid pixel range.
    // Rounding here makes RD distortion operate on the same integer samples that are
    // actually written to the reconstructed picture instead of accumulating float
    // noise from the f32 scratch representation.
    (v + 0.5) as i32
}

#[inline(always)]
pub(crate) fn sq_diff_f32(a: i32, b: i32) -> f32 {
    let d = (a - b) as f32;
    d * d
}

pub(crate) type PixelSseRoundedFn = unsafe fn(&[f32], &[f32]) -> f32;
pub(crate) type PixelSseRoundedConstFn = unsafe fn(&[f32], f32) -> f32;
pub(crate) type PixelSseF32Fn = unsafe fn(&[f32], &[f32]) -> f32;
pub(crate) type PixelSseF32U16Fn = unsafe fn(&[f32], &[u16]) -> f32;
pub(crate) type WeightedPixelSseF32Fn = unsafe fn(&[f32], &[f32], &[f32]) -> f32;
pub(crate) type SadU8Fn = unsafe fn(&[u8], &[u8]) -> u32;
pub(crate) type SumSumsqF32Fn = unsafe fn(&[f32]) -> (f32, f32);
pub(crate) type CflSseI32Fn = unsafe fn(&[i32], &[i32], i32, i32, i32) -> f32;
pub(crate) type CoeffRateF32Fn = unsafe fn(&[f32]) -> f32;
pub(crate) type CoeffAbsRateF32Fn = unsafe fn(&[f32]) -> f32;

static PIXEL_SSE_ROUNDED: OnceLock<PixelSseRoundedFn> = OnceLock::new();
static PIXEL_SSE_ROUNDED_CONST: OnceLock<PixelSseRoundedConstFn> = OnceLock::new();
static PIXEL_SSE_F32: OnceLock<PixelSseF32Fn> = OnceLock::new();
static PIXEL_SSE_F32_U16: OnceLock<PixelSseF32U16Fn> = OnceLock::new();
static WEIGHTED_PIXEL_SSE_F32: OnceLock<WeightedPixelSseF32Fn> = OnceLock::new();
static SAD_U8: OnceLock<SadU8Fn> = OnceLock::new();
static SUM_SUMSQ_F32: OnceLock<SumSumsqF32Fn> = OnceLock::new();
static CFL_SSE_I32: OnceLock<CflSseI32Fn> = OnceLock::new();
static COEFF_RATE_F32: OnceLock<CoeffRateF32Fn> = OnceLock::new();
static COEFF_ABS_RATE_F32: OnceLock<CoeffAbsRateF32Fn> = OnceLock::new();

#[inline]
fn pixel_sse_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[inline]
fn pixel_sse_f32_u16_scalar(a: &[f32], b: &[u16]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y as f32;
            d * d
        })
        .sum()
}

#[inline]
fn weighted_pixel_sse_f32_scalar(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .zip(weights)
        .map(|((&x, &y), &w)| {
            let d = x - y;
            d * d * w
        })
        .sum()
}

#[inline]
fn sad_u8_scalar(a: &[u8], b: &[u8]) -> u32 {
    a.iter().zip(b).fold(0u32, |sum, (&x, &y)| {
        sum.saturating_add((x as i32 - y as i32).unsigned_abs())
    })
}

#[inline]
fn sum_sumsq_f32_scalar(v: &[f32]) -> (f32, f32) {
    v.iter()
        .fold((0.0, 0.0), |(sum, sumsq), &x| (sum + x, sumsq + x * x))
}

#[inline]
fn cfl_sse_i32_scalar(src: &[i32], ac: &[i32], alpha_q3: i32, dc: i32, maxv: i32) -> f32 {
    src.iter()
        .zip(ac)
        .map(|(&s, &a)| {
            let p = alpha_q3 * a;
            let sign = p >> 31;
            let abs = (p ^ sign) - sign;
            let q = (abs + (1 << 10)) >> 11;
            let scaled = (q ^ sign) - sign;
            let pred = (dc + scaled).clamp(0, maxv);
            sq_diff_f32(s, pred)
        })
        .sum()
}

#[inline]
fn resolve_pixel_sse_f32() -> PixelSseF32Fn {
    *PIXEL_SSE_F32.get_or_init(|| {
        let mut _f = pixel_sse_f32_scalar as PixelSseF32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::pixel_sse_f32_neon as PixelSseF32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::pixel_sse_f32_avx2 as PixelSseF32Fn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_pixel_sse_f32_u16() -> PixelSseF32U16Fn {
    *PIXEL_SSE_F32_U16.get_or_init(|| {
        let mut _f = pixel_sse_f32_u16_scalar as PixelSseF32U16Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::pixel_sse_f32_u16_neon as PixelSseF32U16Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::pixel_sse_f32_u16_avx2 as PixelSseF32U16Fn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_weighted_pixel_sse_f32() -> WeightedPixelSseF32Fn {
    *WEIGHTED_PIXEL_SSE_F32.get_or_init(|| {
        let mut _f = weighted_pixel_sse_f32_scalar as WeightedPixelSseF32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::weighted_pixel_sse_f32_neon as WeightedPixelSseF32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::weighted_pixel_sse_f32_avx2 as WeightedPixelSseF32Fn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_sad_u8() -> SadU8Fn {
    *SAD_U8.get_or_init(|| {
        let mut _f = sad_u8_scalar as SadU8Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::sad_u8_neon as SadU8Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::sad_u8_avx2 as SadU8Fn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_sum_sumsq_f32() -> SumSumsqF32Fn {
    *SUM_SUMSQ_F32.get_or_init(|| {
        let mut _f = sum_sumsq_f32_scalar as SumSumsqF32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::sum_sumsq_f32_neon as SumSumsqF32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::sum_sumsq_f32_avx2 as SumSumsqF32Fn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_cfl_sse_i32() -> CflSseI32Fn {
    *CFL_SSE_I32.get_or_init(|| {
        let mut _f = cfl_sse_i32_scalar as CflSseI32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::cfl_sse_i32_neon as CflSseI32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::cfl_sse_i32_avx2 as CflSseI32Fn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn pixel_sse_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let f = resolve_pixel_sse_f32();
    unsafe { f(a, b) }
}

#[inline]
pub(crate) fn weighted_pixel_sse_f32(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), weights.len());
    let f = resolve_weighted_pixel_sse_f32();
    unsafe { f(a, b, weights) }
}

#[inline]
pub(crate) fn sad_u8(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let f = resolve_sad_u8();
    unsafe { f(a, b) }
}

#[inline]
pub(crate) fn sum_sumsq_f32(v: &[f32]) -> (f32, f32) {
    let f = resolve_sum_sumsq_f32();
    unsafe { f(v) }
}

#[inline]
pub(crate) fn cfl_sse_i32(src: &[i32], ac: &[i32], alpha_q3: i32, dc: i32, maxv: i32) -> f32 {
    debug_assert_eq!(src.len(), ac.len());
    let f = resolve_cfl_sse_i32();
    unsafe { f(src, ac, alpha_q3, dc, maxv) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_sse_f32_u16_block(
    src: &[f32],
    src_stride: usize,
    src_y: usize,
    src_x: usize,
    rec: &[u16],
    rec_stride: usize,
    w: usize,
    h: usize,
) -> f32 {
    let f = resolve_pixel_sse_f32_u16();
    rect_rows(src, src_stride, src_y, src_x, w, h)
        .zip(rec.chunks_exact(rec_stride).map(|row| &row[..w]))
        .map(|(src_row, rec_row)| unsafe { f(src_row, rec_row) })
        .sum()
}

#[inline]
fn pixel_sse_rounded_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| sq_diff_f32(pixel_to_i32(x), pixel_to_i32(y)))
        .sum()
}

#[inline]
fn pixel_sse_rounded_const_scalar(a: &[f32], v: f32) -> f32 {
    let vi = pixel_to_i32(v);
    a.iter().map(|&x| sq_diff_f32(pixel_to_i32(x), vi)).sum()
}

#[inline]
fn resolve_pixel_sse_rounded() -> PixelSseRoundedFn {
    *PIXEL_SSE_ROUNDED.get_or_init(|| {
        let mut _f = pixel_sse_rounded_scalar as PixelSseRoundedFn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::pixel_sse_rounded_neon as PixelSseRoundedFn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::pixel_sse_rounded_avx2 as PixelSseRoundedFn;
            }
        }
        _f
    })
}

#[inline]
fn resolve_pixel_sse_rounded_const() -> PixelSseRoundedConstFn {
    *PIXEL_SSE_ROUNDED_CONST.get_or_init(|| {
        let mut _f = pixel_sse_rounded_const_scalar as PixelSseRoundedConstFn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::pixel_sse_rounded_const_neon as PixelSseRoundedConstFn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::pixel_sse_rounded_const_avx2 as PixelSseRoundedConstFn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn pixel_sse_rounded(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let f = resolve_pixel_sse_rounded();
    unsafe { f(a, b) }
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_sse_rounded_block(
    src: &[f32],
    src_stride: usize,
    src_y: usize,
    src_x: usize,
    rec: &[f32],
    rec_stride: usize,
    w: usize,
    h: usize,
) -> f32 {
    let f = resolve_pixel_sse_rounded();
    rect_rows(src, src_stride, src_y, src_x, w, h)
        .zip(rec.chunks_exact(rec_stride).map(|row| &row[..w]))
        .map(|(src_row, rec_row)| unsafe { f(src_row, rec_row) })
        .sum()
}

#[inline]
#[allow(clippy::too_many_arguments)]
pub(crate) fn pixel_sse_rounded_block_const(
    src: &[f32],
    src_stride: usize,
    src_y: usize,
    src_x: usize,
    w: usize,
    h: usize,
    v: f32,
) -> f32 {
    let f = resolve_pixel_sse_rounded_const();
    rect_rows(src, src_stride, src_y, src_x, w, h)
        .map(|src_row| unsafe { f(src_row, v) })
        .sum()
}

#[inline]
pub(crate) fn log2p1_approx_f32(x: f32) -> f32 {
    let y = 1.0 + x;
    let bits = y.to_bits();
    let e = ((bits >> 23) as i32 - 127) as f32;
    let m = f32::from_bits((bits & 0x007f_ffff) | 0x3f80_0000);
    let t = m - 1.0;

    // Same Sollya-generated polynomial as the SIMD paths.
    let mut p = 2.096841670572757720947265625e-2f32;
    p = -9.749893844127655029296875e-2f32 + t * p;
    p = 0.21719777584075927734375f32 + t * p;
    p = -0.340080082416534423828125f32 + t * p;
    p = 0.477900087833404541015625f32 + t * p;
    p = -0.721179187297821044921875f32 + t * p;
    p = 1.4426934719085693359375f32 + t * p;

    e + t * p
}

#[inline]
fn coeff_rate_f32_scalar(lev: &[f32]) -> f32 {
    lev.iter()
        .filter(|&&v| v != 0.0)
        .map(|&v| 2.0 + 2.0 * log2p1_approx_f32(v.abs()))
        .sum()
}

#[inline]
fn coeff_abs_rate_f32_scalar(lev: &[f32]) -> f32 {
    lev.iter().map(|&v| v.abs()).sum()
}

#[inline]
fn resolve_coeff_rate_f32() -> CoeffRateF32Fn {
    *COEFF_RATE_F32.get_or_init(|| {
        let mut _f = coeff_rate_f32_scalar as CoeffRateF32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::coeff_rate_f32_neon as CoeffRateF32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
                _f = crate::av2::avx::coeff_rate_f32_avx2 as CoeffRateF32Fn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn coeff_rate_f32(lev: &[f32]) -> f32 {
    debug_assert!(lev.iter().all(|v| v.is_finite()));
    let f = resolve_coeff_rate_f32();
    unsafe { f(lev) }
}

#[inline]
fn resolve_coeff_abs_rate_f32() -> CoeffAbsRateF32Fn {
    *COEFF_ABS_RATE_F32.get_or_init(|| {
        let mut _f = coeff_abs_rate_f32_scalar as CoeffAbsRateF32Fn;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = crate::av2::neon::coeff_abs_rate_f32_neon as CoeffAbsRateF32Fn;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = crate::av2::avx::coeff_abs_rate_f32_avx2 as CoeffAbsRateF32Fn;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn coeff_abs_rate_f32(lev: &[f32]) -> f32 {
    debug_assert!(lev.iter().all(|v| v.is_finite()));
    let f = resolve_coeff_abs_rate_f32();
    unsafe { f(lev) }
}

#[inline]
pub(crate) fn coeff_tu_rate_proxy_f32(tu: &[Coeff]) -> f32 {
    // Sparse coefficients are `(scan_index, level)` pairs, so their levels are not
    // contiguous in memory. Pack every run into a zero-padded stack block and send it
    // through the dense SIMD rate kernel. Padding is free because zero levels are
    // masked by `coeff_rate_f32`; this also keeps the common short sparse TU out of a
    // scalar-only tail path.
    const N: usize = 32;
    let mut packed = [0f32; N];
    let mut bits = 0f32;
    for chunk in tu.chunks(N) {
        packed.fill(0.0);
        for (dst, &(_, level)) in packed.iter_mut().zip(chunk) {
            *dst = level.unsigned_abs() as f32;
        }
        bits += coeff_rate_f32(&packed);
    }
    bits
}

#[inline]
pub(crate) fn coeff_tus_rate_proxy_f32(tus: &[Vec<Coeff>], tu_overhead_bits: f32) -> f32 {
    tus.iter()
        .map(|tu| tu_overhead_bits + coeff_tu_rate_proxy_f32(tu))
        .sum()
}

/// Incremental EXT_NEW_TX_SET signalling costs relative to DCT_DCT for the
/// mode-independent TX_16X16 entries: DCT_DCT, ADST_ADST, ADST_DCT, DCT_ADST.
/// Derived from TXTP_EXT8 rather than hand-tuned approximations.
pub(crate) const TX16_TYPE_RATE_DELTA: [f32; 4] = [0.0, 0.527_067_7, 3.126_745_2, 2.660_032_9];

/// Edge-aware TX_16X16 distortion. Plain SSE systematically underprices the
/// directional ringing produced by ADST candidates: their pixel error may be
/// smaller while the error's horizontal/vertical derivatives are much larger.
/// The derivative term is averaged over the two directions so this remains on
/// approximately the same scale as pixel SSE (and therefore the existing lambda).
pub(crate) fn tx16_distortion(src: &[f32; 256], rec: &[f32]) -> f32 {
    debug_assert_eq!(rec.len(), 256);
    let mut pixel = 0.0f32;
    let mut edge = 0.0f32;
    for y in 0..16 {
        for x in 0..16 {
            let i = y * 16 + x;
            let s = src[i].round();
            let r = rec[i].round();
            let e = s - r;
            pixel += e * e;
            if x != 0 {
                let de = (s - src[i - 1].round()) - (r - rec[i - 1].round());
                edge += de * de;
            }
            if y != 0 {
                let de = (s - src[i - 16].round()) - (r - rec[i - 16].round());
                edge += de * de;
            }
        }
    }
    pixel + 0.5 * edge
}

/// Choose a non-DCT TX_16X16 only when it wins the edge-aware RD comparison.
///
/// `dc_only[i]` marks a candidate whose quantised levels have no AC coefficient.
/// A non-DCT tx_type is only *signaled* when the leaf has an AC coefficient
/// (`encode_luma_leaf_16x16_full` gates it on `eob >= 1`, mirroring AVM which
/// reads the intra ext-tx symbol only for eob>1), so a DC-only leaf always
/// decodes as DCT_DCT. Under ADST/mixed the DC basis function is a DST-VII ramp
/// (not flat), so a DC-only ADST reconstruction is a gradient while the decoder
/// produces a flat DCT block — the 4:4:4 "content from nowhere" mismatch. Never
/// pick a DC-only non-DCT candidate; DCT is exact (and optimal) for flat leaves.
pub(crate) fn choose_tx16_type(cost: [f32; 4], distortion: [f32; 4], dc_only: [bool; 4]) -> usize {
    let mut best = cost[0];
    let mut choice = 0;
    for i in 1..4 {
        if !dc_only[i] && distortion[i] < distortion[0] && cost[i] < best {
            best = cost[i];
            choice = i;
        }
    }
    choice
}

/// True when the scan-ordered levels carry no AC coefficient (only `lev[0]`, the
/// DC term, may be non-zero) — such a 16×16 leaf can only decode as DCT_DCT.
#[inline]
pub(crate) fn tx16_dc_only(lev: &[f32]) -> bool {
    !lev.iter().skip(1).any(|&l| l != 0.0)
}

pub(crate) fn levels_to_coeffs(lev: &[f32]) -> Vec<Coeff> {
    lev.iter()
        .enumerate()
        .filter(|&(_, l)| *l != 0.0)
        .map(|(k, &l)| (k, l as i32))
        .collect()
}

pub(crate) fn sb_tu4_contexts(
    tus: &[Vec<Coeff>],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    rem_rows: usize,
    rem_cols: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = rem_rows * rem_cols;
    let mut skip_ctx = vec![0usize; n];
    let mut dc_sign_ctx = vec![0usize; n];
    for by in 0..rem_rows {
        for bx in 0..rem_cols {
            let i = by * rem_cols + bx;
            let cx = sb_x / 4 + bx;
            let cy = sb_y / 4 + by;
            let a = above[cx];
            let l = left[cy];
            let top = (a & 0x3F).min(4) as usize;
            let lft = (l & 0x3F).min(4) as usize;
            skip_ctx[i] = (top + lft + 3) >> 1;
            // each neighbor unit contributes its packed sign (0x40 -> 1 baseline)
            let dcs = ((a & 0xC0) >> 6) as i32 + ((l & 0xC0) >> 6) as i32;
            let sgn = dcs - 2; // two neutral units sum to 2
            dc_sign_ctx[i] = ((sgn != 0) as usize) + ((sgn > 0) as usize);
            // update the grid with this TU's result context
            let nz: Vec<Coeff> = tus[i].iter().cloned().filter(|&(_, l)| l != 0).collect();
            let res = if nz.is_empty() {
                0x40u8
            } else {
                let cul = nz
                    .iter()
                    .map(|&(_, l)| l.unsigned_abs())
                    .sum::<u32>()
                    .min(7) as u8;
                let dc = nz
                    .iter()
                    .find(|&&(s, _)| s == 0)
                    .map(|&(_, l)| l)
                    .unwrap_or(0);
                let dcbits = if dc > 0 {
                    0x80
                } else if dc < 0 {
                    0x00
                } else {
                    0x40
                };
                (cul & 0x3F) | dcbits
            };
            above[cx] = res;
            left[cy] = res;
        }
    }
    (skip_ctx, dc_sign_ctx)
}

/// Compute one superblock's 256 4x4 lossless TUs for a single plane: per 4x4 block,
/// DC-predict from the reconstruction, forward-WHT the residual, reconstruct (so later
/// blocks predict from exact neighbors), and emit the coded levels as a coeff list.
pub(crate) fn lossless_sb_tus(
    src: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    neutral: f32,
    rem_rows: usize,
    rem_cols: usize,
) -> Vec<Vec<Coeff>> {
    let mut tus: Vec<Vec<Coeff>> = Vec::with_capacity(rem_rows * rem_cols);
    for by in 0..rem_rows {
        for bx in 0..rem_cols {
            let (y0, x0) = (sb_y + by * 4, sb_x + bx * 4);
            let pred = dc_pred(src, pw, y0, x0, 4, neutral);
            let mut resid = [0i32; 16];
            for (dst_row, src_row) in resid
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(rect_rows(src, pw, y0, x0, 4, 4))
            {
                for (dst, &src) in dst_row.iter_mut().zip(src_row) {
                    *dst = (src - pred) as i32;
                }
            }
            let lev = fwht4x4(&resid);
            tus.push(levels_to_coeffs_4x4(&lev));
        }
    }
    tus
}

/// U: ctx = (above_nz + left_nz) + 6, indexed into the shared txb_skip table.
/// V: ctx = (above_nz + left_nz) + 3 + (co-located U non-zero ? 6 : 0), indexed into
/// the separate v_txb_skip table. Entropy context inits to 0; grids store cul.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sb_tu4_chroma_skip(
    tus: &[Vec<Coeff>],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    plane_v: bool,
    eob_u_last: bool,
    rem_rows: usize,
    rem_cols: usize,
) -> Vec<usize> {
    // avm reads all U txbs then all V txbs; xd->eob_u_flag is a single field left at the
    // LAST U txb's value, so every V TU's skip context uses (last U TU nonzero), not the
    // co-located one.
    let v_off = 3 + if eob_u_last { 6 } else { 0 };
    let mut skip = vec![0usize; rem_rows * rem_cols];
    for by in 0..rem_rows {
        for bx in 0..rem_cols {
            let i = by * rem_cols + bx;
            let cx = sb_x / 4 + bx;
            let cy = sb_y / 4 + by;
            let base = ((above[cx] != 0) as usize) + ((left[cy] != 0) as usize);
            skip[i] = if plane_v { base + v_off } else { base + 6 };
            let cul = tus[i]
                .iter()
                .filter(|&&(_, l)| l != 0)
                .map(|&(_, l)| l.unsigned_abs())
                .sum::<u32>()
                .min(7) as u8;
            above[cx] = cul;
            left[cy] = cul;
        }
    }
    skip
}

/// Work-stealing parallel map (same scheme as the AV1 coder): workers atomically
/// claim the next index, so no thread idles on a static chunk while work remains.
pub(crate) fn par_map_indexed<T, F>(nthreads: usize, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    if nthreads <= 1 || n <= 1 {
        return (0..n).map(f).collect();
    }
    let next = std::sync::atomic::AtomicUsize::new(0);
    let work = || {
        let mut got: Vec<(usize, T)> = Vec::new();
        loop {
            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if i >= n {
                break got;
            }
            got.push((i, f(i)));
        }
    };
    let parts: Vec<Vec<(usize, T)>> = std::thread::scope(|scope| {
        // Caller thread participates too, so spawn one fewer worker.
        let spawned = nthreads.min(n) - 1;
        let handles: Vec<_> = (0..spawned).map(|_| scope.spawn(work)).collect();
        std::iter::once(work())
            .chain(
                handles
                    .into_iter()
                    .map(|h| h.join().expect("worker panicked")),
            )
            .collect()
    });
    let mut slots: Vec<Option<T>> = std::iter::repeat_with(|| None).take(n).collect();
    for (i, v) in parts.into_iter().flatten() {
        slots[i] = Some(v);
    }
    slots
        .into_iter()
        .map(|s| s.expect("index produced"))
        .collect()
}

/// Dependency-ordered wavefront over an `rows × cols` grid of superblocks: cell
/// `(r, c)` is only visited after `(r-1, c)` and `(r, c-1)`, so all cells on the
/// same anti-diagonal `r + c` are mutually independent and run in parallel (one
/// barrier per diagonal). Peak parallelism is `min(rows, cols)`.
///
/// This is the **format-agnostic Stage-A engine** for the staged-threading
/// decouple: each chroma core (444/420/422/400) supplies the same shape of
/// per-SB *decide* closure — read the already-finished top/left neighbour
/// reconstruction, decide the SB, write its own SB region + record entry — and
/// this driver parallelises it so a single tile saturates all cores without the
/// compression cost of extra tiles. Emit stays a serial raster replay of the
/// records (entropy is inherently sequential; the decide is CDF-independent, so
/// only the recon dependency constrains ordering — exactly this wavefront).
///
/// The closure runs on worker threads, so any shared state it mutates (the recon
/// planes, the per-SB record slots) must use interior mutability. Soundness of
/// such disjoint writes rests on the invariant this driver guarantees: when
/// `(r, c)` runs, every cell on diagonals `< r + c` has finished, and no other
/// cell on the current diagonal touches `(r, c)`'s SB region — so each closure
/// writes ONLY its own SB block and reads ONLY prior-diagonal neighbours.
///
/// Not yet wired into a core's Stage A (that needs the full per-SB decide/emit
/// decouple + a disjoint-recon-write wrapper); it is the shared driver every
/// format will call once that lands. See `av2-staged-threading-decouple` memory.
#[allow(dead_code)]
pub(crate) fn par_wavefront<F>(nthreads: usize, rows: usize, cols: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if rows == 0 || cols == 0 {
        return;
    }
    // Anti-diagonals d = 0 ..= (rows-1)+(cols-1); each diagonal is a barrier.
    for d in 0..=(rows + cols - 2) {
        let r_lo = d.saturating_sub(cols - 1);
        let r_hi = d.min(rows - 1);
        let cells: Vec<(usize, usize)> = (r_lo..=r_hi).map(|r| (r, d - r)).collect();
        par_map_indexed(nthreads, cells.len(), |i| {
            let (r, c) = cells[i];
            f(r, c);
        });
    }
}

/// Row-skewed (WPP-style) wavefront that additionally satisfies the **above-right**
/// dependency AV2 intra prediction needs: cell `(r, c)` runs only after `(r-1,c-1)`,
/// `(r-1,c)`, `(r-1,c+1)` (above-right!) and `(r,c-1)` have all finished.
///
/// Ordering key `d = 2r + c` achieves this: every one of those four predecessors
/// maps to a strictly smaller `d` (`2r+c-3`, `2r+c-2`, `2r+c-1`, `2r+c-1`), so they
/// lie on an earlier barrier; and no two cells sharing a `d` depend on each other
/// (a same-`d` neighbour of `(r,c)` would be `(r-1,c+2)`, which is not a
/// predecessor). Peak parallelism is `~min(rows, (cols+1)/2)` — half the plain
/// anti-diagonal's, the price of the extra dependency, but correct for AV2 intra.
///
/// Unlike [`par_wavefront`] (`d = r+c`), this is the schedule a 4:4:4 SB decide can
/// actually run under: the plain anti-diagonal places `(r-1,c+1)` on the *same*
/// diagonal (concurrent), so an SB reading its above-right neighbour's recon would
/// race a half-written block. Barrier per diagonal via [`par_map_indexed`].
#[allow(dead_code)]
pub(crate) fn par_wavefront_wpp<F>(nthreads: usize, rows: usize, cols: usize, f: F)
where
    F: Fn(usize, usize) + Sync,
{
    if rows == 0 || cols == 0 {
        return;
    }
    // d = 2r + c, d ∈ 0 ..= 2(rows-1) + (cols-1). For each d, r ranges over the cells
    // whose column c = d - 2r is in [0, cols-1] and whose row is in [0, rows-1].
    let d_max = 2 * (rows - 1) + (cols - 1);
    for d in 0..=d_max {
        // c = d - 2r ≥ 0  ⇒ r ≤ d/2;  c ≤ cols-1 ⇒ 2r ≥ d-cols+1 ⇒ r ≥ ⌈(d-cols+1)/2⌉.
        let r_lo = if d + 1 > cols { (d - cols + 2) / 2 } else { 0 };
        let r_hi = (d / 2).min(rows - 1);
        if r_lo > r_hi {
            continue;
        }
        let cells: Vec<(usize, usize)> = (r_lo..=r_hi).map(|r| (r, d - 2 * r)).collect();
        par_map_indexed(nthreads, cells.len(), |i| {
            let (r, c) = cells[i];
            f(r, c);
        });
    }
}

/// **Persistent-pool** wavefront. With `needs_above_right`, it uses the WPP
/// `d = 2r + c` schedule; otherwise it uses the wider `d = r + c` top/left-only
/// schedule. Workers spawn **once** for the whole wavefront and loop over its
/// diagonals with a barrier between each.
///
/// Why it matters: each format's per-SB decide closure keeps its recon scratch in
/// a `thread_local!` full-plane buffer (~100+ MB for large frames). Under the
/// old per-diagonal `par_map_indexed` driver the worker threads are re-created
/// every diagonal, so that `thread_local` is RE-ALLOCATED per diagonal per worker
/// → memory-bandwidth saturation that REGRESSES scaling past ~4 threads. Here the
/// workers persist across all diagonals, so each worker's `thread_local` is
/// allocated exactly once.
///
/// Returns per-cell results indexed `r * cols + c`; every cell is produced exactly
/// once (all `Some` on return). Correctness rests on three invariants:
///  1. The `barrier.wait()` after each diagonal enforces the finished-neighbour
///     dependency — every cell of diagonal `d` is written (and every recon write
///     its closure made is globally visible, the barrier being a full fence)
///     before ANY worker starts diagonal `d+1`, which reads those cells' SB
///     regions as its halo.
///  2. Result writes are disjoint: each cell `(r, c)` writes `results[r*cols+c]`
///     exactly once, so the raw-pointer view never aliases across workers.
///  3. `f` keeps using the existing `thread_local!` scratch unchanged; persistence
///     across diagonals is what makes that allocation one-time.
///
/// `nthreads <= 1` runs serially in diagonal order (matching the parallel visit
/// order). See `av2-staged-threading-decouple` memory.
pub(crate) fn par_wavefront_pool<T, F>(
    nthreads: usize,
    rows: usize,
    cols: usize,
    needs_above_right: bool,
    f: F,
) -> Vec<Option<T>>
where
    T: Send,
    F: Fn(usize, usize) -> T + Sync,
{
    let n = rows * cols;
    let mut results: Vec<Option<T>> = (0..n).map(|_| None).collect();
    if rows == 0 || cols == 0 {
        return results;
    }
    let mut diagonals: Vec<Vec<(usize, usize)>> = Vec::new();
    if needs_above_right {
        // d = 2r+c keeps the above-right SB on an earlier diagonal.
        let d_max = 2 * (rows - 1) + (cols - 1);
        for d in 0..=d_max {
            let r_lo = if d + 1 > cols { (d - cols + 2) / 2 } else { 0 };
            let r_hi = (d / 2).min(rows - 1);
            if r_lo <= r_hi {
                diagonals.push((r_lo..=r_hi).map(|r| (r, d - 2 * r)).collect());
            }
        }
    } else {
        // DC-only decisions need top and left but never above-right. The ordinary
        // anti-diagonal doubles useful parallel width for the monochrome fast path.
        for d in 0..=(rows + cols - 2) {
            let r_lo = d.saturating_sub(cols - 1);
            let r_hi = d.min(rows - 1);
            diagonals.push((r_lo..=r_hi).map(|r| (r, d - r)).collect());
        }
    }
    if nthreads <= 1 {
        for cells in &diagonals {
            for &(r, c) in cells {
                results[r * cols + c] = Some(f(r, c));
            }
        }
        return results;
    }

    // Raw-pointer Send wrapper over the result slots. Each cell writes its own
    // slot exactly once (disjoint), so concurrent writes never alias.
    struct ResultsPtr<T>(*mut Option<T>);
    // SAFETY: writes are disjoint per the wavefront invariant (one cell → one
    // slot), and the buffer outlives the scope below.
    unsafe impl<T: Send> Send for ResultsPtr<T> {}
    unsafe impl<T: Send> Sync for ResultsPtr<T> {}
    let base = ResultsPtr(results.as_mut_ptr());

    // Per-diagonal work counter (no reset needed: each diagonal has its own).
    let counters: Vec<std::sync::atomic::AtomicUsize> = diagonals
        .iter()
        .map(|_| std::sync::atomic::AtomicUsize::new(0))
        .collect();
    let barrier = std::sync::Barrier::new(nthreads);

    let f = &f;
    let diagonals = &diagonals;
    let counters = &counters;
    let barrier = &barrier;
    let base = &base;
    let worker = || {
        for (d, cells) in diagonals.iter().enumerate() {
            loop {
                let i = counters[d].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if i >= cells.len() {
                    break;
                }
                let (r, c) = cells[i];
                let v = f(r, c);
                // SAFETY: slot (r*cols+c) is written by exactly one worker; no
                // other worker aliases it.
                unsafe { base.0.add(r * cols + c).write(Some(v)) };
            }
            // Barrier: diagonal `d` fully done (and its recon writes visible)
            // before any worker starts `d+1`. All `nthreads` threads participate.
            barrier.wait();
        }
    };
    std::thread::scope(|scope| {
        // Main thread is one of the `nthreads` workers (mirrors par_map_indexed).
        for _ in 0..nthreads - 1 {
            scope.spawn(worker);
        }
        worker();
    });
    results
}

/// A raw mutable view over one reconstruction plane that lets `par_wavefront`
/// closures write their own superblock blocks concurrently. `Send + Sync` is
/// asserted unsafely; soundness rests on the wavefront invariant — each cell
/// writes ONLY its own SB block, so no two concurrent writers touch the same
/// byte, and any bytes read (finished neighbours from earlier diagonals) are
/// never being written at the same time.
///
/// Not yet wired into a core (integration also needs the `DecideOnly` mode + the
/// per-SB decouple); shipped now as the tested companion to [`par_wavefront`].
#[allow(dead_code)]
pub(crate) struct PlaneWriter {
    ptr: *mut f32,
    len: usize,
    stride: usize,
}

// SAFETY: the pointer addresses a caller-owned `&mut [f32]` that outlives the
// writer; concurrent use is sound only under the disjoint-block contract above.
unsafe impl Send for PlaneWriter {}
unsafe impl Sync for PlaneWriter {}

#[allow(dead_code)]
impl PlaneWriter {
    pub(crate) fn new(plane: &mut [f32], stride: usize) -> Self {
        Self {
            ptr: plane.as_mut_ptr(),
            len: plane.len(),
            stride,
        }
    }

    /// Write an `h × w` block (row-major `src`) at plane offset `(y, x)`.
    ///
    /// # Safety
    /// No other thread may access this `h × w` region concurrently. The region
    /// must lie within the plane.
    pub(crate) unsafe fn write_block(&self, y: usize, x: usize, h: usize, w: usize, src: &[f32]) {
        debug_assert!((y + h - 1) * self.stride + x + w <= self.len);
        for r in 0..h {
            let off = (y + r) * self.stride + x;
            // SAFETY: caller guarantees exclusive access to this disjoint block.
            let dst = unsafe { std::slice::from_raw_parts_mut(self.ptr.add(off), w) };
            dst.copy_from_slice(&src[r * w..r * w + w]);
        }
    }

    /// Copy an `h × w` region at `(y, x)` FROM this plane into `dst` (a full-plane
    /// buffer with the same stride) at the SAME coordinates. Used by a wavefront
    /// worker to pull its halo (finished neighbours) out of the shared recon plane.
    ///
    /// # Safety
    /// No thread may be WRITING this region concurrently. Under `par_wavefront_wpp`
    /// the halo touches only earlier-diagonal (finished) superblocks, so this holds.
    /// The region must lie within the plane and `dst.len() == self.len`.
    pub(crate) unsafe fn copy_region_to(
        &self,
        dst: &mut [f32],
        y: usize,
        x: usize,
        h: usize,
        w: usize,
    ) {
        debug_assert_eq!(dst.len(), self.len);
        for r in 0..h {
            let off = (y + r) * self.stride + x;
            // SAFETY: caller guarantees this region is finished (not being written).
            let src = unsafe { std::slice::from_raw_parts(self.ptr.add(off), w) };
            dst[off..off + w].copy_from_slice(src);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wavefront_respects_top_left_deps() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for &(rows, cols) in &[(1usize, 1usize), (1, 8), (8, 1), (5, 7), (16, 12), (3, 20)] {
            let done: Vec<AtomicBool> = (0..rows * cols).map(|_| AtomicBool::new(false)).collect();
            let ok = AtomicBool::new(true);
            let idx = |r: usize, c: usize| r * cols + c;
            par_wavefront(4, rows, cols, |r, c| {
                // Both causal neighbours must already be finished when we run.
                if r > 0 && !done[idx(r - 1, c)].load(Ordering::Acquire) {
                    ok.store(false, Ordering::Relaxed);
                }
                if c > 0 && !done[idx(r, c - 1)].load(Ordering::Acquire) {
                    ok.store(false, Ordering::Relaxed);
                }
                done[idx(r, c)].store(true, Ordering::Release);
            });
            assert!(
                ok.load(Ordering::Relaxed),
                "{rows}x{cols}: cell ran before top/left neighbour"
            );
            assert!(
                done.iter().all(|b| b.load(Ordering::Relaxed)),
                "{rows}x{cols}: not every cell was visited"
            );
        }
    }

    #[test]
    fn wpp_wavefront_respects_above_right_dep() {
        use std::sync::atomic::{AtomicBool, Ordering};
        for &(rows, cols) in &[
            (1usize, 1usize),
            (1, 8),
            (8, 1),
            (5, 7),
            (16, 12),
            (3, 20),
            (12, 3),
            (2, 2),
        ] {
            let done: Vec<AtomicBool> = (0..rows * cols).map(|_| AtomicBool::new(false)).collect();
            let ok = AtomicBool::new(true);
            let idx = |r: usize, c: usize| r * cols + c;
            par_wavefront_wpp(4, rows, cols, |r, c| {
                // All four causal predecessors — including above-right (r-1,c+1) —
                // must already be finished when this cell runs.
                let check = |rr: i64, cc: i64| {
                    if rr >= 0
                        && cc >= 0
                        && (rr as usize) < rows
                        && (cc as usize) < cols
                        && !done[idx(rr as usize, cc as usize)].load(Ordering::Acquire)
                    {
                        ok.store(false, Ordering::Relaxed);
                    }
                };
                let (ri, ci) = (r as i64, c as i64);
                check(ri - 1, ci - 1);
                check(ri - 1, ci);
                check(ri - 1, ci + 1); // above-right
                check(ri, ci - 1);
                done[idx(r, c)].store(true, Ordering::Release);
            });
            assert!(
                ok.load(Ordering::Relaxed),
                "{rows}x{cols}: cell ran before a causal (incl. above-right) neighbour"
            );
            assert!(
                done.iter().all(|b| b.load(Ordering::Relaxed)),
                "{rows}x{cols}: not every cell was visited exactly once"
            );
        }
    }

    #[test]
    fn wavefront_pool_respects_deps_and_runs_each_cell_once() {
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
        for &(rows, cols) in &[
            (1usize, 1usize),
            (1, 8),
            (8, 1),
            (5, 7),
            (16, 12),
            (3, 20),
            (12, 3),
            (2, 2),
        ] {
            let done: Vec<AtomicBool> = (0..rows * cols).map(|_| AtomicBool::new(false)).collect();
            let runs: Vec<AtomicUsize> = (0..rows * cols).map(|_| AtomicUsize::new(0)).collect();
            let ok = AtomicBool::new(true);
            let idx = |r: usize, c: usize| r * cols + c;
            let out = par_wavefront_pool(4, rows, cols, true, |r, c| {
                // All four causal predecessors — including above-right (r-1,c+1) —
                // must already be finished (the barrier guarantees it).
                let check = |rr: i64, cc: i64| {
                    if rr >= 0
                        && cc >= 0
                        && (rr as usize) < rows
                        && (cc as usize) < cols
                        && !done[idx(rr as usize, cc as usize)].load(Ordering::Acquire)
                    {
                        ok.store(false, Ordering::Relaxed);
                    }
                };
                let (ri, ci) = (r as i64, c as i64);
                check(ri - 1, ci - 1);
                check(ri - 1, ci);
                check(ri - 1, ci + 1); // above-right
                check(ri, ci - 1);
                runs[idx(r, c)].fetch_add(1, Ordering::Relaxed);
                done[idx(r, c)].store(true, Ordering::Release);
                r * cols + c
            });
            assert!(
                ok.load(Ordering::Relaxed),
                "{rows}x{cols}: cell ran before a causal (incl. above-right) neighbour"
            );
            for r in 0..rows {
                for c in 0..cols {
                    assert_eq!(
                        out[idx(r, c)],
                        Some(r * cols + c),
                        "{rows}x{cols}: cell ({r},{c}) missing or produced wrong value"
                    );
                    assert_eq!(
                        runs[idx(r, c)].load(Ordering::Relaxed),
                        1,
                        "{rows}x{cols}: cell ({r},{c}) ran {} times, expected once",
                        runs[idx(r, c)].load(Ordering::Relaxed)
                    );
                }
            }
        }
    }

    #[test]
    fn wavefront_plane_writer_disjoint_blocks_are_correct() {
        // Each SB cell writes its own bs×bs block in parallel via the wavefront;
        // verify every block landed with no corruption or lost writes.
        let (rows, cols, bs) = (6usize, 9usize, 4usize);
        let (ph, pw) = (rows * bs, cols * bs);
        let mut plane = vec![-1f32; ph * pw];
        {
            let writer = PlaneWriter::new(&mut plane, pw);
            par_wavefront(4, rows, cols, |r, c| {
                let val = (r * cols + c) as f32;
                let block = vec![val; bs * bs];
                // SAFETY: each cell writes only its own disjoint bs×bs block.
                unsafe { writer.write_block(r * bs, c * bs, bs, bs, &block) };
            });
        }
        for r in 0..rows {
            for c in 0..cols {
                let val = (r * cols + c) as f32;
                for yy in 0..bs {
                    for xx in 0..bs {
                        assert_eq!(
                            plane[(r * bs + yy) * pw + c * bs + xx],
                            val,
                            "block ({r},{c}) pixel ({yy},{xx}) corrupted"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn tx16_choice_never_spends_distortion_for_proxy_rate() {
        assert_eq!(
            choose_tx16_type(
                [100.0, 90.0, 80.0, 70.0],
                [100.0, 101.0, 102.0, 103.0],
                [false; 4]
            ),
            0
        );
        assert_eq!(
            choose_tx16_type(
                [100.0, 95.0, 90.0, 92.0],
                [100.0, 89.0, 80.0, 85.0],
                [false; 4]
            ),
            2
        );
    }

    fn coeff_rate_case(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed
            .wrapping_mul(747_796_405)
            .wrapping_add((len as u32) << 9)
            .wrapping_add(2_891_336_453);
        let mut out = vec![0.0f32; len];
        for (i, v) in out.iter_mut().enumerate() {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            if (state & 7) == 0 {
                continue;
            }
            let sign = if (state & 8) == 0 { 1.0 } else { -1.0 };
            let mag = ((state >> 10) & 0x7ff) as f32 + ((i & 15) as f32 * 0.125);
            *v = sign * mag;
        }
        out
    }

    fn assert_coeff_rate_impl_matches_scalar(name: &str, simd: CoeffRateF32Fn) {
        for len in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 256, 1024,
        ] {
            for seed in 0..16u32 {
                let levels = coeff_rate_case(len, seed);
                let expected = coeff_rate_f32_scalar(&levels);
                let actual = unsafe { simd(&levels) };
                let tol = 0.001 + expected.abs() * 2.0e-6;
                assert!(
                    (actual - expected).abs() <= tol,
                    "{name} coeff-rate mismatch len {len} seed {seed}: actual={actual} expected={expected} tol={tol}"
                );
            }
        }
    }

    fn assert_coeff_abs_rate_impl_matches_scalar(name: &str, simd: CoeffAbsRateF32Fn) {
        for len in [
            0usize, 1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32, 63, 64, 127, 256, 1024,
        ] {
            for seed in 0..16u32 {
                let levels = coeff_rate_case(len, seed);
                let expected = coeff_abs_rate_f32_scalar(&levels);
                let actual = unsafe { simd(&levels) };
                let tol = 0.001 + expected.abs() * 2.0e-6;
                assert!(
                    (actual - expected).abs() <= tol,
                    "{name} coeff-abs-rate mismatch len {len} seed {seed}: actual={actual} expected={expected} tol={tol}"
                );
            }
        }
    }

    struct MetricCase {
        a8: Vec<u8>,
        b8: Vec<u8>,
        a16: Vec<u16>,
        b16: Vec<u16>,
        af: Vec<f32>,
        bf: Vec<f32>,
        weights: Vec<f32>,
    }

    fn metric_case(len: usize, seed: u32) -> MetricCase {
        let mut state = seed.wrapping_mul(747_796_405).wrapping_add(len as u32);
        let mut a8 = Vec::with_capacity(len);
        let mut b8 = Vec::with_capacity(len);
        let mut a16 = Vec::with_capacity(len);
        let mut b16 = Vec::with_capacity(len);
        let mut af = Vec::with_capacity(len);
        let mut bf = Vec::with_capacity(len);
        let mut w = Vec::with_capacity(len);
        for i in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            a8.push(state as u8);
            b8.push((state >> 8) as u8);
            a16.push((state & 0x0fff) as u16);
            b16.push(((state >> 12) & 0x0fff) as u16);
            af.push(((state & 0x0fff) as f32) + (i & 3) as f32 * 0.25);
            bf.push((((state >> 12) & 0x0fff) as f32) + (i & 7) as f32 * 0.125);
            w.push(0.25 + ((state >> 24) as f32) / 255.0);
        }
        MetricCase {
            a8,
            b8,
            a16,
            b16,
            af,
            bf,
            weights: w,
        }
    }

    #[derive(Clone, Copy)]
    struct MetricImpls {
        name: &'static str,
        sad8: SadU8Fn,
        sse: PixelSseF32Fn,
        sse_u16: PixelSseF32U16Fn,
        weighted: WeightedPixelSseF32Fn,
        stats: SumSumsqF32Fn,
        cfl: CflSseI32Fn,
    }

    fn assert_metric_impls_match_scalar(imp: &MetricImpls) {
        let MetricImpls {
            name,
            sad8,
            sse,
            sse_u16,
            weighted,
            stats,
            cfl,
        } = *imp;
        for len in [0usize, 1, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 255, 1024] {
            for seed in 0..8u32 {
                let MetricCase {
                    a8,
                    b8,
                    a16,
                    b16,
                    af,
                    bf,
                    weights,
                } = metric_case(len, seed);
                assert_eq!(
                    unsafe { sad8(&a8, &b8) },
                    sad_u8_scalar(&a8, &b8),
                    "{name} u8 SAD len={len} seed={seed}"
                );
                let expected = pixel_sse_f32_scalar(&af, &bf);
                let actual = unsafe { sse(&af, &bf) };
                let tol = 2.0e-5 * expected.max(1.0);
                assert!(
                    (actual - expected).abs() <= tol,
                    "{name} f32 SSE len={len} seed={seed}: actual={actual} expected={expected}"
                );

                let expected = pixel_sse_f32_u16_scalar(&af, &b16);
                let actual = unsafe { sse_u16(&af, &b16) };
                let tol = 2.0e-5 * expected.max(1.0);
                assert!(
                    (actual - expected).abs() <= tol,
                    "{name} mixed SSE len={len} seed={seed}: actual={actual} expected={expected}"
                );

                let expected = weighted_pixel_sse_f32_scalar(&af, &bf, &weights);
                let actual = unsafe { weighted(&af, &bf, &weights) };
                let tol = 2.0e-5 * expected.max(1.0);
                assert!(
                    (actual - expected).abs() <= tol,
                    "{name} weighted SSE len={len} seed={seed}: actual={actual} expected={expected}"
                );

                let expected = sum_sumsq_f32_scalar(&af);
                let actual = unsafe { stats(&af) };
                let sum_tol = 2.0e-5 * expected.0.abs().max(1.0);
                let sq_tol = 2.0e-5 * expected.1.abs().max(1.0);
                assert!(
                    (actual.0 - expected.0).abs() <= sum_tol,
                    "{name} sum len={len} seed={seed}"
                );
                assert!(
                    (actual.1 - expected.1).abs() <= sq_tol,
                    "{name} sumsq len={len} seed={seed}"
                );

                let src: Vec<i32> = a16.iter().map(|&v| v as i32).collect();
                let ac: Vec<i32> = b16
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| v as i32 - 2048 + (i as i32 & 7))
                    .collect();
                for alpha in [-256, -96, 0, 32, 160, 256] {
                    let expected = cfl_sse_i32_scalar(&src, &ac, alpha, 1023, 4095);
                    let actual = unsafe { cfl(&src, &ac, alpha, 1023, 4095) };
                    let tol = 2.0e-5 * expected.max(1.0);
                    assert!(
                        (actual - expected).abs() <= tol,
                        "{name} CfL SSE len={len} seed={seed} alpha={alpha}: actual={actual} expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn log2p1_approx_close_to_libm() {
        for i in 0..8192u32 {
            let x = i as f32 * 0.5;
            let actual = log2p1_approx_f32(x);
            let expected = (x + 1.0).log2();
            assert!(
                (actual - expected).abs() <= 4.0e-6,
                "log2p1 mismatch x={x}: actual={actual} expected={expected}"
            );
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn coeff_rate_neon_matches_scalar() {
        assert_coeff_rate_impl_matches_scalar("neon", crate::av2::neon::coeff_rate_f32_neon);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn coeff_abs_rate_neon_matches_scalar() {
        assert_coeff_abs_rate_impl_matches_scalar(
            "neon",
            crate::av2::neon::coeff_abs_rate_f32_neon,
        );
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn rd_metrics_neon_match_scalar() {
        assert_metric_impls_match_scalar(&MetricImpls {
            name: "neon",
            sad8: crate::av2::neon::sad_u8_neon,
            sse: crate::av2::neon::pixel_sse_f32_neon,
            sse_u16: crate::av2::neon::pixel_sse_f32_u16_neon,
            weighted: crate::av2::neon::weighted_pixel_sse_f32_neon,
            stats: crate::av2::neon::sum_sumsq_f32_neon,
            cfl: crate::av2::neon::cfl_sse_i32_neon,
        });
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn coeff_rate_avx2_matches_scalar() {
        if !(std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")) {
            return;
        }
        assert_coeff_rate_impl_matches_scalar("avx2", crate::av2::avx::coeff_rate_f32_avx2);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn coeff_abs_rate_avx2_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        assert_coeff_abs_rate_impl_matches_scalar("avx2", crate::av2::avx::coeff_abs_rate_f32_avx2);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn rd_metrics_avx2_match_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        assert_metric_impls_match_scalar(&MetricImpls {
            name: "avx2",
            sad8: crate::av2::avx::sad_u8_avx2,
            sse: crate::av2::avx::pixel_sse_f32_avx2,
            sse_u16: crate::av2::avx::pixel_sse_f32_u16_avx2,
            weighted: crate::av2::avx::weighted_pixel_sse_f32_avx2,
            stats: crate::av2::avx::sum_sumsq_f32_avx2,
            cfl: crate::av2::avx::cfl_sse_i32_avx2,
        });
    }
}
