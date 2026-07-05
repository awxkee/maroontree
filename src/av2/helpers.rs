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
                .min(63)) as u8;
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
        // clamp is a no-op and behaviour is unchanged.
        let in_cols = (mc - cx as i64).clamp(0, 8) as usize;
        let in_rows = (mr - cy as i64).clamp(0, 8) as usize;
        for k in 0..8 {
            above[cx + k] = if k < in_cols { res } else { 0x40 };
            left[cy + k] = if k < in_rows { res } else { 0x40 };
        }
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
                .min(63) as u8;
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
        for k in 0..8 {
            above[cx + k] = if k < in_cols { res } else { 0x40 };
            left[cy + k] = if k < in_rows { res } else { 0x40 };
        }
    }
    (skip_cdfs, dc_sign_ctxs)
}

/// Per-TU skip / DC-sign contexts for an arbitrary list of TX_32X32 sub-TUs within
/// an SB (used by the 32X64 / 32X32 partition leaves). `pos` holds the SB-relative
/// pixel offsets in coding order; updates `above`/`left` coeff-context arrays with
/// the same edge-clamp as `sb_tu_contexts`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sb_tu_contexts_pos(
    pos: &[(usize, usize)],
    tus: &[Vec<Coeff>],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    qc: usize,
    mc: i64,
    mr: i64,
    block_eq_tx: bool,
) -> (Vec<u32>, Vec<usize>) {
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
                .min(63) as u8;
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
        for k in 0..8 {
            above[cx + k] = if k < in_cols { res } else { 0x40 };
            left[cy + k] = if k < in_rows { res } else { 0x40 };
        }
    }
    (skip_cdfs, dc_sign_ctxs)
}

/// Context for a single rectangular luma TU (16-tap family: TX_16X64 4×16 mi,
/// TX_64X16 16×4 mi, TX_16X16 4×4 mi). `wu`/`hu` are the tx width/height in mi units
/// (tx_size_wide_unit/high_unit). Skip ctx is 0 when `block_eq_tx` (block == tx, which
/// holds for all single-TX 16-family leaves); dc_sign ctx sums neighbor sign bits over
/// the tx units. Updates `wu` above + `hu` left entries with this TU's cul/DC byte.
/// Returns `(skip_cdf, dc_sign_ctx)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sb_tu_contexts_rect(
    tu: &[Coeff],
    sb_y: usize,
    sb_x: usize,
    above: &mut [u8],
    left: &mut [u8],
    qc: usize,
    mc: i64,
    mr: i64,
    wu: usize,
    hu: usize,
    block_eq_tx: bool,
) -> (u32, usize) {
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
    } else if wu.min(hu) == 2 && wu.max(hu) >= 4 {
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
            .min(63) as u8;
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
    for k in 0..wu {
        above[cx + k] = if k < in_cols { res } else { 0x40 };
    }
    for k in 0..hu {
        left[cy + k] = if k < in_rows { res } else { 0x40 };
    }
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
    for y in 0..ph {
        let sy = y.min(h - 1);
        for x in 0..pw {
            out[y * pw + x] = src[sy * w + x.min(w - 1)];
        }
    }
    out
}
/// SB-aligned (multiple of 64) size for a given dimension.
pub(crate) fn sb_align(n: usize) -> usize {
    n.div_ceil(64) * 64
}

/// DC prediction for a `bw`-wide × `bh`-tall block (4:2:2 chroma is 32×64).
pub(crate) fn dc_pred(rec: &[f32], w: usize, y0: usize, x0: usize, bs: usize, neutral: f32) -> f32 {
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
pub(crate) fn get_residual(
    plane: &[f32],
    w: usize,
    y0: usize,
    x0: usize,
    bs: usize,
    pred: f32,
) -> Vec<f32> {
    let mut r = vec![0f32; bs * bs];
    for yy in 0..bs {
        for (dst, &src) in r[yy * bs..yy * bs + bs]
            .iter_mut()
            .zip(plane[(y0 + yy) * w + x0..(y0 + yy) * w + x0 + bs].iter())
        {
            *dst = src - pred;
        }
    }
    r
}
pub(crate) fn put_block(plane: &mut [f32], w: usize, y0: usize, x0: usize, bs: usize, rec: &[f32]) {
    for yy in 0..bs {
        plane[(y0 + yy) * w + x0..(y0 + yy) * w + x0 + bs]
            .copy_from_slice(&rec[yy * bs..yy * bs + bs]);
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
    let (ha, hl) = (y0 > 0, x0 > 0);
    let sa: i64 = if ha {
        (0..bw).map(|i| rec[(y0 - 1) * w + x0 + i] as i64).sum()
    } else {
        0
    };
    let sl: i64 = if hl {
        (0..bh).map(|i| rec[(y0 + i) * w + x0 - 1] as i64).sum()
    } else {
        0
    };
    // avm `highbd_dc_predictor` (reconintra.h) averages the `count` reference
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
    for yy in 0..bh {
        for xx in 0..bw {
            r[yy * bw + xx] = plane[(y0 + yy) * w + x0 + xx] - pred;
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
    for yy in 0..bh {
        let plane_dst = &mut plane[(y0 + yy) * w + x0..(y0 + yy) * w + x0 + bw];
        let plane_src = &rec[yy * bw..yy * bw + bw];
        plane_dst.copy_from_slice(plane_src);
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
pub(crate) fn sq_diff_u64(a: i32, b: i32) -> u64 {
    let d = a as i64 - b as i64;
    (d * d) as u64
}

#[inline]
pub(crate) fn pixel_sse_rounded(a: &[f32], b: &[f32]) -> u64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| sq_diff_u64(pixel_to_i32(x), pixel_to_i32(y)))
        .sum()
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
) -> u64 {
    let mut sse = 0u64;
    for r in 0..h {
        let src_row = &src[(src_y + r) * src_stride + src_x..][..w];
        let rec_row = &rec[r * rec_stride..][..w];
        sse += pixel_sse_rounded(src_row, rec_row);
    }
    sse
}

pub(crate) type CoeffRateF32Fn = unsafe fn(&[f32]) -> f32;
pub(crate) type CoeffAbsRateF32Fn = unsafe fn(&[f32]) -> f32;

static COEFF_RATE_F32: OnceLock<CoeffRateF32Fn> = OnceLock::new();
static COEFF_ABS_RATE_F32: OnceLock<CoeffAbsRateF32Fn> = OnceLock::new();

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
pub(crate) fn coeff_count_rate_f32(lev: &[f32], bits_per_nonzero: f32) -> f32 {
    lev.iter().filter(|&&v| v != 0.0).count() as f32 * bits_per_nonzero
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
                    .min(63) as u8;
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
            for yy in 0..4 {
                let row = (y0 + yy) * pw + x0;
                for xx in 0..4 {
                    resid[yy * 4 + xx] = (src[row + xx] - pred) as i32;
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
                .min(63) as u8;
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
