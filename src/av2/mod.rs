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
mod avif;
mod cdfs_qctx;
mod cdfx_4tx;
mod coder;
mod entropy;
mod headers;
mod helpers;
mod intrapred;
mod itx422;
mod layout;
mod lossless;
mod partition;
mod proj;
mod quant;
mod tables;
mod tables_tx32;
mod wht;

use crate::av2::avif::{Av2Color, Av2Format};
use crate::av2::cdfs_qctx::{
    CHROMA_EOB_HI_BIT_QC, CHROMA_EOB256_QC, CHROMA_EOB512_QC, CHROMA_SKIP_TX32_QC,
    CHROMA_SKIP_V_QC, SKIP_TX16_QC,
};
use crate::av2::cdfx_4tx::{TXB_SKIP_TX4_Q0, V_TXB_SKIP_TX4_Q0};
use crate::av2::coder::{
    Coeff, encode_chroma_block, encode_chroma_block_rect, encode_chroma_tu4,
    encode_lossless_luma_sb, encode_luma_block_split, encode_luma_leaf_16x16,
    encode_luma_leaf_16x64, encode_luma_leaf_32x32, encode_luma_leaf_32x64, encode_luma_leaf_64x16,
    encode_luma_leaf_64x32,
};
use crate::av2::entropy::RangeEncoder;
use crate::av2::headers::{Config, frame_header, obu, sequence_header};
use crate::av2::helpers::{
    dc_pred, dc_pred_rect, get_residual, get_residual_rect, levels_to_coeffs, lossless_sb_tus,
    pad_plane, put_block, put_block_rect, sb_align, sb_tu_contexts, sb_tu_contexts_64x32,
    sb_tu_contexts_pos, sb_tu_contexts_rect, sb_tu4_chroma_skip, sb_tu4_contexts,
};
use crate::av2::layout::Layout;
use crate::av2::proj::Basis;
use crate::av2::tables::{SCAN16, SCAN16X32, SCAN32X16};
use crate::err::EncodeError;
use crate::{ChromaFormat, ColorEncoding, Pixel, PlanarImage};

/// Build the prediction block for luma candidate `m` (0=DC, 1=SMOOTH, 4=PAETH)
/// at TX index `i` (raster within the 64x64 SB) and pixel origin `(y0,x0)`.
#[allow(clippy::too_many_arguments)]
fn predict_luma(
    recy: &[f32],
    pw: usize,
    width: usize,
    height: usize,
    i: usize,
    y0: usize,
    x0: usize,
    m: usize,
    neutral: f32,
) -> Vec<f32> {
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    // Exact avm reference-sample availability for TX_32X32 in a 64x64
    // PARTITION_NONE luma block (has_top_right / has_bottom_left specialised;
    // mi_cols/mi_rows = round_up(dim,8)>>2; single tile). `i` is raster TX index.
    // avm tile bounds are superblock-aligned: tile.mi_col_end =
    // sb_cols<<mib_size_log2 = ((dim+63)&~63)>>2, NOT the 8-pixel-aligned
    // mi_params value. Using the SB-aligned bound is what makes a transform
    // block's top-right / bottom-left correctly resolve to already-decoded
    // samples in the same superblock (e.g. TX(32,0) reads TX(0,32)'s edge).
    let mi_col_end = (((width + 63) & !63) >> 2) as i64;
    let mi_row_end = (((height + 63) & !63) >> 2) as i64;
    let sb_y0 = (y0 / 64) * 64;
    let sb_x0 = (x0 / 64) * 64;
    let mi_row = (sb_y0 >> 2) as i64;
    let mi_col = (sb_x0 >> 2) as i64;
    let (row_off, col_off) = ((y0 - sb_y0) as i64 / 4, (x0 - sb_x0) as i64 / 4);
    let (lx, ly) = ((x0 - sb_x0) as i64, (y0 - sb_y0) as i64); // TX offset px within SB
    let xr = ((mi_col_end - mi_col - 16) << 2) + 32 - lx;
    let yd = ((mi_row_end - mi_row - 16) << 2) + 32 - ly;
    let right_available = (mi_col + col_off + 8) < mi_col_end;
    let bottom_available = (yd > 0) && ((mi_row + row_off + 8) < mi_row_end);
    // top-right: needed by TX 0/1/2 (TX 3 has col_off+txw==block width -> none)
    let tr_ok = matches!(i, 0 | 1 | 2) && have_above && right_available && xr > 0;
    let tr_px = if tr_ok {
        (xr.min(32)).max(0) as usize
    } else {
        0
    };
    // bottom-left: only TX 0 (others sit at/under the block's bottom-left edge)
    let bl_ok = i == 0 && have_left && bottom_available && yd > 0;
    let bl_px = if bl_ok {
        (yd.min(32)).max(0) as usize
    } else {
        0
    };
    let (ab, lf, corner) =
        intrapred::build_refs(recy, pw, y0, x0, 32, have_above, have_left, tr_px, bl_px);
    if m == 1 {
        intrapred::smooth(32, &ab, &lf)
    } else {
        intrapred::paeth(32, &ab, &lf, corner)
    }
}

/// Encode one 64x64 luma superblock as a single PARTITION_NONE block with a
/// per-SB intra mode chosen from {DC, SMOOTH, PAETH}. Each candidate is fully
/// trialled (4x TX_32X32 with intra-SB reconstruction feedback); the mode with
/// the smallest total coefficient magnitude (rate proxy) wins. Mutates `recy`
/// (leaving the winner's reconstruction) and returns the four TX coefficient
/// lists plus the chosen `mode_idx` (0=DC, 1=SMOOTH, 4=PAETH).
#[allow(clippy::too_many_arguments)]
fn encode_luma_sb(
    recy: &mut [f32],
    yp: &[f32],
    pw: usize,
    width: usize,
    height: usize,
    sb_y: usize,
    sb_x: usize,
    luma: &Basis,
    qstep: i32,
    scan: &[u16],
    neutral: f32,
) -> ([Vec<Coeff>; 4], usize) {
    const POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 64 * 64];
    let cands: &[usize] = &[0usize, 1, 4];
    for &m in cands {
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, &(ty, tx)) in POS.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x + tx);
            let pblk = predict_luma(recy, pw, width, height, i, y0, x0, m, neutral);
            let mut resid = vec![0f32; 1024];
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = luma.project(&resid, 0.0);
            // Bit-cost estimate: each nonzero coefficient costs roughly a
            // significance+sign pair plus a magnitude term ~ log2(|lev|).
            // This tracks coded size far better than Sum|lev|, which
            // over-rewards numerous tiny-magnitude coefficients.
            cost += lev
                .iter()
                .filter(|&&v| v != 0.0)
                .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                .sum::<f64>();
            let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[i] = levels_to_coeffs(&lev);
        }
        // Mode-signaling cost (once per 64x64 block). DC is the cheapest to
        // signal; SMOOTH/PAETH cost a few extra bits, so only pick them when
        // they save more than they cost.
        if m != 0 {
            cost += 6.0;
        }
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_tus = tus;
            for ry in 0..64 {
                let dst = ry * 64;
                let src = (sb_y + ry) * pw + sb_x;
                best_region[dst..dst + 64].copy_from_slice(&recy[src..src + 64]);
            }
        }
    }
    for ry in 0..64 {
        let src = ry * 64;
        let dst = (sb_y + ry) * pw + sb_x;
        recy[dst..dst + 64].copy_from_slice(&best_region[src..src + 64]);
    }
    (best_tus, best_mode)
}

/// Intra prediction for one TX_32X32 of a bottom-edge 64x32 luma leaf. `ti` is the
/// sub-TU index (0=left, 1=right) within the SB-wide leaf at (`sb_y`,`sb_x`).
/// Unlike `predict_luma`, availability uses the NATIVE mi grid (`mi_cols`,`mi_rows`)
/// and bottom-left is always off: the leaf is the bottom partition, so everything
/// below it is out of frame / not yet decoded.
#[allow(clippy::too_many_arguments)]
fn predict_luma_leaf32(
    recy: &[f32],
    pw: usize,
    mi_cols: i64,
    _mi_rows: i64,
    sb_y: usize,
    sb_x: usize,
    ti: usize,
    m: usize,
    neutral: f32,
) -> Vec<f32> {
    let (y0, x0) = (sb_y, sb_x + ti * 32);
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    let mi_col = (sb_x >> 2) as i64;
    let lx = (ti * 32) as i64;
    let col_off = lx / 4;
    // top-right reference width (px), clamped to 32. Same geometry as predict_luma
    // but with the native column bound.
    let xr = ((mi_cols - mi_col - 16) << 2) + 32 - lx;
    let right_available = (mi_col + col_off + 8) < mi_cols;
    let tr_ok = have_above && right_available && xr > 0;
    let tr_px = if tr_ok {
        (xr.min(32)).max(0) as usize
    } else {
        0
    };
    let (ab, lf, corner) =
        intrapred::build_refs(recy, pw, y0, x0, 32, have_above, have_left, tr_px, 0);
    if m == 1 {
        intrapred::smooth(32, &ab, &lf)
    } else {
        intrapred::paeth(32, &ab, &lf, corner)
    }
}

/// Project + trial-code a bottom-edge 64x32 luma leaf as two side-by-side TX_32X32.
/// Mirrors `encode_luma_sb` (mode trial over {DC,SMOOTH,PAETH} with intra-leaf
/// reconstruction feedback) but only the top two TUs and with leaf-aware prediction.
/// Mutates `recy` (top 64x32 of the SB) and returns the two TU coefficient lists
/// plus the chosen mode.
#[allow(clippy::too_many_arguments)]
fn encode_luma_leaf32(
    recy: &mut [f32],
    yp: &[f32],
    pw: usize,
    mi_cols: i64,
    mi_rows: i64,
    sb_y: usize,
    sb_x: usize,
    luma: &Basis,
    qstep: i32,
    scan: &[u16],
    neutral: f32,
) -> ([Vec<Coeff>; 2], usize) {
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 64 * 32];
    for &m in &[0usize, 1, 4] {
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for ti in 0..2 {
            let (y0, x0) = (sb_y, sb_x + ti * 32);
            let pblk = predict_luma_leaf32(recy, pw, mi_cols, mi_rows, sb_y, sb_x, ti, m, neutral);
            let mut resid = vec![0f32; 1024];
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = luma.project(&resid, 0.0);
            cost += lev
                .iter()
                .filter(|&&v| v != 0.0)
                .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                .sum::<f64>();
            let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[ti] = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_tus = tus;
            for ry in 0..32 {
                let src = (sb_y + ry) * pw + sb_x;
                best_region[ry * 64..ry * 64 + 64].copy_from_slice(&recy[src..src + 64]);
            }
        }
    }
    for ry in 0..32 {
        let dst = (sb_y + ry) * pw + sb_x;
        recy[dst..dst + 64].copy_from_slice(&best_region[ry * 64..ry * 64 + 64]);
    }
    (best_tus, best_mode)
}

/// General intra prediction for one TX_32X32 sub-block of a partition leaf, using
/// the NATIVE mi grid for reference availability. `(ty,tx)` is the TU's pixel offset
/// within the SB; `i` is the equivalent 64x64-raster index that selects avm's
/// top-right (i∈{0,1,2}) / bottom-left (i==0) eligibility rules.
#[allow(clippy::too_many_arguments)]
fn predict_luma_leaf_tu(
    recy: &[f32],
    pw: usize,
    mc: i64,
    mr: i64,
    sb_y: usize,
    sb_x: usize,
    ty: usize,
    tx: usize,
    i: usize,
    m: usize,
    neutral: f32,
) -> Vec<f32> {
    let (y0, x0) = (sb_y + ty, sb_x + tx);
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    let mi_col = (sb_x >> 2) as i64;
    let mi_row = (sb_y >> 2) as i64;
    let (lx, ly) = (tx as i64, ty as i64);
    let (col_off, row_off) = (lx / 4, ly / 4);
    let xr = ((mc - mi_col - 16) << 2) + 32 - lx;
    let yd = ((mr - mi_row - 16) << 2) + 32 - ly;
    let right_available = (mi_col + col_off + 8) < mc;
    let bottom_available = (yd > 0) && ((mi_row + row_off + 8) < mr);
    let tr_ok = matches!(i, 0 | 1 | 2) && have_above && right_available && xr > 0;
    let tr_px = if tr_ok {
        (xr.min(32)).max(0) as usize
    } else {
        0
    };
    let bl_ok = i == 0 && have_left && bottom_available && yd > 0;
    let bl_px = if bl_ok {
        (yd.min(32)).max(0) as usize
    } else {
        0
    };
    let (ab, lf, corner) =
        intrapred::build_refs(recy, pw, y0, x0, 32, have_above, have_left, tr_px, bl_px);
    if m == 1 {
        intrapred::smooth(32, &ab, &lf)
    } else {
        intrapred::paeth(32, &ab, &lf, corner)
    }
}

/// Project + trial-code a right-edge 32x64 luma leaf as two stacked TX_32X32
/// (top i=0, bottom i=2). Mirrors `encode_luma_leaf32` but vertical.
#[allow(clippy::too_many_arguments)]
fn encode_luma_leaf_v32x64(
    recy: &mut [f32],
    yp: &[f32],
    pw: usize,
    mc: i64,
    mr: i64,
    sb_y: usize,
    sb_x: usize,
    luma: &Basis,
    qstep: i32,
    scan: &[u16],
    neutral: f32,
) -> ([Vec<Coeff>; 2], usize) {
    let tu_i = [(0usize, 0usize), (32usize, 2usize)]; // (ty, raster-i)
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 32 * 64];
    for &m in &[0usize, 1, 4] {
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for (k, &(ty, i)) in tu_i.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x);
            let pblk = predict_luma_leaf_tu(recy, pw, mc, mr, sb_y, sb_x, ty, 0, i, m, neutral);
            let mut resid = vec![0f32; 1024];
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = luma.project(&resid, 0.0);
            cost += lev
                .iter()
                .filter(|&&v| v != 0.0)
                .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                .sum::<f64>();
            let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[k] = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_tus = tus;
            for ry in 0..64 {
                let src = (sb_y + ry) * pw + sb_x;
                best_region[ry * 32..ry * 32 + 32].copy_from_slice(&recy[src..src + 32]);
            }
        }
    }
    for ry in 0..64 {
        let dst = (sb_y + ry) * pw + sb_x;
        recy[dst..dst + 32].copy_from_slice(&best_region[ry * 32..ry * 32 + 32]);
    }
    (best_tus, best_mode)
}

/// Project + trial-code a corner 32x32 luma leaf as a single TX_32X32 (i=0).
#[allow(clippy::too_many_arguments)]
fn encode_luma_leaf_s32x32(
    recy: &mut [f32],
    yp: &[f32],
    pw: usize,
    mc: i64,
    mr: i64,
    sb_y: usize,
    sb_x: usize,
    luma: &Basis,
    qstep: i32,
    scan: &[u16],
    neutral: f32,
) -> (Vec<Coeff>, usize) {
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tu: Vec<Coeff> = Vec::new();
    let mut best_region = vec![0f32; 32 * 32];
    for &m in &[0usize, 1, 4] {
        let pblk = predict_luma_leaf_tu(recy, pw, mc, mr, sb_y, sb_x, 0, 0, 0, m, neutral);
        let mut resid = vec![0f32; 1024];
        for r in 0..32 {
            let base = (sb_y + r) * pw + sb_x;
            for c in 0..32 {
                resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
            }
        }
        let lev = luma.project(&resid, 0.0);
        let mut cost: f64 = lev
            .iter()
            .filter(|&&v| v != 0.0)
            .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
            .sum();
        if m != 0 {
            cost += 6.0;
        }
        let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_tu = levels_to_coeffs(&lev);
            best_region.copy_from_slice(&rb);
        }
    }
    put_block(recy, pw, sb_y, sb_x, 32, &best_region);
    (best_tu, best_mode)
}

// Q0.13 coefficients  (value = round(f * 8192))
const Q: i32 = 13;
const HALF: i32 = 1 << (Q - 1); // 0.5 rounding bias

const Y_R: i32 = 2449; // round( 0.299    * 8192)
const Y_G: i32 = 4809; // round( 0.587    * 8192)
const Y_B: i32 = 934; // round( 0.114    * 8192)

const CB_R: i32 = -1382; // round(-0.168736 * 8192)
const CB_G: i32 = -2714; // round(-0.331264 * 8192)
const CB_B: i32 = 4096; // round( 0.5 * 8192)

const CR_R: i32 = 4096; // round( 0.5 * 8192)
const CR_G: i32 = -3430; // round(-0.418688 * 8192)
const CR_B: i32 = -666; // round(-0.081312 * 8192)

const MAX_DIM: u32 = 65535;
const MIN_DIM: u32 = 1;

pub fn get_q_ctx(q: u8) -> usize {
    if q <= 90 {
        0
    } else if q <= 140 {
        1
    } else if q <= 190 {
        2
    } else {
        3
    }
}

fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

/// Result of an encode: the AV2 bitstream plus the metadata needed to interpret it.
pub struct Av2Frame {
    data: Vec<u8>,
    width: usize,
    height: usize,
    /// Coded (decoder-output) dimensions = the size signaled in the OBU. Equal to
    /// width/height for lossless and SB-aligned lossy; for padded lossy they are the
    /// 64-aligned size, and the AVIF muxer adds a `clap` box cropping to width/height.
    coded_width: usize,
    coded_height: usize,
    bit_depth: u8,
    color: ColorEncoding,
    chroma_format: ChromaFormat,
}

impl Av2Frame {
    pub fn view(&self) -> &[u8] {
        self.data.as_slice()
    }
}

/// A reusable still-image encoder configured for one quality.
///
/// `Av2Encoder::new(q)` loads the bundled q120 bases and rescales them to the target
/// `base_q_idx` once (see [`proj::Bases::rescaled_to_q`]); the per-superblock encode
/// then reuses that precomputed set. Lower `base_q_idx` → finer quantizer → larger,
/// higher-quality output; higher → coarser/smaller.
pub struct Av2Encoder {
    bases: proj::Bases,
    base_q_idx: u8,
    bit_depth: u8,
}

/// Returns the AV2 mi-unit frame extents `(mc, mr)` for a native (no-pad) lossy 4:4:4
/// encode, iff both dimensions are "boundary-safe". A dimension is boundary-safe when
/// the last superblock has >8 mi in-frame: mc%16==0 || mc%16>8, where mc =
/// ALIGN_POWER_OF_TWO(W,3)>>2 (avm's mi_cols). Returns None if either dimension is
/// not boundary-safe; the encoder then falls back to padding.
fn lossy_native_mi(width: usize, height: usize) -> Option<(i64, i64)> {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    // The mi grid is 8-px aligned, so mc/mr are always even; the right/bottom SB has
    // (m mod 16) mi in frame. Supported partial-edge residues:
    //   0,10,12,14  → whole 64X64 leaves (m%16==0, or >8 so the implied split never
    //                 triggers; ≥9 mi in frame, coded with edge-clamped TUs);
    //   6,8         → 32-family force-split leaves (32X64 / 64X32 / 32X32 corner);
    //   4           → 16-tap family: 16X64 (right) / 64X16 (bottom) single edges, and
    //                 the 16X16 corner when BOTH dims are residue 4 (DC-only luma).
    // A residue-4 edge combined with a residue-{6,8} edge would need a 16X32 / 32X16
    // corner that is not built yet, so those fall back to padding+clap. Residue 2
    // (8px edge) also still falls back.
    let ok = |m: i64| m % 16 == 0 || m % 16 >= 6 || m % 16 == 4;
    if !(ok(mc) && ok(mr)) {
        return None;
    }
    // residue-4 in one dim is only supported when the perpendicular dim is a whole SB
    // (residue 0) or also residue 4 (→ 16X16 corner); a 6/8 perpendicular is unsupported.
    let perp_ok = |a: i64, b: i64| a % 16 != 4 || b % 16 == 0 || b % 16 == 4;
    if !(perp_ok(mc, mr) && perp_ok(mr, mc)) {
        return None;
    }
    Some((mc, mr))
}

/// True when the size needs a force-split partition walk (any edge residue in
/// {6,8}); residues {0,10,12,14} tile into whole 64X64 leaves and use the fast path.
fn lossy_needs_partition(width: usize, height: usize) -> bool {
    let mc = (((width + 7) & !7) / 4) as i64;
    let mr = (((height + 7) & !7) / 4) as i64;
    let part = |m: i64| m % 16 == 6 || m % 16 == 8 || m % 16 == 4;
    part(mc) || part(mr)
}

impl Av2Encoder {
    /// Build an 8-bit encoder for `base_q_idx`. Honors the `BASES` env override for
    /// the source basis file, otherwise uses the embedded q120 set, then rescales.
    pub fn new(base_q_idx: u8) -> Self {
        Self::with_bit_depth(base_q_idx, 8)
    }

    /// Build an encoder for `base_q_idx` at a given coded bit depth (8, 10 or 12).
    /// The avm quantiser step is bit-depth-independent, so only the sample range,
    /// reconstruction clamp, DC-prediction neutral and the sequence-header signalling
    /// differ; the bases are unchanged.
    pub fn with_bit_depth(base_q_idx: u8, bit_depth: u8) -> Self {
        assert!(
            matches!(bit_depth, 8 | 10 | 12),
            "bit_depth must be 8, 10 or 12, got {bit_depth}"
        );
        let mut bases = match std::env::var("BASES") {
            Ok(p) => proj::load_bases(&p),
            Err(_) => proj::default_bases(),
        }
        .rescaled_to_q(base_q_idx as u32);
        bases.set_bit_depth(bit_depth);
        Av2Encoder {
            bases,
            base_q_idx,
            bit_depth,
        }
    }

    /// The quality this encoder is configured for.
    pub fn base_q_idx(&self) -> u8 {
        self.base_q_idx
    }

    fn config(&self, layout: Layout) -> Config {
        Config {
            layout,
            base_q: self.base_q_idx as u32,
            deblock: false,
            delta_q: 0,
            tx_switchable: true,
            guided_deblock: None,
            bit_depth: self.bit_depth,
            lossless: self.base_q_idx == 0,
        }
    }

    /// DC-prediction neutral value for the first block (1 << (bit_depth-1)).
    fn dc_neutral(&self) -> f32 {
        (1u32 << (self.bit_depth - 1)) as f32
    }

    /// Resolve a caller-supplied thread budget: `0` = use all available cores,
    /// `1` = serial, `N` = up to N threads. Replaces the old `SLIMAV_THREADS` env.
    fn resolve_threads(threads: usize) -> usize {
        if threads == 0 {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        } else {
            threads
        }
    }

    /// Encode a 4:4:4 YCbCr still. `y`, `cb`, `cr` are full-resolution
    /// (`width × height`). Luma is four 32x32 transform units per 64x64 superblock;
    /// each chroma plane is one 64x64 transform per superblock.
    pub fn encode_yuv444<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        if self.base_q_idx == 0 {
            return self.encode_yuv444_lossless(planar_image, color, threads);
        }
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        // Native-size 444: boundary-safe non-aligned sizes can signal real W×H so the
        // decoder reconstructs the full padded SB and crops — no AVIF clap box needed.
        let (tmc, tmr) =
            lossy_native_mi(width, height).unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);

        let layout = Layout::I444;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pw * ph];
        let mut recv = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];
        let qstep_i = crate::av2::quant::qstep(self.base_q_idx as u32) as i32;
        // Bottom-edge force-split: the last SB row is 32 px tall in frame, so each
        // 64X64 force-splits HORZ (implied, no bits) into a top 64X32 leaf coded by
        // the partition leaf path. Partition context `above_pctx` persists down
        // columns; `left_pctx` is len-16 and reset per SB row.
        // Force-split partition walk. When any edge residue is 6 or 8 the right/bottom
        // SBs split into 32-family leaves (32X64 / 64X32 / 32X32); otherwise every SB is
        // a whole 64X64. The walk drives `sb_partition_ops`, which also maintains the
        // partition contexts (`above_pctx` down columns, `left_pctx` reset per SB row).
        let needs_partition =
            lossy_native_mi(width, height).is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = vec![0u8; 16];

        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                // Chroma neighbour (above/left) coeff-present contexts from the SB grid.
                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };

                // Helper closures capture nothing mutable; chroma coeff encode is inlined
                // per leaf because basis/size/skip-table differ.
                if !needs_partition {
                    // Fast path: whole 64X64 SB.
                    let (tus, mode_idx) = encode_luma_sb(
                        &mut recy,
                        &yp,
                        pw,
                        width,
                        height,
                        sb_y,
                        sb_x,
                        &bases.luma,
                        qstep_i,
                        &crate::av2::tables::SCAN,
                        neutral,
                    );
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr);
                    encode_luma_block_split(
                        &mut enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        true,
                        12276,
                    );
                    let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                    let levu = bases
                        .chroma444
                        .project(&get_residual(&up, pw, sb_y, sb_x, 64, predu), 0.0);
                    put_block(
                        &mut recu,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        &bases.chroma444.reconstruct(predu, &levu),
                    );
                    let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                    let levv = bases
                        .chroma444
                        .project(&get_residual(&vp, pw, sb_y, sb_x, 64, predv), 0.0);
                    put_block(
                        &mut recv,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        &bases.chroma444.reconstruct(predv, &levv),
                    );
                    let ucoeffs = levels_to_coeffs(&levu);
                    let vcoeffs = levels_to_coeffs(&levv);
                    let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                    encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                    let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                    let v_skip =
                        CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                    encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                    u_has[row * sb_cols + col] = u_present as i32;
                    v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
                    continue;
                }

                // Walk + dispatch. For residues {6,8} each SB yields exactly one Leaf and
                // no RectType ops; RectType is handled generically for forward-compat.
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    tmr as usize,
                    tmc as usize,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    let (bw_mi, bh_mi, pc) = match op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(*cdf, *val);
                            continue;
                        }
                        partition::Op::Leaf {
                            bw_mi,
                            bh_mi,
                            part_cdf,
                            ..
                        } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276)),
                    };
                    let (u_present, v_present) = match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx) = encode_luma_sb(
                                &mut recy,
                                &yp,
                                pw,
                                width,
                                height,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                                &tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_block_split(
                                &mut enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                pc,
                            );
                            let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                            let levu = bases
                                .chroma444
                                .project(&get_residual(&up, pw, sb_y, sb_x, 64, predu), 0.0);
                            put_block(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                &bases.chroma444.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                            let levv = bases
                                .chroma444
                                .project(&get_residual(&vp, pw, sb_y, sb_x, 64, predv), 0.0);
                            put_block(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                &bases.chroma444.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 8) => {
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_64x32(
                                &tus2, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_leaf_64x32(
                                &mut enc, &tus2, &skip2, &dcs2, mode_idx, true, pc,
                            );
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 64, 32, neutral);
                            let levu = bases.chroma444_64x32.project(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 64, 32, predu),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &bases.chroma444_64x32.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 64, 32, neutral);
                            let levv = bases.chroma444_64x32.project(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 32, predv),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &bases.chroma444_64x32.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 16) => {
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0), (32, 0)],
                                &tus2,
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                false,
                            );
                            let s2 = [skip2[0], skip2[1]];
                            let d2 = [dcs2[0], dcs2[1]];
                            encode_luma_leaf_32x64(&mut enc, &tus2, &s2, &d2, mode_idx, true, pc);
                            // chroma TX_32X64 (32 wide x 64 tall): chroma422 basis, TX64 skip ctx.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 32, 64, neutral);
                            let levu = bases.chroma422.project(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 32, 64, predu),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &bases.chroma422.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 32, 64, neutral);
                            let levv = bases.chroma422.project(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 32, 64, predv),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &bases.chroma422.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 8) => {
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let tus_v = [tu.clone()];
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0)],
                                &tus_v,
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            encode_luma_leaf_32x32(
                                &mut enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc,
                            );
                            // chroma TX_32X32: chroma420 basis, TX32 U skip, shared V skip.
                            let predu = dc_pred(&recu, pw, sb_y, sb_x, 32, neutral);
                            let levu = bases
                                .chroma420
                                .project(&get_residual(&up, pw, sb_y, sb_x, 32, predu), 0.0);
                            put_block(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                &bases.chroma420.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred(&recv, pw, sb_y, sb_x, 32, neutral);
                            let levv = bases
                                .chroma420
                                .project(&get_residual(&vp, pw, sb_y, sb_x, 32, predv), 0.0);
                            put_block(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                &bases.chroma420.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = crate::av2::cdfs_qctx::CHROMA_SKIP_TX32_QC[qc]
                                [(6 + ua + ul) as usize]
                                as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (4, 16) => {
                            // Right-edge 16×64 luma leaf (residue 4), DC pred (mode 0),
                            // single TX_16X64, coeff region 16×32 (SCAN16X32, eob 512).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 64, neutral);
                            let lev = bases.luma16x64.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 64, pred),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(pred, &lev, &SCAN16X32),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 16, true,
                            );
                            encode_luma_leaf_16x64(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 16×64 (TX_16X64): reuse luma16x64 basis for projection
                            // (validity-only); chroma eob class 512, TX_32X32 skip ctx.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 16, 64, neutral);
                            let levu = bases.luma16x64.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 16, 64, predu),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(predu, &levu, &SCAN16X32),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 16, 64, neutral);
                            let levv = bases.luma16x64.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 16, 64, predv),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(predv, &levv, &SCAN16X32),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN16X32,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN16X32,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 4) => {
                            // Bottom-edge 64×16 luma leaf (residue 4), DC pred, single
                            // TX_64X16, coeff region 32×16 (SCAN32X16, eob 512).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 64, 16, neutral);
                            let lev = bases.luma64x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 64, 16, pred),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(pred, &lev, &SCAN32X16),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 16, 4, true,
                            );
                            encode_luma_leaf_64x16(&mut enc, &tu, skip, dcs, 0, true, pc);
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 64, 16, neutral);
                            let levu = bases.luma64x16.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 64, 16, predu),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(predu, &levu, &SCAN32X16),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 64, 16, neutral);
                            let levv = bases.luma64x16.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 16, predv),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(predv, &levv, &SCAN32X16),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN32X16,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN32X16,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner leaf (residue 4 in both dims).
                            // Luma is DC-only (eob count 1) so the decoder skips the
                            // EXT_NEW_TX_SET tx_type; chroma codes full AC (tx_type is
                            // luma-only). TX_16X16 = entropy class 2, eob class 256.
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 16, neutral);
                            let mut lev = bases.luma16x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 16, pred),
                                0.0,
                                &SCAN16,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0; // keep DC only → eob count 1
                            }
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(pred, &lev, &SCAN16),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 4, true,
                            );
                            // TX_16X16 luma skip = class-2 cdf, block_eq_tx → ctx 0.
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_16x16(&mut enc, dc_level, skip, dcs, 0, true, pc);
                            // chroma 16×16 (TX_16X16): full AC, reuse luma16x16 basis,
                            // chroma eob class 256, class-2 U skip / shared V skip.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 16, 16, neutral);
                            let levu = bases.luma16x16.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 16, 16, predu),
                                0.0,
                                &SCAN16,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(predu, &levu, &SCAN16),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 16, 16, neutral);
                            let levv = bases.luma16x16.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 16, 16, predv),
                                0.0,
                                &SCAN16,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(predv, &levv, &SCAN16),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = SKIP_TX16_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN16,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN16,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        other => unreachable!("unsupported lossy leaf {:?}", other),
                    };
                    u_has[row * sb_cols + col] = u_present as i32;
                    v_has[row * sb_cols + col] = v_present as i32;
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode a 4:2:0 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height/2`. Luma is four 32x32 TUs per superblock; each chroma plane
    /// is one 32x32 transform per superblock. `width`/`height` must be even.
    pub fn encode_yuv420<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        // Lossy 4:2:x is reconstruction-sequential (intra prediction reads the
        // running reconstruction), so it codes serially regardless of `threads`.
        let _ = threads;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        planar_image.validate_420()?;
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph / 2);
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width / 2, height / 2, pcw, pch);
        let vp = pad_plane(&to_plane(cr), width / 2, height / 2, pcw, pch);

        let layout = Layout::I420;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let (tus, mode_idx) = encode_luma_sb(
                    &mut recy,
                    &yp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    &bases.luma,
                    crate::av2::quant::qstep(self.base_q_idx as u32) as i32,
                    &crate::av2::tables::SCAN,
                    neutral,
                );
                let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                    &tus,
                    sb_y,
                    sb_x,
                    &mut above,
                    &mut left,
                    qc,
                    (pw / 4) as i64,
                    (ph / 4) as i64,
                );
                encode_luma_block_split(
                    &mut enc,
                    &tus,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    mode_idx,
                    true,
                    12276,
                );

                let (cy, cx) = (sb_y / 2, sb_x / 2);
                let predu = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                let levu = bases
                    .chroma420
                    .project(&get_residual(&up, pcw, cy, cx, 32, predu), 0.0);
                put_block(
                    &mut recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    &bases.chroma420.reconstruct(predu, &levu),
                );
                let predv = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                let levv = bases
                    .chroma420
                    .project(&get_residual(&vp, pcw, cy, cx, 32, predv), 0.0);
                put_block(
                    &mut recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    &bases.chroma420.reconstruct(predv, &levv),
                );
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode a 4:2:2 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height` (half width, full height). Luma is four 32×32 TUs per
    /// superblock; each chroma plane is one 32-wide × 64-tall (TX_32X64) transform per
    /// superblock. `width` must be even. Chroma coefficient coding is identical to 4:4:4
    /// (avm codes TX_32X64 with the 32×32 scan and TX_64X64 entropy context); only the
    /// basis is the rectangular `chroma422` set.
    pub fn encode_yuv422<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        // Lossy 4:2:x is reconstruction-sequential (intra prediction reads the
        // running reconstruction), so it codes serially regardless of `threads`.
        let _ = threads;
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        planar_image.validate_422()?;
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph); // chroma: half width, full height
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width / 2, height, pcw, pch);
        let vp = pad_plane(&to_plane(cr), width / 2, height, pcw, pch);

        let layout = Layout::I422;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let (tus, mode_idx) = encode_luma_sb(
                    &mut recy,
                    &yp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    &bases.luma,
                    crate::av2::quant::qstep(self.base_q_idx as u32) as i32,
                    &crate::av2::tables::SCAN,
                    neutral,
                );
                let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                    &tus,
                    sb_y,
                    sb_x,
                    &mut above,
                    &mut left,
                    qc,
                    (pw / 4) as i64,
                    (ph / 4) as i64,
                );
                encode_luma_block_split(
                    &mut enc,
                    &tus,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    mode_idx,
                    true,
                    12276,
                );

                // Chroma block: 32 wide (sb_x/2) × 64 tall (sb_y), one TX_32X64 per plane.
                let (cy, cx) = (sb_y, sb_x / 2);
                let predu = dc_pred_rect(&recu, pcw, cy, cx, 32, 64, neutral);
                let levu = bases
                    .chroma422
                    .project(&get_residual_rect(&up, pcw, cy, cx, 32, 64, predu), 0.0);
                put_block_rect(
                    &mut recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &crate::av2::itx422::reconstruct_422(
                        predu,
                        &levu,
                        crate::av2::quant::qstep(self.base_q_idx as u32) as i32,
                        &crate::av2::tables::SCAN,
                    ),
                );
                let predv = dc_pred_rect(&recv, pcw, cy, cx, 32, 64, neutral);
                let levv = bases
                    .chroma422
                    .project(&get_residual_rect(&vp, pcw, cy, cx, 32, 64, predv), 0.0);
                put_block_rect(
                    &mut recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &crate::av2::itx422::reconstruct_422(
                        predv,
                        &levv,
                        crate::av2::quant::qstep(self.base_q_idx as u32) as i32,
                        &crate::av2::tables::SCAN,
                    ),
                );
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode a 4:0:0 (monochrome / luma-only) still. `y` is `width × height`.
    /// Four 32x32 luma TUs per superblock; no chroma is coded or signalled
    /// (`has_chroma = false` ⇒ no chroma intra mode, profile 0, layout uvlc 1).
    pub fn encode_yuv400<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_400()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);

        let layout = Layout::Monochrome;
        let config = self.config(layout);

        if config.lossless {
            return Ok(
                self.encode_yuv400_lossless(&yp, pw, ph, width, height, &config, color, threads)
            );
        }

        let mut recy = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let (tus, mode_idx) = encode_luma_sb(
                    &mut recy,
                    &yp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    &bases.luma,
                    crate::av2::quant::qstep(self.base_q_idx as u32) as i32,
                    &crate::av2::tables::SCAN,
                    neutral,
                );
                let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                    &tus,
                    sb_y,
                    sb_x,
                    &mut above,
                    &mut left,
                    qc,
                    (pw / 4) as i64,
                    (ph / 4) as i64,
                );
                encode_luma_block_split(
                    &mut enc,
                    &tus,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    mode_idx,
                    false,
                    12276,
                );
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Lossless (base_q=0) monochrome encode: each 64x64 superblock is coded as 256
    /// 4x4 transform units (forced TX_4X4), DC-predicted per TU and carried by the 4x4
    /// WHT. `yp` is the SB-padded source plane. The pixel reconstruction is bit-exact;
    /// the 4x4 coefficient CDFs/contexts are still being validated against the decoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv400_lossless(
        &self,
        yp: &[f32],
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        config: &Config,
        color: &ColorEncoding,
        threads: usize,
    ) -> Av2Frame {
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx); // base_q=0 -> q-context 0
        let neutral = self.dc_neutral();
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        // mi grid is 8px-aligned (avm dec_set_mb_mi); the recursive forced-split coder
        // handles every boundary geometry, so we always code the real (8-aligned) grid.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;

        // Phase A: per-SB TU generation is independent in lossless (recon == source),
        // so generate the clipped SB TU grids in parallel across `threads`.
        let nsb = sb_rows * sb_cols;
        let mut sbtus: Vec<Vec<Vec<Coeff>>> = (0..nsb).map(|_| Vec::new()).collect();
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                let (row, col) = (idx / sb_cols, idx % sb_cols);
                let (rr, rc) = ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16));
                *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
                        }
                    });
                }
            });
        }

        let mut above_pctx = vec![0u8; code_mc + 16];

        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let rr = (code_mr - row * 16).min(16);
                let rc = (code_mc - col * 16).min(16);
                // SB grid of in-frame 4x4 TUs (precomputed in Phase A).
                let tus = &sbtus[row * sb_cols + col];
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    code_mr,
                    code_mc,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(cdf, val);
                        }
                        partition::Op::Leaf {
                            mi_row,
                            mi_col,
                            bw_mi,
                            bh_mi,
                            part_cdf,
                        } => {
                            let lr = mi_row - row * 16;
                            let lc = mi_col - col * 16;
                            let lrows = bh_mi.min(rr - lr);
                            let lcols = bw_mi.min(rc - lc);
                            let mut ltus = Vec::with_capacity(lrows * lcols);
                            for i in 0..lrows {
                                for j in 0..lcols {
                                    ltus.push(tus[(lr + i) * rc + (lc + j)].clone());
                                }
                            }
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (skip_ctx, dc_sign_ctxs) =
                                sb_tu4_contexts(&ltus, ly, lx, &mut above, &mut left, lrows, lcols);
                            let skip_cdfs: Vec<u32> = skip_ctx
                                .iter()
                                .map(|&c| TXB_SKIP_TX4_Q0[c] as u32)
                                .collect();
                            encode_lossless_luma_sb(
                                &mut enc,
                                &ltus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                0,
                                false,
                                part_cdf,
                            );
                        }
                    }
                }
            }
        }
        self.finish(enc, config, pw, ph, width, height, color)
    }

    /// 4:4:4 lossless (q=0): luma + full-resolution U/V, all TX_4X4 WHT. Per superblock
    /// the block codes intra modes (incl. use_dpcm_y/uv = 0 and DC uv mode), then 256
    /// luma TUs, 256 U TUs, 256 V TUs — matching avm's shared-tree plane order.
    fn encode_yuv444_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);
        let config = self.config(Layout::I444);
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        // mi grid is 8px-aligned; recursion handles every boundary -> always exact.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        // luma ctx grids (0x40 = neutral DC-sign packing); chroma grids store cul (init 0).
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        let mut ua = vec![0u8; pw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        // Phase A: per-SB TU generation (DC-pred + WHT + levels). Independent across SBs
        // (lossless reconstruction == source), so this is data-parallel.
        type PackedCoeff = Vec<Coeff>;
        let mut sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> = (0..nsb)
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let gen_tile =
            |idx: usize, slot: &mut (Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)| {
                let (sb_y, sb_x) = ((idx / sb_cols) * 64, (idx % sb_cols) * 64);
                let (rr, rc) = rem(idx / sb_cols, idx % sb_cols);
                *slot = (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&vp, pw, sb_y, sb_x, neutral, rr, rc),
                );
            };
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                gen_tile(idx, slot);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (yp, up, vp) = (&yp, &up, &vp);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let (sb_y, sb_x) = (row * 64, col * 64);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = (
                                lossless_sb_tus(yp, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(up, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(vp, pw, sb_y, sb_x, neutral, rr, rc),
                            );
                        }
                    });
                }
            });
        }
        // Phase B: serial context derivation (cross-SB grids) + entropy coding.
        // Partition context arrays (av2 update_partition_context): `above` persists
        // down columns frame-wide; `left` is len-16 and zeroed per SB row.
        let mut above_pctx = vec![0u8; code_mc + 16];
        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                let (ytus, utus, vtus) = &sbtus[row * sb_cols + col];
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    code_mr,
                    code_mc,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(cdf, val);
                        }
                        partition::Op::Leaf {
                            mi_row,
                            mi_col,
                            bw_mi,
                            bh_mi,
                            part_cdf,
                        } => {
                            let lr = mi_row - row * 16;
                            let lc = mi_col - col * 16;
                            let lrows = bh_mi.min(rr - lr);
                            let lcols = bw_mi.min(rc - lc);
                            let slice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * lcols);
                                for i in 0..lrows {
                                    for j in 0..lcols {
                                        v.push(g[(lr + i) * rc + (lc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (lytus, lutus, lvtus) = (slice(ytus), slice(utus), slice(vtus));
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&lytus, ly, lx, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly, lx, &mut ua, &mut ul, false, false, lrows, lcols,
                            );
                            // avm's eob_u_flag is the LAST U TU of the block, used by every V TU.
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly, lx, &mut va, &mut vl, true, u_last_nz, lrows, lcols,
                            );
                            // modes (incl. uv) + luma coeffs, then U, then V (shared-tree order)
                            encode_lossless_luma_sb(
                                &mut enc,
                                &lytus,
                                &yskip_cdfs,
                                &ydcs,
                                0,
                                true,
                                part_cdf,
                            );
                            for (i, tu) in lutus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    TXB_SKIP_TX4_Q0[uskip[i]] as u32,
                                    false,
                                );
                            }
                            for (i, tu) in lvtus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    V_TXB_SKIP_TX4_Q0[vskip[i]] as u32,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:4:4 AV2. Converts RGB→YCbCr internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_444<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let bd = img.bit_depth;
        let maxv = (1i32 << bd.bits()) - 1;
        let off_q = (1i32 << (bd.bits() - 1)) << Q;
        let mx_i = maxv;
        let n = img.planes[0].len();
        let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        for (((((yv, cbv), crv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(cb.iter_mut())
            .zip(cr.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *cbv = ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
            *crv = ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
        }
        self.encode_yuv444(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
            threads,
        )
    }

    /// Encode an RGB image to 4:2:0 AV2. Converts RGB→YCbCr and downsamples
    /// chroma with a 2×2 box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_420<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        if self.base_q_idx == 0 {
            return Err(EncodeError::InvalidQuality);
        }
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = (1i32 << (bd - 1)) << Q;
        let mx_i = maxv;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut y = vec![0i32; w * h];
        let mut fcb_q = vec![0i32; w * h];
        let mut fcr_q = vec![0i32; w * h];
        for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(fcb_q.iter_mut())
            .zip(fcr_q.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
            *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
        }
        const HALF_AVG: i32 = 1 << (Q + 1); // rounding bias for >> (Q+2)
        let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);
        for row in 0..ch {
            for c in 0..cw {
                let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
                let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));
                let avg_q =
                    |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];
                cb[row * cw + c] = ((avg_q(&fcb_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
                cr[row * cw + c] = ((avg_q(&fcr_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            }
        }
        self.encode_yuv420(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
            threads,
        )
    }

    /// Encode an RGB image to 4:2:2 AV2. Converts RGB→YCbCr and downsamples
    /// chroma horizontally with a 2-tap box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_422<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        if self.base_q_idx == 0 {
            return Err(EncodeError::InvalidQuality);
        }
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = (1i32 << (bd - 1)) << Q;
        let mx_i = maxv;
        let cw = w.div_ceil(2);
        let mut y = vec![0i32; w * h];
        let mut fcb_q = vec![0i32; w * h];
        let mut fcr_q = vec![0i32; w * h];
        for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(fcb_q.iter_mut())
            .zip(fcr_q.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
            *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
        }
        const HALF_AVG: i32 = 1 << Q;
        let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);
        for row in 0..h {
            for c in 0..cw {
                let x0 = 2 * c;
                let x1 = (2 * c + 1).min(w - 1);
                let cb0 = fcb_q[row * w + x0];
                let cb1 = fcb_q[row * w + x1];
                let cr0 = fcr_q[row * w + x0];
                let cr1 = fcr_q[row * w + x1];
                cb[row * cw + c] = ((cb0 + cb1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
                cr[row * cw + c] = ((cr0 + cr1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            }
        }
        self.encode_yuv422(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
            threads,
        )
    }

    /// Encode a luma-only (4:0:0 / monochrome) image to AV2.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_400<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_400()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let plane = img.planes[0].to_vec();
        self.encode_yuv400(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [plane, vec![], vec![]],
            },
            color,
            threads,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        enc: RangeEncoder,
        config: &Config,
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        color: &ColorEncoding,
    ) -> Av2Frame {
        let tile = enc.finish();
        // AV2 derives its mode-info grid by rounding the frame to 4px
        // (ALIGN_POWER_OF_TWO(dim, MI_SIZE_LOG2)); superblocks are 64px (16 mi).
        // A square superblock at the right/bottom edge is force-split (no bits read)
        // only when *less than half* of it (<=32px, i.e. <=8 mi) is in-frame — see
        // is_partition_implied_at_boundary. When >32px is in-frame, every SB stays
        // PARTITION_NONE exactly as in the padded encode, so we can signal the real
        // size and let the decoder crop: the coded tile is byte-identical.
        // mi grid is 8px-aligned (avm dec_set_mb_mi); superblocks are 64px (16 mi).
        let mi_cols = ((width + 7) & !7) / 4;
        let mi_rows = ((height + 7) & !7) / 4;
        const MIB: usize = 16; // 64px superblock in 4px mode-info units
        // Lossless now codes every boundary geometry via the recursive forced-split
        // partition coder, so it always signals the real size (decoder crops to W x H).
        // Lossy doesn't clip its tx blocks at boundaries, so it pads unless SB-aligned.
        let aligned = mi_cols % MIB == 0 && mi_rows % MIB == 0;
        // Boundary-safe lossy 4:4:4 can also signal real W×H natively (the partial-edge
        // superblock decodes correctly with the edge-clamped entropy contexts).
        let lossy_native = !config.lossless
            && config.layout == Layout::I444
            && lossy_native_mi(width, height).is_some();
        let exact = config.lossless || aligned || lossy_native;
        // Signaled dimensions: real size when boundary-safe, else the padded size.
        let (sw, sh) = if exact { (width, height) } else { (pw, ph) };
        let mut frame = frame_header(config, sw as u32, sh as u32);
        frame.extend(&tile);
        let mut data = vec![];
        data.extend(obu(2, &[]));
        data.extend(obu(1, &sequence_header(config, sw as u32, sh as u32)));
        data.extend(obu(4, &frame));
        Av2Frame {
            data,
            width,
            height,
            // Coded size = the OBU-signaled size (decoder output). The muxer crops to
            // width/height via `clap` when this is larger (padded lossy).
            coded_width: sw,
            coded_height: sh,
            // Coded bit depth signaled in the sequence header (8/10/12). av2C/pixi in
            // the AVIF muxer must use this.
            bit_depth: self.bit_depth,
            color: *color,
            chroma_format: match config.layout {
                Layout::Monochrome => ChromaFormat::Monochrome,
                Layout::I420 => ChromaFormat::Yuv420,
                Layout::I422 => ChromaFormat::Yuv422,
                Layout::I444 => ChromaFormat::Yuv444,
            },
        }
    }

    /// Finish wrapping a color AV1 OBU stream in an AVIF container.
    pub fn wrap_avif(
        frame: &Av2Frame,
        icc_profile: Option<Vec<u8>>,
        exif: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        if let (Some(exif), Some(icc_profile)) = (exif, icc_profile.as_ref()) {
            return Ok(avif::to_avif_full(
                frame,
                &format,
                Some(icc_profile),
                Some(&exif),
            ));
        }
        if let Some(icc_profile) = icc_profile.as_ref() {
            return Ok(avif::to_avif_cicp_icc(frame, &format, icc_profile.to_vec()));
        }
        Ok(avif::to_avif(frame, &format))
    }

    /// Wrap a color frame together with a monochrome alpha auxiliary item into an
    /// AVIF (alpha = an `encode_yuv400` result, typically of the alpha plane). The
    /// alpha item is linked via `auxl` and tagged with the standard alpha `auxC` URN.
    pub fn wrap_avif_alpha(
        frame: &Av2Frame,
        alpha: &Av2Frame,
        icc_profile: Option<Vec<u8>>,
        exif: Option<Vec<u8>>,
    ) -> Result<Vec<u8>, EncodeError> {
        let format = Av2Format {
            bit_depth: frame.bit_depth,
            monochrome: frame.chroma_format == ChromaFormat::Monochrome,
            chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                || frame.chroma_format == ChromaFormat::Yuv420,
            chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
        };
        let color = match icc_profile {
            Some(icc) => Av2Color::Both {
                cicp: frame.color,
                icc,
            },
            None => Av2Color::Cicp(frame.color),
        };
        Ok(avif::to_avif_color_alpha(
            frame,
            alpha,
            &format,
            &color,
            exif.as_deref(),
        ))
    }
}
