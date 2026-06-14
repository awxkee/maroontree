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

use super::*;
use crate::Speed;

/// Build the prediction block for luma candidate `m` (0=DC, 1=SMOOTH, 4=PAETH)
/// at TX index `i` (raster within the 64x64 SB) and pixel origin `(y0,x0)`.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_luma(
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
    let tr_ok = matches!(i, 0..=2) && have_above && right_available && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
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
    } else if m == 2 {
        intrapred::smooth_v(32, &ab, &lf)
    } else if m == 3 {
        intrapred::smooth_h(32, &ab, &lf)
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
/// RD lambda for the per-SB luma tx-partition choice (pixel-SSE vs coded bits).
/// Scales ~qstep^2 like a standard mode-decision lambda; `c` is the multiplier in
/// lambda = c*qstep^2 (see [`crate::av2::Tuning::part_lambda_c`]).
pub(crate) fn part_lambda(qstep: i32, c: f64) -> f64 {
    c * (qstep as f64) * (qstep as f64)
}

fn project_luma_rdoq(
    luma: &Basis,
    resid: &[f32],
    scan: &[u16],
    qc: usize,
    cost: &mut f64,
    lambda: f64,
) -> Vec<f32> {
    if lambda > 0.0 {
        let (mut l, prm) = luma.project_scan_with_prm(resid, scan);
        *cost += coder::rdoq_luma(&prm, &mut l, qc, scan, 1024, lambda);
        l
    } else {
        let l = luma.project(resid, 0.0);
        *cost += l
            .iter()
            .filter(|&&v| v != 0.0)
            .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
            .sum::<f64>();
        l
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_luma_sb(
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
    qc: usize,
    rdoq_lambda: f64,
    speed: Speed,
) -> ([Vec<Coeff>; 4], usize) {
    const POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 64 * 64];
    // Fast reduces the intra candidate set; non-Full tiers rank candidates with a
    // cheap coeff cost (RDOQ disabled) and re-RDOQ the winner only.
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Encode one mode into `recy`, returning its TU coeffs and RD cost.
    let encode_mode = |recy: &mut [f32], m: usize, lambda: f64| -> ([Vec<Coeff>; 4], f64) {
        let mut resid = vec![0f32; 1024];
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for (i, &(ty, tx)) in POS.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x + tx);
            let pblk = predict_luma(recy, pw, width, height, i, y0, x0, m, neutral);
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = if lambda > 0.0 {
                // Trellis RDOQ: pick coefficient levels by real rate-distortion
                // (rate = true coded bits), then RD-trim the EOB.
                let (mut l, prm) = luma.project_with_prm(&resid);
                cost += crate::av2::coder::rdoq_luma(&prm, &mut l, qc, scan, 1024, lambda);
                l
            } else {
                let l = luma.project(&resid, 0.0);
                cost += l
                    .iter()
                    .filter(|&&v| v != 0.0)
                    .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                    .sum::<f64>();
                l
            };
            let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[i] = levels_to_coeffs(&lev);
        }
        // Mode-signaling cost (once per 64x64 block). DC is cheapest to signal;
        // SMOOTH/PAETH cost a few extra bits, so only win when they earn them.
        if m != 0 {
            cost += 6.0;
        }
        (tus, cost)
    };
    for &m in cands {
        let (tus, cost) = encode_mode(recy, m, search_lambda);
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
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        // Winner already coded at the final RDOQ setting (or RDOQ is off):
        // restore the saved winning reconstruction.
        for ry in 0..64 {
            let src = ry * 64;
            let dst = (sb_y + ry) * pw + sb_x;
            recy[dst..dst + 64].copy_from_slice(&best_region[src..src + 64]);
        }
    } else {
        // Winner-only RDOQ: re-encode the chosen mode with real RDOQ. `recy` ends
        // holding this reconstruction and the coeffs come from the same levels, so
        // bitstream and recon stay consistent by construction.
        let (tus, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode)
}

/// Intra prediction for one TX_32X32 of a bottom-edge 64x32 luma leaf. `ti` is the
/// sub-TU index (0=left, 1=right) within the SB-wide leaf at (`sb_y`,`sb_x`).
/// Unlike `predict_luma`, availability uses the NATIVE mi grid (`mi_cols`,`mi_rows`)
/// and bottom-left is always off: the leaf is the bottom partition, so everything
/// below it is out of frame / not yet decoded.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_luma_leaf32(
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
    } else if m == 2 {
        intrapred::smooth_v(32, &ab, &lf)
    } else if m == 3 {
        intrapred::smooth_h(32, &ab, &lf)
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
pub(super) fn encode_luma_leaf32(
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
    qc: usize,
    rdoq_lambda: f64,
    speed: Speed,
) -> ([Vec<Coeff>; 2], usize) {
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 64 * 32];
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    let encode_mode = |recy: &mut [f32], m: usize, lambda: f64| -> ([Vec<Coeff>; 2], f64) {
        let mut resid = vec![0f32; 1024];
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for (ti, tu) in tus.iter_mut().enumerate() {
            let (y0, x0) = (sb_y, sb_x + ti * 32);
            let pblk = predict_luma_leaf32(recy, pw, mi_cols, mi_rows, sb_y, sb_x, ti, m, neutral);
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            let rb = reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            *tu = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        (tus, cost)
    };
    for &m in cands {
        let (tus, cost) = encode_mode(recy, m, search_lambda);
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
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for ry in 0..32 {
            let dst = (sb_y + ry) * pw + sb_x;
            recy[dst..dst + 64].copy_from_slice(&best_region[ry * 64..ry * 64 + 64]);
        }
    } else {
        let (tus, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode)
}

/// General intra prediction for one TX_32X32 sub-block of a partition leaf, using
/// the NATIVE mi grid for reference availability. `(ty,tx)` is the TU's pixel offset
/// within the SB; `i` is the equivalent 64x64-raster index that selects avm's
/// top-right (i∈{0,1,2}) / bottom-left (i==0) eligibility rules.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_luma_leaf_tu(
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
    let tr_ok = matches!(i, 0..=2) && have_above && right_available && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    let bl_ok = i == 0 && have_left && bottom_available && yd > 0;
    let bl_px = if bl_ok { yd.min(32).max(0) as usize } else { 0 };
    let (ab, lf, corner) =
        intrapred::build_refs(recy, pw, y0, x0, 32, have_above, have_left, tr_px, bl_px);
    if m == 1 {
        intrapred::smooth(32, &ab, &lf)
    } else if m == 2 {
        intrapred::smooth_v(32, &ab, &lf)
    } else if m == 3 {
        intrapred::smooth_h(32, &ab, &lf)
    } else {
        intrapred::paeth(32, &ab, &lf, corner)
    }
}

/// Project + trial-code a right-edge 32x64 luma leaf as two stacked TX_32X32
/// (top i=0, bottom i=2). Mirrors `encode_luma_leaf32` but vertical.
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_luma_leaf_v32x64(
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
    qc: usize,
    rdoq_lambda: f64,
    speed: Speed,
) -> ([Vec<Coeff>; 2], usize) {
    let tu_i = [(0usize, 0usize), (32usize, 2usize)]; // (ty, raster-i)
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = vec![0f32; 32 * 64];
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    let encode_mode = |recy: &mut [f32], m: usize, lambda: f64| -> ([Vec<Coeff>; 2], f64) {
        let mut resid = vec![0f32; 1024];
        let mut cost = 0f64;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for (k, &(ty, i)) in tu_i.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x);
            let pblk = predict_luma_leaf_tu(recy, pw, mc, mr, sb_y, sb_x, ty, 0, i, m, neutral);
            for r in 0..32 {
                let base = (y0 + r) * pw + x0;
                for c in 0..32 {
                    resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
                }
            }
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            let rb = crate::av2::itx422::reconstruct_luma(&pblk, &lev, qstep, scan);
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[k] = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        (tus, cost)
    };
    for &m in cands {
        let (tus, cost) = encode_mode(recy, m, search_lambda);
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
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for ry in 0..64 {
            let dst = (sb_y + ry) * pw + sb_x;
            recy[dst..dst + 32].copy_from_slice(&best_region[ry * 32..ry * 32 + 32]);
        }
    } else {
        let (tus, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode)
}

/// Project + trial-code a corner 32x32 luma leaf as a single TX_32X32 (i=0).
#[allow(clippy::too_many_arguments)]
pub(super) fn encode_luma_leaf_s32x32(
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
    qc: usize,
    rdoq_lambda: f64,
    speed: Speed,
) -> (Vec<Coeff>, usize) {
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Single TU: no intra-leaf feedback, so a mode is fully described by its
    // (recon, coeffs, cost) and the winner can simply be re-projected with RDOQ.
    let encode_mode = |recy: &[f32], m: usize, lambda: f64| -> ([f32; 1024], Vec<Coeff>, f64) {
        let pblk = predict_luma_leaf_tu(recy, pw, mc, mr, sb_y, sb_x, 0, 0, 0, m, neutral);
        let mut resid = vec![0f32; 1024];
        for r in 0..32 {
            let base = (sb_y + r) * pw + sb_x;
            for c in 0..32 {
                resid[r * 32 + c] = yp[base + c] - pblk[r * 32 + c];
            }
        }
        let mut cost = 0f64;
        let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
        if m != 0 {
            cost += 6.0;
        }
        let rb = reconstruct_luma(&pblk, &lev, qstep, scan);
        (rb, levels_to_coeffs(&lev), cost)
    };
    let mut best_cost = f64::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tu: Vec<Coeff> = Vec::new();
    let mut best_region = vec![0f32; 32 * 32];
    for &m in cands {
        let (rb, tu, cost) = encode_mode(recy, m, search_lambda);
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_tu = tu;
            best_region.copy_from_slice(&rb);
        }
    }
    if !speed.per_candidate_rdoq() && rdoq_lambda > 0.0 {
        // Winner-only RDOQ: re-project the chosen mode with real RDOQ.
        let (rb, tu, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tu = tu;
        best_region.copy_from_slice(&rb);
    }
    put_block(recy, pw, sb_y, sb_x, 32, &best_region);
    (best_tu, best_mode)
}
