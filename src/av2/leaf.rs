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

#[allow(clippy::too_many_arguments)]
fn fill_scaled_residual(
    dst: &mut [f32],
    src: &[f32],
    src_stride: usize,
    src_y: usize,
    src_x: usize,
    pred: &[f32],
    width: usize,
    height: usize,
    scale: f32,
) {
    let dst_rows = dst.chunks_exact_mut(width);
    let src_rows = rect_rows(src, src_stride, src_y, src_x, width, height);
    let pred_rows = pred.chunks_exact(width);
    for ((dst_row, src_row), pred_row) in dst_rows.zip(src_rows).zip(pred_rows) {
        for ((dst, &src), &pred) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
            *dst = (src - pred) * scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_intra_pred(
    m: usize,
    adelta: i32,
    ab: &[i32],
    lf: &[i32],
    corner: i32,
    have_top: bool,
    have_left: bool,
    max_w: i32,
    max_h: i32,
) -> Vec<f32> {
    use directional::Dir::*;
    let dir = match m {
        1 => return intrapred::smooth(32, ab, lf),
        2 => return intrapred::smooth_v(32, ab, lf),
        3 => return intrapred::smooth_h(32, ab, lf),
        4 => return intrapred::paeth(32, ab, lf, corner),
        5 => V,
        6 => H,
        7 => D45,
        8 => D135,
        9 => D113,
        10 => D157,
        11 => D203,
        _ => D67,
    };
    directional::directional(
        dir,
        ab,
        lf,
        corner,
        &directional::DirectionalSpec {
            angle_delta: adelta,
            block_size: 32,
            is_luma: true,
            have_top,
            have_left,
            edge_filter: true,
            max_width: max_w,
            max_height: max_h,
        },
    )
}

/// Source luma plane and its row stride.
#[derive(Clone, Copy)]
pub(super) struct LumaSource<'a> {
    pub(super) plane: &'a [f32],
    pub(super) stride: usize,
}

/// Frame-bounded location of one superblock-sized luma coding region.
#[derive(Clone, Copy)]
pub(super) struct LumaFrameBlock {
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
    pub(super) y: usize,
    pub(super) x: usize,
}

/// MI-grid-bounded location shared by the split luma-leaf encoders.
#[derive(Clone, Copy)]
pub(super) struct LumaGridBlock {
    pub(super) mi_cols: i64,
    pub(super) mi_rows: i64,
    pub(super) y: usize,
    pub(super) x: usize,
}

/// Transform, quantization and speed state shared by luma mode searches.
#[derive(Clone, Copy)]
pub(super) struct LumaQuantSpec<'a> {
    pub(super) basis: &'a Basis,
    pub(super) qstep: i32,
    pub(super) scan: &'a [u16],
    pub(super) neutral: f32,
    pub(super) quant_context: usize,
    pub(super) rdoq_lambda: f32,
    pub(super) speed: Speed,
    pub(super) bit_depth: i32,
}

/// Search options specific to a full 64x64 luma superblock.
#[derive(Clone, Copy)]
pub(super) struct LumaSbSearch {
    pub(super) residual_scale: f32,
    pub(super) allow_directional: bool,
}

/// Geometry and mode state for a frame-bounded 32x32 luma prediction.
#[derive(Clone, Copy)]
pub(super) struct LumaPredictSpec {
    pub(super) stride: usize,
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
    pub(super) tx_index: usize,
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) mode: usize,
    pub(super) angle_delta: i32,
    pub(super) neutral: f32,
}

/// Geometry and mode state for one TX_32X32 within a split luma leaf.
#[derive(Clone, Copy)]
pub(super) struct LumaLeafPredictSpec {
    pub(super) stride: usize,
    pub(super) mi_cols: i64,
    pub(super) mi_rows: i64,
    pub(super) block_y: usize,
    pub(super) block_x: usize,
    pub(super) tx_y: usize,
    pub(super) tx_x: usize,
    pub(super) tx_index: usize,
    pub(super) mode: usize,
    pub(super) neutral: f32,
    pub(super) split_leaf: bool,
}

/// Build the prediction block for luma candidate `m` (0=DC, 1=SMOOTH, 4=PAETH)
/// at TX index `i` (raster within the 64x64 SB) and pixel origin `(y0,x0)`.
pub(super) fn predict_luma(recy: &[f32], spec: &LumaPredictSpec) -> Vec<f32> {
    let &LumaPredictSpec {
        stride: pw,
        frame_width: width,
        frame_height: height,
        tx_index: i,
        y: y0,
        x: x0,
        mode: m,
        angle_delta: adelta,
        neutral,
    } = spec;
    if m == 0 {
        return vec![
            crate::av2::helpers::dc_pred_bounded(
                recy, pw, y0, x0, 32, neutral, width, height
            );
            1024
        ];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    // Match the decoder's mi grid: mi_cols/mi_rows = ALIGN(dim, 8) >> MI_SIZE_LOG2
    // (av2 aligns the frame to 8 luma px for CDEF, not to the 64-px superblock),
    // so reference-sample availability at partial edge SBs matches tile.mi_col_end.
    let mi_col_end = (((width + 7) & !7) >> 2) as i64;
    let mi_row_end = (((height + 7) & !7) >> 2) as i64;
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
    let avail_above = (((mi_col_end << 2) as usize).saturating_sub(x0))
        .min(32)
        .max(1);
    let avail_left = (((mi_row_end << 2) as usize).saturating_sub(y0))
        .min(32)
        .max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred(
        m,
        adelta,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32 + bl_px as i32,
    )
}

pub(crate) fn part_lambda(qstep: i32, c: f32) -> f32 {
    c * (qstep as f32) * (qstep as f32)
}

#[inline]
fn luma_mode_lambda(rdoq_lambda: f32, qstep: i32) -> f32 {
    rdoq_lambda.max(proj::DEFAULT_RDOQ_LAMBDA) * (qstep as f32) * (qstep as f32)
}

fn project_luma_rdoq(
    luma: &Basis,
    resid: &[f32],
    scan: &[u16],
    qc: usize,
    cost: &mut f32,
    lambda: f32,
) -> Vec<f32> {
    if lambda > 0.0 {
        let (mut l, prm) = luma.project_scan_with_prm(resid, scan);
        *cost += coder::rdoq_luma(&prm, &mut l, qc, scan, 1024, lambda);
        l
    } else {
        let l = luma.project(resid, 0.0);
        *cost += coeff_rate_f32(&l);
        l
    }
}

pub(super) fn encode_luma_sb(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaFrameBlock,
    quant: &LumaQuantSpec<'_>,
    search: &LumaSbSearch,
) -> ([Vec<Coeff>; 4], usize, i8, f32) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaFrameBlock {
        frame_width: width,
        frame_height: height,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        speed,
        bit_depth: bd,
    } = quant;
    let &LumaSbSearch {
        residual_scale: resid_scale,
        allow_directional: allow_dir,
    } = search;
    static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
    let mut best_cost = f32::INFINITY;
    let mut best_mode = 0usize;
    let mut best_delta = 0i32;
    let mut best_tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut best_region = [0f32; 64 * 64];
    // Fast reduces the intra candidate set; non-Full tiers rank candidates with a
    // cheap coeff cost (RDOQ disabled) and re-RDOQ the winner only.
    let base_modes: &[usize] = if speed.reduced_modes() {
        if allow_dir {
            &[0usize, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
        } else {
            &[0usize, 1, 2]
        }
    } else if allow_dir {
        &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    // Directional modes trial all angle deltas (Δ = -3..=3, ×3°) only on the
    // Slow tier; faster tiers use the nominal angle (Δ=0) to avoid 7× the
    // directional RD candidates. Non-directional modes are always Δ=0.
    let dir_deltas: &[i32] = if speed.try_angle_deltas() {
        &[0, -1, 1, -2, 2, -3, 3]
    } else {
        &[0]
    };

    let mut cands: Vec<(usize, i32)> = Vec::new();
    for &m in base_modes {
        if m >= 5 {
            for &d in dir_deltas {
                cands.push((m, d));
            }
        } else {
            cands.push((m, 0));
        }
    }
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Residual scratch reused across every (mode, angle) candidate of this SB,
    // instead of allocating a fresh 1024-element buffer per candidate.
    let resid_buf = std::cell::RefCell::new(vec![0f32; 1024]);
    // Encode one (mode, angle_delta) into `recy`, returning its TU coeffs, coded-bit
    // cost, and reconstruction SSE (for a proper rate-distortion mode decision).
    let encode_mode =
        |recy: &mut [f32], m: usize, adelta: i32, lambda: f32| -> ([Vec<Coeff>; 4], f32, f32) {
            let mut resid = resid_buf.borrow_mut();
            let mut cost = 0f32;
            let mut sse = 0f32;
            let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for (i, &(ty, tx)) in POS.iter().enumerate() {
                let (y0, x0) = (sb_y + ty, sb_x + tx);
                let pblk = predict_luma(
                    recy,
                    &LumaPredictSpec {
                        stride: pw,
                        frame_width: width,
                        frame_height: height,
                        tx_index: i,
                        y: y0,
                        x: x0,
                        mode: m,
                        angle_delta: adelta,
                        neutral,
                    },
                );
                fill_scaled_residual(&mut resid, yp, pw, y0, x0, &pblk, 32, 32, resid_scale);
                let lev = if lambda > 0.0 {
                    // Trellis RDOQ: pick coefficient levels by real rate-distortion
                    // (rate = true coded bits), then RD-trim the EOB.
                    let (mut l, prm) = luma.project_with_prm(&resid[..]);
                    cost += coder::rdoq_luma(&prm, &mut l, qc, scan, 1024, lambda);
                    l
                } else {
                    let l = luma.project(&resid[..], 0.0);
                    cost += coeff_rate_f32(&l) as f32;
                    l
                };
                let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
                // real reconstruction SSE vs source for this TU
                sse += rect_sse_f32(
                    &PlaneRect {
                        plane: yp,
                        stride: pw,
                        y: y0,
                        x: x0,
                    },
                    &PlaneRect {
                        plane: &rb,
                        stride: 32,
                        y: 0,
                        x: 0,
                    },
                    32,
                    32,
                );
                put_block(recy, pw, y0, x0, 32, &rb);
                tus[i] = levels_to_coeffs(&lev);
            }
            // Mode-signaling cost (once per 64x64 block). DC is cheapest; SMOOTH/PAETH
            // and directional cost a few extra bits (directional a bit more for the
            // set/idx symbols), so only win when they earn them.
            if m != 0 {
                cost += if m >= 5 { 9.0 } else { 6.0 };
            }
            (tus, cost, sse)
        };
    // Stage 1 — cheap SATD prune. Score each (mode, angle) by the SATD of its
    // prediction vs source (predict-only: no forward transform, RDOQ or
    // reconstruct), keep only the top-K, and run the expensive SSE/RD pass on
    // those. SATD tracks post-transform coding cost well, so the true winner is
    // essentially always in the top few. The prediction is written into `recy` so
    // subsequent 32x32 sub-blocks predict from the prior ones (as the real coder
    // does); TU[0] always predicts from the SB's finalized neighbors, and the
    // full Stage-2 pass overwrites `recy` correctly, so this scratch use is safe.
    let satd_of_mode = |recy: &mut [f32], m: usize, adelta: i32| -> u64 {
        let mut satd = 0u64;
        for (i, &(ty, tx)) in POS.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x + tx);
            let pblk = predict_luma(
                recy,
                &LumaPredictSpec {
                    stride: pw,
                    frame_width: width,
                    frame_height: height,
                    tx_index: i,
                    y: y0,
                    x: x0,
                    mode: m,
                    angle_delta: adelta,
                    neutral,
                },
            );
            satd += crate::av2::metrics::satd_f32(&yp[y0 * pw + x0..], pw, &pblk, 32, 32, 32);
            put_block(recy, pw, y0, x0, 32, &pblk);
        }
        satd
    };
    let keep = if speed.try_angle_deltas() { 8 } else { 4 };
    let cands: Vec<(usize, i32)> = if cands.len() > keep {
        let mut ranked: Vec<(u64, (usize, i32))> = cands
            .iter()
            .map(|&(m, d)| (satd_of_mode(recy, m, d), (m, d)))
            .collect();
        ranked.sort_by_key(|&(s, _)| s);
        ranked.truncate(keep);
        ranked.into_iter().map(|(_, md)| md).collect()
    } else {
        cands
    };
    // Mode-RD weight in pixel-SSE per bit: rdoq_lambda is level^2/bit, qstep^2 is
    // pixel^2/level^2, so their product converts coded bits to the SSE domain. This makes
    // the mode search a proper rate-distortion decision instead of cheapest-to-code.
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep);
    let mut best_bits = 0f32;
    for &(m, d) in &cands {
        let (tus, cost, sse) = encode_mode(recy, m, d, search_lambda);
        let j = sse + mode_lambda * cost;
        if j < best_cost {
            best_cost = j;
            best_bits = cost;
            best_mode = m;
            best_delta = d;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 64))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        // Winner already coded at the final RDOQ setting (or RDOQ is off):
        // restore the saved winning reconstruction.
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 64, 64).zip(best_region.as_chunks::<64>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        // Winner-only RDOQ: re-encode the chosen mode with real RDOQ.
        let (tus, cost, _) = encode_mode(recy, best_mode, best_delta, rdoq_lambda);
        best_tus = tus;
        best_bits = cost;
    }
    (best_tus, best_mode, best_delta as i8, best_bits)
}

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
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    let avail_above = ((mi_cols << 2) as usize).saturating_sub(x0).min(32).max(1);
    let avail_left = ((_mi_rows << 2) as usize).saturating_sub(y0).min(32).max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: 0,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred(
        m,
        0,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32,
    )
}

pub(super) fn encode_luma_leaf32(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
) -> ([Vec<Coeff>; 2], usize) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols,
        mi_rows,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        speed,
        bit_depth: bd,
    } = quant;
    let mut best_cost = f32::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = [0f32; 64 * 32];
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
    let encode_mode = |recy: &mut [f32], m: usize, lambda: f32| -> ([Vec<Coeff>; 2], f32, f32) {
        let mut resid = [0f32; 1024];
        let mut cost = 0f32;
        let mut sse = 0f32;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for (ti, tu) in tus.iter_mut().enumerate() {
            let (y0, x0) = (sb_y, sb_x + ti * 32);
            let pblk = predict_luma_leaf32(recy, pw, mi_cols, mi_rows, sb_y, sb_x, ti, m, neutral);
            fill_scaled_residual(
                &mut resid,
                yp,
                pw,
                y0,
                x0,
                &pblk,
                32,
                32,
                luma.qstep as f32 / qstep as f32,
            );
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
            sse += rect_sse_f32(
                &PlaneRect {
                    plane: yp,
                    stride: pw,
                    y: y0,
                    x: x0,
                },
                &PlaneRect {
                    plane: &rb,
                    stride: 32,
                    y: 0,
                    x: 0,
                },
                32,
                32,
            );
            put_block(recy, pw, y0, x0, 32, &rb);
            *tu = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        (tus, cost, sse)
    };
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep);
    for &m in cands {
        let (tus, rate, sse) = encode_mode(recy, m, search_lambda);
        let j = sse + mode_lambda * rate;
        if j < best_cost {
            best_cost = j;
            best_mode = m;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 32))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 64, 32).zip(best_region.as_chunks::<64>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        let (tus, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode)
}

pub(crate) fn predict_luma_leaf_tu(recy: &[f32], spec: &LumaLeafPredictSpec) -> Vec<f32> {
    let &LumaLeafPredictSpec {
        stride: pw,
        mi_cols: mc,
        mi_rows: mr,
        block_y: sb_y,
        block_x: sb_x,
        tx_y: ty,
        tx_x: tx,
        tx_index: i,
        mode: m,
        neutral,
        split_leaf,
    } = spec;
    let (y0, x0) = (sb_y + ty, sb_x + tx);
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    let sb_x0 = (x0 / 64) * 64;
    let sb_y0 = (y0 / 64) * 64;
    let is_right = (x0 - sb_x0) >= 32;
    let is_bottom = (y0 - sb_y0) >= 32;
    let mi_col = (sb_x >> 2) as i64;
    let mi_row = (sb_y >> 2) as i64;
    let (lx, ly) = (tx as i64, ty as i64);
    let (col_off, row_off) = (lx / 4, ly / 4);
    let xr = ((mc - mi_col - 16) << 2) + 32 - lx;
    let yd = ((mr - mi_row - 16) << 2) + 32 - ly;
    // top-right unavailable for the bottom-right leaf; bottom-left only for the top-left leaf.
    // avm has_top_right: top-row leaves (tr_mask_row<0) read above-right from the
    // coded SB row above, so TL and TR both qualify; only bottom-row leaves don't.
    #[allow(clippy::overly_complex_bool_expr)]
    let tr_avail = !(is_right && is_bottom) && !is_bottom && (mi_col + col_off + 8) < mc;
    #[allow(clippy::overly_complex_bool_expr)]
    let bl_avail = !is_right && !is_bottom && (mi_row + row_off + 8) < mr;
    // SMOOTH/SMOOTH_H/D45/D67 need above-right; SMOOTH/SMOOTH_V/D203 need bottom-left.
    let need_tr = matches!(m, 1 | 3 | 7) || m >= 12;
    let need_bl = matches!(m, 1 | 2 | 11);
    let tr_ok = need_tr && matches!(i, 0..=2) && have_above && tr_avail && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    let bl_ok = need_bl
        && i == 0
        && have_left
        && (bl_avail || (split_leaf && is_right && !is_bottom))
        && yd > 0;
    let bl_px = if bl_ok { yd.min(32).max(0) as usize } else { 0 };
    let avail_above = (((mc << 2) as usize).saturating_sub(x0)).min(32).max(1);
    let avail_left = (((mr << 2) as usize).saturating_sub(y0)).min(32).max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred(
        m,
        0,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32 + bl_px as i32,
    )
}

/// Project + trial-code a right-edge 32x64 luma leaf as two stacked TX_32X32
/// (top i=0, bottom i=2). Mirrors `encode_luma_leaf32` but vertical.
pub(super) fn encode_luma_leaf_v32x64(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
) -> ([Vec<Coeff>; 2], usize) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols: mc,
        mi_rows: mr,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        speed,
        bit_depth: bd,
    } = quant;
    let tu_i = [(0usize, 0usize), (32usize, 2usize)]; // (ty, raster-i)
    let mut best_cost = f32::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = [0f32; 32 * 64];
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
    let encode_mode = |recy: &mut [f32], m: usize, lambda: f32| -> ([Vec<Coeff>; 2], f32, f32) {
        let mut resid = [0f32; 1024];
        let mut cost = 0f32;
        let mut sse = 0f32;
        let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
        for (k, &(ty, i)) in tu_i.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x);
            let pblk = predict_luma_leaf_tu(
                recy,
                &LumaLeafPredictSpec {
                    stride: pw,
                    mi_cols: mc,
                    mi_rows: mr,
                    block_y: sb_y,
                    block_x: sb_x,
                    tx_y: ty,
                    tx_x: 0,
                    tx_index: i,
                    mode: m,
                    neutral,
                    split_leaf: false,
                },
            );
            fill_scaled_residual(
                &mut resid,
                yp,
                pw,
                y0,
                x0,
                &pblk,
                32,
                32,
                luma.qstep as f32 / qstep as f32,
            );
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
            sse += rect_sse_f32(
                &PlaneRect {
                    plane: yp,
                    stride: pw,
                    y: y0,
                    x: x0,
                },
                &PlaneRect {
                    plane: &rb,
                    stride: 32,
                    y: 0,
                    x: 0,
                },
                32,
                32,
            );
            put_block(recy, pw, y0, x0, 32, &rb);
            tus[k] = levels_to_coeffs(&lev);
        }
        if m != 0 {
            cost += 6.0;
        }
        (tus, cost, sse)
    };
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep);
    for &m in cands {
        let (tus, rate, sse) = encode_mode(recy, m, search_lambda);
        let j = sse + mode_lambda * rate;
        if j < best_cost {
            best_cost = j;
            best_mode = m;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 32, 64))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 32, 64).zip(best_region.as_chunks::<32>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        let (tus, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode)
}

/// Project + trial-code a corner 32x32 luma leaf as a single TX_32X32 (i=0).
pub(super) fn encode_luma_leaf_s32x32(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
) -> (Vec<Coeff>, usize) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols: mc,
        mi_rows: mr,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        speed,
        bit_depth: bd,
    } = quant;
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
    let encode_mode =
        |recy: &[f32], m: usize, lambda: f32| -> ([f32; 1024], Vec<Coeff>, f32, f32) {
            let pblk = predict_luma_leaf_tu(
                recy,
                &LumaLeafPredictSpec {
                    stride: pw,
                    mi_cols: mc,
                    mi_rows: mr,
                    block_y: sb_y,
                    block_x: sb_x,
                    tx_y: 0,
                    tx_x: 0,
                    tx_index: 0,
                    mode: m,
                    neutral,
                    split_leaf: true,
                },
            );
            let mut resid = [0f32; 1024];
            fill_scaled_residual(
                &mut resid,
                yp,
                pw,
                sb_y,
                sb_x,
                &pblk,
                32,
                32,
                luma.qstep as f32 / qstep as f32,
            );
            let mut cost = 0f32;
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            if m != 0 {
                cost += 6.0;
            }
            let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
            let sse = rect_sse_f32(
                &PlaneRect {
                    plane: yp,
                    stride: pw,
                    y: sb_y,
                    x: sb_x,
                },
                &PlaneRect {
                    plane: &rb,
                    stride: 32,
                    y: 0,
                    x: 0,
                },
                32,
                32,
            );
            (rb, levels_to_coeffs(&lev), cost, sse)
        };
    // SATD prune (same as encode_luma_sb): this is a single 32x32 TU with no
    // intra-leaf feedback, so every mode's prediction is independent — SATD-rank
    // them (predict-only) and full-encode just the top-K.
    let keep = if speed.reduced_modes() { 2 } else { 3 };
    let cands: Vec<usize> = if cands.len() > keep {
        let mut ranked: Vec<(u64, usize)> = cands
            .iter()
            .map(|&m| {
                let pblk = predict_luma_leaf_tu(
                    recy,
                    &LumaLeafPredictSpec {
                        stride: pw,
                        mi_cols: mc,
                        mi_rows: mr,
                        block_y: sb_y,
                        block_x: sb_x,
                        tx_y: 0,
                        tx_x: 0,
                        tx_index: 0,
                        mode: m,
                        neutral,
                        split_leaf: true,
                    },
                );
                (
                    crate::av2::metrics::satd_f32(&yp[sb_y * pw + sb_x..], pw, &pblk, 32, 32, 32),
                    m,
                )
            })
            .collect();
        ranked.sort_by_key(|&(s, _)| s);
        ranked.truncate(keep);
        ranked.into_iter().map(|(_, m)| m).collect()
    } else {
        cands.to_vec()
    };
    let mut best_cost = f32::INFINITY;
    let mut best_mode = 0usize;
    let mut best_tu: Vec<Coeff> = Vec::new();
    let mut best_region = [0f32; 32 * 32];
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep);
    for &m in &cands {
        let (rb, tu, rate, sse) = encode_mode(recy, m, search_lambda);
        let j = sse + mode_lambda * rate;
        if j < best_cost {
            best_cost = j;
            best_mode = m;
            best_tu = tu;
            best_region.copy_from_slice(&rb);
        }
    }
    if !speed.per_candidate_rdoq() && rdoq_lambda > 0.0 {
        // Winner-only RDOQ: re-project the chosen mode with real RDOQ.
        let (rb, tu, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tu = tu;
        best_region.copy_from_slice(&rb);
    }
    put_block(recy, pw, sb_y, sb_x, 32, &best_region);
    (best_tu, best_mode)
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct LumaPartitionDecision {
    pub(super) split64: bool,
    pub(super) split32: [bool; 4],
    pub(super) split16: [bool; 16],
}

#[derive(Clone, Copy)]
pub(super) struct LumaPartitionSearch<'a> {
    pub(super) quant: LumaQuantSpec<'a>,
    pub(super) sb: LumaSbSearch,
    pub(super) basis16: &'a Basis,
    pub(super) basis8: &'a Basis,
    /// Subsampled leaf coders currently support 16x16 only for boundary residue,
    /// not as a regular interior chroma-reference leaf.
    pub(super) allow_16x16: bool,
    /// 8x8 luma leaves require chroma-reference sharing in subsampled formats.
    /// Keep them out of the search until that syntax is emitted by the leaf coder.
    pub(super) allow_8x8: bool,
}

pub(super) fn choose_luma_64x64_partition(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaFrameBlock,
    grid: &LumaGridBlock,
    search: &LumaPartitionSearch<'_>,
) -> LumaPartitionDecision {
    let quant = &search.quant;
    let pw = source.stride;
    let mut saved = [0f32; 64 * 64];
    for (dst, src) in saved
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .zip(rect_rows(recy, pw, block.y, block.x, 64, 64))
    {
        dst.copy_from_slice(src);
    }
    let restore = |recy: &mut [f32]| {
        for (dst, src) in
            rect_rows_mut(recy, pw, block.y, block.x, 64, 64).zip(saved.as_chunks::<64>().0.iter())
        {
            dst.copy_from_slice(src);
        }
    };

    let (_, _, _, whole_rate) = encode_luma_sb(recy, source, block, quant, &search.sb);
    let whole_sse = pixel_sse_rounded_block(
        source.plane,
        pw,
        block.y,
        block.x,
        &recy[block.y * pw + block.x..],
        pw,
        64,
        64,
    );
    restore(recy);

    let mut split_rate = 0.0f32;
    let mut split32 = [false; 4];
    let mut split16 = [false; 16];
    for (child, (dy, dx)) in [(0usize, 0usize), (0, 32), (32, 0), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut child_saved = [0f32; 32 * 32];
        for (dst, src) in child_saved
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(rect_rows(recy, pw, block.y + dy, block.x + dx, 32, 32))
        {
            dst.copy_from_slice(src);
        }
        let restore_child = |recy: &mut [f32]| {
            for (dst, src) in rect_rows_mut(recy, pw, block.y + dy, block.x + dx, 32, 32)
                .zip(child_saved.as_chunks::<32>().0.iter())
            {
                dst.copy_from_slice(src);
            }
        };
        let leaf = LumaGridBlock {
            mi_cols: grid.mi_cols,
            mi_rows: grid.mi_rows,
            y: block.y + dy,
            x: block.x + dx,
        };
        let (tu, mode) = encode_luma_leaf_s32x32(recy, source, &leaf, quant);
        let rate32 = coeff_tu_rate_proxy_f32(&tu);
        let rate32 = rate32 + if mode != 0 { 6.0 } else { 0.0 };
        // Cache the 32x32 leaf reconstruction so the not-split branch can restore it
        // instead of re-encoding the identical leaf (rate32 already is its cost).
        let mut leaf32_recon = [0f32; 32 * 32];
        for (dst, src) in leaf32_recon
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(rect_rows(recy, pw, block.y + dy, block.x + dx, 32, 32))
        {
            dst.copy_from_slice(src);
        }
        let sse32 = pixel_sse_rounded_block(
            source.plane,
            pw,
            block.y + dy,
            block.x + dx,
            &recy[(block.y + dy) * pw + block.x + dx..],
            pw,
            32,
            32,
        );
        restore_child(recy);

        let mut rate16 = 0.0;
        for (sy, sx) in [(0usize, 0usize), (0, 16), (16, 0), (16, 16)] {
            let (y, x) = (block.y + dy + sy, block.x + dx + sx);
            let mut saved16 = [0f32; 256];
            for (dst, src) in saved16
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, y, x, 16, 16))
            {
                dst.copy_from_slice(src);
            }
            let restore16 = |recy: &mut [f32]| {
                for (dst, src) in
                    rect_rows_mut(recy, pw, y, x, 16, 16).zip(saved16.as_chunks::<16>().0.iter())
                {
                    dst.copy_from_slice(src);
                }
            };
            let pred = dc_pred_rect(recy, pw, y, x, 16, 16, quant.neutral, quant.bit_depth);
            // Compensate for the basis calibration vs the per-SB qstep (as the 32x32 path
            // does). Omitting this made the 16x16 probe reconstruct residuals at the wrong
            // amplitude under dark AQ, systematically making smaller partitions look worse.
            let scale16 = search.basis16.qstep as f32 / quant.qstep as f32;
            let resid = aq::scale_resid(
                &get_residual_rect(source.plane, pw, y, x, 16, 16, pred),
                scale16,
            );
            let lev = search.basis16.project_scan(&resid, 0.0, &SCAN16);
            let pred_block = [pred; 256];
            let rec = itx422::reconstruct_luma16(
                &pred_block,
                &lev,
                quant.qstep,
                &SCAN16,
                quant.bit_depth,
            );
            put_block_rect(recy, pw, y, x, 16, 16, &rec);
            let candidate16_rate = coeff_rate_f32(&lev);
            let candidate16_sse =
                pixel_sse_rounded_block(source.plane, pw, y, x, &recy[y * pw + x..], pw, 16, 16);
            restore16(recy);

            let index16 = ((dy + sy) / 16) * 4 + (dx + sx) / 16;
            let mut rate8 = 0.0;
            if search.allow_8x8 {
                for (ey, ex) in [(0usize, 0usize), (0, 8), (8, 0), (8, 8)] {
                    let (y8, x8) = (y + ey, x + ex);
                    let p8 = dc_pred_rect(recy, pw, y8, x8, 8, 8, quant.neutral, quant.bit_depth);
                    let scale8 = search.basis8.qstep as f32 / quant.qstep as f32;
                    let r8 = aq::scale_resid(
                        &get_residual_rect(source.plane, pw, y8, x8, 8, 8, p8),
                        scale8,
                    );
                    let l8 = search.basis8.project_scan(&r8, 0.0, &SCAN8X8);
                    let rec8 = itx422::reconstruct_chroma(
                        p8,
                        &l8,
                        quant.qstep,
                        &SCAN8X8,
                        8,
                        8,
                        quant.bit_depth,
                    );
                    put_block_rect(recy, pw, y8, x8, 8, 8, &rec8);
                    rate8 += coeff_rate_f32(&l8);
                }
                let sse8 = pixel_sse_rounded_block(
                    source.plane,
                    pw,
                    y,
                    x,
                    &recy[y * pw + x..],
                    pw,
                    16,
                    16,
                );
                let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep);
                split16[index16] =
                    sse8 + lambda * (rate8 + 5.0) < candidate16_sse + lambda * candidate16_rate;
            }
            if split16[index16] {
                rate16 += rate8 + 5.0;
            } else {
                restore16(recy);
                put_block_rect(recy, pw, y, x, 16, 16, &rec);
                rate16 += candidate16_rate;
            }
        }
        let sse16 = pixel_sse_rounded_block(
            source.plane,
            pw,
            block.y + dy,
            block.x + dx,
            &recy[(block.y + dy) * pw + block.x + dx..],
            pw,
            32,
            32,
        );
        let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep);
        split32[child] =
            search.allow_16x16 && sse16 + lambda * (rate16 + 5.0) < sse32 + lambda * rate32;
        if split32[child] {
            split_rate += rate16 + 5.0;
        } else {
            // Not split: restore the cached 32x32 leaf recon (no re-encode).
            put_block(recy, pw, block.y + dy, block.x + dx, 32, &leaf32_recon);
            split_rate += rate32;
        }
    }
    let split_sse = pixel_sse_rounded_block(
        source.plane,
        pw,
        block.y,
        block.x,
        &recy[block.y * pw + block.x..],
        pw,
        64,
        64,
    );
    restore(recy);

    split_rate += 5.0;
    let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep);
    LumaPartitionDecision {
        split64: split_sse + lambda * split_rate < whole_sse + lambda * whole_rate,
        split32,
        split16,
    }
}
