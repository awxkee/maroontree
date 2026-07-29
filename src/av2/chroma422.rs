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
use crate::av2::coder::EobCdf;
use crate::util::dirty_log2f;

#[allow(clippy::too_many_arguments)]
pub(super) fn project_chroma_rdoq(
    basis: &Basis,
    resid: &[f32],
    scan: &[u16],
    qc: usize,
    area: usize,
    plane_offset: usize,
    lambda: f32,
) -> Vec<f32> {
    if lambda > 0.0 {
        let (mut l, prm) = basis.project_scan_with_prm(resid, scan);
        coder::rdoq_chroma(&prm, &mut l, qc, scan, area, plane_offset, lambda);
        l
    } else {
        basis.project_scan(resid, 0.0, scan)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn recon_422_chroma(
    pred: f32,
    lev: &[f32],
    qstep: i32,
    scan: &[u16],
    cw: usize,
    ch: usize,
    _basis: &Basis,
    bd: i32,
) -> Vec<f32> {
    itx422::reconstruct_chroma(pred, lev, qstep, scan, cw, ch, bd)
}

/// Reconstruction (`rec_*`, written) and source (`src_*`, read) chroma plane refs plus
/// their shared `stride`. Replaces the five plane/stride positional args of the chroma
/// TU coder.
pub(super) struct ChromaPlanes<'a> {
    pub(super) rec_u: &'a mut [f32],
    pub(super) rec_v: &'a mut [f32],
    pub(super) src_u: &'a [f32],
    pub(super) src_v: &'a [f32],
    pub(super) stride: usize,
    /// Tile-local coded chroma extent; `stride`/allocation may be SB-padded.
    pub(super) coded_width: usize,
    pub(super) coded_height: usize,
}

/// Read-only planes and strides used while selecting an MHCCP chroma leaf.
#[derive(Clone, Copy)]
pub(super) struct ChromaLeafPlanes<'a> {
    pub(super) reconstructed_luma: &'a [f32],
    pub(super) reconstructed_u: &'a [f32],
    pub(super) reconstructed_v: &'a [f32],
    pub(super) source_u: &'a [f32],
    pub(super) source_v: &'a [f32],
    pub(super) luma_stride: usize,
    pub(super) chroma_stride: usize,
}

/// Luma/chroma placement and availability for one chroma leaf.
#[derive(Clone, Copy)]
pub(super) struct ChromaLeafGeometry {
    pub(super) bounds: cfl::MhccpBounds,
    pub(super) luma_y: usize,
    pub(super) luma_x: usize,
    pub(super) chroma_y: usize,
    pub(super) chroma_x: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) subsample_x: bool,
    pub(super) subsample_y: bool,
    pub(super) have_top: bool,
    pub(super) have_left: bool,
}

/// Quantization and distortion parameters shared by chroma-leaf mode decisions.
#[derive(Clone, Copy)]
pub(super) struct ChromaLeafRd<'a> {
    pub(super) neutral: f32,
    pub(super) basis: &'a Basis,
    pub(super) scan: &'a [u16],
    pub(super) qstep: i32,
    pub(super) lambda: f32,
    pub(super) bit_depth: i32,
}

/// Planes used by the regular chroma intra-mode search.
#[derive(Clone, Copy)]
pub(super) struct ChromaModePlanes<'a> {
    pub(super) reconstructed_u: &'a [f32],
    pub(super) reconstructed_v: &'a [f32],
    pub(super) source_u: &'a [f32],
    pub(super) source_v: &'a [f32],
    pub(super) stride: usize,
}

/// Position and coded extent of a square chroma mode-search block.
#[derive(Clone, Copy)]
pub(super) struct ChromaModeBlock {
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) size: usize,
    pub(super) coded_width: usize,
    pub(super) coded_height: usize,
}

/// Search and quantization settings for regular chroma intra-mode selection.
#[derive(Clone, Copy)]
pub(super) struct ChromaModeSearch<'a> {
    pub(super) neutral: f32,
    pub(super) basis: &'a Basis,
    pub(super) scan: &'a [u16],
    pub(super) quant_context: usize,
    pub(super) qstep: i32,
    pub(super) sb_qstep: i32,
    pub(super) residual_scale: f32,
    pub(super) rdoq_lambda: f32,
    pub(super) lambda: f32,
    pub(super) reduced: bool,
    pub(super) angle_directional: bool,
    pub(super) bit_depth: i32,
}

pub(super) fn mhccp_decide_leaf(
    enc: &mut RangeEncoder,
    planes: &ChromaLeafPlanes,
    block: &ChromaLeafGeometry,
    rd: &ChromaLeafRd,
) -> Option<cfl::CflChoice> {
    // Snapshot the current SB's coded-mi mask (see `RangeEncoder::sb_coded`) so
    // MHCCP's top-right / bottom-left reference extents follow the true coding
    // order rather than a raster assumption.
    let coded_mask = enc.sb_coded;
    let &ChromaLeafPlanes {
        reconstructed_luma: recy,
        reconstructed_u: recu,
        reconstructed_v: recv,
        source_u: up,
        source_v: vp,
        luma_stride: pw,
        chroma_stride: pcw,
    } = planes;
    let &ChromaLeafGeometry {
        bounds,
        luma_y: ly,
        luma_x: lx,
        chroma_y: cy,
        chroma_x: cx,
        width: cw,
        height: ch,
        subsample_x: ssx,
        subsample_y: ssy,
        have_top,
        have_left,
    } = block;
    let &ChromaLeafRd {
        neutral,
        basis,
        scan,
        qstep,
        lambda,
        bit_depth: bd,
    } = rd;
    if !enc.mhccp
        || !cfl::is_mhccp_allowed(
            (cw << (ssx as usize)) / 4,
            (ch << (ssy as usize)) / 4,
            ssx,
            ssy,
        )
    {
        return None;
    }
    let mut suf = vec![0f32; cw * ch];
    let mut svf = vec![0f32; cw * ch];

    let up_rows = up[cy * pcw + cx..].chunks(pcw);
    let vp_rows = vp[cy * pcw + cx..].chunks(pcw);

    for ((suf_row, svf_row), (up_row, vp_row)) in suf
        .chunks_exact_mut(cw)
        .zip(svf.chunks_exact_mut(cw))
        .zip(up_rows.zip(vp_rows))
        .take(ch)
    {
        suf_row.copy_from_slice(&up_row[..cw]);
        svf_row.copy_from_slice(&vp_row[..cw]);
    }

    let dc_u = dc_pred_rect_bounded(
        recu,
        &BoundedIntraRect {
            stride: pcw,
            y: cy,
            x: cx,
            width: cw,
            height: ch,
            frame_width: bounds.chroma_width,
            frame_height: bounds.chroma_height,
        },
        neutral,
        bd,
    );
    let dc_v = dc_pred_rect_bounded(
        recv,
        &BoundedIntraRect {
            stride: pcw,
            y: cy,
            x: cx,
            width: cw,
            height: ch,
            frame_width: bounds.chroma_width,
            frame_height: bounds.chroma_height,
        },
        neutral,
        bd,
    );
    let choice = cfl::mhccp_eval_leaf(&cfl::MhccpLeafInput {
        reconstructed_luma: recy,
        luma_stride: pw,
        reconstructed_u: recu,
        reconstructed_v: recv,
        chroma_stride: pcw,
        bounds,
        luma_y: ly,
        luma_x: lx,
        chroma_y: cy,
        chroma_x: cx,
        width: cw,
        height: ch,
        subsample_x: ssx,
        subsample_y: ssy,
        have_top,
        have_left,
        source_u: &suf,
        source_v: &svf,
        dc_u,
        dc_v,
        rd: cfl::ChromaRdSpec {
            basis,
            qstep,
            lambda,
            bit_depth: bd,
        },
        scan,
        coded_mask: &coded_mask,
    });
    if let Some(ref ch_choice) = choice
        && let Some(ref mh) = ch_choice.mhccp
    {
        enc.cfl_use = true;
        enc.mhccp_use = true;
        enc.mhccp_dir = mh.mh_dir;
        enc.mhccp_size_group = mh.size_group;
        enc.uv_mode = 0;
    }
    choice
}

pub(super) fn chroma_mode_decide_leaf(
    enc: &mut RangeEncoder,
    planes: &ChromaModePlanes,
    block: &ChromaModeBlock,
    search: &ChromaModeSearch,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let &ChromaModePlanes {
        reconstructed_u: recu,
        reconstructed_v: recv,
        source_u: up,
        source_v: vp,
        stride: pcw,
    } = planes;
    let &ChromaModeBlock {
        y: cy,
        x: cx,
        size: bs,
        coded_width: cw_px,
        coded_height: ch_px,
    } = block;
    let &ChromaModeSearch {
        neutral,
        basis,
        scan,
        quant_context: qc,
        qstep,
        sb_qstep,
        residual_scale: resid_scale,
        rdoq_lambda,
        lambda,
        reduced,
        angle_directional: angle_dir,
        bit_depth: bd,
    } = search;
    // Candidate set. DC + SMOOTH family + PAETH is the base. V/H (nominal
    // directional) are searched in every speed tier. The diagonal directionals
    // (D45..D203) are searched only when `angle_dir` (Medium speed and above).
    // Fast (`reduced`) keeps the cheapest non-directional subset plus V/H.
    let cand_modes: &[usize] = match (reduced, angle_dir) {
        (true, _) => &[0, 1, 4, 5, 6],
        (false, false) => &[0, 1, 2, 3, 4, 5, 6],
        (false, true) => &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    };
    let dcu = dc_pred_rect(recu, pcw, cy, cx, bs, bs, neutral, bd);
    let dcv = dc_pred_rect(recv, pcw, cy, cx, bs, bs, neutral, bd);
    let cscale = basis.qstep as f32 / qstep as f32;
    let mut best_mode = 0usize;
    let mut best_cost = f32::INFINITY;
    let mut best_pred: Option<(Vec<f32>, Vec<f32>)> = None;
    let predict_uv = |m: usize| -> (Vec<f32>, Vec<f32>) {
        if m == 0 {
            (vec![dcu; bs * bs], vec![dcv; bs * bs])
        } else {
            (
                predict_chroma_sq(recu, pcw, cy, cx, bs, m, neutral, cw_px, ch_px),
                predict_chroma_sq(recv, pcw, cy, cx, bs, m, neutral, cw_px, ch_px),
            )
        }
    };
    // SATD prune: each chroma mode is one bs×bs prediction from neighbors
    // (independent) — rank by SATD over U+V, full-encode only the top-K.
    let keep_uv = if reduced {
        2
    } else if angle_dir {
        4
    } else {
        3
    };
    let cand_modes: Vec<usize> = if cand_modes.len() > keep_uv {
        let mut r: Vec<(u64, usize)> = cand_modes
            .iter()
            .map(|&m| {
                let (pu, pv) = predict_uv(m);
                (
                    crate::av2::metrics::satd_f32(&up[cy * pcw + cx..], pcw, &pu, bs, bs, bs)
                        + crate::av2::metrics::satd_f32(&vp[cy * pcw + cx..], pcw, &pv, bs, bs, bs),
                    m,
                )
            })
            .collect();
        r.sort_by_key(|&(s, _)| s);
        r.truncate(keep_uv);
        r.into_iter().map(|(_, m)| m).collect()
    } else {
        cand_modes.to_vec()
    };
    for &m in &cand_modes {
        let (pu, pv) = predict_uv(m);
        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5) as i32).collect();
        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5) as i32).collect();
        let mut ru = vec![0f32; bs * bs];
        let mut rv = vec![0f32; bs * bs];
        let dst_rows = ru.chunks_exact_mut(bs).zip(rv.chunks_exact_mut(bs));
        let src_rows = rect_rows(up, pcw, cy, cx, bs, bs).zip(rect_rows(vp, pcw, cy, cx, bs, bs));
        let pred_rows = pu.chunks_exact(bs).zip(pv.chunks_exact(bs));
        for (((ru_row, rv_row), (up_row, vp_row)), (pu_row, pv_row)) in
            dst_rows.zip(src_rows).zip(pred_rows)
        {
            for ((dst, &src), &pred) in ru_row.iter_mut().zip(up_row).zip(pu_row) {
                *dst = src - pred;
            }
            for ((dst, &src), &pred) in rv_row.iter_mut().zip(vp_row).zip(pv_row) {
                *dst = src - pred;
            }
        }
        let lu = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&ru, cscale * resid_scale),
            scan,
            qc,
            bs * bs,
            0,
            rdoq_lambda,
        );
        let lv = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&rv, cscale * resid_scale),
            scan,
            qc,
            bs * bs,
            4,
            rdoq_lambda,
        );
        let recu_b = itx422::reconstruct_chroma_cfl(&pu_i, &lu, sb_qstep, scan, bs, bs, bd);
        let recv_b = itx422::reconstruct_chroma_cfl(&pv_i, &lv, sb_qstep, scan, bs, bs, bd);
        let sse = pixel_sse_rounded_block(up, pcw, cy, cx, &recu_b, bs, bs, bs)
            + pixel_sse_rounded_block(vp, pcw, cy, cx, &recv_b, bs, bs, bs);
        let rate = coeff_abs_rate_f32(&lu) + coeff_abs_rate_f32(&lv);
        let mode_bits = if m == 0 { 0.0 } else { 2.0 };
        let cost = sse as f32 + lambda * (rate as f32 + mode_bits);
        if cost < best_cost {
            best_cost = cost;
            best_mode = m;
            best_pred = if m == 0 { None } else { Some((pu, pv)) };
        }
    }
    enc.uv_mode = best_mode;
    best_pred
}

#[allow(clippy::too_many_arguments)]
fn predict_chroma_sq(
    rec: &[f32],
    pcw: usize,
    cy: usize,
    cx: usize,
    bs: usize,
    m: usize,
    neutral: f32,
    cw_px: usize,
    ch_px: usize,
) -> Vec<f32> {
    let have_above = cy > 0;
    let have_left = cx > 0;
    // Chroma mi grid (4 chroma px per mi). SB-aligned plane bounds.
    let mi_col_end = (((cw_px + 31) & !31) >> 2) as i64;
    let mi_row_end = (((ch_px + 31) & !31) >> 2) as i64;
    let mi_col = (cx >> 2) as i64;
    let mi_row = (cy >> 2) as i64;
    let bs_mi = (bs >> 2) as i64;
    // px to the right / below this block within the SB-aligned plane.
    let xr = ((mi_col_end - mi_col - bs_mi) << 2) + bs as i64;
    let right_available = (mi_col + bs_mi) < mi_col_end;
    let _ = mi_row_end;
    let _ = mi_row;
    let tr_px = if have_above && right_available && xr > 0 {
        xr.min(bs as i64).max(0) as usize
    } else {
        0
    };
    let bl_px = 0usize;
    let (ab, lf, corner) = intrapred::build_refs(
        rec,
        &intrapred::IntraRefSpec {
            stride: pcw,
            y: cy,
            x: cx,
            block_size: bs,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: bs,
            available_left: bs,
        },
    );
    match m {
        1 => intrapred::smooth(bs, &ab, &lf),
        2 => intrapred::smooth_v(bs, &ab, &lf),
        3 => intrapred::smooth_h(bs, &ab, &lf),
        4 => intrapred::paeth(bs, &ab, &lf, corner),
        // Directional (nominal angle; UV angle_delta is 0 when the co-located luma
        // mode is non-directional, which is the case for these leaves). The mode
        // order after PAETH follows AVM's `default_mode_list_uv`:
        // 5=V, 6=H, 7=D45, 8=D135, 9=D67, 10=D113, 11=D157, 12=D203.
        _ => {
            use crate::av2::directional::Dir;
            let dir = match m {
                5 => Dir::V,
                6 => Dir::H,
                7 => Dir::D45,
                8 => Dir::D135,
                9 => Dir::D67,
                10 => Dir::D113,
                11 => Dir::D157,
                _ => Dir::D203,
            };
            directional::directional(
                dir,
                &ab,
                &lf,
                corner,
                &directional::DirectionalSpec {
                    angle_delta: 0,
                    block_size: bs,
                    is_luma: false,
                    have_top: have_above,
                    have_left,
                    edge_filter: false,
                    max_width: bs as i32 + tr_px as i32,
                    max_height: bs as i32 + bl_px as i32,
                },
            )
        }
    }
}
/// TU of a given leaf shape, so it can be built once per match arm instead of spelled
/// out as eight positional args at every call.
pub(super) struct ChromaTxSpec<'a> {
    pub(super) cw: usize,
    pub(super) ch: usize,
    pub(super) basis: &'a Basis,
    pub(super) scan: &'a [u16],
    pub(super) eob_cdf: EobCdf,
    pub(super) eob_hi: u16,
    pub(super) area: usize,
    pub(super) u_skip_row: &'a [u16; 10],
}

/// Above/left coefficient-presence flags for the U and V planes at this TU.
#[derive(Clone, Copy)]
pub(super) struct ChromaNeighbors {
    pub(super) ua: i32,
    pub(super) ul: i32,
    pub(super) va: i32,
    pub(super) vl: i32,
}

/// Per-call position and prediction state for one 4:2:2 chroma transform unit.
#[derive(Clone, Copy)]
pub(super) struct ChromaTuInput<'a> {
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) bit_depth: i32,
    pub(super) cfl: Option<&'a cfl::CflChoice>,
    pub(super) mode_predictors: Option<(&'a [f32], &'a [f32])>,
}

pub(super) fn code_422_chroma_tu(
    enc: &mut RangeEncoder,
    planes: ChromaPlanes,
    spec: &ChromaTxSpec,
    quant: QuantCtx,
    nb: ChromaNeighbors,
    input: &ChromaTuInput,
) -> (bool, bool) {
    let &ChromaTuInput {
        y: cy,
        x: cx,
        bit_depth: bd,
        cfl,
        mode_predictors: uvmode_pred,
    } = input;
    let ChromaPlanes {
        rec_u: recu,
        rec_v: recv,
        src_u: up,
        src_v: vp,
        stride: pcw,
        coded_width,
        coded_height,
    } = planes;
    let &ChromaTxSpec {
        cw,
        ch,
        basis,
        scan,
        eob_cdf,
        eob_hi,
        area,
        u_skip_row,
    } = spec;
    let QuantCtx {
        qc,
        neutral,
        qstep,
        rdoq_lambda,
    } = quant;
    // MHCCP (if any) is decided BEFORE the luma mode is emitted and handed in via
    // `cfl` (its pred_u/pred_v carry the MHCCP predictor). See `mhccp_decide_leaf`.
    // Variance Boost: when the SB quantizer differs from the basis (frame) qstep, scale the
    // residual so the projection (which quantizes at `basis.qstep`) effectively quantizes at
    // `qstep`, matching the reconstruction that dequantizes at `qstep`. Identity when equal.
    let cscale = basis.qstep as f32 / qstep as f32;
    let ChromaNeighbors { ua, ul, va, vl } = nb;
    let (levu, levv) = if let Some(cflc) = cfl {
        // CfL: residual against the per-pixel prediction; reconstruct with that base.
        let n = cw * ch;
        let mut ru = vec![0f32; n];
        let mut rv = vec![0f32; n];
        let dst_rows = ru.chunks_exact_mut(cw).zip(rv.chunks_exact_mut(cw));
        let src_rows = rect_rows(up, pcw, cy, cx, cw, ch).zip(rect_rows(vp, pcw, cy, cx, cw, ch));
        let pred_rows = cflc
            .pred_u
            .chunks_exact(cw)
            .zip(cflc.pred_v.chunks_exact(cw));
        for (((ru_row, rv_row), (up_row, vp_row)), (pu_row, pv_row)) in
            dst_rows.zip(src_rows).zip(pred_rows)
        {
            for ((dst, &src), &pred) in ru_row.iter_mut().zip(up_row).zip(pu_row) {
                *dst = src - pred as f32;
            }
            for ((dst, &src), &pred) in rv_row.iter_mut().zip(vp_row).zip(pv_row) {
                *dst = src - pred as f32;
            }
        }
        let levu = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&ru, cscale),
            scan,
            qc,
            cw * ch,
            0,
            rdoq_lambda,
        );
        let levv = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&rv, cscale),
            scan,
            qc,
            cw * ch,
            4,
            rdoq_lambda,
        );
        put_block_rect(
            recu,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&cflc.pred_u, &levu, qstep, scan, cw, ch, bd),
        );
        put_block_rect(
            recv,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&cflc.pred_v, &levv, qstep, scan, cw, ch, bd),
        );
        (levu, levv)
    } else if let Some((pu, pv)) = uvmode_pred {
        // Chroma intra mode (SMOOTH family / PAETH): per-pixel predictor decided
        // before the luma emit (see chroma_mode_decide_leaf); reconstruct with that
        // base, exactly like the CfL per-pixel path.
        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5) as i32).collect();
        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5) as i32).collect();
        let mut ru = vec![0f32; cw * ch];
        let mut rv = vec![0f32; cw * ch];
        let dst_rows = ru.chunks_exact_mut(cw).zip(rv.chunks_exact_mut(cw));
        let src_rows = rect_rows(up, pcw, cy, cx, cw, ch).zip(rect_rows(vp, pcw, cy, cx, cw, ch));
        let pred_rows = pu.chunks_exact(cw).zip(pv.chunks_exact(cw));
        for (((ru_row, rv_row), (up_row, vp_row)), (pu_row, pv_row)) in
            dst_rows.zip(src_rows).zip(pred_rows)
        {
            for ((dst, &src), &pred) in ru_row.iter_mut().zip(up_row).zip(pu_row) {
                *dst = src - pred;
            }
            for ((dst, &src), &pred) in rv_row.iter_mut().zip(vp_row).zip(pv_row) {
                *dst = src - pred;
            }
        }
        let levu = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&ru, cscale),
            scan,
            qc,
            cw * ch,
            0,
            rdoq_lambda,
        );
        let levv = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&rv, cscale),
            scan,
            qc,
            cw * ch,
            4,
            rdoq_lambda,
        );
        put_block_rect(
            recu,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&pu_i, &levu, qstep, scan, cw, ch, bd),
        );
        put_block_rect(
            recv,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&pv_i, &levv, qstep, scan, cw, ch, bd),
        );
        (levu, levv)
    } else {
        let predu = dc_pred_rect_bounded(
            recu,
            &BoundedIntraRect {
                stride: pcw,
                y: cy,
                x: cx,
                width: cw,
                height: ch,
                frame_width: coded_width,
                frame_height: coded_height,
            },
            neutral,
            bd,
        );
        let levu = project_chroma_rdoq(
            basis,
            &aq::scale_resid(&get_residual_rect(up, pcw, cy, cx, cw, ch, predu), cscale),
            scan,
            qc,
            cw * ch,
            0,
            rdoq_lambda,
        );
        put_block_rect(
            recu,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &recon_422_chroma(predu, &levu, qstep, scan, cw, ch, basis, bd),
        );
        let predv = dc_pred_rect_bounded(
            recv,
            &BoundedIntraRect {
                stride: pcw,
                y: cy,
                x: cx,
                width: cw,
                height: ch,
                frame_width: coded_width,
                frame_height: coded_height,
            },
            neutral,
            bd,
        );
        let levv = project_chroma_rdoq(
            basis,
            &crate::av2::aq::scale_resid(
                &get_residual_rect(vp, pcw, cy, cx, cw, ch, predv),
                cscale,
            ),
            scan,
            qc,
            cw * ch,
            4,
            rdoq_lambda,
        );
        put_block_rect(
            recv,
            pcw,
            cy,
            cx,
            cw,
            ch,
            &recon_422_chroma(predv, &levv, qstep, scan, cw, ch, basis, bd),
        );
        (levu, levv)
    };
    let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
    let cbwl = dirty_log2f(cw.min(32) as f32) as i32;
    let u_skip = u_skip_row[(6 + ua + ul) as usize] as u32;
    encode_chroma_block_rect_w(enc, &uc, u_skip, true, scan, eob_cdf, eob_hi, area, cbwl);
    let up_ = uc.iter().any(|&(_, l)| l != 0);
    let v_skip = (6 * (up_ as i32) + va + vl) as u32;
    encode_chroma_block_rect_w(enc, &vc, v_skip, false, scan, eob_cdf, eob_hi, area, cbwl);
    (up_, vc.iter().any(|&(_, l)| l != 0))
}

/// As [`predict_chroma_mode`] but with explicit chroma-plane dimensions so the
/// top-right / bottom-left reference availability can be resolved the way avmdec
/// does. `cw_px`/`ch_px` are the chroma plane's coded width/height in pixels.
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_chroma_mode_dims(
    rec: &[f32],
    pcw: usize,
    cy: usize,
    cx: usize,
    bs: usize,
    m: usize,
    neutral: f32,
    cw_px: usize,
    ch_px: usize,
) -> Vec<f32> {
    let have_above = cy > 0;
    let have_left = cx > 0;
    // Top-right / bottom-left availability for the chroma 32x32 block, mirroring
    // the luma TX-index-0 calc in `predict_luma` (the chroma block fills one 32x32
    // chroma SB region, analogous to a single TX_32X32 at the SB origin). avmdec
    // resolves these against SB-aligned plane bounds; getting them wrong replaces
    // the top-right/bottom-left anchor with an extended edge sample and diverges.
    // The chroma plane's mi grid is the luma grid >> 1 (4:2:0). One chroma "mi"
    // unit here is 4 chroma px; the chroma SB is 32 px = 8 mi units.
    // Match decoder tile mi extent: ALIGN(dim, 8) >> 2, not chroma-SB-aligned.
    let mi_col_end = (((cw_px + 7) & !7) >> 2) as i64;
    let mi_row_end = (((ch_px + 7) & !7) >> 2) as i64;
    let mi_row = (cy >> 2) as i64;
    let mi_col = (cx >> 2) as i64;
    // px to the right / below this 32-px block within the SB-aligned plane.
    let xr = ((mi_col_end - mi_col - 8) << 2) + 32 - (cx as i64 - ((cx / 32) * 32) as i64);
    let yd = ((mi_row_end - mi_row - 8) << 2) + 32 - (cy as i64 - ((cy / 32) * 32) as i64);
    let right_available = (mi_col + 8) < mi_col_end;
    let bottom_available = (yd > 0) && ((mi_row + 8) < mi_row_end);
    let tr_px = if have_above && right_available && xr > 0 {
        xr.min(bs as i64).max(0) as usize
    } else {
        0
    };
    // Bottom-left for a full-SB chroma block resolves to the SB below-left, which
    // is later in raster order and therefore not yet reconstructed — so it is never
    // available here and the reference is extended from left[bs-1].
    let _ = (bottom_available, yd);
    let bl_px = 0usize;
    let (ab, lf, corner) = intrapred::build_refs(
        rec,
        &intrapred::IntraRefSpec {
            stride: pcw,
            y: cy,
            x: cx,
            block_size: bs,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: bs,
            available_left: bs,
        },
    );
    match m {
        1 => intrapred::smooth(bs, &ab, &lf),
        2 => intrapred::smooth_v(bs, &ab, &lf),
        3 => intrapred::smooth_h(bs, &ab, &lf),
        _ => intrapred::paeth(bs, &ab, &lf, corner),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn predict_chroma444_whole64(
    rec: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    m: usize,
    neutral: f32,
    cw_px: usize,
    ch_px: usize,
) -> Vec<f32> {
    let have_above = sb_y > 0;
    let have_left = sb_x > 0;
    // Luma-grid mi units (4 px each). The block is one 64x64 SB = 16 mi.
    // Match the decoder's tile mi extent: ALIGN(dim, 8) >> 2, not SB-aligned.
    let mi_col_end = (((cw_px + 7) & !7) >> 2) as i64;
    let mi_row_end = (((ch_px + 7) & !7) >> 2) as i64;
    let mi_col = (sb_x >> 2) as i64;
    let mi_row = (sb_y >> 2) as i64;
    // px available to the right of / below this 64-px block within the SB-aligned
    // plane, mirroring `predict_luma`'s xr/yd for a block at TX offset 0.
    let xr = ((mi_col_end - mi_col - 16) << 2) + 64;
    let yd = ((mi_row_end - mi_row - 16) << 2) + 64;
    let right_available = (mi_col + 16) < mi_col_end;
    // The bottom-left SB is later in raster order (not yet reconstructed), so the
    // whole-SB block never has a real bottom-left; the reference extends left[bs-1].
    // AVM `has_top_right` (av2/common/reconintra.c): for chroma planes, a
    // transform wider than 32 px is never allowed top-right reference samples
    // ("Do not allow more than 64 top reference samples"). The whole-SB chroma
    // block is a single TX_64X64, so top-right is unavailable regardless of the
    // neighbor reconstruction — the reference extends the top edge instead.
    let tr_ok = have_above && right_available && xr > 0;
    let tr_px = 0usize;
    let _ = tr_ok;
    let _ = yd;
    let bl_px = 0usize;
    let (ab, lf, corner) = intrapred::build_refs(
        rec,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: sb_y,
            x: sb_x,
            block_size: 64,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: 64,
            available_left: 64,
        },
    );
    match m {
        1 => intrapred::smooth(64, &ab, &lf),
        2 => intrapred::smooth_v(64, &ab, &lf),
        3 => intrapred::smooth_h(64, &ab, &lf),
        4 => intrapred::paeth(64, &ab, &lf, corner),
        _ => {
            use crate::av2::directional::Dir;
            let dir = match m {
                5 => Dir::V,
                6 => Dir::H,
                7 => Dir::D45,
                8 => Dir::D135,
                9 => Dir::D67,
                10 => Dir::D113,
                11 => Dir::D157,
                _ => Dir::D203,
            };
            directional::directional(
                dir,
                &ab,
                &lf,
                corner,
                &directional::DirectionalSpec {
                    angle_delta: 0,
                    block_size: 64,
                    is_luma: false,
                    have_top: have_above,
                    have_left,
                    edge_filter: false,
                    max_width: 64 + tr_px as i32,
                    max_height: 64 + bl_px as i32,
                },
            )
        }
    }
}
