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

#[allow(clippy::too_many_arguments)]
pub(super) fn project_chroma_rdoq(
    basis: &Basis,
    resid: &[f32],
    scan: &[u16],
    qc: usize,
    area: usize,
    plane_offset: usize,
    lambda: f64,
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
}

#[allow(clippy::too_many_arguments)]
pub(super) fn mhccp_decide_leaf(
    enc: &mut RangeEncoder,
    recy: &[f32],
    recu: &[f32],
    recv: &[f32],
    up: &[f32],
    vp: &[f32],
    pw: usize,
    pcw: usize,
    ly: usize,
    lx: usize,
    cy: usize,
    cx: usize,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
    have_top: bool,
    have_left: bool,
    neutral: f32,
    basis: &Basis,
    scan: &[u16],
    qstep: i32,
    lambda: f64,
    bd: i32,
) -> Option<cfl::CflChoice> {
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

    let dc_u = dc_pred_rect(recu, pcw, cy, cx, cw, ch, neutral, bd);
    let dc_v = dc_pred_rect(recv, pcw, cy, cx, cw, ch, neutral, bd);
    let choice = cfl::mhccp_eval_leaf(
        recy, pw, recu, recv, pcw, ly, lx, cy, cx, cw, ch, ssx, ssy, have_top, have_left, &suf,
        &svf, dc_u, dc_v, basis, qstep, lambda, scan, bd,
    );
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

#[allow(clippy::too_many_arguments)]
pub(super) fn chroma_mode_decide_leaf(
    enc: &mut RangeEncoder,
    recu: &[f32],
    recv: &[f32],
    up: &[f32],
    vp: &[f32],
    pcw: usize,
    cy: usize,
    cx: usize,
    bs: usize,
    neutral: f32,
    basis: &Basis,
    scan: &[u16],
    qc: usize,
    qstep: i32,
    sb_qstep: i32,
    resid_scale: f32,
    rdoq_lambda: f64,
    lambda: f64,
    cw_px: usize,
    ch_px: usize,
    reduced: bool,
    angle_dir: bool,
    bd: i32,
) -> Option<(Vec<f32>, Vec<f32>)> {
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
    let mut best_cost = f64::INFINITY;
    let mut best_pred: Option<(Vec<f32>, Vec<f32>)> = None;
    for &m in cand_modes {
        let (pu, pv): (Vec<f32>, Vec<f32>) = if m == 0 {
            (vec![dcu; bs * bs], vec![dcv; bs * bs])
        } else {
            (
                predict_chroma_sq(recu, pcw, cy, cx, bs, m, neutral, cw_px, ch_px),
                predict_chroma_sq(recv, pcw, cy, cx, bs, m, neutral, cw_px, ch_px),
            )
        };
        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5).floor() as i32).collect();
        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5).floor() as i32).collect();
        let mut ru = vec![0f32; bs * bs];
        let mut rv = vec![0f32; bs * bs];
        for r in 0..bs {
            let b = (cy + r) * pcw + cx;
            for c in 0..bs {
                ru[r * bs + c] = up[b + c] - pu[r * bs + c];
                rv[r * bs + c] = vp[b + c] - pv[r * bs + c];
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
        let mut sse = 0f64;
        for r in 0..bs {
            let b = (cy + r) * pcw + cx;
            for c in 0..bs {
                let du = up[b + c] - recu_b[r * bs + c];
                let dv = vp[b + c] - recv_b[r * bs + c];
                sse += (du * du + dv * dv) as f64;
            }
        }
        let rate: f64 = lu.iter().chain(lv.iter()).map(|&l| l.abs() as f64).sum();
        let mode_bits = if m == 0 { 0.0 } else { 2.0 };
        let cost = sse + lambda * (rate + mode_bits);
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
        rec, pcw, cy, cx, bs, have_above, have_left, tr_px, bl_px, neutral,
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
                0,
                bs,
                &ab,
                &lf,
                corner,
                false,
                have_above,
                have_left,
                false,
                bs as i32 + tr_px as i32,
                bs as i32 + bl_px as i32,
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

#[allow(clippy::too_many_arguments)]
pub(super) fn code_422_chroma_tu(
    enc: &mut RangeEncoder,
    planes: ChromaPlanes,
    cy: usize,
    cx: usize,
    spec: &ChromaTxSpec,
    quant: QuantCtx,
    nb: ChromaNeighbors,
    bd: i32,
    cfl: Option<&cfl::CflChoice>,
    uvmode_pred: Option<(&[f32], &[f32])>,
) -> (bool, bool) {
    let ChromaPlanes {
        rec_u: recu,
        rec_v: recv,
        src_u: up,
        src_v: vp,
        stride: pcw,
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
        for r in 0..ch {
            let b = (cy + r) * pcw + cx;
            let ru_d = &mut ru[r * cw..];
            let rv_d = &mut rv[r * cw..];
            let up_s = &up[b..b + cw];
            let vp_s = &vp[b..b + cw];
            let pred_us = &cflc.pred_u[r * cw..r * cw + cw];
            let pred_vs = &cflc.pred_v[r * cw..r * cw + cw];
            for (((((ru, rv), &up), &vp), &pred_us), &pred_vs) in ru_d[..cw]
                .iter_mut()
                .zip(rv_d[..cw].iter_mut())
                .zip(up_s.iter())
                .zip(vp_s.iter())
                .zip(pred_us.iter())
                .zip(pred_vs.iter())
            {
                *ru = up - pred_us as f32;
                *rv = vp - pred_vs as f32;
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
        let pu_i: Vec<i32> = pu.iter().map(|&p| (p + 0.5).floor() as i32).collect();
        let pv_i: Vec<i32> = pv.iter().map(|&p| (p + 0.5).floor() as i32).collect();
        let mut ru = vec![0f32; cw * ch];
        let mut rv = vec![0f32; cw * ch];
        for r in 0..ch {
            let b = (cy + r) * pcw + cx;
            for c in 0..cw {
                ru[r * cw + c] = up[b + c] - pu[r * cw + c];
                rv[r * cw + c] = vp[b + c] - pv[r * cw + c];
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
        let predu = dc_pred_rect(recu, pcw, cy, cx, cw, ch, neutral, bd);
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
        let predv = dc_pred_rect(recv, pcw, cy, cx, cw, ch, neutral, bd);
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
    let u_skip = u_skip_row[(6 + ua + ul) as usize] as u32;
    encode_chroma_block_rect(enc, &uc, u_skip, true, scan, eob_cdf, eob_hi, area);
    let up_ = uc.iter().any(|&(_, l)| l != 0);
    let v_skip = CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
    encode_chroma_block_rect(enc, &vc, v_skip, false, scan, eob_cdf, eob_hi, area);
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
    let mi_col_end = (((cw_px + 31) & !31) >> 2) as i64; // chroma SB-aligned, /4
    let mi_row_end = (((ch_px + 31) & !31) >> 2) as i64;
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
        rec, pcw, cy, cx, bs, have_above, have_left, tr_px, bl_px, neutral,
    );
    match m {
        1 => intrapred::smooth(bs, &ab, &lf),
        2 => intrapred::smooth_v(bs, &ab, &lf),
        3 => intrapred::smooth_h(bs, &ab, &lf),
        _ => intrapred::paeth(bs, &ab, &lf, corner),
    }
}

pub(super) fn predict_chroma444_whole64(
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
    let mi_col_end = (((cw_px + 63) & !63) >> 2) as i64;
    let mi_row_end = (((ch_px + 63) & !63) >> 2) as i64;
    let mi_col = (sb_x >> 2) as i64;
    let mi_row = (sb_y >> 2) as i64;
    // px available to the right of / below this 64-px block within the SB-aligned
    // plane, mirroring `predict_luma`'s xr/yd for a block at TX offset 0.
    let xr = ((mi_col_end - mi_col - 16) << 2) + 64;
    let yd = ((mi_row_end - mi_row - 16) << 2) + 64;
    let right_available = (mi_col + 16) < mi_col_end;
    // The bottom-left SB is later in raster order (not yet reconstructed), so the
    // whole-SB block never has a real bottom-left; the reference extends left[bs-1].
    let tr_ok = have_above && right_available && xr > 0;
    let tr_px = if tr_ok { xr.min(64).max(0) as usize } else { 0 };
    let _ = yd;
    let bl_px = 0usize;
    let (ab, lf, corner) = intrapred::build_refs(
        rec, pw, sb_y, sb_x, 64, have_above, have_left, tr_px, bl_px, neutral,
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
                0,
                64,
                &ab,
                &lf,
                corner,
                false,
                have_above,
                have_left,
                false,
                64 + tr_px as i32,
                64 + bl_px as i32,
            )
        }
    }
}
