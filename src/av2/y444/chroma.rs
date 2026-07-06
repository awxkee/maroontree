/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use super::super::*;

/// Plane set for one 4:4:4 chroma leaf. Mutable reconstruction planes are kept
/// together so the leaf coder receives one coherent image view.
pub(super) struct Chroma444Planes<'a> {
    pub(super) reconstructed_luma: &'a [f32],
    pub(super) reconstructed_u: &'a mut [f32],
    pub(super) reconstructed_v: &'a mut [f32],
    pub(super) source_u: &'a [f32],
    pub(super) source_v: &'a [f32],
    pub(super) stride: usize,
}

/// Placement, extent and neighbor availability of one 4:4:4 chroma leaf.
#[derive(Clone, Copy)]
pub(super) struct Chroma444Block {
    pub(super) bounds: cfl::MhccpBounds,
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) have_top: bool,
    pub(super) have_left: bool,
}

/// Transform shape and entropy tables used by one 4:4:4 chroma leaf.
#[derive(Clone, Copy)]
pub(super) struct Chroma444Tx<'a> {
    pub(super) basis: &'a Basis,
    pub(super) scan: &'a [u16],
    pub(super) eob_cdf: EobCdf,
    pub(super) eob_hi: u16,
    pub(super) area: usize,
    pub(super) u_skip_row: &'a [u16],
}

/// Quantization and distortion settings for one 4:4:4 chroma leaf.
#[derive(Clone, Copy)]
pub(super) struct Chroma444Quant {
    pub(super) neutral: f32,
    pub(super) qstep: i32,
    pub(super) lambda: f32,
    pub(super) bit_depth: i32,
}

/// One 4:4:4 chroma leaf (U then V), DC-predicted unless MHCCP wins. Mirrors the
/// per-arm inline path (project_scan + reconstruct_chroma) so the eligible
/// partition arms can share it. Returns (u_present, v_present).
/// Decide MHCCP for a 4:4:4 chroma leaf and set `enc.cfl_use / mhccp_use` BEFORE
/// the luma leaf is emitted. The luma emit (`encode_luma_leaf_*`) signals the
/// chroma mode via `encode_intra_modes`, which reads those flags; without this
/// pre-pass the mode is coded as non-MHCCP and the decoder never applies MHCCP
/// (chroma reconstructs from DC while the encoder coded the residual against the
/// MHCCP predictor). `code_444_chroma_leaf` re-derives the same (deterministic)
/// choice for the coefficient emit.
#[allow(clippy::too_many_arguments)]
pub(super) fn preset_444_mhccp(
    enc: &mut RangeEncoder,
    recy: &[f32],
    recu: &[f32],
    recv: &[f32],
    up: &[f32],
    vp: &[f32],
    pw: usize,
    block: &Chroma444Block,
    tx: &Chroma444Tx<'_>,
    quant: &Chroma444Quant,
    mhccp_on: bool,
    // Staged threading: `Some(cached)` reuses a previously-fitted choice (replay,
    // skipping the 3-direction filter fit); `None` performs the fit. Returns the
    // choice so the caller can log it during capture.
    replay_choice: Option<Option<cfl::CflChoice>>,
) -> Option<cfl::CflChoice> {
    let coded_mask = enc.sb_coded;
    let &Chroma444Block {
        bounds: mhccp_bounds,
        y: sb_y,
        x: sb_x,
        width: cw,
        height: ch,
        have_top,
        have_left,
    } = block;
    let &Chroma444Quant {
        neutral,
        qstep,
        lambda,
        bit_depth: bd,
    } = quant;
    if !(mhccp_on && cfl::is_mhccp_allowed(cw / 4, ch / 4, false, false)) {
        return None;
    }
    let choice = if let Some(cached) = replay_choice {
        cached
    } else {
        let dcu = dc_pred_rect(recu, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let dcv = dc_pred_rect(recv, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let mut suf = vec![0f32; cw * ch];
        let mut svf = vec![0f32; cw * ch];
        for (((suf_row, svf_row), up_row), vp_row) in suf
            .chunks_exact_mut(cw)
            .zip(svf.chunks_exact_mut(cw))
            .zip(rect_rows(up, pw, sb_y, sb_x, cw, ch))
            .zip(rect_rows(vp, pw, sb_y, sb_x, cw, ch))
        {
            suf_row.copy_from_slice(up_row);
            svf_row.copy_from_slice(vp_row);
        }
        cfl::mhccp_eval_leaf(&cfl::MhccpLeafInput {
            reconstructed_luma: recy,
            luma_stride: pw,
            reconstructed_u: recu,
            reconstructed_v: recv,
            chroma_stride: pw,
            bounds: mhccp_bounds,
            luma_y: sb_y,
            luma_x: sb_x,
            chroma_y: sb_y,
            chroma_x: sb_x,
            width: cw,
            height: ch,
            subsample_x: false,
            subsample_y: false,
            have_top,
            have_left,
            source_u: &suf,
            source_v: &svf,
            dc_u: dcu,
            dc_v: dcv,
            rd: cfl::ChromaRdSpec {
                basis: tx.basis,
                qstep,
                lambda,
                bit_depth: bd,
            },
            scan: tx.scan,
            coded_mask: &coded_mask,
        })
    };
    if let Some(mh) = choice.as_ref().and_then(|c| c.mhccp.as_ref()) {
        enc.cfl_use = true;
        enc.mhccp_use = true;
        enc.mhccp_dir = mh.mh_dir;
        enc.mhccp_size_group = mh.size_group;
        enc.uv_mode = 0;
    }
    // Cache the fit so the matching chroma-leaf encode reuses this exact predictor
    // (bit-identical inputs) instead of re-running the 3-direction filter fit.
    enc.mhccp_cache = Some((sb_y, sb_x, cw, ch, choice.clone()));
    choice
}

pub(super) fn code_444_chroma_leaf(
    enc: &mut RangeEncoder,
    planes: &mut Chroma444Planes<'_>,
    block: &Chroma444Block,
    tx: &Chroma444Tx<'_>,
    quant: &Chroma444Quant,
    neighbors: ChromaNeighbors,
    mhccp_on: bool,
) -> (bool, bool) {
    let coded_mask = enc.sb_coded;
    let Chroma444Planes {
        reconstructed_luma,
        reconstructed_u,
        reconstructed_v,
        source_u,
        source_v,
        stride,
    } = planes;
    let recy = *reconstructed_luma;
    let recu = &mut **reconstructed_u;
    let recv = &mut **reconstructed_v;
    let up = *source_u;
    let vp = *source_v;
    let pw = *stride;
    let &Chroma444Block {
        bounds: mhccp_bounds,
        y: sb_y,
        x: sb_x,
        width: cw,
        height: ch,
        have_top,
        have_left,
    } = block;
    let &Chroma444Tx {
        basis,
        scan,
        eob_cdf,
        eob_hi,
        area,
        u_skip_row,
    } = tx;
    let &Chroma444Quant {
        neutral,
        qstep,
        lambda,
        bit_depth: bd,
    } = quant;
    let ChromaNeighbors { ua, ul, va, vl } = neighbors;
    // Reuse the preset's cached MHCCP fit for this exact leaf (position-keyed,
    // one-shot) — the preset fed the fit identical inputs, so this is bit-exact and
    // skips the redundant 3-direction filter fit.
    let matches = matches!(
        &enc.mhccp_cache,
        Some((cy, cx, ccw, cch, _)) if (*cy, *cx, *ccw, *cch) == (sb_y, sb_x, cw, ch)
    );
    // Take (one-shot) only on a position match; a mismatch leaves the cache intact
    // for the leaf it belongs to.
    let cached = if matches {
        enc.mhccp_cache.take().map(|(_, _, _, _, choice)| choice)
    } else {
        None
    };
    // MHCCP vs DC incumbent (4:4:4 => ssx = ssy = false).
    let mhccp_choice = if let Some(choice) = cached {
        choice
    } else if mhccp_on && cfl::is_mhccp_allowed(cw / 4, ch / 4, false, false) {
        let dcu = dc_pred_rect(recu, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let dcv = dc_pred_rect(recv, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let mut suf = vec![0f32; cw * ch];
        let mut svf = vec![0f32; cw * ch];
        for (((suf_row, svf_row), up_row), vp_row) in suf
            .chunks_exact_mut(cw)
            .zip(svf.chunks_exact_mut(cw))
            .zip(rect_rows(up, pw, sb_y, sb_x, cw, ch))
            .zip(rect_rows(vp, pw, sb_y, sb_x, cw, ch))
        {
            suf_row.copy_from_slice(up_row);
            svf_row.copy_from_slice(vp_row);
        }
        cfl::mhccp_eval_leaf(&cfl::MhccpLeafInput {
            reconstructed_luma: recy,
            luma_stride: pw,
            reconstructed_u: recu,
            reconstructed_v: recv,
            chroma_stride: pw,
            bounds: mhccp_bounds,
            luma_y: sb_y,
            luma_x: sb_x,
            chroma_y: sb_y,
            chroma_x: sb_x,
            width: cw,
            height: ch,
            subsample_x: false,
            subsample_y: false,
            have_top,
            have_left,
            source_u: &suf,
            source_v: &svf,
            dc_u: dcu,
            dc_v: dcv,
            rd: cfl::ChromaRdSpec {
                basis,
                qstep,
                lambda,
                bit_depth: bd,
            },
            scan,
            coded_mask: &coded_mask,
        })
    } else {
        None
    };
    let mh = mhccp_choice.as_ref().and_then(|c| c.mhccp.as_ref());
    if let Some(mh) = mh {
        enc.cfl_use = true;
        enc.mhccp_use = true;
        enc.mhccp_dir = mh.mh_dir;
        enc.mhccp_size_group = mh.size_group;
        enc.uv_mode = 0;
    }
    let win = mh.is_some();
    // U plane.
    let levu = if win {
        let ch_choice = mhccp_choice.as_ref().unwrap();
        let mut ru = vec![0f32; cw * ch];
        for ((dst_row, src_row), pred_row) in ru
            .chunks_exact_mut(cw)
            .zip(rect_rows(up, pw, sb_y, sb_x, cw, ch))
            .zip(ch_choice.pred_u.chunks_exact(cw))
        {
            for ((dst, &src), &pred) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
                *dst = src - pred as f32;
            }
        }
        let levu = basis.project_scan(&ru, 0.0, scan);
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&ch_choice.pred_u, &levu, qstep, scan, cw, ch, bd),
        );
        levu
    } else {
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let levu = basis.project_scan(
            &get_residual_rect(up, pw, sb_y, sb_x, cw, ch, predu),
            0.0,
            scan,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma(predu, &levu, qstep, scan, cw, ch, bd),
        );
        levu
    };
    // V plane.
    let levv = if win {
        let ch_choice = mhccp_choice.as_ref().unwrap();
        let mut rv = vec![0f32; cw * ch];
        for ((dst_row, src_row), pred_row) in rv
            .chunks_exact_mut(cw)
            .zip(rect_rows(vp, pw, sb_y, sb_x, cw, ch))
            .zip(ch_choice.pred_v.chunks_exact(cw))
        {
            for ((dst, &src), &pred) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
                *dst = src - pred as f32;
            }
        }
        let levv = basis.project_scan(&rv, 0.0, scan);
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma_cfl(&ch_choice.pred_v, &levv, qstep, scan, cw, ch, bd),
        );
        levv
    } else {
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, cw, ch, neutral, bd);
        let levv = basis.project_scan(
            &get_residual_rect(vp, pw, sb_y, sb_x, cw, ch, predv),
            0.0,
            scan,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            cw,
            ch,
            &itx422::reconstruct_chroma(predv, &levv, qstep, scan, cw, ch, bd),
        );
        levv
    };
    let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
    let cbwl = (cw.min(32) as f32).log2() as i32;
    let u_skip = u_skip_row[(6 + ua + ul) as usize] as u32;
    encode_chroma_block_rect_w(enc, &uc, u_skip, true, scan, eob_cdf, eob_hi, area, cbwl);
    let up_ = uc.iter().any(|&(_, l)| l != 0);
    let v_skip = (6 * (up_ as i32) + va + vl) as u32;
    encode_chroma_block_rect_w(enc, &vc, v_skip, false, scan, eob_cdf, eob_hi, area, cbwl);
    (up_, vc.iter().any(|&(_, l)| l != 0))
}
