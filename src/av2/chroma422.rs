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

/// Shape-constant transform parameters for one chroma TU geometry; identical for every
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
    cfl: Option<&crate::av2::cfl::CflChoice>,
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
    let QuantCtx { qc, neutral, qstep } = quant;
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
        let levu = basis.project_scan(&aq::scale_resid(&ru, cscale), 0.0, scan);
        let levv = basis.project_scan(&aq::scale_resid(&rv, cscale), 0.0, scan);
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
    } else {
        let predu = dc_pred_rect(recu, pcw, cy, cx, cw, ch, neutral, bd);
        let levu = basis.project_scan(
            &crate::av2::aq::scale_resid(
                &get_residual_rect(up, pcw, cy, cx, cw, ch, predu),
                cscale,
            ),
            0.0,
            scan,
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
        let levv = basis.project_scan(
            &crate::av2::aq::scale_resid(
                &get_residual_rect(vp, pcw, cy, cx, cw, ch, predv),
                cscale,
            ),
            0.0,
            scan,
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
