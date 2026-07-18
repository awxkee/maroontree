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
use super::chroma::{
    Chroma444Block, Chroma444Planes, Chroma444Quant, Chroma444Tx, code_444_chroma_leaf,
};
use crate::av2::cfl::{Cfl64Input, ChromaRdSpec, cfl_decide_64, cfl_prediction};
use crate::av2::video::mc;

/// Outline independent 4:4:4 candidate searches and emit phases. Calls happen
/// per superblock, keeping the boundary well outside transform inner loops.
#[inline(never)]
fn outline_444<R>(f: impl FnOnce() -> R) -> R {
    f()
}

#[derive(Clone, Copy)]
pub(super) struct Core444Planes<'a> {
    pub(super) y: &'a [f32],
    pub(super) u: &'a [f32],
    pub(super) v: &'a [f32],
}

/// One superblock's parallel-decide output: the captured record + its deblock TU
/// rectangles. Merged in raster order after the wavefront so the serial Replay
/// consumes the exact sequence the serial Capture would have produced.
struct WfSlot {
    record: replay::DecisionRecord,
    db_tx: Vec<(usize, usize, usize, usize)>,
    db_tx_c: Vec<(usize, usize, usize, usize)>,
}

/// Per-worker reusable scratch for the wavefront decide: a private full-plane recon
/// (zero own-block + halo of finished neighbours each cell) plus the emit-side
/// neighbour context arrays. Each cell must see the frame-initial context values,
/// not arbitrary values left by an unrelated SB previously handled by the worker.
/// One instance per thread via `thread_local!`, sized on first use.
#[derive(Default)]
struct WfScratch {
    ry: Vec<f32>,
    ru: Vec<f32>,
    rv: Vec<f32>,
    above: Vec<u8>,
    left: Vec<u8>,
    apctx: Vec<u8>,
    midx: Vec<u8>,
    ua: Vec<i32>,
    va: Vec<i32>,
    ul: Vec<i32>,
    vl: Vec<i32>,
    cfa: Vec<i32>,
    cfl: Vec<i32>,
    isa: Vec<u8>,
    iia: Vec<u8>,
    ina: Vec<u8>,
    ima: Vec<Option<video::mv::Mv>>,
    dbq: Vec<u16>,
    me: video::me::MeScratch<u16>,
}

struct Leaf4x4Finish<'a, 'record> {
    enc: &'a mut RangeEncoder,
    recy: &'a mut [f32],
    recu: &'a mut [f32],
    recv: &'a mut [f32],
    up: &'a [f32],
    vp: &'a [f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    lmr: usize,
    lmc: usize,
    split_qstep: i32,
    split_qstep_c: i32,
    mhccp_bounds: cfl::MhccpBounds,
    tu: Vec<Coeff>,
    tx_idx: usize,
    dcs: usize,
    skip: u32,
    pc: u32,
    ua: i32,
    ul: i32,
    va: i32,
    vl: i32,
    decide_mode: &'a mut replay::DecideMode<'record>,
}

impl WfScratch {
    fn ensure(&mut self, pw: usize, ph: usize, tmc: i64, tmr: i64, sb_cols: usize, sb_rows: usize) {
        let n = pw * ph;
        if self.ry.len() != n {
            self.ry = vec![0f32; n];
            self.ru = vec![0f32; n];
            self.rv = vec![0f32; n];
        }
        let (mc, mr) = (tmc as usize + 16, tmr as usize + 16);
        // A pool worker can next encode a frame with the same width but a different
        // height. Size every dimension-dependent array independently; keying all of
        // them off `above.len()` leaves `left`, `midx`, and `dbq` stale in that case.
        self.above.resize(pw / 4 + 16, 0x40);
        self.left.resize(ph / 4 + 16, 0x40);
        self.apctx.resize(mc, 0);
        self.midx.resize(sb_cols * sb_rows, 0xff);
        self.ua.resize(mc, 0);
        self.va.resize(mc, 0);
        self.ul.resize(mr, 0);
        self.vl.resize(mr, 0);
        self.cfa.resize(mc, 0);
        self.cfl.resize(mr, 0);
        self.isa.resize(sb_cols.max(1), 0);
        self.iia.resize(sb_cols.max(1), 0);
        self.ina.resize(sb_cols.max(1), 0);
        self.ima.resize(sb_cols.max(1), None);
        self.dbq.resize(sb_cols * sb_rows, 0);
    }

    /// Restore the frame-initial context state before every independently-decided
    /// SB. Thread-pool assignment is deliberately unstable,
    /// so carrying these arrays between cells makes the result depend on which
    /// unrelated cell happened to run previously on this worker.
    fn reset_contexts(&mut self) {
        self.above.fill(0x40);
        self.left.fill(0x40);
        self.apctx.fill(0);
        self.midx.fill(0xff);
        self.ua.fill(0);
        self.va.fill(0);
        self.ul.fill(0);
        self.vl.fill(0);
        self.cfa.fill(0);
        self.cfl.fill(0);
        self.isa.fill(0);
        self.iia.fill(0);
        self.ina.fill(0);
        self.ima.fill(None);
        self.dbq.fill(0);
    }
}

thread_local! {
    static WF_SCRATCH: std::cell::RefCell<WfScratch> = std::cell::RefCell::new(WfScratch::default());
}

impl Av2Encoder {
    /// SB-loop core for one 4:4:4 region (a whole frame, or one tile treated as a
    /// sub-frame). Returns the finished entropy coder; assembly (frame header/OBU,
    /// or multi-tile concatenation) happens in the caller.
    /// Encode one full-64 (16×16-mi) leaf on the partition-walk path — the
    /// extracted body of the walk's `(16, 16)` match arm. Behaviour-preserving
    /// (byte-identical); this is the first step of decoupling the `ops` walk into
    /// a per-leaf function so its search can later be recorded/replayed. Returns
    /// whether the U / V planes coded any coefficients.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf16(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        width: usize,
        height: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
        split_resid_scale: f32,
        split_qstep_c: i32,
        split_resid_scale_c: f32,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let rdoq_lambda = self.tune.rdoq_lambda;
        let part_lambda_c = self.tune.part_lambda_c;
        let (tus, mode_idx, _, _) = encode_luma_sb(
            recy,
            &LumaSource {
                plane: yp,
                stride: pw,
            },
            &LumaFrameBlock {
                frame_width: width,
                frame_height: height,
                y: sb_y,
                x: sb_x,
            },
            &LumaQuantSpec {
                basis: &bases.luma,
                qstep: split_qstep,
                scan: &tables::SCAN,
                neutral,
                quant_context: qc,
                rdoq_lambda,
                speed: self.speed,
                bit_depth: self.bit_depth as i32,
            },
            &LumaSbSearch {
                residual_scale: split_resid_scale,
                allow_directional: false,
            },
        );
        let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
        let cfl_choice = if enc.cfl {
            cfl_decide_64(
                &Cfl64Input {
                    reconstructed_luma: &*recy,
                    source_u: up,
                    source_v: vp,
                    reconstructed_u: &*recu,
                    reconstructed_v: &*recv,
                    stride: pw,
                    y: sb_y,
                    x: sb_x,
                    neutral,
                },
                &ChromaRdSpec {
                    basis: &bases.chroma444,
                    qstep: split_qstep_c,
                    lambda: leaf::part_lambda(split_qstep, part_lambda_c),
                    bit_depth: self.bit_depth as i32,
                },
            )
        } else {
            None
        };
        if let Some(ref ch) = cfl_choice {
            enc.cfl_use = true;
            enc.cfl_js = ch.js;
            enc.cfl_mag_u = ch.mag_u;
            enc.cfl_mag_v = ch.mag_v;
            enc.cfl_ctx_u = ch.ctx_u;
            enc.cfl_ctx_v = ch.ctx_v;
        }
        encode_luma_block_split(enc, &tus, &skip_cdfs, &dc_sign_ctxs, mode_idx, true, pc);
        let bd = self.bit_depth as i32;
        let (levu, levv) = if let Some(ref ch) = cfl_choice {
            let mut ru = [0f32; 64 * 64];
            let mut rv = [0f32; 64 * 64];
            cfl_prediction::<64>(pw, up, vp, sb_y, sb_x, &ch, &mut ru, &mut rv);
            let levu = chroma422::project_chroma_rdoq(
                &bases.chroma444,
                &aq::scale_resid(&ru, split_resid_scale_c),
                &tables::SCAN,
                qc,
                1024,
                0,
                self.tune.chroma_rdoq_lambda,
            );
            let levv = chroma422::project_chroma_rdoq(
                &bases.chroma444,
                &aq::scale_resid(&rv, split_resid_scale_c),
                &tables::SCAN,
                qc,
                1024,
                4,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recu,
                pw,
                sb_y,
                sb_x,
                64,
                &itx422::reconstruct_chroma_cfl(
                    &ch.pred_u,
                    &levu,
                    split_qstep_c,
                    &tables::SCAN,
                    64,
                    64,
                    bd,
                ),
            );
            put_block(
                recv,
                pw,
                sb_y,
                sb_x,
                64,
                &itx422::reconstruct_chroma_cfl(
                    &ch.pred_v,
                    &levv,
                    split_qstep_c,
                    &tables::SCAN,
                    64,
                    64,
                    bd,
                ),
            );
            (levu, levv)
        } else {
            let predu = dc_pred(recu, pw, sb_y, sb_x, 64, neutral);
            let levu = chroma422::project_chroma_rdoq(
                &bases.chroma444,
                &aq::scale_resid(
                    &get_residual(up, pw, sb_y, sb_x, 64, predu),
                    split_resid_scale_c,
                ),
                &tables::SCAN,
                qc,
                1024,
                0,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recu,
                pw,
                sb_y,
                sb_x,
                64,
                &itx422::reconstruct_chroma(predu, &levu, split_qstep_c, &tables::SCAN, 64, 64, bd),
            );
            let predv = dc_pred(recv, pw, sb_y, sb_x, 64, neutral);
            let levv = chroma422::project_chroma_rdoq(
                &bases.chroma444,
                &aq::scale_resid(
                    &get_residual(vp, pw, sb_y, sb_x, 64, predv),
                    split_resid_scale_c,
                ),
                &tables::SCAN,
                qc,
                1024,
                4,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recv,
                pw,
                sb_y,
                sb_x,
                64,
                &itx422::reconstruct_chroma(predv, &levv, split_qstep_c, &tables::SCAN, 64, 64, bd),
            );
            (levu, levv)
        };
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = (6 + ua + ul) as u32;
        encode_chroma_block(enc, &uc, u_skip, true);
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block(enc, &vc, v_skip, false);
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(16, 8)` arm — a 64×32 luma leaf with DC
    /// chroma. Behaviour-preserving; byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf16x8(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        split_resid_scale_c: f32,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let rdoq_lambda = self.tune.rdoq_lambda;
        let bd = self.bit_depth as i32;
        let (tus2, mode_idx) = encode_luma_leaf32(
            recy,
            &LumaSource {
                plane: yp,
                stride: pw,
            },
            &LumaGridBlock {
                mi_cols: tmc,
                mi_rows: tmr,
                y: sb_y,
                x: sb_x,
            },
            &LumaQuantSpec {
                basis: &bases.luma,
                qstep: split_qstep,
                scan: &tables::SCAN,
                neutral,
                quant_context: qc,
                rdoq_lambda,
                speed: self.speed,
                bit_depth: bd,
            },
        );
        let (skip2, dcs2) = sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
        encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, true, pc);
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, 64, 32, neutral, bd);
        let levu = chroma422::project_chroma_rdoq(
            &bases.chroma444_64x32,
            &aq::scale_resid(
                &get_residual_rect(up, pw, sb_y, sb_x, 64, 32, predu),
                split_resid_scale_c,
            ),
            &tables::SCAN,
            qc,
            1024,
            0,
            self.tune.chroma_rdoq_lambda,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            64,
            32,
            &itx422::reconstruct_chroma(predu, &levu, split_qstep_c, &tables::SCAN, 64, 32, bd),
        );
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, 64, 32, neutral, bd);
        let levv = chroma422::project_chroma_rdoq(
            &bases.chroma444_64x32,
            &aq::scale_resid(
                &get_residual_rect(vp, pw, sb_y, sb_x, 64, 32, predv),
                split_resid_scale_c,
            ),
            &tables::SCAN,
            qc,
            1024,
            4,
            self.tune.chroma_rdoq_lambda,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            64,
            32,
            &itx422::reconstruct_chroma(predv, &levv, split_qstep_c, &tables::SCAN, 64, 32, bd),
        );
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = (6 + ua + ul) as u32;
        encode_chroma_block(enc, &uc, u_skip, true);
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block(enc, &vc, v_skip, false);
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(8, 16)` arm — a 32×64 vertical luma leaf
    /// with DC chroma (chroma422 basis). Behaviour-preserving; byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf8x16(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        split_resid_scale_c: f32,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let rdoq_lambda = self.tune.rdoq_lambda;
        let bd = self.bit_depth as i32;
        let (tus2, mode_idx) = encode_luma_leaf_v32x64(
            recy,
            &LumaSource {
                plane: yp,
                stride: pw,
            },
            &LumaGridBlock {
                mi_cols: tmc,
                mi_rows: tmr,
                y: sb_y,
                x: sb_x,
            },
            &LumaQuantSpec {
                basis: &bases.luma,
                qstep: split_qstep,
                scan: &tables::SCAN,
                neutral,
                quant_context: qc,
                rdoq_lambda,
                speed: self.speed,
                bit_depth: bd,
            },
        );
        let (skip2, dcs2) = sb_tu_contexts_pos(
            &[(0, 0), (32, 0)],
            &tus2,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: false,
            },
        );
        let s2 = [skip2[0], skip2[1]];
        let d2 = [dcs2[0], dcs2[1]];
        encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, true, pc);
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, 32, 64, neutral, bd);
        let levu = chroma422::project_chroma_rdoq(
            &bases.chroma422,
            &aq::scale_resid(
                &get_residual_rect(up, pw, sb_y, sb_x, 32, 64, predu),
                split_resid_scale_c,
            ),
            &tables::SCAN,
            qc,
            1024,
            0,
            self.tune.chroma_rdoq_lambda,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            32,
            64,
            &itx422::reconstruct_chroma(predu, &levu, split_qstep_c, &tables::SCAN, 32, 64, bd),
        );
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, 32, 64, neutral, bd);
        let levv = chroma422::project_chroma_rdoq(
            &bases.chroma422,
            &aq::scale_resid(
                &get_residual_rect(vp, pw, sb_y, sb_x, 32, 64, predv),
                split_resid_scale_c,
            ),
            &tables::SCAN,
            qc,
            1024,
            4,
            self.tune.chroma_rdoq_lambda,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            32,
            64,
            &itx422::reconstruct_chroma(predv, &levv, split_qstep_c, &tables::SCAN, 32, 64, bd),
        );
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = (6 + ua + ul) as u32;
        encode_chroma_block(enc, &uc, u_skip, true);
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block(enc, &vc, v_skip, false);
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(8, 8)` arm — a 32×32 luma leaf whose chroma
    /// competes MHCCP against the DC incumbent (chroma420 basis). Behaviour-
    /// preserving; byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf8x8(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        split_resid_scale_c: f32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
        decide_mode: &mut crate::av2::replay::DecideMode<'_>,
        walk: crate::av2::replay::LeafWalk<'_>,
    ) -> (bool, bool) {
        use crate::av2::replay::{LeafDecision, LeafWalk};
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let rdoq_lambda = self.tune.rdoq_lambda;
        // Luma mode search (`encode_luma_leaf_s32x32`) is the only walk-leaf search
        // here (MHCCP is cached below). Replay restores the captured 32×32 recon +
        // reuses the captured coeffs/mode, skipping the search entirely.
        let (tu, mode_idx) = if let LeafWalk::Replay(d) = &walk {
            put_block(recy, pw, sb_y, sb_x, 32, &d.recon_y);
            (d.tu.clone(), d.mode_idx)
        } else {
            encode_luma_leaf_s32x32(
                recy,
                &LumaSource {
                    plane: yp,
                    stride: pw,
                },
                &LumaGridBlock {
                    mi_cols: tmc,
                    mi_rows: tmr,
                    y: sb_y,
                    x: sb_x,
                },
                &LumaQuantSpec {
                    basis: &bases.luma,
                    qstep: split_qstep,
                    scan: &tables::SCAN,
                    neutral,
                    quant_context: qc,
                    rdoq_lambda,
                    speed: self.speed,
                    bit_depth: self.bit_depth as i32,
                },
            )
        };
        if let LeafWalk::Capture(v) = walk {
            let mut recon_y = vec![0f32; 32 * 32];
            for (dst, src) in recon_y
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(rect_rows(&*recy, pw, sb_y, sb_x, 32, 32))
            {
                dst.copy_from_slice(src);
            }
            v.push(LeafDecision {
                bw_mi: 8,
                bh_mi: 8,
                tu: tu.clone(),
                mode_idx,
                recon_y,
            });
        }
        let (skip2, dcs2) = sb_tu_contexts_pos(
            &[(0, 0)],
            std::slice::from_ref(&tu),
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
        );
        let bd444 = self.bit_depth as i32;
        // Replay reuses the captured MHCCP choice, skipping the 3-direction filter
        // fit; Capture logs it. Exhausted cursor (shouldn't happen) falls through
        // to a re-search.
        let replayed_mh = if let crate::av2::replay::DecideMode::Replay(cur) = &mut *decide_mode {
            cur.next_mhccp()
        } else {
            None
        };
        let mh444 = match replayed_mh {
            Some(m) => m,
            None => {
                let m = if enc.mhccp && !enc.in_interior_split {
                    let dcu = dc_pred(recu, pw, sb_y, sb_x, 32, neutral);
                    let dcv = dc_pred(recv, pw, sb_y, sb_x, 32, neutral);
                    let baseline_j = pixel_sse_rounded_block_const(up, pw, sb_y, sb_x, 32, 32, dcu)
                        + pixel_sse_rounded_block_const(vp, pw, sb_y, sb_x, 32, 32, dcv);
                    let mut suf = [0f32; 32 * 32];
                    let mut svf = [0f32; 32 * 32];
                    for (((suf_row, svf_row), up_row), vp_row) in suf
                        .as_chunks_mut::<32>()
                        .0
                        .iter_mut()
                        .zip(svf.as_chunks_mut::<32>().0.iter_mut())
                        .zip(rect_rows(up, pw, sb_y, sb_x, 32, 32))
                        .zip(rect_rows(vp, pw, sb_y, sb_x, 32, 32))
                    {
                        suf_row.copy_from_slice(up_row);
                        svf_row.copy_from_slice(vp_row);
                    }
                    let mctx = cfl::MhccpCtx {
                        recy: &*recy,
                        pw,
                        recu: &*recu,
                        recv: &*recv,
                        pcw: pw,
                        bounds: mhccp_bounds,
                        ly: sb_y,
                        lx: sb_x,
                        cy: sb_y,
                        cx: sb_x,
                        ssx: false,
                        ssy: false,
                        have_top: lmr > 0,
                        have_left: lmc > 0,
                        is_top_sb_boundary: (sb_y & 63) == 0,
                        size_group: cfl::mhccp_size_group_wh4(8, 8),
                        coded: &[],
                    };
                    cfl::mhccp_decide(
                        &mctx,
                        &cfl::MhccpDecisionInput {
                            source_u: &suf,
                            source_v: &svf,
                            width: 32,
                            height: 32,
                            rd: cfl::ChromaRdSpec {
                                basis: &bases.chroma420,
                                qstep: split_qstep_c,
                                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                                bit_depth: bd444,
                            },
                            scan: &tables::SCAN,
                            baseline_j,
                        },
                    )
                } else {
                    None
                };
                if let crate::av2::replay::DecideMode::Capture(rec) = &mut *decide_mode {
                    rec.push_mhccp(m.clone());
                }
                m
            }
        };
        let mh_win = mh444.as_ref().and_then(|c| c.mhccp.as_ref()).is_some();
        enc.uv_mode = 0;
        if let Some(ch) = if mh_win { mh444.as_ref() } else { None } {
            enc.cfl_use = true;
            enc.mhccp_use = true;
            if let Some(ref mh) = ch.mhccp {
                enc.mhccp_dir = mh.mh_dir;
                enc.mhccp_size_group = mh.size_group;
            }
            enc.uv_mode = 0;
        }
        encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc);
        let (levu, levv) = if mh_win {
            let ch = mh444.as_ref().unwrap();
            let mut ru = [0f32; 32 * 32];
            let mut rv = [0f32; 32 * 32];
            cfl_prediction::<32>(pw, up, vp, sb_y, sb_x, &ch, &mut ru, &mut rv);
            let levu = chroma422::project_chroma_rdoq(
                &bases.chroma420,
                &aq::scale_resid(&ru, split_resid_scale_c),
                &tables::SCAN,
                qc,
                1024,
                0,
                self.tune.chroma_rdoq_lambda,
            );
            let levv = chroma422::project_chroma_rdoq(
                &bases.chroma420,
                &aq::scale_resid(&rv, split_resid_scale_c),
                &tables::SCAN,
                qc,
                1024,
                4,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recu,
                pw,
                sb_y,
                sb_x,
                32,
                &itx422::reconstruct_chroma_cfl(
                    &ch.pred_u,
                    &levu,
                    split_qstep_c,
                    &tables::SCAN,
                    32,
                    32,
                    bd444,
                ),
            );
            put_block(
                recv,
                pw,
                sb_y,
                sb_x,
                32,
                &itx422::reconstruct_chroma_cfl(
                    &ch.pred_v,
                    &levv,
                    split_qstep_c,
                    &tables::SCAN,
                    32,
                    32,
                    bd444,
                ),
            );
            (levu, levv)
        } else {
            let predu = dc_pred(recu, pw, sb_y, sb_x, 32, neutral);
            let levu = chroma422::project_chroma_rdoq(
                &bases.chroma420,
                &aq::scale_resid(
                    &get_residual(up, pw, sb_y, sb_x, 32, predu),
                    split_resid_scale_c,
                ),
                &tables::SCAN,
                qc,
                1024,
                0,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recu,
                pw,
                sb_y,
                sb_x,
                32,
                &itx422::reconstruct_chroma(
                    predu,
                    &levu,
                    split_qstep_c,
                    &tables::SCAN,
                    32,
                    32,
                    self.bit_depth as i32,
                ),
            );
            let predv = dc_pred(recv, pw, sb_y, sb_x, 32, neutral);
            let levv = chroma422::project_chroma_rdoq(
                &bases.chroma420,
                &aq::scale_resid(
                    &get_residual(vp, pw, sb_y, sb_x, 32, predv),
                    split_resid_scale_c,
                ),
                &tables::SCAN,
                qc,
                1024,
                4,
                self.tune.chroma_rdoq_lambda,
            );
            put_block(
                recv,
                pw,
                sb_y,
                sb_x,
                32,
                &itx422::reconstruct_chroma(
                    predv,
                    &levv,
                    split_qstep_c,
                    &tables::SCAN,
                    32,
                    32,
                    self.bit_depth as i32,
                ),
            );
            (levu, levv)
        };
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = (6 + ua + ul) as u32;
        encode_chroma_block_ex(enc, &uc, u_skip, true, false);
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block(enc, &vc, v_skip, false);
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(16, 4)` arm — a bottom-edge 64×16 luma leaf
    /// that RD-picks single TX_64X16 vs a 2×TX_32X16 vertical tx-partition, with DC
    /// chroma. Behaviour-preserving; byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf16x4(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
        split_resid_scale: f32,
        split_qstep_c: i32,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 64, 16, neutral, bd);
        let resid = aq::scale_resid(
            &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
            split_resid_scale,
        );
        let rate = coeff_rate_f32;
        let sse_vs = |rec: &[f32], w: usize, xoff: usize| -> f32 {
            pixel_sse_rounded_block(yp, pw, sb_y, sb_x + xoff, rec, w, w, 16)
        };
        let lambda = leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
        let lev = bases.luma64x16.project_scan(&resid, 0.0, &SCAN32X16);
        let rec_a = itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &SCAN32X16, 64, 16, bd);
        let j_a = sse_vs(&rec_a, 64, 0) + lambda * rate(&lev);
        let mut levs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut recs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut j_b = lambda * 4.0;
        {
            let mut scratch = recy.to_vec();
            for half in 0..2 {
                let tux = sb_x + half * 32;
                let p = dc_pred_rect(&scratch, pw, sb_y, tux, 32, 16, neutral, bd);
                let r = aq::scale_resid(
                    &get_residual_rect(yp, pw, sb_y, tux, 32, 16, p),
                    split_resid_scale,
                );
                let l = bases.luma32x16.project_scan(&r, 0.0, &SCAN32X16);
                let rec = itx422::reconstruct_chroma(p, &l, split_qstep_c, &SCAN32X16, 32, 16, bd);
                put_block_rect(&mut scratch, pw, sb_y, tux, 32, 16, &rec);
                j_b += sse_vs(&rec, 32, half * 32) + lambda * rate(&l);
                levs_b[half] = l;
                recs_b[half] = rec;
            }
        }
        if j_b < j_a {
            for (half, src) in recs_b.iter().enumerate() {
                put_block_rect(recy, pw, sb_y, sb_x + half * 32, 32, 16, src);
            }
            let tus: [Vec<Coeff>; 2] = [levels_to_coeffs(&levs_b[0]), levels_to_coeffs(&levs_b[1])];
            let mut skips = [0u32; 2];
            let mut dcss = [0usize; 2];
            for half in 0..2 {
                let (sk, dc) = sb_tu_contexts_rect(
                    &tus[half],
                    above,
                    left,
                    &TxbContextSpec {
                        sb_y,
                        sb_x: sb_x + half * 32,
                        qc,
                        mi_cols: tmc,
                        mi_rows: tmr,
                        block_eq_tx: false,
                    },
                    8,
                    4,
                );
                skips[half] = sk;
                dcss[half] = dc;
            }
            coder::encode_luma_leaf_64x16_vert(enc, &tus, &skips, &dcss, 0, true, pc);
        } else {
            put_block_rect(recy, pw, sb_y, sb_x, 64, 16, &rec_a);
            let tu = levels_to_coeffs(&lev);
            let (skip, dcs) = sb_tu_contexts_rect(
                &tu,
                above,
                left,
                &TxbContextSpec {
                    sb_y,
                    sb_x,
                    qc,
                    mi_cols: tmc,
                    mi_rows: tmr,
                    block_eq_tx: true,
                },
                16,
                4,
            );
            encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, true, pc);
        }
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, 64, 16, neutral, self.bit_depth as i32);
        let levu = bases.luma64x16.project_scan(
            &get_residual_rect(up, pw, sb_y, sb_x, 64, 16, predu),
            0.0,
            &SCAN32X16,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            64,
            16,
            &itx422::reconstruct_chroma(
                predu,
                &levu,
                split_qstep_c,
                &SCAN32X16,
                64,
                16,
                self.bit_depth as i32,
            ),
        );
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, 64, 16, neutral, self.bit_depth as i32);
        let levv = bases.luma64x16.project_scan(
            &get_residual_rect(vp, pw, sb_y, sb_x, 64, 16, predv),
            0.0,
            &SCAN32X16,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            64,
            16,
            &itx422::reconstruct_chroma(
                predv,
                &levv,
                split_qstep_c,
                &SCAN32X16,
                64,
                16,
                self.bit_depth as i32,
            ),
        );
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
        encode_chroma_block_rect(
            enc,
            &uc,
            u_skip,
            true,
            &SCAN32X16,
            EobCdf::ChrEob512,
            CHROMA_EOB_HI_BIT_QC[qc],
            512,
        );
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block_rect(
            enc,
            &vc,
            v_skip,
            false,
            &SCAN32X16,
            EobCdf::ChrEob512,
            CHROMA_EOB_HI_BIT_QC[qc],
            512,
        );
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(4, 4)` arm — the bottom-right 16×16 corner:
    /// RD-picks the luma 16×16 tx-type (DCT/ADST variants) + `preset_444_mhccp` +
    /// `code_444_chroma_leaf`. Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf4x4(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
        decide_mode: &mut crate::av2::replay::DecideMode<'_>,
        walk: crate::av2::replay::LeafWalk<'_>,
    ) -> (bool, bool) {
        use crate::av2::replay::{LeafDecision, LeafWalk};
        let qc = enc.qc;
        // Luma tx-type RD (four 16×16 transforms + `choose_tx16_type`) is the only
        // walk-leaf search here (MHCCP is cached below). Replay restores the
        // captured 16×16 recon + reuses the captured coeffs/tx-index, skipping it.
        let (tu, tx_idx) = if let LeafWalk::Replay(d) = &walk {
            put_block_rect(recy, pw, sb_y, sb_x, 16, 16, &d.recon_y);
            (d.tu.clone(), d.mode_idx)
        } else {
            self.leaf4x4_luma_search(recy, yp, pw, sb_y, sb_x, split_qstep)
        };
        if let LeafWalk::Capture(v) = walk {
            let mut recon_y = vec![0f32; 16 * 16];
            for (dst, src) in recon_y
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(rect_rows(&*recy, pw, sb_y, sb_x, 16, 16))
            {
                dst.copy_from_slice(src);
            }
            v.push(LeafDecision {
                bw_mi: 4,
                bh_mi: 4,
                tu: tu.clone(),
                mode_idx: tx_idx,
                recon_y,
            });
        }
        let (_s, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            4,
            4,
        );
        let skip = SKIP_TX16_QC[qc][0] as u32;
        self.leaf4x4_finish_emit(Leaf4x4Finish {
            enc,
            recy,
            recu,
            recv,
            up,
            vp,
            pw,
            sb_y,
            sb_x,
            lmr,
            lmc,
            split_qstep,
            split_qstep_c,
            mhccp_bounds,
            tu,
            tx_idx,
            dcs,
            skip,
            pc,
            ua,
            ul,
            va,
            vl,
            decide_mode,
        })
    }

    /// The luma tx-type RD search for the walk's `(4,4)` leaf (a 16×16 luma leaf):
    /// projects the four 16×16 tx types (DCT / ADST / ADST-DCT / DCT-ADST) and
    /// `choose_tx16_type`s the winner. Writes the winning recon into `recy` and
    /// returns `(coeffs, tx_index)`. Split out so replay can skip it wholesale.
    fn leaf4x4_luma_search(
        &self,
        recy: &mut [f32],
        yp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
    ) -> (Vec<Coeff>, usize) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 16, neutral, self.bit_depth as i32);
        let resid = {
            let mut r = get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred);
            let sc = bases.luma16x16.qstep as f32 / split_qstep as f32;
            if sc != 1.0 {
                for v in r.iter_mut() {
                    *v *= sc;
                }
            }
            r
        };
        let pred_flat = [pred; 256];
        let mut src16 = [0f32; 256];
        for (dst_row, src_row) in src16
            .as_chunks_mut::<16>()
            .0
            .iter_mut()
            .zip(rect_rows(yp, pw, sb_y, sb_x, 16, 16))
        {
            dst_row.copy_from_slice(src_row);
        }
        let rate = coeff_rate_f32;
        let sse = |rec: &[f32]| -> f32 { tx16_distortion(&src16, rec) };
        let lambda = crate::av2::leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
        let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
        let rec_dct = crate::av2::itx422::reconstruct_luma16(
            &pred_flat,
            &lev_dct,
            split_qstep,
            &SCAN16,
            self.bit_depth as i32,
        );
        let dist_dct = sse(&rec_dct);
        let cost_dct = dist_dct + lambda * rate(&lev_dct);
        let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
        let rec_adst = itx422::reconstruct_luma16_adst(
            &pred_flat,
            &lev_adst,
            split_qstep,
            &SCAN16,
            true,
            true,
            self.bit_depth as i32,
        );
        let dist_adst = sse(&rec_adst);
        let cost_adst = dist_adst + lambda * (rate(&lev_adst) + TX16_TYPE_RATE_DELTA[1]);
        let lev_ad = bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
        let rec_ad = itx422::reconstruct_luma16_adst(
            &pred_flat,
            &lev_ad,
            split_qstep,
            &SCAN16,
            false,
            true,
            self.bit_depth as i32,
        );
        let dist_ad = sse(&rec_ad);
        let cost_ad = dist_ad + lambda * (rate(&lev_ad) + TX16_TYPE_RATE_DELTA[2]);
        let lev_da = bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
        let rec_da = itx422::reconstruct_luma16_adst(
            &pred_flat,
            &lev_da,
            split_qstep,
            &SCAN16,
            true,
            false,
            self.bit_depth as i32,
        );
        let dist_da = sse(&rec_da);
        let cost_da = dist_da + lambda * (rate(&lev_da) + TX16_TYPE_RATE_DELTA[3]);
        let choice = choose_tx16_type(
            [cost_dct, cost_adst, cost_ad, cost_da],
            [dist_dct, dist_adst, dist_ad, dist_da],
            [
                false,
                tx16_dc_only(&lev_adst),
                tx16_dc_only(&lev_ad),
                tx16_dc_only(&lev_da),
            ],
        );
        let (lev, rec, tx_idx): (&[f32], &[f32; 256], usize) = match choice {
            1 => (&lev_adst, &rec_adst, 1),
            2 => (&lev_ad, &rec_ad, 2),
            3 => (&lev_da, &rec_da, 3),
            _ => (&lev_dct, &rec_dct, 0),
        };
        put_block_rect(recy, pw, sb_y, sb_x, 16, 16, rec);
        let tu: Vec<Coeff> = levels_to_coeffs(lev);
        (tu, tx_idx)
    }

    /// The MHCCP chroma decide (cached via `decide_mode`) + entropy emit tail for
    /// the walk's `(4,4)` leaf. Split from the luma search so a replay run reuses
    /// it verbatim while skipping the search. `dcs`/`skip` are the luma DC-sign ctx
    /// and skip-cdf derived from the (captured) coeffs.
    fn leaf4x4_finish_emit(&self, finish: Leaf4x4Finish<'_, '_>) -> (bool, bool) {
        let Leaf4x4Finish {
            enc,
            recy,
            recu,
            recv,
            up,
            vp,
            pw,
            sb_y,
            sb_x,
            lmr,
            lmc,
            split_qstep,
            split_qstep_c,
            mhccp_bounds,
            tu,
            tx_idx,
            dcs,
            skip,
            pc,
            ua,
            ul,
            va,
            vl,
            decide_mode,
        } = finish;
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        {
            let mh_on = enc.mhccp;
            let __cached = if let crate::av2::replay::DecideMode::Replay(cur) = &mut *decide_mode {
                cur.next_mhccp()
            } else {
                None
            };
            let __choice = crate::av2::y444::chroma::preset_444_mhccp(
                enc,
                &*recy,
                &*recu,
                &*recv,
                up,
                vp,
                pw,
                &Chroma444Block {
                    bounds: mhccp_bounds,
                    y: sb_y,
                    x: sb_x,
                    width: 16,
                    height: 16,
                    have_top: lmr > 0,
                    have_left: lmc > 0,
                },
                &Chroma444Tx {
                    basis: &bases.luma16x16,
                    scan: &SCAN16,
                    eob_cdf: EobCdf::ChrEob256,
                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                    area: 256,
                    u_skip_row: &SKIP_TX16_QC[qc],
                },
                &Chroma444Quant {
                    neutral,
                    qstep: split_qstep_c,
                    lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                    bit_depth: self.bit_depth as i32,
                },
                mh_on,
                __cached,
            );
            if let crate::av2::replay::DecideMode::Capture(rec) = &mut *decide_mode {
                rec.push_mhccp(__choice);
            }
        }
        encode_luma_leaf_16x16_full(enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx);
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 16,
                height: 16,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.luma16x16,
                scan: &SCAN16,
                eob_cdf: EobCdf::ChrEob256,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 256,
                u_skip_row: &SKIP_TX16_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(4, 16)` arm — a right-edge 16×64 luma leaf
    /// that RD-picks single TX_16X64 vs a 2×TX_16X32 horizontal tx-partition, with
    /// DC chroma. Behaviour-preserving; byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf4x16(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        split_qstep: i32,
        split_resid_scale: f32,
        split_qstep_c: i32,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 64, neutral, bd);
        let resid = aq::scale_resid(
            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
            split_resid_scale,
        );
        let rate = coeff_rate_f32;
        let sse_vs = |rec: &[f32], h2: usize, yoff: usize| -> f32 {
            pixel_sse_rounded_block(yp, pw, sb_y + yoff, sb_x, rec, 16, 16, h2)
        };
        let lambda = leaf::part_lambda(split_qstep, self.tune.part_lambda_c);
        let lev = bases.luma16x64.project_scan(&resid, 0.0, &SCAN16X32);
        let rec_a = itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &SCAN16X32, 16, 64, bd);
        let j_a = sse_vs(&rec_a, 64, 0) + lambda * rate(&lev);
        let mut levs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut recs_b: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        let mut j_b = lambda * 4.0;
        {
            let mut scratch = recy.to_vec();
            for half in 0..2 {
                let tuy = sb_y + half * 32;
                let p = dc_pred_rect(&scratch, pw, tuy, sb_x, 16, 32, neutral, bd);
                let r = aq::scale_resid(
                    &get_residual_rect(yp, pw, tuy, sb_x, 16, 32, p),
                    split_resid_scale,
                );
                let l = bases.luma16x32.project_scan(&r, 0.0, &SCAN16X32);
                let rec = itx422::reconstruct_chroma(p, &l, split_qstep_c, &SCAN16X32, 16, 32, bd);
                put_block_rect(&mut scratch, pw, tuy, sb_x, 16, 32, &rec);
                j_b += sse_vs(&rec, 32, half * 32) + lambda * rate(&l);
                levs_b[half] = l;
                recs_b[half] = rec;
            }
        }
        if j_b < j_a {
            for (half, src) in recs_b.iter().enumerate() {
                put_block_rect(recy, pw, sb_y + half * 32, sb_x, 16, 32, src);
            }
            let tus: [Vec<Coeff>; 2] = [levels_to_coeffs(&levs_b[0]), levels_to_coeffs(&levs_b[1])];
            let mut skips = [0u32; 2];
            let mut dcss = [0usize; 2];
            for half in 0..2 {
                let (sk, dc) = sb_tu_contexts_rect(
                    &tus[half],
                    above,
                    left,
                    &TxbContextSpec {
                        sb_y: sb_y + half * 32,
                        sb_x,
                        qc,
                        mi_cols: tmc,
                        mi_rows: tmr,
                        block_eq_tx: false,
                    },
                    4,
                    8,
                );
                skips[half] = sk;
                dcss[half] = dc;
            }
            coder::encode_luma_leaf_16x64_horz(enc, &tus, &skips, &dcss, 0, true, pc);
        } else {
            put_block_rect(recy, pw, sb_y, sb_x, 16, 64, &rec_a);
            let tu = levels_to_coeffs(&lev);
            let (skip, dcs) = sb_tu_contexts_rect(
                &tu,
                above,
                left,
                &TxbContextSpec {
                    sb_y,
                    sb_x,
                    qc,
                    mi_cols: tmc,
                    mi_rows: tmr,
                    block_eq_tx: true,
                },
                4,
                16,
            );
            encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, true, pc);
        }
        let predu = dc_pred_rect(recu, pw, sb_y, sb_x, 16, 64, neutral, self.bit_depth as i32);
        let levu = bases.luma16x64.project_scan(
            &get_residual_rect(up, pw, sb_y, sb_x, 16, 64, predu),
            0.0,
            &SCAN16X32,
        );
        put_block_rect(
            recu,
            pw,
            sb_y,
            sb_x,
            16,
            64,
            &itx422::reconstruct_chroma(
                predu,
                &levu,
                split_qstep_c,
                &SCAN16X32,
                16,
                64,
                self.bit_depth as i32,
            ),
        );
        let predv = dc_pred_rect(recv, pw, sb_y, sb_x, 16, 64, neutral, self.bit_depth as i32);
        let levv = bases.luma16x64.project_scan(
            &get_residual_rect(vp, pw, sb_y, sb_x, 16, 64, predv),
            0.0,
            &SCAN16X32,
        );
        put_block_rect(
            recv,
            pw,
            sb_y,
            sb_x,
            16,
            64,
            &itx422::reconstruct_chroma(
                predv,
                &levv,
                split_qstep_c,
                &SCAN16X32,
                16,
                64,
                self.bit_depth as i32,
            ),
        );
        let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
        let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
        encode_chroma_block_rect_w(
            enc,
            &uc,
            u_skip,
            true,
            &SCAN16X32,
            EobCdf::ChrEob512,
            CHROMA_EOB_HI_BIT_QC[qc],
            512,
            4,
        );
        let up_ = uc.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (up_ as i32) + va + vl) as u32;
        encode_chroma_block_rect_w(
            enc,
            &vc,
            v_skip,
            false,
            &SCAN16X32,
            EobCdf::ChrEob512,
            CHROMA_EOB_HI_BIT_QC[qc],
            512,
            4,
        );
        (up_, vc.iter().any(|&(_, l)| l != 0))
    }

    /// Extracted body of the walk's `(2, 2)` arm — the both-axis residue-2 corner
    /// (8×8 luma with AQ residual rescale + a `preset_444_mhccp` mode decision +
    /// `code_444_chroma_leaf` chroma, eob64). Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf2x2(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
        decide_mode: &mut crate::av2::replay::DecideMode<'_>,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 8, neutral, bd);
        let resid8 = {
            let mut r = get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred);
            let sc = bases.c8x8.qstep as f32 / split_qstep as f32;
            if sc != 1.0 {
                for v in r.iter_mut() {
                    *v *= sc;
                }
            }
            r
        };
        let lev = bases.c8x8.project_scan(&resid8, 0.0, &SCAN8X8);
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            8,
            8,
            &itx422::reconstruct_chroma(pred, &lev, split_qstep, &SCAN8X8, 8, 8, bd),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            2,
            2,
        );
        {
            let mh_on = enc.mhccp;
            let __cached = if let crate::av2::replay::DecideMode::Replay(cur) = &mut *decide_mode {
                cur.next_mhccp()
            } else {
                None
            };
            let __choice = crate::av2::y444::chroma::preset_444_mhccp(
                enc,
                &*recy,
                &*recu,
                &*recv,
                up,
                vp,
                pw,
                &Chroma444Block {
                    bounds: mhccp_bounds,
                    y: sb_y,
                    x: sb_x,
                    width: 8,
                    height: 8,
                    have_top: lmr > 0,
                    have_left: lmc > 0,
                },
                &Chroma444Tx {
                    basis: &bases.c8x8,
                    scan: &SCAN8X8,
                    eob_cdf: EobCdf::ChrEob64,
                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                    area: 64,
                    u_skip_row: &SKIP_TX8_QC[qc],
                },
                &Chroma444Quant {
                    neutral,
                    qstep: split_qstep_c,
                    lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                    bit_depth: self.bit_depth as i32,
                },
                mh_on,
                __cached,
            );
            if let crate::av2::replay::DecideMode::Capture(rec) = &mut *decide_mode {
                rec.push_mhccp(__choice);
            }
        }
        encode_luma_leaf_8x8(
            enc,
            &tu,
            skip,
            dcs,
            0,
            true,
            pc,
            3148,
            Some((&crate::av2::coder::TXTP_EXT8, 0, 6)),
        );
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 8,
                height: 8,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.c8x8,
                scan: &SCAN8X8,
                eob_cdf: EobCdf::ChrEob64,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 64,
                u_skip_row: &SKIP_TX8_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(4, 2)` arm — a 16×8 residue leaf
    /// (rect128 luma) with `code_444_chroma_leaf` chroma (eob128). Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf4x2(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
        let lev = bases.c16x8.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 8, pred),
            0.0,
            &tables::SCAN16X8,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            16,
            8,
            &itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &tables::SCAN16X8, 16, 8, bd),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            4,
            2,
        );
        coder::encode_luma_leaf_rect128(
            enc,
            &tu,
            &coder::LumaLeafRect128Spec {
                skip_cdf: skip,
                dc_sign_ctx: dcs,
                mode_idx: 0,
                has_chroma: true,
                width_mi: 4,
                height_mi: 2,
                part_cdf: pc,
                tx_part_cdf: 12348,
                scan: &tables::SCAN16X8,
                tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
            },
        );
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 16,
                height: 8,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.c16x8,
                scan: &tables::SCAN16X8,
                eob_cdf: EobCdf::ChrEob128,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 128,
                u_skip_row: &SKIP_TX16_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(2, 4)` arm — an 8×16 residue leaf
    /// (rect128 luma) with `code_444_chroma_leaf` chroma (eob128). Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf2x4(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
        let lev = bases.c8x16.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 8, 16, pred),
            0.0,
            &tables::SCAN8X16,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            8,
            16,
            &itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &tables::SCAN8X16, 8, 16, bd),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            2,
            4,
        );
        coder::encode_luma_leaf_rect128(
            enc,
            &tu,
            &coder::LumaLeafRect128Spec {
                skip_cdf: skip,
                dc_sign_ctx: dcs,
                mode_idx: 0,
                has_chroma: true,
                width_mi: 2,
                height_mi: 4,
                part_cdf: pc,
                tx_part_cdf: 12348,
                scan: &tables::SCAN8X16,
                tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
            },
        );
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 8,
                height: 16,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.c8x16,
                scan: &tables::SCAN8X16,
                eob_cdf: EobCdf::ChrEob128,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 128,
                u_skip_row: &SKIP_TX16_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(8, 4)` arm — a 32×16 residue leaf (DC luma)
    /// with `code_444_chroma_leaf` chroma (eob512). Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf8x4(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
        let lev = bases.luma32x16.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 32, 16, pred),
            0.0,
            &SCAN32X16,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            32,
            16,
            &itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &SCAN32X16, 32, 16, bd),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            8,
            4,
        );
        encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, true, pc);
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 32,
                height: 16,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.luma32x16,
                scan: &SCAN32X16,
                eob_cdf: EobCdf::ChrEob512,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 512,
                u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(4, 8)` arm — a 16×32 residue leaf (DC luma)
    /// with `code_444_chroma_leaf` chroma (eob512). Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf4x8(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let bd = self.bit_depth as i32;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
        let lev = bases.luma16x32.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 32, pred),
            0.0,
            &SCAN16X32,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            16,
            32,
            &itx422::reconstruct_chroma(pred, &lev, split_qstep_c, &SCAN16X32, 16, 32, bd),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            4,
            8,
        );
        encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, true, pc);
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 16,
                height: 32,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.luma16x32,
                scan: &SCAN16X32,
                eob_cdf: EobCdf::ChrEob512,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 512,
                u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(8, 2)` arm — a bottom-edge 32×8 residue-2
    /// luma leaf (DC-only) with `code_444_chroma_leaf` chroma. Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf8x2(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 8, neutral, self.bit_depth as i32);
        let lev = bases.luma32x8.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 32, 8, pred),
            0.0,
            &SCAN32X8,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            32,
            8,
            &itx422::reconstruct_chroma(
                pred,
                &lev,
                split_qstep_c,
                &SCAN32X8,
                32,
                8,
                self.bit_depth as i32,
            ),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            8,
            2,
        );
        encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, true, pc);
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 32,
                height: 8,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.luma32x8,
                scan: &SCAN32X8,
                eob_cdf: EobCdf::ChrEob256,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 256,
                u_skip_row: &SKIP_TX16_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    /// Extracted body of the walk's `(2, 8)` arm — a right-edge 8×32 residue-2
    /// luma leaf (DC-only) with `code_444_chroma_leaf` chroma. Byte-identical.
    #[allow(clippy::too_many_arguments)]
    fn encode_walk_leaf2x8(
        &self,
        enc: &mut RangeEncoder,
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        pw: usize,
        sb_y: usize,
        sb_x: usize,
        lmr: usize,
        lmc: usize,
        split_qstep: i32,
        split_qstep_c: i32,
        mhccp_bounds: cfl::MhccpBounds,
        above: &mut [u8],
        left: &mut [u8],
        tmc: i64,
        tmr: i64,
        pc: u32,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
    ) -> (bool, bool) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 32, neutral, self.bit_depth as i32);
        let lev = bases.luma8x32.project_scan(
            &get_residual_rect(yp, pw, sb_y, sb_x, 8, 32, pred),
            0.0,
            &SCAN8X32,
        );
        put_block_rect(
            recy,
            pw,
            sb_y,
            sb_x,
            8,
            32,
            &itx422::reconstruct_chroma(
                pred,
                &lev,
                split_qstep_c,
                &SCAN8X32,
                8,
                32,
                self.bit_depth as i32,
            ),
        );
        let tu: Vec<Coeff> = levels_to_coeffs(&lev);
        let (skip, dcs) = sb_tu_contexts_rect(
            &tu,
            above,
            left,
            &TxbContextSpec {
                sb_y,
                sb_x,
                qc,
                mi_cols: tmc,
                mi_rows: tmr,
                block_eq_tx: true,
            },
            2,
            8,
        );
        encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, true, pc);
        let mh_on = enc.mhccp;
        code_444_chroma_leaf(
            enc,
            &mut Chroma444Planes {
                reconstructed_luma: &*recy,
                reconstructed_u: recu,
                reconstructed_v: recv,
                source_u: up,
                source_v: vp,
                stride: pw,
            },
            &Chroma444Block {
                bounds: mhccp_bounds,
                y: sb_y,
                x: sb_x,
                width: 8,
                height: 32,
                have_top: lmr > 0,
                have_left: lmc > 0,
            },
            &Chroma444Tx {
                basis: &bases.luma8x32,
                scan: &SCAN8X32,
                eob_cdf: EobCdf::ChrEob256,
                eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                area: 256,
                u_skip_row: &SKIP_TX16_QC[qc],
            },
            &Chroma444Quant {
                neutral,
                qstep: split_qstep_c,
                lambda: leaf::part_lambda(split_qstep, self.tune.part_lambda_c),
                bit_depth: self.bit_depth as i32,
            },
            ChromaNeighbors { ua, ul, va, vl },
            mh_on,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_sb_whole64_444(
        &self,
        enc: &mut RangeEncoder,
        aqs: &mut aq::AqState,
        mut decide_mode: &mut crate::av2::replay::DecideMode<'_>,
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        above: &mut [u8],
        left: &mut [u8],
        above_pctx: &mut [u8],
        left_pctx: &mut [u8],
        midx_grid: &mut [u8],
        u_above: &mut [i32],
        v_above: &mut [i32],
        u_left: &mut [i32],
        v_left: &mut [i32],
        cfl_above: &mut [i32],
        cfl_left: &mut [i32],
        db_sb_qidx: &mut [u16],
        db_tx: &mut Vec<(usize, usize, usize, usize)>,
        db_tx_c: &mut Vec<(usize, usize, usize, usize)>,
        i444_inter_above: &mut [u8],
        i444_inter_left: &mut u8,
        i444_mv_above: &mut [Option<video::mv::Mv>],
        i444_mv_left: &mut Option<video::mv::Mv>,
        pw: usize,
        width: usize,
        height: usize,
        sb_y: usize,
        sb_x: usize,
        row: usize,
        col: usize,
        sb_cols: usize,
        tmc: i64,
        tmr: i64,
        ua: i32,
        ul: i32,
        va: i32,
        vl: i32,
        fmr: usize,
        fmc: usize,
        cell: aq::AqCell,
        core_has_last: bool,
    ) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let part_lambda_c = self.tune.part_lambda_c;
        let txpart = self.tune.txpart;
        let rdoq_lambda = self.tune.rdoq_lambda;
        let push_whole64 =
            |db: &mut Vec<(usize, usize, usize, usize)>, r: usize, c: usize, part: u8| {
                match part {
                    1 => {
                        for i in 0..4 {
                            db.push((r, c + i * 4, 4, 16));
                        }
                    } // VERT4: 4x TX_16X64
                    2 => {
                        for i in 0..4 {
                            db.push((r + i * 4, c, 16, 4));
                        }
                    } // HORZ4: 4x TX_64X16
                    _ => {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                db.push((r + qy * 8, c + qx * 8, 8, 8));
                            }
                        }
                    } // SPLIT
                }
            };
        // Staged threading: in replay mode pop this whole-64 SB's captured
        // winner (one push/pop per whole-64 SB, same order in decide & replay
        // since `choose_luma_64x64_partition` is deterministic). A `Whole64`
        // hit lets the emit run from the record; anything else re-searches.
        let replay_w: Option<Box<replay::Whole64Decision>> =
            if let replay::DecideMode::Replay(cur) = &mut decide_mode {
                match cur.next() {
                    replay::SbDecision::Whole64(d) => Some(d.clone()),
                    _ => None,
                }
            } else {
                None
            };
        // Fast path: whole 64X64 SB. RD-choose luma tx-partition between
        // SPLIT (4×TX_32X32) and VERT4 (4×TX_16X64), cheap SSE + rate proxy.
        // AQ: stills (the wavefront target) read the precomputed grid cell so a
        // parallel decide needs no serial `last_qidx` accumulator; video/inter keeps
        // the raster-serial `per_sb` (its accumulation interleaves with inter-skip,
        // which the grid does not model). Both set `delta_q_signaled` + qstep tuple
        // identically for stills — proven byte-identical by the gate + AqCell unit test.
        let (sb_qstep, sb_resid_scale, sb_qstep_c, sb_resid_scale_c) = if core_has_last {
            aqs.per_sb(enc, yp, pw, sb_y, sb_x, width, height)
        } else {
            enc.delta_q_signaled = cell.sig;
            (cell.qs, cell.resid_scale, cell.qs_c, cell.resid_scale_c)
        };
        db_sb_qidx[row * sb_cols + col] = if core_has_last {
            aqs.current_qidx() as u16
        } else {
            cell.qidx as u16
        };
        // In an inter tile every block signals intra_inter. AVM scans
        // neighbors in order {bottom-left, above-right, left, above} and
        // keeps the first two non-null; ctx = both-intra?3 : (either-intra).
        // For our whole-64 raster bottom-left is never coded yet (null).
        if enc.inter_tile {
            let up = row > 0;
            let lf = col > 0;
            let ia = i444_inter_above[col] == 1;
            let il = *i444_inter_left == 1;
            // For a whole-64 SB, AVM's first two non-null neighbors are the
            // left SB (bottom-left+left) and above SB (above-right+above); a
            // lone neighbor is scanned twice.
            enc.intra_inter_ctx = if up && lf {
                if !il && !ia { 3 } else { (!il || !ia) as usize }
            } else if lf {
                if il { 0 } else { 3 }
            } else if up {
                if ia { 0 } else { 3 }
            } else {
                0
            };
        }
        // do_split cdf for this whole-64 PARTITION_NONE, from the real partition
        // context (12276 in an all-whole-64 frame; differs next to a split SB).
        let none_do_split_cdf = partition::sb_none_do_split_cdf(row, col, above_pctx, left_pctx);
        let sse_region =
            |rec: &[f32]| -> f32 { pixel_sse_rounded_block(yp, pw, sb_y, sb_x, rec, pw, 64, 64) };
        let rate_proxy = coeff_tus_rate_proxy_f32;
        // Use THIS SB's qstep, not the frame base: dark AQ lowers sb_qstep, and
        // the base-q lambda would overprice rate and discourage the smaller /
        // coefficient-heavy partitions in exactly the protected dark blocks.
        let lambda = leaf::part_lambda(sb_qstep, part_lambda_c);
        // ---- SPLIT candidate (existing mode search) ----
        let (mut tus_s, mut mode_idx, mut adelta, split_bits) = outline_444(|| {
            if replay_w.is_some() {
                // Replay: the SPLIT mode search is skipped; the commit below is fed
                // the captured winner. Dummy values (overwritten by the inject).
                (
                    [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
                    0usize,
                    0i8,
                    0.0f32,
                )
            } else {
                encode_luma_sb(
                    recy,
                    &LumaSource {
                        plane: yp,
                        stride: pw,
                    },
                    &LumaFrameBlock {
                        frame_width: width,
                        frame_height: height,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaQuantSpec {
                        basis: &bases.luma,
                        qstep: sb_qstep,
                        scan: &tables::SCAN,
                        neutral,
                        quant_context: qc,
                        rdoq_lambda,
                        speed: self.speed,
                        bit_depth: self.bit_depth as i32,
                    },
                    &LumaSbSearch {
                        residual_scale: sb_resid_scale,
                        allow_directional: self.speed.try_directional()
                            && sb_x + 64 <= width
                            && sb_y + 64 <= height
                            && (width.is_multiple_of(64) || sb_x + 128 <= width)
                            && (height.is_multiple_of(64) || sb_y + 128 <= height),
                    },
                )
            }
        });
        let j_s = sse_region(recy) + lambda * split_bits;
        // partition strategy from tuning (was AV2_TXPART env)
        // Rect tx-partition (VERT4/HORZ4) is only safe on FULL interior 64x64
        // SBs: on a partial edge SB the rect strips cross the frame boundary
        // and the edge-clamped coding desyncs the decoder. Restrict rect
        // candidates to whole SBs; partial edge SBs fall back to SPLIT.
        let whole_sb = sb_x + 64 <= width && sb_y + 64 <= height;
        let want_vert4 = whole_sb
            && replay_w.is_none()
            && matches!(txpart, TxPart::ThreeWay | TxPart::Rd2 | TxPart::Vert4);
        let want_horz4 =
            whole_sb && replay_w.is_none() && matches!(txpart, TxPart::ThreeWay | TxPart::Horz4);
        let force_vert4 = txpart == TxPart::Vert4;
        let force_horz4 = txpart == TxPart::Horz4;
        let mut snap_split = [0f32; 64 * 64];
        let mut snap_best = [0f32; 64 * 64];
        for (dst_row, src_row) in snap_split
            .as_chunks_mut::<64>()
            .0
            .iter_mut()
            .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 64))
        {
            dst_row.copy_from_slice(src_row);
        }
        snap_best.copy_from_slice(&snap_split);
        let restore = |recy: &mut [f32], snap: &[f32]| {
            for (dst_row, src_row) in
                rect_rows_mut(recy, pw, sb_y, sb_x, 64, 64).zip(snap.as_chunks::<64>().0.iter())
            {
                dst_row.copy_from_slice(src_row);
            }
        };
        #[derive(PartialEq, Debug)]
        enum Part {
            Split,
            Vert4,
            Horz4,
        }
        let mut best = Part::Split;
        let mut best_j = j_s;
        let mut tus_v: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        let mut tus_h: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        // ---- VERT4 candidate (4× TX_16X64, strips L→R) ----
        outline_444(|| {
            if want_vert4 {
                for (i, tus_v) in tus_v[..4].iter_mut().enumerate() {
                    let x0 = sb_x + i * 16;
                    let predv =
                        dc_pred_rect(recy, pw, sb_y, x0, 16, 64, neutral, self.bit_depth as i32);
                    let lev = bases.luma16x64.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, x0, 16, 64, predv),
                            sb_resid_scale,
                        ),
                        0.0,
                        &SCAN16X32,
                    );
                    let pred_flat = [predv; 1024];
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        x0,
                        16,
                        64,
                        &itx422::reconstruct_luma_16x64(
                            &pred_flat,
                            &lev,
                            sb_qstep,
                            &SCAN16X32,
                            self.bit_depth as i32,
                        ),
                    );
                    *tus_v = levels_to_coeffs(&lev);
                }
                let j_v = sse_region(recy) * self.tune.rect_part_penalty
                    + lambda * rate_proxy(&tus_v, 4.0);
                let take = force_vert4 || j_v < best_j;
                if take {
                    best = Part::Vert4;
                    best_j = j_v;
                    snap_best.copy_from_slice(&{
                        let mut s = [0f32; 64 * 64];
                        for (dst_row, src_row) in s
                            .as_chunks_mut::<64>()
                            .0
                            .iter_mut()
                            .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 64))
                        {
                            dst_row.copy_from_slice(src_row);
                        }
                        s
                    });
                }
                restore(recy, &snap_split);
            }
        });
        // ---- HORZ4 candidate (4× TX_64X16, strips T→B) ----
        outline_444(|| {
            if want_horz4 {
                for (i, tus_h) in tus_h[..4].iter_mut().enumerate() {
                    let y0 = sb_y + i * 16;
                    let predh =
                        dc_pred_rect(recy, pw, y0, sb_x, 64, 16, neutral, self.bit_depth as i32);
                    let lev = bases.luma64x16.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, y0, sb_x, 64, 16, predh),
                            sb_resid_scale,
                        ),
                        0.0,
                        &SCAN32X16,
                    );
                    let pred_flat = [predh; 1024];
                    put_block_rect(
                        recy,
                        pw,
                        y0,
                        sb_x,
                        64,
                        16,
                        &itx422::reconstruct_luma_64x16(
                            &pred_flat,
                            &lev,
                            sb_qstep,
                            &SCAN32X16,
                            self.bit_depth as i32,
                        ),
                    );
                    *tus_h = levels_to_coeffs(&lev);
                }
                let j_h = sse_region(recy) * self.tune.rect_part_penalty
                    + lambda * rate_proxy(&tus_h, 4.0);
                let take = force_horz4 || j_h < best_j;
                if take {
                    best = Part::Horz4;
                    // best_j no longer read past the last candidate.
                    for (dst_row, src_row) in snap_best
                        .as_chunks_mut::<64>()
                        .0
                        .iter_mut()
                        .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 64))
                    {
                        dst_row.copy_from_slice(src_row);
                    }
                }
                restore(recy, &snap_split);
            }
        });
        // ---- commit winner ----
        restore(recy, &snap_best);
        // CfL decision (4:4:4 whole-64 fast path). recy is final here; recu/
        // recv hold the neighbor reconstructions for the DC prediction. The
        // is_cfl context comes from CfL-usage neighbors. This sets the
        // per-block CfL state consumed by encode_intra_modes during the
        // luma-block encode below (which emits is_cfl + alphas).
        let cfl_a = if fmr > 0 { cfl_above[fmc] } else { 0 };
        let cfl_l = if fmc > 0 { cfl_left[fmr] } else { 0 };
        enc.cfl_ctx = (cfl_a + cfl_l) as usize;
        let mut cfl_choice = outline_444(|| {
            if enc.cfl && replay_w.is_none() {
                cfl_decide_64(
                    &Cfl64Input {
                        reconstructed_luma: recy,
                        source_u: up,
                        source_v: vp,
                        reconstructed_u: recu,
                        reconstructed_v: recv,
                        stride: pw,
                        y: sb_y,
                        x: sb_x,
                        neutral,
                    },
                    &ChromaRdSpec {
                        basis: &bases.chroma444,
                        qstep: sb_qstep_c,
                        lambda,
                        bit_depth: self.bit_depth as i32,
                    },
                )
            } else {
                None
            }
        });
        if let Some(ref ch) = cfl_choice {
            enc.cfl_use = true;
            enc.cfl_js = ch.js;
            enc.cfl_mag_u = ch.mag_u;
            enc.cfl_mag_v = ch.mag_v;
            enc.cfl_ctx_u = ch.ctx_u;
            enc.cfl_ctx_v = ch.ctx_v;
        } else {
            enc.cfl_use = false;
            enc.cfl_signaled = false;
        }
        // Chroma intra mode search MUST run before the luma-block encode
        // below, because that encode emits the uv_mode symbol. Decide the
        // winning predictor now (when not CfL), set enc.uv_mode, and reuse
        // the predictor when coding the chroma residual further down.
        let mut uv444_pred: Option<(Vec<f32>, Vec<f32>)> = outline_444(|| {
            if cfl_choice.is_none() && self.tune.chroma_mode_search && replay_w.is_none() {
                let cand_modes: &[usize] = if self.speed.reduced_modes() {
                    &[0, 1, 4, 5, 6]
                } else if self.speed.chroma_angle_directional() {
                    &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                } else {
                    &[0, 1, 2, 3, 4, 5, 6]
                };
                let mode_lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
                let dcu = dc_pred(recu, pw, sb_y, sb_x, 64, neutral);
                let dcv = dc_pred(recv, pw, sb_y, sb_x, 64, neutral);
                let mut best_mode = 0usize;
                let mut best_cost = f32::INFINITY;
                let mut best_pred: Option<(Vec<f32>, Vec<f32>)> = None;
                // Chroma internal (5..) -> luma internal for the same
                // direction; a uv mode equal to a directional luma with a
                // non-zero angle delta cannot be signaled with the nominal
                // angle (the "same as luma" list slot copies the delta).
                let luma_dir_as_uv: usize = match mode_idx {
                    5 => 5,   // V
                    6 => 6,   // H
                    7 => 7,   // D45
                    8 => 8,   // D135
                    9 => 10,  // D113
                    10 => 11, // D157
                    11 => 12, // D203
                    12 => 9,  // D67
                    _ => usize::MAX,
                };
                let predict_uv = |m: usize| -> (Vec<f32>, Vec<f32>) {
                    if m == 0 {
                        (vec![dcu; 64 * 64], vec![dcv; 64 * 64])
                    } else {
                        (
                            chroma422::predict_chroma444_whole64(
                                recu, pw, sb_y, sb_x, m, neutral, width, height,
                            ),
                            chroma422::predict_chroma444_whole64(
                                recv, pw, sb_y, sb_x, m, neutral, width, height,
                            ),
                        )
                    }
                };
                // SATD prune: each chroma mode is one 64x64 prediction from
                // neighbours (independent), so rank by SATD(pred, source) over
                // U+V and full-encode only the top-K.
                let keep_uv = if self.speed.reduced_modes() {
                    2
                } else if self.speed.chroma_angle_directional() {
                    4
                } else {
                    3
                };
                let cand_modes: Vec<usize> = {
                    let valid: Vec<usize> = cand_modes
                        .iter()
                        .copied()
                        .filter(|&m| !(m == luma_dir_as_uv && adelta != 0))
                        .collect();
                    if valid.len() > keep_uv {
                        let mut r: Vec<(u64, usize)> = valid
                            .iter()
                            .map(|&m| {
                                let (pu, pv) = predict_uv(m);
                                (
                                    crate::av2::metrics::satd_f32(
                                        &up[sb_y * pw + sb_x..],
                                        pw,
                                        &pu,
                                        64,
                                        64,
                                        64,
                                    ) + crate::av2::metrics::satd_f32(
                                        &vp[sb_y * pw + sb_x..],
                                        pw,
                                        &pv,
                                        64,
                                        64,
                                        64,
                                    ),
                                    m,
                                )
                            })
                            .collect();
                        r.sort_by_key(|&(s, _)| s);
                        r.truncate(keep_uv);
                        r.into_iter().map(|(_, m)| m).collect()
                    } else {
                        valid
                    }
                };
                for &m in &cand_modes {
                    let (pu, pv) = predict_uv(m);
                    let mut pu_i = vec![0i32; 64 * 64];
                    let mut pv_i = vec![0i32; 64 * 64];
                    metrics::prediction_f32_to_i32(&mut pu_i, &pu, 64, 64, 64);
                    metrics::prediction_f32_to_i32(&mut pv_i, &pv, 64, 64, 64);
                    let mut ru = [0f32; 64 * 64];
                    let mut rv = [0f32; 64 * 64];
                    let residual_spec = metrics::ResidualSpec {
                        src_stride: pw,
                        pred_stride: 64,
                        width: 64,
                        height: 64,
                        scale: sb_resid_scale_c,
                    };
                    metrics::scaled_residual_f32(
                        &mut ru,
                        &up[sb_y * pw + sb_x..],
                        &pu,
                        residual_spec,
                    );
                    metrics::scaled_residual_f32(
                        &mut rv,
                        &vp[sb_y * pw + sb_x..],
                        &pv,
                        residual_spec,
                    );
                    let lu = chroma422::project_chroma_rdoq(
                        &bases.chroma444,
                        &ru,
                        &tables::SCAN,
                        qc,
                        1024,
                        0,
                        self.tune.chroma_rdoq_lambda,
                    );
                    let lv = chroma422::project_chroma_rdoq(
                        &bases.chroma444,
                        &rv,
                        &tables::SCAN,
                        qc,
                        1024,
                        4,
                        self.tune.chroma_rdoq_lambda,
                    );
                    let recu_b = itx422::reconstruct_chroma_cfl(
                        &pu_i,
                        &lu,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        self.bit_depth as i32,
                    );
                    let recv_b = itx422::reconstruct_chroma_cfl(
                        &pv_i,
                        &lv,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        self.bit_depth as i32,
                    );
                    let sse = pixel_sse_rounded_block(up, pw, sb_y, sb_x, &recu_b, 64, 64, 64)
                        + pixel_sse_rounded_block(vp, pw, sb_y, sb_x, &recv_b, 64, 64, 64);
                    let rate = coeff_abs_rate_f32(&lu) + coeff_abs_rate_f32(&lv);
                    let mode_bits = if m == 0 { 0.0 } else { 2.0 };
                    let cost = sse + mode_lambda * (rate + mode_bits);
                    if cost < best_cost {
                        best_cost = cost;
                        best_mode = m;
                        best_pred = if m == 0 { None } else { Some((pu, pv)) };
                    }
                }
                enc.uv_mode = best_mode;
                best_pred
            } else {
                None
            }
        });
        // Staged threading (replay): overwrite the just-searched winner with
        // the captured decision so the commit/emit below runs entirely from the
        // record. (3a: search still ran; injecting cached values here proves the
        // record faithfully reproduces the bitstream. 3b will skip the search.)
        if let Some(w) = replay_w.as_ref() {
            use crate::av2::replay::{WholeChroma, WholePart};
            best = match w.part {
                WholePart::Split => Part::Split,
                WholePart::Vert4 => Part::Vert4,
                WholePart::Horz4 => Part::Horz4,
            };
            match w.part {
                WholePart::Split => tus_s = w.tus.clone(),
                WholePart::Vert4 => tus_v = w.tus.clone(),
                WholePart::Horz4 => tus_h = w.tus.clone(),
            }
            mode_idx = w.mode_idx;
            adelta = w.angle_delta;
            match &w.chroma {
                WholeChroma::Cfl(ch) => {
                    cfl_choice = Some(ch.clone());
                    uv444_pred = None;
                    enc.cfl_use = true;
                    enc.cfl_js = ch.js;
                    enc.cfl_mag_u = ch.mag_u;
                    enc.cfl_mag_v = ch.mag_v;
                    enc.cfl_ctx_u = ch.ctx_u;
                    enc.cfl_ctx_v = ch.ctx_v;
                }
                WholeChroma::Mode(m) => {
                    cfl_choice = None;
                    enc.cfl_use = false;
                    enc.cfl_signaled = false;
                    enc.uv_mode = *m;
                    uv444_pred = if *m == 0 {
                        None
                    } else {
                        Some((
                            chroma422::predict_chroma444_whole64(
                                recu, pw, sb_y, sb_x, *m, neutral, width, height,
                            ),
                            chroma422::predict_chroma444_whole64(
                                recv, pw, sb_y, sb_x, *m, neutral, width, height,
                            ),
                        ))
                    };
                }
            }
            // Restore the winner's luma recon (the commit emits coeffs but does
            // not rewrite recy); chroma recon is reproduced by the commit half.
            for r in 0..64 {
                let off = (sb_y + r) * pw + sb_x;
                recy[off..off + 64].copy_from_slice(&w.recon_y[r * 64..r * 64 + 64]);
            }
        }
        enc.delta_q_pending = enc.delta_q_present;
        // Directional-mode neighbor context for this SB (decoder get_y_mode_idx_ctx):
        // number of directional left/above neighbors. Used by every luma-mode emit arm.
        let lmidx = if col > 0 {
            midx_grid[row * sb_cols + col - 1]
        } else {
            0xff
        };
        let amidx = if row > 0 {
            midx_grid[(row - 1) * sb_cols + col]
        } else {
            0xff
        };
        let sb_y_ctx = (lmidx != 0xff) as usize + (amidx != 0xff) as usize;
        push_whole64(
            db_tx,
            row * 16,
            col * 16,
            match best {
                Part::Vert4 => 1,
                Part::Horz4 => 2,
                _ => 0,
            },
        );
        // Whole-64 chroma is always a single TX_64X64 (no internal TU edges),
        // regardless of the luma SPLIT/VERT4/HORZ4 tiling.
        db_tx_c.push((row * 16, col * 16, 16, 16));
        outline_444(|| match best {
            Part::Vert4 => {
                let mut skip_cdfs = [0u32; 4];
                let mut dc_sign_ctxs = [0usize; 4];
                for i in 0..4 {
                    let (s, d) = sb_tu_contexts_rect(
                        &tus_v[i],
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y,
                            sb_x: sb_x + i * 16,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: false,
                        },
                        4,
                        16,
                    );
                    skip_cdfs[i] = s;
                    dc_sign_ctxs[i] = d;
                }
                encode_luma_block_vert4(
                    enc,
                    &tus_v,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    0,
                    true,
                    none_do_split_cdf,
                    sb_y_ctx,
                );
            }
            Part::Horz4 => {
                let mut skip_cdfs = [0u32; 4];
                let mut dc_sign_ctxs = [0usize; 4];
                for i in 0..4 {
                    let (s, d) = sb_tu_contexts_rect(
                        &tus_h[i],
                        above,
                        left,
                        &TxbContextSpec {
                            sb_y: sb_y + i * 16,
                            sb_x,
                            qc,
                            mi_cols: tmc,
                            mi_rows: tmr,
                            block_eq_tx: false,
                        },
                        16,
                        4,
                    );
                    skip_cdfs[i] = s;
                    dc_sign_ctxs[i] = d;
                }
                encode_luma_block_horz4(
                    enc,
                    &tus_h,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    0,
                    true,
                    none_do_split_cdf,
                    sb_y_ctx,
                );
            }
            Part::Split => {
                let (skip_cdfs, dc_sign_ctxs) =
                    sb_tu_contexts(&tus_s, sb_y, sb_x, above, left, qc, tmc, tmr);
                let (_cul, sb_midx) = encode_luma_block_split_dir(
                    enc,
                    &LumaSplitDirSpec {
                        tus: &tus_s,
                        skip_cdfs: &skip_cdfs,
                        dc_sign_ctxs: &dc_sign_ctxs,
                        mode_idx,
                        angle_delta: adelta,
                        has_chroma: true,
                        part_cdf: none_do_split_cdf,
                        left_midx: lmidx,
                        above_midx: amidx,
                    },
                );
                midx_grid[row * sb_cols + col] = sb_midx;
            }
        });
        let bd = self.bit_depth as i32;
        let (levu, levv) = outline_444(|| {
            if let Some(ref ch) = cfl_choice {
                // CfL: residual against the per-pixel prediction; reconstruct
                // with that prediction as the base.
                let mut ru = [0f32; 64 * 64];
                let mut rv = [0f32; 64 * 64];
                cfl_prediction::<64>(pw, up, vp, sb_y, sb_x, &ch, &mut ru, &mut rv);
                let levu = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &aq::scale_resid(&ru, sb_resid_scale_c),
                    &tables::SCAN,
                    qc,
                    1024,
                    0,
                    self.tune.chroma_rdoq_lambda,
                );
                let levv = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &aq::scale_resid(&rv, sb_resid_scale_c),
                    &tables::SCAN,
                    qc,
                    1024,
                    4,
                    self.tune.chroma_rdoq_lambda,
                );
                put_block(
                    recu,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma_cfl(
                        &ch.pred_u,
                        &levu,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                put_block(
                    recv,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma_cfl(
                        &ch.pred_v,
                        &levv,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                (levu, levv)
            } else if let Some((pu, pv)) = uv444_pred.as_ref() {
                // Non-DC chroma intra mode chosen above (enc.uv_mode already
                // set and emitted by the luma-block encode). Code the residual
                // against the per-pixel predictor, reconstruct with it as base.
                let mut pu_i = vec![0i32; 64 * 64];
                let mut pv_i = vec![0i32; 64 * 64];
                metrics::prediction_f32_to_i32(&mut pu_i, pu, 64, 64, 64);
                metrics::prediction_f32_to_i32(&mut pv_i, pv, 64, 64, 64);
                let mut ru = [0f32; 64 * 64];
                let mut rv = [0f32; 64 * 64];
                let residual_spec = metrics::ResidualSpec {
                    src_stride: pw,
                    pred_stride: 64,
                    width: 64,
                    height: 64,
                    scale: sb_resid_scale_c,
                };
                metrics::scaled_residual_f32(&mut ru, &up[sb_y * pw + sb_x..], pu, residual_spec);
                metrics::scaled_residual_f32(&mut rv, &vp[sb_y * pw + sb_x..], pv, residual_spec);
                let levu = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &ru,
                    &tables::SCAN,
                    qc,
                    1024,
                    0,
                    self.tune.chroma_rdoq_lambda,
                );
                let levv = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &rv,
                    &tables::SCAN,
                    qc,
                    1024,
                    4,
                    self.tune.chroma_rdoq_lambda,
                );
                put_block(
                    recu,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma_cfl(
                        &pu_i,
                        &levu,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                put_block(
                    recv,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma_cfl(
                        &pv_i,
                        &levv,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                (levu, levv)
            } else {
                // DC (either search disabled, or DC won the search above).
                let predu = dc_pred(recu, pw, sb_y, sb_x, 64, neutral);
                let levu = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &aq::scale_resid(
                        &get_residual(up, pw, sb_y, sb_x, 64, predu),
                        sb_resid_scale_c,
                    ),
                    &tables::SCAN,
                    qc,
                    1024,
                    0,
                    self.tune.chroma_rdoq_lambda,
                );
                put_block(
                    recu,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma(
                        predu,
                        &levu,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                let predv = dc_pred(recv, pw, sb_y, sb_x, 64, neutral);
                let levv = chroma422::project_chroma_rdoq(
                    &bases.chroma444,
                    &aq::scale_resid(
                        &get_residual(vp, pw, sb_y, sb_x, 64, predv),
                        sb_resid_scale_c,
                    ),
                    &tables::SCAN,
                    qc,
                    1024,
                    4,
                    self.tune.chroma_rdoq_lambda,
                );
                put_block(
                    recv,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &itx422::reconstruct_chroma(
                        predv,
                        &levv,
                        sb_qstep_c,
                        &tables::SCAN,
                        64,
                        64,
                        bd,
                    ),
                );
                (levu, levv)
            }
        });
        let ucoeffs = levels_to_coeffs(&levu);
        let vcoeffs = levels_to_coeffs(&levv);
        let u_skip = (6 + ua + ul) as u32;
        encode_chroma_block(enc, &ucoeffs, u_skip, true);
        let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
        let v_skip = (6 * (u_present as i32) + va + vl) as u32;
        encode_chroma_block(enc, &vcoeffs, v_skip, false);
        let v_present = vcoeffs.iter().any(|&(_, l)| l != 0);
        let cfl_used = cfl_choice.is_some() as i32;
        for c in fmc..fmc + 16 {
            u_above[c] = u_present as i32;
            v_above[c] = v_present as i32;
            cfl_above[c] = cfl_used;
        }
        for r in fmr..fmr + 16 {
            u_left[r] = u_present as i32;
            v_left[r] = v_present as i32;
            cfl_left[r] = cfl_used;
        }
        // Maintain partition contexts for this whole-64 PARTITION_NONE so that
        // SBs neighboring a chroma-motivated split observe correct contexts.
        partition::sb_none_pctx(row, col, above_pctx, left_pctx);
        // Fast-path intra SB: clear neighbor inter flags for this column.
        i444_inter_above[col] = 0;
        *i444_inter_left = 0;
        i444_mv_above[col] = None;
        *i444_mv_left = None;
        // Staged threading: log this whole-64 winner so an emit-only
        // replay pass can skip the search half. One push per fast-path
        // SB (visited identically in decide/replay), so the call-order
        // cursor stays aligned without touching the other paths.
        if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
            use crate::av2::replay::{SbDecision, Whole64Decision, WholeChroma, WholePart};
            let (part, tus) = match best {
                Part::Split => (WholePart::Split, tus_s.clone()),
                Part::Vert4 => (WholePart::Vert4, tus_v.clone()),
                Part::Horz4 => (WholePart::Horz4, tus_h.clone()),
            };
            let chroma = match cfl_choice {
                Some(ref ch) => WholeChroma::Cfl(ch.clone()),
                None => WholeChroma::Mode(enc.uv_mode),
            };
            // Grab the winner's 64×64 recon (already final in recy/recu/recv)
            // so replay can put_block it instead of reconstructing.
            let grab = |plane: &[f32]| -> Vec<f32> {
                let mut b = Vec::with_capacity(64 * 64);
                for r in 0..64 {
                    let off = (sb_y + r) * pw + sb_x;
                    b.extend_from_slice(&plane[off..off + 64]);
                }
                b
            };
            rec.push(SbDecision::Whole64(Box::new(Whole64Decision {
                part,
                tus,
                mode_idx,
                angle_delta: adelta,
                chroma,
                recon_y: grab(recy),
                recon_u: grab(recu),
                recon_v: grab(recv),
            })));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decide_sb(
        &self,
        enc: &mut RangeEncoder,
        aqs: &mut aq::AqState,
        mut decide_mode: &mut crate::av2::replay::DecideMode<'_>,
        y: &[f32],
        cb: &[f32],
        cr: &[f32],
        yp: &[f32],
        up: &[f32],
        vp: &[f32],
        recy: &mut [f32],
        recu: &mut [f32],
        recv: &mut [f32],
        above: &mut [u8],
        left: &mut [u8],
        above_pctx: &mut [u8],
        left_pctx: &mut [u8],
        midx_grid: &mut [u8],
        u_above: &mut [i32],
        v_above: &mut [i32],
        u_left: &mut [i32],
        v_left: &mut [i32],
        cfl_above: &mut [i32],
        cfl_left: &mut [i32],
        db_sb_qidx: &mut [u16],
        db_tx: &mut Vec<(usize, usize, usize, usize)>,
        db_tx_c: &mut Vec<(usize, usize, usize, usize)>,
        i444_skip_above: &mut [u8],
        i444_inter_above: &mut [u8],
        i444_newmv_above: &mut [u8],
        i444_mv_above: &mut [Option<video::mv::Mv>],
        i444_skip_left: &mut u8,
        i444_inter_left: &mut u8,
        i444_newmv_left: &mut u8,
        i444_mv_left: &mut Option<video::mv::Mv>,
        core_last_ref: &[Vec<f32>],
        inter_luma: &Option<(Vec<u16>, Vec<u16>, usize, usize)>,
        inter_chroma: &Option<[(Vec<u16>, usize); 2]>,
        me_scratch: &mut video::me::MeScratch<u16>,
        mhccp_bounds: cfl::MhccpBounds,
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        sb_cols: usize,
        tmc: i64,
        tmr: i64,
        qstep_i: i32,
        ref_x0: usize,
        ref_y0: usize,
        ref_ls: usize,
        core_has_last: bool,
        frame_mv_seed: video::mv::Mv,
        needs_partition: bool,
        aq_grid: &[aq::AqCell],
        row: usize,
        col: usize,
    ) {
        let bases = &self.bases;
        let neutral = self.dc_neutral();
        let qc = enc.qc;
        let rdoq_lambda = self.tune.rdoq_lambda;
        // Per-SB AQ decision from the serial pre-pass grid (used for stills; video
        // still accumulates via `aqs`). `qs_before` = accumulator entering this SB
        // (the partition probe's `current()`); `qs`/`sig`/`qidx` = the committed SB.
        let cell = aq_grid[row * sb_cols + col];
        const BRD: usize = 72;
        let push_whole64 =
            |db: &mut Vec<(usize, usize, usize, usize)>, r: usize, c: usize, part: u8| {
                match part {
                    1 => {
                        for i in 0..4 {
                            db.push((r, c + i * 4, 4, 16));
                        }
                    } // VERT4: 4× TX_16X64
                    2 => {
                        for i in 0..4 {
                            db.push((r + i * 4, c, 16, 4));
                        }
                    } // HORZ4: 4× TX_64X16
                    _ => {
                        for qy in 0..2 {
                            for qx in 0..2 {
                                db.push((r + qy * 8, c + qx * 8, 8, 8));
                            }
                        }
                    } // SPLIT
                }
            };
        let sb_y = row * 64;
        let sb_x = col * 64;
        // Arm the per-SB CDEF index emission (consumed once, at this SB's
        // first coded block). Inert unless pass 2 installed a grid.
        if enc.cdef_nb >= 2 {
            enc.cdef_pending = true;
            enc.cdef_sb_rc = (row, col);
        }
        let mut i444_intra_this_sb = false;
        // --- 4:4:4 inter (NEWMV) attempt: whole-64 LAST-ref block ---------
        let inter_committed = outline_444(|| {
            if core_has_last && sb_x + 64 <= width && sb_y + 64 <= height {
                use crate::av2::video::{me, mv::Mv};
                let bd = self.bit_depth as i32;
                let (ly, lu, lv) = (&core_last_ref[0], &core_last_ref[1], &core_last_ref[2]);
                let (nmv_qstep, nmv_scale, nmv_qstep_c, nmv_scale_c) =
                    aqs.per_sb_probe(y, pw, sb_y, sb_x, width, height);
                // GLOBALMV zero-motion skip: a static full 64x64 block copies LAST
                // (all three planes at full res) instead of falling back to intra.
                // RD bound scales q^2 (AVM rdmult); mirrors the 4:2:0 skip branch.
                {
                    let sse = rect_sse_f32(
                        &PlaneRect {
                            plane: y,
                            stride: pw,
                            y: sb_y,
                            x: sb_x,
                        },
                        &PlaneRect {
                            plane: ly,
                            stride: ref_ls,
                            y: ref_y0 + sb_y,
                            x: ref_x0 + sb_x,
                        },
                        64,
                        64,
                    );
                    use crate::av2::video::rd;
                    let skip_dist = sse * rd::SS2_INTER_DIST_W;
                    let intra_bound = rd::rd_cost(0.0, 2.0 * 64.0 * 64.0, qstep_i as u32);
                    if skip_dist < intra_bound {
                        let up = row > 0;
                        let lf = col > 0;
                        let ia = i444_inter_above[col] == 1;
                        let il = (*i444_inter_left) == 1;
                        enc.intra_inter_ctx = if up && lf {
                            let n_intra = (!il as u8) + (!ia as u8);
                            if n_intra == 2 { 3 } else { n_intra as usize }
                        } else if up {
                            if ia { 0 } else { 3 }
                        } else if lf {
                            if il { 0 } else { 3 }
                        } else {
                            0
                        };
                        let sa = i444_skip_above[col];
                        let sl = *i444_skip_left;
                        let skip_ctx = if up && lf {
                            (sl + sa) as usize
                        } else if up {
                            (2 * sa) as usize
                        } else if lf {
                            (2 * sl) as usize
                        } else {
                            0
                        };
                        let mode_ctx = (i444_inter_above[col] + (*i444_inter_left)) as usize
                            + if i444_newmv_above[col] != 0 || (*i444_newmv_left) != 0 {
                                2
                            } else {
                                0
                            };
                        coder::emit_inter_skip_block(enc, 12276, skip_ctx, mode_ctx);
                        // Copy LAST into recon: Y, U, V all full-res, same MV=0.
                        for (rec, refp) in [
                            (&mut *recy, &core_last_ref[0]),
                            (&mut *recu, &core_last_ref[1]),
                            (&mut *recv, &core_last_ref[2]),
                        ] {
                            for (dst_row, src_row) in rect_rows_mut(rec, pw, sb_y, sb_x, 64, 64)
                                .zip(rect_rows(
                                    refp,
                                    ref_ls,
                                    ref_y0 + sb_y,
                                    ref_x0 + sb_x,
                                    64,
                                    64,
                                ))
                            {
                                dst_row.copy_from_slice(src_row);
                            }
                        }
                        // Update neighbor/entropy grids identically to the NEWMV
                        // skip case (skip_txfm=1: no residual, cleared contexts).
                        let (fmr, fmc) = (row * 16, col * 16);
                        for c in fmc..fmc + 16 {
                            u_above[c] = 0;
                            v_above[c] = 0;
                            cfl_above[c] = 0;
                        }
                        for r in fmr..fmr + 16 {
                            u_left[r] = 0;
                            v_left[r] = 0;
                            cfl_left[r] = 0;
                        }
                        partition::sb_none_pctx(row, col, above_pctx, left_pctx);
                        // Clear this SB's luma coeff entropy context (0x40 marker).
                        {
                            let cx = sb_x / 4;
                            let cy = sb_y / 4;
                            let ae = (cx + 16).min(above.len());
                            let le = (cy + 16).min(left.len());
                            for v in above[cx..ae].iter_mut() {
                                *v = 0x40;
                            }
                            for v in left[cy..le].iter_mut() {
                                *v = 0x40;
                            }
                        }
                        i444_skip_above[col] = 1;
                        *i444_skip_left = 1;
                        i444_inter_above[col] = 1;
                        *i444_inter_left = 1;
                        i444_newmv_above[col] = 0;
                        *i444_newmv_left = 0;
                        i444_mv_above[col] = Some(Mv::ZERO);
                        *i444_mv_left = Some(Mv::ZERO);
                        return true;
                    }
                }
                // Frame-scoped buffers built once before the loop; borrow them.
                let (cur_u, bref, bstride, refh) = {
                    let (c, b, s, h) = inter_luma.as_ref().unwrap();
                    (c.as_slice(), b.as_slice(), *s, *h)
                };
                // Spatial MV predictors (zero, left, above, above-right) seed the
                // search; ref_mv matches the DRL[0] predictor used for MVD coding.
                let mut preds = me::MeCandidates::new();
                if frame_mv_seed != Mv::ZERO {
                    preds.push_unique(frame_mv_seed);
                }
                for cand in [
                    (*i444_mv_left),
                    if row > 0 { i444_mv_above[col] } else { None },
                    if row > 0 && col + 1 < sb_cols {
                        i444_mv_above[col + 1]
                    } else {
                        None
                    },
                ]
                .into_iter()
                .flatten()
                {
                    preds.push_unique(cand);
                }
                let ref_mv = (*i444_mv_left)
                    .or(if row > 0 { i444_mv_above[col] } else { None })
                    .unwrap_or(Mv::ZERO);
                let (mv, _) = me::search(
                    &me::MePlanes {
                        current: &cur_u[sb_y * pw + sb_x..],
                        current_stride: pw,
                        reference: bref,
                        reference_stride: bstride,
                    },
                    preds.as_slice(),
                    &me::MeSearchSpec {
                        origin_x: (ref_x0 + sb_x + BRD) as isize,
                        origin_y: (ref_y0 + sb_y + BRD) as isize,
                        width: 64,
                        height: 64,
                        reference_mv: ref_mv,
                        lambda_mv: (nmv_qstep as u32).max(1),
                        max_dx: self.video_search_range,
                        max_dy: self.video_search_range,
                        predictor_gate_sad_per_pixel: self.video_predictor_gate,
                        integer_satd_radius: self.video_integer_satd_radius,
                        bit_depth: self.bit_depth,
                        frame_width: bstride,
                        frame_height: refh + 2 * BRD,
                    },
                    me_scratch,
                );
                let mv = mc::clamp_umv(
                    mv,
                    (ref_x0 + sb_x) as i32,
                    (ref_y0 + sb_y) as i32,
                    64,
                    64,
                    ref_ls as i32,
                    refh as i32,
                );
                // Accept zero-motion (static-block skip) and drop the fixed
                // magnitude clamp; motion found by the search is used directly.
                {
                    let mut pred_u = [0u16; 64 * 64];
                    mc::predict(
                        &mut pred_u,
                        64,
                        bref,
                        bstride,
                        &mc::MotionBlock {
                            origin_x: (ref_x0 + sb_x + BRD) as isize,
                            origin_y: (ref_y0 + sb_y + BRD) as isize,
                            mv,
                            width: 64,
                            height: 64,
                            bit_depth: self.bit_depth,
                        },
                    );
                    let sse = pixel_sse_f32_u16_block(y, pw, sb_y, sb_x, &pred_u, 64, 64, 64);
                    // Inter vs intra RD bound scales q^2 (AVM rdmult); accepts
                    // zero-motion static blocks now that the MV gate is removed.
                    use crate::av2::video::rd;
                    let inter_dist = sse * rd::SS2_INTER_DIST_W;
                    let intra_bound = rd::rd_cost(0.0, 16.0 * 64.0 * 64.0, nmv_qstep as u32);
                    if inter_dist < intra_bound {
                        static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
                        let mut tus: [Vec<Coeff>; 4] =
                            [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                        for (i, &(ty, tx)) in POS.iter().enumerate() {
                            let (y0, x0) = (sb_y + ty, sb_x + tx);
                            let mut pblk = [0f32; 1024];
                            let mut resid = [0f32; 1024];
                            metrics::u16_prediction_and_scaled_residual_f32(
                                &mut pblk,
                                &mut resid,
                                &y[y0 * pw + x0..],
                                &pred_u[ty * 64 + tx..],
                                metrics::ResidualSpec {
                                    src_stride: pw,
                                    pred_stride: 64,
                                    width: 32,
                                    height: 32,
                                    scale: nmv_scale,
                                },
                            );
                            let lev = bases.luma.project(&resid[..], 0.0);
                            let rb = reconstruct_luma(&pblk, &lev, nmv_qstep, &tables::SCAN, bd);
                            put_block(recy, pw, y0, x0, 32, &rb);
                            tus[i] = levels_to_coeffs(&lev);
                        }
                        let mut uv_coeffs: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
                        for (pi, (src_c, ref_c, rec_c)) in
                            [(cb, lu, &mut *recu), (cr, lv, &mut *recv)]
                                .into_iter()
                                .enumerate()
                        {
                            let _ = ref_c; // reference now borrowed frame-scoped
                            let (bcref, bcstride) = {
                                let (b, s) = &inter_chroma.as_ref().unwrap()[pi];
                                (b.as_slice(), *s)
                            };
                            let mut cpredu = [0u16; 64 * 64];
                            mc::predict(
                                &mut cpredu,
                                64,
                                bcref,
                                bcstride,
                                &mc::MotionBlock {
                                    origin_x: (ref_x0 + sb_x + BRD) as isize,
                                    origin_y: (ref_y0 + sb_y + BRD) as isize,
                                    mv,
                                    width: 64,
                                    height: 64,
                                    bit_depth: self.bit_depth,
                                },
                            );
                            let mut cres = [0f32; 64 * 64];
                            let mut cpred = [0f32; 64 * 64];
                            metrics::u16_prediction_and_scaled_residual_f32(
                                &mut cpred,
                                &mut cres,
                                &src_c[sb_y * pw + sb_x..],
                                &cpredu,
                                metrics::ResidualSpec {
                                    src_stride: pw,
                                    pred_stride: 64,
                                    width: 64,
                                    height: 64,
                                    scale: nmv_scale_c,
                                },
                            );
                            let lev = chroma422::project_chroma_rdoq(
                                &bases.chroma444,
                                &cres[..],
                                &tables::SCAN,
                                qc,
                                1024,
                                pi * 4,
                                self.tune.chroma_rdoq_lambda,
                            );
                            let cpred_i: Vec<i32> = cpred.iter().map(|&p| p as i32).collect();
                            let rb = itx422::reconstruct_chroma_pred(
                                &cpred_i,
                                &lev,
                                nmv_qstep_c,
                                &tables::SCAN,
                                64,
                                64,
                                bd,
                            );
                            put_block(rec_c, pw, sb_y, sb_x, 64, &rb);
                            uv_coeffs[pi] = levels_to_coeffs(&lev);
                        }
                        let up = row > 0;
                        let lf = col > 0;
                        let skip_ctx = if up && lf {
                            (i444_skip_above[col] + (*i444_skip_left)) as usize
                        } else if up {
                            (2 * i444_skip_above[col]) as usize
                        } else if lf {
                            (2 * (*i444_skip_left)) as usize
                        } else {
                            0
                        };
                        let mode_ctx = (i444_inter_above[col] + (*i444_inter_left)) as usize
                            + if i444_newmv_above[col] != 0 || (*i444_newmv_left) != 0 {
                                2
                            } else {
                                0
                            };
                        // DRL[0] predictor (AVM setup_ref_mv_list scan order with
                        // reorder disabled, single LAST ref, no TMVP): left SB's MV
                        // first, then above, then above-right; else zero.
                        let pred_mv = (*i444_mv_left)
                            .or(if row > 0 { i444_mv_above[col] } else { None })
                            .or(if row > 0 && col + 1 < sb_cols {
                                i444_mv_above[col + 1]
                            } else {
                                None
                            })
                            .unwrap_or(crate::av2::video::mv::Mv::ZERO);
                        let mvd_row = mv.row - pred_mv.row;
                        let mvd_col = mv.col - pred_mv.col;
                        // mv == DRL[0]: signal NEARMV (copies the predictor, no MVD);
                        // otherwise NEWMV with mvd = mv - DRL[0].
                        let inter_mode = if mvd_row == 0 && mvd_col == 0 { 0 } else { 2 };
                        {
                            enc.delta_q_pending = enc.delta_q_present;
                            let any_resid = tus.iter().any(|t| !t.is_empty())
                                || !uv_coeffs[0].is_empty()
                                || !uv_coeffs[1].is_empty();
                            let ds_cdf =
                                partition::sb_none_do_split_cdf(row, col, above_pctx, left_pctx);
                            // intra_inter ctx for the is_inter flag (see rule below).
                            {
                                let up = row > 0;
                                let lf = col > 0;
                                let ia = i444_inter_above[col] == 1;
                                let il = (*i444_inter_left) == 1;
                                // AVM neighbors for a whole-64 SB resolve to the left SB
                                // (bottom-left+left) and above SB (above-right+above); a
                                // lone neighbor is scanned twice.
                                enc.intra_inter_ctx = if up && lf {
                                    if !il && !ia { 3 } else { (!il || !ia) as usize }
                                } else if lf {
                                    if il { 0 } else { 3 }
                                } else if up {
                                    if ia { 0 } else { 3 }
                                } else {
                                    0
                                };
                            }
                            // Chroma skip contexts (TX64): neighbor any-nonzero from the
                            // chroma entropy grids at this SB's mi origin.
                            let (fmr, fmc) = (row * 16, col * 16);
                            let ua = if fmr > 0 {
                                u_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
                            } else {
                                0
                            };
                            let ul = if fmc > 0 {
                                u_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
                            } else {
                                0
                            };
                            let va = if fmr > 0 {
                                v_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
                            } else {
                                0
                            };
                            let vl = if fmc > 0 {
                                v_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
                            } else {
                                0
                            };
                            let u_present = uv_coeffs[0].iter().any(|&(_, l)| l != 0);
                            let v_present = uv_coeffs[1].iter().any(|&(_, l)| l != 0);
                            if any_resid {
                                // Commit AQ only for residual blocks: whole-SB skips emit
                                // no delta_q, so the decoder's qindex must stay unchanged.
                                let _ = aqs.per_sb(enc, y, pw, sb_y, sb_x, width, height);
                                let (tu_skip_cdfs, tu_dc_sign) = sb_tu_contexts(
                                    &tus,
                                    sb_y,
                                    sb_x,
                                    above,
                                    left,
                                    enc.qc,
                                    (pw / 4) as i64,
                                    (ph / 4) as i64,
                                );
                                let u_skip = (6 + ua + ul) as u32;
                                let v_skip = (6 * (u_present as i32) + va + vl) as u32;
                                coder::emit_inter_newmv_residual_block_multi(
                                    enc,
                                    &coder::InterResidualSpec {
                                        part_cdf: ds_cdf,
                                        skip_ctx,
                                        mode_ctx,
                                        drl_ctx: mode_ctx,
                                        mode: inter_mode,
                                        scaled_row: mvd_row / 2,
                                        scaled_col: mvd_col / 2,
                                        luma_tus: &tus,
                                        luma_skip_cdfs: &tu_skip_cdfs,
                                        luma_dc_sign_ctxs: &tu_dc_sign,
                                    },
                                    &coder::InterChromaTus {
                                        u_tus: std::slice::from_ref(&uv_coeffs[0]),
                                        v_tus: std::slice::from_ref(&uv_coeffs[1]),
                                        u_skip_cdfs: &[u_skip],
                                        v_skip_cdfs: &[v_skip],
                                        u_tx64: true,
                                    },
                                );
                            } else {
                                coder::emit_inter_mode_block(
                                    enc,
                                    ds_cdf,
                                    skip_ctx,
                                    mode_ctx,
                                    mode_ctx,
                                    inter_mode,
                                    mvd_row / 2,
                                    mvd_col / 2,
                                );
                                let cx = sb_x / 4;
                                let cy = sb_y / 4;
                                let ae = (cx + 16).min(above.len());
                                let le = (cy + 16).min(left.len());
                                for v in above[cx..ae].iter_mut() {
                                    *v = 0x40;
                                }
                                for v in left[cy..le].iter_mut() {
                                    *v = 0x40;
                                }
                            }
                            // Update chroma entropy grids with this block's coeff presence.
                            for c in fmc..fmc + 16 {
                                u_above[c] = u_present as i32;
                                v_above[c] = v_present as i32;
                                cfl_above[c] = 0; // inter blocks are never CfL-coded
                            }
                            for r in fmr..fmr + 16 {
                                u_left[r] = u_present as i32;
                                v_left[r] = v_present as i32;
                                cfl_left[r] = 0;
                            }
                            partition::sb_none_pctx(row, col, above_pctx, left_pctx);
                            i444_skip_above[col] = if any_resid { 0 } else { 1 };
                            (*i444_skip_left) = if any_resid { 0 } else { 1 };
                            i444_inter_above[col] = 1;
                            (*i444_inter_left) = 1;
                            i444_newmv_above[col] = (inter_mode == 2) as u8;
                            (*i444_newmv_left) = (inter_mode == 2) as u8;
                            i444_mv_above[col] = Some(mv);
                            (*i444_mv_left) = Some(mv);
                            return true;
                        }
                    }
                }
                // NOTE: intra fall-through. The intra leaf reads i444_inter_above[col]
                // and i444_inter_left as NEIGHBOUR inter-ness for its intra_inter ctx,
                // so we must NOT clear them here — they are cleared after the SB below.
                i444_skip_above[col] = 0;
                (*i444_skip_left) = 0;
                i444_newmv_above[col] = 0;
                (*i444_newmv_left) = 0;
                i444_mv_above[col] = None;
                (*i444_mv_left) = None;
                i444_intra_this_sb = true;
            }
            false
        });
        if inter_committed {
            return;
        }
        // Fast-path SB chroma context at the SB-origin mi (col*16, row*16).
        let (fmr, fmc) = (row * 16, col * 16);
        let ua = if fmr > 0 {
            u_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
        } else {
            0
        };
        let ul = if fmc > 0 {
            u_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
        } else {
            0
        };
        let va = if fmr > 0 {
            v_above[fmc..fmc + 16].iter().any(|&x| x != 0) as i32
        } else {
            0
        };
        let vl = if fmc > 0 {
            v_left[fmr..fmr + 16].iter().any(|&x| x != 0) as i32
        } else {
            0
        };

        // TX_64X64 only carries a 32x32 coefficient region. That is acceptable
        // for subsampled chroma, but in 4:4:4 it creates a fixed spatial-bandwidth
        // ceiling: lowering q spends more bits on the retained low frequencies while
        // the missing full-resolution chroma detail cannot improve. Keep one uniform
        // policy over the frame instead of mixing TX64 and four-TX32 SBs through an
        // SSE proxy, which produced visible 64-pixel changes in chroma sharpness.
        //
        // The static-CDF q-context bug that originally made low-quality split leaves
        // unsafe has been fixed, so there is no longer a qidx safety gate here.
        // Edge SBs already use the native partition walk and smaller transforms.
        const AUTO_FULLRES_CHROMA_QIDX: u8 = 140;
        let full_interior = sb_x + 64 <= width && sb_y + 64 <= height;
        let sb_walk = needs_partition && !full_interior;
        let preserve_fullres_chroma = self.base_q_idx <= AUTO_FULLRES_CHROMA_QIDX;
        // NB: this deliberately uses the *previous* SB's accumulated qindex
        // (`current()`), not this SB's `per_sb_probe`. Feeding the dark-AQ-boosted
        // (lower) qstep into `choose_luma_64x64_partition` makes its SSE-RD over-select
        // the fragmented 32x32-leaf split, which is strongly RD-inferior to the whole-64
        // large-transform path (whole-64 is +8..+13 SS2 at matched-or-fewer bytes across
        // a diverse image set). The real over-split fix is `prefer_whole64` below.
        let (probe_qstep, probe_resid_scale) = if core_has_last {
            let (q, r, _, _) = aqs.current();
            (q, r)
        } else {
            (cell.qs_before, cell.resid_scale_before)
        };
        let luma_partition = if full_interior && self.tune.chroma_split {
            // Replay: reuse the cached partition decision, skipping the ~35%
            // `choose_luma_64x64_partition` search. The probe recon it would have
            // written is overwritten by the real leaf/whole-64 encode below, so
            // not running it is safe.
            let replayed = if let crate::av2::replay::DecideMode::Replay(cur) = &mut decide_mode {
                cur.next_part()
            } else {
                None
            };
            if let Some(p) = replayed {
                p
            } else {
                let p = choose_luma_64x64_partition(
                    recy,
                    &LumaSource {
                        plane: yp,
                        stride: pw,
                    },
                    &LumaFrameBlock {
                        frame_width: width,
                        frame_height: height,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaGridBlock {
                        mi_cols: tmc,
                        mi_rows: tmr,
                        y: sb_y,
                        x: sb_x,
                    },
                    &LumaPartitionSearch {
                        quant: LumaQuantSpec {
                            basis: &bases.luma,
                            qstep: probe_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                        sb: LumaSbSearch {
                            residual_scale: probe_resid_scale,
                            allow_directional: self.speed.try_directional(),
                        },
                        basis16: &bases.luma16x16,
                        basis8: &bases.c8x8,
                        allow_16x16: true,
                        allow_8x8: true,
                    },
                );
                if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
                    rec.push_part(p);
                }
                p
            }
        } else {
            LumaPartitionDecision::default()
        };
        // At low-to-mid quality the 444 whole-64 large-transform path is strongly
        // RD-superior to the 32x32-leaf split path: forcing whole-64 gains +8..+13 SS2
        // at matched-or-fewer bytes on a diverse image set (photos, skylines, fractals,
        // abstract), because SSE-based partition RD systematically under-values the
        // coherent large transform. The win fades toward high quality (where fine
        // splitting pays off), so gate the preference by base qindex. This also drops
        // the `preserve_fullres_chroma` forced split, which was spending luma bits on a
        // net-negative full-res-chroma trade in this quality band.
        // Threshold sits at 131 (one above the 444 deblock gate of base_q<=130) so the
        // two features own disjoint quality bands: deblock keeps [<=130] on the
        // byte-exact recursive split-leaf path, prefer_whole64 takes [>=131] where
        // deblock is already off. The whole-64 fast path's own loop-filter TU map is not
        // byte-exact yet, so it must not enter the deblock-active band; recovering the
        // [110,130] whole-64 win needs that map fixed (or 444-specific level signaling).
        const WHOLE64_PREFER_QIDX: u8 = 110;
        let prefer_whole64 = self.base_q_idx >= WHOLE64_PREFER_QIDX;
        let sb_use_split = !sb_walk
            && full_interior
            && !prefer_whole64
            && (luma_partition.split64 || preserve_fullres_chroma);
        if !sb_walk && !sb_use_split {
            self.encode_sb_whole64_444(
                enc,
                aqs,
                decide_mode,
                yp,
                up,
                vp,
                recy,
                recu,
                recv,
                above,
                left,
                above_pctx,
                left_pctx,
                midx_grid,
                u_above,
                v_above,
                u_left,
                v_left,
                cfl_above,
                cfl_left,
                db_sb_qidx,
                db_tx,
                db_tx_c,
                i444_inter_above,
                &mut (*i444_inter_left),
                i444_mv_above,
                &mut (*i444_mv_left),
                pw,
                width,
                height,
                sb_y,
                sb_x,
                row,
                col,
                sb_cols,
                tmc,
                tmr,
                ua,
                ul,
                va,
                vl,
                fmr,
                fmc,
                cell,
                core_has_last,
            );
            return;
        }

        // Walk + dispatch. For residues {6,8} each SB yields exactly one Leaf and
        // no RectType ops; RectType is handled generically for forward-compat.
        // A chroma-motivated interior split emits a 4x32x32 PARTITION_SPLIT instead.
        let ops = if sb_use_split {
            partition::sb_rd_split_ops(
                row,
                col,
                luma_partition.split32,
                luma_partition.split16,
                above_pctx,
                left_pctx,
            )
        } else {
            partition::sb_partition_ops(row, col, tmr as usize, tmc as usize, above_pctx, left_pctx)
        };
        // Per-SB AQ: on the split path, commit the delta-q and reconstruct at the
        // accumulated qstep (matching the decoder), not the base qstep.
        // Stills read the AQ grid cell (see the whole-64 note); `cell.sig` is 0 for
        // edge SBs (matching the `!sb_use_split` `delta_q_signaled = 0`), and for
        // edge cells `cell.{qs,resid_scale,qs_c,resid_scale_c}` equal `current()`.
        let (split_qstep, split_resid_scale, split_qstep_c, split_resid_scale_c) = if core_has_last
        {
            if sb_use_split {
                aqs.per_sb(enc, yp, pw, sb_y, sb_x, width, height)
            } else {
                enc.delta_q_signaled = 0;
                aqs.current()
            }
        } else {
            enc.delta_q_signaled = cell.sig;
            (cell.qs, cell.resid_scale, cell.qs_c, cell.resid_scale_c)
        };
        db_sb_qidx[row * sb_cols + col] = if core_has_last {
            aqs.current_qidx() as u16
        } else {
            cell.qidx as u16
        };
        enc.delta_q_pending = enc.delta_q_present;
        enc.in_interior_split = sb_use_split;
        // Walk leaves code DC chroma; clear any CMS mode left by a fast-path SB.
        enc.uv_mode = 0;
        enc.y_ctx = 0;
        // Reset the per-SB coded-mi mask consumed by MHCCP (see y420).
        enc.sb_coded = [0u8; 256];
        // Staged decouple — the walk pushes exactly one `entries` entry per SB
        // (mirroring the whole-64 branch): `Walk(leaves)` when every leaf shape is
        // replayable ({8×8, 4×4, 2×2}), else `Fallback` (replay re-searches). In
        // Replay we pop that entry now and reuse each captured luma winner, skipping
        // the per-leaf mode/tx search entirely (MHCCP + partition are already
        // cached separately). One `next()` per walk SB keeps the whole-64 cursor
        // aligned (each SB takes exactly one branch, deterministically).
        let replay_walk: Option<Vec<replay::LeafDecision>> =
            if let replay::DecideMode::Replay(cur) = &mut decide_mode {
                match cur.next() {
                    crate::av2::replay::SbDecision::Walk(v) => Some(v.clone()),
                    // `Fallback` (uncaptured) or an unexpected variant ⇒ re-search.
                    _ => None,
                }
            } else {
                None
            };
        let mut leaf_recs: Vec<crate::av2::replay::LeafDecision> = Vec::new();
        let mut walk_capturable = true;
        let mut cap_idx = 0usize;
        for op in &ops {
            let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
                partition::Op::RectType { cdf, val, ctx } => {
                    enc.bool_rect_type(*cdf, *val, *ctx);
                    continue;
                }
                partition::Op::Split {
                    do_split_cdf,
                    square_cdf,
                } => {
                    enc.bool_do_split(*do_split_cdf, 1);
                    if *square_cdf != 0 {
                        enc.bool_do_square_split(*square_cdf, 1);
                    }
                    continue;
                }
                partition::Op::Leaf {
                    bw_mi,
                    bh_mi,
                    part_cdf,
                    mi_row,
                    mi_col,
                } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276), *mi_row, *mi_col),
            };
            // Only the {8×8, 4×4, 2×2} residue-edge shapes are replayable so far
            // (8×8/4×4 skip their luma search from the captured winner; 2×2 is
            // search-free). Any other shape ⇒ the whole SB falls back to a re-search.
            if !matches!((bw_mi, bh_mi), (8, 8) | (4, 4) | (2, 2)) {
                walk_capturable = false;
            }
            // Deblock: record this leaf's TU rectangles. A 16×16-MI (64px) leaf is
            // coded SPLIT (4× TX_32X32); smaller leaves are a single TU.
            if bw_mi == 16 && bh_mi == 16 {
                push_whole64(db_tx, lmr, lmc, 0);
                // Chroma of a full-64 walk leaf is a single TX_64X64.
                db_tx_c.push((lmr, lmc, 16, 16));
            } else {
                db_tx.push((lmr, lmc, bw_mi, bh_mi));
                db_tx_c.push((lmr, lmc, bw_mi, bh_mi));
            }
            // Per-leaf position (a single SB may contain several leaves, e.g. the
            // two stacked 8×32 residue-2 edges). Shadow sb_y/sb_x so the arms below
            // address the leaf, not the SB origin.
            let sb_y = lmr * 4;
            let sb_x = lmc * 4;
            // Decoder y-mode index context (get_y_mode_idx_ctx): count of
            // directional above-right / bottom-left neighbor modes. Walk
            // leaves are never directional, so a leaf only sees a directional
            // neighbor when it touches its SB's top/left edge and the
            // adjacent SB was coded whole with a directional luma mode.
            enc.y_ctx = ((lmr % 16 == 0 && row > 0 && midx_grid[(row - 1) * sb_cols + col] != 0xff)
                as usize)
                + ((lmc % 16 == 0 && col > 0 && midx_grid[row * sb_cols + col - 1] != 0xff)
                    as usize);
            // intra_inter ctx: mirror the decoder's neighbor-inter logic. In an
            // inter tile, neighbors may be inter; AVM doubles a lone neighbor.
            // Fall back to the all-intra rule for still tiles.
            enc.intra_inter_ctx = if enc.inter_tile {
                let up = lmr > 0;
                let lf = lmc > 0;
                let ia = i444_inter_above[col] == 1;
                let il = (*i444_inter_left) == 1;
                if up && lf {
                    if !il && !ia { 3 } else { (!il || !ia) as usize }
                } else if lf {
                    if il { 0 } else { 3 }
                } else if up {
                    if ia { 0 } else { 3 }
                } else {
                    0
                }
            } else if lmr > 0 || lmc > 0 {
                3
            } else {
                0
            };
            let ua = if lmr > 0 {
                u_above[lmc..lmc + bw_mi].iter().any(|&x| x != 0) as i32
            } else {
                0
            };
            let ul = if lmc > 0 {
                u_left[lmr..lmr + bh_mi].iter().any(|&x| x != 0) as i32
            } else {
                0
            };
            let va = if lmr > 0 {
                v_above[lmc..lmc + bw_mi].iter().any(|&x| x != 0) as i32
            } else {
                0
            };
            let vl = if lmc > 0 {
                v_left[lmr..lmr + bh_mi].iter().any(|&x| x != 0) as i32
            } else {
                0
            };
            {
                let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
                let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
                enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                enc.cfl_use = false;
                enc.cfl_signaled = false;
                enc.mhccp_use = false;
            }
            let (u_present, v_present) = match (bw_mi, bh_mi) {
                (16, 16) => self.encode_walk_leaf16(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    split_qstep,
                    split_resid_scale,
                    split_qstep_c,
                    split_resid_scale_c,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (16, 8) => self.encode_walk_leaf16x8(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    split_qstep,
                    split_qstep_c,
                    split_resid_scale_c,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (8, 16) => self.encode_walk_leaf8x16(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    split_qstep,
                    split_qstep_c,
                    split_resid_scale_c,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (8, 8) => {
                    let walk = if let Some(v) = &replay_walk {
                        let d = &v[cap_idx];
                        cap_idx += 1;
                        crate::av2::replay::LeafWalk::Replay(d)
                    } else if matches!(decide_mode, crate::av2::replay::DecideMode::Capture(_)) {
                        crate::av2::replay::LeafWalk::Capture(&mut leaf_recs)
                    } else {
                        crate::av2::replay::LeafWalk::Off
                    };
                    self.encode_walk_leaf8x8(
                        enc,
                        recy,
                        recu,
                        recv,
                        yp,
                        up,
                        vp,
                        pw,
                        sb_y,
                        sb_x,
                        lmr,
                        lmc,
                        split_qstep,
                        split_qstep_c,
                        split_resid_scale_c,
                        mhccp_bounds,
                        above,
                        left,
                        tmc,
                        tmr,
                        pc,
                        ua,
                        ul,
                        va,
                        vl,
                        decide_mode,
                        walk,
                    )
                }
                (4, 16) => self.encode_walk_leaf4x16(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    split_qstep,
                    split_resid_scale,
                    split_qstep_c,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (16, 4) => self.encode_walk_leaf16x4(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    split_qstep,
                    split_resid_scale,
                    split_qstep_c,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (2, 8) => self.encode_walk_leaf2x8(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (8, 2) => self.encode_walk_leaf8x2(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (4, 4) => {
                    let walk = if let Some(v) = &replay_walk {
                        let d = &v[cap_idx];
                        cap_idx += 1;
                        crate::av2::replay::LeafWalk::Replay(d)
                    } else if matches!(decide_mode, crate::av2::replay::DecideMode::Capture(_)) {
                        crate::av2::replay::LeafWalk::Capture(&mut leaf_recs)
                    } else {
                        crate::av2::replay::LeafWalk::Off
                    };
                    self.encode_walk_leaf4x4(
                        enc,
                        recy,
                        recu,
                        recv,
                        yp,
                        up,
                        vp,
                        pw,
                        sb_y,
                        sb_x,
                        lmr,
                        lmc,
                        split_qstep,
                        split_qstep_c,
                        mhccp_bounds,
                        above,
                        left,
                        tmc,
                        tmr,
                        pc,
                        ua,
                        ul,
                        va,
                        vl,
                        decide_mode,
                        walk,
                    )
                }
                (2, 2) => self.encode_walk_leaf2x2(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                    decide_mode,
                ),
                (2, 4) => self.encode_walk_leaf2x4(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (4, 2) => self.encode_walk_leaf4x2(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (4, 8) => self.encode_walk_leaf4x8(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                (8, 4) => self.encode_walk_leaf8x4(
                    enc,
                    recy,
                    recu,
                    recv,
                    yp,
                    up,
                    vp,
                    pw,
                    sb_y,
                    sb_x,
                    lmr,
                    lmc,
                    split_qstep,
                    split_qstep_c,
                    mhccp_bounds,
                    above,
                    left,
                    tmc,
                    tmr,
                    pc,
                    ua,
                    ul,
                    va,
                    vl,
                ),
                other => unreachable!("unsupported lossy leaf {:?}", other),
            };
            // CfL-usage neighbor update: enc.cfl_use holds this leaf's decision
            // (true only for a (16,16) leaf that picked CfL; false otherwise).
            let cfl_used = enc.cfl_signaled as i32;
            for c in lmc..lmc + bw_mi {
                u_above[c] = u_present as i32;
                v_above[c] = v_present as i32;
                cfl_above[c] = cfl_used;
            }
            for r in lmr..lmr + bh_mi {
                u_left[r] = u_present as i32;
                v_left[r] = v_present as i32;
                cfl_left[r] = cfl_used;
            }
            // Mark this leaf's luma mi as coded for the next leaf's MHCCP.
            for r in (lmr & 15)..((lmr & 15) + bh_mi).min(16) {
                for c in (lmc & 15)..((lmc & 15) + bw_mi).min(16) {
                    enc.sb_coded[r * 16 + c] = 1;
                }
            }
        }
        // One `entries` push per walk SB (paired with the pop above), keeping the
        // cursor exactly one entry/SB across whole-64 and walk branches.
        if let crate::av2::replay::DecideMode::Capture(rec) = &mut decide_mode {
            rec.push(if walk_capturable {
                crate::av2::replay::SbDecision::Walk(std::mem::take(&mut leaf_recs))
            } else {
                crate::av2::replay::SbDecision::Fallback
            });
        }
        enc.in_interior_split = false;
        // Deferred intra fall-through reset: clear this SB's inter/left
        // neighbor flags now that the intra leaf has consumed them.
        if i444_intra_this_sb {
            i444_inter_above[col] = 0;
            (*i444_inter_left) = 0;
            i444_mv_above[col] = None;
            (*i444_mv_left) = None;
        }
    }

    pub(super) fn encode_444_core(
        &self,
        planes: Core444Planes<'_>,
        width: usize,
        height: usize,
        // Per-tile inter reference. `None` on the whole-frame path. When `Some`,
        // carries the FULL reference frame + this tile's origin; 4:4:4 chroma is
        // full-resolution, so chroma shares the luma stride and origin.
        tile_ref: Option<&crate::av2::tiling::TileRefCtx>,
        // Per-block CDEF pass 2: the frame decision (strengths + per-SB grid). When
        // `Some`, this pass emits the per-SB index symbols; when `None` (pass 1) the
        // core instead derives the decision from its own reconstruction.
        cdef_pre: Option<&crate::av2::cdef_est::CdefDecision>,
        // Staged-threading decision record. `Off` = search+emit inline (today's
        // behavior); `Capture` also logs each SB winner; `Replay` reuses logged
        // winners (re-searching only not-yet-converted `Fallback` SBs).
        mut decide_mode: crate::av2::replay::DecideMode<'_>,
    ) -> RangeEncoder {
        let Core444Planes { y, u: cb, v: cr } = planes;
        // Encode-time tuning (was AV2_* env). Captured once per region.
        let (pw, ph) = (sb_align(width), sb_align(height));
        // MHCCP reference padding is defined against the coded tile extent, not
        // the larger SB-aligned reconstruction allocation. Keep these bounds
        // separate from `pw`/`ph` so edge leaves cannot consume uncoded padding.
        let mhccp_bounds = cfl::MhccpBounds::from_luma(width, height, false, false);
        // Native-size 444: boundary-safe non-aligned sizes can signal real W×H so the
        // decoder reconstructs the full padded SB and crops — no AVIF clap box needed.
        let native_mi = lossy_native_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        let yp = pad_plane(y, width, height, pw, ph);
        let up = pad_plane(cb, width, height, pw, ph);
        let vp = pad_plane(cr, width, height, pw, ph);

        let _layout = Layout::I444;
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pw * ph];
        let mut recv = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.inter_tile = self.inter_tile.load(std::sync::atomic::Ordering::Relaxed);
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = false;
        enc.mhccp_ssy = false;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        // Pass 2: install the per-SB CDEF grid so the block-emit path signals the
        // per-unit index (nb_cdef_strengths = 2) at each SB's first coded block.
        if let Some(pre) = cdef_pre {
            enc.cdef_nb = 2;
            enc.cdef_cols = pre.sb_cols;
            enc.cdef_grid = pre.grid.clone();
            enc.cdef_decided = Some(pre.clone());
        }
        // Per-SB directional midx (0xff = non-directional) for the y-mode neighbor
        // context, mirroring the decoder's bottom_left/above_right joint modes.
        let mut midx_grid = vec![0xff_u8; sb_cols * sb_rows];
        // Per-mi chroma neighbor coeff-presence (mirrors the luma above/left arrays):
        // `*_above[mi_col]` / `*_left[mi_row]` hold whether the most recent TU covering
        // that column/row had U/V coeffs. Per-mi (not per-SB) so that multiple chroma
        // TUs within one SB — e.g. the two vertically stacked 8×32 residue-2 leaves —
        // see each other as neighbors.
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        // Per-mi CfL-usage neighbors for get_cfl_ctx: `cfl_above[mi_col]` / `cfl_left
        // [mi_row]` hold whether the chroma block covering that column/row used CfL
        // (uv_mode == UV_CFL_PRED). is_cfl context = above_used + left_used (0..2).
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        // Bottom-edge force-split: the last SB row is 32 px tall in frame, so each
        // 64X64 force-splits HORZ (implied, no bits) into a top 64X32 leaf coded by
        // the partition leaf path. Partition context `above_pctx` persists down
        // columns; `left_pctx` is len-16 and reset per SB row.
        // Force-split partition walk. When any edge residue is 6 or 8 the right/bottom
        // SBs split into 32-family leaves (32X64 / 64X32 / 32X32); otherwise every SB is
        // a whole 64X64. The walk drives `sb_partition_ops`, which also maintains the
        // partition contexts (`above_pctx` down columns, `left_pctx` reset per SB row).
        let needs_partition = native_mi.is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = [0u8; 16];
        // Inter (LAST-ref) support for 4:4:4: reference planes + per-SB neighbor
        // grids mirroring the 4:2:0 CORE path. Chroma is full-res (same MV, TX64).
        let (core_last_ref, ref_x0, ref_y0, ref_ls): (
            std::sync::Arc<Vec<Vec<f32>>>,
            usize,
            usize,
            usize,
        ) = if enc.inter_tile {
            match tile_ref {
                Some(r) => (std::sync::Arc::clone(&r.planes), r.x0, r.y0, r.luma_stride),
                None => (
                    std::sync::Arc::clone(&self.last_ref.lock().unwrap()),
                    0,
                    0,
                    pw,
                ),
            }
        } else {
            (std::sync::Arc::new(Vec::new()), 0, 0, pw)
        };
        let core_has_last = core_last_ref.len() >= 3 && !core_last_ref[0].is_empty();
        let frame_mv_seed = *self.video_mv_seed.lock().unwrap();
        // Frame-scoped inter preparation: convert current luma and build the
        // bordered reference ONCE per frame (was rebuilt per superblock).
        const BRD: usize = 72;
        let inter_luma: Option<(Vec<u16>, Vec<u16>, usize, usize)> = if core_has_last {
            let cur_u: Vec<u16> = y.iter().map(|&v| v.round().max(0.0) as u16).collect();
            let ref_u: Vec<u16> = core_last_ref[0]
                .iter()
                .map(|&v| v.round().max(0.0) as u16)
                .collect();
            let refh = ref_u.len() / ref_ls;
            let (bref, bstride) = mc::bordered(&ref_u, ref_ls, refh, ref_ls, BRD);
            Some((cur_u, bref, bstride, refh))
        } else {
            None
        };
        // Frame-scoped bordered chroma references (U, V), full-res for 4:4:4, built
        // once instead of per superblock in the NEWMV residual path.
        let inter_chroma: Option<[(Vec<u16>, usize); 2]> = if core_has_last {
            let build = |plane: &[f32]| {
                let u: Vec<u16> = plane.iter().map(|&v| v.round().max(0.0) as u16).collect();
                let ch = u.len() / ref_ls;
                mc::bordered(&u, ref_ls, ch, ref_ls, BRD)
            };
            Some([build(&core_last_ref[1]), build(&core_last_ref[2])])
        } else {
            None
        };
        let mut i444_skip_above = vec![0u8; sb_cols.max(1)];
        let mut i444_inter_above = vec![0u8; sb_cols.max(1)];
        let mut i444_newmv_above = vec![0u8; sb_cols.max(1)];
        // Per-SB decoded MVs (eighth-pel) for the DRL[0] predictor model.
        let mut i444_mv_above: Vec<Option<video::mv::Mv>> = vec![None; sb_cols.max(1)];
        let mut me_scratch = video::me::MeScratch::default();
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
            self.tune.uv_ac_delta_q,
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        )
        .with_dark_aq(self.tune.dark_aq);
        // Serial AQ pre-pass: the per-SB grid a wavefront (diagonal) decide reads
        // instead of the raster-serial `last_qidx` accumulator. Bit-exact with the
        // `per_sb`/`current` sequence (unit-tested). Used for stills; video keeps the
        // serial `aqs` (its accumulation interleaves with inter-skip).
        let aq_grid = aqs.precompute_grid(&yp, pw, width, height, needs_partition);

        // Deblock: TX-granular luma rectangles (4:4:4 mixes SPLIT 32×32 and VERT4/HORZ4
        // 16×64/64×16 tilings, so each recorded rect is exactly one TU) + per-SB qindex.
        let mut db_tx: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut db_tx_c: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut db_sb_qidx = vec![self.base_q_idx as u16; sb_rows * sb_cols];
        // The serial path retains its accumulated contexts. Wavefront workers reset
        // their private contexts before each independently-decided SB.
        let fresh_ctx = false;
        // Step 2b (wavefront recon plumbing, serial gate): route each SB's decide
        // through a private full-plane buffer whose only valid data is the halo
        // (finished neighbours) copied from the real recon `out`. Poisoning the rest
        // proves the halo is a complete superset of every read `decide_sb` makes
        // OUTSIDE its own 64×64 block — the exact isolation a parallel wavefront
        // worker has. The parallel branch below always uses the same geometry.
        let halo_mode = false;
        // Halo geometry (audited): intra uses a single reference line (row sb_y-1,
        // col sb_x-1) + above-right extension of the top row (≤ sb_x+96); MHCCP/CfL
        // read the same 1px border; bottom-left stays within the SB's own row band.
        // Copy a generous superset: 32px perpendicular band + 64px above-right margin.
        const HALO_BAND: usize = 32;
        const HALO_AR: usize = 64;
        // The private buffer must mirror B1's zero-initialised recon plane as seen by
        // decide: zero everywhere except the finished-neighbour halo. Own-block and
        // not-yet-coded regions read as 0 exactly as in B1's fresh plane. (A missed
        // *non-zero* causal read still diverges, so the gate keeps its detecting power.)
        const POISON: f32 = 0.0;
        let (mut recy_p, mut recu_p, mut recv_p) = if halo_mode {
            (
                vec![0f32; pw * ph],
                vec![0f32; pw * ph],
                vec![0f32; pw * ph],
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new())
        };
        let copy_halo =
            |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize, copy_rd_anchor: bool| {
                let bx = sb_x.saturating_sub(HALO_BAND);
                let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                let by = sb_y.saturating_sub(HALO_BAND);
                // Top band (above + above-right + corner): rows [by, sb_y) × [bx, tx_end).
                for r in by..sb_y {
                    dst[r * pw + bx..r * pw + tx_end]
                        .copy_from_slice(&src[r * pw + bx..r * pw + tx_end]);
                }
                // Left band (left ref, within the SB's own row band): rows [sb_y, sb_y+64) × [bx, sb_x).
                let ly_end = (sb_y + 64).min(ph);
                for r in sb_y..ly_end {
                    dst[r * pw + bx..r * pw + sb_x]
                        .copy_from_slice(&src[r * pw + bx..r * pw + sb_x]);
                }
                // The whole-64 RD path's historical `sse_region` call passes the full
                // recon plane where `pixel_sse_rounded_block` expects a block-origin
                // slice. Consequently every non-origin SB reads the finished 64x64
                // reconstruction at (0,0) while selecting its tx partition. Preserve
                // that shipping decision dependency without copying the full causal
                // plane. (Changing the call itself would alter serial output.)
                if copy_rd_anchor && (sb_y != 0 || sb_x != 0) {
                    for r in 0..64.min(ph) {
                        dst[r * pw..r * pw + 64].copy_from_slice(&src[r * pw..r * pw + 64]);
                    }
                }
            };
        let copy_own = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let ly_end = (sb_y + 64).min(ph);
            for r in sb_y..ly_end {
                dst[r * pw + sb_x..r * pw + sb_x + 64]
                    .copy_from_slice(&src[r * pw + sb_x..r * pw + sb_x + 64]);
            }
        };
        // Step 3: parallel SB-wavefront decide (stills only). Each SB is decided by a
        // worker under the `par_wavefront_wpp` (d=2r+c) schedule — top+left+above-right
        // neighbours finished — reading its halo from the shared recon and writing its
        // own 64×64 block back (sound raw-ptr disjoint access), capturing into a per-SB
        // record. Merged raster → the serial Replay emits.
        // Worker scratch is reset to the same frame-initial contexts used by the
        // context-independence proof. It must not inherit contexts from the worker's
        // previous (unrelated) diagonal cell: pool assignment is nondeterministic and
        // stale values affect the captured record even though fresh values do not.
        // The SB-wavefront fires whenever the caller selected a multi-threaded
        // single-tile whole-frame still Capture pass. It is suppressed inside
        // per-tile sub-encodes, where tiles already provide the parallelism.
        let wavefront = Self::resolve_threads(self.threads) > 1
            && !crate::av2::replay::in_tile_subencode()
            && !self.video_mode.load(std::sync::atomic::Ordering::Relaxed)
            && matches!(decide_mode, crate::av2::replay::DecideMode::Capture(_))
            && !core_has_last;
        if wavefront {
            let enc_qc = enc.qc;
            let enc_cfl = enc.cfl;
            let enc_mhccp = enc.mhccp;
            let enc_dqp = enc.delta_q_present;
            let enc_inter = enc.inter_tile;
            let adaptive = self.tune.updating_cdf && self.base_q_idx != 0;
            let base_q = self.base_q_idx as i32;
            let uv_delta = self.tune.uv_ac_delta_q;
            let nthreads = Self::resolve_threads(self.threads);
            let wy = helpers::PlaneWriter::new(&mut recy, pw);
            let wu = helpers::PlaneWriter::new(&mut recu, pw);
            let wv = helpers::PlaneWriter::new(&mut recv, pw);
            // Persistent-pool WPP wavefront: workers spawn once and loop over the
            // diagonals with a barrier between each, so each worker's thread_local
            // recon scratch is allocated once (not re-allocated per diagonal).
            let slots: Vec<Option<WfSlot>> =
                helpers::par_wavefront_pool(nthreads, sb_rows, sb_cols, true, |r, c| {
                    let sb_y = r * 64;
                    let sb_x = c * 64;

                    WF_SCRATCH.with(|sc| {
                        let s = &mut *sc.borrow_mut();
                        s.ensure(pw, ph, tmc, tmr, sb_cols, sb_rows);
                        s.reset_contexts();
                        let ly_end = (sb_y + 64).min(ph);
                        let bx = sb_x.saturating_sub(HALO_BAND);
                        let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                        let by = sb_y.saturating_sub(HALO_BAND);
                        // CRITICAL for parallel correctness: the thread-local buffer is
                        // REUSED across cells, so it must present decide the SAME view B1's
                        // fresh plane does — ZERO everywhere except the finished-neighbour
                        // halo. Rather than enumerate every region decide might read (the
                        // walk path reaches beyond the obvious halo box), we keep the whole
                        // buffer ZERO by INVARIANT: it starts zero (allocated in `ensure`),
                        // and each cell RESTORES (re-zeroes) everything it dirtied — the
                        // copied halo + its own-block — at the end (below). So on entry the
                        // buffer is all-zero, decide reads its halo (copied real) and zero
                        // everywhere else, exactly like B1. Bounded cost (dirty region), no
                        // per-cell whole-plane zero.
                        // Halo (finished neighbours): top band + left band, from `out`.
                        // SAFETY: under WPP these regions are earlier-diagonal (finished),
                        // not being written concurrently; own-block writes are disjoint.
                        unsafe {
                            if sb_y != 0 || sb_x != 0 {
                                wy.copy_region_to(&mut s.ry, 0, 0, 64.min(ph), 64);
                            }
                            if sb_y > by {
                                wy.copy_region_to(&mut s.ry, by, bx, sb_y - by, tx_end - bx);
                                wu.copy_region_to(&mut s.ru, by, bx, sb_y - by, tx_end - bx);
                                wv.copy_region_to(&mut s.rv, by, bx, sb_y - by, tx_end - bx);
                            }
                            if ly_end > sb_y && sb_x > bx {
                                wy.copy_region_to(&mut s.ry, sb_y, bx, ly_end - sb_y, sb_x - bx);
                                wu.copy_region_to(&mut s.ru, sb_y, bx, ly_end - sb_y, sb_x - bx);
                                wv.copy_region_to(&mut s.rv, sb_y, bx, ly_end - sb_y, sb_x - bx);
                            }
                        }
                        let mut e = RangeEncoder::new();
                        e.inter_tile = enc_inter;
                        e.qc = enc_qc;
                        // DecideOnly: the Capture pass's emitted bytes are discarded (only
                        // the serial Replay produces the real bitstream) and no decision
                        // reads the emit state (context-independence, proven byte-identical).
                        // So run the encoder as a non-adaptive SINK — no per-SB `CdfState`
                        // (hundreds of nested CDF Vecs), and every `encode_symbol`/`_bool`
                        // short-circuits (`sink`), skipping all range coding, CDF adaptation
                        // and `output` growth. Coefficient emits route through the guarded
                        // static path since `cdf_state` stays `None`. Decisions + captured
                        // recon are unchanged; the record drives Replay's real emit.
                        e.sink = true;
                        let _ = adaptive;
                        e.cfl = enc_cfl;
                        e.mhccp = enc_mhccp;
                        e.mhccp_ssx = false;
                        e.mhccp_ssy = false;
                        e.delta_q_present = enc_dqp;
                        let mut a = aq::AqState::new(enc_dqp, base_q, qstep_i, 0.0, uv_delta);
                        let mut rec = crate::av2::replay::DecisionRecord::new();
                        let mut dm = crate::av2::replay::DecideMode::Capture(&mut rec);
                        let mut db_tx: Vec<(usize, usize, usize, usize)> = Vec::new();
                        let mut db_tx_c: Vec<(usize, usize, usize, usize)> = Vec::new();
                        let mut lpctx = [0u8; 16];
                        let (mut isl, mut iil, mut inl) = (0u8, 0u8, 0u8);
                        let mut iml: Option<video::mv::Mv> = None;
                        self.decide_sb(
                            &mut e,
                            &mut a,
                            &mut dm,
                            y,
                            cb,
                            cr,
                            &yp,
                            &up,
                            &vp,
                            &mut s.ry,
                            &mut s.ru,
                            &mut s.rv,
                            &mut s.above,
                            &mut s.left,
                            &mut s.apctx,
                            &mut lpctx,
                            &mut s.midx,
                            &mut s.ua,
                            &mut s.va,
                            &mut s.ul,
                            &mut s.vl,
                            &mut s.cfa,
                            &mut s.cfl,
                            &mut s.dbq,
                            &mut db_tx,
                            &mut db_tx_c,
                            &mut s.isa,
                            &mut s.iia,
                            &mut s.ina,
                            &mut s.ima,
                            &mut isl,
                            &mut iil,
                            &mut inl,
                            &mut iml,
                            &core_last_ref,
                            &inter_luma,
                            &inter_chroma,
                            &mut s.me,
                            mhccp_bounds,
                            pw,
                            ph,
                            width,
                            height,
                            sb_cols,
                            tmc,
                            tmr,
                            qstep_i,
                            ref_x0,
                            ref_y0,
                            ref_ls,
                            false,
                            frame_mv_seed,
                            needs_partition,
                            &aq_grid,
                            r,
                            c,
                        );
                        // Write own 64×64 block back to `out`.
                        let h = ly_end - sb_y;
                        let mut blk = [0f32; 64 * 64];
                        let gather = |plane: &[f32], blk: &mut [f32; 64 * 64]| {
                            for rr in 0..h {
                                blk[rr * 64..rr * 64 + 64].copy_from_slice(
                                    &plane[(sb_y + rr) * pw + sb_x..(sb_y + rr) * pw + sb_x + 64],
                                );
                            }
                        };
                        // SAFETY: this SB's own block is disjoint from every other worker's.
                        gather(&s.ry, &mut blk);
                        unsafe { wy.write_block(sb_y, sb_x, h, 64, &blk[..h * 64]) };
                        gather(&s.ru, &mut blk);
                        unsafe { wu.write_block(sb_y, sb_x, h, 64, &blk[..h * 64]) };
                        gather(&s.rv, &mut blk);
                        unsafe { wv.write_block(sb_y, sb_x, h, 64, &blk[..h * 64]) };
                        // Restore the buffer to all-zero: re-zero exactly what was dirtied
                        // this cell — the copied halo (top band [by,sb_y) + left band
                        // [sb_y,ly_end)) plus the own-block decide wrote — i.e. the box
                        // [by, ly_end) × [bx, tx_end). Keeps the all-zero-except-halo
                        // invariant for the next cell on this worker (see note above).
                        for rr in by..ly_end {
                            s.ry[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                            s.ru[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                            s.rv[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                        }
                        if sb_y != 0 || sb_x != 0 {
                            for rr in 0..64.min(ph) {
                                s.ry[rr * pw..rr * pw + 64].fill(0.0);
                            }
                        }
                        WfSlot {
                            record: rec,
                            db_tx,
                            db_tx_c,
                        }
                    })
                });
            // Merge per-SB records in raster order → the Capture record (the serial
            // Replay consumes the exact sequence a serial Capture would have logged).
            if let replay::DecideMode::Capture(rec) = &mut decide_mode {
                for slot in slots.into_iter() {
                    let slot = slot.expect("every SB decided");
                    rec.append(slot.record);
                    db_tx.extend(slot.db_tx);
                    db_tx_c.extend(slot.db_tx_c);
                }
            }
            for (i, cell) in aq_grid.iter().enumerate() {
                db_sb_qidx[i] = cell.qidx as u16;
            }
        } else {
            for row in 0..sb_rows {
                left_pctx.iter_mut().for_each(|p| *p = 0);
                let mut i444_skip_left = 0u8;
                let mut i444_inter_left = 0u8;
                let mut i444_newmv_left = 0u8;
                let mut i444_mv_left: Option<video::mv::Mv> = None;
                for col in 0..sb_cols {
                    if fresh_ctx {
                        above.iter_mut().for_each(|v| *v = 0x40);
                        left.iter_mut().for_each(|v| *v = 0x40);
                        above_pctx.iter_mut().for_each(|v| *v = 0);
                        left_pctx.iter_mut().for_each(|v| *v = 0);
                        midx_grid.iter_mut().for_each(|v| *v = 0xff);
                        u_above.iter_mut().for_each(|v| *v = 0);
                        v_above.iter_mut().for_each(|v| *v = 0);
                        u_left.iter_mut().for_each(|v| *v = 0);
                        v_left.iter_mut().for_each(|v| *v = 0);
                        cfl_above.iter_mut().for_each(|v| *v = 0);
                        cfl_left.iter_mut().for_each(|v| *v = 0);
                        i444_skip_above.iter_mut().for_each(|v| *v = 0);
                        i444_inter_above.iter_mut().for_each(|v| *v = 0);
                        i444_newmv_above.iter_mut().for_each(|v| *v = 0);
                        i444_mv_above.iter_mut().for_each(|v| *v = None);
                        i444_skip_left = 0;
                        i444_inter_left = 0;
                        i444_newmv_left = 0;
                        i444_mv_left = None;
                    }
                    let sb_y = row * 64;
                    let sb_x = col * 64;
                    if halo_mode {
                        recy_p.iter_mut().for_each(|v| *v = POISON);
                        recu_p.iter_mut().for_each(|v| *v = POISON);
                        recv_p.iter_mut().for_each(|v| *v = POISON);
                        copy_halo(&mut recy_p, &recy, sb_y, sb_x, true);
                        copy_halo(&mut recu_p, &recu, sb_y, sb_x, false);
                        copy_halo(&mut recv_p, &recv, sb_y, sb_x, false);
                    }
                    let (ry, ru, rv): (&mut [f32], &mut [f32], &mut [f32]) = if halo_mode {
                        (
                            recy_p.as_mut_slice(),
                            recu_p.as_mut_slice(),
                            recv_p.as_mut_slice(),
                        )
                    } else {
                        (
                            recy.as_mut_slice(),
                            recu.as_mut_slice(),
                            recv.as_mut_slice(),
                        )
                    };
                    self.decide_sb(
                        &mut enc,
                        &mut aqs,
                        &mut decide_mode,
                        y,
                        cb,
                        cr,
                        &yp,
                        &up,
                        &vp,
                        ry,
                        ru,
                        rv,
                        &mut above,
                        &mut left,
                        &mut above_pctx,
                        &mut left_pctx,
                        &mut midx_grid,
                        &mut u_above,
                        &mut v_above,
                        &mut u_left,
                        &mut v_left,
                        &mut cfl_above,
                        &mut cfl_left,
                        &mut db_sb_qidx,
                        &mut db_tx,
                        &mut db_tx_c,
                        &mut i444_skip_above,
                        &mut i444_inter_above,
                        &mut i444_newmv_above,
                        &mut i444_mv_above,
                        &mut i444_skip_left,
                        &mut i444_inter_left,
                        &mut i444_newmv_left,
                        &mut i444_mv_left,
                        &core_last_ref,
                        &inter_luma,
                        &inter_chroma,
                        &mut me_scratch,
                        mhccp_bounds,
                        pw,
                        ph,
                        width,
                        height,
                        sb_cols,
                        tmc,
                        tmr,
                        qstep_i,
                        ref_x0,
                        ref_y0,
                        ref_ls,
                        core_has_last,
                        frame_mv_seed,
                        needs_partition,
                        &aq_grid,
                        row,
                        col,
                    );
                    if halo_mode {
                        copy_own(&mut recy, &recy_p, sb_y, sb_x);
                        copy_own(&mut recu, &recu_p, sb_y, sb_x);
                        copy_own(&mut recv, &recv_p, sb_y, sb_x);
                    }
                }
            }
        }
        // Real deblocking (4:4:4). Leaves recorded at TX granularity across the whole-64
        // (SPLIT/VERT4/HORZ4) and split/walk paths, so luma_tx_cap=(16,16) leaves each
        // recorded rect as one TU. Chroma is full-res and follows the luma tx (ssx=ssy=0).
        let df_quant = {
            let qs = quant::qstep(self.base_q_idx as u32) as i32;
            ((qs + 4) >> 3) >> 6
        };
        // Byte-exact for the split path (base_q<=140 = `preserve_fullres_chroma`, where
        // every full-interior SB is coded via the split ops → Op::Leaf recording). The
        // whole-64 fast path (base_q>140) RD-picks SPLIT/VERT4/HORZ4 whose TX tilings the
        // recorder doesn't yet reproduce exactly (VERT4/HORZ4), so it's gated out for now.
        // SB-aligned only for now: partial edge SBs take the whole-64 path but the
        // synthesized full-64 leaf produces spurious TU edges (and out-of-range reads).
        let sb_aligned_444 = width.is_multiple_of(64) && height.is_multiple_of(64);
        if self.tune.deblock && df_quant >= 1 && !needs_partition && sb_aligned_444 {
            let eff_dy = if self.tune.db_delta_y == i32::MIN {
                if df_quant >= 5 { 0 } else { 1 }
            } else {
                self.tune.db_delta_y
            };
            crate::av2::deblock::deblock_frame(crate::av2::deblock::FrameDeblock {
                recy: &mut recy,
                recu: &mut recu,
                recv: &mut recv,
                luma_stride: pw,
                chroma_stride: pw,
                width,
                height,
                ssx: 0,
                ssy: 0,
                has_chroma: true,
                chroma_deblock: self.tune.chroma_deblock,
                bit_depth: self.bit_depth as u32,
                delta_y: eff_dy,
                delta_uv: self.tune.db_delta_uv,
                base_q: self.base_q_idx as u16,
                sb_cols,
                sb_qidx: &db_sb_qidx,
                leaves: &db_tx,
                chroma_leaves: &db_tx_c,
                skip_leaves: &[],
                luma_tx_cap: (16, 16),
                chroma_tx_cap: (16, 16),
            });
        }
        // Video: stash final recon (Y,U,V) for the DPB; skipped for still tiles.
        if self
            .capture_recon
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.recon = vec![recy.clone(), recu.clone(), recv.clone()];
        }
        // Pass 1 (single-tile whole-frame): derive the per-block CDEF decision from
        // this reconstruction for a pass-2 re-emit. Skipped when a grid is already
        // installed (pass 2), or when recon is captured for the tiled path.
        if self.tune.cdef
            && tile_ref.is_none()
            && cdef_pre.is_none()
            && !self
                .capture_recon
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.cdef_decided = crate::av2::cdef_est::search_per_block(
                &[recy, recu, recv],
                &[yp, up, vp],
                pw,
                ph,
                pw,
                ph,
                0,
                0,
                true,
                self.base_q_idx,
                self.bit_depth,
            );
        }
        enc
    }
}

#[cfg(test)]
mod wavefront_scratch_tests {
    use super::WfScratch;
    use crate::av2::video;

    #[test]
    fn reused_worker_scratch_is_fresh_and_resized_per_cell() {
        let mut s = WfScratch::default();
        s.ensure(128, 64, 32, 16, 2, 1);

        // Model a completed SB contaminating every context family on its worker.
        s.above.fill(1);
        s.left.fill(1);
        s.apctx.fill(1);
        s.midx.fill(1);
        s.ua.fill(1);
        s.va.fill(1);
        s.ul.fill(1);
        s.vl.fill(1);
        s.cfa.fill(1);
        s.cfl.fill(1);
        s.isa.fill(1);
        s.iia.fill(1);
        s.ina.fill(1);
        s.ima.fill(Some(video::mv::Mv { row: 1, col: 1 }));
        s.dbq.fill(1);

        // Same width, different height used to skip resizing because `ensure`
        // incorrectly keyed every array off the width-dependent `above.len()`.
        s.ensure(128, 128, 32, 32, 2, 2);
        s.reset_contexts();

        assert_eq!(s.left.len(), 128 / 4 + 16);
        assert_eq!(s.midx.len(), 4);
        assert_eq!(s.dbq.len(), 4);
        assert!(s.above.iter().all(|&v| v == 0x40));
        assert!(s.left.iter().all(|&v| v == 0x40));
        assert!(s.apctx.iter().all(|&v| v == 0));
        assert!(s.midx.iter().all(|&v| v == 0xff));
        assert!(
            s.ua.iter()
                .chain(&s.va)
                .chain(&s.ul)
                .chain(&s.vl)
                .all(|&v| v == 0)
        );
        assert!(s.cfa.iter().chain(&s.cfl).all(|&v| v == 0));
        assert!(s.isa.iter().chain(&s.iia).chain(&s.ina).all(|&v| v == 0));
        assert!(s.ima.iter().all(Option::is_none));
        assert!(s.dbq.iter().all(|&v| v == 0));
    }
}
