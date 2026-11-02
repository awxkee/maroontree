/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! SDP (semi-decoupled partitioning) chroma tree for 4:2:0 key frames. With SDP
//! a superblock codes its whole luma tree first (leaves carry no chroma syntax),
//! then chroma with its own partition: plane-1 partition contexts and CDFs, a
//! chroma-only mode (CfL / MHCCP / uv mode) and one U + one V TU per leaf.

use super::super::*;
use crate::av2::partition::{ChromaOp, sdp_chroma_ops};

/// The SB state the chroma tree reads and writes.
pub(crate) struct ChromaTree420<'a, 'r> {
    pub(crate) enc: &'a mut RangeEncoder,
    pub(crate) decide_mode: &'a mut replay::DecideMode<'r>,
    pub(crate) recy: &'a [f32],
    pub(crate) recu: &'a mut [f32],
    pub(crate) recv: &'a mut [f32],
    pub(crate) up: &'a [f32],
    pub(crate) vp: &'a [f32],
    pub(crate) pw: usize,
    pub(crate) pcw: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) u_above: &'a mut [i32],
    pub(crate) v_above: &'a mut [i32],
    pub(crate) u_left: &'a mut [i32],
    pub(crate) v_left: &'a mut [i32],
    pub(crate) cfl_above: &'a mut [i32],
    pub(crate) cfl_left: &'a mut [i32],
    /// The SB's luma leaves `(mi_row, mi_col, bw_mi, bh_mi)`.
    pub(crate) luma_leaves: &'a [(usize, usize, usize, usize)],
    /// Output: the chroma tree's leaves (luma-mi rects) for the chroma deblock.
    pub(crate) chroma_leaves: &'a mut Vec<(usize, usize, usize, usize)>,
    /// Luma `midx` per SB mi (0xff = non-directional): the uv-mode context.
    pub(crate) luma_midx: &'a [u8; 256],
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) tmr: i64,
    pub(crate) tmc: i64,
    pub(crate) qc: usize,
    pub(crate) neutral: f32,
    pub(crate) sb_qstep: i32,
    pub(crate) mhccp_bounds: cfl::MhccpBounds,
    /// Chroma subsampling `(ssx, ssy)`: (1, 1) 4:2:0, (1, 0) 4:2:2.
    pub(crate) ss: (usize, usize),
}

/// TU spec of a chroma block (`cw` x `ch` px); 4:2:0 4x4 is coded separately and
/// 4:4:4 64x64 / 64x32 / 32x64 emit through `encode_chroma_block` (TX64 slot).
fn chroma_spec<'b>(bases: &'b proj::Bases, cw: usize, ch: usize, qc: usize) -> ChromaTxSpec<'b> {
    let mk = |basis, scan, eob_cdf, area, u_skip_row| ChromaTxSpec {
        cw,
        ch,
        basis,
        scan,
        eob_cdf,
        eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
        area,
        u_skip_row,
    };
    match (cw, ch) {
        (32, 32) => mk(
            &bases.chroma420,
            &tables::SCAN,
            EobCdf::ChrEobBin,
            1024,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (64, 64) => mk(
            &bases.chroma444,
            &tables::SCAN,
            EobCdf::ChrEobBin,
            1024,
            &CHROMA_SKIP_TX64_QC[qc],
        ),
        (64, 32) => mk(
            &bases.chroma444_64x32,
            &tables::SCAN,
            EobCdf::ChrEobBin,
            1024,
            &CHROMA_SKIP_TX64_QC[qc],
        ),
        (64, 16) => mk(
            &bases.luma64x16,
            &SCAN32X16,
            EobCdf::ChrEob512,
            512,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (32, 64) => mk(
            &bases.chroma422,
            &tables::SCAN,
            EobCdf::ChrEobBin,
            1024,
            &CHROMA_SKIP_TX64_QC[qc],
        ),
        (16, 64) => mk(
            &bases.luma16x64,
            &SCAN16X32,
            EobCdf::ChrEob512,
            512,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (8, 64) => mk(
            &bases.c8x64,
            &SCAN8X32,
            EobCdf::ChrEob256,
            256,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (4, 32) => mk(
            &bases.c4x32,
            &tables::SCAN4X32,
            EobCdf::ChrEob128,
            128,
            &SKIP_TX16_QC[qc],
        ),
        (32, 16) => mk(
            &bases.c32x16,
            &SCAN32X16,
            EobCdf::ChrEob512,
            512,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (16, 32) => mk(
            &bases.c16x32,
            &SCAN16X32,
            EobCdf::ChrEob512,
            512,
            &CHROMA_SKIP_TX32_QC[qc],
        ),
        (16, 16) => mk(
            &bases.luma16x16,
            &SCAN16,
            EobCdf::ChrEob256,
            256,
            &SKIP_TX16_QC[qc],
        ),
        (32, 8) => mk(
            &bases.luma32x8,
            &SCAN32X8,
            EobCdf::ChrEob256,
            256,
            &SKIP_TX16_QC[qc],
        ),
        (8, 32) => mk(
            &bases.luma8x32,
            &SCAN8X32,
            EobCdf::ChrEob256,
            256,
            &SKIP_TX16_QC[qc],
        ),
        (16, 8) => mk(
            &bases.c16x8,
            &tables::SCAN16X8,
            EobCdf::ChrEob128,
            128,
            &SKIP_TX16_QC[qc],
        ),
        (8, 16) => mk(
            &bases.c8x16,
            &tables::SCAN8X16,
            EobCdf::ChrEob128,
            128,
            &SKIP_TX16_QC[qc],
        ),
        (8, 8) => mk(
            &bases.c8x8,
            &SCAN8X8,
            EobCdf::ChrEob64,
            64,
            &SKIP_TX8_QC[qc],
        ),
        (16, 4) => mk(
            &bases.c16x4,
            &tables::SCAN16X4,
            EobCdf::ChrEob64,
            64,
            &SKIP_TX8_QC[qc],
        ),
        (4, 16) => mk(
            &bases.c4x16,
            &tables::SCAN4X16,
            EobCdf::ChrEob64,
            64,
            &SKIP_TX8_QC[qc],
        ),
        (8, 4) => mk(
            &bases.c8x4,
            &tables::SCAN8X4,
            EobCdf::ChrEob32,
            32,
            &SKIP_TX8_QC[qc],
        ),
        (4, 8) => mk(
            &bases.c4x8,
            &tables::SCAN4X8,
            EobCdf::ChrEob32,
            32,
            &SKIP_TX8_QC[qc],
        ),
        _ => unreachable!("4:2:0 chroma-tree block {cw}x{ch}"),
    }
}

impl Av2Encoder {
    /// Code the chroma tree of one SB after its luma tree. The tree is chosen by
    /// coding a few candidate plans for real into a copy of the encoder (exact
    /// bits) and keeping the lowest U+V SSE + lambda * bits.
    pub(crate) fn code_chroma_tree_420(&self, t: &mut ChromaTree420<'_, '_>) {
        let _qm = crate::av2::qm::ChromaScope::new();
        let (mi_rows, mi_cols) = (t.tmr as usize, t.tmc as usize);
        if t.enc.sdp_above_c.len() < mi_cols + 16 {
            t.enc.sdp_above_c.resize(mi_cols + 16, 0);
        }
        let leaves = t.luma_leaves;
        let luma_part = |r: usize, c: usize, b: usize| {
            partition::luma_part_from_leaves(leaves, r, c, b, mi_rows, mi_cols)
        };
        let (r0, c0) = (t.row * 16, t.col * 16);
        let ss_is_420 = t.ss == (1, 1);
        let ss_is_422 = t.ss == (1, 0);
        const V: u8 = 1;
        const H: u8 = 0;
        const NONE: u8 = partition::PART_NONE;
        // Plans by block size (luma coords): 12 = 64x64, 10 = 32x64, 9 = 32x32,
        // 7 = 16x32, 6 = 16x16. Boundary-forced and inferred nodes override them.
        // The luma tree as a chroma plan, except where it would make 8x4 / 4x8 or
        // 4:1 chroma blocks (those chroma coders are frame-edge only and drift in the
        // interior): a 16x16 luma node split into a 16x8 / 8x16 pair keeps one 8x8
        // chroma block, a 2:1 luma node split along its short side keeps its 2:1
        // chroma block.
        let mirror = |r: usize, c: usize, b: usize| -> u8 {
            let p = luma_part(r, c, b);
            // 4:2:2 keeps a 16x16 luma node's chroma whole: below it the chroma
            // blocks would be 4x8, whose coders are frame-edge only.
            if ss_is_422 && b == 6 {
                return NONE;
            }
            if !ss_is_420 || p == NONE {
                return p;
            }
            match b {
                6 => {
                    // 16x16 = 4 mi: halves at +2 mi along the split.
                    let (half, r2, c2) = if p == V { (4, r, c + 2) } else { (5, r + 2, c) };
                    if luma_part(r, c, half) == NONE || luma_part(r2, c2, half) == NONE {
                        NONE
                    } else {
                        p
                    }
                }
                8 | 11 if p == H => NONE,
                7 | 10 if p == V => NONE,
                _ => p,
            }
        };
        let plan = |k: usize, r: usize, c: usize, b: usize| -> u8 {
            match k {
                0 => mirror(r, c, b),
                1 => NONE,
                2 => {
                    if b == 12 {
                        V
                    } else {
                        NONE
                    }
                }
                3 => match b {
                    12 => V,
                    10 => H,
                    _ => NONE,
                },
                4 => match b {
                    12 | 9 => V,
                    10 | 7 => H,
                    _ => NONE,
                },
                5 => {
                    if b == 6 {
                        NONE
                    } else {
                        mirror(r, c, b)
                    }
                }
                // Square mirror: the luma split structure with every VERT/HORZ pair
                // turned into a quad (chroma keeps square blocks where luma went rect).
                _ => match b {
                    12 | 9 | 6 => {
                        if partition::luma_node_covered(leaves, r, c, b) {
                            NONE
                        } else {
                            V
                        }
                    }
                    10 | 7 | 4 => H,
                    _ => mirror(r, c, b),
                },
            }
        };
        let in_frame = r0 + 16 <= mi_rows && c0 + 16 <= mi_cols;
        let replayed = t.decide_mode.replay_chroma_plan();
        // The winning trial's state, adopted instead of coding that plan again:
        // (encoder, U and V recon of the SB, the six context spans, chroma leaves,
        // logged MHCCP choices).
        type Adopted = (
            RangeEncoder,
            Vec<f32>,
            Vec<f32>,
            [Vec<i32>; 6],
            Vec<(usize, usize, usize, usize)>,
            Vec<Option<cfl::CflChoice>>,
        );
        let mut adopted: Option<Adopted> = None;
        let best = if let Some(k) = replayed {
            k as usize
        } else if in_frame && luma_part(r0, c0, 12) != NONE {
            let lambda = leaf::part_lambda(t.sb_qstep, self.subsampled_part_lambda_c());
            let (cy0, cx0) = ((r0 * 4) >> t.ss.1, (c0 * 4) >> t.ss.0);
            let (sw, sh) = (64 >> t.ss.0, 64 >> t.ss.1);
            let save = |pl: &[f32]| -> Vec<f32> {
                rect_rows(pl, t.pcw, cy0, cx0, sw, sh)
                    .flatten()
                    .copied()
                    .collect()
            };
            let (ru0, rv0) = (save(&*t.recu), save(&*t.recv));
            let span = |g: &[i32], s: usize| g[s..(s + 16).min(g.len())].to_vec();
            let ctx0 = [
                span(t.u_above, c0),
                span(t.v_above, c0),
                span(t.cfl_above, c0),
                span(t.u_left, r0),
                span(t.v_left, r0),
                span(t.cfl_left, r0),
            ];
            let n_ch = t.chroma_leaves.len();
            let bits0 = t.enc.tell_bits();
            let mut best = (f32::INFINITY, 0usize);
            let mut best_rel = f32::INFINITY;
            // Plan 2 (tall halves) is never picked: with the usual VERT luma root
            // the chroma root is inferred VERT, which makes it plan 1.
            let sq_plan = partition::luma_has_rects(t.luma_leaves);
            let mut te = t.enc.trial_clone();
            let capture = t.decide_mode.is_capture();
            // A plan whose chroma ops equal an earlier plan's codes identically (and
            // ties keep the earlier plan), so only distinct trees are trialled.
            let (above0, left0) = (t.enc.sdp_above_c.clone(), t.enc.sdp_left_c);
            let (row, col, ss) = (t.row, t.col, t.ss);
            let ops_of = |k: usize| {
                let mut above_c = above0.clone();
                let mut left_c = left0;
                partition::sdp_chroma_ops(
                    row,
                    col,
                    mi_rows,
                    mi_cols,
                    ss,
                    &mut above_c,
                    &mut left_c,
                    &luma_part,
                    &|r, c, b| plan(k, r, c, b),
                )
            };
            let mut tried: Vec<(Vec<partition::ChromaOp>, bool)> = Vec::new();
            for k in [0usize, 1, 3, 4, 5, 6] {
                if k == 6 && !sq_plan {
                    continue;
                }
                let ops = ops_of(k);
                if tried.contains(&ops) {
                    continue;
                }
                tried.push(ops);
                t.enc.trial_clone_into(&mut te);
                let mut trial_rec = replay::DecisionRecord::default();
                let mut trial_mode = if capture {
                    replay::DecideMode::Capture(&mut trial_rec)
                } else {
                    replay::DecideMode::Off
                };
                let mut tt = ChromaTree420 {
                    enc: &mut te,
                    decide_mode: &mut trial_mode,
                    recy: t.recy,
                    recu: &mut *t.recu,
                    recv: &mut *t.recv,
                    up: t.up,
                    vp: t.vp,
                    pw: t.pw,
                    pcw: t.pcw,
                    width: t.width,
                    height: t.height,
                    u_above: &mut *t.u_above,
                    v_above: &mut *t.v_above,
                    u_left: &mut *t.u_left,
                    v_left: &mut *t.v_left,
                    cfl_above: &mut *t.cfl_above,
                    cfl_left: &mut *t.cfl_left,
                    luma_leaves: t.luma_leaves,
                    chroma_leaves: &mut *t.chroma_leaves,
                    luma_midx: t.luma_midx,
                    row: t.row,
                    col: t.col,
                    tmr: t.tmr,
                    tmc: t.tmc,
                    qc: t.qc,
                    neutral: t.neutral,
                    sb_qstep: t.sb_qstep,
                    mhccp_bounds: t.mhccp_bounds,
                    ss: t.ss,
                };
                // The abort test measures bits from this trial's start (the plan
                // decision keeps its own formula), against the best plan's cost on
                // the same footing.
                let tb0 = tt.enc.tell_bits();
                let bound = (best_rel.is_finite()).then_some((best_rel, lambda, tb0));
                let complete =
                    self.code_chroma_plan_420(&mut tt, &|r, c, b| plan(k, r, c, b), bound);
                let sse = |rec: &[f32], src: &[f32]| -> f32 {
                    if crate::av2::helpers::wsse_on() {
                        let o = cy0 * t.pcw + cx0;
                        return crate::av2::helpers::dist_rounded(
                            &src[o..],
                            t.pcw,
                            &rec[o..],
                            t.pcw,
                            sw,
                            sh,
                        );
                    }
                    rect_rows(rec, t.pcw, cy0, cx0, sw, sh)
                        .zip(rect_rows(src, t.pcw, cy0, cx0, sw, sh))
                        .map(|(a, b)| {
                            a.iter()
                                .zip(b)
                                .map(|(&x, &y)| (x.round() - y).powi(2))
                                .sum::<f32>()
                        })
                        .sum()
                };
                if complete {
                    let d = sse(&*t.recu, t.up) + sse(&*t.recv, t.vp);
                    let j = d + lambda * (te.tell_bits() - bits0);
                    if j < best.0 {
                        best = (j, k);
                        best_rel = d + lambda * (te.tell_bits() - tb0);
                        let mut enc = adopted.take().map_or_else(|| t.enc.trial_clone(), |a| a.0);
                        std::mem::swap(&mut enc, &mut te);
                        adopted = Some((
                            enc,
                            save(&*t.recu),
                            save(&*t.recv),
                            [
                                span(t.u_above, c0),
                                span(t.v_above, c0),
                                span(t.cfl_above, c0),
                                span(t.u_left, r0),
                                span(t.v_left, r0),
                                span(t.cfl_left, r0),
                            ],
                            t.chroma_leaves[n_ch..].to_vec(),
                            trial_rec.take_mhccp(),
                        ));
                    }
                }
                for (dst, src) in
                    rect_rows_mut(t.recu, t.pcw, cy0, cx0, sw, sh).zip(ru0.chunks_exact(sw))
                {
                    dst.copy_from_slice(src);
                }
                for (dst, src) in
                    rect_rows_mut(t.recv, t.pcw, cy0, cx0, sw, sh).zip(rv0.chunks_exact(sw))
                {
                    dst.copy_from_slice(src);
                }
                let put = |g: &mut [i32], s: usize, v: &[i32]| g[s..s + v.len()].copy_from_slice(v);
                put(t.u_above, c0, &ctx0[0]);
                put(t.v_above, c0, &ctx0[1]);
                put(t.cfl_above, c0, &ctx0[2]);
                put(t.u_left, r0, &ctx0[3]);
                put(t.v_left, r0, &ctx0[4]);
                put(t.cfl_left, r0, &ctx0[5]);
                t.chroma_leaves.truncate(n_ch);
            }
            best.1
        } else {
            0
        };
        t.decide_mode.capture_chroma_plan(best as u8);
        match adopted {
            Some((enc, ru, rv, ctx, leaves, mhccp)) => {
                t.enc.adopt_trial(enc);
                let (cy0, cx0) = ((r0 * 4) >> t.ss.1, (c0 * 4) >> t.ss.0);
                let sw = 64 >> t.ss.0;
                for (pl, saved) in [(&mut *t.recu, &ru), (&mut *t.recv, &rv)] {
                    for (dst, src) in rect_rows_mut(pl, t.pcw, cy0, cx0, sw, 64 >> t.ss.1)
                        .zip(saved.chunks_exact(sw))
                    {
                        dst.copy_from_slice(src);
                    }
                }
                let put = |g: &mut [i32], s: usize, v: &[i32]| g[s..s + v.len()].copy_from_slice(v);
                put(t.u_above, c0, &ctx[0]);
                put(t.v_above, c0, &ctx[1]);
                put(t.cfl_above, c0, &ctx[2]);
                put(t.u_left, r0, &ctx[3]);
                put(t.v_left, r0, &ctx[4]);
                put(t.cfl_left, r0, &ctx[5]);
                t.chroma_leaves.extend(leaves);
                for m in &mhccp {
                    t.decide_mode.capture_mhccp(m);
                }
            }
            None => {
                self.code_chroma_plan_420(t, &|r, c, b| plan(best, r, c, b), None);
            }
        }
    }

    /// Code the chroma tree with `want` choosing the signalled partitions.
    /// `abort`: `(bound, lambda, bits0)` for a plan trial — stop (returning
    /// false) once the running cost, coded-leaf SSE + lambda * (bits - bits0),
    /// passes `bound`: the cost only grows, so such a plan cannot win.
    fn code_chroma_plan_420(
        &self,
        t: &mut ChromaTree420<'_, '_>,
        want: &dyn Fn(usize, usize, usize) -> u8,
        abort: Option<(f32, f32, f32)>,
    ) -> bool {
        let (mi_rows, mi_cols) = (t.tmr as usize, t.tmc as usize);
        let leaves = t.luma_leaves;
        let luma_part = |r: usize, c: usize, b: usize| {
            partition::luma_part_from_leaves(leaves, r, c, b, mi_rows, mi_cols)
        };
        let mut above_c = std::mem::take(&mut t.enc.sdp_above_c);
        let mut left_c = t.enc.sdp_left_c;
        let (ops, cfl_off) = sdp_chroma_ops(
            t.row,
            t.col,
            mi_rows,
            mi_cols,
            t.ss,
            &mut above_c,
            &mut left_c,
            &luma_part,
            want,
        );
        t.enc.sdp_above_c = above_c;
        t.enc.sdp_left_c = left_c;
        // Chroma top-right / bottom-left availability follows the chroma tree's own
        // coded mask (in luma mi units of the chroma blocks).
        t.enc.sb_coded = [0u8; 256];
        let mut sse_so_far = 0f32;
        let mut done = true;
        for op in ops {
            match op {
                ChromaOp::Split { ctx, bit } => t.enc.bool_do_split_c(ctx, bit),
                ChromaOp::Dir { ctx, bit } => t.enc.bool_rect_type_c(ctx, bit),
                ChromaOp::Leaf {
                    mi_row,
                    mi_col,
                    bw_mi,
                    bh_mi,
                } => {
                    self.chroma_leaf_420(t, mi_row, mi_col, bw_mi, bh_mi, !cfl_off);
                    if let Some((bound, lambda, bits0)) = abort {
                        let (ssx, ssy) = t.ss;
                        let (cy, cx) = ((mi_row * 4) >> ssy, (mi_col * 4) >> ssx);
                        let (cw, ch) = ((bw_mi * 4) >> ssx, (bh_mi * 4) >> ssy);
                        let sse = |rec: &[f32], src: &[f32]| -> f32 {
                            if crate::av2::helpers::wsse_on() {
                                let o = cy * t.pcw + cx;
                                return crate::av2::helpers::dist_rounded(
                                    &src[o..],
                                    t.pcw,
                                    &rec[o..],
                                    t.pcw,
                                    cw,
                                    ch,
                                );
                            }
                            rect_rows(rec, t.pcw, cy, cx, cw, ch)
                                .zip(rect_rows(src, t.pcw, cy, cx, cw, ch))
                                .map(|(a, b)| {
                                    a.iter()
                                        .zip(b)
                                        .map(|(&x, &y)| (x.round() - y).powi(2))
                                        .sum::<f32>()
                                })
                                .sum()
                        };
                        sse_so_far += sse(&*t.recu, t.up) + sse(&*t.recv, t.vp);
                        let j = sse_so_far + lambda * (t.enc.tell_bits() - bits0);
                        // Margin: the full cost sums the same terms in another order.
                        if j > bound + bound.abs() * 1e-4 + 1.0 {
                            done = false;
                            break;
                        }
                    }
                }
            }
        }
        done
    }

    fn chroma_leaf_420(
        &self,
        t: &mut ChromaTree420<'_, '_>,
        r: usize,
        c: usize,
        bw: usize,
        bh: usize,
        cfl_ok: bool,
    ) {
        let (ssx, ssy) = t.ss;
        let (cw, ch) = ((bw * 4) >> ssx, (bh * 4) >> ssy);
        let (cy, cx) = ((r * 4) >> ssy, (c * 4) >> ssx);
        let any = |g: &[i32], s: usize, n: usize| {
            g[s..(s + n).min(g.len())].iter().any(|&x| x != 0) as i32
        };
        let ua = if r > 0 { any(t.u_above, c, bw) } else { 0 };
        let ul = if c > 0 { any(t.u_left, r, bh) } else { 0 };
        let va = if r > 0 { any(t.v_above, c, bw) } else { 0 };
        let vl = if c > 0 { any(t.v_left, r, bh) } else { 0 };
        let cfl_a = if r > 0 { t.cfl_above[c] } else { 0 };
        let cfl_l = if c > 0 { t.cfl_left[r] } else { 0 };
        let enc = &mut *t.enc;
        enc.cfl_ctx = (cfl_a + cfl_l) as usize;
        enc.cfl_use = false;
        enc.cfl_signaled = false;
        enc.mhccp_use = false;
        enc.uv_mode = 0;
        enc.cur_bw4 = bw;
        enc.cur_bh4 = bh;
        let luma_dir = t.luma_midx[(r & 15) * 16 + (c & 15)] != 0xff;
        let bd = self.bit_depth as i32;
        let lambda = leaf::part_lambda(t.sb_qstep, self.subsampled_part_lambda_c());
        let (u_nz, v_nz) = if t.ss == (1, 1) && cw == 4 && ch == 4 {
            self.chroma4_leaf_420(t, r, c, cfl_ok, luma_dir, (ua, ul, va, vl))
        } else {
            let spec = chroma_spec(&self.bases, cw, ch, t.qc);
            let mhccp = if cfl_ok {
                chroma422::mhccp_decide_leaf(
                    t.enc,
                    t.decide_mode,
                    &chroma422::ChromaLeafPlanes {
                        reconstructed_luma: t.recy,
                        reconstructed_u: &*t.recu,
                        reconstructed_v: &*t.recv,
                        source_u: t.up,
                        source_v: t.vp,
                        luma_stride: t.pw,
                        chroma_stride: t.pcw,
                    },
                    &chroma422::ChromaLeafGeometry {
                        bounds: t.mhccp_bounds,
                        luma_y: r * 4,
                        luma_x: c * 4,
                        chroma_y: cy,
                        chroma_x: cx,
                        width: cw,
                        height: ch,
                        subsample_x: ssx == 1,
                        subsample_y: ssy == 1,
                        have_top: r > 0,
                        have_left: c > 0,
                    },
                    &chroma422::ChromaLeafRd {
                        neutral: t.neutral,
                        basis: spec.basis,
                        scan: spec.scan,
                        qstep: t.sb_qstep,
                        lambda,
                        bit_depth: bd,
                    },
                )
            } else {
                None
            };
            // Candidate predictors scored on the real TU coding (chroma RDOQ, DCT):
            // DC, the MHCCP fit, classic CfL and, at 32x32 (where the derived tx
            // type is DCT), the uv intra modes.
            let n = cw * ch;
            let cscale = spec.basis.qstep as f32 / t.sb_qstep as f32;
            let rq = self.tune.chroma_rdoq_lambda;
            let src = |p: &[f32]| -> Vec<f32> {
                rect_rows(p, t.pcw, cy, cx, cw, ch)
                    .flatten()
                    .copied()
                    .collect()
            };
            let (su, sv) = (src(t.up), src(t.vp));
            let lev_scratch = std::cell::RefCell::new(Vec::new());
            let score = |pu: &[i32], pv: &[i32], bits: f32| -> f32 {
                let mut j = lambda * bits;
                // Chroma blocks reach 64x64 at 4:4:4.
                let (mut resid, mut rec) = ([0f32; 4096], [0f32; 4096]);
                for (s, p, voff) in [(&su, pu, 0usize), (&sv, pv, 4usize)] {
                    for ((r, &a), &b) in resid.iter_mut().zip(s.iter()).zip(p) {
                        *r = (a - b as f32) * cscale;
                    }
                    let lev = &mut *lev_scratch.borrow_mut();
                    chroma422::project_chroma_rdoq_into(
                        spec.basis,
                        &resid[..n],
                        spec.scan,
                        t.qc,
                        n,
                        voff,
                        rq,
                        lev,
                    );
                    let rec = &mut rec[..n];
                    itx422::reconstruct_chroma_pred_tx_into(
                        p, lev, t.sb_qstep, spec.scan, cw, ch, bd, 0, rec,
                    );
                    let d = crate::av2::helpers::dist_rounded(s, cw, rec, cw, cw, ch);
                    j += d + lambda * coeff_rate_f32(lev);
                }
                j
            };
            let crect = |pl: &[f32]| {
                dc_pred_rect_bounded(
                    pl,
                    &BoundedIntraRect {
                        stride: t.pcw,
                        y: cy,
                        x: cx,
                        width: cw,
                        height: ch,
                        frame_width: t.mhccp_bounds.chroma_width,
                        frame_height: t.mhccp_bounds.chroma_height,
                    },
                    t.neutral,
                    bd,
                )
                .round() as i32
            };
            let (dcu, dcv) = (crect(&*t.recu), crect(&*t.recv));
            // 0 = DC, 1 = MHCCP, 2 = CfL, 3 = uv mode.
            let mut best = (score(&vec![dcu; n], &vec![dcv; n], 0.0), 0u8);
            if let Some(m) = &mhccp {
                let jm = score(&m.pred_u, &m.pred_v, 2.0);
                if jm < best.0 {
                    best = (jm, 1);
                }
            }
            let mut cfl_c: Option<cfl::CflChoice> = None;
            if cfl_ok && t.enc.cfl {
                let (lw, lh) = (cw << ssx, ch << ssy);
                let luma: Vec<i32> = rect_rows(t.recy, t.pw, r * 4, c * 4, lw, lh)
                    .flatten()
                    .map(|&v| v.round() as i32)
                    .collect();
                let q3 = cfl::subsample_luma_q3(&luma, lw, cw, ch, ssx == 1, ssy == 1);
                let avg_l = cfl::cfl_avg_l(
                    t.recy,
                    t.pw,
                    r * 4,
                    c * 4,
                    cw,
                    ch,
                    ssx == 1,
                    ssy == 1,
                    bd,
                    (t.width, t.height),
                );
                let ac: Vec<i32> = q3.iter().map(|&q| q - avg_l).collect();
                let to_i =
                    |v: &[f32]| -> Vec<i32> { v.iter().map(|&x| x.round() as i32).collect() };
                // CfL on a chroma TX wider or taller than 32 predicts from the
                // subsampled DC (AVM highbd_dc_predictor_subsampled).
                let (cdu, cdv) = if cw.max(ch) > 32 {
                    let sub = |pl: &[f32]| {
                        dc_pred_rect_subsampled_bounded(
                            pl,
                            &BoundedIntraRect {
                                stride: t.pcw,
                                y: cy,
                                x: cx,
                                width: cw,
                                height: ch,
                                frame_width: t.mhccp_bounds.chroma_width,
                                frame_height: t.mhccp_bounds.chroma_height,
                            },
                            t.neutral,
                            bd,
                        )
                        .round() as i32
                    };
                    (sub(&*t.recu), sub(&*t.recv))
                } else {
                    (dcu, dcv)
                };
                if let Some(cc) = cfl::cfl_candidate(&to_i(&su), &to_i(&sv), &ac, cdu, cdv, bd) {
                    let bits = 2.0
                        + 3.0
                        + if cc.sign_u != 0 { 3.0 } else { 0.0 }
                        + if cc.sign_v != 0 { 3.0 } else { 0.0 };
                    let jc = score(&cc.pred_u, &cc.pred_v, bits);
                    if jc < best.0 {
                        best = (jc, 2);
                    }
                    cfl_c = Some(cc);
                }
            }
            let mut uv_pred: Option<(Vec<f32>, Vec<f32>)> = None;
            let mut uv_m = 0usize;
            // 32x32 chroma (derived tx type DCT), 4:2:0 only.
            if t.ss == (1, 1) && cw == 32 && ch == 32 && self.tune.chroma_mode_search {
                let (tr, bl) = leaf::square_edge_px(
                    &leaf::SquareLumaBlock {
                        stride: t.pw,
                        frame_width: t.width,
                        frame_height: t.height,
                        y: r * 4,
                        x: c * 4,
                        size: cw << ssx,
                    },
                    &t.enc.sb_coded,
                    true,
                    true,
                );
                let modes: &[usize] = if luma_dir {
                    &[1, 2, 3, 4]
                } else {
                    &[1, 2, 3, 4, 5, 6]
                };
                for &m in modes {
                    let pred = |rec: &[f32]| {
                        chroma422::predict_chroma_sq(
                            rec,
                            t.pcw,
                            cy,
                            cx,
                            cw,
                            m,
                            t.neutral,
                            t.mhccp_bounds.chroma_width,
                            t.mhccp_bounds.chroma_height,
                            Some((tr >> ssx, bl >> ssy)),
                        )
                    };
                    let (pu, pv) = (pred(&*t.recu), pred(&*t.recv));
                    let ri =
                        |v: &[f32]| -> Vec<i32> { v.iter().map(|&x| (x + 0.5) as i32).collect() };
                    let ju = score(&ri(&pu), &ri(&pv), 2.0);
                    if ju < best.0 {
                        best = (ju, 3);
                        uv_m = m;
                        uv_pred = Some((pu, pv));
                    }
                }
            }
            let enc = &mut *t.enc;
            let choice = match best.1 {
                1 => mhccp,
                2 => {
                    let cc = cfl_c.expect("CfL candidate");
                    enc.cfl_use = true;
                    enc.mhccp_use = false;
                    enc.cfl_js = cc.js;
                    enc.cfl_mag_u = cc.mag_u;
                    enc.cfl_mag_v = cc.mag_v;
                    enc.cfl_ctx_u = cc.ctx_u;
                    enc.cfl_ctx_v = cc.ctx_v;
                    Some(cc)
                }
                _ => {
                    enc.cfl_use = false;
                    enc.mhccp_use = false;
                    enc.uv_mode = uv_m;
                    None
                }
            };
            coder::encode_chroma_tree_mode(t.enc, bw, bh, cfl_ok, luma_dir);
            if t.ss == (0, 0) && cw.max(ch) == 64 && cw.min(ch) >= 32 {
                // 4:4:4 TX_64X64 / 64X32 / 32X64 chroma, as the shared-tree leaves code it.
                let (pu, pv): (Vec<i32>, Vec<i32>) = match &choice {
                    Some(cc) => (cc.pred_u.clone(), cc.pred_v.clone()),
                    None => (vec![dcu; n], vec![dcv; n]),
                };
                let code = |s: &[f32], p: &[i32], voff: usize| {
                    let resid: Vec<f32> = s
                        .iter()
                        .zip(p)
                        .map(|(&a, &b)| (a - b as f32) * cscale)
                        .collect();
                    let lev = chroma422::project_chroma_rdoq(
                        spec.basis, &resid, spec.scan, t.qc, 1024, voff, rq,
                    );
                    let rec =
                        itx422::reconstruct_chroma_cfl(p, &lev, t.sb_qstep, spec.scan, cw, ch, bd);
                    (lev, rec)
                };
                let (levu, recu_b) = code(&su, &pu, 0);
                let (levv, recv_b) = code(&sv, &pv, 4);
                put_block_rect(t.recu, t.pcw, cy, cx, cw, ch, &recu_b);
                put_block_rect(t.recv, t.pcw, cy, cx, cw, ch, &recv_b);
                let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                coder::encode_chroma_block(t.enc, &uc, (6 + ua + ul) as u32, true);
                let u_nz = uc.iter().any(|&(_, l)| l != 0);
                coder::encode_chroma_block(t.enc, &vc, (6 * u_nz as i32 + va + vl) as u32, false);
                (u_nz, vc.iter().any(|&(_, l)| l != 0))
            } else {
                code_422_chroma_tu(
                    t.enc,
                    ChromaPlanes {
                        rec_u: &mut *t.recu,
                        rec_v: &mut *t.recv,
                        src_u: t.up,
                        src_v: t.vp,
                        stride: t.pcw,
                        coded_width: t.mhccp_bounds.chroma_width,
                        coded_height: t.mhccp_bounds.chroma_height,
                    },
                    &spec,
                    QuantCtx {
                        qc: t.qc,
                        neutral: t.neutral,
                        qstep: t.sb_qstep,
                        rdoq_lambda: self.tune.chroma_rdoq_lambda,
                    },
                    ChromaNeighbors { ua, ul, va, vl },
                    &ChromaTuInput {
                        y: cy,
                        x: cx,
                        bit_depth: bd,
                        cfl: choice.as_ref(),
                        mode_predictors: uv_pred.as_ref().map(|(u, v)| (&u[..], &v[..])),
                    },
                )
            }
        };
        t.enc.uv_mode = 0;
        t.chroma_leaves.push((r, c, bw, bh));
        let cfl_used = t.enc.cfl_signaled as i32;
        for cc in c..c + bw {
            t.u_above[cc] = u_nz as i32;
            t.v_above[cc] = v_nz as i32;
            t.cfl_above[cc] = cfl_used;
        }
        for rr in r..r + bh {
            t.u_left[rr] = u_nz as i32;
            t.v_left[rr] = v_nz as i32;
            t.cfl_left[rr] = cfl_used;
        }
        for rr in (r & 15)..((r & 15) + bh).min(16) {
            for cc in (c & 15)..((c & 15) + bw).min(16) {
                t.enc.sb_coded[rr * 16 + cc] = 1;
            }
        }
    }

    /// 4x4 chroma leaf (luma-coord 8x8): DC, the non-directional / V / H uv modes
    /// (their derived ADST tx types) or classic CfL, each scored on the real 4x4
    /// coding with chroma RDOQ; then the mode syntax and the U, V TUs.
    fn chroma4_leaf_420(
        &self,
        t: &mut ChromaTree420<'_, '_>,
        r: usize,
        c: usize,
        cfl_ok: bool,
        luma_dir: bool,
        (ua, ul, va, vl): (i32, i32, i32, i32),
    ) -> (bool, bool) {
        use crate::av2::coder::{SCAN4X4_LOSSY, SCAN4X4_LOSSY_PACKED, encode_chroma_tu4_scan};
        use crate::av2::itx::{Tx1d, TxClass, txtp};
        let bases = &self.bases;
        let bd = self.bit_depth as i32;
        let (cy, cx) = (r * 2, c * 2);
        let (sb_y, sb_x) = (r * 4, c * 4);
        let pcw = t.pcw;
        let sb_qstep = t.sb_qstep;
        let qc = t.qc;
        let neutral = t.neutral;
        let scale4 = bases.c4x4.qstep as f32 / sb_qstep as f32;
        let dc_u = dc_pred_rect(t.recu, pcw, cy, cx, 4, 4, neutral, bd);
        let dc_v = dc_pred_rect(t.recv, pcw, cy, cx, 4, 4, neutral, bd);
        let src4 = |plane: &[f32]| -> [f32; 16] {
            let mut out = [0f32; 16];
            for (d, &v) in out
                .iter_mut()
                .zip(rect_rows(plane, pcw, cy, cx, 4, 4).flatten())
            {
                *d = v;
            }
            out
        };
        let (su4, sv4) = (src4(t.up), src4(t.vp));
        let rq4 = self.tune.chroma_rdoq_lambda;
        let uv_tx = |m: usize| -> usize {
            match m {
                1 | 4 => 1,
                2 | 5 => 2,
                3 | 6 => 3,
                _ => 0,
            }
        };
        let tx4_basis = |col_adst: bool, row_adst: bool| {
            let mut b = bases.c4x4.clone();
            b.fwd = crate::av2::fdct::FwdKind::Tx4 { col_adst, row_adst };
            b
        };
        let bases4 = [
            bases.c4x4.clone(),
            tx4_basis(true, true),
            tx4_basis(true, false),
            tx4_basis(false, true),
        ];
        let txtp4 = [
            0,
            txtp(Tx1d::Adst, Tx1d::Adst, TxClass::TwoD),
            txtp(Tx1d::Dct, Tx1d::Adst, TxClass::TwoD),
            txtp(Tx1d::Adst, Tx1d::Dct, TxClass::TwoD),
        ];
        let code4 = |src: &[f32], pred: &[i32], voff: usize, ti: usize, lev: &mut Vec<f32>| {
            let mut resid = [0f32; 16];
            for ((r, &s), &q) in resid.iter_mut().zip(src).zip(pred) {
                *r = (s - q as f32) * scale4;
            }
            chroma422::project_chroma_rdoq_into(
                &bases4[ti],
                &resid,
                &SCAN4X4_LOSSY_PACKED,
                qc,
                16,
                voff,
                rq4,
                lev,
            );
            let mut rec = [0f32; 16];
            itx422::reconstruct_chroma_pred_tx_into(
                pred,
                lev,
                sb_qstep,
                &SCAN4X4_LOSSY_PACKED,
                4,
                4,
                bd,
                txtp4[ti],
                &mut rec,
            );
            rec
        };
        let lambda4 = leaf::part_lambda(sb_qstep, self.subsampled_part_lambda_c());
        let lev_scratch = std::cell::RefCell::new(Vec::new());
        let j = |preds: [&[i32]; 2], extra_bits: f32, ti: usize| -> f32 {
            let mut j = lambda4 * extra_bits;
            for (src, pred, voff) in [(&su4, preds[0], 0), (&sv4, preds[1], 4)] {
                let lev = &mut *lev_scratch.borrow_mut();
                let rec = code4(src, pred, voff, ti, lev);
                let d = crate::av2::helpers::dist_rounded(&src[..], 4, &rec, 4, 4, 4);
                j += crate::av2::rd::rd_cost(d, coeff_rate_f32(lev), lambda4);
            }
            j
        };
        let (dcu, dcv) = (dc_u.round() as i32, dc_v.round() as i32);
        let enc = &mut *t.enc;
        let mut pred_uv: ([i32; 16], [i32; 16]) = ([dcu; 16], [dcv; 16]);
        let mut tx_i = 0usize;
        let mut best_j = j([&pred_uv.0, &pred_uv.1], 0.0, 0);
        if self.tune.chroma_mode_search {
            let (tr, bl) = leaf::square_edge_px(
                &leaf::SquareLumaBlock {
                    stride: t.pw,
                    frame_width: t.width,
                    frame_height: t.height,
                    y: sb_y,
                    x: sb_x,
                    size: 8,
                },
                &enc.sb_coded,
                true,
                true,
            );
            let edges4 = (tr / 2, bl / 2);
            // Under a directional luma block a V / H uv mode would be signalled
            // relative to the luma direction; keep the non-directional set there.
            let modes: &[usize] = match (self.speed.reduced_modes(), luma_dir) {
                (_, true) => &[1, 2, 3, 4],
                (true, false) => &[1, 4, 5, 6],
                (false, false) => &[1, 2, 3, 4, 5, 6],
            };
            let pred4 = |m: usize, rec: &[f32]| -> [i32; 16] {
                let mut pf = [0f32; 16];
                chroma422::predict_chroma_sq_into(
                    rec,
                    pcw,
                    cy,
                    cx,
                    4,
                    m,
                    neutral,
                    t.mhccp_bounds.chroma_width,
                    t.mhccp_bounds.chroma_height,
                    Some(edges4),
                    &mut pf,
                );
                pf.map(|p| (p + 0.5) as i32)
            };
            for &m in modes {
                let pred = |rec: &[f32]| pred4(m, rec);
                let (pu, pv) = (pred(&*t.recu), pred(&*t.recv));
                let cj = j([&pu, &pv], 2.0, uv_tx(m));
                if cj < best_j {
                    best_j = cj;
                    enc.uv_mode = m;
                    tx_i = uv_tx(m);
                    pred_uv = (pu, pv);
                }
            }
        }
        if enc.cfl && cfl_ok {
            let luma: Vec<i32> = rect_rows(t.recy, t.pw, sb_y, sb_x, 8, 8)
                .flatten()
                .map(|&v| v.round() as i32)
                .collect();
            let luma_q3 = cfl::subsample_luma_q3(&luma, 8, 4, 4, true, true);
            let avg_l = cfl::cfl_avg_l(
                t.recy,
                t.pw,
                sb_y,
                sb_x,
                4,
                4,
                true,
                true,
                bd,
                (t.width, t.height),
            );
            let ac: Vec<i32> = luma_q3.iter().map(|&q| q - avg_l).collect();
            let to_i = |v: &[f32]| -> Vec<i32> { v.iter().map(|&x| x.round() as i32).collect() };
            if let Some(cc) = cfl::cfl_candidate(&to_i(&su4), &to_i(&sv4), &ac, dcu, dcv, bd) {
                let alpha_bits = 2.0
                    + 3.0
                    + if cc.sign_u != 0 { 3.0 } else { 0.0 }
                    + if cc.sign_v != 0 { 3.0 } else { 0.0 };
                if j([&cc.pred_u, &cc.pred_v], alpha_bits, 0) < best_j {
                    enc.cfl_use = true;
                    enc.cfl_js = cc.js;
                    enc.cfl_mag_u = cc.mag_u;
                    enc.cfl_mag_v = cc.mag_v;
                    enc.cfl_ctx_u = cc.ctx_u;
                    enc.cfl_ctx_v = cc.ctx_v;
                    enc.uv_mode = 0;
                    tx_i = 0;
                    pred_uv.0.copy_from_slice(&cc.pred_u);
                    pred_uv.1.copy_from_slice(&cc.pred_v);
                }
            }
        }
        coder::encode_chroma_tree_mode(enc, 2, 2, cfl_ok, luma_dir);
        enc.uv_mode = 0;
        let mut levu = Vec::new();
        let recu4 = code4(&su4, &pred_uv.0, 0, tx_i, &mut levu);
        put_block_rect(t.recu, pcw, cy, cx, 4, 4, &recu4);
        let uc = levels_to_coeffs(&levu);
        let u_ctx = (6 + ua + ul) as usize;
        let u_skip = cdfs_qctx::SKIP_TX4_QC[enc.qc][u_ctx] as u32;
        encode_chroma_tu4_scan(enc, &uc, u_skip, false, &SCAN4X4_LOSSY, u_ctx);
        let u_nz = uc.iter().any(|&(_, l)| l != 0);
        let mut levv = Vec::new();
        let recv4 = code4(&sv4, &pred_uv.1, 4, tx_i, &mut levv);
        put_block_rect(t.recv, pcw, cy, cx, 4, 4, &recv4);
        let vc = levels_to_coeffs(&levv);
        let v_ctx = (6 * (u_nz as i32) + va + vl) as usize;
        encode_chroma_tu4_scan(enc, &vc, v_ctx as u32, true, &SCAN4X4_LOSSY, v_ctx);
        (u_nz, vc.iter().any(|&(_, l)| l != 0))
    }
}
