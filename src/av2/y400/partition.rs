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

/// Copy an `w × h` reconstruction rectangle at `(y, x)` out of a plane (row-major).
/// Used to snapshot a walk leaf's luma reconstruction for the decouple record.
fn gather_rect(recy: &[f32], pw: usize, y: usize, x: usize, w: usize, h: usize) -> Vec<f32> {
    let mut v = vec![0f32; w * h];
    for (dst, src) in v.chunks_exact_mut(w).zip(rect_rows(recy, pw, y, x, w, h)) {
        dst.copy_from_slice(src);
    }
    v
}

/// One superblock's parallel-decide output for 4:0:0: the captured record plus the
/// deblock TU rectangles it recorded. Merged in raster order after the wavefront so
/// the serial Replay consumes the exact sequence a serial Capture would produce.
struct WfSlot400 {
    record: replay::DecisionRecord,
    tx: Vec<(usize, usize, usize, usize)>,
}

/// Per-worker reusable scratch for the 4:0:0 wavefront decide: a private luma recon
/// (`ry`, full-res `pw×ph`) plus the emit-side neighbor context arrays. 4:0:0 is
/// luma-only, so there is no chroma recon/halo/context here. Each cell sees the
/// frame-initial context values, reset per cell so the result never depends on
/// which cell a worker handled previously.
#[derive(Default)]
struct WfScratch400 {
    ry: Vec<f32>,
    above: Vec<u8>,
    left: Vec<u8>,
    apctx: Vec<u8>,
    dbq: Vec<u16>,
}

impl WfScratch400 {
    fn ensure(&mut self, pw: usize, ph: usize, tmc: i64, sb_cols: usize, sb_rows: usize) {
        let ny = pw * ph;
        if self.ry.len() != ny {
            self.ry = vec![0f32; ny];
        }
        self.above.resize(pw / 4 + 16, 0x40);
        self.left.resize(ph / 4 + 16, 0x40);
        self.apctx.resize(tmc as usize + 16, 0);
        self.dbq.resize(sb_cols * sb_rows, 0);
    }

    /// Restore the exact frame-initial context state before every independently
    /// decided SB (thread-pool assignment is unstable, so nothing may carry over).
    fn reset_contexts(&mut self) {
        self.above.fill(0x40);
        self.left.fill(0x40);
        self.apctx.fill(0);
        self.dbq.fill(0);
    }
}

thread_local! {
    static WF_SCRATCH_400: std::cell::RefCell<WfScratch400> =
        std::cell::RefCell::new(WfScratch400::default());
}

struct Sb400Decision<'a, 'record> {
    enc: &'a mut RangeEncoder,
    aqs: &'a mut aq::AqState,
    decide_mode: &'a mut replay::DecideMode<'record>,
    yp: &'a [f32],
    recy: &'a mut [f32],
    above: &'a mut [u8],
    left: &'a mut [u8],
    above_pctx: &'a mut [u8],
    left_pctx: &'a mut [u8],
    tx_leaves: &'a mut Vec<(usize, usize, usize, usize)>,
    sb_qidx: &'a mut [u16],
    pw: usize,
    width: usize,
    height: usize,
    sb_cols: usize,
    tmc: i64,
    tmr: i64,
    qc: usize,
    neutral: f32,
    qstep_i: i32,
    rdoq_lambda: f32,
    aq_grid: &'a [aq::AqCell],
    use_grid: bool,
    row: usize,
    col: usize,
}

impl Av2Encoder {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_yuv400_partition(
        &self,
        enc: &mut RangeEncoder,
        luma: LumaPlanes,
        ctx: &PartitionPass,
        nb: PartitionNeighbors,
        // Deblock support: luma leaf rectangles and per-SB settled qindex (see 4:2:0).
        tx_leaves: &mut Vec<(usize, usize, usize, usize)>,
        sb_qidx: &mut [u16],
        // Staged replay: `Capture` logs each walk leaf's luma winner; `Replay` reuses
        // it (skips the leaf mode/tx searches). `Off` = today's inline behavior.
        mut decide_mode: crate::av2::replay::DecideMode<'_>,
    ) {
        let LumaPlanes { rec: recy, src: yp } = luma;
        let &PartitionPass {
            luma_stride: pw,
            width,
            height,
            sb_rows,
            sb_cols,
            tmc,
            tmr,
            quant:
                QuantCtx {
                    qc,
                    neutral,
                    qstep: qstep_i,
                    rdoq_lambda,
                },
            ..
        } = ctx;
        let PartitionNeighbors {
            above,
            left,
            above_pctx,
            left_pctx,
        } = nb;
        // This method is only entered on the native/edge partition path, i.e.
        // `needs_partition == true`. The AQ grid mirrors the serial `per_sb`
        // (whole-64/full-interior) / `current()` (native edge) sequence exactly for
        // this path (see `precompute_grid`), so a wavefront decide reads it instead of
        // the raster-serial accumulator.
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
            0, /* monochrome: no chroma */
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        )
        .with_dark_aq(self.tune.dark_aq);
        let aq_grid = aqs.precompute_grid(yp, pw, width, height, true);

        let fresh_ctx = false;

        // Halo geometry (LUMA-ONLY): intra reads a single reference line + above-right
        // extension of the top row. Copy a generous superset — a 32px perpendicular
        // band + 64px above-right margin — plus the whole-64 RD path's full-plane SSE
        // anchor at (0,0). This is the complete read-set required to isolate a
        // wavefront worker.
        const HALO_BAND: usize = 32;
        const HALO_AR: usize = 64;
        const POISON: f32 = 0.0;
        let (ph_full, _) = (recy.len() / pw, 0usize);
        let halo_mode = false;
        let mut recy_p = if halo_mode {
            vec![0f32; recy.len()]
        } else {
            Vec::new()
        };
        let copy_halo_luma =
            |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize, copy_rd_anchor: bool| {
                let bx = sb_x.saturating_sub(HALO_BAND);
                let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                let by = sb_y.saturating_sub(HALO_BAND);
                for r in by..sb_y {
                    dst[r * pw + bx..r * pw + tx_end]
                        .copy_from_slice(&src[r * pw + bx..r * pw + tx_end]);
                }
                let ly_end = (sb_y + 64).min(ph_full);
                for r in sb_y..ly_end {
                    dst[r * pw + bx..r * pw + sb_x]
                        .copy_from_slice(&src[r * pw + bx..r * pw + sb_x]);
                }
                if copy_rd_anchor && (sb_y != 0 || sb_x != 0) {
                    for r in 0..64.min(ph_full) {
                        dst[r * pw..r * pw + 64].copy_from_slice(&src[r * pw..r * pw + 64]);
                    }
                }
            };
        let copy_own_luma = |dst: &mut [f32], src: &[f32], sb_y: usize, sb_x: usize| {
            let ly_end = (sb_y + 64).min(ph_full);
            for r in sb_y..ly_end {
                dst[r * pw + sb_x..r * pw + sb_x + 64]
                    .copy_from_slice(&src[r * pw + sb_x..r * pw + sb_x + 64]);
            }
        };

        // Wavefront gate: stills only (all 4:0:0 lossy), Capture only. The AQ grid is
        // bit-exact with the serial `per_sb`/`current` sequence for this partition path
        // regardless of base_q (whole-64 accumulates, native edge uses `current()`),
        // so there is no base_q restriction like the 4:2:2 padded band.
        // Default parallelism for stills: fires on any multi-threaded single-tile
        // Capture pass. Suppressed inside per-tile sub-encodes, which already
        // parallelise.
        let wavefront = Self::resolve_threads(self.threads) > 1
            && !replay::in_tile_subencode()
            && !self.video_mode.load(std::sync::atomic::Ordering::Relaxed)
            && matches!(decide_mode, replay::DecideMode::Capture(_));
        if wavefront {
            let enc_qc = enc.qc;
            let enc_dqp = enc.delta_q_present;
            let base_q = self.base_q_idx as i32;
            let nthreads = Self::resolve_threads(self.threads);
            let wy = crate::av2::helpers::PlaneWriter::new(recy, pw);
            // Persistent-pool wavefront: fast/DC-only mode uses the wider top/left
            // schedule; directional search retains the above-right-safe WPP schedule.
            // Worker scratch is allocated once, not once per diagonal.
            let slots: Vec<Option<WfSlot400>> = crate::av2::helpers::par_wavefront_pool(
                nthreads,
                sb_rows,
                sb_cols,
                self.speed.try_directional(),
                |r, c| {
                    let sb_y = r * 64;
                    let sb_x = c * 64;

                    WF_SCRATCH_400.with(|sc| {
                        let s = &mut *sc.borrow_mut();
                        s.ensure(pw, ph_full, tmc, sb_cols, sb_rows);
                        s.reset_contexts();
                        let ly_end = (sb_y + 64).min(ph_full);
                        let bx = sb_x.saturating_sub(HALO_BAND);
                        let tx_end = (sb_x + 64 + HALO_AR).min(pw);
                        let by = sb_y.saturating_sub(HALO_BAND);
                        // The thread-local buffer is REUSED across cells; it must present
                        // decide the same all-zero-except-halo view the serial fresh plane
                        // does. It starts zero (in `ensure`) and each cell RE-ZEROES exactly
                        // what it dirtied (below), so on entry it is all-zero. Copy the halo
                        // (finished top/left neighbors) from the shared recon.
                        // SAFETY: under WPP these regions are earlier-diagonal (finished),
                        // not being written concurrently; own-block writes are disjoint.
                        unsafe {
                            if sb_y != 0 || sb_x != 0 {
                                wy.copy_region_to(&mut s.ry, 0, 0, 64.min(ph_full), 64);
                            }
                            if sb_y > by {
                                wy.copy_region_to(&mut s.ry, by, bx, sb_y - by, tx_end - bx);
                            }
                            if ly_end > sb_y && sb_x > bx {
                                wy.copy_region_to(&mut s.ry, sb_y, bx, ly_end - sb_y, sb_x - bx);
                            }
                        }
                        let mut e = RangeEncoder::new();
                        e.qc = enc_qc;
                        // Decision workers discard their entropy stream. No decision
                        // reads coder state, so avoid allocating/adapting a full CDF
                        // tree and skip range-output work for every superblock.
                        e.sink = true;
                        e.delta_q_present = enc_dqp;
                        let mut a = aq::AqState::new(enc_dqp, base_q, qstep_i, 0.0, 0);
                        let mut rec = replay::DecisionRecord::new();
                        let mut dm = replay::DecideMode::Capture(&mut rec);
                        let mut lpctx = [0u8; 16];
                        let mut cell_tx: Vec<(usize, usize, usize, usize)> = Vec::new();
                        self.decide_sb_400(Sb400Decision {
                            enc: &mut e,
                            aqs: &mut a,
                            decide_mode: &mut dm,
                            yp,
                            recy: &mut s.ry,
                            above: &mut s.above,
                            left: &mut s.left,
                            above_pctx: &mut s.apctx,
                            left_pctx: &mut lpctx,
                            tx_leaves: &mut cell_tx,
                            sb_qidx: &mut s.dbq,
                            pw,
                            width,
                            height,
                            sb_cols,
                            tmc,
                            tmr,
                            qc,
                            neutral,
                            qstep_i,
                            rdoq_lambda,
                            aq_grid: &aq_grid,
                            use_grid: true,
                            row: r,
                            col: c,
                        });
                        // Write this SB's own 64×64 block back to the shared recon.
                        // SAFETY: disjoint from every other worker's own block.
                        let hl = ly_end - sb_y;
                        let mut blk = [0f32; 64 * 64];
                        for rr in 0..hl {
                            blk[rr * 64..rr * 64 + 64].copy_from_slice(
                                &s.ry[(sb_y + rr) * pw + sb_x..(sb_y + rr) * pw + sb_x + 64],
                            );
                        }
                        unsafe { wy.write_block(sb_y, sb_x, hl, 64, &blk[..hl * 64]) };
                        // Restore the buffer to all-zero: re-zero exactly what was dirtied
                        // (halo bands + own block) so the next cell sees all-zero-except-halo.
                        for rr in by..ly_end {
                            s.ry[rr * pw + bx..rr * pw + tx_end].fill(0.0);
                        }
                        if sb_y != 0 || sb_x != 0 {
                            for rr in 0..64.min(ph_full) {
                                s.ry[rr * pw..rr * pw + 64].fill(0.0);
                            }
                        }
                        WfSlot400 {
                            record: rec,
                            tx: cell_tx,
                        }
                    })
                },
            );
            // Merge per-SB records + deblock rectangles in raster order → the exact
            // sequence a serial Capture would have logged, so the serial Replay stays
            // byte-identical.
            if let replay::DecideMode::Capture(rec) = &mut decide_mode {
                for slot in slots.into_iter() {
                    let mut slot = slot.expect("every SB decided");
                    rec.append(slot.record);
                    tx_leaves.append(&mut slot.tx);
                }
            }
            for (i, cell) in aq_grid.iter().enumerate() {
                sb_qidx[i] = cell.qidx as u16;
            }
            return;
        }

        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                if fresh_ctx {
                    above.iter_mut().for_each(|v| *v = 0x40);
                    left.iter_mut().for_each(|v| *v = 0x40);
                    above_pctx.iter_mut().for_each(|v| *v = 0);
                    left_pctx.iter_mut().for_each(|v| *v = 0);
                }
                let sb_y = row * 64;
                let sb_x = col * 64;
                if halo_mode {
                    recy_p.iter_mut().for_each(|v| *v = POISON);
                    copy_halo_luma(&mut recy_p, recy, sb_y, sb_x, true);
                }
                let ry: &mut [f32] = if halo_mode {
                    recy_p.as_mut_slice()
                } else {
                    &mut *recy
                };
                self.decide_sb_400(Sb400Decision {
                    enc,
                    aqs: &mut aqs,
                    decide_mode: &mut decide_mode,
                    yp,
                    recy: ry,
                    above,
                    left,
                    above_pctx,
                    left_pctx,
                    tx_leaves,
                    sb_qidx,
                    pw,
                    width,
                    height,
                    sb_cols,
                    tmc,
                    tmr,
                    qc,
                    neutral,
                    qstep_i,
                    rdoq_lambda,
                    aq_grid: &aq_grid,
                    use_grid: fresh_ctx,
                    row,
                    col,
                });
                if halo_mode {
                    copy_own_luma(recy, &recy_p, sb_y, sb_x);
                }
            }
        }
    }

    /// Decide + emit one 4:0:0 superblock (the extracted body of the `for col`
    /// loop). Behaviour-preserving; byte-identical. Split out so the SB-wavefront
    /// can drive it per cell with a private luma recon + fresh contexts.
    fn decide_sb_400(&self, decision: Sb400Decision<'_, '_>) {
        let Sb400Decision {
            enc,
            aqs,
            decide_mode,
            yp,
            recy,
            above,
            left,
            above_pctx,
            left_pctx,
            tx_leaves,
            sb_qidx,
            pw,
            width,
            height,
            sb_cols,
            tmc,
            tmr,
            qc,
            neutral,
            qstep_i: _qstep_i,
            rdoq_lambda,
            aq_grid,
            use_grid,
            row,
            col,
        } = decision;
        use crate::av2::replay::{DecideMode, Leaf400, Sb400};
        let bases = &self.bases;
        let cell = aq_grid[row * sb_cols + col];
        let whole_sb = col * 64 + 64 <= width && row * 64 + 64 <= height;
        // Whole-64 interior SBs get AQ (the (16,16) leaf below); split edge SBs stay
        // quantization-neutral. delta_q is emitted once per SB.
        let (sb_qstep, sb_resid_scale) = if use_grid {
            enc.delta_q_signaled = if whole_sb { cell.sig } else { 0 };
            (cell.qs, cell.resid_scale)
        } else if whole_sb {
            let (q, r, _, _) = aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height);
            (q, r)
        } else {
            enc.delta_q_signaled = 0;
            let (q, r, _, _) = aqs.current();
            (q, r)
        };
        sb_qidx[row * sb_cols + col] = if use_grid {
            cell.qidx as u16
        } else {
            aqs.current_qidx() as u16
        };
        let (sb_y, sb_x) = (row * 64, col * 64);
        let luma_partition = if whole_sb && self.tune.chroma_split {
            // Replay: reuse the cached partition decision, skipping the ~35%
            // `choose_luma_64x64_partition` search. The probe recon it would have
            // written is overwritten by the real leaf encode below, so skipping is safe.
            let replayed = if let DecideMode::Replay(cur) = &mut *decide_mode {
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
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                        sb: LumaSbSearch {
                            residual_scale: sb_resid_scale,
                            allow_directional: self.speed.try_directional(),
                        },
                        basis16: &bases.luma16x16,
                        basis8: &bases.c8x8,
                        allow_16x16: true,
                        allow_8x8: true,
                    },
                );
                if let DecideMode::Capture(rec) = &mut *decide_mode {
                    rec.push_part(p);
                }
                p
            }
        } else {
            LumaPartitionDecision::default()
        };
        let ops = if whole_sb && luma_partition.split64 {
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
        enc.delta_q_pending = enc.delta_q_present;
        // Staged decouple — the walk pushes exactly one `luma400` entry per SB:
        // `Walk(leaves)` when every leaf shape is replayable ({(16,16),(8,8),(4,4),
        // (2,2)} = all interior shapes), else `Fallback` (replay re-searches, byte-safe
        // because serial replay recon == Off recon there). In Replay we pop that entry
        // and reuse each captured luma winner, skipping the leaf mode/tx search.
        let replay_walk: Option<Vec<Leaf400>> = if let DecideMode::Replay(cur) = &mut *decide_mode {
            match cur.next_sb400() {
                Some(Sb400::Walk(v)) => Some(v.clone()),
                _ => None,
            }
        } else {
            None
        };
        let capturing = matches!(*decide_mode, DecideMode::Capture(_));
        let mut leaf_recs: Vec<Leaf400> = Vec::new();
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
            tx_leaves.push((lmr, lmc, bw_mi, bh_mi));
            let sb_y = lmr * 4;
            let sb_x = lmc * 4;
            let leaf_replay: Option<&Leaf400> = replay_walk.as_ref().and_then(|v| v.get(cap_idx));
            match (bw_mi, bh_mi) {
                (16, 16) => {
                    let (tus, mode_idx) = if let Some(d) = leaf_replay {
                        put_block_rect(recy, pw, sb_y, sb_x, 64, 64, &d.recon_y);
                        let t: [Vec<Coeff>; 4] = std::array::from_fn(|k| d.tus[k].clone());
                        (t, d.mode_idx)
                    } else {
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
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                            &LumaSbSearch {
                                residual_scale: sb_resid_scale,
                                allow_directional: false,
                            },
                        );
                        (tus, mode_idx)
                    };
                    if capturing {
                        leaf_recs.push(Leaf400 {
                            bw_mi: 16,
                            bh_mi: 16,
                            tus: tus.to_vec(),
                            mode_idx,
                            tx_idx: 0,
                            recon_y: gather_rect(recy, pw, sb_y, sb_x, 64, 64),
                        });
                    }
                    cap_idx += 1;
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                    encode_luma_block_split(
                        enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        false,
                        pc,
                    );
                }
                (8, 8) => {
                    let (tu, mode_idx) = if let Some(d) = leaf_replay {
                        put_block(recy, pw, sb_y, sb_x, 32, &d.recon_y);
                        (d.tus[0].clone(), d.mode_idx)
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
                                qstep: sb_qstep,
                                scan: &tables::SCAN,
                                neutral,
                                quant_context: qc,
                                rdoq_lambda: self.tune.rdoq_lambda,
                                speed: self.speed,
                                bit_depth: self.bit_depth as i32,
                            },
                        )
                    };
                    if capturing {
                        leaf_recs.push(Leaf400 {
                            bw_mi: 8,
                            bh_mi: 8,
                            tus: vec![tu.clone()],
                            mode_idx,
                            tx_idx: 0,
                            recon_y: gather_rect(recy, pw, sb_y, sb_x, 32, 32),
                        });
                    }
                    cap_idx += 1;
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
                    encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, false, pc);
                }
                (4, 4) => {
                    // 16x16 corner leaf: tx_type RD over DCT/ADST mixes (ported from 4:4:4).
                    let bd = self.bit_depth as i32;
                    let (tu, tx_idx) = if let Some(d) = leaf_replay {
                        put_block_rect(recy, pw, sb_y, sb_x, 16, 16, &d.recon_y);
                        (d.tus[0].clone(), d.tx_idx)
                    } else {
                        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 16, neutral, bd);
                        let resid = aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred),
                            sb_resid_scale,
                        );
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
                        let lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
                        let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                        let rec_dct =
                            itx422::reconstruct_luma16(&pred_flat, &lev_dct, sb_qstep, &SCAN16, bd);
                        let dist_dct = sse(&rec_dct);
                        let cost_dct = dist_dct + lambda * rate(&lev_dct);
                        let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                        let rec_adst = itx422::reconstruct_luma16_adst(
                            &pred_flat, &lev_adst, sb_qstep, &SCAN16, true, true, bd,
                        );
                        let dist_adst = sse(&rec_adst);
                        let cost_adst =
                            dist_adst + lambda * (rate(&lev_adst) + TX16_TYPE_RATE_DELTA[1]);
                        let lev_ad = bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                        let rec_ad = itx422::reconstruct_luma16_adst(
                            &pred_flat, &lev_ad, sb_qstep, &SCAN16, false, true, bd,
                        );
                        let dist_ad = sse(&rec_ad);
                        let cost_ad = dist_ad + lambda * (rate(&lev_ad) + TX16_TYPE_RATE_DELTA[2]);
                        let lev_da = bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                        let rec_da = itx422::reconstruct_luma16_adst(
                            &pred_flat, &lev_da, sb_qstep, &SCAN16, true, false, bd,
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
                        (levels_to_coeffs(lev), tx_idx)
                    };
                    if capturing {
                        leaf_recs.push(Leaf400 {
                            bw_mi: 4,
                            bh_mi: 4,
                            tus: vec![tu.clone()],
                            mode_idx: 0,
                            tx_idx,
                            recon_y: gather_rect(recy, pw, sb_y, sb_x, 16, 16),
                        });
                    }
                    cap_idx += 1;
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
                    coder::encode_luma_leaf_16x16_full(
                        enc, &tu, skip, dcs, 0, false, pc, 11074, tx_idx,
                    );
                }
                (2, 2) => {
                    // 8x8 luma leaf (TX_8X8), search-free single DCT.
                    let bd = self.bit_depth as i32;
                    let tu: Vec<Coeff> = if let Some(d) = leaf_replay {
                        put_block_rect(recy, pw, sb_y, sb_x, 8, 8, &d.recon_y);
                        d.tus[0].clone()
                    } else {
                        let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 8, neutral, bd);
                        let lev = bases.c8x8.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred),
                                sb_resid_scale,
                            ),
                            0.0,
                            &tables::SCAN8X8,
                        );
                        put_block_rect(
                            recy,
                            pw,
                            sb_y,
                            sb_x,
                            8,
                            8,
                            &itx422::reconstruct_chroma(pred, &lev, sb_qstep, &SCAN8X8, 8, 8, bd),
                        );
                        levels_to_coeffs(&lev)
                    };
                    if capturing {
                        leaf_recs.push(Leaf400 {
                            bw_mi: 2,
                            bh_mi: 2,
                            tus: vec![tu.clone()],
                            mode_idx: 0,
                            tx_idx: 0,
                            recon_y: gather_rect(recy, pw, sb_y, sb_x, 8, 8),
                        });
                    }
                    cap_idx += 1;
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
                    encode_luma_leaf_8x8(
                        enc,
                        &tu,
                        skip,
                        dcs,
                        0,
                        false,
                        pc,
                        3148,
                        Some((&coder::TXTP_EXT8, 0, 6)),
                    );
                }
                (16, 8) => {
                    // Native edge shape — not yet captured; whole SB falls back to a
                    // serial re-search in replay (byte-safe).
                    walk_capturable = false;
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
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
                        },
                    );
                    let (skip2, dcs2) =
                        sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                    encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, false, pc);
                }
                (8, 16) => {
                    walk_capturable = false;
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
                            qstep: sb_qstep,
                            scan: &tables::SCAN,
                            neutral,
                            quant_context: qc,
                            rdoq_lambda: self.tune.rdoq_lambda,
                            speed: self.speed,
                            bit_depth: self.bit_depth as i32,
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
                    encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, false, pc);
                }
                (4, 16) => {
                    walk_capturable = false;
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 16, 64, neutral, self.bit_depth as i32);
                    let lev = bases.luma16x64.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
                            sb_resid_scale,
                        ),
                        0.0,
                        &SCAN16X32,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        16,
                        64,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &SCAN16X32,
                            16,
                            64,
                            self.bit_depth as i32,
                        ),
                    );
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
                    encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, false, pc);
                }
                (16, 4) => {
                    walk_capturable = false;
                    let pred =
                        dc_pred_rect(recy, pw, sb_y, sb_x, 64, 16, neutral, self.bit_depth as i32);
                    let lev = bases.luma64x16.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
                            sb_resid_scale,
                        ),
                        0.0,
                        &SCAN32X16,
                    );
                    put_block_rect(
                        recy,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        16,
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &SCAN32X16,
                            64,
                            16,
                            self.bit_depth as i32,
                        ),
                    );
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
                    encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, false, pc);
                }
                (2, 8) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 32, neutral, bd);
                    let lev = bases.luma8x32.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 8, 32, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(pred, &lev, sb_qstep, &SCAN8X32, 8, 32, bd),
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
                    coder::encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, false, pc);
                }
                (8, 2) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 8, neutral, bd);
                    let lev = bases.luma32x8.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 32, 8, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(pred, &lev, sb_qstep, &SCAN32X8, 32, 8, bd),
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
                    encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, false, pc);
                }
                (8, 4) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
                    let lev = bases.luma32x16.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 32, 16, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(pred, &lev, sb_qstep, &SCAN32X16, 32, 16, bd),
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
                    coder::encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, false, pc);
                }
                (4, 8) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
                    let lev = bases.luma16x32.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 32, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(pred, &lev, sb_qstep, &SCAN16X32, 16, 32, bd),
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
                    encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, false, pc);
                }
                (4, 2) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                    let lev = bases.c16x8.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 16, 8, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &tables::SCAN16X8,
                            16,
                            8,
                            bd,
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
                        4,
                        2,
                    );
                    coder::encode_luma_leaf_rect128(
                        enc,
                        &tu,
                        &LumaLeafRect128Spec {
                            skip_cdf: skip,
                            dc_sign_ctx: dcs,
                            mode_idx: 0,
                            has_chroma: false,
                            width_mi: 4,
                            height_mi: 2,
                            part_cdf: pc,
                            tx_part_cdf: 12348,
                            scan: &tables::SCAN16X8,
                            tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                        },
                    );
                }
                (2, 4) => {
                    walk_capturable = false;
                    let bd = self.bit_depth as i32;
                    let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                    let lev = bases.c8x16.project_scan(
                        &aq::scale_resid(
                            &get_residual_rect(yp, pw, sb_y, sb_x, 8, 16, pred),
                            sb_resid_scale,
                        ),
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
                        &itx422::reconstruct_chroma(
                            pred,
                            &lev,
                            sb_qstep,
                            &tables::SCAN8X16,
                            8,
                            16,
                            bd,
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
                        4,
                    );
                    coder::encode_luma_leaf_rect128(
                        enc,
                        &tu,
                        &coder::LumaLeafRect128Spec {
                            skip_cdf: skip,
                            dc_sign_ctx: dcs,
                            mode_idx: 0,
                            has_chroma: false,
                            width_mi: 2,
                            height_mi: 4,
                            part_cdf: pc,
                            tx_part_cdf: 12348,
                            scan: &tables::SCAN8X16,
                            tx_type_cdf: Some((&coder::TXTP_EXT8, 0, 6)),
                        },
                    );
                }
                other => unreachable!("unsupported native 4:0:0 leaf {:?}", other),
            }
        }
        if let DecideMode::Capture(rec) = &mut *decide_mode {
            rec.push_sb400(if walk_capturable {
                Sb400::Walk(leaf_recs)
            } else {
                Sb400::Fallback
            });
        }
    }
}
