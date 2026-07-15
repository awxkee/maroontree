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

#[derive(Clone, Copy)]
pub(super) struct Core400Frame<'a> {
    pub(super) yp: &'a [f32],
    pub(super) pw: usize,
    pub(super) ph: usize,
    pub(super) width: usize,
    pub(super) height: usize,
}

impl Av2Encoder {
    /// SB-loop core for one monochrome region. The source plane is already padded to
    /// `pw × ph`; header/finish assembly remains in the public API layer.
    pub(super) fn encode_400_core(
        &self,
        frame: Core400Frame<'_>,
        cdef_pre: Option<&cdef_est::CdefDecision>,
        // Staged replay: `Capture` logs the whole-64 luma winner; `Replay` reuses it.
        mut decide_mode: replay::DecideMode<'_>,
    ) -> RangeEncoder {
        let Core400Frame {
            yp,
            pw,
            ph,
            width,
            height,
        } = frame;
        let bases = &self.bases;
        let mut recy = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        // Pass 2: install the per-SB CDEF grid for per-unit index emission.
        if let Some(pre) = cdef_pre {
            enc.cdef_nb = 2;
            enc.cdef_cols = pre.sb_cols;
            enc.cdef_grid = pre.grid.clone();
            enc.cdef_decided = Some(pre.clone());
        }

        let native_mi = lossy_native_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // Same edge fix as 4:2:0: residues {10,12,14} return native (un-padded) mi
        // extents from `lossy_native_mi` but `lossy_needs_partition` is false, so the old
        // code took the fast whole-SB path with a native/padded extent mismatch and
        // desynced the decoder on the partial edge SB (any side ≡ 40/48/56 mod 64). Route
        // every non-64-aligned dimension through the edge-aware partition walk instead.
        let mc_edge = (((width + 7) & !7) / 4) as i64 % 16;
        let mr_edge = (((height + 7) & !7) / 4) as i64 % 16;
        let needs_partition = native_mi.is_some()
            && (lossy_needs_partition(width, height)
                || mc_edge != 0
                || mr_edge != 0
                || self.tune.chroma_split);
        if needs_partition {
            let mut above_pctx = vec![0u8; tmc as usize + 16];
            let mut left_pctx = vec![0u8; 16];
            let mut tx_leaves: Vec<(usize, usize, usize, usize)> = Vec::new();
            let mut db_sb_qidx = vec![self.base_q_idx as u16; sb_rows * sb_cols];
            self.encode_yuv400_partition(
                &mut enc,
                LumaPlanes {
                    rec: &mut recy,
                    src: yp,
                },
                &PartitionPass {
                    luma_stride: pw,
                    chroma_stride: 0,
                    width,
                    height,
                    sb_rows,
                    sb_cols,
                    tmc,
                    tmr,
                    quant: QuantCtx {
                        qc,
                        neutral,
                        qstep: qstep_i,
                        rdoq_lambda: self.tune.rdoq_lambda,
                    },
                },
                PartitionNeighbors {
                    above: &mut above,
                    left: &mut left,
                    above_pctx: &mut above_pctx,
                    left_pctx: &mut left_pctx,
                },
                &mut tx_leaves,
                &mut db_sb_qidx,
                decide_mode,
            );
            // Real deblocking (monochrome, luma only). Leaves recorded during the walk.
            let df_quant = {
                let qs = quant::qstep(self.base_q_idx as u32) as i32;
                ((qs + 4) >> 3) >> 6
            };
            if self.tune.deblock && df_quant >= 1 {
                let eff_dy = if self.tune.db_delta_y == i32::MIN {
                    if df_quant >= 5 { 0 } else { 1 }
                } else {
                    self.tune.db_delta_y
                };
                let mut empty: [f32; 0] = [];
                let mut empty2: [f32; 0] = [];
                deblock::deblock_frame(deblock::FrameDeblock {
                    recy: &mut recy,
                    recu: &mut empty,
                    recv: &mut empty2,
                    luma_stride: pw,
                    chroma_stride: 0,
                    width,
                    height,
                    ssx: 0,
                    ssy: 0,
                    has_chroma: false,
                    chroma_deblock: false,
                    bit_depth: self.bit_depth as u32,
                    delta_y: eff_dy,
                    delta_uv: 0,
                    base_q: self.base_q_idx as u16,
                    sb_cols,
                    sb_qidx: &db_sb_qidx,
                    leaves: &tx_leaves,
                    chroma_leaves: &tx_leaves,
                    skip_leaves: &[],
                    luma_tx_cap: (8, 8),

                    chroma_tx_cap: (8, 8),
                });
            }
            if self
                .capture_recon
                .load(std::sync::atomic::Ordering::Relaxed)
            {
                enc.recon = vec![recy];
            }
            return enc;
        }

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
        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                if enc.cdef_nb >= 2 {
                    enc.cdef_pending = true;
                    enc.cdef_sb_rc = (row, col);
                }
                let (sb_qstep, sb_resid_scale, _, _) =
                    aqs.per_sb(&mut enc, yp, pw, sb_y, sb_x, width, height);
                // Staged replay: capture/replay the whole-64 luma mode-search winner.
                let replay_l = if let replay::DecideMode::Replay(cur) = &mut decide_mode {
                    cur.next_luma420()
                } else {
                    None
                };
                let (tus, mode_idx) = if let Some(l) = replay_l {
                    for r in 0..64 {
                        let off = (sb_y + r) * pw + sb_x;
                        recy[off..off + 64].copy_from_slice(&l.recon_y[r * 64..r * 64 + 64]);
                    }
                    (l.tus, l.mode_idx)
                } else {
                    let (tus, mode_idx, adelta, _) = encode_luma_sb(
                        &mut recy,
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
                    if let replay::DecideMode::Capture(rec) = &mut decide_mode {
                        let mut recon_y = Vec::with_capacity(64 * 64);
                        for r in 0..64 {
                            let off = (sb_y + r) * pw + sb_x;
                            recon_y.extend_from_slice(&recy[off..off + 64]);
                        }
                        rec.push_luma420(replay::Luma420 {
                            tus: tus.clone(),
                            mode_idx,
                            adelta,
                            recon_y,
                        });
                    }
                    (tus, mode_idx)
                };
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
                enc.delta_q_pending = enc.delta_q_present;
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
        // Pass 1: derive the per-block CDEF decision from this reconstruction (400
        // lossy is always whole-frame — no tiling — so no capture_recon guard).
        if self.tune.cdef
            && cdef_pre.is_none()
            && !self
                .capture_recon
                .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.cdef_decided = cdef_est::search_per_block(
                std::slice::from_ref(&recy),
                &[yp.to_vec()],
                pw,
                ph,
                pw,
                ph,
                0,
                0,
                false,
                self.base_q_idx,
                self.bit_depth,
            );
        }
        if self
            .capture_recon
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            enc.recon = vec![recy];
        }
        enc
    }
}
