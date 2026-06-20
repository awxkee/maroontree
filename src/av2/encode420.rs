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
use crate::av2::cfl::{cfl_partition_prediction, cfl_prediction};

impl Av2Encoder {
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv420_partition(
        &self,
        enc: &mut RangeEncoder,
        luma: LumaPlanes,
        chroma: ChromaPlaneRefs,
        ctx: &PartitionPass,
        nb: PartitionNeighbors,
        cnb: ChromaNeighborBufs,
    ) {
        let LumaPlanes { rec: recy, src: yp } = luma;
        let ChromaPlaneRefs {
            rec_u: recu,
            rec_v: recv,
            src_u: up,
            src_v: vp,
        } = chroma;
        let &PartitionPass {
            luma_stride: pw,
            chroma_stride: pcw,
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
                },
        } = ctx;
        let PartitionNeighbors {
            above,
            left,
            above_pctx,
            left_pctx,
        } = nb;
        let ChromaNeighborBufs {
            u_above,
            v_above,
            u_left,
            v_left,
        } = cnb;
        // Per-mi CfL-usage neighbours for get_cfl_ctx (one bit per chroma block).
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let bases = &self.bases;
        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    tmr as usize,
                    tmc as usize,
                    above_pctx,
                    left_pctx,
                );
                // Arm this SB's delta-Q once; the first leaf's mode emitter consumes it
                // (after its partition bit), so it is coded exactly once per SB. Edge
                // SBs on the partition path stay quantization-neutral (signaled = 0).
                enc.delta_q_signaled = 0;
                enc.delta_q_pending = enc.delta_q_present;
                for op in &ops {
                    let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(*cdf, *val);
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
                    let sb_y = lmr * 4;
                    let sb_x = lmc * 4;
                    let ua = if lmr > 0 { u_above[lmc] } else { 0 };
                    let ul = if lmc > 0 { u_left[lmr] } else { 0 };
                    let va = if lmr > 0 { v_above[lmc] } else { 0 };
                    let vl = if lmc > 0 { v_left[lmr] } else { 0 };
                    // 4:2:0 chroma origin: half the luma origin in BOTH axes.
                    let (cy, cx) = (sb_y / 2, sb_x / 2);
                    // CfL neighbor context + per-leaf default (every eligible leaf emits an
                    // is_cfl bit; only the whole-64 leaf may set it).
                    let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
                    let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
                    enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                    enc.cfl_use = false;
                    let (u_present, v_present) = match (bw_mi, bh_mi) {
                        (16, 16) => {
                            // 64x64 luma → 32x32 chroma (TX_32X32, eob 1024, skip TX32).
                            let (tus, mode_idx, _) = encode_luma_sb(
                                recy,
                                yp,
                                pw,
                                width,
                                height,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                1.0, // resid_scale: no AQ on this path
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                                false, // partition path: keep non-directional (not wired)
                            );
                            let (skip_cdfs, dc_sign_ctxs) =
                                sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                            const PARTITION_CFL: bool = true;
                            let bd = self.bit_depth as i32;
                            let cfl_choice = if enc.cfl && PARTITION_CFL {
                                let avg_l =
                                    cfl::cfl_avg_l(recy, pw, sb_y, sb_x, 32, 32, true, true, bd);
                                let mut suf = [0f32; 32 * 32];
                                let mut svf = [0f32; 32 * 32];
                                cfl_partition_prediction::<32>(
                                    pcw, up, vp, cy, cx, &mut suf, &mut svf,
                                );
                                let dc_u_f = dc_pred_rect(recu, pcw, cy, cx, 32, 32, neutral, bd);
                                let dc_v_f = dc_pred_rect(recv, pcw, cy, cx, 32, 32, neutral, bd);
                                cfl::cfl_decide(
                                    recy,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    &suf,
                                    &svf,
                                    dc_u_f,
                                    dc_v_f,
                                    32,
                                    32,
                                    true,
                                    true,
                                    avg_l,
                                    bd,
                                    &bases.chroma420,
                                    qstep_i,
                                    leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
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
                            encode_luma_block_split(
                                enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                pc,
                            );
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 32,
                                    basis: &bases.chroma420,
                                    scan: &tables::SCAN,
                                    eob_bin: &CHROMA_EOB_BIN_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 1024,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                cfl_choice.as_ref(),
                            )
                        }
                        (16, 8) => {
                            // 64x32 luma → 32x16 chroma (TX_32X16, eob 512, skip TX32).
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) =
                                sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                            encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 16,
                                    basis: &bases.c32x16,
                                    scan: &SCAN32X16,
                                    eob_bin: &CHROMA_EOB512_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 512,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (8, 16) => {
                            // 32x64 luma → 16x32 chroma (TX_16X32, eob 512, skip TX32).
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0), (32, 0)],
                                &tus2,
                                sb_y,
                                sb_x,
                                above,
                                left,
                                qc,
                                tmc,
                                tmr,
                                false,
                            );
                            let s2 = [skip2[0], skip2[1]];
                            let d2 = [dcs2[0], dcs2[1]];
                            encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 32,
                                    basis: &bases.c16x32,
                                    scan: &crate::av2::tables::SCAN16X32,
                                    eob_bin: &CHROMA_EOB512_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 512,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (8, 8) => {
                            // 32x32 luma → 16x16 chroma (TX_16X16, eob 256, skip TX16).
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0)],
                                std::slice::from_ref(&tu),
                                sb_y,
                                sb_x,
                                above,
                                left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            encode_luma_leaf_32x32(enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 16,
                                    basis: &bases.luma16x16,
                                    scan: &SCAN16,
                                    eob_bin: &CHROMA_EOB256_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (4, 16) => {
                            // Right-edge 16x64 luma leaf → 4:2:0 chroma 8x32 (TX_8X32,
                            // coeff 8x32, SCAN8X32, eob 256, ctx-2 SKIP_TX16). Reuses the
                            // luma8x32 basis (identical 8x32 geometry).
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma16x64.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
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
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN16X32,
                                    16,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 16, true,
                            );
                            encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 32,
                                    basis: &bases.luma8x32,
                                    scan: &SCAN8X32,
                                    eob_bin: &CHROMA_EOB256_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (16, 4) => {
                            // Bottom-edge 64x16 luma leaf → 4:2:0 chroma 32x8 (TX_32X8,
                            // coeff 32x8, SCAN32X8, eob 256, ctx-2 SKIP_TX16). Reuses the
                            // luma32x8 basis.
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma64x16.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
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
                                    qstep_i,
                                    &SCAN32X16,
                                    64,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 16, 4, true,
                            );
                            encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 8,
                                    basis: &bases.luma32x8,
                                    scan: &SCAN32X8,
                                    eob_bin: &CHROMA_EOB256_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (2, 8) => {
                            // Right-edge 8×32 luma leaf (residue-2 width) → 4:2:0 chroma
                            // 4×16 (TX_4X16, SCAN4X16, eob 64, ctx-1 SKIP_TX8). Luma
                            // TX_8X32 long-side-32 (min=1 short cdf).
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                    qstep_i,
                                    &SCAN8X32,
                                    8,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 8, true,
                            );
                            encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 4,
                                    ch: 16,
                                    basis: &bases.c4x16,
                                    scan: &tables::SCAN4X16,
                                    eob_bin: &CHROMA_EOB64_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 64,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (8, 2) => {
                            // Bottom-edge 32×8 luma leaf (residue-2 height) → 4:2:0 chroma
                            // 16×4 (TX_16X4, SCAN16X4, eob 64, ctx-1 SKIP_TX8).
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                    qstep_i,
                                    &SCAN32X8,
                                    32,
                                    8,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 8, 2, true,
                            );
                            encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 4,
                                    basis: &bases.c16x4,
                                    scan: &tables::SCAN16X4,
                                    eob_bin: &CHROMA_EOB64_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 64,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (4, 8) => {
                            // Bottom-right 16×32 corner leaf (residue-4 width ×
                            // residue-{6,8} height) → 4:2:0 chroma 8×16 (TX_8X16,
                            // SCAN8X16, eob class 128, ctx-2 SKIP_TX16). Luma TX_16X32
                            // is EXT_TX long-side-32 (DCT_DCT via the long32 coder).
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN16X32,
                                    16,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 8, true,
                            );
                            encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 16,
                                    basis: &bases.c8x16,
                                    scan: &tables::SCAN8X16,
                                    eob_bin: &CHROMA_EOB128_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 128,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (8, 4) => {
                            // Bottom-right 32×16 corner leaf (residue-{6,8} width ×
                            // residue-4 height) → 4:2:0 chroma 16×8 (TX_16X8, SCAN16X8,
                            // eob class 128, ctx-2 SKIP_TX16). Luma TX_32X16 long-side-32.
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN32X16,
                                    32,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 8, 4, true,
                            );
                            encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, true, pc);
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 8,
                                    basis: &bases.c16x8,
                                    scan: &tables::SCAN16X8,
                                    eob_bin: &CHROMA_EOB128_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 128,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner leaf (residue 4 in both dims):
                            // full-AC TX_16X16 luma with 4-way ADST RD (DCT_DCT /
                            // ADST_ADST / ADST_DCT / DCT_ADST, DC mode) → 4:2:0 chroma
                            // 8×8 (TX_8X8, SCAN8X8, eob class 64, skip txs_ctx 1).
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let resid = get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred);
                            let pred_flat = [pred; 256];
                            let mut src16 = [0f32; 256];
                            for r in 0..16 {
                                for c in 0..16 {
                                    src16[r * 16 + c] = yp[(sb_y + r) * pw + sb_x + c];
                                }
                            }
                            let rate = |lev: &[f32]| -> f64 {
                                lev.iter()
                                    .filter(|&&v| v != 0.0)
                                    .map(|&v| 2.0 + 2.0 * ((v.abs() as f64) + 1.0).log2())
                                    .sum::<f64>()
                            };
                            let sse = |rec: &[f32]| -> f64 {
                                (0..256)
                                    .map(|i| {
                                        let d = src16[i] as f64 - rec[i] as f64;
                                        d * d
                                    })
                                    .sum()
                            };
                            let lambda = leaf::part_lambda(qstep_i, self.tune.part_lambda_c);
                            let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                            let rec_dct = itx422::reconstruct_luma16(
                                &pred_flat,
                                &lev_dct,
                                qstep_i,
                                &SCAN16,
                                self.bit_depth as i32,
                            );
                            let cost_dct = sse(&rec_dct) + lambda * rate(&lev_dct);
                            let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_adst = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_adst,
                                qstep_i,
                                &SCAN16,
                                true,
                                true,
                                self.bit_depth as i32,
                            );
                            let cost_adst = sse(&rec_adst) + lambda * (rate(&lev_adst) + 0.2);
                            let lev_ad =
                                bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                            let rec_ad = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_ad,
                                qstep_i,
                                &SCAN16,
                                false,
                                true,
                                self.bit_depth as i32,
                            );
                            let cost_ad = sse(&rec_ad) + lambda * (rate(&lev_ad) + 3.12);
                            let lev_da =
                                bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_da = itx422::reconstruct_luma16_adst(
                                &pred_flat,
                                &lev_da,
                                qstep_i,
                                &SCAN16,
                                true,
                                false,
                                self.bit_depth as i32,
                            );
                            let cost_da = sse(&rec_da) + lambda * (rate(&lev_da) + 2.71);
                            let mut best = cost_dct;
                            let mut choice = 0usize;
                            if cost_adst < best {
                                best = cost_adst;
                                choice = 1;
                            }
                            if cost_ad < best {
                                best = cost_ad;
                                choice = 2;
                            }
                            if cost_da < best {
                                choice = 3;
                            }
                            let (lev, rec, tx_idx): (&[f32], &[f32; 256], usize) = match choice {
                                1 => (&lev_adst, &rec_adst, 1),
                                2 => (&lev_ad, &rec_ad, 2),
                                3 => (&lev_da, &rec_da, 3),
                                _ => (&lev_dct, &rec_dct, 0),
                            };
                            put_block_rect(recy, pw, sb_y, sb_x, 16, 16, rec);
                            let tu: Vec<Coeff> = levels_to_coeffs(lev);
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 4, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_16x16_full(
                                enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx,
                            );
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 8,
                                    basis: &bases.c8x8,
                                    scan: &SCAN8X8,
                                    eob_bin: &CHROMA_EOB64_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 64,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (2, 2) => {
                            // Bottom-right 8×8 corner leaf (residue-2 both axes), TX_8X8 ctx-1.
                            // Luma TX_8X8 (szctx=1, do_part_cdf=3148, ext-tx txtp_ext(min=1)
                            // DCT_DCT idx 0); 4:2:0 chroma is one 4×4 (TX_4X4) TU per plane.
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                8,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.c8x8.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 8, pred),
                                0.0,
                                &SCAN8X8,
                            );
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                8,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN8X8,
                                    8,
                                    8,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 2, true,
                            );
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
                            use crate::av2::coder::{
                                SCAN4X4_LOSSY, SCAN4X4_LOSSY_PACKED, encode_chroma_tu4_scan,
                            };
                            let bd = self.bit_depth as i32;
                            let predu = dc_pred_rect(recu, pcw, cy, cx, 4, 4, neutral, bd);
                            let levu = bases.c4x4.project_scan(
                                &get_residual_rect(up, pcw, cy, cx, 4, 4, predu),
                                0.0,
                                &SCAN4X4_LOSSY_PACKED,
                            );
                            put_block_rect(
                                recu,
                                pcw,
                                cy,
                                cx,
                                4,
                                4,
                                &itx422::reconstruct_chroma(
                                    predu,
                                    &levu,
                                    qstep_i,
                                    &SCAN4X4_LOSSY_PACKED,
                                    4,
                                    4,
                                    bd,
                                ),
                            );
                            let uc = levels_to_coeffs(&levu);
                            let u_skip = TXB_SKIP_TX4_Q0[(6 + ua + ul) as usize] as u32;
                            encode_chroma_tu4_scan(enc, &uc, u_skip, false, &SCAN4X4_LOSSY);
                            let u_nz = uc.iter().any(|&(_, l)| l != 0);
                            let predv = dc_pred_rect(recv, pcw, cy, cx, 4, 4, neutral, bd);
                            let levv = bases.c4x4.project_scan(
                                &get_residual_rect(vp, pcw, cy, cx, 4, 4, predv),
                                0.0,
                                &SCAN4X4_LOSSY_PACKED,
                            );
                            put_block_rect(
                                recv,
                                pcw,
                                cy,
                                cx,
                                4,
                                4,
                                &itx422::reconstruct_chroma(
                                    predv,
                                    &levv,
                                    qstep_i,
                                    &SCAN4X4_LOSSY_PACKED,
                                    4,
                                    4,
                                    bd,
                                ),
                            );
                            let vc = levels_to_coeffs(&levv);
                            let v_skip =
                                V_TXB_SKIP_TX4_Q0[(6 * (u_nz as i32) + va + vl) as usize] as u32;
                            encode_chroma_tu4_scan(enc, &vc, v_skip, true, &SCAN4X4_LOSSY);
                            (u_nz, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (2, 4) => {
                            // residue-2 width × residue-4 height corner: 8×16 luma
                            // (TX_8X16) + 4×8 chroma per plane (4:2:0).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                            let lev = bases.c8x16.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 16, pred),
                                0.0,
                                &crate::av2::tables::SCAN8X16,
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
                                    qstep_i,
                                    &crate::av2::tables::SCAN8X16,
                                    8,
                                    16,
                                    bd,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 4, true,
                            );
                            crate::av2::coder::encode_luma_leaf_rect128(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                pc,
                                12348,
                                &crate::av2::tables::SCAN8X16,
                                Some((&crate::av2::coder::TXTP_EXT8, 0, 6)),
                            );
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 4,
                                    ch: 8,
                                    basis: &bases.c4x8,
                                    scan: &tables::SCAN4X8,
                                    eob_bin: &crate::av2::cdfs_qctx::CHROMA_EOB32_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 32,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        (4, 2) => {
                            // residue-4 width × residue-2 height corner: 16×8 luma
                            // (TX_16X8) + 8×4 chroma per plane (4:2:0).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                            let lev = bases.c16x8.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 8, pred),
                                0.0,
                                &crate::av2::tables::SCAN16X8,
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
                                    qstep_i,
                                    &crate::av2::tables::SCAN16X8,
                                    16,
                                    8,
                                    bd,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 2, true,
                            );
                            crate::av2::coder::encode_luma_leaf_rect128(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                pc,
                                12348,
                                &crate::av2::tables::SCAN16X8,
                                Some((&crate::av2::coder::TXTP_EXT8, 0, 6)),
                            );
                            code_422_chroma_tu(
                                enc,
                                ChromaPlanes {
                                    rec_u: &mut *recu,
                                    rec_v: &mut *recv,
                                    src_u: up,
                                    src_v: vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 4,
                                    basis: &bases.c8x4,
                                    scan: &tables::SCAN8X4,
                                    eob_bin: &crate::av2::cdfs_qctx::CHROMA_EOB32_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 32,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                            )
                        }
                        other => unreachable!("unsupported native 4:2:0 leaf {:?}", other),
                    };
                    let cfl_used = enc.cfl_use as i32;
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
                }
            }
        }
    }

    /// Encode a 4:2:0 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height/2`. Luma is four 32x32 TUs per superblock; each chroma plane
    /// is one 32x32 transform per superblock. `width`/`height` must be even.
    pub fn encode_yuv420<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        planar_image.validate_420()?;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let yf = to_plane(&planar_image.planes[0]);
        let cbf = to_plane(&planar_image.planes[1]);
        let crf = to_plane(&planar_image.planes[2]);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I420);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_420_tiled(
                &yf,
                &cbf,
                &crf,
                width,
                height,
                &config,
                color,
                log2c,
                log2r,
                self.threads,
            ));
        }
        let enc = self.encode_420_core(&yf, &cbf, &crf, width, height);
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// SB-loop core for one 4:2:0 region (whole frame or one tile). Pads the region
    /// planes, runs the per-SB (or native-edge-partition) encode, and returns the
    /// entropy coder; header/finish (or multi-tile assembly) happens in the caller.
    fn encode_420_core(
        &self,
        yf: &[f32],
        cbf: &[f32],
        crf: &[f32],
        width: usize,
        height: usize,
    ) -> RangeEncoder {
        let bases = &self.bases;
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph / 2);
        let yp = pad_plane(yf, width, height, pw, ph);
        let up = pad_plane(cbf, width.div_ceil(2), height.div_ceil(2), pcw, pch);
        let vp = pad_plane(crf, width.div_ceil(2), height.div_ceil(2), pcw, pch);
        let layout = Layout::I420;
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];
        // Per-SB CfL-usage grid for get_cfl_ctx (whole-64 fast path).
        let mut cfl_has = vec![0i32; sb_cols * sb_rows];

        // Native edge partitioning (no padding) when the geometry is supported and the
        // image has residue edges. Otherwise the whole-SB path codes one 32x32 chroma TU
        // per SB (padded to SB on non-aligned dims).
        let native_mi = native_420_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // 4:2:0 edge handling. The fast whole-SB path codes one 32×32 chroma TU per SB
        // against PADDED extents, but `native_420_mi` hands the context functions the
        // NATIVE (un-padded) mi extents for residues ≥6/==4. For residues {10,12,14}
        // `lossy_needs_partition` is false, so the old code took the fast path with that
        // native/padded extent mismatch — desyncing the decoder on the partial edge SB
        // (e.g. any side ≡ 40 mod 64, like 1000). Routing every non-64-aligned dimension
        // through the edge-aware partition walk fixes it; 64-aligned sizes are unchanged.
        let mc_edge = (((width + 7) & !7) / 4) as i64 % 16;
        let mr_edge = (((height + 7) & !7) / 4) as i64 % 16;
        let needs_partition = native_mi.is_some()
            && (lossy_needs_partition(width, height) || mc_edge != 0 || mr_edge != 0);
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = vec![0u8; 16];

        if needs_partition {
            self.encode_yuv420_partition(
                &mut enc,
                LumaPlanes {
                    rec: &mut recy,
                    src: &yp,
                },
                ChromaPlaneRefs {
                    rec_u: &mut recu,
                    rec_v: &mut recv,
                    src_u: &up,
                    src_v: &vp,
                },
                &PartitionPass {
                    luma_stride: pw,
                    chroma_stride: pcw,
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
                    },
                },
                PartitionNeighbors {
                    above: &mut above,
                    left: &mut left,
                    above_pctx: &mut above_pctx,
                    left_pctx: &mut left_pctx,
                },
                ChromaNeighborBufs {
                    u_above: &mut u_above,
                    v_above: &mut v_above,
                    u_left: &mut u_left,
                    v_left: &mut v_left,
                },
            );
            return enc;
        }

        let mut midx_grid = vec![0xff_u8; sb_cols * sb_rows];
        // Adaptive-quantization reference: center the per-SB qindex delta on this
        // tile's mean log-activity so the deltas are zero-mean (flat SBs get finer
        // q, busy SBs coarser) without a net quantizer bias that would just trade
        // size for PSNR. Computed in a cheap pre-pass over the tile's SBs.
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
        );
        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                // Per-SB adaptive quantization: pick this SB's qstep from source
                // activity (signaling the delta), then quantize+reconstruct luma and
                // chroma at it. (qstep_i, 1.0) when AQ is off.
                let (sb_qstep, sb_resid_scale) =
                    aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                let (tus, mode_idx, adelta) = encode_luma_sb(
                    &mut recy,
                    &yp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    &bases.luma,
                    sb_qstep,
                    sb_resid_scale,
                    &tables::SCAN,
                    neutral,
                    qc,
                    self.tune.rdoq_lambda,
                    self.speed,
                    self.bit_depth as i32,
                    true, // allow directional intra modes (core path)
                );
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
                let (cy, cx) = (sb_y / 2, sb_x / 2);
                // CfL decision (4:2:0 whole-64 fast path). recy is final after
                // encode_luma_sb; sets the per-block CfL state read by encode_intra_modes
                // during the luma encode below. Context from the per-SB CfL-usage grid.
                let cfl_a = if row > 0 {
                    cfl_has[(row - 1) * sb_cols + col]
                } else {
                    0
                };
                let cfl_l = if col > 0 {
                    cfl_has[row * sb_cols + col - 1]
                } else {
                    0
                };
                enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                let cfl_choice = if enc.cfl {
                    let bd = self.bit_depth as i32;
                    let avg_l = cfl::cfl_avg_l(&recy, pw, sb_y, sb_x, 32, 32, true, true, bd);
                    let mut suf = [0f32; 32 * 32];
                    let mut svf = [0f32; 32 * 32];
                    cfl_partition_prediction::<32>(pcw, &up, &vp, cy, cx, &mut suf, &mut svf);
                    let dc_u_f = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                    let dc_v_f = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                    cfl::cfl_decide(
                        &recy,
                        pw,
                        sb_y,
                        sb_x,
                        suf.as_slice(),
                        svf.as_slice(),
                        dc_u_f,
                        dc_v_f,
                        32,
                        32,
                        true,
                        true,
                        avg_l,
                        bd,
                        &bases.chroma420,
                        qstep_i,
                        leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
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
                } else {
                    enc.cfl_use = false;
                }
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
                // Arm this SB's delta-Q (value already set above); the mode emitter
                // consumes it after the partition bit, exactly once per SB.
                enc.delta_q_pending = enc.delta_q_present;
                let (_cul, sb_midx) = encode_luma_block_split_dir(
                    &mut enc,
                    &tus,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    mode_idx,
                    adelta,
                    true,
                    12276,
                    lmidx,
                    amidx,
                );
                midx_grid[row * sb_cols + col] = sb_midx;

                let bd = self.bit_depth as i32;
                let (levu, levv) = if let Some(ref ch) = cfl_choice {
                    let mut ru = [0f32; 32 * 32];
                    let mut rv = [0f32; 32 * 32];
                    cfl_prediction::<32>(pcw, &up, &vp, cy, cx, &ch, &mut ru, &mut rv);
                    let levu = bases
                        .chroma420
                        .project(&aq::scale_resid(&ru, sb_resid_scale), 0.0);
                    let levv = bases
                        .chroma420
                        .project(&aq::scale_resid(&rv, sb_resid_scale), 0.0);
                    put_block(
                        &mut recu,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma_cfl(
                            &ch.pred_u,
                            &levu,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    put_block(
                        &mut recv,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma_cfl(
                            &ch.pred_v,
                            &levv,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    (levu, levv)
                } else {
                    let predu = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                    let levu = bases.chroma420.project(
                        &aq::scale_resid(
                            &get_residual(&up, pcw, cy, cx, 32, predu),
                            sb_resid_scale,
                        ),
                        0.0,
                    );
                    put_block(
                        &mut recu,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma(
                            predu,
                            &levu,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    let predv = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                    let levv = bases.chroma420.project(
                        &aq::scale_resid(
                            &get_residual(&vp, pcw, cy, cx, 32, predv),
                            sb_resid_scale,
                        ),
                        0.0,
                    );
                    put_block(
                        &mut recv,
                        pcw,
                        cy,
                        cx,
                        32,
                        &itx422::reconstruct_chroma(
                            predv,
                            &levv,
                            sb_qstep,
                            &tables::SCAN,
                            32,
                            32,
                            bd,
                        ),
                    );
                    (levu, levv)
                };
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
                cfl_has[row * sb_cols + col] = cfl_choice.is_some() as i32;
            }
        }
        enc
    }

    /// Multi-tile 4:2:0 assembly. Each tile is an independent sub-frame encode; tiles
    /// run in parallel across `threads` workers (raster order preserved). 4:2:0 chroma
    /// is half-width/half-height, so a luma tile at `(x0, y0, tw, th)` maps to chroma
    /// `(x0/2, y0/2, tw/2, th/2)` — all even because SB boundaries are multiples of 64.
    #[allow(clippy::too_many_arguments)]
    fn encode_420_tiled(
        &self,
        yf: &[f32],
        cbf: &[f32],
        crf: &[f32],
        width: usize,
        height: usize,
        config: &Config,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
        threads: usize,
    ) -> Av2Frame {
        // See encode_444_tiled: when any edge tile isn't boundary-exact, pad the whole
        // frame SB-aligned, carve all-SB-aligned tiles, signal the padded size, and let
        // the AVIF muxer clap back to width×height. Otherwise, signal the real size.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| native_420_mi(tw, th).is_some());
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (sig_w, sig_h, lstride, cstride, planes, specs) = if exact {
            (
                width,
                height,
                width,
                width.div_ceil(2),
                (yf.to_vec(), cbf.to_vec(), crf.to_vec()),
                native_specs,
            )
        } else {
            (
                pw,
                ph,
                pw,
                pw / 2,
                (
                    pad_plane(yf, width, height, pw, ph),
                    pad_plane(cbf, width.div_ceil(2), height.div_ceil(2), pw / 2, ph / 2),
                    pad_plane(crf, width.div_ceil(2), height.div_ceil(2), pw / 2, ph / 2),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let cw = cstride; // chroma plane stride (4:2:0)
        let n = specs.len();
        let mut tiles_bytes: Vec<Vec<u8>> = vec![Vec::new(); n];
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        if nthreads <= 1 || n <= 1 {
            for (slot, &(x0, y0, tw, th)) in tiles_bytes.iter_mut().zip(&specs) {
                let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
                let tv = extract_subplane(crf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
                *slot = self.encode_420_core(&ty, &tu, &tv, tw, th).finish();
            }
        } else {
            let chunk = n.div_ceil(nthreads);
            let me = self;
            let (yf, cbf, crf) = (&yf, &cbf, &crf);
            std::thread::scope(|sc| {
                for (out_chunk, spec_chunk) in
                    tiles_bytes.chunks_mut(chunk).zip(specs.chunks(chunk))
                {
                    sc.spawn(move || {
                        for (slot, &(x0, y0, tw, th)) in out_chunk.iter_mut().zip(spec_chunk) {
                            let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
                            let tu = extract_subplane(
                                cbf,
                                cw,
                                x0 / 2,
                                y0 / 2,
                                tw.div_ceil(2),
                                th.div_ceil(2),
                            );
                            let tv = extract_subplane(
                                crf,
                                cw,
                                x0 / 2,
                                y0 / 2,
                                tw.div_ceil(2),
                                th.div_ceil(2),
                            );
                            *slot = me.encode_420_core(&ty, &tu, &tv, tw, th).finish();
                        }
                    });
                }
            });
        }
        assemble_multitile(
            config,
            sig_w,
            sig_h,
            width,
            height,
            color,
            log2c,
            log2r,
            self.bit_depth,
            ChromaFormat::Yuv420,
            &tiles_bytes,
        )
    }

    /// Encode an RGB image to 4:2:0 AV2. Converts RGB→YCbCr and downsamples
    /// chroma with a 2×2 box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_420<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        if self.base_q_idx == 0 {
            return Err(EncodeError::InvalidQuality);
        }
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = (1i32 << (bd - 1)) << Q;
        let mx_i = maxv;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let mut y = vec![0i32; w * h];
        let mut fcb_q = vec![0i32; w * h];
        let mut fcr_q = vec![0i32; w * h];
        for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(fcb_q.iter_mut())
            .zip(fcr_q.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
            *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
        }
        const HALF_AVG: i32 = 1 << (Q + 1); // rounding bias for >> (Q+2)
        let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);
        for row in 0..ch {
            for c in 0..cw {
                let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
                let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));
                let avg_q =
                    |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];
                cb[row * cw + c] = ((avg_q(&fcb_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
                cr[row * cw + c] = ((avg_q(&fcr_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            }
        }
        self.encode_yuv420(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr, Vec::new()],
            },
            color,
        )
    }
}
