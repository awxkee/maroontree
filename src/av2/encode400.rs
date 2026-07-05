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

impl Av2Encoder {
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv400_partition(
        &self,
        enc: &mut RangeEncoder,
        luma: LumaPlanes,
        ctx: &PartitionPass,
        nb: PartitionNeighbors,
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
                    rdoq_lambda: _rdoq_lambda,
                },
            ..
        } = ctx;
        let PartitionNeighbors {
            above,
            left,
            above_pctx,
            left_pctx,
        } = nb;
        let bases = &self.bases;
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        );
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
                // Whole-64 interior SBs get AQ (the (16,16) leaf below); split edge
                // SBs stay quantization-neutral. delta_q is emitted once per SB.
                let whole_sb = col * 64 + 64 <= width && row * 64 + 64 <= height;
                let (sb_qstep, sb_resid_scale) = if whole_sb {
                    aqs.per_sb(enc, yp, pw, row * 64, col * 64, width, height)
                } else {
                    enc.delta_q_signaled = 0;
                    aqs.current()
                };
                enc.delta_q_pending = enc.delta_q_present;
                for op in &ops {
                    let (bw_mi, bh_mi, pc, _lmr, _lmc) = match op {
                        partition::Op::RectType { cdf, val } => {
                            enc.bool_rect_type(*cdf, *val);
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
                    let sb_y = _lmr * 4;
                    let sb_x = _lmc * 4;
                    match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx, _) = encode_luma_sb(
                                recy,
                                yp,
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
                                false, // non-directional path
                            );
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
                        (16, 8) => {
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                sb_qstep,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) =
                                sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                            encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, false, pc);
                        }
                        (8, 16) => {
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                sb_qstep,
                                &tables::SCAN,
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
                            encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, false, pc);
                        }
                        (8, 8) => {
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                sb_qstep,
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
                            encode_luma_leaf_32x32(
                                enc, &tu, skip2[0], dcs2[0], mode_idx, false, pc,
                            );
                        }
                        (4, 16) => {
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
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 16, true,
                            );
                            encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (16, 4) => {
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
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 16, 4, true,
                            );
                            encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (2, 8) => {
                            // 8x32 luma leaf, full AC (ported from 4:4:4).
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
                                &itx422::reconstruct_chroma(
                                    pred, &lev, sb_qstep, &SCAN8X32, 8, 32, bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 8, true,
                            );
                            coder::encode_luma_leaf_8x32(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (8, 2) => {
                            // 32x8 luma leaf, full AC (ported from 4:4:4).
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
                                &itx422::reconstruct_chroma(
                                    pred, &lev, sb_qstep, &SCAN32X8, 32, 8, bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 8, 2, true,
                            );
                            encode_luma_leaf_32x8(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (4, 4) => {
                            // 16x16 corner leaf: tx_type RD over DCT/ADST mixes (ported from 4:4:4).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 16, neutral, bd);
                            let resid = aq::scale_resid(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred),
                                sb_resid_scale,
                            );
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
                            let lambda = leaf::part_lambda(sb_qstep, self.tune.part_lambda_c);
                            let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                            let rec_dct = itx422::reconstruct_luma16(
                                &pred_flat, &lev_dct, sb_qstep, &SCAN16, bd,
                            );
                            let cost_dct = sse(&rec_dct) + lambda * rate(&lev_dct);
                            let lev_adst = bases.luma16x16_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_adst = itx422::reconstruct_luma16_adst(
                                &pred_flat, &lev_adst, sb_qstep, &SCAN16, true, true, bd,
                            );
                            let cost_adst = sse(&rec_adst) + lambda * (rate(&lev_adst) + 0.2);
                            let lev_ad =
                                bases.luma16x16_adst_dct.project_scan(&resid, 0.0, &SCAN16);
                            let rec_ad = itx422::reconstruct_luma16_adst(
                                &pred_flat, &lev_ad, sb_qstep, &SCAN16, false, true, bd,
                            );
                            let cost_ad = sse(&rec_ad) + lambda * (rate(&lev_ad) + 3.12);
                            let lev_da =
                                bases.luma16x16_dct_adst.project_scan(&resid, 0.0, &SCAN16);
                            let rec_da = itx422::reconstruct_luma16_adst(
                                &pred_flat, &lev_da, sb_qstep, &SCAN16, true, false, bd,
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
                            coder::encode_luma_leaf_16x16_full(
                                enc, &tu, skip, dcs, 0, false, pc, 11074, tx_idx,
                            );
                        }
                        (8, 4) => {
                            // 32x16 luma leaf (TX_32X16), ported from the 4:4:4 arm.
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
                            let lev = bases.luma32x16.project_scan(
                                &aq::scale_resid(
                                    &get_residual_rect(yp, pw, sb_y, sb_x, 32, 16, pred),
                                    sb_resid_scale,
                                ),
                                0.0,
                                &tables::SCAN32X16,
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
                                    sb_qstep,
                                    &tables::SCAN32X16,
                                    32,
                                    16,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 8, 4, true,
                            );
                            coder::encode_luma_leaf_32x16(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (4, 8) => {
                            // 16x32 luma leaf (TX_16X32), ported from the 4:4:4 arm.
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
                            let lev = bases.luma16x32.project_scan(
                                &aq::scale_resid(
                                    &get_residual_rect(yp, pw, sb_y, sb_x, 16, 32, pred),
                                    sb_resid_scale,
                                ),
                                0.0,
                                &tables::SCAN16X32,
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
                                    sb_qstep,
                                    &tables::SCAN16X32,
                                    16,
                                    32,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 8, true,
                            );
                            coder::encode_luma_leaf_16x32(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (4, 2) => {
                            // 16x8 luma leaf (TX_16X8, rect128), ported from the 4:4:4 arm.
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
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 2, true,
                            );
                            coder::encode_luma_leaf_rect128(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                false,
                                4,
                                2,
                                pc,
                                12348,
                                &tables::SCAN16X8,
                                Some((&coder::TXTP_EXT8, 0, 6)),
                            );
                        }
                        (2, 4) => {
                            // 8x16 luma leaf (TX_8X16, rect128), ported from the 4:4:4 arm.
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
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 4, true,
                            );
                            coder::encode_luma_leaf_rect128(
                                enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                false,
                                2,
                                4,
                                pc,
                                12348,
                                &tables::SCAN8X16,
                                Some((&coder::TXTP_EXT8, 0, 6)),
                            );
                        }
                        (2, 2) => {
                            // 8x8 luma leaf (TX_8X8), ported from the 4:4:4 arm.
                            let bd = self.bit_depth as i32;
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
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    sb_qstep,
                                    &tables::SCAN8X8,
                                    8,
                                    8,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 2, true,
                            );
                            coder::encode_luma_leaf_8x8(
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
                        other => unreachable!("unsupported native 4:0:0 leaf {:?}", other),
                    }
                }
            }
        }
        if let Ok(p) = std::env::var("DUMP_REC") {
            let mut o = Vec::with_capacity(width * height);
            for r in 0..height {
                o.extend(
                    recy[r * pw..r * pw + width]
                        .iter()
                        .map(|&v| v.clamp(0.0, 255.0) as u8),
                );
            }
            std::fs::write(p, o).unwrap();
        }
    }

    /// Encode a 4:0:0 (monochrome / luma-only) still. `y` is `width × height`.
    /// Four 32x32 luma TUs per superblock; no chroma is coded or signalled
    /// (`has_chroma = false` ⇒ no chroma intra mode, profile 0, layout uvlc 1).
    pub fn encode_yuv400<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_400()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);

        let layout = Layout::Monochrome;
        let config = self.config(layout);

        if config.lossless {
            return Ok(self.encode_yuv400_lossless(
                &yp,
                pw,
                ph,
                width,
                height,
                &config,
                color,
                self.threads,
            ));
        }

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
            && (lossy_needs_partition(width, height) || mc_edge != 0 || mr_edge != 0);
        if needs_partition {
            let mut above_pctx = vec![0u8; tmc as usize + 16];
            let mut left_pctx = vec![0u8; 16];
            self.encode_yuv400_partition(
                &mut enc,
                LumaPlanes {
                    rec: &mut recy,
                    src: &yp,
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
                        rdoq_lambda: self.tune.chroma_rdoq_lambda,
                    },
                },
                PartitionNeighbors {
                    above: &mut above,
                    left: &mut left,
                    above_pctx: &mut above_pctx,
                    left_pctx: &mut left_pctx,
                },
            );
            return Ok(self.finish(enc, &config, pw, ph, width, height, color));
        }

        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present {
                aq::tile_ref_activity(&yp, pw, sb_rows, sb_cols, width, height)
            } else {
                0.0
            },
        )
        .with_variance_boost(
            self.tune.vb_octile,
            self.tune.vb_strength,
            self.tune.vb_boost_only,
        );
        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let (sb_qstep, sb_resid_scale) =
                    aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                let (tus, mode_idx, _) = encode_luma_sb(
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
                    false, // non-directional path
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
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Lossless (base_q=0) monochrome encode: each 64x64 superblock is coded as 256
    /// 4x4 transform units (forced TX_4X4), DC-predicted per TU and carried by the 4x4
    /// WHT. `yp` is the SB-padded source plane. The pixel reconstruction is bit-exact;
    /// the 4x4 coefficient CDFs/contexts are still being validated against the decoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv400_lossless(
        &self,
        yp: &[f32],
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        config: &Config,
        color: &Cicp,
        threads: usize,
    ) -> Av2Frame {
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0; // base_q=0 -> q-context 0
        let neutral = self.dc_neutral();
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        // mi grid is 8px-aligned (avm dec_set_mb_mi); the recursive forced-split coder
        // handles every boundary geometry, so we always code the real (8-aligned) grid.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;

        // Phase A: per-SB TU generation is independent in lossless (recon == source),
        // so generate the clipped SB TU grids in parallel across `threads`.
        let nsb = sb_rows * sb_cols;
        let mut sbtus: Vec<Vec<Vec<Coeff>>> = (0..nsb).map(|_| Vec::new()).collect();
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                let (row, col) = (idx / sb_cols, idx % sb_cols);
                let (rr, rc) = ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16));
                *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
                        }
                    });
                }
            });
        }

        let mut above_pctx = vec![0u8; code_mc + 16];

        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let rr = (code_mr - row * 16).min(16);
                let rc = (code_mc - col * 16).min(16);
                // SB grid of in-frame 4x4 TUs (precomputed in Phase A).
                let tus = &sbtus[row * sb_cols + col];
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    code_mr,
                    code_mc,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val } => {
                            enc.bool_rect_type(cdf, val);
                        }
                        partition::Op::Split {
                            do_split_cdf,
                            square_cdf,
                        } => {
                            enc.bool_do_split(do_split_cdf, 1);
                            if square_cdf != 0 {
                                enc.bool_do_square_split(square_cdf, 1);
                            }
                        }
                        partition::Op::Leaf {
                            mi_row,
                            mi_col,
                            bw_mi,
                            bh_mi,
                            part_cdf,
                        } => {
                            let lr = mi_row - row * 16;
                            let lc = mi_col - col * 16;
                            let lrows = bh_mi.min(rr - lr);
                            let lcols = bw_mi.min(rc - lc);
                            let mut ltus = Vec::with_capacity(lrows * lcols);
                            for i in 0..lrows {
                                for j in 0..lcols {
                                    ltus.push(tus[(lr + i) * rc + (lc + j)].clone());
                                }
                            }
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (skip_ctx, dc_sign_ctxs) =
                                sb_tu4_contexts(&ltus, ly, lx, &mut above, &mut left, lrows, lcols);
                            let skip_cdfs: Vec<u32> = skip_ctx
                                .iter()
                                .map(|&c| TXB_SKIP_TX4_Q0[c] as u32)
                                .collect();
                            encode_lossless_luma_sb(
                                &mut enc,
                                &ltus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                0,
                                false,
                                part_cdf,
                            );
                        }
                    }
                }
            }
        }
        self.finish(enc, config, pw, ph, width, height, color)
    }

    /// Encode a luma-only (4:0:0 / monochrome) image to AV2.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_400<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_400()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let plane = img.planes[0].to_vec();
        self.encode_yuv400(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [plane, Vec::new(), Vec::new(), Vec::new()],
            },
            color,
        )
    }
}
