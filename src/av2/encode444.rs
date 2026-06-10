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
    /// Encode a 4:4:4 YCbCr still. `y`, `cb`, `cr` are full-resolution
    /// (`width × height`). Luma is four 32x32 transform units per 64x64 superblock;
    /// each chroma plane is one 64x64 transform per superblock.
    pub fn encode_yuv444<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        if self.base_q_idx == 0 {
            return self.encode_yuv444_lossless(planar_image, color, threads);
        }
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        // Native-size 444: boundary-safe non-aligned sizes can signal real W×H so the
        // decoder reconstructs the full padded SB and crops — no AVIF clap box needed.
        let native_mi = lossy_native_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);

        let layout = Layout::I444;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pw * ph];
        let mut recv = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        // Per-mi chroma neighbour coeff-presence (mirrors the luma above/left arrays):
        // `*_above[mi_col]` / `*_left[mi_row]` hold whether the most recent TU covering
        // that column/row had U/V coeffs. Per-mi (not per-SB) so that multiple chroma
        // TUs within one SB — e.g. the two vertically stacked 8×32 residue-2 leaves —
        // see each other as neighbours.
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        let qstep_i = crate::av2::quant::qstep(self.base_q_idx as u32) as i32;
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
        let mut left_pctx = vec![0u8; 16];

        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                // Fast-path SB chroma context at the SB-origin mi (col*16, row*16).
                let (fmr, fmc) = (row * 16, col * 16);
                let ua = if fmr > 0 { u_above[fmc] } else { 0 };
                let ul = if fmc > 0 { u_left[fmr] } else { 0 };
                let va = if fmr > 0 { v_above[fmc] } else { 0 };
                let vl = if fmc > 0 { v_left[fmr] } else { 0 };

                // Helper closures capture nothing mutable; chroma coeff encode is inlined
                // per leaf because basis/size/skip-table differ.
                if !needs_partition {
                    // Fast path: whole 64X64 SB.
                    let (tus, mode_idx) = encode_luma_sb(
                        &mut recy,
                        &yp,
                        pw,
                        width,
                        height,
                        sb_y,
                        sb_x,
                        &bases.luma,
                        qstep_i,
                        &crate::av2::tables::SCAN,
                        neutral,
                    );
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr);
                    encode_luma_block_split(
                        &mut enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        true,
                        12276,
                    );
                    let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                    let levu = bases
                        .chroma444
                        .project(&get_residual(&up, pw, sb_y, sb_x, 64, predu), 0.0);
                    put_block(
                        &mut recu,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        &bases.chroma444.reconstruct(predu, &levu),
                    );
                    let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                    let levv = bases
                        .chroma444
                        .project(&get_residual(&vp, pw, sb_y, sb_x, 64, predv), 0.0);
                    put_block(
                        &mut recv,
                        pw,
                        sb_y,
                        sb_x,
                        64,
                        &bases.chroma444.reconstruct(predv, &levv),
                    );
                    let ucoeffs = levels_to_coeffs(&levu);
                    let vcoeffs = levels_to_coeffs(&levv);
                    let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                    encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                    let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                    let v_skip =
                        CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                    encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                    let v_present = vcoeffs.iter().any(|&(_, l)| l != 0);
                    for c in fmc..fmc + 16 {
                        u_above[c] = u_present as i32;
                        v_above[c] = v_present as i32;
                    }
                    for r in fmr..fmr + 16 {
                        u_left[r] = u_present as i32;
                        v_left[r] = v_present as i32;
                    }
                    continue;
                }

                // Walk + dispatch. For residues {6,8} each SB yields exactly one Leaf and
                // no RectType ops; RectType is handled generically for forward-compat.
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    tmr as usize,
                    tmc as usize,
                    &mut above_pctx,
                    &mut left_pctx,
                );
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
                    // Per-leaf position (a single SB may contain several leaves, e.g. the
                    // two stacked 8×32 residue-2 edges). Shadow sb_y/sb_x so the arms below
                    // address the leaf, not the SB origin.
                    let sb_y = lmr * 4;
                    let sb_x = lmc * 4;
                    let ua = if lmr > 0 { u_above[lmc] } else { 0 };
                    let ul = if lmc > 0 { u_left[lmr] } else { 0 };
                    let va = if lmr > 0 { v_above[lmc] } else { 0 };
                    let vl = if lmc > 0 { v_left[lmr] } else { 0 };
                    let (u_present, v_present) = match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx) = encode_luma_sb(
                                &mut recy,
                                &yp,
                                pw,
                                width,
                                height,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                                &tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_block_split(
                                &mut enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                true,
                                pc,
                            );
                            let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                            let levu = bases
                                .chroma444
                                .project(&get_residual(&up, pw, sb_y, sb_x, 64, predu), 0.0);
                            put_block(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                &bases.chroma444.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                            let levv = bases
                                .chroma444
                                .project(&get_residual(&vp, pw, sb_y, sb_x, 64, predv), 0.0);
                            put_block(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                &bases.chroma444.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 8) => {
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_64x32(
                                &tus2, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            encode_luma_leaf_64x32(
                                &mut enc, &tus2, &skip2, &dcs2, mode_idx, true, pc,
                            );
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 64, 32, neutral);
                            let levu = bases.chroma444_64x32.project(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 64, 32, predu),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &bases.chroma444_64x32.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 64, 32, neutral);
                            let levv = bases.chroma444_64x32.project(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 32, predv),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                32,
                                &bases.chroma444_64x32.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 16) => {
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0), (32, 0)],
                                &tus2,
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                false,
                            );
                            let s2 = [skip2[0], skip2[1]];
                            let d2 = [dcs2[0], dcs2[1]];
                            encode_luma_leaf_32x64(&mut enc, &tus2, &s2, &d2, mode_idx, true, pc);
                            // chroma TX_32X64 (32 wide x 64 tall): chroma422 basis, TX64 skip ctx.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 32, 64, neutral);
                            let levu = bases.chroma422.project(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 32, 64, predu),
                                0.0,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &bases.chroma422.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 32, 64, neutral);
                            let levv = bases.chroma422.project(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 32, 64, predv),
                                0.0,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                64,
                                &bases.chroma422.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 8) => {
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                &mut recy,
                                &yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &crate::av2::tables::SCAN,
                                neutral,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0)],
                                std::slice::from_ref(&tu),
                                sb_y,
                                sb_x,
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            encode_luma_leaf_32x32(
                                &mut enc, &tu, skip2[0], dcs2[0], mode_idx, true, pc,
                            );
                            // chroma TX_32X32: chroma420 basis, TX32 U skip, shared V skip.
                            let predu = dc_pred(&recu, pw, sb_y, sb_x, 32, neutral);
                            let levu = bases
                                .chroma420
                                .project(&get_residual(&up, pw, sb_y, sb_x, 32, predu), 0.0);
                            put_block(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                &bases.chroma420.reconstruct(predu, &levu),
                            );
                            let predv = dc_pred(&recv, pw, sb_y, sb_x, 32, neutral);
                            let levv = bases
                                .chroma420
                                .project(&get_residual(&vp, pw, sb_y, sb_x, 32, predv), 0.0);
                            put_block(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                &bases.chroma420.reconstruct(predv, &levv),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = crate::av2::cdfs_qctx::CHROMA_SKIP_TX32_QC[qc]
                                [(6 + ua + ul) as usize]
                                as u32;
                            encode_chroma_block(&mut enc, &uc, u_skip, true);
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block(&mut enc, &vc, v_skip, false);
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (4, 16) => {
                            // Right-edge 16×64 luma leaf (residue 4), DC pred (mode 0),
                            // single TX_16X64, coeff region 16×32 (SCAN16X32, eob 512).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 64, neutral);
                            let lev = bases.luma16x64.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 64, pred),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(pred, &lev, &SCAN16X32),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 16, true,
                            );
                            encode_luma_leaf_16x64(&mut enc, &tu, skip, dcs, 0, true, pc);
                            // chroma 16×64 (TX_16X64): reuse luma16x64 basis for projection
                            // (validity-only); chroma eob class 512, TX_32X32 skip ctx.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 16, 64, neutral);
                            let levu = bases.luma16x64.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 16, 64, predu),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(predu, &levu, &SCAN16X32),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 16, 64, neutral);
                            let levv = bases.luma16x64.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 16, 64, predv),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &bases.luma16x64.reconstruct_scan(predv, &levv, &SCAN16X32),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN16X32,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN16X32,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (16, 4) => {
                            // Bottom-edge 64×16 luma leaf (residue 4), DC pred, single
                            // TX_64X16, coeff region 32×16 (SCAN32X16, eob 512).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 64, 16, neutral);
                            let lev = bases.luma64x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 64, 16, pred),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(pred, &lev, &SCAN32X16),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 16, 4, true,
                            );
                            encode_luma_leaf_64x16(&mut enc, &tu, skip, dcs, 0, true, pc);
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 64, 16, neutral);
                            let levu = bases.luma64x16.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 64, 16, predu),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(predu, &levu, &SCAN32X16),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 64, 16, neutral);
                            let levv = bases.luma64x16.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 64, 16, predv),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &bases.luma64x16.reconstruct_scan(predv, &levv, &SCAN32X16),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = CHROMA_SKIP_TX32_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN32X16,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN32X16,
                                &CHROMA_EOB512_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                512,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (2, 8) => {
                            // Right-edge 8×32 luma leaf (residue 2). 8×64 would be 1:8
                            // aspect (disallowed) so the SB partitions to 8×32 leaves.
                            // TX_8X32 = entropy class 2; luma is DC-only (eob count 1 →
                            // no LONG_SIDE_32 tx_type). do_part group 8 → cdf 18958.
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 32, neutral);
                            let mut lev = bases.luma8x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 32, pred),
                                0.0,
                                &SCAN8X32,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0; // keep DC only → eob count 1
                            }
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &bases.luma8x32.reconstruct_scan(pred, &lev, &SCAN8X32),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 8, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32; // class-2 skip, ctx 0
                            encode_luma_leaf_dc_class2(
                                &mut enc, dc_level, skip, dcs, 0, true, pc, 18958,
                            );
                            // chroma 8×32 (TX_8X32): full AC, reuse luma8x32 basis, eob
                            // class 256, class-2 U skip / shared V skip.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 8, 32, neutral);
                            let levu = bases.luma8x32.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 8, 32, predu),
                                0.0,
                                &SCAN8X32,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &bases.luma8x32.reconstruct_scan(predu, &levu, &SCAN8X32),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 8, 32, neutral);
                            let levv = bases.luma8x32.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 8, 32, predv),
                                0.0,
                                &SCAN8X32,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &bases.luma8x32.reconstruct_scan(predv, &levv, &SCAN8X32),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = SKIP_TX16_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN8X32,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN8X32,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (8, 2) => {
                            // Bottom-edge 32×8 luma leaf (residue 2). TX_32X8 = class 2,
                            // DC-only luma, do_part group 8 → cdf 18958, scan SCAN32X8.
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 32, 8, neutral);
                            let mut lev = bases.luma32x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 32, 8, pred),
                                0.0,
                                &SCAN32X8,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
                            }
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &bases.luma32x8.reconstruct_scan(pred, &lev, &SCAN32X8),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 8, 2, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                &mut enc, dc_level, skip, dcs, 0, true, pc, 18958,
                            );
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 32, 8, neutral);
                            let levu = bases.luma32x8.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 32, 8, predu),
                                0.0,
                                &SCAN32X8,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &bases.luma32x8.reconstruct_scan(predu, &levu, &SCAN32X8),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 32, 8, neutral);
                            let levv = bases.luma32x8.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 32, 8, predv),
                                0.0,
                                &SCAN32X8,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &bases.luma32x8.reconstruct_scan(predv, &levv, &SCAN32X8),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = SKIP_TX16_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN32X8,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN32X8,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner leaf (residue 4 in both dims).
                            // Luma is DC-only (eob count 1) so the decoder skips the
                            // EXT_NEW_TX_SET tx_type; chroma codes full AC (tx_type is
                            // luma-only). TX_16X16 = entropy class 2, eob class 256.
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 16, neutral);
                            let mut lev = bases.luma16x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 16, pred),
                                0.0,
                                &SCAN16,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0; // keep DC only → eob count 1
                            }
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(pred, &lev, &SCAN16),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 4, true,
                            );
                            // TX_16X16 luma skip = class-2 cdf, block_eq_tx → ctx 0.
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                &mut enc, dc_level, skip, dcs, 0, true, pc, 11074,
                            );
                            // chroma 16×16 (TX_16X16): full AC, reuse luma16x16 basis,
                            // chroma eob class 256, class-2 U skip / shared V skip.
                            let predu = dc_pred_rect(&recu, pw, sb_y, sb_x, 16, 16, neutral);
                            let levu = bases.luma16x16.project_scan(
                                &get_residual_rect(&up, pw, sb_y, sb_x, 16, 16, predu),
                                0.0,
                                &SCAN16,
                            );
                            put_block_rect(
                                &mut recu,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(predu, &levu, &SCAN16),
                            );
                            let predv = dc_pred_rect(&recv, pw, sb_y, sb_x, 16, 16, neutral);
                            let levv = bases.luma16x16.project_scan(
                                &get_residual_rect(&vp, pw, sb_y, sb_x, 16, 16, predv),
                                0.0,
                                &SCAN16,
                            );
                            put_block_rect(
                                &mut recv,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &bases.luma16x16.reconstruct_scan(predv, &levv, &SCAN16),
                            );
                            let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                            let u_skip = SKIP_TX16_QC[qc][(6 + ua + ul) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &uc,
                                u_skip,
                                true,
                                &SCAN16,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            let up_ = uc.iter().any(|&(_, l)| l != 0);
                            let v_skip =
                                CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                            encode_chroma_block_rect(
                                &mut enc,
                                &vc,
                                v_skip,
                                false,
                                &SCAN16,
                                &CHROMA_EOB256_QC[qc],
                                CHROMA_EOB_HI_BIT_QC[qc],
                                256,
                            );
                            (up_, vc.iter().any(|&(_, l)| l != 0))
                        }
                        other => unreachable!("unsupported lossy leaf {:?}", other),
                    };
                    for c in lmc..lmc + bw_mi {
                        u_above[c] = u_present as i32;
                        v_above[c] = v_present as i32;
                    }
                    for r in lmr..lmr + bh_mi {
                        u_left[r] = u_present as i32;
                        v_left[r] = v_present as i32;
                    }
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// 4:4:4 lossless (q=0): luma + full-resolution U/V, all TX_4X4 WHT. Per superblock
    /// the block codes intra modes (incl. use_dpcm_y/uv = 0 and DC uv mode), then 256
    /// luma TUs, 256 U TUs, 256 V TUs — matching avm's shared-tree plane order.
    fn encode_yuv444_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);
        let config = self.config(Layout::I444);
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        // mi grid is 8px-aligned; recursion handles every boundary -> always exact.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        // luma ctx grids (0x40 = neutral DC-sign packing); chroma grids store cul (init 0).
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        let mut ua = vec![0u8; pw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        // Phase A: per-SB TU generation (DC-pred + WHT + levels). Independent across SBs
        // (lossless reconstruction == source), so this is data-parallel.
        type PackedCoeff = Vec<Coeff>;
        let mut sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> = (0..nsb)
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let gen_tile =
            |idx: usize, slot: &mut (Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)| {
                let (sb_y, sb_x) = ((idx / sb_cols) * 64, (idx % sb_cols) * 64);
                let (rr, rc) = rem(idx / sb_cols, idx % sb_cols);
                *slot = (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&vp, pw, sb_y, sb_x, neutral, rr, rc),
                );
            };
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                gen_tile(idx, slot);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (yp, up, vp) = (&yp, &up, &vp);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let (sb_y, sb_x) = (row * 64, col * 64);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = (
                                lossless_sb_tus(yp, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(up, pw, sb_y, sb_x, neutral, rr, rc),
                                lossless_sb_tus(vp, pw, sb_y, sb_x, neutral, rr, rc),
                            );
                        }
                    });
                }
            });
        }
        // Phase B: serial context derivation (cross-SB grids) + entropy coding.
        // Partition context arrays (av2 update_partition_context): `above` persists
        // down columns frame-wide; `left` is len-16 and zeroed per SB row.
        let mut above_pctx = vec![0u8; code_mc + 16];
        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                let (ytus, utus, vtus) = &sbtus[row * sb_cols + col];
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
                            enc.encode_bool(cdf, val);
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
                            let slice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * lcols);
                                for i in 0..lrows {
                                    for j in 0..lcols {
                                        v.push(g[(lr + i) * rc + (lc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (lytus, lutus, lvtus) = (slice(ytus), slice(utus), slice(vtus));
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&lytus, ly, lx, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly, lx, &mut ua, &mut ul, false, false, lrows, lcols,
                            );
                            // avm's eob_u_flag is the LAST U TU of the block, used by every V TU.
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly, lx, &mut va, &mut vl, true, u_last_nz, lrows, lcols,
                            );
                            // modes (incl. uv) + luma coeffs, then U, then V (shared-tree order)
                            encode_lossless_luma_sb(
                                &mut enc,
                                &lytus,
                                &yskip_cdfs,
                                &ydcs,
                                0,
                                true,
                                part_cdf,
                            );
                            for (i, tu) in lutus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    TXB_SKIP_TX4_Q0[uskip[i]] as u32,
                                    false,
                                );
                            }
                            for (i, tu) in lvtus.iter().enumerate() {
                                encode_chroma_tu4(
                                    &mut enc,
                                    tu,
                                    V_TXB_SKIP_TX4_Q0[vskip[i]] as u32,
                                    true,
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:4:4 AV2. Converts RGB→YCbCr internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_444<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let bd = img.bit_depth;
        let maxv = (1i32 << bd.bits()) - 1;
        let off_q = (1i32 << (bd.bits() - 1)) << Q;
        let mx_i = maxv;
        let n = img.planes[0].len();
        let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
        for (((((yv, cbv), crv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(cb.iter_mut())
            .zip(cr.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *cbv = ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
            *crv = ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
        }
        self.encode_yuv444(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
            threads,
        )
    }
}
