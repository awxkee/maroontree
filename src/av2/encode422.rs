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
    /// Encode a 4:2:2 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height` (half width, full height). Luma is four 32×32 TUs per
    /// superblock; each chroma plane is one 32-wide × 64-tall (TX_32X64) transform per
    /// superblock. `width` must be even. Chroma coefficient coding is identical to 4:4:4
    /// (avm codes TX_32X64 with the 32×32 scan and TX_64X64 entropy context); only the
    /// basis is the rectangular `chroma422` set.
    pub fn encode_yuv422<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        // Lossy 4:2:x is reconstruction-sequential (intra prediction reads the
        // running reconstruction), so it codes serially regardless of `threads`.
        let _ = threads;
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        planar_image.validate_422()?;
        if self.base_q_idx == 0 {
            return self.encode_yuv422_lossless(planar_image, color, threads);
        }
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph); // chroma: half width, full height
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width / 2, height, pcw, pch);
        let vp = pad_plane(&to_plane(cr), width / 2, height, pcw, pch);

        let layout = Layout::I422;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let qstep_i = crate::av2::quant::qstep(self.base_q_idx as u32) as i32;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        // Native (no-pad) 4:2:2 luma extents for the 32-family; else pad to whole SBs.
        let native_mi = native_422_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // Per-mi chroma neighbour presence, indexed in luma-mi space (the shared tree
        // drives both planes, so neighbour relations match the luma footprint).
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        let needs_partition = native_mi.is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = vec![0u8; 16];

        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                if !needs_partition {
                    // Whole 64×64 luma SB → one 32×64 (TX_32X64) chroma TU per plane.
                    let (fmr, fmc) = (row * 16, col * 16);
                    let ua = if fmr > 0 { u_above[fmc] } else { 0 };
                    let ul = if fmc > 0 { u_left[fmr] } else { 0 };
                    let va = if fmr > 0 { v_above[fmc] } else { 0 };
                    let vl = if fmc > 0 { v_left[fmr] } else { 0 };
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
                    let (cy, cx) = (sb_y, sb_x / 2);
                    let predu = dc_pred_rect(&recu, pcw, cy, cx, 32, 64, neutral);
                    let levu = bases.chroma422.project_scan(
                        &get_residual_rect(&up, pcw, cy, cx, 32, 64, predu),
                        0.0,
                        &crate::av2::tables::SCAN,
                    );
                    put_block_rect(
                        &mut recu,
                        pcw,
                        cy,
                        cx,
                        32,
                        64,
                        &recon_422_chroma(
                            predu,
                            &levu,
                            qstep_i,
                            &crate::av2::tables::SCAN,
                            32,
                            64,
                            &bases.chroma422,
                        ),
                    );
                    let predv = dc_pred_rect(&recv, pcw, cy, cx, 32, 64, neutral);
                    let levv = bases.chroma422.project_scan(
                        &get_residual_rect(&vp, pcw, cy, cx, 32, 64, predv),
                        0.0,
                        &crate::av2::tables::SCAN,
                    );
                    put_block_rect(
                        &mut recv,
                        pcw,
                        cy,
                        cx,
                        32,
                        64,
                        &recon_422_chroma(
                            predv,
                            &levv,
                            qstep_i,
                            &crate::av2::tables::SCAN,
                            32,
                            64,
                            &bases.chroma422,
                        ),
                    );
                    let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                    let u_skip = CHROMA_SKIP_TX64_QC[qc][(6 + ua + ul) as usize] as u32;
                    encode_chroma_block_rect(
                        &mut enc,
                        &uc,
                        u_skip,
                        true,
                        &crate::av2::tables::SCAN,
                        &CHROMA_EOB_BIN_QC[qc],
                        CHROMA_EOB_HI_BIT_QC[qc],
                        1024,
                    );
                    let up_ = uc.iter().any(|&(_, l)| l != 0);
                    let v_skip = CHROMA_SKIP_V_QC[qc][(6 * (up_ as i32) + va + vl) as usize] as u32;
                    encode_chroma_block_rect(
                        &mut enc,
                        &vc,
                        v_skip,
                        false,
                        &crate::av2::tables::SCAN,
                        &CHROMA_EOB_BIN_QC[qc],
                        CHROMA_EOB_HI_BIT_QC[qc],
                        1024,
                    );
                    let v_present = vc.iter().any(|&(_, l)| l != 0);
                    for c in fmc..fmc + 16 {
                        u_above[c] = up_ as i32;
                        v_above[c] = v_present as i32;
                    }
                    for r in fmr..fmr + 16 {
                        u_left[r] = up_ as i32;
                        v_left[r] = v_present as i32;
                    }
                    continue;
                }

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
                    let sb_y = lmr * 4;
                    let sb_x = lmc * 4;
                    let ua = if lmr > 0 { u_above[lmc] } else { 0 };
                    let ul = if lmc > 0 { u_left[lmr] } else { 0 };
                    let va = if lmr > 0 { v_above[lmc] } else { 0 };
                    let vl = if lmc > 0 { v_left[lmr] } else { 0 };
                    let (cy, cx) = (sb_y, sb_x / 2);
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 64,
                                    basis: &bases.chroma422,
                                    scan: &crate::av2::tables::SCAN,
                                    eob_bin: &CHROMA_EOB_BIN_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 1024,
                                    u_skip_row: &CHROMA_SKIP_TX64_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                            )
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 32,
                                    basis: &bases.chroma420,
                                    scan: &crate::av2::tables::SCAN,
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
                            )
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 64,
                                    basis: &bases.luma16x64,
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
                            )
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
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
                            )
                        }
                        (4, 16) => {
                            // Right-edge 16×64 luma leaf → 4:2:2 chroma 8×64 (TX_8X64,
                            // coeff 8×32, SCAN8X32, eob 256, skip class 3).
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 64,
                                    basis: &bases.c8x64,
                                    scan: &crate::av2::tables::SCAN8X32,
                                    eob_bin: &CHROMA_EOB256_QC[qc],
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                            )
                        }
                        (16, 4) => {
                            // Bottom-edge 64×16 luma leaf → 4:2:2 chroma 32×16 (TX_32X16,
                            // coeff 32×16, SCAN32X16, eob 512, skip class 3).
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 32,
                                    ch: 16,
                                    basis: &bases.c32x16,
                                    scan: &crate::av2::tables::SCAN32X16,
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
                            )
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner → DC-only luma; 4:2:2 chroma 8×16
                            // (TX_8X16, coeff 8×16, SCAN8X16, eob 128 NO-ESCAPE, skip class 2).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 16, neutral);
                            let mut lev = bases.luma16x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 16, pred),
                                0.0,
                                &SCAN16,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
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
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                &mut enc, dc_level, skip, dcs, 0, true, pc, 11074,
                            );
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 8,
                                    ch: 16,
                                    basis: &bases.c8x16,
                                    scan: &crate::av2::tables::SCAN8X16,
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
                            )
                        }
                        (2, 8) => {
                            // Right-edge 8×32 DC-only luma leaf → 4:2:2 chroma 4×32
                            // (TX_4X32, coeff 4×32, SCAN4X32, eob 128 NO-ESCAPE, class 2).
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 32, neutral);
                            let mut lev = bases.luma8x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 32, pred),
                                0.0,
                                &SCAN8X32,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
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
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                &mut enc, dc_level, skip, dcs, 0, true, pc, 18958,
                            );
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 4,
                                    ch: 32,
                                    basis: &bases.c4x32,
                                    scan: &crate::av2::tables::SCAN4X32,
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
                            )
                        }
                        (8, 2) => {
                            // Bottom-edge 32×8 DC-only luma leaf → 4:2:2 chroma 16×8
                            // (TX_16X8, coeff 16×8, SCAN16X8, eob 128 NO-ESCAPE, class 2).
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
                            code_422_chroma_tu(
                                &mut enc,
                                ChromaPlanes {
                                    rec_u: &mut recu,
                                    rec_v: &mut recv,
                                    src_u: &up,
                                    src_v: &vp,
                                    stride: pcw,
                                },
                                cy,
                                cx,
                                &ChromaTxSpec {
                                    cw: 16,
                                    ch: 8,
                                    basis: &bases.c16x8,
                                    scan: &crate::av2::tables::SCAN16X8,
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
                            )
                        }
                        other => unreachable!("unsupported native 4:2:2 leaf {:?}", other),
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

    /// Lossless 4:2:2: identical recursion to the 4:4:4 lossless path, but the chroma
    /// planes are half-width, so each luma leaf of `lcols × lrows` 4×4 WHT TUs maps to a
    /// `(lcols/2) × lrows` chroma block at chroma column `lc/2`. The mode-info grid is
    /// 8px-aligned so every leaf starts on an even mi column, keeping the halving exact.
    fn encode_yuv422_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_422()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph);
        let (cw, chh) = (width.div_ceil(2), height);
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), cw, chh, pcw, pch);
        let vp = pad_plane(&to_plane(cr), cw, chh, pcw, pch);
        let config = self.config(Layout::I422);
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        // Chroma neighbour grids live in chroma-pixel space (half-width above grid).
        let mut ua = vec![0u8; pcw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pcw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        type PackedCoeff = Vec<Coeff>;
        // Phase A: per-SB 4×4 WHT TUs. Luma over the full SB (rr×rc), chroma over the
        // half-width SB (rr × rc/2) at chroma origin sb_x/2.
        let mut sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> = (0..nsb)
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let gen_block =
            |idx: usize, slot: &mut (Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)| {
                let (row, col) = (idx / sb_cols, idx % sb_cols);
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                *slot = (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                    lossless_sb_tus(&vp, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                );
            };
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                gen_block(idx, slot);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (yp, up, vp) = (&yp, &up, &vp);
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
                                lossless_sb_tus(up, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                                lossless_sb_tus(vp, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                            );
                        }
                    });
                }
            });
        }
        // Phase B: serial partition + entropy walk (shared luma tree drives both planes).
        let mut above_pctx = vec![0u8; code_mc + 16];
        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                let (ytus, utus, vtus) = &sbtus[row * sb_cols + col];
                let ccols = rc / 2; // chroma TU columns in this SB's grid
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
                            let yslice = {
                                let mut v = Vec::with_capacity(lrows * lcols);
                                for i in 0..lrows {
                                    for j in 0..lcols {
                                        v.push(ytus[(lr + i) * rc + (lc + j)].clone());
                                    }
                                }
                                v
                            };
                            // Chroma is half-width: clc = lc/2, ccols_leaf = lcols/2.
                            let clc = lc / 2;
                            let ccols_leaf = lcols / 2;
                            let cslice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * ccols_leaf);
                                for i in 0..lrows {
                                    for j in 0..ccols_leaf {
                                        v.push(g[(lr + i) * ccols + (clc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (lutus, lvtus) = (cslice(utus), cslice(vtus));
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let cx = lx / 2;
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&yslice, ly, lx, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly, cx, &mut ua, &mut ul, false, false, lrows, ccols_leaf,
                            );
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly, cx, &mut va, &mut vl, true, u_last_nz, lrows,
                                ccols_leaf,
                            );
                            encode_lossless_luma_sb(
                                &mut enc,
                                &yslice,
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

    /// Encode an RGB image to 4:2:2 AV2. Converts RGB→YCbCr and downsamples
    /// chroma horizontally with a 2-tap box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_422<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
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
        let cw = w.div_ceil(2);
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
        const HALF_AVG: i32 = 1 << Q;
        let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);
        for row in 0..h {
            for c in 0..cw {
                let x0 = 2 * c;
                let x1 = (2 * c + 1).min(w - 1);
                let cb0 = fcb_q[row * w + x0];
                let cb1 = fcb_q[row * w + x1];
                let cr0 = fcr_q[row * w + x0];
                let cr1 = fcr_q[row * w + x1];
                cb[row * cw + c] = ((cb0 + cb1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
                cr[row * cw + c] = ((cr0 + cr1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            }
        }
        self.encode_yuv422(
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
