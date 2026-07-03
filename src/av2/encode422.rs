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
    /// Encode a 4:2:2 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height` (half width, full height). Luma is four 32×32 TUs per
    /// superblock; each chroma plane is one 32-wide × 64-tall (TX_32X64) transform per
    /// superblock. `width` must be even. Chroma coefficient coding is identical to 4:4:4
    /// (avm codes TX_32X64 with the 32×32 scan and TX_64X64 entropy context); only the
    /// basis is the rectangular `chroma422` set.
    pub fn encode_yuv422<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        let width = planar_image.width;
        let height = planar_image.height;
        planar_image.validate_422()?;
        if self.base_q_idx == 0 {
            return self.encode_yuv422_lossless(planar_image, color, self.threads);
        }
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let yf = to_plane(&planar_image.planes[0]);
        let cbf = to_plane(&planar_image.planes[1]);
        let crf = to_plane(&planar_image.planes[2]);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I422);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_422_tiled(
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
        let enc = self.encode_422_core(&yf, &cbf, &crf, width, height);
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// SB-loop core for one 4:2:2 region (whole frame or one tile). Pads the region
    /// planes, runs the per-SB encode, and returns the entropy coder; header/finish
    /// (or multi-tile assembly) happens in the caller.
    fn encode_422_core(
        &self,
        yf: &[f32],
        cbf: &[f32],
        crf: &[f32],
        width: usize,
        height: usize,
    ) -> RangeEncoder {
        let bases = &self.bases;
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph); // chroma: half width, full height
        let yp = pad_plane(yf, width, height, pw, ph);
        let up = pad_plane(cbf, width.div_ceil(2), height, pcw, pch);
        let vp = pad_plane(crf, width.div_ceil(2), height, pcw, pch);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = true;
        enc.mhccp_ssy = false;
        enc.delta_q_present = self.tune.aq && self.base_q_idx != 0;
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
        // Per-mi chroma neighbor presence, indexed in luma-mi space (the shared tree
        // drives both planes, so neighbor relations match the luma footprint).
        let mut u_above = vec![0i32; tmc as usize + 16];
        let mut v_above = vec![0i32; tmc as usize + 16];
        let mut u_left = vec![0i32; tmr as usize + 16];
        let mut v_left = vec![0i32; tmr as usize + 16];
        // Per-mi CfL-usage neighbors for get_cfl_ctx.
        let mut cfl_above = vec![0i32; tmc as usize + 16];
        let mut cfl_left = vec![0i32; tmr as usize + 16];
        let needs_partition = native_mi.is_some() && lossy_needs_partition(width, height);
        let mut above_pctx = vec![0u8; tmc as usize + 16];
        let mut left_pctx = vec![0u8; 16];
        let mut aqs = aq::AqState::new(
            enc.delta_q_present,
            self.base_q_idx as i32,
            qstep_i,
            if enc.delta_q_present && !needs_partition {
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
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                if !needs_partition {
                    // Whole 64×64 luma SB → one 32×64 (TX_32X64) chroma TU per plane.
                    let (fmr, fmc) = (row * 16, col * 16);
                    let (sb_qstep, sb_resid_scale) =
                        aqs.per_sb(&mut enc, &yp, pw, sb_y, sb_x, width, height);
                    let ua = if fmr > 0 { u_above[fmc] } else { 0 };
                    let ul = if fmc > 0 { u_left[fmr] } else { 0 };
                    let va = if fmr > 0 { v_above[fmc] } else { 0 };
                    let vl = if fmc > 0 { v_left[fmr] } else { 0 };
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
                    let (skip_cdfs, dc_sign_ctxs) =
                        sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr);
                    let (cy, cx) = (sb_y, sb_x / 2);
                    let bd = self.bit_depth as i32;
                    // CfL decision (4:2:2 whole-64 fast path). 32x64 chroma; luma subsampled
                    // horizontally (<<2); neighbor DC via cfl_avg_l(ssx=true, ssy=false).
                    let cfl_a = if fmr > 0 { cfl_above[fmc] } else { 0 };
                    let cfl_l = if fmc > 0 { cfl_left[fmr] } else { 0 };
                    enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                    let cfl_choice = if enc.cfl {
                        let avg_l = cfl::cfl_avg_l(&recy, pw, sb_y, sb_x, 32, 64, true, false, bd);
                        let mut suf = [0f32; 32 * 64];
                        let mut svf = [0f32; 32 * 64];
                        for r in 0..64 {
                            let b = (cy + r) * pcw + cx;
                            for c in 0..32 {
                                suf[r * 32 + c] = up[b + c];
                                svf[r * 32 + c] = vp[b + c];
                            }
                        }
                        let dc_u_f =
                            dc_pred_rect_subsampled(&recu, pcw, cy, cx, 32, 64, neutral, bd);
                        let dc_v_f =
                            dc_pred_rect_subsampled(&recv, pcw, cy, cx, 32, 64, neutral, bd);
                        cfl::cfl_decide(
                            &recy,
                            pw,
                            sb_y,
                            sb_x,
                            &suf,
                            &svf,
                            dc_u_f,
                            dc_v_f,
                            32,
                            64,
                            true,
                            false,
                            avg_l,
                            bd,
                            &bases.chroma422,
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
                    enc.delta_q_pending = enc.delta_q_present;
                    encode_luma_block_split(
                        &mut enc,
                        &tus,
                        &skip_cdfs,
                        &dc_sign_ctxs,
                        mode_idx,
                        true,
                        12276,
                    );
                    let scan = &tables::SCAN;
                    let (levu, levv) = if let Some(ref ch) = cfl_choice {
                        let mut ru = [0f32; 32 * 64];
                        let mut rv = [0f32; 32 * 64];
                        cfl_prediction::<32>(pcw, &up, &vp, cy, cx, &ch, &mut ru, &mut rv);
                        let levu = bases.chroma422.project_scan(
                            &aq::scale_resid(&ru, sb_resid_scale),
                            0.0,
                            scan,
                        );
                        let levv = bases.chroma422.project_scan(
                            &aq::scale_resid(&rv, sb_resid_scale),
                            0.0,
                            scan,
                        );
                        put_block_rect(
                            &mut recu,
                            pcw,
                            cy,
                            cx,
                            32,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &ch.pred_u, &levu, sb_qstep, scan, 32, 64, bd,
                            ),
                        );
                        put_block_rect(
                            &mut recv,
                            pcw,
                            cy,
                            cx,
                            32,
                            64,
                            &itx422::reconstruct_chroma_cfl(
                                &ch.pred_v, &levv, sb_qstep, scan, 32, 64, bd,
                            ),
                        );
                        (levu, levv)
                    } else {
                        let predu = dc_pred_rect(&recu, pcw, cy, cx, 32, 64, neutral, bd);
                        let levu = bases.chroma422.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(&up, pcw, cy, cx, 32, 64, predu),
                                sb_resid_scale,
                            ),
                            0.0,
                            scan,
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
                                sb_qstep,
                                scan,
                                32,
                                64,
                                &bases.chroma422,
                                bd,
                            ),
                        );
                        let predv = dc_pred_rect(&recv, pcw, cy, cx, 32, 64, neutral, bd);
                        let levv = bases.chroma422.project_scan(
                            &aq::scale_resid(
                                &get_residual_rect(&vp, pcw, cy, cx, 32, 64, predv),
                                sb_resid_scale,
                            ),
                            0.0,
                            scan,
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
                                sb_qstep,
                                scan,
                                32,
                                64,
                                &bases.chroma422,
                                bd,
                            ),
                        );
                        (levu, levv)
                    };
                    let (uc, vc) = (levels_to_coeffs(&levu), levels_to_coeffs(&levv));
                    let u_skip = CHROMA_SKIP_TX64_QC[qc][(6 + ua + ul) as usize] as u32;
                    encode_chroma_block_rect(
                        &mut enc,
                        &uc,
                        u_skip,
                        true,
                        &tables::SCAN,
                        EobCdf::ChrEobBin,
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
                        &tables::SCAN,
                        EobCdf::ChrEobBin,
                        CHROMA_EOB_HI_BIT_QC[qc],
                        1024,
                    );
                    let v_present = vc.iter().any(|&(_, l)| l != 0);
                    let cfl_used = cfl_choice.is_some() as i32;
                    for c in fmc..fmc + 16 {
                        u_above[c] = up_ as i32;
                        v_above[c] = v_present as i32;
                        cfl_above[c] = cfl_used;
                    }
                    for r in fmr..fmr + 16 {
                        u_left[r] = up_ as i32;
                        v_left[r] = v_present as i32;
                        cfl_left[r] = cfl_used;
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
                // Edge SBs: quantization-neutral, but emit delta_q (0) once per SB.
                enc.delta_q_signaled = 0;
                enc.delta_q_pending = enc.delta_q_present;
                for op in &ops {
                    let (bw_mi, bh_mi, pc, lmr, lmc) = match op {
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
                    let sb_y = lmr * 4;
                    let sb_x = lmc * 4;
                    let ua = if lmr > 0 { u_above[lmc] } else { 0 };
                    let ul = if lmc > 0 { u_left[lmr] } else { 0 };
                    let va = if lmr > 0 { v_above[lmc] } else { 0 };
                    let vl = if lmc > 0 { v_left[lmr] } else { 0 };
                    let (cy, cx) = (sb_y, sb_x / 2);
                    let cfl_a = if lmr > 0 { cfl_above[lmc] } else { 0 };
                    let cfl_l = if lmc > 0 { cfl_left[lmr] } else { 0 };
                    enc.cfl_ctx = (cfl_a + cfl_l) as usize;
                    enc.cfl_use = false;
                    enc.mhccp_use = false;
                    let (u_present, v_present) = match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx, _) = encode_luma_sb(
                                &mut recy,
                                &yp,
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
                                false, // non-directional path
                            );
                            let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                                &tus, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            const PARTITION_CFL: bool = true;
                            let bd = self.bit_depth as i32;
                            let cfl_choice = if enc.cfl && PARTITION_CFL {
                                let avg_l =
                                    cfl::cfl_avg_l(&recy, pw, sb_y, sb_x, 32, 64, true, false, bd);
                                let mut suf = [0f32; 32 * 64];
                                let mut svf = [0f32; 32 * 64];
                                cfl_partition_prediction::<32>(
                                    pcw, &up, &vp, cy, cx, &mut suf, &mut svf,
                                );
                                let dc_u_f = dc_pred_rect_subsampled(
                                    &recu, pcw, cy, cx, 32, 64, neutral, bd,
                                );
                                let dc_v_f = dc_pred_rect_subsampled(
                                    &recv, pcw, cy, cx, 32, 64, neutral, bd,
                                );
                                cfl::cfl_decide(
                                    &recy,
                                    pw,
                                    sb_y,
                                    sb_x,
                                    &suf,
                                    &svf,
                                    dc_u_f,
                                    dc_v_f,
                                    32,
                                    64,
                                    true,
                                    false,
                                    avg_l,
                                    bd,
                                    &bases.chroma422,
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
                                    scan: &tables::SCAN,
                                    eob_cdf: EobCdf::ChrEobBin,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 1024,
                                    u_skip_row: &CHROMA_SKIP_TX64_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                cfl_choice.as_ref(),
                                None,
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
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_64x32(
                                &tus2, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                32,
                                32,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.chroma420,
                                &tables::SCAN,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            let uv_pred = if mh_choice.is_none() && self.tune.chroma_mode_search {
                                chroma422::chroma_mode_decide_leaf(
                                    &mut enc,
                                    &recu,
                                    &recv,
                                    &up,
                                    &vp,
                                    pcw,
                                    cy,
                                    cx,
                                    32,
                                    neutral,
                                    &bases.chroma420,
                                    &tables::SCAN,
                                    qc,
                                    qstep_i,
                                    qstep_i,
                                    1.0,
                                    self.tune.chroma_rdoq_lambda,
                                    leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                    width / 2,
                                    height,
                                    self.speed.reduced_modes(),
                                    self.speed.chroma_angle_directional(),
                                    self.bit_depth as i32,
                                )
                            } else {
                                None
                            };
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
                                    scan: &tables::SCAN,
                                    eob_cdf: EobCdf::ChrEobBin,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 1024,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                uv_pred.as_ref().map(|(u, v)| (u.as_slice(), v.as_slice())),
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
                                    scan: &SCAN16X32,
                                    eob_cdf: EobCdf::ChrEob512,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 512,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                                None,
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
                                &mut above,
                                &mut left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                16,
                                32,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c16x32,
                                &SCAN16X32,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
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
                                    scan: &SCAN16X32,
                                    eob_cdf: EobCdf::ChrEob512,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 512,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (4, 16) => {
                            // Right-edge 16×64 luma leaf → 4:2:2 chroma 8×64 (TX_8X64,
                            // coeff 8×32, SCAN8X32, eob 256, skip class 3).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                &itx422::reconstruct_chroma(
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
                                    scan: &SCAN8X32,
                                    eob_cdf: EobCdf::ChrEob256,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                                None,
                            )
                        }
                        (16, 4) => {
                            // Bottom-edge 64×16 luma leaf → 4:2:2 chroma 32×16 (TX_32X16,
                            // coeff 32×16, SCAN32X16, eob 512, skip class 3).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
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
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 16, 4, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                32,
                                16,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c32x16,
                                &SCAN32X16,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
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
                                    scan: &SCAN32X16,
                                    eob_cdf: EobCdf::ChrEob512,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 512,
                                    u_skip_row: &CHROMA_SKIP_TX32_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (4, 4) => {
                            // Bottom-right 16×16 corner leaf (residue 4 in both dims).
                            // Full-AC native TX_16X16 luma (entropy class 2, eob class 256),
                            // tx_type RD-chosen among DCT_DCT / ADST_ADST / ADST_DCT /
                            // DCT_ADST (EXT_NEW_TX_SET, DC mode); 4:2:2 chroma 8×16 stays
                            // DCT (tx_type is luma-only).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let resid = get_residual_rect(&yp, pw, sb_y, sb_x, 16, 16, pred);
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
                            // DCT_DCT candidate (idx 0).
                            let lev_dct = bases.luma16x16.project_scan(&resid, 0.0, &SCAN16);
                            let rec_dct = itx422::reconstruct_luma16(
                                &pred_flat,
                                &lev_dct,
                                qstep_i,
                                &SCAN16,
                                self.bit_depth as i32,
                            );
                            let cost_dct = sse(&rec_dct) + lambda * rate(&lev_dct);
                            // ADST_ADST candidate (idx 1, DST-VII both axes).
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
                            // ADST_DCT candidate (idx 2: ADST vertical, DCT horizontal →
                            // inverse row_adst=false, col_adst=true; ~3.1 extra bits).
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
                            // DCT_ADST candidate (idx 3: DCT vertical, ADST horizontal →
                            // inverse row_adst=true, col_adst=false; ~2.7 extra bits).
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
                            // Strict-improvement tie-break (DCT_DCT default).
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
                            put_block_rect(&mut recy, pw, sb_y, sb_x, 16, 16, rec);
                            let tu: Vec<Coeff> = levels_to_coeffs(lev);
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 4, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                8,
                                16,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c8x16,
                                &tables::SCAN8X16,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            encode_luma_leaf_16x16_full(
                                &mut enc, &tu, skip, dcs, 0, true, pc, 11074, tx_idx,
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
                                    scan: &tables::SCAN8X16,
                                    eob_cdf: EobCdf::ChrEob128,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 128,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (2, 8) => {
                            // Right-edge 8×32 DC-only luma leaf → 4:2:2 chroma 4×32
                            // (TX_4X32, coeff 4×32, SCAN4X32, eob 128 NO-ESCAPE, class 2).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma8x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 32, pred),
                                0.0,
                                &SCAN8X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN8X32,
                                    8,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 8, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                4,
                                32,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c4x32,
                                &tables::SCAN4X32,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            encode_luma_leaf_8x32(&mut enc, &tu, skip, dcs, 0, true, pc);
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
                                    scan: &tables::SCAN4X32,
                                    eob_cdf: EobCdf::ChrEob128,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 128,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (8, 2) => {
                            // Bottom-edge 32×8 DC-only luma leaf → 4:2:2 chroma 16×8
                            // (TX_16X8, coeff 16×8, SCAN16X8, eob 128 NO-ESCAPE, class 2).
                            let pred = dc_pred_rect(
                                &recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma32x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 32, 8, pred),
                                0.0,
                                &SCAN32X8,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN32X8,
                                    32,
                                    8,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 8, 2, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                16,
                                8,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c16x8,
                                &tables::SCAN16X8,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            encode_luma_leaf_32x8(&mut enc, &tu, skip, dcs, 0, true, pc);
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
                                    scan: &tables::SCAN16X8,
                                    eob_cdf: EobCdf::ChrEob128,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 128,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (2, 2) => {
                            // Both-axis residue-2 corner: 8×8 luma (TX_8X8) + 4×8 chroma per
                            // plane (4:2:2: half-width, full-height → TX_4X8).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 8, neutral, bd);
                            let lev = bases.c8x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 8, pred),
                                0.0,
                                &tables::SCAN8X8,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                8,
                                &itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &tables::SCAN8X8,
                                    8,
                                    8,
                                    bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 2, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                4,
                                8,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.c4x8,
                                &tables::SCAN4X8,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            encode_luma_leaf_8x8(
                                &mut enc,
                                &tu,
                                skip,
                                dcs,
                                0,
                                true,
                                pc,
                                3148,
                                Some((&crate::av2::coder::TXTP_EXT8, 0, 6)),
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
                                    ch: 8,
                                    basis: &bases.c4x8,
                                    scan: &tables::SCAN4X8,
                                    eob_cdf: EobCdf::ChrEob32,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 32,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (2, 4) => {
                            // residue-2 W × residue-4 H: 8×16 luma (TX_8X16) + 4×16 chroma
                            // per plane (4:2:2 half-width/full-height → TX_4X16, eob64 ctx1).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 8, 16, neutral, bd);
                            let lev = bases.c8x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 8, 16, pred),
                                0.0,
                                &crate::av2::tables::SCAN8X16,
                            );
                            put_block_rect(
                                &mut recy,
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
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 2, 4, true,
                            );
                            crate::av2::coder::encode_luma_leaf_rect128(
                                &mut enc,
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
                                    ch: 16,
                                    basis: &bases.c4x16,
                                    scan: &tables::SCAN4X16,
                                    eob_cdf: EobCdf::ChrEob64,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 64,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                                None,
                            )
                        }
                        (4, 2) => {
                            // residue-4 W × residue-2 H: 16×8 luma (TX_16X8) + 8×8 chroma
                            // per plane (4:2:2 half-width/full-height → TX_8X8, eob64 ctx1).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 8, neutral, bd);
                            let lev = bases.c16x8.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 8, pred),
                                0.0,
                                &crate::av2::tables::SCAN16X8,
                            );
                            put_block_rect(
                                &mut recy,
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
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 2, true,
                            );
                            let uv_pred = if self.tune.chroma_mode_search {
                                chroma422::chroma_mode_decide_leaf(
                                    &mut enc,
                                    &recu,
                                    &recv,
                                    &up,
                                    &vp,
                                    pcw,
                                    cy,
                                    cx,
                                    8,
                                    neutral,
                                    &bases.c8x8,
                                    &tables::SCAN8X8,
                                    qc,
                                    qstep_i,
                                    qstep_i,
                                    1.0,
                                    self.tune.chroma_rdoq_lambda,
                                    leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                    width / 2,
                                    height,
                                    self.speed.reduced_modes(),
                                    self.speed.chroma_angle_directional(),
                                    self.bit_depth as i32,
                                )
                            } else {
                                None
                            };
                            crate::av2::coder::encode_luma_leaf_rect128(
                                &mut enc,
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
                                    ch: 8,
                                    basis: &bases.c8x8,
                                    scan: &tables::SCAN8X8,
                                    eob_cdf: EobCdf::ChrEob64,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 64,
                                    u_skip_row: &SKIP_TX8_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                None,
                                uv_pred.as_ref().map(|(u, v)| (u.as_slice(), v.as_slice())),
                            )
                        }
                        (4, 8) => {
                            // residue-4 W × residue-{6,8} H: 16×32 luma (TX_16X32 long-side-32)
                            // + 8×32 chroma per plane (4:2:2 half-width → TX_8X32, eob256 ctx2).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 16, 32, neutral, bd);
                            let lev = bases.luma16x32.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 16, 32, pred),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                32,
                                &itx422::reconstruct_chroma(
                                    pred, &lev, qstep_i, &SCAN16X32, 16, 32, bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 4, 8, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                8,
                                32,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.luma8x32,
                                &SCAN8X32,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            encode_luma_leaf_16x32(&mut enc, &tu, skip, dcs, 0, true, pc);
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
                                    ch: 32,
                                    basis: &bases.luma8x32,
                                    scan: &SCAN8X32,
                                    eob_cdf: EobCdf::ChrEob256,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                None,
                            )
                        }
                        (8, 4) => {
                            // residue-{6,8} W × residue-4 H: 32×16 luma (TX_32X16 long-side-32)
                            // + 16×16 chroma per plane (4:2:2 half-width → TX_16X16, eob256 ctx2).
                            let bd = self.bit_depth as i32;
                            let pred = dc_pred_rect(&recy, pw, sb_y, sb_x, 32, 16, neutral, bd);
                            let lev = bases.luma32x16.project_scan(
                                &get_residual_rect(&yp, pw, sb_y, sb_x, 32, 16, pred),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                &mut recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                16,
                                &itx422::reconstruct_chroma(
                                    pred, &lev, qstep_i, &SCAN32X16, 32, 16, bd,
                                ),
                            );
                            let tu: Vec<Coeff> = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, &mut above, &mut left, qc, tmc, tmr, 8, 4, true,
                            );
                            let mh_choice = chroma422::mhccp_decide_leaf(
                                &mut enc,
                                &recy,
                                &recu,
                                &recv,
                                &up,
                                &vp,
                                pw,
                                pcw,
                                sb_y,
                                sb_x,
                                cy,
                                cx,
                                16,
                                16,
                                true,
                                false,
                                lmr > 0,
                                lmc > 0,
                                neutral,
                                &bases.luma16x16,
                                &tables::SCAN16,
                                qstep_i,
                                leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                self.bit_depth as i32,
                            );
                            let uv_pred = if mh_choice.is_none() && self.tune.chroma_mode_search {
                                chroma422::chroma_mode_decide_leaf(
                                    &mut enc,
                                    &recu,
                                    &recv,
                                    &up,
                                    &vp,
                                    pcw,
                                    cy,
                                    cx,
                                    16,
                                    neutral,
                                    &bases.luma16x16,
                                    &tables::SCAN16,
                                    qc,
                                    qstep_i,
                                    qstep_i,
                                    1.0,
                                    self.tune.chroma_rdoq_lambda,
                                    leaf::part_lambda(qstep_i, self.tune.part_lambda_c),
                                    width / 2,
                                    height,
                                    self.speed.reduced_modes(),
                                    self.speed.chroma_angle_directional(),
                                    self.bit_depth as i32,
                                )
                            } else {
                                None
                            };
                            encode_luma_leaf_32x16(&mut enc, &tu, skip, dcs, 0, true, pc);
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
                                    ch: 16,
                                    basis: &bases.luma16x16,
                                    scan: &tables::SCAN16,
                                    eob_cdf: EobCdf::ChrEob256,
                                    eob_hi: CHROMA_EOB_HI_BIT_QC[qc],
                                    area: 256,
                                    u_skip_row: &SKIP_TX16_QC[qc],
                                },
                                QuantCtx {
                                    qc,
                                    neutral,
                                    qstep: qstep_i,
                                    rdoq_lambda: self.tune.chroma_rdoq_lambda,
                                },
                                ChromaNeighbors { ua, ul, va, vl },
                                self.bit_depth as i32,
                                mh_choice.as_ref(),
                                uv_pred.as_ref().map(|(u, v)| (u.as_slice(), v.as_slice())),
                            )
                        }
                        other => unreachable!("unsupported native 4:2:2 leaf {:?}", other),
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
        enc
    }

    /// Multi-tile 4:2:2 assembly. Each tile is an independent sub-frame encode; tiles
    /// run in parallel across `threads` workers (raster order preserved). 4:2:2 chroma
    /// is half-width/full-height, so a luma tile at `(x0, tw)` maps to chroma
    /// `(x0/2, tw/2)` — both even because SB boundaries and 4:2:2 width are even.
    #[allow(clippy::too_many_arguments)]
    fn encode_422_tiled(
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
        // See encode_444_tiled: pad the whole frame SB-aligned (one frame-level clap)
        // when any edge tile isn't boundary-exact; otherwise signal the real size.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| native_422_mi(tw, th).is_some());
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
                    pad_plane(cbf, width.div_ceil(2), height, pw / 2, ph),
                    pad_plane(crf, width.div_ceil(2), height, pw / 2, ph),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let cw = cstride; // chroma plane stride (4:2:2)
        let n = specs.len();
        let mut tiles_bytes: Vec<Vec<u8>> = vec![Vec::new(); n];
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        if nthreads <= 1 || n <= 1 {
            for (slot, &(x0, y0, tw, th)) in tiles_bytes.iter_mut().zip(&specs) {
                let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                let tv = extract_subplane(crf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                *slot = self.encode_422_core(&ty, &tu, &tv, tw, th).finish();
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
                            let tu = extract_subplane(cbf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                            let tv = extract_subplane(crf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                            *slot = me.encode_422_core(&ty, &tu, &tv, tw, th).finish();
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
            ChromaFormat::Yuv422,
            &tiles_bytes,
        )
    }

    /// Lossless 4:2:2: identical recursion to the 4:4:4 lossless path, but the chroma
    /// planes are half-width, so each luma leaf of `lcols × lrows` 4×4 WHT TUs maps to a
    /// `(lcols/2) × lrows` chroma block at chroma column `lc/2`. The mode-info grid is
    /// 8px-aligned so every leaf starts on an even mi column, keeping the halving exact.
    fn encode_yuv422_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
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
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = true;
        enc.mhccp_ssy = false;
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        // Chroma neighbor grids live in chroma-pixel space (half-width above grid).
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
                planes: [y, cb, cr, Vec::new()],
            },
            color,
        )
    }
}
