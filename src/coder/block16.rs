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

impl<'a> LossyTile<'a> {
    /// Shared header + luma for a TX_16X16 block: codes the block-level skip
    /// flag, `DC_PRED` y/uv modes, the luma TX_16X16 coefficients, updates the
    /// 4-unit (16-sample) luma skip/coef footprint, and reconstructs luma. The
    /// caller has already decided `block_skip` (needs all planes) and passes the
    /// luma coefficients + DC prediction.
    /// Emit the chroma `uv_mode` symbol: plain DC (`None`) or CfL (`Some(alphas)`),
    /// in which case also the joint-sign and per-plane magnitude symbols.
    /// `uv_delta` is the chroma angle delta coded for a directional `uv_mode`.
    #[allow(clippy::too_many_arguments)]
    fn emit_uv_mode(
        &mut self,
        y_mode: usize,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        uv_delta: i32,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
    ) {
        let mut coded_mode = uv_mode;
        match cfl {
            Some(a) => {
                let su = if a[0] == 0 {
                    0
                } else if a[0] < 0 {
                    1
                } else {
                    2
                };
                let sv = if a[1] == 0 {
                    0
                } else if a[1] < 0 {
                    1
                } else {
                    2
                };
                let sign = su * 3 + sv;
                if sign == 0 {
                    // Both alphas zero: zero-alpha CfL reconstructs exactly as DC
                    // prediction, so signal plain DC and avoid an invalid
                    // `sign - 1` joint-sign symbol.
                    self.ent.enc
                        .encode_symbol(DC_PRED, &mut self.ent.cdfs.uv_mode[13 + y_mode]);
                    coded_mode = DC_PRED;
                } else {
                    self.ent.enc
                        .encode_symbol(CFL_PRED, &mut self.ent.cdfs.uv_mode[13 + y_mode]);
                    self.ent.enc.encode_symbol(sign - 1, &mut self.ent.cdfs.cfl_sign);
                    if su != 0 {
                        let c = (su == 2) as usize * 3 + sv;
                        self.ent.enc
                            .encode_symbol((a[0].abs() - 1) as usize, &mut self.ent.cdfs.cfl_alpha[c]);
                    }
                    if sv != 0 {
                        let c = (sv == 2) as usize * 3 + su;
                        self.ent.enc
                            .encode_symbol((a[1].abs() - 1) as usize, &mut self.ent.cdfs.cfl_alpha[c]);
                    }
                    coded_mode = CFL_PRED;
                }
            }
            None => {
                self.ent.enc
                    .encode_symbol(uv_mode, &mut self.ent.cdfs.uv_mode[13 + y_mode]);
                // Directional chroma modes (here only V_PRED/H_PRED, used at 8x8
                // 4:4:4 chroma where `use_angle_delta` holds) emit a chroma
                // angle_delta symbol. The encoder only offers delta 0, so emit the
                // center bucket (delta + 3 == 3).
                if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
                    self.ent.enc.encode_symbol(
                        (uv_delta + 3) as usize,
                        &mut self.ent.cdfs.angle_delta[uv_mode - V_PRED],
                    );
                }
            }
        }
        self.commit_uv_mode(px, py, w, h, coded_mode);
    }

    /// Emit a 16x16 block's header and luma: skip flag, modes, palette and
    /// filter-intra syntax, transform depth, then the luma coefficients and
    /// reconstruction of the winner `l`.
    #[allow(clippy::too_many_arguments)]
    fn code_header_luma16(
        &mut self,
        x8: usize,
        y8: usize,
        l: &Luma16<'_>,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        uv_delta: i32,
        uv_palette: Option<&LossyUvPalette>,
    ) {
        let Luma16 {
            cf: lcf,
            pred: lpred,
            y_mode,
            txtp16,
            s8_txtps,
            s16_txtps,
            angle_delta,
            filter_intra,
            palette,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 2, block_skip);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc.encode_symbol(
                (angle_delta + 3) as usize,
                &mut self.ent.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        if !self.cfg.is_mono() {
            self.emit_uv_mode(y_mode, uv_mode, cfl, uv_delta, px, py, 16, 16);
        }
        self.emit_palette_mode_info(px, py, 16, 16, y_mode, !self.cfg.is_mono(), palette, uv_palette);
        if palette.is_none() {
            self.emit_filter_intra(y_mode, 16, 16, filter_intra);
        }
        if let Some(p) = palette {
            self.emit_palette_map(p);
        }
        if let Some(up) = uv_palette {
            self.emit_palette_uv_map(up);
        }
        self.code_tx_depth(
            px,
            py,
            16,
            16,
            match txtp16 {
                4 => 1,
                6 => 2,
                _ => 0,
            },
        );
        // Intra-edge smooth-filter flag: dav1d derives it ONCE at the block
        // origin from the NEIGHBOR modes, before the per-TX loop — capture it
        // before the a_mode/l_mode fills below overwrite the neighbor state.
        let block_ftype = self.luma_filter_type(px, py);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.ctx.a_skip[bx4..bx4 + 4].fill(sv);
        self.ctx.l_skip[by4..by4 + 4].fill(sv);
        self.ctx.a_mode[bx4..bx4 + 4].fill(mv);
        self.ctx.l_mode[by4..by4 + 4].fill(mv);
        if txtp16 == 6 {
            // tx_depth = 2: SIXTEEN TX_4X4 in raster order, per-TX prediction
            // and per-TX txtp4 (see split16_depth2_try).
            for j in 0..4usize {
                for i in 0..4usize {
                    let ti = j * 4 + i;
                    let (bx, by) = (px + i * 4, py + j * 4);
                    let (cx4, cy4) = (bx / 4, by / 4);
                    let mut cfq = [0i32; 16];
                    cfq.copy_from_slice(&lcf[ti * 16..ti * 16 + 16]);
                    let res_ctx = if block_skip {
                        0x40
                    } else {
                        let sk = self.skip_ctx_split(cx4, cy4, 1, 1);
                        let ds = self.dc_sign_ctx_420(0, cx4, cy4);
                        let t = s16_txtps[ti];
                        if t == 2 || t == 3 {
                            encode_tx4_coeffs_1d(
                                &mut self.ent.enc,
                                &mut self.ent.cdfs,
                                &cfq,
                                t == 2,
                                sk,
                                ds,
                                y_mode,
                            )
                        } else {
                            encode_tx4_luma_coeffs_adapt(
                                &mut self.ent.enc,
                                &mut self.ent.cdfs,
                                &cfq,
                                sk,
                                ds,
                                y_mode,
                                t as usize,
                            )
                        }
                    };
                    self.ctx.a_coef[0][cx4] = res_ctx;
                    self.ctx.l_coef[0][cy4] = res_ctx;
                    let tr = if j == 0 {
                        py > 0 && (i < 3 || have_tr)
                    } else {
                        i < 3
                    };
                    let bl = if i == 0 {
                        px > 0 && (j < 3 || have_bl)
                    } else {
                        false
                    };
                    let mut pred = [0i32; 16];
                    if y_mode == DC_PRED && angle_delta == 0 {
                        let d = self.k.intrapred.dc_pred_4x4(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                        pred = [d; 16];
                    } else {
                        self.k.intrapred.predict_nd_ad(
                            y_mode,
                            angle_delta,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            bx,
                            by,
                            4,
                            4,
                            tr,
                            bl,
                            self.cfg.w,
                            self.cfg.h,
                            block_ftype,
                            &mut pred,
                            self.cfg.bd,
                        );
                    }
                    let rr = if block_skip {
                        [0i32; 16]
                    } else {
                        match s16_txtps[ti] {
                            4 => self.k.idct.iadst_dequant_4x4(&cfq, &self.ctx.quant),
                            0 => self.k.idct.iidentity_dequant_4x4(&cfq, &self.ctx.quant),
                            2 => self.k.idct.ivdct_dequant_4x4(&cfq, &self.ctx.quant),
                            3 => self.k.idct.ihdct_dequant_4x4(&cfq, &self.ctx.quant),
                            _ => self.k.idct.idct_dequant_4x4(&cfq, &self.ctx.quant),
                        }
                    };
                    self.k.rd.reconstruct(
                        &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                        self.cfg.w,
                        None,
                        &pred,
                        &rr,
                        4,
                        4,
                        self.cfg.bd,
                    );
                }
            }
            return;
        }
        if txtp16 == 4 {
            // tx_depth = 1: four TX_8X8 in raster order, DCT sub-transforms,
            // coefficients packed quadrant-major in `lcf`. Per the spec, intra
            // prediction runs per TRANSFORM block: each quadrant predicts from
            // the running reconstruction (including earlier quadrants), with
            // per-quadrant edge availability mirroring dav1d's per-TX flags.
            for (qi, &(sx, sy)) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)]
                .iter()
                .enumerate()
            {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let mut cfq = [0i32; 64];
                cfq.copy_from_slice(&lcf[qi * 64..qi * 64 + 64]);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
                    let ds = self.dc_sign_ctx(0, qbx4, qby4);
                    let t = s8_txtps[qi];
                    if t == 2 || t == 3 {
                        encode_tx8_coeffs_1d(
                            &mut self.ent.enc,
                            &mut self.ent.cdfs,
                            &cfq,
                            t == 2,
                            sk,
                            ds,
                            filter_intra_tx_mode(None, y_mode),
                        )
                    } else {
                        encode_tx8_coeffs_adapt(
                            &mut self.ent.enc,
                            &mut self.ent.cdfs,
                            &cfq,
                            false,
                            sk,
                            ds,
                            filter_intra_tx_mode(None, y_mode),
                            t as usize,
                        )
                    }
                };
                self.ctx.a_coef[0][qbx4] = res_ctx;
                self.ctx.a_coef[0][qbx4 + 1] = res_ctx;
                self.ctx.l_coef[0][qby4] = res_ctx;
                self.ctx.l_coef[0][qby4 + 1] = res_ctx;
                let (tr, bl) = match (sx, sy) {
                    (0, 0) => (py > 0, px > 0),
                    (8, 0) => (have_tr, false),
                    (0, 8) => (true, have_bl),
                    _ => (false, false),
                };
                let mut pred = [0i32; 64];
                if y_mode == DC_PRED {
                    let d = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                    pred = [d; 64];
                } else {
                    self.k.intrapred.predict_nd_ad(
                        y_mode,
                        angle_delta,
                        &self.ctx.recon[0],
                        self.cfg.w,
                        bx,
                        by,
                        8,
                        8,
                        tr,
                        bl,
                        self.cfg.w,
                        self.cfg.h,
                        block_ftype,
                        &mut pred,
                        self.cfg.bd,
                    );
                }
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    match s8_txtps[qi] {
                        4 => self.k.idct.iadst_dequant_8x8(&cfq, &self.ctx.quant),
                        5 => self.k.idct.iadstdct_dequant_8x8(&cfq, &self.ctx.quant),
                        6 => self.k.idct.idctadst_dequant_8x8(&cfq, &self.ctx.quant),
                        0 => self.k.idct.iidentity_dequant_8x8(&cfq, &self.ctx.quant),
                        2 => self.k.idct.ivdct_dequant_8x8(&cfq, &self.ctx.quant),
                        3 => self.k.idct.ihdct_dequant_8x8(&cfq, &self.ctx.quant),
                        _ => self.k.idct.idct_dequant_8x8(&cfq, &self.ctx.quant),
                    }
                };
                self.k.rd.reconstruct(
                    &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                    self.cfg.w,
                    None,
                    &pred,
                    &rr,
                    8,
                    8,
                    self.cfg.bd,
                );
            }
            return;
        }
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_16(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_16(0, bx4, by4);
            encode_tx16_coeffs_adapt(
                &mut self.ent.enc,
                &mut self.ent.cdfs,
                lcf,
                false,
                sk,
                ds,
                filter_intra_tx_mode(filter_intra, y_mode),
                match txtp16 {
                    1 => ADST_ADST_TX16_IDX,
                    2 => ADST_DCT_TX16_IDX,
                    3 => DCT_ADST_TX16_IDX,
                    5 => IDTX_TX16_IDX,
                    _ => 1,
                },
            )
        };
        self.ctx.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
        self.ctx.l_coef[0][by4..by4 + 4].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 256]
        } else {
            match txtp16 {
                1 => self.k.idct.iadst_dequant_16x16(lcf, &self.ctx.quant),
                2 => self.k.idct.iadstdct_dequant_16x16(lcf, &self.ctx.quant),
                3 => self.k.idct.idctadst_dequant_16x16(lcf, &self.ctx.quant),
                5 => self.k.idct.iidentity_dequant_16x16(lcf, &self.ctx.quant),
                _ => self.k.idct.idct_dequant_16x16(lcf, &self.ctx.quant),
            }
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lrr.as_chunks::<16>().0.iter())
            .enumerate()
        {
            let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
            recon_add_pred(drow, prow, rrow, (1 << self.cfg.bd) - 1);
        }
    }

    /// Chroma search of a 16x16 block (444) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block16_444`.
    fn pick_uv16_444(&self, x8: usize, y8: usize, l: &Luma16<'_>) -> UvPick<SBuf<[i32; 256]>, SBuf<[i32; 256]>> {
        let Luma16 {
            cf: lcf,
            pred: lpred,
            y_mode,
            txtp16,
            palette,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let mut ccf = [self.sbuf_i256(), self.sbuf_i256()];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[plane], self.cfg.w, px, py, self.cfg.bd as i32);
            cpred[ci] = pred;
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_dc(
                &mut resid[..],
                &self.cfg.src[plane],
                self.cfg.w,
                px,
                py,
                16,
                16,
                pred,
            );
            let (q, qt) = self.k.dct.dct16x16_t(&resid, &self.ctx.cquant);
            *ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci][..],
                &qt,
                self.ctx.cquant.dc_q() as f32,
                self.ctx.cquant.ac_q() as f32,
                &SCAN_16X16,
                self.trellis_lambda(),
            );
            self.k.rd.preserve_dc(&mut ccf[ci][0], &resid[..]);
        }
        // 4:4:4 CfL for the 16x16 chroma blocks (mirrors the 8x8 path).
        let mut cpred16 = [self.sbuf_i256(), self.sbuf_i256()];
        let mut cfl_opt: Option<[i32; 2]> = None;
        if self.cfg.speed.full_chroma_rdo() {
            // Luma reconstruction as the decoder will see it — with TX split the
            // block reconstructs per sub-TX (sequential per-quadrant prediction),
            // so `lpred + inverse(lcf)` no longer applies.
            let mut luma_rec = self.sbuf_u256();
            if txtp16 == 4 || txtp16 == 6 {
                *luma_rec = *l.split_rec.expect("a TX-split luma winner carries its reconstruction");
            } else {
                let lrr_cfl = match txtp16 {
                    1 => self.k.idct.iadst_dequant_16x16(lcf, &self.ctx.quant),
                    2 => self.k.idct.iadstdct_dequant_16x16(lcf, &self.ctx.quant),
                    3 => self.k.idct.idctadst_dequant_16x16(lcf, &self.ctx.quant),
                    5 => self.k.idct.iidentity_dequant_16x16(lcf, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_16x16(lcf, &self.ctx.quant),
                };
                recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, (1 << self.cfg.bd) - 1);
            }
            let mut ac = self.sbuf_i256();
            self.k.intrapred
                .cfl_ac_444(&luma_rec[..], 16, 16, &mut ac[..]);
            let (dcq, acq, lam) = (
                self.ctx.cquant.dc_q() as f32,
                self.ctx.cquant.ac_q() as f32,
                self.trellis_lambda(),
            );
            let mut cfl_ccf = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f32; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f32; 2]);
            let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
            let steps = crate::tuning::get().cfl_alpha_rd_steps;
            // Energy-optimal alphas for both planes up front: the rate-aware
            // walk prices each plane's alpha bits against the other plane's
            // current alpha (one coordinate-descent pass).
            let a_e = [0, 1].map(|ci| {
                self.k.intrapred.cfl_best_alpha(
                    &ac[..],
                    self.src_blk(ci + 1, px, py, 16, 16),
                    cpred[ci],
                    self.cfg.bd,
                )
            });
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let src = self.src_blk(plane, px, py, 16, 16);
                let dcrr = self.k.idct.idct_dequant_16x16(&ccf[ci], &self.ctx.cquant);
                dc_sse[ci] = sse_recon::<256, 16>(&self.k.rd, &[dc; 256], &dcrr, src, self.cfg.bd);
                dc_bits[ci] = self.chroma_bits(&ccf[ci][..], &SCAN_16X16, 16, plane, px, py);
                let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
                let realise = |a: i32| {
                    let mut cpr = self.sbuf_i256();
                    self.k.intrapred.cfl_pred(&mut cpr[..], &ac[..256], dc, a, self.cfg.bd);
                    let mut resid = self.sbuf_i256();
                    self.k.rd.residual_pred_blk(&mut resid[..], &cpr[..], src);
                    let (mut q, qt) = self.k.dct.dct16x16_t(&resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q,
                        &qt,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        16,
                        plane,
                        px,
                        py,
                    );
                    let rr = self.k.idct.idct_dequant_16x16(&q, &self.ctx.cquant);
                    let sse = sse_recon::<256, 16>(&self.k.rd, &cpr, &rr, src, self.cfg.bd);
                    let bits = self.chroma_bits(&q, &SCAN_16X16, 16, plane, px, py);
                    let pair = if ci == 0 { [a, other] } else { [other, a] };
                    let cost = cfl_alpha_walk_cost(
                        sse,
                        mlam,
                        bits,
                        self.uv_mode_bits(y_mode, CFL_PRED, Some(pair)),
                    );
                    (cost, (q, cpr, sse, bits))
                };
                let (c0, p0) = realise(a_e[ci]);
                let (a, (q, cpr, sse, bits)) =
                    cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                cfl_a[ci] = a;
                *cfl_ccf[ci] = q;
                cfl_sse[ci] = sse;
                cfl_bits[ci] = bits;
                *cpred16[ci] = *cpr;
            }
            let sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
            let dc_total = rd_cost_i64(
                dc_sse[0] + dc_sse[1],
                mlam,
                dc_bits[0] + dc_bits[1] + self.uv_mode_bits(y_mode, DC_PRED, None),
            );
            let cfl_total = rd_cost_i64(
                cfl_sse[0] + cfl_sse[1],
                mlam,
                cfl_bits[0] + cfl_bits[1] + sig,
            );
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                cfl_opt = Some(cfl_a);
                for ci in 0..2 {
                    *ccf[ci] = *cfl_ccf[ci];
                }
            } else {
                for ci in 0..2 {
                    *cpred16[ci] = [cpred[ci]; 256];
                }
            }
        }
        let mut chosen_uv_16 = DC_PRED;
        let mut uv_delta_16 = 0i32;
        let mut uv_pal: Option<LossyUvPalette> = None;
        if self.cfg.speed.full_chroma_rdo() {
            let (dcq, acq, lam) = (
                self.ctx.cquant.dc_q() as f32,
                self.ctx.cquant.ac_q() as f32,
                self.trellis_lambda(),
            );
            // Unification V9: the chroma lambda for a chroma decision
            // (`uv16_444_mlam_c`); the luma lambda is the pre-unification law.
            let mlam = if crate::tuning::get().uv16_444_mlam_c {
                self.emit_mlam_c(x8 * 8, y8 * 8, 16)
            } else {
                self.emit_mlam(x8 * 8, y8 * 8, 16)
            };
            // Reconstructed R-D of the CURRENT chroma choice (DC or CfL), using the
            // coeffs/prediction already selected above.
            let mut cur_total = rate_cost(
                mlam,
                if let Some(a) = cfl_opt {
                    self.uv_mode_bits(y_mode, CFL_PRED, Some(a))
                } else {
                    self.uv_mode_bits(y_mode, DC_PRED, None)
                },
            );
            for ci in 0..2 {
                let plane = ci + 1;
                let rr = self.k.idct.idct_dequant_16x16(&ccf[ci], &self.ctx.cquant);
                let sse = sse_recon::<256, 16>(&self.k.rd, &cpred16[ci], &rr, self.src_blk(plane, px, py, 16, 16), self.cfg.bd);
                cur_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&ccf[ci][..], &SCAN_16X16, 16, plane, px, py),
                );
            }

            // Directional / smooth chroma modes, each with its decoder-derived
            // chroma tx (PAETH/SMOOTH -> ADST_ADST, SMOOTH_V -> ADST_DCT,
            // SMOOTH_H -> DCT_ADST). PAETH is empirically the strongest non-DC
            // chroma mode, so it is searched alongside the SMOOTH family. The
            // winner must beat the current DC/CfL choice on the libaom-style
            // R-D cost computed in `cur_total`.
            let mut best_total = cur_total;
            let mut best_mode_uv = DC_PRED;
            let mut uv_delta16 = 0i32;
            let mut best_ccf16 = [self.sbuf_i256(), self.sbuf_i256()];
            let mut best_pred16 = [self.sbuf_i256(), self.sbuf_i256()];
            // D45/D67/D203 read the top-right / bottom-left edge. A square leaf's
            // single chroma transform inherits the luma leaf's edge flags (dav1d
            // recon_tmpl.c `uv_sb_has_tr/bl`), so they predict exactly here.
            let candidates = &[
                SMOOTH_V_PRED,
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
                D135_PRED,
                D113_PRED,
                D157_PRED,
                D45_PRED,
                VERT_LEFT_PRED,
                D203_PRED,
            ];
            let directional_top = self.rank_chroma_modes::<256>(candidates, px, py, px, py, 16, 16, have_tr, have_bl);
            for &cand in candidates {
                // V/H are cheap enough for every tier; Fast skips diagonal angles.
                if cand != V_PRED
                    && cand != H_PRED
                    && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                    && !self.cfg.speed.chroma_angle_directional()
                {
                    continue;
                }
                if !directional_top.contains(cand) {
                    continue;
                }
                let tx = chroma_tx_for_mode(cand);
                let mut cand_ccf = [self.sbuf_i256(), self.sbuf_i256()];
                let mut cand_pred = [self.sbuf_i256(), self.sbuf_i256()];
                // V/H and the Z2 angular modes (D135/D113/D157) all emit a chroma
                // angle_delta symbol (~3 bits); they sit in the 1..=8 directional range.
                let sig_bits = self.uv_mode_bits(y_mode, cand, None);
                let mut cand_total = rate_cost(mlam, sig_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    self.k.intrapred.predict_nd(
                        cand,
                        &self.ctx.recon[plane],
                        self.cfg.w,
                        px,
                        py,
                        16,
                        16,
                        have_tr,
                        have_bl,
                        self.cfg.w,
                        self.cfg.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci][..],
                        self.cfg.bd,
                    );
                    let mut resid = self.sbuf_i256();
                    self.k.rd.residual_pred(
                        &mut resid[..],
                        &cand_pred[ci][..],
                        &self.cfg.src[plane],
                        self.cfg.w,
                        px,
                        py,
                        16,
                        16,
                    );
                    let (mut q, qt) = fwd_chroma_16x16(&self.k.dct, tx, &resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q,
                        &qt,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        16,
                        plane,
                        px,
                        py,
                    );
                    self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    *cand_ccf[ci] = q;
                    let rr = inv_chroma_16x16(&self.k.idct, tx, &q, &self.ctx.cquant);
                    let sse = sse_recon::<256, 16>(&self.k.rd, &cand_pred[ci], &rr, self.src_blk(plane, px, py, 16, 16), self.cfg.bd);
                    cand_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_bits(&q, &SCAN_16X16, 16, plane, px, py),
                    );
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf16 = cand_ccf;
                    best_pred16 = cand_pred;
                }
            }
            if crate::tuning::get().uv_ad
                && self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx)
                && (V_PRED..=VERT_LEFT_PRED).contains(&best_mode_uv)
            {
                let cand = best_mode_uv;
                let tx = chroma_tx_for_mode(cand);
                let ad = &self.dcdf().angle_delta[cand - V_PRED];
                let base_sig = self.uv_mode_bits(y_mode, cand, None) - cdf_cost(ad, 3);
                for &d in uv_zone2_deltas(cand) {
                    let mut cand_ccf = [self.sbuf_i256(), self.sbuf_i256()];
                    let mut cand_pred = [self.sbuf_i256(), self.sbuf_i256()];
                    let mut cand_total = rate_cost(mlam, base_sig + cdf_cost(ad, (d + 3) as usize));
                    for ci in 0..2 {
                        let plane = ci + 1;
                        self.k.intrapred.predict_nd_ad(
                            cand,
                            d,
                            &self.ctx.recon[plane],
                            self.cfg.w,
                            px,
                            py,
                            16,
                            16,
                            false,
                            false,
                            self.cfg.w,
                            self.cfg.h,
                            self.chroma_filter_type(px, py),
                            &mut cand_pred[ci][..],
                            self.cfg.bd,
                        );
                        let mut resid = self.sbuf_i256();
                        self.k.rd.residual_pred(&mut resid[..], &cand_pred[ci][..], &self.cfg.src[plane], self.cfg.w, px, py, 16, 16);
                        let (mut q, qt) = fwd_chroma_16x16(&self.k.dct, tx, &resid, &self.ctx.cquant);
                        self.chroma_rect_trellis(&mut q, &qt, dcq, acq, &SCAN_16X16, lam, 16, 16, plane, px, py);
                        self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                        *cand_ccf[ci] = q;
                        let rr = inv_chroma_16x16(&self.k.idct, tx, &q, &self.ctx.cquant);
                        let sse = sse_recon::<256, 16>(&self.k.rd, &cand_pred[ci], &rr, self.src_blk(plane, px, py, 16, 16), self.cfg.bd);
                        cand_total += rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_16X16, 16, plane, px, py));
                    }
                    if cand_total < best_total {
                        best_total = cand_total;
                        uv_delta16 = d;
                        best_ccf16 = cand_ccf;
                        best_pred16 = cand_pred;
                    }
                }
            }
            if best_mode_uv != DC_PRED {
                for ci in 0..2 {
                    *ccf[ci] = *best_ccf16[ci];
                }
                for ci in 0..2 {
                    *cpred16[ci] = *best_pred16[ci];
                }
                cfl_opt = None; // a non-DC chroma mode overrides CfL if it wins
                chosen_uv_16 = best_mode_uv;
                uv_delta_16 = uv_delta16;
            }
            // UV palette candidates (4:4:4): exact when the block holds
            // <= 8 distinct (U, V) pairs (distortion 0), otherwise LOSSY
            // k-means clusterings — the unlock for anti-aliased screen
            // content where the exact gate always fails. Palette + two
            // zero-coefficient blocks, signaled as uv DC + palette_uv;
            // competes on the same R-D bar with its real chroma SSE.
            if self.try_palette() {
                let exact = exact_uv_palette(&self.cfg.src[1], &self.cfg.src[2], self.cfg.w, px, py, 16, 16);
                let cands: Vec<LossyUvPalette> = if let Some(up) = exact {
                    vec![up]
                } else {
                    let uv_hist = uv_pair_histogram(&self.cfg.src[1], &self.cfg.src[2], self.cfg.w, px, py, 16, 16);
                    let uv_map_opts = self.uv_palette_map_opts();
                    [(8usize, false), (4, false), (8, true), (4, true)]
                        .iter()
                        .filter_map(|&(k, top)| {
                            lossy_uv_palette_from_hist(
                                &self.k.kmeans,
                                &uv_hist,
                                &self.cfg.src[1],
                                &self.cfg.src[2],
                                self.cfg.w,
                                px,
                                py,
                                16,
                                16,
                                k,
                                top,
                                &uv_map_opts,
                            )
                        })
                        .collect()
                };
                for up in cands {
                    // Residual-over-palette: the palette map is the chroma
                    // PREDICTION; the residual is coded through the normal
                    // DCT path (uv DC mode -> DCT_DCT).
                    let (dcq2, acq2) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
                    let mut pal_pred = [self.sbuf_i256(), self.sbuf_i256()];
                    let [pred_u, pred_v] = &mut pal_pred;
                    palette_uv_pred(&mut pred_u[..], &mut pred_v[..], &up.map, &up.u, &up.v);
                    let mut bits = self.uv_mode_bits(y_mode, DC_PRED, None)
                        + self.palette_uv_rate_bits(px, py, palette.is_some(), &up);
                    let mut sse = 0i64;
                    let mut pal_ccf = [self.sbuf_i256(), self.sbuf_i256()];
                    let mut resids = [self.sbuf_i256(), self.sbuf_i256()];
                    for ci in 0..2 {
                        self.k.rd.residual_pred(
                            &mut resids[ci][..],
                            &pal_pred[ci][..],
                            &self.cfg.src[ci + 1],
                            self.cfg.w,
                            px,
                            py,
                            16,
                            16,
                        );
                    }
                    let raw_sse = self.uv_pal_gate_dist(&resids[0][..], &resids[1][..], 16, 16);
                    if self.uv_palette_gate(raw_sse, bits, mlam, best_total) {
                        continue;
                    }
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let resid = &resids[ci];
                        let (mut q, qt) = self.k.dct.dct16x16_t(resid, &self.ctx.cquant);
                        trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_16X16, self.trellis_lambda());
                        let rr = self.k.idct.idct_dequant_16x16(&q, &self.ctx.cquant);
                        sse += sse_recon::<256, 16>(&self.k.rd, &pal_pred[ci], &rr, self.src_blk(plane, px, py, 16, 16), self.cfg.bd);
                        *pal_ccf[ci] = q;
                        bits += self.chroma_bits(&q, &SCAN_16X16, 16, plane, px, py);
                    }
                    let cand_total = rd_cost_i64(sse, mlam, bits);
                    if cand_total < best_total {
                        best_total = cand_total;
                        chosen_uv_16 = DC_PRED;
                        uv_delta_16 = 0;
                        cfl_opt = None;
                        ccf = pal_ccf;
                        cpred16 = pal_pred;
                        uv_pal = Some(up);
                    }
                }
            }
        } // end directional/smooth chroma (4:4:4 16x16)
        UvPick {
            coeffs: ccf,
            pred: cpred16,
            resid: None,
            dc: cpred,
            mode: chosen_uv_16,
            cfl: cfl_opt,
            delta: uv_delta_16,
            palette: uv_pal,
        }
    }

    /// Emit a 16x16 block (444) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block16_444(&mut self, x8: usize, y8: usize, l: &Luma16<'_>, uv: UvPick<SBuf<[i32; 256]>, SBuf<[i32; 256]>>) {
        let (palette, luma_zero) = (l.palette, l.luma_zero);
        let UvPick {
            coeffs: ccf,
            pred: cpred16,
            dc: cpred,
            mode: chosen_uv_16,
            cfl: cfl_opt,
            delta: uv_delta_16,
            palette: uv_pal,
            ..
        } = uv;
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let block_skip = palette.is_none()
            && uv_pal.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&ccf[0][..])
            && self.k.rd.all_zero_i32(&ccf[1][..]);
        let uv_delta_sig = if chosen_uv_16 != DC_PRED { uv_delta_16 } else { 0 };
        self.code_header_luma16(
            x8,
            y8,
            l,
            block_skip,
            chosen_uv_16,
            cfl_opt,
            uv_delta_sig,
            uv_pal.as_ref(),
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_16(plane, bx4, by4);
                encode_tx16_coeffs_adapt(
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.ctx.a_coef[plane][bx4..bx4 + 4].fill(res_ctx);
            self.ctx.l_coef[plane][by4..by4 + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else if chosen_uv_16 != DC_PRED {
                // A directional/smooth chroma mode: invert with its decoder-derived
                // chroma tx (PAETH/SMOOTH -> ADST_ADST, SMOOTH_V -> ADST_DCT,
                // SMOOTH_H -> DCT_ADST).
                inv_chroma_16x16(&self.k.idct, chroma_tx_for_mode(chosen_uv_16), &ccf[ci], &self.ctx.cquant)
            } else {
                self.k.idct.idct_dequant_16x16(&ccf[ci], &self.ctx.cquant)
            };
            let max = (1 << self.cfg.bd) - 1;
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.w + px..];
                if cfl_opt.is_some() || chosen_uv_16 != DC_PRED || uv_pal.is_some() {
                    // CfL and every non-DC mode store their per-pixel prediction in
                    // `cpred16`.
                    recon_add_pred(drow, &cpred16[ci][ry * 16..], rrow, max);
                } else {
                    // Plain DC chroma: use the scalar predictor directly so recon
                    // never depends on the CfL block having populated `cpred16`.
                    recon_add_dc(drow, cpred[ci], rrow, max);
                }
            }
        }
    }

    /// Chroma search of a 16x16 block (420) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block16_420`.
    fn pick_uv16_420(&self, x8: usize, y8: usize, l: &Luma16<'_>) -> UvPick<[i32; 64], [i32; 64]> {
        let Luma16 {
            cf: lcf,
            pred: lpred,
            y_mode,
            txtp16,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let (dcq, acq, lam) = (
            self.ctx.cquant.dc_q() as f32,
            self.ctx.cquant.ac_q() as f32,
            self.trellis_lambda(),
        );
        let maxval = (1 << self.cfg.bd) - 1;
        // DC path.
        let mut ccf_dc = [[0i32; 64]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[plane], self.cfg.cw, cx, cy, self.cfg.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 64];
            self.k.rd.residual_dc(&mut resid, &self.cfg.src[plane], self.cfg.cw, cx, cy, 8, 8, dc);
            let (q, qt) = self.k.dct.dct8x8_t(&resid, &self.ctx.cquant);
            ccf_dc[ci] = q;
            self.chroma_rect_trellis(
                &mut ccf_dc[ci],
                &qt,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                8,
                plane,
                cx,
                cy,
            );
            self.k.rd.preserve_dc(&mut ccf_dc[ci][0], &resid[..]);
        }
        // DC baseline R-D (libaom-style: SSE + mlam*coeff_bits, summed over U+V).
        let mut rr_dc = [[0i32; 64]; 2];
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
        let mut dc_total = 0f32;
        for ci in 0..2 {
            let plane = ci + 1;
            rr_dc[ci] = self.k.idct.idct_dequant_8x8(&ccf_dc[ci], &self.ctx.cquant);
            let dc = dc_preds[ci];
            let sse = sse_recon::<64, 8>(&self.k.rd, &[dc; 64], &rr_dc[ci], self.src_blk(plane, cx, cy, 8, 8), self.cfg.bd);
            dc_total += rd_cost_i64(
                sse,
                mlam,
                self.chroma_bits(&ccf_dc[ci], &SCAN_8X8, 8, plane, cx, cy),
            );
        }
        // Directional / smooth chroma modes, each with its decoder-derived chroma tx.
        // PAETH is empirically the strongest non-DC chroma mode, searched alongside
        // the SMOOTH family. Winner must beat DC on the same R-D metric.
        let mut best_total = dc_total;
        let mut chosen_uv = DC_PRED;
        let mut best_ccf = ccf_dc;
        let mut best_rr = rr_dc;
        let mut best_pred = [[0i32; 64]; 2];
        let mut use_cfl = false;
        let mut cfl_alpha = [0i32; 2];
        if !self.cfg.is_mono() {
            let mut luma_rec = self.sbuf_u256();
            if txtp16 == 4 || txtp16 == 6 {
                *luma_rec = *l.split_rec.expect("a TX-split luma winner carries its reconstruction");
            } else {
                let lrr = match txtp16 {
                    1 => self.k.idct.iadst_dequant_16x16(lcf, &self.ctx.quant),
                    2 => self.k.idct.iadstdct_dequant_16x16(lcf, &self.ctx.quant),
                    3 => self.k.idct.idctadst_dequant_16x16(lcf, &self.ctx.quant),
                    5 => self.k.idct.iidentity_dequant_16x16(lcf, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_16x16(lcf, &self.ctx.quant),
                };
                recon_add_pred(&mut luma_rec[..], lpred, &lrr, maxval);
            }
            let mut ac = [0i32; 64];
            self.k.intrapred
                .cfl_ac_sub(&luma_rec[..], 16, 8, 8, true, true, &mut ac);
            let mut cfl_ccf = [[0i32; 64]; 2];
            let mut cfl_rr = [[0i32; 64]; 2];
            let mut cfl_px = [[0i32; 64]; 2];
            let mut cfl_a = [0i32; 2];
            let mut cfl_body = 0f32;
            let steps = crate::tuning::get().cfl_alpha_rd_steps;
            let a_e = [0, 1].map(|ci| {
                self.k.intrapred.cfl_best_alpha(&ac, self.src_blk(ci + 1, cx, cy, 8, 8), dc_preds[ci], self.cfg.bd)
            });
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_preds[ci];
                let src = self.src_blk(plane, cx, cy, 8, 8);
                let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
                let realise = |a: i32| {
                    let mut cpr = [0i32; 64];
                    self.k.intrapred.cfl_pred(&mut cpr, &ac[..64], dc, a, self.cfg.bd);
                    let mut resid = [0i32; 64];
                    self.k.rd.residual_pred_blk(&mut resid, &cpr, src);
                    let (mut q, qt) = self.k.dct.dct8x8_t(&resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
                    );
                    let rr = self.k.idct.idct_dequant_8x8(&q, &self.ctx.cquant);
                    let sse = sse_recon::<64, 8>(&self.k.rd, &cpr, &rr, src, self.cfg.bd);
                    let bits = self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy);
                    let pair = if ci == 0 { [a, other] } else { [other, a] };
                    let cost = cfl_alpha_walk_cost(
                        sse,
                        mlam,
                        bits,
                        self.uv_mode_bits(y_mode, CFL_PRED, Some(pair)),
                    );
                    (cost, (q, rr, cpr, sse, bits))
                };
                let (c0, p0) = realise(a_e[ci]);
                let (a, (q, rr, cpr, sse, bits)) =
                    cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                cfl_a[ci] = a;
                cfl_ccf[ci] = q;
                cfl_rr[ci] = rr;
                cfl_px[ci] = cpr;
                cfl_body += rd_cost_i64(sse, mlam, bits);
            }
            let cfl_total = cfl_body
                + rate_cost(mlam, self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a)))
                - rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None));
            if (cfl_a[0] != 0 || cfl_a[1] != 0) && cfl_total < best_total {
                best_total = cfl_total;
                use_cfl = true;
                cfl_alpha = cfl_a;
                best_ccf = cfl_ccf;
                best_rr = cfl_rr;
                best_pred = cfl_px;
            }
        }
        // D45/D67/D203 read the top-right / bottom-left edge. A square leaf's
        // single chroma transform inherits the luma leaf's edge flags (dav1d
        // recon_tmpl.c `uv_sb_has_tr/bl`), so they predict exactly here.
        let candidates = &[
            SMOOTH_V_PRED,
            PAETH_PRED,
            SMOOTH_PRED,
            SMOOTH_H_PRED,
            V_PRED,
            H_PRED,
            D135_PRED,
            D113_PRED,
            D157_PRED,
            D45_PRED,
            VERT_LEFT_PRED,
            D203_PRED,
        ];
        // `uv16_joint`: joint (mode, delta) SATD shortlist (see `uv8_joint`).
        let joint16 = crate::tuning::get().uv16_joint;
        let mut joint_delta = 0i32;
        let directional_top = if !joint16 {
            self.rank_chroma_modes::<64>(candidates, px, py, cx, cy, 8, 8, have_tr, have_bl)
        } else {
            DirectionalTopK::new()
        };
        if joint16 {
            let k = (crate::tuning::get().uv8_joint_k as usize).clamp(1, 12);
            let ad_ok = self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx);
            let mut all = FixedList::<(u64, usize, i32), 84>::new((0, DC_PRED, 0));
            for &m in candidates {
                let deltas: &[i32] = if ad_ok && (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                    &[-3, -2, -1, 0, 1, 2, 3]
                } else {
                    &[0]
                };
                for &d in deltas {
                    let mut cost = 0u64;
                    for plane in 1..=2 {
                        let mut pp = [0i32; 64];
                        self.k.intrapred.predict_nd_ad(
                            m,
                            d,
                            &self.ctx.recon[plane],
                            self.cfg.cw,
                            cx,
                            cy,
                            8,
                            8,
                            have_tr,
                            have_bl,
                            self.cfg.cw,
                            self.cfg.h / 2,
                            self.chroma_filter_type(px, py),
                            &mut pp,
                            self.cfg.bd,
                        );
                        cost += self.k.rd.satd_sad_proxy(self.src_blk(plane, cx, cy, 8, 8), &pp, 8);
                    }
                    all.push((cost, m, d));
                }
            }
            all.as_mut_slice().sort_unstable_by_key(|&(c, m, d)| (c, m, d));
            for &(_, cand, jd) in all.as_slice().iter().take(k) {
                let tx = chroma_tx_for_mode(cand);
                let ad_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    let ad = &self.dcdf().angle_delta[cand - V_PRED];
                    cdf_cost(ad, (jd + 3) as usize) - cdf_cost(ad, 3)
                } else {
                    0.0
                };
                let mut cand_ccf = [[0i32; 64]; 2];
                let mut cand_rr = [[0i32; 64]; 2];
                let mut cand_pred = [[0i32; 64]; 2];
                let mut cand_total = rate_cost(mlam, self.uv_mode_bits(y_mode, cand, None) + ad_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    self.k.intrapred.predict_nd_ad(
                        cand,
                        jd,
                        &self.ctx.recon[plane],
                        self.cfg.cw,
                        cx,
                        cy,
                        8,
                        8,
                        have_tr,
                        have_bl,
                        self.cfg.cw,
                        self.cfg.h / 2,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        self.cfg.bd,
                    );
                    let mut resid = [0i32; 64];
                    self.k.rd.residual_pred(&mut resid, &cand_pred[ci], &self.cfg.src[plane], self.cfg.cw, cx, cy, 8, 8);
                    let (mut q, qt) = fwd_chroma_8x8(&self.k.dct, tx, &resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(&mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, cx, cy);
                    self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    cand_ccf[ci] = q;
                    cand_rr[ci] = inv_chroma_8x8(&self.k.idct, tx, &q, &self.ctx.cquant);
                    let sse = sse_recon::<64, 8>(&self.k.rd, &cand_pred[ci], &cand_rr[ci], self.src_blk(plane, cx, cy, 8, 8), self.cfg.bd);
                    cand_total += rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy));
                    if ci == 0 && cand_total >= best_total {
                        break; // cannot win: skip the second plane (non-negative costs)
                    }
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    chosen_uv = cand;
                    joint_delta = jd;
                    use_cfl = false;
                    best_ccf = cand_ccf;
                    best_rr = cand_rr;
                    best_pred = cand_pred;
                }
            }
        }
        for &cand in candidates {
            if joint16 {
                break;
            }
            // V/H are cheap enough for every tier; Fast skips diagonal angles.
            if cand != V_PRED
                && cand != H_PRED
                && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                && !self.cfg.speed.chroma_angle_directional()
            {
                continue;
            }
            if !directional_top.contains(cand) {
                continue;
            }
            let tx = chroma_tx_for_mode(cand);
            let mut cand_ccf = [[0i32; 64]; 2];
            let mut cand_rr = [[0i32; 64]; 2];
            let mut cand_pred = [[0i32; 64]; 2];
            let sig_bits = self.uv_mode_bits(y_mode, cand, None);
            let mut cand_total = rate_cost(mlam, sig_bits);
            for ci in 0..2 {
                let plane = ci + 1;
                self.k.intrapred.predict_nd(
                    cand,
                    &self.ctx.recon[plane],
                    self.cfg.cw,
                    cx,
                    cy,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.cfg.cw,
                    self.cfg.h / 2,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci],
                    self.cfg.bd,
                );
                let mut resid = [0i32; 64];
                self.k.rd.residual_pred(
                    &mut resid,
                    &cand_pred[ci],
                    &self.cfg.src[plane],
                    self.cfg.cw,
                    cx,
                    cy,
                    8,
                    8,
                );
                let (mut q, qt) = fwd_chroma_8x8(&self.k.dct, tx, &resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
                );
                self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                cand_ccf[ci] = q;
                cand_rr[ci] = inv_chroma_8x8(&self.k.idct, tx, &q, &self.ctx.cquant);
                let sse = sse_recon::<64, 8>(&self.k.rd, &cand_pred[ci], &cand_rr[ci], self.src_blk(plane, cx, cy, 8, 8), self.cfg.bd);
                cand_total +=
                    rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy));
            }
            if cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                use_cfl = false;
                best_ccf = cand_ccf;
                best_rr = cand_rr;
                best_pred = cand_pred;
            }
        }
        // Chroma angle-delta refinement (`uv_ad`) of a directional winner:
        // the zone-2 deltas only (V +1..3, H -1..-3, D113/D135/D157 +-1..3), so
        // the tr/bl-free chroma edge setup stays exact.
        let mut uv_delta = if joint16 && !use_cfl { joint_delta } else { 0 };
        if !joint16
            && !use_cfl
            && crate::tuning::get().uv_ad
            && self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx)
            && (V_PRED..=VERT_LEFT_PRED).contains(&chosen_uv)
        {
            let cand = chosen_uv;
            let tx = chroma_tx_for_mode(cand);
            let ad = &self.dcdf().angle_delta[cand - V_PRED];
            let base_sig = self.uv_mode_bits(y_mode, cand, None) - cdf_cost(ad, 3);
            for &d in uv_zone2_deltas(cand) {
                let mut cand_ccf = [[0i32; 64]; 2];
                let mut cand_rr = [[0i32; 64]; 2];
                let mut cand_pred = [[0i32; 64]; 2];
                let mut cand_total = rate_cost(mlam, base_sig + cdf_cost(ad, (d + 3) as usize));
                for ci in 0..2 {
                    let plane = ci + 1;
                    self.k.intrapred.predict_nd_ad(
                        cand,
                        d,
                        &self.ctx.recon[plane],
                        self.cfg.cw,
                        cx,
                        cy,
                        8,
                        8,
                        false,
                        false,
                        self.cfg.cw,
                        self.cfg.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        self.cfg.bd,
                    );
                    let mut resid = [0i32; 64];
                    self.k.rd.residual_pred(&mut resid, &cand_pred[ci], &self.cfg.src[plane], self.cfg.cw, cx, cy, 8, 8);
                    let (mut q, qt) = fwd_chroma_8x8(&self.k.dct, tx, &resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(&mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, cx, cy);
                    self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    cand_ccf[ci] = q;
                    cand_rr[ci] = inv_chroma_8x8(&self.k.idct, tx, &q, &self.ctx.cquant);
                    let sse = sse_recon::<64, 8>(&self.k.rd, &cand_pred[ci], &cand_rr[ci], self.src_blk(plane, cx, cy, 8, 8), self.cfg.bd);
                    cand_total += rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy));
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    uv_delta = d;
                    best_ccf = cand_ccf;
                    best_rr = cand_rr;
                    best_pred = cand_pred;
                }
            }
        }
        UvPick {
            coeffs: best_ccf,
            pred: best_pred,
            resid: Some(best_rr),
            dc: dc_preds,
            mode: chosen_uv,
            cfl: use_cfl.then_some(cfl_alpha),
            delta: uv_delta,
            palette: None,
        }
    }

    /// Emit a 16x16 block (420) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block16_420(&mut self, x8: usize, y8: usize, l: &Luma16<'_>, uv: UvPick<[i32; 64], [i32; 64]>) {
        let (palette, luma_zero) = (l.palette, l.luma_zero);
        let UvPick {
            coeffs: ccf,
            pred: sv_preds,
            resid,
            dc: dc_preds,
            mode: chosen_uv,
            cfl,
            delta: uv_delta,
            ..
        } = uv;
        let rr_cache = resid.expect("the 4:2:0 search caches the inverse residual");
        let use_cfl = cfl.is_some();
        let (cx, cy) = (x8 * 4, y8 * 4);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let maxval = (1 << self.cfg.bd) - 1;
        let use_nondc = chosen_uv != DC_PRED || use_cfl;
        // Palette color indices ride with the transform-token payload; a skip
        // block has none, so palette blocks always code skip_txfm = 0.
        let block_skip = palette.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&ccf[0])
            && self.k.rd.all_zero_i32(&ccf[1]);
        self.code_header_luma16(
            x8,
            y8,
            l,
            block_skip,
            chosen_uv,
            cfl,
            uv_delta,
            None,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx(plane, bx4c, by4c);
                encode_tx8_coeffs_adapt(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], true, sk, ds, 0, 1)
            };
            self.ctx.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.ctx.l_coef[plane][by4c..by4c + 2].fill(res_ctx);
            let rr = if block_skip { [0i32; 64] } else { rr_cache[ci] };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
                if use_nondc {
                    recon_add_pred(&mut drow[..8], &sv_preds[ci][ry * 8..], rrow, maxval);
                } else {
                    recon_add_dc(&mut drow[..8], dc_preds[ci], rrow, maxval);
                }
            }
        }
    }

    /// 4:2:2: a 16x16 luma region maps to an 8-wide x 16-tall chroma region per
    /// plane (`RTX_8X16`, coef-CDF class 2). Chroma is full-height, half-width, so
    /// the chroma block sits at `(cx, py)` with `cx = px/2` and spans 2 coef units
    /// horizontally and 4 vertically on the chroma grid.
    /// Chroma search of a 16x16 block (422) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block16_422`.
    fn pick_uv16_422(&self, x8: usize, y8: usize, l: &Luma16<'_>) -> UvPick<[i32; 128], [i32; 128]> {
        let Luma16 {
            cf: lcf,
            pred: lpred,
            y_mode,
            txtp16,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let maxv = (1 << self.cfg.bd) - 1;
        let (dcq, acq, lam) = (
            self.ctx.cquant.dc_q() as f32,
            self.ctx.cquant.ac_q() as f32,
            self.trellis_lambda(),
        );
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        // Per-pixel chroma prediction (DC broadcast, or CfL dc+alpha*ac).
        let mut cpred_px = [[0i32; 128]; 2];
        let src_planes = [1, 2].map(|plane| self.src_blk(plane, cx, py, 8, 16));
        // DC option (always computed).
        let mut dc_ccf = [[0i32; 128]; 2];
        let mut dc_sse = [0i64; 2];
        let mut dc_bits = [0f32; 2];
        // DC option.
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = self.k.intrapred.dc_pred_8x16(&self.ctx.recon[plane], self.cfg.cw, cx, py, self.cfg.bd as i32);
            cpred[ci] = pred;
            let src = src_planes[ci];
            let mut resid = [0i32; 128];
            self.k.rd.residual_dc_blk(&mut resid, src, pred);
            let (mut q, qt) = self.k.dct.dct8x16_t(&resid, &self.ctx.cquant);
            self.chroma_rect_trellis(
                &mut q,
                &qt,
                dcq,
                acq,
                &SCAN_8X16,
                lam,
                8,
                16,
                ci + 1,
                cx,
                py,
            );
            let rr = self.k.idct.idct_dequant_8x16(&q, &self.ctx.cquant);
            dc_ccf[ci] = q;
            dc_sse[ci] = self.k.rd.sse_recon(&[pred; 128], &rr, src, self.cfg.bd);
            dc_bits[ci] = self.chroma_rect_bits(&q, &SCAN_8X16, 8, 16, plane, cx, py);
        }

        // CfL option: predict 8x16 U/V from the horizontally-subsampled 16x16
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0). Mirrors the
        // 4:2:2 4x8 CfL in `code_block` at the larger 8x16 chroma size.
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        {
            // Luma reconstruction as the decoder will see it: with TX split the
            // block reconstructs per sub-TX (sequential per-quadrant
            // prediction), so `lpred + inverse(lcf)` no longer applies.
            // Missing this branch fed CfL a wrong AC — the 4:2:2 V-plane
            // recon drift (x_fractal q100), V-visible because alpha_v is
            // typically large while alpha_u rounds away.
            let mut luma_rec = self.sbuf_u256();
            if txtp16 == 4 || txtp16 == 6 {
                *luma_rec = *l.split_rec.expect("a TX-split luma winner carries its reconstruction");
            } else {
                let lrr_cfl = match txtp16 {
                    1 => self.k.idct.iadst_dequant_16x16(lcf, &self.ctx.quant),
                    2 => self.k.idct.iadstdct_dequant_16x16(lcf, &self.ctx.quant),
                    3 => self.k.idct.idctadst_dequant_16x16(lcf, &self.ctx.quant),
                    5 => self.k.idct.iidentity_dequant_16x16(lcf, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_16x16(lcf, &self.ctx.quant),
                };
                recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, maxv);
            }
            let mut ac = [0i32; 128];
            self.k.intrapred
                .cfl_ac_sub(&luma_rec[..], 16, 8, 16, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let mut cfl_sse = [0i64; 2];
            let mut cfl_bits = [0f32; 2];
            let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
            let steps = crate::tuning::get().cfl_alpha_rd_steps;
            let a_e = [0, 1].map(|ci| {
                self.k.intrapred.cfl_best_alpha(&ac, src_planes[ci], cpred[ci], self.cfg.bd)
            });
            for ci in 0..2 {
                let dc = cpred[ci];
                let src = src_planes[ci];
                let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
                let realise = |a: i32| {
                    let mut cpr = [0i32; 128];
                    self.k.intrapred.cfl_pred(&mut cpr, &ac[..128], dc, a, self.cfg.bd);
                    let mut resid = [0i32; 128];
                    self.k.rd.residual_pred_blk(&mut resid, &cpr, src);
                    let (mut q, qt) = self.k.dct.dct8x16_t(&resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q,
                        &qt,
                        dcq,
                        acq,
                        &SCAN_8X16,
                        lam,
                        8,
                        16,
                        ci + 1,
                        cx,
                        py,
                    );
                    let rr = self.k.idct.idct_dequant_8x16(&q, &self.ctx.cquant);
                    let sse = self.k.rd.sse_recon(&cpr, &rr, src, self.cfg.bd);
                    let bits = self.chroma_rect_bits(&q, &SCAN_8X16, 8, 16, ci + 1, cx, py);
                    let pair = if ci == 0 { [a, other] } else { [other, a] };
                    let cost = cfl_alpha_walk_cost(
                        sse,
                        mlam,
                        bits,
                        self.uv_mode_bits(y_mode, CFL_PRED, Some(pair)),
                    );
                    (cost, (q, cpr, sse, bits))
                };
                let (c0, p0) = realise(a_e[ci]);
                let (a, (q, cpr, sse, bits)) =
                    cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                cfl_a[ci] = a;
                cfl_ccf[ci] = q;
                cfl_sse[ci] = sse;
                cfl_bits[ci] = bits;
                cpred_px[ci] = cpr;
            }
            let sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
            let dc_total = rd_cost_i64(
                dc_sse[0] + dc_sse[1],
                mlam,
                dc_bits[0] + dc_bits[1] + self.uv_mode_bits(y_mode, DC_PRED, None),
            );
            let cfl_total = rd_cost_i64(
                cfl_sse[0] + cfl_sse[1],
                mlam,
                cfl_bits[0] + cfl_bits[1] + sig,
            );
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        if !use_cfl {
            for ci in 0..2 {
                ccf[ci] = dc_ccf[ci];
                cpred_px[ci] = [cpred[ci]; 128];
            }
        }
        // Directional / smooth chroma (`uv422_modes`): the same candidate set
        // as the 4:2:0 / 4:4:4 16x16 leaves, on the 8x16 chroma block with its
        // uv-mode-derived transform, then the zone-2 angle-delta refinement.
        let mut chosen_uv = DC_PRED;
        let mut uv_delta = 0i32;
        if crate::tuning::get().uv422_modes {
            let cur_mode_bits = if use_cfl {
                self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_alpha_uv))
            } else {
                self.uv_mode_bits(y_mode, DC_PRED, None)
            };
            let mut best_total = rate_cost(mlam, cur_mode_bits);
            for ci in 0..2 {
                let plane = ci + 1;
                let rr = self.k.idct.idct_dequant_8x16(&ccf[ci], &self.ctx.cquant);
                let sse = self.k.rd.sse_recon(&cpred_px[ci], &rr, src_planes[ci], self.cfg.bd);
                best_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci], &SCAN_8X16, 8, 16, plane, cx, py),
                );
            }
            let candidates = &[
                SMOOTH_V_PRED,
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
                D135_PRED,
                D113_PRED,
                D157_PRED,
            ];
            let directional_top =
                self.rank_chroma_modes::<128>(candidates, px, py, cx, py, 8, 16, false, false);
            let eval = |this: &Self, cand: usize, d: i32| {
                let tx = chroma_tx_for_mode(cand);
                let mut sig = this.uv_mode_bits(y_mode, cand, None);
                if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    let ad = &this.dcdf().angle_delta[cand - V_PRED];
                    sig += cdf_cost(ad, (d + 3) as usize) - cdf_cost(ad, 3);
                }
                let mut total = rate_cost(mlam, sig);
                let mut cand_ccf = [[0i32; 128]; 2];
                let mut cand_pred = [[0i32; 128]; 2];
                for ci in 0..2 {
                    let plane = ci + 1;
                    this.k.intrapred.predict_nd_ad(
                        cand,
                        d,
                        &this.ctx.recon[plane],
                        this.cfg.cw,
                        cx,
                        py,
                        8,
                        16,
                        false,
                        false,
                        this.cfg.cw,
                        this.cfg.h,
                        this.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        this.cfg.bd,
                    );
                    let mut resid = [0i32; 128];
                    this.k.rd.residual_pred_blk(&mut resid, &cand_pred[ci], src_planes[ci]);
                    let (mut q, qt) = fwd_chroma_8x16(&this.k.dct, tx, &resid, &this.ctx.cquant);
                    this.chroma_rect_trellis(
                        &mut q, &qt, dcq, acq, &SCAN_8X16, lam, 8, 16, plane, cx, py,
                    );
                    this.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    let rr = inv_chroma_8x16(&this.k.idct, tx, &q, &this.ctx.cquant);
                    let sse = this.k.rd.sse_recon(&cand_pred[ci], &rr, src_planes[ci], this.cfg.bd);
                    total += rd_cost_i64(
                        sse,
                        mlam,
                        this.chroma_rect_bits(&q, &SCAN_8X16, 8, 16, plane, cx, py),
                    );
                    cand_ccf[ci] = q;
                }
                (total, cand_ccf, cand_pred)
            };
            let mut win: Option<UvRect422Win> = None;
            for &cand in candidates {
                if cand != V_PRED
                    && cand != H_PRED
                    && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                    && !self.cfg.speed.chroma_angle_directional()
                {
                    continue;
                }
                if !directional_top.contains(cand) {
                    continue;
                }
                let (t, c, p) = eval(self, cand, 0);
                if t < best_total {
                    best_total = t;
                    win = Some((cand, 0, c, p));
                }
            }
            if let Some((m, _, _, _)) = win
                && crate::tuning::get().uv_ad
                && self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx)
            {
                for &d in uv_zone2_deltas(m) {
                    let (t, c, p) = eval(self, m, d);
                    if t < best_total {
                        best_total = t;
                        win = Some((m, d, c, p));
                    }
                }
            }
            if let Some((m, d, c, p)) = win {
                chosen_uv = m;
                uv_delta = d;
                use_cfl = false;
                ccf = c;
                cpred_px = p;
            }
        }
        UvPick {
            coeffs: ccf,
            pred: cpred_px,
            resid: None,
            dc: cpred,
            mode: chosen_uv,
            cfl: use_cfl.then_some(cfl_alpha_uv),
            delta: uv_delta,
            palette: None,
        }
    }

    /// Emit a 16x16 block (422) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block16_422(&mut self, x8: usize, y8: usize, l: &Luma16<'_>, uv: UvPick<[i32; 128], [i32; 128]>) {
        let (palette, luma_zero) = (l.palette, l.luma_zero);
        let UvPick {
            coeffs: ccf,
            pred: cpred_px,
            mode: chosen_uv,
            cfl,
            delta: uv_delta,
            ..
        } = uv;
        let use_cfl = cfl.is_some();
        let py = y8 * 8;
        let cx = x8 * 4;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let maxv = (1 << self.cfg.bd) - 1;
        // Palette color indices ride with the transform-token payload; a skip
        // block has none, so palette blocks always code skip_txfm = 0.
        let block_skip = palette.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&ccf[0])
            && self.k.rd.all_zero_i32(&ccf[1]);
        self.code_header_luma16(
            x8,
            y8,
            l,
            block_skip,
            if use_cfl { CFL_PRED } else { chosen_uv },
            cfl,
            uv_delta,
            None,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_8x16_422(plane, bx4c, by4c);
                encode_8x16_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            };
            // RTX_8X16: 2 coef-context units wide, 4 units tall.
            self.ctx.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.ctx.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 128]
            } else {
                inv_chroma_8x16(&self.k.idct, chroma_tx_for_mode(chosen_uv), &ccf[ci], &self.ctx.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.cw + cx..];
                recon_add_pred(drow, &cpred_px[ci][ry * 8..], rrow, maxv);
            }
        }
    }

    /// Code an 8x8 luma region as PARTITION_SPLIT into four BLOCK_4X4 luma
    /// sub-blocks (z-order), with the shared 4:2:0 4x4 chroma attached to the
    /// bottom-right sub-block. DC-only luma + DC chroma for now: this is the
    /// bit-exactness scaffold for sub-8x8 luma; richer modes/CfL layer on once
    /// the entropy/recon path is verified against dav1d. Caller has already
    /// emitted the PARTITION_SPLIT symbol.
    /// Search of 4x4 block `si` (z-order) of an 8x8 SPLIT: luma mode and
    /// transform type, then chroma where this block carries it.
    fn pick_sub4(&self, x8: usize, y8: usize, si: usize, have_tr: bool, have_bl: bool) -> Sub4Pick {
        let (px, py) = (x8 * 8, y8 * 8);
        let maxv = (1 << self.cfg.bd) - 1;
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 8);


        // Chroma layout differs by subsampling:
        //   4:2:0 -> the four 4x4 luma units share ONE 4x4 chroma block, coded on
        //            the bottom-right sub-block (origin px/2, py/2).
        //   4:4:4 -> chroma is full resolution: EVERY 4x4 luma sub-block carries a
        //            co-located 4x4 chroma block (dav1d has_chroma is true for each
        //            BLOCK_4X4 in I444). Each sub-block emits its own uv_mode and
        //            U/V residual at the same (bx, by) with stride w.
        //   4:2:2 -> the horizontal PAIRS share chroma: the RIGHT sub-block of
        //            each pair (TR, BR — odd bx4, dav1d has_chroma rule at
        //            ss_hor=1/ss_ver=0) carries a 4x4 chroma block at
        //            (px/2, by) covering both halves. DC-only, like 4:2:0.
        let ss420 = self.cfg.is_420();
        let ss422 = self.cfg.is_422();
        let (cx420, cy420) = (px / 2, py / 2);
        // z-order: TL, TR, BL, BR
        let (sx, sy) = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)][si];
        let (bx, by) = (px + sx * 4, py + sy * 4);
        // Spec edge availability per z-order sub-block (must match the
        // decoder EXACTLY — conservative flags mispredict D45 etc.).
        let (sub_tr, sub_bl) = match si {
            0 => (py > 0, px > 0),
            1 => (have_tr, false),
            2 => (true, have_bl),
            _ => (false, false),
        };
        // 4:2:0: bottom-right unit only; 4:2:2: right unit of each
        // horizontal pair; 4:4:4: every unit.
        let has_chroma = !self.cfg.is_mono()
            && if ss420 {
                si == 3
            } else if ss422 {
                si == 1 || si == 3
            } else {
                true
            };
        // Chroma origin / stride for this unit: full-res co-located for
        // 4:4:4, half-res shared block for 4:2:0, half-width pair block
        // for 4:2:2.
        let (chx, chy) = if ss420 {
            (cx420, cy420)
        } else if ss422 {
            (px / 2, by)
        } else {
            (bx, by)
        };
        let cstride = self.cfg.cw;

        // --- luma 4x4: intra mode search (DCT in-loop), then a tx-type
        // refinement on the winner. ---
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 8);
        let full_set4 = !self.cfg.is_444();
        let modes = if self.cfg.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        let mut best_mode = DC_PRED;
        let mut lpred = [0i32; 16];
        let mut lcf = [0i32; 16];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        for &m in modes {
            if !full_set4
                && (m == SMOOTH_V_PRED
                    || m == SMOOTH_H_PRED
                    || (is_directional_mode(m) && m != V_PRED && m != H_PRED))
            {
                continue;
            }
            let mut pred = [0i32; 16];
            if m == DC_PRED {
                let d = self.k.intrapred.dc_pred_4x4(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                pred = [d; 16];
            } else {
                self.k.intrapred.predict_nd(
                    m,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    bx,
                    by,
                    4,
                    4,
                    sub_tr,
                    sub_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(bx, by),
                    &mut pred,
                    self.cfg.bd,
                );
            }
            let mut resid = [0i32; 16];
            self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, bx, by, 4, 4);
            let (mut cf, tf) = self.k.dct.dct4x4_t(&resid, &self.ctx.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
            let rr = self.k.idct.idct_dequant_4x4(&cf, &self.ctx.quant);
            let sse = sse_recon::<16, 4>(&self.k.rd, &pred, &rr, self.src_blk(0, bx, by, 4, 4), self.cfg.bd);
            let bits = self.luma_bits(&cf, &SCAN_4X4, 4, bx, by, m, 1);
            // The mode symbol is coded for every 4x4 prediction block. At
            // high-quality 4:4:4, pricing it prevents tiny SSE edges from
            // beating much cheaper contextual DC. It also pays across the
            // measured 4:2:2 low/mid band; 4:2:0 and the remaining bands
            // keep their established decision law.
            let top444 = self.top_band() && !self.cfg.is_420() && !self.cfg.is_422();
            let low_mid_422 = self.cfg.speed.at_least_slow()
                && self.ctx.aq.enabled
                && self.cfg.is_422()
                && self.ctx.aq.base_q >= 28;
            // `exact_8x8_mode_rate` charges it unconditionally: the symbol
            // IS emitted for every 4x4, and at 4x4 it is a large share of
            // block rate, so omitting it lets a directional mode with a
            // tiny SSE edge win while costing several bits. The band gates
            // below are the pre-existing tuned law, kept as the fallback.
            let mode_bits = if (crate::tuning::get().exact_8x8_mode_rate
                && !self.cfg.is_422()
                && !self.cfg.is_mono())
                || top444
                || low_mid_422
            {
                self.mode_bits(bx, by, m)
            } else {
                0.0
            };
            let eff = rd_cost_i64(sse, mlam, bits + mode_bits);
            if eff < best_eff {
                best_eff = eff;
                best_mode = m;
                lpred = pred;
                lcf = cf;
                best_dct_sse = sse;
                best_dct_bits = bits;
            }
        }
        // Per-sub-block tx-type refinement on the winning prediction:
        // ADST_ADST only. It shares SCAN_4X4 and the standard coefficient
        // classes with DCT, so `encode_tx4_luma_coeffs_adapt` codes it
        // unchanged; accepted under the 8x8-style raw-SSE guard (no
        // distortion-for-rate trades). MEASURED: the asymmetric hybrids
        // (ADST_DCT/DCT_ADST, txtp 5/6) regress holdout at this size
        // (+0.02% 420 / +0.06% 444 vs −0.06%/−0.02% for ADST alone) —
        // the extra signaled symbols don't pay at 16 coefficients. IDTX
        // and V/H at 4x4 would need the 1-D class coder (still open).
        let mut tx4 = TxSel::Dct;
        if self.cfg.speed.try_adst() && best_eff.is_finite() {
            let mut resid = [0i32; 16];
            self.k.rd.residual_pred(
                &mut resid,
                &lpred,
                &self.cfg.src[0],
                self.cfg.w,
                bx,
                by,
                4,
                4,
            );
            let mut best_txtp_sse = best_dct_sse;
            let mut best_txtp_bits = best_dct_bits;
            for (sel, txtp) in [(TxSel::Adst, 4usize), (TxSel::Idtx, 0)] {
                let (mut acf, atf) = match sel {
                    TxSel::Adst => self.k.dct.adst4x4_t(&resid, &self.ctx.quant),
                    TxSel::Idtx => self.k.dct.idtx4x4_t(&resid, &self.ctx.quant),
                    TxSel::VDct => self.k.dct.fvdct4x4_t(&resid, &self.ctx.quant),
                    TxSel::HDct => self.k.dct.fhdct4x4_t(&resid, &self.ctx.quant),
                    _ => self.k.dct.dctadst4x4_t(&resid, &self.ctx.quant),
                };
                // ADST gets RDOQ; IDTX and the 1-D classes keep plain
                // forward levels (the 2026-07-22 full-RDOQ retry — ctx
                // trellis for IDTX, 1-D class trellis for V/H — measured
                // flat/negative on holdout at every format; these winners
                // live on exactness).
                if sel == TxSel::Adst {
                    trellis_optimize(&mut acf, &atf, dcq, acq, &SCAN_4X4, lam);
                }
                let arr = match sel {
                    TxSel::Adst => self.k.idct.iadst_dequant_4x4(&acf, &self.ctx.quant),
                    TxSel::Idtx => self.k.idct.iidentity_dequant_4x4(&acf, &self.ctx.quant),
                    TxSel::VDct => self.k.idct.ivdct_dequant_4x4(&acf, &self.ctx.quant),
                    TxSel::HDct => self.k.idct.ihdct_dequant_4x4(&acf, &self.ctx.quant),
                    _ => self.k.idct.idctadst_dequant_4x4(&acf, &self.ctx.quant),
                };
                let asse =
                    sse_recon::<16, 4>(&self.k.rd, &lpred, &arr, self.src_blk(0, bx, by, 4, 4), self.cfg.bd);
                let base_rd = rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits);
                let bits_bound = (base_rd - asse as f32) / mlam;
                let abits = match sel {
                    TxSel::VDct => self.luma_bits_1d_4x4(&acf, true, bx, by, best_mode),
                    TxSel::HDct => self.luma_bits_1d_4x4(&acf, false, bx, by, best_mode),
                    _ => self.luma_bits_bounded(
                        &acf, &SCAN_4X4, 4, bx, by, best_mode, txtp, bits_bound,
                    ),
                };
                let candidate_rd = rd_cost_i64(asse, mlam, abits);
                if asse <= best_dct_sse + self.tx_guard(best_dct_sse) && candidate_rd < base_rd {
                    lcf = acf;
                    tx4 = sel;
                    best_txtp_sse = asse;
                    best_txtp_bits = abits;
                }
            }
        }
        let luma_zero = self.k.rd.all_zero_i32(&lcf);

        // Chroma winner (popped here, pushed after the DC-vs-CfL decision;
        // exactly one per sub-block — chroma-less units push a DC dummy so
        // the cursor stays aligned across formats/mono).
        // --- chroma: DC (and, for 4:4:4, CfL) prediction + forward transform.
        // Per-unit chroma in 4:4:4; BR-only shared block in 4:2:0. ---
        let mut ccf = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2]; // chroma DC value per plane
        // Per-pixel chroma prediction (DC fills flat; CfL fills dc + alpha*ac).
        let mut cpred_px = [[0i32; 16]; 2];
        let mut chroma_zero = true;
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        if has_chroma {
            let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
            // DC option (always computed; the per-pixel prediction is the DC
            // value broadcast across the block).
            let mut dc_ccf = [[0i32; 16]; 2];
            let mut dc_sse = [0i64; 2];
            let mut dc_bits = [0f32; 2];
            let src_planes = [1, 2].map(|plane| self.src_blk(plane, chx, chy, 4, 4));
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = self.k.intrapred.dc_pred_4x4(&self.ctx.recon[plane], cstride, chx, chy, self.cfg.bd as i32);
                cpred[ci] = dc;
                let src = src_planes[ci];
                let mut cres = [0i32; 16];
                self.k.rd.residual_dc_blk(&mut cres, src, dc);
                let (mut q, qt) = self.k.dct.dct4x4_t(&cres, &self.ctx.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_4X4, lam);
                let rr = self.k.idct.idct_dequant_4x4(&q, &self.ctx.cquant);
                dc_ccf[ci] = q;
                dc_sse[ci] = sse_recon::<16, 4>(&self.k.rd, &[dc; 16], &rr, src, self.cfg.bd);
                dc_bits[ci] = self.chroma_bits(&q, &SCAN_4X4, 4, plane, chx, chy);
            }

            // CfL option (4:4:4 and 4:2:2; the 4:2:0 shared half-res block
            // stays DC-only). The AC reference is the chroma-co-located luma
            // reconstruction (this unit at 4:4:4; the 8x4 horizontal pair at
            // 4:2:2), inverse matching the winner's tx type.
            let cfl_eligible =
                !ss420 && !ss422;
            let mut cfl_ccf = [[0i32; 16]; 2];
            let mut cfl_a = [0i32; 2];
            let mut cfl_px = [[0i32; 16]; 2];
            let mut cfl_sse = [0i64; 2];
            let mut cfl_bits = [0f32; 2];
            let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 8);
            if cfl_eligible {
                let lrr_cfl = match tx4 {
                    TxSel::Adst => self.k.idct.iadst_dequant_4x4(&lcf, &self.ctx.quant),
                    TxSel::Idtx => self.k.idct.iidentity_dequant_4x4(&lcf, &self.ctx.quant),
                    TxSel::VDct => self.k.idct.ivdct_dequant_4x4(&lcf, &self.ctx.quant),
                    TxSel::HDct => self.k.idct.ihdct_dequant_4x4(&lcf, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_4x4(&lcf, &self.ctx.quant),
                };
                let mut luma_rec = [0u16; 16];
                recon_add_pred(&mut luma_rec, &lpred, &lrr_cfl, maxv);
                let mut ac = [0i32; 16];
                self.k.intrapred.cfl_ac_444(&luma_rec, 4, 4, &mut ac);
                let steps = crate::tuning::get().cfl_alpha_rd_steps;
                let a_e = [0, 1].map(|ci| {
                    self.k.intrapred.cfl_best_alpha(&ac, src_planes[ci], cpred[ci], self.cfg.bd)
                });
                for ci in 0..2 {
                    let dc = cpred[ci];
                    let src = src_planes[ci];
                    let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
                    let realise = |a: i32| {
                        let mut cpr = [0i32; 16];
                        self.k.intrapred.cfl_pred(&mut cpr, &ac[..16], dc, a, self.cfg.bd);
                        let mut resid = [0i32; 16];
                        self.k.rd.residual_pred_blk(&mut resid, &cpr, src);
                        let (mut q, qt) = self.k.dct.dct4x4_t(&resid, &self.ctx.cquant);
                        trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_4X4, lam);
                        let rr = self.k.idct.idct_dequant_4x4(&q, &self.ctx.cquant);
                        let sse = sse_recon::<16, 4>(&self.k.rd, &cpr, &rr, src, self.cfg.bd);
                        let bits = self.chroma_bits(&q, &SCAN_4X4, 4, ci + 1, chx, chy);
                        let pair = if ci == 0 { [a, other] } else { [other, a] };
                        let cost = cfl_alpha_walk_cost(
                    sse,
                    mlam,
                    bits,
                    self.uv_mode_bits(best_mode, CFL_PRED, Some(pair)),
                );
                        (cost, (q, cpr, sse, bits))
                    };
                    let (c0, p0) = realise(a_e[ci]);
                    let (a, (q, cpr, sse, bits)) =
                        cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                    cfl_a[ci] = a;
                    cfl_ccf[ci] = q;
                    cfl_px[ci] = cpr;
                    cfl_sse[ci] = sse;
                    cfl_bits[ci] = bits;
                }
            }

            // RD: pick CfL over DC only when it has a non-zero alpha and wins
            // including the joint signaling cost (sign symbol + a magnitude
            // per non-zero plane), mirroring the 8x8 4:4:4 path.
            let sig = self.uv_mode_bits(best_mode, CFL_PRED, Some(cfl_a));
            let dc_total = rd_cost_i64(
                dc_sse[0] + dc_sse[1],
                mlam,
                dc_bits[0] + dc_bits[1] + self.uv_mode_bits(best_mode, DC_PRED, None),
            );
            let cfl_total = rd_cost_i64(
                cfl_sse[0] + cfl_sse[1],
                mlam,
                cfl_bits[0] + cfl_bits[1] + sig,
            );
            if cfl_eligible && cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                for ci in 0..2 {
                    ccf[ci] = cfl_ccf[ci];
                    cpred_px[ci] = cfl_px[ci];
                    if !self.k.rd.all_zero_i32(&cfl_ccf[ci]) {
                        chroma_zero = false;
                    }
                }
            } else {
                for ci in 0..2 {
                    ccf[ci] = dc_ccf[ci];
                    cpred_px[ci] = [cpred[ci]; 16];
                    if !self.k.rd.all_zero_i32(&dc_ccf[ci]) {
                        chroma_zero = false;
                    }
                }
            }
        }

        let block_skip = if has_chroma {
            luma_zero && chroma_zero
        } else {
            luma_zero
        };

        // --- mode info: skip, y_mode (DC), [uv_mode (DC) if has_chroma] ---
        Sub4Pick {
            block_skip,
            best_mode,
            has_chroma,
            use_cfl,
            cfl_alpha_uv,
            tx4,
            lcf,
            lpred,
            chx,
            chy,
            ccf,
            cstride,
            cpred_px,
        }
    }

    /// Emit 4x4 block `si` of an 8x8 SPLIT from its search result.
    fn emit_sub4(&mut self, x8: usize, y8: usize, si: usize, pick: Sub4Pick) {
        let Sub4Pick {
            block_skip,
            best_mode,
            has_chroma,
            use_cfl,
            cfl_alpha_uv,
            tx4,
            lcf,
            lpred,
            chx,
            chy,
            ccf,
            cstride,
            cpred_px,
        } = pick;
        let (px, py) = (x8 * 8, y8 * 8);
        let maxv = (1 << self.cfg.bd) - 1;
        // z-order: TL, TR, BL, BR
        let (sx, sy) = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)][si];
        let (bx, by) = (px + sx * 4, py + sy * 4);
        let (bx4, by4) = (bx / 4, by / 4);
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        self.ent.enc.encode_symbol(best_mode, &mut self.ent.cdfs.kf_y[yctx]);
        if has_chroma {
            // uv context uses the luma mode of this unit. CfL signals the
            // joint sign + per-plane alpha; otherwise plain DC.
            if use_cfl {
                self.emit_uv_mode(best_mode, CFL_PRED, Some(cfl_alpha_uv), 0, bx, by, 4, 4);
            } else {
                self.emit_uv_mode(best_mode, DC_PRED, None, 0, bx, by, 4, 4);
            }
        }
        self.emit_palette_mode_info(bx, by, 4, 4, best_mode, has_chroma, None, None);
        self.emit_filter_intra(best_mode, 4, 4, None);
        self.tx_ctx_update4(bx, by);

        // --- residual: luma 4x4, then chroma U/V 4x4 (if has_chroma) ---
        let lres_ctx = if block_skip {
            0x40
        } else {
            let ds = self.dc_sign_ctx_420(0, bx4, by4);
            if tx4 == TxSel::VDct || tx4 == TxSel::HDct {
                encode_tx4_coeffs_1d(
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    &lcf,
                    tx4 == TxSel::VDct,
                    0, // luma TX_4X4 (tx == block) -> txb_skip ctx 0
                    ds,
                    best_mode,
                )
            } else {
                encode_tx4_luma_coeffs_adapt(
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    &lcf,
                    0, // luma TX_4X4 (tx == block) -> txb_skip ctx 0
                    ds,
                    best_mode,
                    match tx4 {
                        TxSel::Adst => 4,
                        TxSel::Idtx => 0,
                        _ => 1, // DCT_DCT
                    },
                )
            }
        };
        self.ctx.a_coef[0][bx4] = lres_ctx;
        self.ctx.l_coef[0][by4] = lres_ctx;

        // luma reconstruction
        let lrr = if block_skip {
            [0i32; 16]
        } else {
            match tx4 {
                TxSel::Adst => self.k.idct.iadst_dequant_4x4(&lcf, &self.ctx.quant),
                TxSel::Idtx => self.k.idct.iidentity_dequant_4x4(&lcf, &self.ctx.quant),
                TxSel::VDct => self.k.idct.ivdct_dequant_4x4(&lcf, &self.ctx.quant),
                TxSel::HDct => self.k.idct.ihdct_dequant_4x4(&lcf, &self.ctx.quant),
                _ => self.k.idct.idct_dequant_4x4(&lcf, &self.ctx.quant),
            }
        };
        for ry in 0..4 {
            let drow = &mut self.ctx.recon[0][(by + ry) * self.cfg.w + bx..];
            recon_add_pred(&mut drow[..4], &lpred[ry * 4..], &lrr[ry * 4..], maxv);
        }

        // chroma residual + reconstruction
        if has_chroma {
            let (bx4c, by4c) = (chx / 4, chy / 4);
            for ci in 0..2 {
                let plane = ci + 1;
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_420(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                    encode_4x4_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
                };
                self.ctx.a_coef[plane][bx4c] = res_ctx;
                self.ctx.l_coef[plane][by4c] = res_ctx;
                let rr = if block_skip {
                    [0i32; 16]
                } else {
                    self.k.idct.idct_dequant_4x4(&ccf[ci], &self.ctx.cquant)
                };
                // When the block is skipped (no residual) the prediction is
                // exactly the chroma reconstruction. For a skipped CfL block,
                // cpred_px already holds the CfL prediction.
                for ry in 0..4 {
                    let drow = &mut self.ctx.recon[plane][(chy + ry) * cstride + chx..];
                    recon_add_pred(
                        &mut drow[..4],
                        &cpred_px[ci][ry * 4..],
                        &rr[ry * 4..],
                        maxv,
                    );
                }
            }
        }

        // --- neighbor context updates for this 4x4 ---
        self.ctx.a_skip[bx4] = block_skip as u8;
        self.ctx.l_skip[by4] = block_skip as u8;
        self.ctx.a_mode[bx4] = best_mode as u8;
        self.ctx.l_mode[by4] = best_mode as u8;
    }

    /// The four 4x4 blocks of an 8x8 PARTITION_SPLIT, in z-order: each is
    /// searched and emitted before the next, since it predicts from the
    /// reconstruction of the ones before it. The caller has already emitted
    /// the PARTITION_SPLIT symbol.
    fn code_block_split4_dc(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        // Record all four 4x4 units through the shared helper rather than
        // hand-writing the transform map. The hand-written version set only
        // `blk4*` and so:
        //   * never advanced `emit_epoch`, leaving partition costs cached from
        //     BEFORE these four blocks changed reconstruction, mode, skip and
        //     coefficient contexts — a later sibling could reuse them;
        //   * never populated `pblk4*`, so each 4x4 PREDICTION block was absent
        //     from the prediction geometry the chroma deblock consults. The
        //     shipped path survives on `.max(1)` coercing the zero size, but
        //     the prediction-start flags stayed wrong.
        // Each 4x4 sub-block is its own prediction block, hence w4 = h4 = 1.
        // MEASURED QUALITY-NEUTRAL (Slow, 20 crops x 6 q): 444 +0.0016,
        // 420 -0.0022, spread -0.12..+0.15. Expected for a correctness fix that
        // removes stale-cache reuse — it makes decisions right, not better on
        // average.
        for uy in 0..2 {
            for ux in 0..2 {
                self.record_blk_rect4(x8 * 2 + ux, y8 * 2 + uy, 1, 1);
            }
        }
        // AV1 CDEF skips a 8x8 only when ALL FOUR covering 4x4 blocks are skip
        // (spec 7.15.1); accumulate the sub-block skips into the 8x8 map.
        let mut all4_skip = true;
        for si in 0..4 {
            let pick = self.pick_sub4(x8, y8, si, have_tr, have_bl);
            all4_skip &= pick.block_skip;
            self.emit_sub4(x8, y8, si, pick);
        }
        self.mark_skip8(x8, y8, 1, all4_skip);
    }
}

/// Chroma angle deltas whose prediction reads neither the top-right nor the
/// bottom-left edge (zone 2, 90 < angle < 180) — the only ones the chroma
/// searches can offer, since they predict with both edge flags off.
fn uv_zone2_deltas(mode: usize) -> &'static [i32] {
    match mode {
        V_PRED => &[1, 2, 3],
        H_PRED => &[-3, -2, -1],
        D113_PRED | D135_PRED | D157_PRED => &[-3, -2, -1, 1, 2, 3],
        _ => &[],
    }
}

/// 4:2:2 16x16-leaf chroma search winner: (uv mode, angle delta, U/V levels,
/// U/V predictions) on the 8x16 chroma block.
type UvRect422Win = (usize, i32, [[i32; 128]; 2], [[i32; 128]; 2]);
