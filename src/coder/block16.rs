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
    #[allow(clippy::too_many_arguments)]
    fn emit_uv_mode(
        &mut self,
        y_mode: usize,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
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
                    self.enc
                        .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                    coded_mode = DC_PRED;
                } else {
                    self.enc
                        .encode_symbol(CFL_PRED, &mut self.cdfs.uv_mode[13 + y_mode]);
                    self.enc.encode_symbol(sign - 1, &mut self.cdfs.cfl_sign);
                    if su != 0 {
                        let c = (su == 2) as usize * 3 + sv;
                        self.enc
                            .encode_symbol((a[0].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                    }
                    if sv != 0 {
                        let c = (sv == 2) as usize * 3 + su;
                        self.enc
                            .encode_symbol((a[1].abs() - 1) as usize, &mut self.cdfs.cfl_alpha[c]);
                    }
                    coded_mode = CFL_PRED;
                }
            }
            None => {
                self.enc
                    .encode_symbol(uv_mode, &mut self.cdfs.uv_mode[13 + y_mode]);
                // Directional chroma modes (here only V_PRED/H_PRED, used at 8x8
                // 4:4:4 chroma where `use_angle_delta` holds) emit a chroma
                // angle_delta symbol. The encoder only offers delta 0, so emit the
                // center bucket (delta + 3 == 3).
                if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
                    self.enc
                        .encode_symbol(3, &mut self.cdfs.angle_delta[uv_mode - V_PRED]);
                }
            }
        }
        self.commit_uv_mode(px, py, w, h, coded_mode);
    }

    #[allow(clippy::too_many_arguments)]
    fn code_header_luma16(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        txtp16: u8,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 2, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (angle_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl, px, py, 16, 16);
        self.emit_filter_intra(y_mode, 16, 16, filter_intra);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 4].fill(sv);
        self.l_skip[by4..by4 + 4].fill(sv);
        self.a_mode[bx4..bx4 + 4].fill(mv);
        self.l_mode[by4..by4 + 4].fill(mv);
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_16(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_16(0, bx4, by4);
            encode_tx16_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                lcf,
                false,
                sk,
                ds,
                filter_intra_tx_mode(filter_intra, y_mode),
                match txtp16 {
                    1 => ADST_ADST_TX16_IDX,
                    2 => ADST_DCT_TX16_IDX,
                    3 => DCT_ADST_TX16_IDX,
                    _ => 1,
                },
            )
        };
        self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
        self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
        // Pure-emit replay: recon is preinstalled from the record; the write
        // below would need the prediction the caller no longer computes.
        if self.sb_mode == SbMode::Replay {
            return;
        }
        let lrr = if block_skip {
            [0i32; 256]
        } else {
            match txtp16 {
                1 => iadst_dequant_16x16(lcf, &self.quant),
                2 => iadstdct_dequant_16x16(lcf, &self.quant),
                3 => idctadst_dequant_16x16(lcf, &self.quant),
                _ => idct_dequant_16x16(lcf, &self.quant),
            }
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<16>()
            .0
            .iter()
            .zip(lrr.as_chunks::<16>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_pred(drow, prow, rrow, (1 << self.bd) - 1);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block16_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let mut ccf = [[0i32; 256]; 2];
        let mut cpred = [0i32; 2];
        // Pure-emit replay: skip the DC baseline (block_skip below reads the
        // FINAL coeffs, installed from the record before it).
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let pred = dc_pred_16x16(&self.recon[plane], self.w, px, py, self.bd as i32);
            cpred[ci] = pred;
            let mut resid = [0i32; 256];
            for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(py + ry) * self.w + px..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - pred;
                }
            }
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(
                &mut ccf[ci],
                &qt,
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                &SCAN_16X16,
                trellis_lambda(),
            );
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // 4:4:4 CfL for the 16x16 chroma blocks (mirrors the 8x8 path).
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // before the emit below.
        let mut cpred16 = [[0i32; 256]; 2];
        let mut cfl_opt: Option<[i32; 2]> = None;
        if ru.is_none() {
            let lrr_cfl = match txtp16 {
                1 => iadst_dequant_16x16(lcf, &self.quant),
                2 => iadstdct_dequant_16x16(lcf, &self.quant),
                3 => idctadst_dequant_16x16(lcf, &self.quant),
                _ => idct_dequant_16x16(lcf, &self.quant),
            };
            let mut luma_rec = [0i32; 256];
            recon_add_pred(&mut luma_rec, lpred, &lrr_cfl, (1 << self.bd) - 1);
            let mut ac = [0i32; 256];
            cfl_ac_444(&luma_rec, 16, 16, &mut ac);
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                trellis_lambda(),
            );
            let mlam = self.mlam();
            let mut cfl_ccf = [[0i32; 256]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f32; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f32; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 256];
                for (ry, drow) in src.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..16]);
                }
                let dcrr = idct_dequant_16x16(&ccf[ci], &self.cquant);
                dc_sse[ci] = sse_recon::<256, 16>(&[dc; 256], &dcrr, &src, 16, 0, 0, self.bd);
                dc_bits[ci] = block_rate_bits(&ccf[ci], &SCAN_16X16);
                let a = cfl_best_alpha(&ac, &src, dc, 256, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 256];
                let mut resid = [0i32; 256];
                for i in 0..256 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                let rr = idct_dequant_16x16(&q, &self.cquant);
                cfl_ccf[ci] = q;
                cfl_sse[ci] = sse_recon::<256, 16>(&cpr, &rr, &src, 16, 0, 0, self.bd);
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_16X16);
                cpred16[ci] = cpr;
            }
            let sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let dc_total = rd_cost_i64(dc_sse[0] + dc_sse[1], mlam, dc_bits[0] + dc_bits[1]);
            let cfl_total = rd_cost_i64(
                cfl_sse[0] + cfl_sse[1],
                mlam,
                cfl_bits[0] + cfl_bits[1] + sig,
            );
            // Let the RD comparison decide DC-vs-CfL across the whole quality
            // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
            // where it helps most (high quality).
            if ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                cfl_opt = Some(cfl_a);
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred16[ci] = [cpred[ci]; 256];
                }
            }
        }
        let mut chosen_uv_16 = DC_PRED;
        // Pure-emit replay never runs the directional search either; the
        // captured winner (mode + coeffs) installs below.
        if ru.is_none() {
            let (dcq, acq, lam) = (
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                trellis_lambda(),
            );
            let mlam = self.mlam();
            // Reconstructed R-D of the CURRENT chroma choice (DC or CfL), using the
            // coeffs/prediction already selected above.
            let mut cur_total = 0f32;
            if let Some(a) = cfl_opt {
                // CfL signals a non-DC uv_mode plus per-plane alpha (~4 bits each).
                cur_total += rate_cost(
                    mlam,
                    4.0f32
                        + if a[0] != 0 { 4.0f32 } else { 0.0f32 }
                        + if a[1] != 0 { 4.0f32 } else { 0.0f32 },
                );
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let rr = idct_dequant_16x16(&ccf[ci], &self.cquant);
                let sse = sse_recon::<256, 16>(
                    &cpred16[ci],
                    &rr,
                    &self.src[plane],
                    self.w,
                    px,
                    py,
                    self.bd,
                );
                cur_total += rd_cost_i64(sse, mlam, block_rate_bits(&ccf[ci], &SCAN_16X16));
            }

            // Directional / smooth chroma modes, each with its decoder-derived
            // chroma tx (PAETH/SMOOTH -> ADST_ADST, SMOOTH_V -> ADST_DCT,
            // SMOOTH_H -> DCT_ADST). PAETH is empirically the strongest non-DC
            // chroma mode, so it is searched alongside the SMOOTH family. The
            // winner must beat the current DC/CfL choice on the libaom-style
            // R-D cost computed in `cur_total`.
            let mut best_total = cur_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf16 = [[0i32; 256]; 2];
            let mut best_pred16 = [[0i32; 256]; 2];
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
            let directional_top = if ru.is_none() {
                self.rank_chroma_directionals::<256>(candidates, px, py, px, py, 16, 16)
            } else {
                DirectionalTopK::new()
            };
            for &cand in candidates {
                if ru.is_some_and(|r| cand as u8 != r.uv) {
                    continue;
                }
                // V/H are cheap enough for every tier; Fast skips diagonal angles.
                if ru.is_none()
                    && cand != V_PRED
                    && cand != H_PRED
                    && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                    && !self.speed.chroma_angle_directional()
                {
                    continue;
                }
                if ru.is_none() && is_directional_mode(cand) && !directional_top.contains(cand) {
                    continue;
                }
                let tx = chroma_tx_for_mode(cand);
                let mut cand_ccf = [[0i32; 256]; 2];
                let mut cand_pred = [[0i32; 256]; 2];
                // V/H and the Z2 angular modes (D135/D113/D157) all emit a chroma
                // angle_delta symbol (~3 bits); they sit in the 1..=8 directional range.
                let sig_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    7.0f32
                } else {
                    4.0f32
                };
                let mut cand_total = rate_cost(mlam, sig_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.w,
                        px,
                        py,
                        16,
                        16,
                        false,
                        false,
                        self.w,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        self.bd,
                    );
                    let mut resid = [0i32; 256];
                    for (ry, drow) in resid.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                        let srow = &self.src[plane][(py + ry) * self.w + px..];
                        let prow = &cand_pred[ci][ry * 16..];
                        for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                            *dv = s - p;
                        }
                    }
                    let (mut q, qt) = fwd_chroma_16x16(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                    let mean_resid = resid.iter().sum::<i32>() / 256;
                    if q[0] == 0 && mean_resid.abs() >= 8 {
                        q[0] = if mean_resid > 0 { 1 } else { -1 };
                    }
                    cand_ccf[ci] = q;
                    let rr = inv_chroma_16x16(tx, &q, &self.cquant);
                    let sse = sse_recon::<256, 16>(
                        &cand_pred[ci],
                        &rr,
                        &self.src[plane],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_16X16));
                }
                if ru.is_some() || cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf16 = cand_ccf;
                    best_pred16 = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                ccf[..2].copy_from_slice(&best_ccf16[..2]);
                cpred16[..2].copy_from_slice(&best_pred16[..2]);
                cfl_opt = None; // a non-DC chroma mode overrides CfL if it wins
                chosen_uv_16 = best_mode_uv;
            }
        } // end directional/smooth chroma (4:4:4 16x16)
        // Pure-emit replay: install the captured chroma winner (the searches
        // above were skipped; recon is preinstalled from the record).
        if let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            for (dst, src) in ccf.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
            if r.uv == CFL_PRED as u8 {
                cfl_opt = Some(*al);
            } else if r.uv != DC_PRED as u8 {
                chosen_uv_16 = r.uv as usize;
            }
        }
        // Capture the final chroma winner (CfL folded in as CFL_PRED).
        self.push_uv_sel(UvSel {
            uv: if chosen_uv_16 != DC_PRED {
                chosen_uv_16 as u8
            } else if cfl_opt.is_some() {
                CFL_PRED as u8
            } else {
                DC_PRED as u8
            },
        });
        self.push_uv_cf(&ccf[0], &ccf[1], cfl_opt.unwrap_or([0, 0]));
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_16,
            cfl_opt,
            txtp16,
            angle_delta,
            filter_intra,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_16(plane, bx4, by4);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    0,
                    1,
                )
            };
            self.a_coef[plane][bx4..bx4 + 4].fill(res_ctx);
            self.l_coef[plane][by4..by4 + 4].fill(res_ctx);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let rr = if block_skip {
                [0i32; 256]
            } else if chosen_uv_16 != DC_PRED {
                // A directional/smooth chroma mode: invert with its decoder-derived
                // chroma tx (PAETH/SMOOTH -> ADST_ADST, SMOOTH_V -> ADST_DCT,
                // SMOOTH_H -> DCT_ADST).
                inv_chroma_16x16(chroma_tx_for_mode(chosen_uv_16), &ccf[ci], &self.cquant)
            } else {
                idct_dequant_16x16(&ccf[ci], &self.cquant)
            };
            let max = (1 << self.bd) - 1;
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if cfl_opt.is_some() || chosen_uv_16 != DC_PRED {
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

    #[allow(clippy::too_many_arguments)]
    fn code_block16_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f32,
            self.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        let maxval = (1 << self.bd) - 1;
        // DC path (skipped in pure-emit replay: block_skip below reads the
        // FINAL coeffs, installed from the record).
        let mut ccf_dc = [[0i32; 64]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let dc = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 64];
            for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                    *dv = s - dc;
                }
            }
            let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 64;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // DC baseline R-D (libaom-style: SSE + mlam*coeff_bits, summed over U+V).
        let mlam = self.mlam();
        let mut rr_dc = [[0i32; 64]; 2];
        let mut dc_total = 0f32;
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_8x8(&ccf_dc[ci], &self.cquant);
            let dc = dc_preds[ci];
            let sse = sse_recon::<64, 8>(
                &[dc; 64],
                &rr_dc[ci],
                &self.src[plane],
                self.cw,
                cx,
                cy,
                self.bd,
            );
            dc_total += rd_cost_i64(sse, mlam, block_rate_bits(&ccf_dc[ci], &SCAN_8X8));
        }
        // Directional / smooth chroma modes, each with its decoder-derived chroma tx.
        // PAETH is empirically the strongest non-DC chroma mode, searched alongside
        // the SMOOTH family. Winner must beat DC on the same R-D metric.
        let mut best_total = dc_total;
        let mut chosen_uv = DC_PRED;
        let mut best_ccf = ccf_dc;
        let mut best_rr = rr_dc;
        let mut best_pred = [[0i32; 64]; 2];
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
        let directional_top = if ru.is_none() {
            self.rank_chroma_directionals::<64>(candidates, px, py, cx, cy, 8, 8)
        } else {
            DirectionalTopK::new()
        };
        for &cand in candidates {
            // Pure-emit replay: no candidate runs; the winner installs below.
            if ru.is_some() {
                break;
            }
            // V/H are cheap enough for every tier; Fast skips diagonal angles.
            if cand != V_PRED
                && cand != H_PRED
                && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                && !self.speed.chroma_angle_directional()
            {
                continue;
            }
            if is_directional_mode(cand) && !directional_top.contains(cand) {
                continue;
            }
            let tx = chroma_tx_for_mode(cand);
            let mut cand_ccf = [[0i32; 64]; 2];
            let mut cand_rr = [[0i32; 64]; 2];
            let mut cand_pred = [[0i32; 64]; 2];
            let sig_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                7.0f32
            } else {
                4.0f32
            };
            let mut cand_total = rate_cost(mlam, sig_bits);
            for ci in 0..2 {
                let plane = ci + 1;
                intra_predict_nd(
                    cand,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    8,
                    8,
                    false,
                    false,
                    self.cw,
                    self.h,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci],
                    self.bd,
                );
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &cand_pred[ci][ry * 8..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (mut q, qt) = fwd_chroma_8x8(tx, &resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                let mean_resid = resid.iter().sum::<i32>() / 64;
                if q[0] == 0 && mean_resid.abs() >= 8 {
                    q[0] = if mean_resid > 0 { 1 } else { -1 };
                }
                cand_ccf[ci] = q;
                cand_rr[ci] = inv_chroma_8x8(tx, &q, &self.cquant);
                let sse = sse_recon::<64, 8>(
                    &cand_pred[ci],
                    &cand_rr[ci],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    self.bd,
                );
                cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_8X8));
            }
            if ru.is_some() || cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                best_ccf = cand_ccf;
                best_rr = cand_rr;
                best_pred = cand_pred;
            }
        }
        // Pure-emit replay: install the captured chroma winner (mode + coeffs).
        if let Some(r) = ru
            && let Some((cf, _al)) = ru_cf.as_ref()
        {
            chosen_uv = r.uv as usize;
            for (dst, src) in best_ccf.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
        }
        let (ccf, rr_cache) = (best_ccf, best_rr);
        let sv_preds = best_pred;
        let use_nondc = chosen_uv != DC_PRED;
        self.push_uv_sel(UvSel {
            uv: chosen_uv as u8,
        });
        self.push_uv_cf(&ccf[0], &ccf[1], [0, 0]);
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            None,
            txtp16,
            angle_delta,
            filter_intra,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx(plane, bx4c, by4c);
                encode_tx8_coeffs_adapt(&mut self.enc, &mut self.cdfs, &ccf[ci], true, sk, ds, 0, 1)
            };
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 2].fill(res_ctx);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let rr = if block_skip { [0i32; 64] } else { rr_cache[ci] };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
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
    #[allow(clippy::too_many_arguments)]
    fn code_block16_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 256],
        lpred: &[i32; 256],
        y_mode: usize,
        luma_zero: bool,
        txtp16: u8,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block16 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let maxv = (1 << self.bd) - 1;
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f32,
            self.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        let mlam = self.mlam();
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        // Per-pixel chroma prediction (DC broadcast, or CfL dc+alpha*ac).
        let mut cpred_px = [[0i32; 128]; 2];
        let mut src_planes = [[0i32; 128]; 2];
        // DC option (always computed).
        let mut dc_ccf = [[0i32; 128]; 2];
        let mut dc_sse = [0i64; 2];
        let mut dc_bits = [0f32; 2];
        // DC option (skipped in pure-emit replay; the captured winner installs
        // below and block_skip reads the FINAL coeffs, matching Off).
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let pred = dc_pred_8x16(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut src = [0i32; 128];
            let mut resid = [0i32; 128];
            for (ry, (drow, srow_dst)) in resid
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(src.as_chunks_mut::<8>().0.iter_mut())
                .enumerate()
            {
                let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                for ((dv, sd), &s) in drow.iter_mut().zip(srow_dst.iter_mut()).zip(srow.iter()) {
                    *dv = s - pred;
                    *sd = s;
                }
            }
            src_planes[ci] = src;
            let (mut q, qt) = forward_dct_quant_8x16_t(&resid, &self.cquant);
            trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X16, lam);
            let rr = idct_dequant_8x16(&q, &self.cquant);
            dc_ccf[ci] = q;
            dc_sse[ci] = crate::rd_sse::sse_recon(&[pred; 128], &rr, &src, 8, 0, 0, 8, 16, self.bd);
            dc_bits[ci] = block_rate_bits(&q, &SCAN_8X16);
        }

        // CfL option: predict 8x16 U/V from the horizontally-subsampled 16x16
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0). Mirrors the
        // 4:2:2 4x8 CfL in `code_block` at the larger 8x16 chroma size.
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // below.
        if ru.is_none() {
            let lrr_cfl = match txtp16 {
                1 => iadst_dequant_16x16(lcf, &self.quant),
                2 => iadstdct_dequant_16x16(lcf, &self.quant),
                3 => idctadst_dequant_16x16(lcf, &self.quant),
                _ => idct_dequant_16x16(lcf, &self.quant),
            };
            let mut luma_rec = [0i32; 256];
            recon_add_pred(&mut luma_rec, lpred, &lrr_cfl, maxv);
            let mut ac = [0i32; 128];
            cfl_ac_sub(&luma_rec, 16, 8, 16, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let mut cfl_sse = [0i64; 2];
            let mut cfl_bits = [0f32; 2];
            for ci in 0..2 {
                let dc = cpred[ci];
                let src = src_planes[ci];
                let a = cfl_best_alpha(&ac, &src, dc, 128, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 128];
                let mut resid = [0i32; 128];
                for i in 0..128 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_8x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X16, lam);
                let rr = idct_dequant_8x16(&q, &self.cquant);
                cfl_ccf[ci] = q;
                cfl_sse[ci] = crate::rd_sse::sse_recon(&cpr, &rr, &src, 8, 0, 0, 8, 16, self.bd);
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_8X16);
                cpred_px[ci] = cpr;
            }
            let sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let dc_total = rd_cost_i64(dc_sse[0] + dc_sse[1], mlam, dc_bits[0] + dc_bits[1]);
            let cfl_total = rd_cost_i64(
                cfl_sse[0] + cfl_sse[1],
                mlam,
                cfl_bits[0] + cfl_bits[1] + sig,
            );
            if ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        if ru.is_none() && !use_cfl {
            for ci in 0..2 {
                ccf[ci] = dc_ccf[ci];
                cpred_px[ci] = [cpred[ci]; 128];
            }
        }
        // Pure-emit replay: install the captured chroma winner (coeffs +
        // CfL alphas; recon is preinstalled from the record).
        if let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
            for (dst, src) in ccf.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
        }
        self.push_uv_sel(UvSel {
            uv: if use_cfl {
                CFL_PRED as u8
            } else {
                DC_PRED as u8
            },
        });
        self.push_uv_cf(
            &ccf[0],
            &ccf[1],
            if use_cfl { cfl_alpha_uv } else { [0, 0] },
        );
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma16(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            if use_cfl { CFL_PRED } else { DC_PRED },
            if use_cfl { Some(cfl_alpha_uv) } else { None },
            txtp16,
            angle_delta,
            filter_intra,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_8x16_422(plane, bx4c, by4c);
                encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            // RTX_8X16: 2 coef-context units wide, 4 units tall.
            self.a_coef[plane][bx4c..bx4c + 2].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let rr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_8x16(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
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
    fn code_block_split4_dc(&mut self, x8: usize, y8: usize) {
        let (px, py) = (x8 * 8, y8 * 8);
        let maxv = (1 << self.bd) - 1;
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
        // mark all four 4x4 luma units for the deblock filter (blk4 == 1)
        let nc4 = self.w / 4;
        for uy in 0..2 {
            for ux in 0..2 {
                self.blk4[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = 1;
                self.blk4h[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = 1;
                self.blk4v[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = true;
                self.blk4t[(y8 * 2 + uy) * nc4 + (x8 * 2 + ux)] = true;
            }
        }
        // Chroma layout differs by subsampling:
        //   4:2:0 -> the four 4x4 luma units share ONE 4x4 chroma block, coded on
        //            the bottom-right sub-block (origin px/2, py/2).
        //   4:4:4 -> chroma is full resolution: EVERY 4x4 luma sub-block carries a
        //            co-located 4x4 chroma block (dav1d has_chroma is true for each
        //            BLOCK_4X4 in I444). Each sub-block emits its own uv_mode and
        //            U/V residual at the same (bx, by) with stride w.
        // (4:2:2 is excluded upstream via `split_eligible`.)
        let ss420 = self.ss420;
        let (cx420, cy420) = (px / 2, py / 2);
        // AV1 CDEF skips a 8x8 only when ALL FOUR covering 4x4 blocks are skip
        // (spec 7.15.1); accumulate the sub-block skips into the 8x8 map.
        let mut all4_skip = true;
        // z-order: TL, TR, BL, BR
        let sub = [(0usize, 0usize), (1, 0), (0, 1), (1, 1)];
        for (si, &(sx, sy)) in sub.iter().enumerate() {
            let (bx, by) = (px + sx * 4, py + sy * 4);
            let (bx4, by4) = (bx / 4, by / 4);
            // In 4:2:0 chroma is coded only on the bottom-right unit; in 4:4:4
            // every unit carries chroma.
            let has_chroma = !self.mono && (!ss420 || si == 3);
            // Chroma origin / stride for this unit: full-res co-located for 4:4:4,
            // half-res shared block for 4:2:0.
            let (chx, chy) = if ss420 { (cx420, cy420) } else { (bx, by) };
            let cstride = self.cw;

            // --- luma 4x4: search the non-directional intra modes DC, SMOOTH
            // and PAETH. SMOOTH_V/SMOOTH_H are intentionally excluded: at the
            // 4x4 size their reconstruction does not match dav1d here, and their
            // win over plain SMOOTH is rare/small. These modes use only the
            // above row, left column and above-left corner, so top-right/
            // bottom-left availability is irrelevant; the tx-type is signaled
            // (DCT_DCT), so the mode choice never desyncs.
            let mlam = self.mlam();
            let modes = fast_nd_modes();
            let mut best_mode = DC_PRED;
            let mut lpred = [0i32; 16];
            let mut lcf = [0i32; 16];
            let mut best_eff = f32::INFINITY;
            // Pure-emit replay: skip the mode loop entirely (one LumaSel per
            // 4x4 sub-block, in z-order; delta/filter/tx are always the
            // defaults here — the SPLIT4 path is DCT-only, no refinements).
            let rl = self.luma_sel_replay();
            let rl_cf = self.luma_cf_replay();
            for &m in modes {
                if rl.is_some() {
                    break;
                }
                let mut pred = [0i32; 16];
                if m == DC_PRED {
                    let d = dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    intra_predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        false,
                        false,
                        self.w,
                        self.h,
                        self.luma_filter_type(bx, by),
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                for ry in 0..4 {
                    let srow = &self.src[0][(by + ry) * self.w + bx..];
                    for rx in 0..4 {
                        resid[ry * 4 + rx] = srow[rx] - pred[ry * 4 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_4x4_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_4X4, lam);
                let rr = idct_dequant_4x4(&cf, &self.quant);
                let sse = sse_recon::<16, 4>(&pred, &rr, &self.src[0], self.w, bx, by, self.bd);
                let bits = block_rate_bits(&cf, &SCAN_4X4);
                let eff = rd_cost_i64(sse, mlam, bits);
                if eff < best_eff {
                    best_eff = eff;
                    best_mode = m;
                    lpred = pred;
                    lcf = cf;
                }
            }
            // Pure-emit replay: install the recorded winner + captured coeffs.
            if let Some(r) = rl {
                best_mode = r.mode as usize;
            }
            if let Some(cf) = rl_cf {
                lcf.copy_from_slice(&cf);
            }
            self.push_luma_sel(LumaSel {
                mode: best_mode as u8,
                delta: 0,
                filter: NO_FILTER,
                tx: TxSel::Dct,
            });
            self.push_luma_cf(&lcf);
            let luma_zero = lcf.iter().all(|&c| c == 0);

            // Chroma winner (popped here, pushed after the DC-vs-CfL decision;
            // exactly one per sub-block — chroma-less units push a DC dummy so
            // the cursor stays aligned across formats/mono).
            let ru = self.uv_sel_replay();
            let ru_cf = self.uv_cf_replay();
            // --- chroma: DC (and, for 4:4:4, CfL) prediction + forward transform.
            // Per-unit chroma in 4:4:4; BR-only shared block in 4:2:0. ---
            let mut ccf = [[0i32; 16]; 2];
            let mut cpred = [0i32; 2]; // chroma DC value per plane
            // Per-pixel chroma prediction (DC fills flat; CfL fills dc + alpha*ac).
            let mut cpred_px = [[0i32; 16]; 2];
            let mut chroma_zero = true;
            let mut use_cfl = false;
            let mut cfl_alpha_uv = [0i32; 2];
            if has_chroma && ru.is_none() {
                let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
                // DC option (always computed; the per-pixel prediction is the DC
                // value broadcast across the block).
                let mut dc_ccf = [[0i32; 16]; 2];
                let mut dc_sse = [0i64; 2];
                let mut dc_bits = [0f32; 2];
                let mut src_planes = [[0i32; 16]; 2];
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = dc_pred_4x4(&self.recon[plane], cstride, chx, chy, self.bd as i32);
                    cpred[ci] = dc;
                    let mut src = [0i32; 16];
                    for ry in 0..4 {
                        let srow = &self.src[plane][(chy + ry) * cstride + chx..];
                        for rx in 0..4 {
                            src[ry * 4 + rx] = srow[rx];
                        }
                    }
                    src_planes[ci] = src;
                    let mut cres = [0i32; 16];
                    for (dst, &src) in cres[..16].iter_mut().zip(src[..16].iter()) {
                        *dst = src - dc;
                    }
                    let (mut q, qt) = forward_dct_quant_4x4_t(&cres, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_4X4, lam);
                    let rr = idct_dequant_4x4(&q, &self.cquant);
                    dc_ccf[ci] = q;
                    dc_sse[ci] = sse_recon::<16, 4>(&[dc; 16], &rr, &src, 4, 0, 0, self.bd);
                    dc_bits[ci] = block_rate_bits(&q, &SCAN_4X4);
                }

                // CfL option (4:4:4 only; 4:2:0 chroma here is the shared half-res
                // block and is left DC-only). The AC reference is this 4x4 luma
                // unit's reconstruction (luma is always DCT_DCT in SPLIT4).
                // Replay: evaluate CfL only when it actually won.
                let cfl_eligible = !ss420 && ru.is_none_or(|r| r.uv == CFL_PRED as u8);
                let mut cfl_ccf = [[0i32; 16]; 2];
                let mut cfl_a = [0i32; 2];
                let mut cfl_px = [[0i32; 16]; 2];
                let mut cfl_sse = [0i64; 2];
                let mut cfl_bits = [0f32; 2];
                if cfl_eligible {
                    let lrr_cfl = idct_dequant_4x4(&lcf, &self.quant);
                    let mut luma_rec = [0i32; 16];
                    recon_add_pred(&mut luma_rec, &lpred, &lrr_cfl, maxv);
                    let mut ac = [0i32; 16];
                    cfl_ac_444(&luma_rec, 4, 4, &mut ac);
                    for ci in 0..2 {
                        let dc = cpred[ci];
                        let src = src_planes[ci];
                        let a = cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                        cfl_a[ci] = a;
                        let mut cpr = [0i32; 16];
                        let mut resid = [0i32; 16];
                        for i in 0..16 {
                            cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                            resid[i] = src[i] - cpr[i];
                        }
                        let (mut q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_4X4, lam);
                        let rr = idct_dequant_4x4(&q, &self.cquant);
                        cfl_ccf[ci] = q;
                        cfl_a[ci] = a;
                        cfl_px[ci] = cpr;
                        cfl_sse[ci] = sse_recon::<16, 4>(&cpr, &rr, &src, 4, 0, 0, self.bd);
                        cfl_bits[ci] = block_rate_bits(&q, &SCAN_4X4);
                    }
                }

                // RD: pick CfL over DC only when it has a non-zero alpha and wins
                // including the joint signaling cost (sign symbol + a magnitude
                // per non-zero plane), mirroring the 8x8 4:4:4 path.
                let sig = 4.0f32
                    + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                    + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
                let dc_total = rd_cost_i64(dc_sse[0] + dc_sse[1], mlam, dc_bits[0] + dc_bits[1]);
                let cfl_total = rd_cost_i64(
                    cfl_sse[0] + cfl_sse[1],
                    mlam,
                    cfl_bits[0] + cfl_bits[1] + sig,
                );
                if cfl_eligible
                    && (ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)))
                {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    for ci in 0..2 {
                        ccf[ci] = cfl_ccf[ci];
                        cpred_px[ci] = cfl_px[ci];
                        if !cfl_ccf[ci].iter().all(|&c| c == 0) {
                            chroma_zero = false;
                        }
                    }
                } else {
                    for ci in 0..2 {
                        ccf[ci] = dc_ccf[ci];
                        cpred_px[ci] = [cpred[ci]; 16];
                        if !dc_ccf[ci].iter().all(|&c| c == 0) {
                            chroma_zero = false;
                        }
                    }
                }
            }

            // Pure-emit replay: install the captured chroma winner for this
            // sub-block (empty record entry when it carries no chroma).
            if has_chroma
                && let Some(r) = ru
                && let Some((cf, al)) = ru_cf.as_ref()
            {
                use_cfl = r.uv == CFL_PRED as u8;
                cfl_alpha_uv = *al;
                for (dst, src) in ccf.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
                chroma_zero =
                    ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
            }
            // Capture the sub-block's chroma winner (DC dummy when it carries
            // no chroma, so pushes and pops stay 1:1 per sub-block).
            self.push_uv_sel(UvSel {
                uv: if use_cfl { CFL_PRED } else { DC_PRED } as u8,
            });
            if has_chroma {
                self.push_uv_cf(
                    &ccf[0],
                    &ccf[1],
                    if use_cfl { cfl_alpha_uv } else { [0, 0] },
                );
            } else {
                self.push_uv_cf(&[], &[], [0, 0]);
            }

            let block_skip = if has_chroma {
                luma_zero && chroma_zero
            } else {
                luma_zero
            };

            // --- mode info: skip, y_mode (DC), [uv_mode (DC) if has_chroma] ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
            if has_chroma {
                // uv context uses the luma mode of this unit. CfL signals the
                // joint sign + per-plane alpha; otherwise plain DC.
                if use_cfl {
                    self.emit_uv_mode(best_mode, CFL_PRED, Some(cfl_alpha_uv), bx, by, 4, 4);
                } else {
                    self.emit_uv_mode(best_mode, DC_PRED, None, bx, by, 4, 4);
                }
            }
            self.emit_filter_intra(best_mode, 4, 4, None);

            // --- residual: luma 4x4, then chroma U/V 4x4 (if has_chroma) ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let ds = self.dc_sign_ctx_420(0, bx4, by4);
                encode_tx4_luma_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &lcf,
                    0, // luma TX_4X4 (tx == block) -> txb_skip ctx 0
                    ds,
                    best_mode,
                    1, // DCT_DCT
                )
            };
            self.a_coef[0][bx4] = lres_ctx;
            self.l_coef[0][by4] = lres_ctx;

            // luma reconstruction (skipped in pure-emit replay: preinstalled)
            if self.sb_mode != SbMode::Replay {
                let lrr = if block_skip {
                    [0i32; 16]
                } else {
                    idct_dequant_4x4(&lcf, &self.quant)
                };
                for ry in 0..4 {
                    let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                    recon_add_pred(&mut drow[..4], &lpred[ry * 4..], &lrr[ry * 4..], maxv);
                }
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
                        encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                    };
                    self.a_coef[plane][bx4c] = res_ctx;
                    self.l_coef[plane][by4c] = res_ctx;
                    if self.sb_mode == SbMode::Replay {
                        continue; // recon preinstalled
                    }
                    let rr = if block_skip {
                        [0i32; 16]
                    } else {
                        idct_dequant_4x4(&ccf[ci], &self.cquant)
                    };
                    // When the block is skipped (no residual) the prediction is
                    // exactly the chroma reconstruction. For a skipped CfL block,
                    // cpred_px already holds the CfL prediction.
                    for ry in 0..4 {
                        let drow = &mut self.recon[plane][(chy + ry) * cstride + chx..];
                        recon_add_pred(&mut drow[..4], &cpred_px[ci][ry * 4..], &rr[ry * 4..], maxv);
                    }
                }
            }

            // --- neighbor context updates for this 4x4 ---
            self.a_skip[bx4] = block_skip as u8;
            self.l_skip[by4] = block_skip as u8;
            self.a_mode[bx4] = best_mode as u8;
            self.l_mode[by4] = best_mode as u8;
            all4_skip &= block_skip;
        }
        self.mark_skip8(x8, y8, 1, all4_skip);
    }
}
