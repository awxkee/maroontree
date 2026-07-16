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
    fn code_block(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 2);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let cx = px / 2; // chroma column for 4:2:2

        // Forward-transform/quantize all planes up front to decide block skip.
        // Luma is always 8x8; chroma is 8x8 (4:4:4) or 4x8 (4:2:2).
        // Luma 8x8: search the non-directional intra modes (DC + SMOOTH*/PAETH)
        // and keep the one minimizing pixel SSE + lambda * estimated bits. The
        // chosen prediction is per-pixel; reconstruction uses the same array so
        // the decoder (which re-derives the identical prediction) stays bit-exact.
        let (dcq, acq, lam) = (
            self.quant.dc_q() as f32,
            self.quant.ac_q() as f32,
            trellis_lambda(),
        );
        let mlam = self.mlam();
        let prdo = self.perceptual_rd_scale(px, py, 8);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best_mode = DC_PRED;
        let mut best_is_adst = false;
        let mut best_is_idtx = false;
        let mut best_is_adstdct = false;
        let mut best_is_dctadst = false;
        let mut lpred_arr = [0i32; 64];
        let mut lcf = [0i32; 64];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        let dc_sgn = self.dc_sign_ctx(0, px / 4, py / 4);
        let mut ltf = [0f32; 64]; // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        let directional_top =
            self.rank_luma_directionals::<64>(modes, px, py, 8, 8, have_tr, have_bl);
        for &m in modes {
            if is_directional_mode(m) && !directional_top.contains(m) {
                continue;
            }
            let mut pred = [0i32; 64];
            if m == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<8>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<8>().0.iter())
                .enumerate()
            {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            // Mode decision uses DCT_DCT only (cheap); the ADST_ADST transform
            // choice is refined once for the winning mode after the loop.
            let blk_sse = |rr: &[i32; 64]| -> i64 {
                let mut sse = 0i64;
                for (ry, (prow, rrow)) in pred
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .zip(rr.as_chunks::<8>().0.iter())
                    .enumerate()
                {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        sse += (d * d) as i64;
                    }
                }
                sse
            };
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    &self.cdfs,
                    1,
                    0,
                    &self.cdfs.eob_bin_64_l,
                    dc_sgn,
                );
            }
            let sse = blk_sse(&idct_dequant_8x8(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_8X8);
            let filter_bits = if m == DC_PRED {
                cdf_cost(&self.cdfs.filter_intra[av1_block_size_index(8, 8)], 0)
            } else {
                0.0
            };
            let cost = rd_cost_i64(sse, mlam, bits + mode_signal_bits(m) + filter_bits);
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
                best_filter_intra = None;
            }
        }
        if self.speed == Speed::Slow {
            let bsize = av1_block_size_index(8, 8);
            for filter_mode in FILTER_INTRA_MODES {
                let mut pred = [0i32; 64];
                filter_intra_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 64];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                );
                let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_8X8,
                        lam,
                        8,
                        &self.cdfs,
                        1,
                        0,
                        &self.cdfs.eob_bin_64_l,
                        dc_sgn,
                    );
                }
                let rr = idct_dequant_8x8(&cf, &self.quant);
                let sse = sse_recon::<64, 8>(
                    &pred,
                    &rr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    self.bd,
                );
                let bits = block_rate_bits(&cf, &SCAN_8X8);
                let syntax_bits = mode_signal_bits(DC_PRED)
                    + cdf_cost(&self.cdfs.filter_intra[bsize], 1)
                    + cdf_cost(&self.cdfs.filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if filter_intra_sse_allowed(sse, best_dct_sse) && cost < best_eff {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                    best_filter_intra = Some(filter_mode);
                }
            }
        }
        // Angle-delta winner refinement: if the winning luma mode is one of the
        // six pure diagonals, try angle_delta in -3..=3 (3 deg steps) and keep the
        // best by SSE + lambda*(coeff bits + angle_delta symbol bits). V/H and the
        // non-directional modes stay at delta 0. ~6 extra predictions per block.
        let mut best_delta: i32 = 0;
        if angle_delta_enabled()
            && self.speed.try_angle_deltas()
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.cdfs.angle_delta[best_mode - V_PRED]);
            let mut best_ad_cost =
                rd_cost_i64(best_dct_sse, mlam, best_dct_bits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 64];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 64];
                for ry in 0..8 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..8 {
                        resid[ry * 8 + rx] = srow[rx] - pred[ry * 8 + rx];
                    }
                }
                let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_8X8,
                        lam,
                        8,
                        &self.cdfs,
                        1,
                        0,
                        &self.cdfs.eob_bin_64_l,
                        dc_sgn,
                    );
                }
                let rr = idct_dequant_8x8(&cf, &self.quant);
                let mut sse = 0i64;
                for ry in 0..8 {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    for rx in 0..8 {
                        let r = (pred[ry * 8 + rx] + rr[ry * 8 + rx]).clamp(0, (1 << self.bd) - 1);
                        let dd = srow[rx] - r;
                        sse += (dd * dd) as i64;
                    }
                }
                let bits = block_rate_bits(&cf, &SCAN_8X8);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
        }
        let mut best_txtp_sse = best_dct_sse;
        let mut best_txtp_bits = best_dct_bits;
        if self.speed.try_adst() {
            let mut resid = [0i32; 64];
            for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (mut acf, atf) = adst8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                &self.cdfs,
                1,
                0,
                &self.cdfs.eob_bin_64_l,
                dc_sgn,
            );
            let rr = iadst_dequant_8x8(&acf, &self.quant);
            let mut asse = 0i64;
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    asse += (d * d) as i64;
                }
            }
            let abits = block_rate_bits(&acf, &SCAN_8X8);
            // Quality guard (see 16x16 ADST note): block low-q distortion-for-rate trades.
            if asse <= best_dct_sse + (best_dct_sse >> 5)
                && rd_cost_i64(asse, mlam, abits) < rd_cost_i64(best_dct_sse, mlam, best_dct_bits)
            {
                lcf = acf;
                best_is_adst = true;
                best_txtp_sse = asse;
                best_txtp_bits = abits;
            }
        }
        // Per-block asymmetric-ADST refinement. Intra residual is anisotropic:
        // it grows away from the reference edge in one direction (wants ADST
        // there) and is flat across it (wants DCT). ADST_DCT = vertical ADST,
        // DCT_ADST = horizontal ADST. Each competes with the running tx winner.
        if self.speed.try_adst() && asym_adst_enabled() {
            for (fwd_t, inv_is_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 64];
                for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                    for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                        *r = s - p;
                    }
                }
                let (mut acf, atf) = if fwd_t {
                    dctadst8x8_t(&resid, &self.quant)
                } else {
                    adstdct8x8_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    &self.cdfs,
                    1,
                    0,
                    &self.cdfs.eob_bin_64_l,
                    dc_sgn,
                );
                let rr = if inv_is_dctadst {
                    idctadst_dequant_8x8(&acf, &self.quant)
                } else {
                    iadstdct_dequant_8x8(&acf, &self.quant)
                };
                let mut asse = 0i64;
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let srow = &self.src[0][(py + ry) * self.w + px..];
                    let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                    for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                        let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                        let d = s - r;
                        asse += (d * d) as i64;
                    }
                }
                let abits = block_rate_bits(&acf, &SCAN_8X8);
                if asse <= best_dct_sse + (best_dct_sse >> 5)
                    && rd_cost_i64(asse, mlam, abits)
                        < rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits)
                {
                    lcf = acf;
                    best_is_adst = false;
                    best_is_idtx = false;
                    best_is_adstdct = !inv_is_dctadst;
                    best_is_dctadst = inv_is_dctadst;
                    best_txtp_sse = asse;
                    best_txtp_bits = abits;
                }
            }
        }
        // Per-block IDTX refinement: the identity transform (no spatial
        // decorrelation) wins on sharp edges / screen-content-like residuals
        // where DCT/ADST spread a step across many coefficients. One extra
        // forward+inverse on the winning prediction; kept only if it beats the
        // current best tx by real recon SSE + estimated bits. Bit-exactness is
        // carried by `iidentity_dequant_8x8` (dav1d's exact TX_8X8 IDTX inverse);
        // the IDTX symbol is 0 in the 7-type intra ext-tx set.
        if self.speed.try_adst() {
            let mut resid = [0i32; 64];
            for (ry, rrow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for (r, (&p, &s)) in rrow.iter_mut().zip(prow.iter().zip(srow.iter())) {
                    *r = s - p;
                }
            }
            let (icf, _itf) = fidentity8x8_t(&resid, &self.quant);
            // No RDOQ on IDTX: because the identity transform spreads a residual
            // across many small coefficients, an aggressive trellis zeros them
            // all and the block-level bit term then picks the collapsed result.
            // Plain forward levels keep IDTX conservative (chosen only on a clear
            // real-SSE win); bit-exactness is carried by the inverse regardless.
            let rr = iidentity_dequant_8x8(&icf, &self.quant);
            let mut isse = 0i64;
            for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                let prow = &lpred_arr[ry * 8..ry * 8 + 8];
                for ((&p, &rv), &s) in prow.iter().zip(rrow.iter()).zip(srow.iter()) {
                    let r = (p + rv).clamp(0, (1 << self.bd) - 1);
                    let d = s - r;
                    isse += (d * d) as i64;
                }
            }
            let ibits = block_rate_bits(&icf, &SCAN_8X8);
            // Quality guard (see ADST note): identity spreads residual energy and
            // is cheap to code, so at low-q lambda a pure RD test over-selects it
            // and flattens detail. Require SSE-non-worsening vs the best real tx.
            if isse <= best_txtp_sse + (best_txtp_sse >> 5)
                && rd_cost_i64(isse, mlam, ibits) < rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits)
            {
                lcf = icf;
                best_is_adst = false;
                best_is_idtx = true;
                best_is_adstdct = false;
                best_is_dctadst = false;
            }
        }
        let mut ccf8 = [[0i32; 64]; 2];
        let mut ccf48 = [[0i32; 32]; 2];
        let mut ccf44 = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2];
        let cy = py / 2; // chroma row for 4:2:0
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let pred = dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                ccf44[ci] = q;
                trellis_optimize(&mut ccf44[ci], &qt, dcq, acq, &SCAN_4X4, lam);
            } else if self.ss422 {
                let pred = dc_pred_4x8(&self.recon[plane], self.cw, cx, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 32];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.cw + cx..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                ccf48[ci] = q;
                trellis_optimize(&mut ccf48[ci], &qt, dcq, acq, &SCAN_4X8, lam);
            } else {
                let pred = dc_pred_8x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 64];
                for (ry, drow) in resid.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for (dv, &s) in drow.iter_mut().zip(srow.iter()) {
                        *dv = s - pred;
                    }
                }
                let (q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                ccf8[ci] = q;
                trellis_optimize(&mut ccf8[ci], &qt, dcq, acq, &SCAN_8X8, lam);
            }
        }

        // 4:4:4 chroma-from-luma: try predicting U/V from the reconstructed luma
        // block (scaled, mean-removed) and pick CfL over plain DC per block.
        let mut cpred444 = [[0i32; 64]; 2];
        let mut cpred420 = [[0i32; 16]; 2];
        let mut cpred422 = [[0i32; 32]; 2];
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        if !self.mono && !self.ss420 && !self.ss422 {
            // CfL luma reference must use the SAME inverse transform the decoder
            // will apply (the signaled luma tx-type), or the chroma CfL prediction
            // desyncs. Previously this was unconditionally idct, which diverged
            // whenever the luma block won with ADST or IDTX.
            let lrr_cfl = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr_cfl[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 64];
            cfl_ac_444(&luma_rec, 8, 8, &mut ac);
            let mut cfl_ccf = [[0i32; 64]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f32; 2]);
            let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f32; 2]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 64];
                for (ry, drow) in src.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                }
                // DC option distortion/rate (from the coeffs already computed)
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.cquant);
                let mut s = 0i64;
                for i in 0..64 {
                    let r = (dc + dcrr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s += (d * d) as i64;
                }
                dc_sse[ci] = s;
                dc_bits[ci] = block_rate_bits(&ccf8[ci], &SCAN_8X8);
                // CfL option
                let a = cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 64];
                let mut resid = [0i32; 64];
                for i in 0..64 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                let rr = idct_dequant_8x8(&q, &self.cquant);
                let mut s2 = 0i64;
                for i in 0..64 {
                    let r = (cpr[i] + rr[i]).clamp(0, (1 << self.bd) - 1);
                    let d = src[i] - r;
                    s2 += (d * d) as i64;
                }
                cfl_ccf[ci] = q;
                cfl_sse[ci] = s2;
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_8X8);
                cpred444[ci] = cpr;
            }
            // joint signalling cost estimate (sign symbol + 1 magnitude per non-zero plane)
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
            if cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf8[..2].copy_from_slice(&cfl_ccf[..2]);
            } else {
                for ci in 0..2 {
                    cpred444[ci] = [cpred[ci]; 64];
                }
            }
        }

        // 4:4:4 directional chroma: PAETH_PRED and SMOOTH_PRED, both mapped to
        // ADST_ADST (the decoder derives the chroma tx-type from uv_mode, so
        // signalling either selects ADST_ADST automatically). These track
        // edges/gradients that plain DC smooths away — exactly the over-smoothing
        // the chroma path suffers from. Only considered when CfL did not win, and
        // chosen on a real RD margin over DC.
        let mut chosen_uv_444 = if use_cfl { CFL_PRED } else { DC_PRED };
        let mut paeth_pred444 = [[0i32; 64]; 2];
        // NOTE: the 4:4:4 8x8 chroma path has a pre-existing reconstruction
        // divergence from the decoder at >8-bit (present in plain DC chroma,
        // independent of directional modes — luma and the 4:2:0/4:2:2 4x4/4x8
        // chroma paths are byte-exact at 10/12-bit). Directional modes propagate
        // that corrupted reconstruction, so restrict them to 8-bit here until the
        // baseline 4:4:4 high-bit-depth chroma issue is fixed. 4:2:0/4:2:2 below
        // are byte-exact at all bit depths and stay enabled.
        if !self.mono && !self.ss420 && !self.ss422 && !use_cfl {
            let maxv = (1 << self.bd) - 1;
            // DC reference cost (current `ccf8`).
            let mut dc_total = 0f32;
            let mut src_planes = [[0i32; 64]; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 64];
                for (ry, drow) in src.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_8x8(&ccf8[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..64 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += rd_cost_i64(sse, mlam, block_rate_bits(&ccf8[ci], &SCAN_8X8));
            }
            // Try each directional candidate with its mode-derived transform;
            // keep the best that also beats DC by the mode-signalling margin.
            // V/H additionally emit a chroma angle_delta symbol (only valid here
            // at 8x8 4:4:4 chroma), costed below.
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf8;
            let mut best_pred = [[0i32; 64]; 2];
            let candidates = &[
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_V_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
                D135_PRED,
                D113_PRED,
                D157_PRED,
            ];
            let directional_top =
                self.rank_chroma_directionals::<64>(candidates, px, py, px, py, 8, 8);
            for &cand in candidates {
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
                // mode symbol (~4 bits) + angle_delta symbol (~3 bits) for the
                // directional modes (V/H and the Z2 angulars D135/D113/D157).
                let sig_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    7.0f32
                } else {
                    4.0f32
                };
                let mut cand_ccf = [[0i32; 64]; 2];
                let mut cand_pred = [[0i32; 64]; 2];
                let mut cand_total = rate_cost(mlam, sig_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 64];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.w,
                        px,
                        py,
                        8,
                        8,
                        false,
                        false,
                        self.w,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 64];
                    for i in 0..64 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_8x8(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                    let rr = inv_chroma_8x8(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..64 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_8X8));
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_444 = best_mode_uv;
                ccf8[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred444[..2].copy_from_slice(&best_pred[..2]);
            }
        }

        let chroma_zero = |ci: usize| {
            if self.ss420 {
                ccf44[ci].iter().all(|&c| c == 0)
            } else if self.ss422 {
                ccf48[ci].iter().all(|&c| c == 0)
            } else {
                ccf8[ci].iter().all(|&c| c == 0)
            }
        };
        let block_skip =
            lcf.iter().all(|&c| c == 0) && (self.mono || (chroma_zero(0) && chroma_zero(1)));

        // block-level mode info: skip (ctx = above_skip + left_skip), y/uv = DC
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 1, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
            // angle_delta refined for diagonals (above); V/H stay at delta 0.
            self.enc.encode_symbol(
                (best_delta + 3) as usize,
                &mut self.cdfs.angle_delta[best_mode - V_PRED],
            );
        }
        // SMOOTH_V check for 4:2:0 4x4 chroma: only at low quality (ac_q > 300)
        let smooth_v_active_ss420 = false; // see note: chroma SMOOTH_V -> ADST_DCT not implemented; would desync decoder
        let mut sv_preds_420 = [[0i32; 16]; 2];
        let mut chosen_uv_block = DC_PRED;
        // 4:2:0 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x4). Populated by
        // the search block below (after CfL); recon uses `iadst_dequant_4x4`.
        let mut chosen_uv_420 = DC_PRED;
        let mut paeth_pred420 = [[0i32; 16]; 2];
        // 4:2:2 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x8).
        let mut chosen_uv_422 = DC_PRED;
        let mut paeth_pred422 = [[0i32; 32]; 2];
        if !self.mono && self.ss420 && smooth_v_active_ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                trellis_lambda(),
            );
            let mut sv_ccf44_2 = [[0i32; 16]; 2];
            let mut sse_cur = 0i64;
            let mut sse_sv = 0i64;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for &sr in srow[..4].iter() {
                        let d = sr - dc;
                        sse_cur += (d * d) as i64;
                    }
                }
                intra_predict_nd(
                    SMOOTH_V_PRED,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    4,
                    4,
                    false,
                    false,
                    self.cw,
                    self.h,
                    self.chroma_filter_type(px, py),
                    &mut sv_preds_420[ci],
                    self.bd,
                );
                let mut resid = [0i32; 16];
                for (ry, drow) in resid.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for (dv, (&s, &p)) in drow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                        *dv = s - p;
                    }
                }
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                sv_ccf44_2[ci] = q;
                trellis_optimize(&mut sv_ccf44_2[ci], &qt, dcq2, acq2, &SCAN_4X4, lam2);
                for ry in 0..4 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    let prow = &sv_preds_420[ci][ry * 4..];
                    for j in 0..4 {
                        let d = srow[j] - prow[j];
                        sse_sv += (d * d) as i64;
                    }
                }
            }
            if sse_sv < sse_cur {
                ccf44[..2].copy_from_slice(&sv_ccf44_2[..2]);
                chosen_uv_block = SMOOTH_V_PRED;
            }
        }
        // Note: SMOOTH_V for 4:4:4 8x8 (code_block small-block path) is intentionally
        // not added here — it introduces too many DC↔SV mode transitions at 8-row
        // boundaries that are visible as faint lines at quality 50-75.
        // 4:2:0 chroma-from-luma: predict the 4x4 U/V from the 2x2-subsampled
        // reconstructed luma of this 8x8 block (dav1d cfl_ac, ss_hor=ss_ver=1).
        // Competes with the current DC/SMOOTH_V choice on rate-distortion.
        if !self.mono && self.ss420 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                trellis_lambda(),
            );
            let lrr = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 16];
            cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
            let mut cfl_ccf = [[0i32; 16]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 16];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                for i in 0..16 {
                    let p = if chosen_uv_block == SMOOTH_V_PRED {
                        sv_preds_420[ci][i]
                    } else {
                        dc
                    };
                    let r = (p + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf44[ci], &SCAN_4X4);
                let a = cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 16];
                let mut resid = [0i32; 16];
                for i in 0..16 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X4, lam2);
                let rr = idct_dequant_4x4(&q, &self.cquant);
                for i in 0..16 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X4);
                cfl_ccf[ci] = q;
                cpred420[ci] = cpr;
            }
            let sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let cur_total = rd_cost_i64(cur_sse, mlam, cur_bits);
            let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf44[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        // 4:2:0 directional chroma: PAETH_PRED / SMOOTH_PRED (both -> ADST_ADST,
        // now available at 4x4). Same rationale and structure as the 4:4:4 path:
        // tracks chroma edges/gradients that plain DC over-smooths. Considered
        // only when CfL did not win; chosen on a real RD margin over DC.
        if !self.mono && self.ss420 && !use_cfl && self.cquant.ac_q() < 120 {
            let maxv = (1 << self.bd) - 1;
            let mut src_planes = [[0i32; 16]; 2];
            let mut dc_total = 0f32;
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 16];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..16 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += rd_cost_i64(sse, mlam, block_rate_bits(&ccf44[ci], &SCAN_4X4));
            }
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf44;
            let mut best_pred = [[0i32; 16]; 2];
            let candidates = &[
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_V_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
                D135_PRED,
                D113_PRED,
                D157_PRED,
            ];
            let directional_top =
                self.rank_chroma_directionals::<16>(candidates, px, py, cx, cy, 4, 4);
            for &cand in candidates {
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
                let mut cand_ccf = [[0i32; 16]; 2];
                let mut cand_pred = [[0i32; 16]; 2];
                let sig_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    7.0f32
                } else {
                    4.0f32
                };
                let mut cand_total = rate_cost(mlam, sig_bits); // non-DC uv_mode (+angle_delta for V/H)
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 16];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                        false,
                        false,
                        self.cw,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 16];
                    for i in 0..16 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_4x4(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X4, lam);
                    let rr = inv_chroma_4x4(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..16 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_4X4));
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_420 = best_mode_uv;
                ccf44[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred420[..2].copy_from_slice(&best_pred[..2]);
            }
        }
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0).
        if !self.mono && self.ss422 {
            let (dcq2, acq2, lam2) = (
                self.cquant.dc_q() as f32,
                self.cquant.ac_q() as f32,
                trellis_lambda(),
            );
            let lrr = if best_is_idtx {
                iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x8(&lcf, &self.quant)
            };
            let mut luma_rec = [0i32; 64];
            for i in 0..64 {
                luma_rec[i] = (lpred_arr[i] + lrr[i]).clamp(0, (1 << self.bd) - 1);
            }
            let mut ac = [0i32; 32];
            cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 32]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
            let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
            let maxv = (1 << self.bd) - 1;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0i32; 32];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                }
                let curr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                for i in 0..32 {
                    let r = (dc + curr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cur_sse += (d * d) as i64;
                }
                cur_bits += block_rate_bits(&ccf48[ci], &SCAN_4X8);
                let a = cfl_best_alpha(&ac, &src, dc, 32, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 32];
                let mut resid = [0i32; 32];
                for i in 0..32 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    resid[i] = src[i] - cpr[i];
                }
                let (mut q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X8, lam2);
                let rr = idct_dequant_4x8(&q, &self.cquant);
                for i in 0..32 {
                    let r = (cpr[i] + rr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    cfl_sse += (d * d) as i64;
                }
                cfl_bits += block_rate_bits(&q, &SCAN_4X8);
                cfl_ccf[ci] = q;
                cpred422[ci] = cpr;
            }
            let sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let cur_total = rd_cost_i64(cur_sse, mlam, cur_bits);
            let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
            if cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                ccf48[..2].copy_from_slice(&cfl_ccf[..2]);
            }
        }
        // 4:2:2 directional chroma: PAETH_PRED / SMOOTH_PRED (-> ADST_ADST 4x8).
        // Same rationale/structure as the 4:2:0 path; block is 4 wide x 8 tall at
        // chroma coords (cx, py). Gated to higher quality (chroma ac_q < 120) and
        // only when CfL did not win; chosen on a real RD margin over DC.
        if !self.mono && self.ss422 && !use_cfl && self.cquant.ac_q() < 120 {
            let maxv = (1 << self.bd) - 1;
            let mut src_planes = [[0i32; 32]; 2];
            let mut dc_total = 0f32;
            for ci in 0..2 {
                let plane = ci + 1;
                let mut src = [0i32; 32];
                for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                }
                src_planes[ci] = src;
                let dcrr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                let mut sse = 0i64;
                for i in 0..32 {
                    let r = (cpred[ci] + dcrr[i]).clamp(0, maxv);
                    let d = src[i] - r;
                    sse += (d * d) as i64;
                }
                dc_total += rd_cost_i64(sse, mlam, block_rate_bits(&ccf48[ci], &SCAN_4X8));
            }
            let mut best_total = dc_total;
            let mut best_mode_uv = DC_PRED;
            let mut best_ccf = ccf48;
            let mut best_pred = [[0i32; 32]; 2];
            let candidates = &[
                PAETH_PRED,
                SMOOTH_PRED,
                SMOOTH_V_PRED,
                SMOOTH_H_PRED,
                V_PRED,
                H_PRED,
                D135_PRED,
                D113_PRED,
                D157_PRED,
            ];
            let directional_top =
                self.rank_chroma_directionals::<32>(candidates, px, py, cx, py, 4, 8);
            for &cand in candidates {
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
                let mut cand_ccf = [[0i32; 32]; 2];
                let mut cand_pred = [[0i32; 32]; 2];
                let sig_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                    7.0f32
                } else {
                    4.0f32
                };
                let mut cand_total = rate_cost(mlam, sig_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut pp = [0i32; 32];
                    intra_predict_nd(
                        cand,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        py,
                        4,
                        8,
                        false,
                        false,
                        self.cw,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut pp,
                        self.bd,
                    );
                    let mut resid = [0i32; 32];
                    for i in 0..32 {
                        resid[i] = src_planes[ci][i] - pp[i];
                    }
                    let (mut q, qt) = fwd_chroma_4x8(tx, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X8, lam);
                    let rr = inv_chroma_4x8(tx, &q, &self.cquant);
                    let mut sse = 0i64;
                    for i in 0..32 {
                        let r = (pp[i] + rr[i]).clamp(0, maxv);
                        let d = src_planes[ci][i] - r;
                        sse += (d * d) as i64;
                    }
                    cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_4X8));
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode_uv = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode_uv != DC_PRED {
                chosen_uv_422 = best_mode_uv;
                ccf48[..2].copy_from_slice(&best_ccf[..2]);
                paeth_pred422[..2].copy_from_slice(&best_pred[..2]);
            }
        }
        if !self.mono {
            // 4:4:4 uses the directional (PAETH) decision; 4:2:0/4:2:2 use their
            // own block-mode choice. CfL overrides via the alpha argument.
            let uv_mode_sym = if !self.ss420 && !self.ss422 {
                chosen_uv_444
            } else if self.ss420 {
                // 4:2:0: directional (PAETH/SMOOTH via ADST4) overrides DC; the
                // legacy SMOOTH_V path stays gated off (chosen_uv_block == DC).
                if chosen_uv_420 != DC_PRED {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                }
            } else if self.ss422 {
                if chosen_uv_422 != DC_PRED {
                    chosen_uv_422
                } else {
                    chosen_uv_block
                }
            } else {
                chosen_uv_block
            };
            self.emit_uv_mode(
                best_mode,
                uv_mode_sym,
                if use_cfl { Some(cfl_alpha_uv) } else { None },
                px,
                py,
                8,
                8,
            );
        }
        self.emit_filter_intra(best_mode, 8, 8, best_filter_intra);
        let sv = block_skip as u8;
        self.a_skip[bx4] = sv;
        self.a_skip[bx4 + 1] = sv;
        self.l_skip[by4] = sv;
        self.l_skip[by4 + 1] = sv;
        let mv = best_mode as u8;
        self.a_mode[bx4] = mv;
        self.a_mode[bx4 + 1] = mv;
        self.l_mode[by4] = mv;
        self.l_mode[by4 + 1] = mv;

        // luma (TX_8X8)
        let lres_ctx = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx(0, bx4, by4, false);
            let ds = self.dc_sign_ctx(0, bx4, by4);
            encode_tx8_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                &lcf,
                false,
                sk,
                ds,
                filter_intra_tx_mode(best_filter_intra, best_mode),
                if best_is_idtx {
                    0
                } else if best_is_adst {
                    ADST_ADST_TX8_IDX
                } else if best_is_adstdct {
                    ADST_DCT_TX8_IDX
                } else if best_is_dctadst {
                    DCT_ADST_TX8_IDX
                } else {
                    1
                },
            )
        };
        self.a_coef[0][bx4] = lres_ctx;
        self.a_coef[0][bx4 + 1] = lres_ctx;
        self.l_coef[0][by4] = lres_ctx;
        self.l_coef[0][by4 + 1] = lres_ctx;
        let lrr = if block_skip {
            [0i32; 64]
        } else if best_is_idtx {
            iidentity_dequant_8x8(&lcf, &self.quant)
        } else if best_is_adst {
            iadst_dequant_8x8(&lcf, &self.quant)
        } else if best_is_adstdct {
            iadstdct_dequant_8x8(&lcf, &self.quant)
        } else if best_is_dctadst {
            idctadst_dequant_8x8(&lcf, &self.quant)
        } else {
            idct_dequant_8x8(&lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred_arr
            .as_chunks::<8>()
            .0
            .iter()
            .zip(lrr.as_chunks::<8>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            for ((dv, &p), &rv) in drow.iter_mut().zip(prow.iter()).zip(rrow.iter()) {
                *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
            }
        }

        // chroma U, V
        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let (bx4c, by4c) = (cx / 4, cy / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_420(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                    encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf44[ci], sk, ds)
                };
                // TX_4X4: 1 coef-context unit wide and tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                let paeth420 = chosen_uv_420 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 16]
                } else if paeth420 {
                    inv_chroma_4x4(chroma_tx_for_mode(chosen_uv_420), &ccf44[ci], &self.cquant)
                } else {
                    idct_dequant_4x4(&ccf44[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred420[ci][ry * 4..];
                        for ((dv, &rv), &p) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if chosen_uv_block == SMOOTH_V_PRED {
                        let prow = &sv_preds_420[ci][ry * 4..];
                        for ((dv, &rv), &prow) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (prow + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth420 {
                        let prow = &paeth_pred420[ci][ry * 4..];
                        for ((dv, &rv), &p) in
                            drow[..4].iter_mut().zip(rrow.iter()).zip(prow.iter())
                        {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else if self.ss422 {
                let (bx4c, by4c) = (cx / 4, py / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_422(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_422(plane, bx4c, by4c);
                    encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf48[ci], sk, ds)
                };
                // RTX_4X8: 1 coef-context unit wide, 2 units tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                self.l_coef[plane][by4c + 1] = res_ctx;
                let paeth422 = chosen_uv_422 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 32]
                } else if paeth422 {
                    inv_chroma_4x8(chroma_tx_for_mode(chosen_uv_422), &ccf48[ci], &self.cquant)
                } else {
                    idct_dequant_4x8(&ccf48[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    if use_cfl {
                        let prow = &cpred422[ci][ry * 4..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth422 {
                        let prow = &paeth_pred422[ci][ry * 4..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            } else {
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx(plane, bx4, by4, true);
                    let ds = self.dc_sign_ctx(plane, bx4, by4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf8[ci],
                        true,
                        sk,
                        ds,
                        0,
                        1,
                    )
                };
                self.a_coef[plane][bx4] = res_ctx;
                self.a_coef[plane][bx4 + 1] = res_ctx;
                self.l_coef[plane][by4] = res_ctx;
                self.l_coef[plane][by4 + 1] = res_ctx;
                let paeth = chosen_uv_444 != DC_PRED && chosen_uv_444 != CFL_PRED;
                let rr = if block_skip {
                    [0i32; 64]
                } else if paeth {
                    // Directional chroma: tx derived from uv_mode (Mode_To_Txfm).
                    inv_chroma_8x8(chroma_tx_for_mode(chosen_uv_444), &ccf8[ci], &self.cquant)
                } else {
                    idct_dequant_8x8(&ccf8[ci], &self.cquant)
                };
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        let prow = &cpred444[ci][ry * 8..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else if paeth {
                        let prow = &paeth_pred444[ci][ry * 8..];
                        for ((dv, &rv), &p) in drow.iter_mut().zip(rrow.iter()).zip(prow.iter()) {
                            *dv = (p + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    } else {
                        // Plain DC chroma: use the scalar predictor directly so the
                        // reconstruction never depends on the CfL evaluation block
                        // having populated `cpred444`.
                        for (dv, &rv) in drow.iter_mut().zip(rrow.iter()) {
                            *dv = (cpred[ci] + rv).clamp(0, (1 << self.bd) - 1);
                        }
                    }
                }
            }
        }
    }
}
