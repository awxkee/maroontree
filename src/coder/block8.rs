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

/// Keep large, independent mode-search phases in separate LLVM optimization
/// units. Each call is once per block, so the call boundary is negligible next
/// to the transform search it outlines.
#[inline(never)]
fn outline_block8<R>(f: impl FnOnce() -> R) -> R {
    f()
}

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
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
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
        let mut best_palette: Option<LossyLumaPalette> = None;
        let dc_sgn = self.dc_sign_ctx(0, px / 4, py / 4);
        let mut ltf = [0f32; 64]; // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Pure-emit replay: the recorded winner + its captured coefficients
        // replace every sub-search below — no candidate is evaluated at all;
        // the winner state installs just before `push_luma_sel`.
        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let directional_top = if rl.is_none() {
            self.rank_luma_directionals::<64>(modes, px, py, 8, 8, have_tr, have_bl)
        } else {
            DirectionalTopK::new()
        };
        outline_block8(|| {
            for &m in modes {
                if rl.is_some() {
                    break;
                }
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
                crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
                // Mode decision uses DCT_DCT only (cheap); the ADST_ADST transform
                // choice is refined once for the winning mode after the loop.
                let blk_sse = |rr: &[i32; 64]| -> i64 {
                    sse_recon::<64, 8>(&pred, rr, &self.src[0], self.w, px, py, self.bd)
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
                        self.dcdf(),
                        1,
                        0,
                        &self.dcdf().eob_bin_64_l,
                        dc_sgn,
                    );
                }
                let sse = blk_sse(&idct_dequant_8x8(&cf, &self.quant));
                let bits = block_rate_bits(&cf, &SCAN_8X8);
                let filter_bits = if m == DC_PRED {
                    cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(8, 8)], 0)
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
        });
        outline_block8(|| {
            if rl.is_none() {
                for n in 2..=8 {
                    let Some(palette) = lossy_luma_palette(&self.src[0], self.w, px, py, n) else {
                        continue;
                    };
                    let mut pred = [0i32; 64];
                    palette_pred(&mut pred, 8, &palette.colors, &palette.packed_map, 8, 8);
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
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                        );
                    }
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let coeff_bits = block_rate_bits(&cf, &SCAN_8X8);
                    let bits = coeff_bits
                        + mode_signal_bits(DC_PRED)
                        + palette_signal_bits(&palette, self.bd);
                    let cost = rd_cost_i64(sse, mlam, bits);
                    if cost < best_eff {
                        best_eff = cost;
                        best_mode = DC_PRED;
                        best_filter_intra = None;
                        best_palette = Some(palette);
                        lpred_arr = pred;
                        lcf = cf;
                        ltf = tf;
                        best_dct_sse = sse;
                        best_dct_bits = coeff_bits;
                    }
                }
            }
        });
        outline_block8(|| {
            if rl.is_none() && self.speed == Speed::Slow {
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
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                        );
                    }
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let bits = block_rate_bits(&cf, &SCAN_8X8);
                    let syntax_bits = mode_signal_bits(DC_PRED)
                        + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                        + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                    let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                    if rl.is_some()
                        || (filter_intra_sse_allowed(sse, best_dct_sse) && cost < best_eff)
                    {
                        best_eff = cost;
                        best_mode = DC_PRED;
                        lpred_arr = pred;
                        lcf = cf;
                        ltf = tf;
                        best_dct_sse = sse;
                        best_dct_bits = bits;
                        best_filter_intra = Some(filter_mode);
                        best_palette = None;
                    }
                }
            }
        });
        // Angle-delta winner refinement: if the winning luma mode is one of the
        // six pure diagonals, try angle_delta in -3..=3 (3 deg steps) and keep the
        // best by SSE + lambda*(coeff bits + angle_delta symbol bits). V/H and the
        // non-directional modes stay at delta 0. ~6 extra predictions per block.
        let mut best_delta: i32 = 0;
        outline_block8(|| {
            if rl.is_none()
                && angle_delta_enabled()
                && self.speed.try_angle_deltas()
                && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
                && best_mode != V_PRED
                && best_mode != H_PRED
            {
                let mut ad_cdf = [0u16; 7];
                ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
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
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                        );
                    }
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                    let bits = block_rate_bits(&cf, &SCAN_8X8);
                    let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                    if rl.is_some() || cost < best_ad_cost {
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
        });
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if rl.is_none() && !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                self.dcdf(),
                1,
                0,
                &self.dcdf().eob_bin_64_l,
                dc_sgn,
            );
        }
        let mut best_txtp_sse = best_dct_sse;
        let mut best_txtp_bits = best_dct_bits;
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 64];
            crate::rd_sse::residual_pred(
                &mut resid,
                &lpred_arr,
                &self.src[0],
                self.w,
                px,
                py,
                8,
                8,
            );
            let (mut acf, atf) = adst8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                self.dcdf(),
                1,
                0,
                &self.dcdf().eob_bin_64_l,
                dc_sgn,
            );
            let rr = iadst_dequant_8x8(&acf, &self.quant);
            let asse = sse_recon::<64, 8>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let abits = block_rate_bits(&acf, &SCAN_8X8);
            // Quality guard (see 16x16 ADST note): block low-q distortion-for-rate trades.
            if rl.is_some()
                || (asse <= best_dct_sse + (best_dct_sse >> 5)
                    && rd_cost_i64(asse, mlam, abits)
                        < rd_cost_i64(best_dct_sse, mlam, best_dct_bits))
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
        if rl.is_none() && self.speed.try_adst() && asym_adst_enabled() {
            for (fwd_t, inv_is_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 64];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &lpred_arr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                );
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
                    self.dcdf(),
                    1,
                    0,
                    &self.dcdf().eob_bin_64_l,
                    dc_sgn,
                );
                let rr = if inv_is_dctadst {
                    idctadst_dequant_8x8(&acf, &self.quant)
                } else {
                    iadstdct_dequant_8x8(&acf, &self.quant)
                };
                let asse =
                    sse_recon::<64, 8>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                let abits = block_rate_bits(&acf, &SCAN_8X8);
                if rl.is_some()
                    || (asse <= best_dct_sse + (best_dct_sse >> 5)
                        && rd_cost_i64(asse, mlam, abits)
                            < rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits))
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
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 64];
            crate::rd_sse::residual_pred(
                &mut resid,
                &lpred_arr,
                &self.src[0],
                self.w,
                px,
                py,
                8,
                8,
            );
            let (icf, _itf) = fidentity8x8_t(&resid, &self.quant);
            // No RDOQ on IDTX: because the identity transform spreads a residual
            // across many small coefficients, an aggressive trellis zeros them
            // all and the block-level bit term then picks the collapsed result.
            // Plain forward levels keep IDTX conservative (chosen only on a clear
            // real-SSE win); bit-exactness is carried by the inverse regardless.
            let rr = iidentity_dequant_8x8(&icf, &self.quant);
            let isse = sse_recon::<64, 8>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let ibits = block_rate_bits(&icf, &SCAN_8X8);
            // Quality guard (see ADST note): identity spreads residual energy and
            // is cheap to code, so at low-q lambda a pure RD test over-selects it
            // and flattens detail. Require SSE-non-worsening vs the best real tx.
            if rl.is_some()
                || (isse <= best_txtp_sse + (best_txtp_sse >> 5)
                    && rd_cost_i64(isse, mlam, ibits)
                        < rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits))
            {
                lcf = icf;
                best_is_adst = false;
                best_is_idtx = true;
                best_is_adstdct = false;
                best_is_dctadst = false;
            }
        }
        // Pure-emit replay: install the recorded winner and its captured
        // post-trellis coefficients (every luma sub-search above was skipped).
        if let Some(r) = rl {
            best_mode = r.mode as usize;
            best_delta = r.delta as i32;
            best_filter_intra = FILTER_INTRA_MODES
                .iter()
                .copied()
                .find(|&f| f as u8 == r.filter);
            best_is_adst = r.tx == TxSel::Adst;
            best_is_idtx = r.tx == TxSel::Idtx;
            best_is_adstdct = r.tx == TxSel::AdstDct;
            best_is_dctadst = r.tx == TxSel::DctAdst;
            best_palette = if r.palette == 0 {
                None
            } else {
                lossy_luma_palette(&self.src[0], self.w, px, py, r.palette as usize)
            };
        }
        if let Some(cf) = rl_cf {
            lcf.copy_from_slice(&cf);
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            palette: best_palette.as_ref().map_or(0, |p| p.colors.len() as u8),
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            tx: TxSel::from_flags(best_is_adst, best_is_idtx, best_is_adstdct, best_is_dctadst),
        });
        self.push_luma_cf(&lcf);
        // Chroma winner (popped here, pushed at the end of the chroma searches;
        // exactly one per call in every format, mono included).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let mut ccf8 = [[0i32; 64]; 2];
        let mut ccf48 = [[0i32; 32]; 2];
        let mut ccf44 = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2];
        let cy = py / 2; // chroma row for 4:2:0
        // Pure-emit replay skips this DC baseline for 4:4:4 (its `block_skip`
        // below reads the FINAL coeffs, installed from the record). 4:2:0 and
        // 4:2:2 must still run it: their `block_skip` is derived from these
        // baseline coeffs BEFORE the CfL/directional searches overwrite them,
        // so its zero-ness can differ from the captured winner's.
        let run_dc_baseline = ru.is_none() || self.ss420 || self.ss422;
        for ci in 0..(if self.mono || !run_dc_baseline { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let pred = dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 16];
                crate::rd_sse::residual_dc(
                    &mut resid,
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    4,
                    4,
                    pred,
                );
                let (q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                ccf44[ci] = q;
                trellis_optimize(&mut ccf44[ci], &qt, dcq, acq, &SCAN_4X4, lam);
            } else if self.ss422 {
                let pred = dc_pred_4x8(&self.recon[plane], self.cw, cx, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 32];
                crate::rd_sse::residual_dc(
                    &mut resid,
                    &self.src[plane],
                    self.cw,
                    cx,
                    py,
                    4,
                    8,
                    pred,
                );
                let (q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                ccf48[ci] = q;
                trellis_optimize(&mut ccf48[ci], &qt, dcq, acq, &SCAN_4X8, lam);
            } else {
                let pred = dc_pred_8x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 64];
                crate::rd_sse::residual_dc(
                    &mut resid,
                    &self.src[plane],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    pred,
                );
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
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // below (recon comes preinstalled from the record).
        outline_block8(|| {
            if !self.mono && !self.ss420 && !self.ss422 && ru.is_none() {
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
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, (1 << self.bd) - 1);
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
                    dc_sse[ci] = sse_recon::<64, 8>(&[dc; 64], &dcrr, &src, 8, 0, 0, self.bd);
                    dc_bits[ci] = block_rate_bits(&ccf8[ci], &SCAN_8X8);
                    // CfL option
                    let a = cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 64];
                    for (cpr, &ac) in cpr[..64].iter_mut().zip(ac[..64].iter()) {
                        *cpr = cfl_pred_pixel(dc, ac, a, self.bd);
                    }
                    let mut resid = [0i32; 64];
                    crate::rd_sse::residual_pred(&mut resid, &cpr, &src, 8, 0, 0, 8, 8);
                    let (mut q, qt) = forward_dct_quant_8x8_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                    let rr = idct_dequant_8x8(&q, &self.cquant);
                    cfl_ccf[ci] = q;
                    cfl_sse[ci] = sse_recon::<64, 8>(&cpr, &rr, &src, 8, 0, 0, self.bd);
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
                if ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf8[..2].copy_from_slice(&cfl_ccf[..2]);
                } else {
                    for ci in 0..2 {
                        cpred444[ci] = [cpred[ci]; 64];
                    }
                }
            }
        });
        // Pure-emit replay (4:4:4): install the captured chroma winner before
        // `chosen_uv_444` / `chroma_zero` read it. Prediction buffers stay
        // empty — recon is preinstalled from the record, never rewritten here.
        if !self.mono
            && !self.ss420
            && !self.ss422
            && let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            for (dst, src) in ccf8.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
        }

        // 4:4:4 directional chroma: PAETH_PRED and SMOOTH_PRED, both mapped to
        // ADST_ADST (the decoder derives the chroma tx-type from uv_mode, so
        // signaling either selects ADST_ADST automatically). These track
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
        outline_block8(|| {
            if !self.mono && !self.ss420 && !self.ss422 && !use_cfl && ru.is_none() {
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
                    let sse = sse_recon::<64, 8>(&[cpred[ci]; 64], &dcrr, &src, 8, 0, 0, self.bd);
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
                static CANDIDATES: [usize; 9] = [
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
                let directional_top = if ru.is_none() {
                    self.rank_chroma_directionals::<64>(&CANDIDATES, px, py, px, py, 8, 8)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in CANDIDATES.iter() {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && is_directional_mode(cand) && !directional_top.contains(cand)
                    {
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
                        crate::rd_sse::residual_pred(
                            &mut resid,
                            &pp,
                            &src_planes[ci],
                            8,
                            0,
                            0,
                            8,
                            8,
                        );
                        let (mut q, qt) = fwd_chroma_8x8(tx, &resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_8X8, lam);
                        let rr = inv_chroma_8x8(tx, &q, &self.cquant);
                        let sse = sse_recon::<64, 8>(&pp, &rr, &src_planes[ci], 8, 0, 0, self.bd);
                        cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_8X8));
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
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
        });
        // Pure-emit replay (4:4:4): a recorded directional winner sets the
        // signaled uv mode directly (coeffs were installed above).
        if !self.mono
            && !self.ss420
            && !self.ss422
            && !use_cfl
            && let Some(r) = ru
            && r.uv != DC_PRED as u8
        {
            chosen_uv_444 = r.uv as usize;
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
        // Palette color indices are carried with the transform-token payload;
        // an intra skip block has no such payload, so palette blocks must code
        // `skip_txfm = 0` even when every quantized residual is zero.
        let block_skip = best_palette.is_none()
            && lcf.iter().all(|&c| c == 0)
            && (self.mono || (chroma_zero(0) && chroma_zero(1)));
        #[cfg(test)]
        if best_palette.is_some() && lcf.iter().any(|&c| c != 0) && !self.enc.sink {
            LOSSY_PALETTE_RESIDUAL_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

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
        // SMOOTH_V chroma derives ADST_DCT (dav1d `txtp_from_uvmode`). The old
        // code predicted SMOOTH_V but transformed with plain DCT_DCT, which is
        // what desynced the decoder — not a missing transform: `fwd_chroma_4x4`
        // / `inv_chroma_4x4` already dispatch AdstDct. Using the derived tx on
        // both sides makes this path legal.
        // NOW CORRECT but measured at -0.00% BD-rate on 4:2:0, so left off: it
        // costs an extra chroma trial per block for nothing. Two real bugs were
        // fixed to get here (keep both if re-enabling):
        //   1. it predicted SMOOTH_V but transformed with plain DCT_DCT, while
        //      the decoder derives ADST_DCT (`chroma_tx_for_mode`) -> desync;
        //   2. the reconstruction ranked SMOOTH_V ahead of the directional
        //      `chosen_uv_420` mode, contradicting the emission precedence, so
        //      when both fired the stream said one predictor and the encoder
        //      used another (silent below q50, -16 SSIMU2 above q60).
        let smooth_v_active_ss420 = false;
        let mut sv_preds_420 = [[0i32; 16]; 2];
        let mut chosen_uv_block = DC_PRED;
        // 4:2:0 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x4). Populated by
        // the search block below (after CfL); recon uses `iadst_dequant_4x4`.
        let mut chosen_uv_420 = DC_PRED;
        let mut paeth_pred420 = [[0i32; 16]; 2];
        // 4:2:2 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x8).
        let mut chosen_uv_422 = DC_PRED;
        let mut paeth_pred422 = [[0i32; 32]; 2];
        outline_block8(|| {
            if !self.mono && self.ss420 && smooth_v_active_ss420 {
                let (dcq2, acq2, lam2) = (
                    self.cquant.dc_q() as f32,
                    self.cquant.ac_q() as f32,
                    trellis_lambda(),
                );
                let maxv = (1i32 << self.bd) - 1;
                let mlam_c = self.mlam_c();
                let sv_tx = chroma_tx_for_mode(SMOOTH_V_PRED);
                let mut sv_ccf44_2 = [[0i32; 16]; 2];
                // Real R-D on both legs. The old test compared raw PREDICTION error
                // (`src - pred`) and ignored rate entirely, so it took SMOOTH_V
                // whenever the predictor looked closer — even when the coded block
                // ended up bigger and worse. Score reconstructed distortion + rate.
                let mut dc_rd = 0f32;
                let mut sv_rd = 0f32;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let dcrr = idct_dequant_4x4(&ccf44[ci], &self.cquant);
                    let mut sse_dc = 0i64;
                    for ry in 0..4 {
                        let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                        for j in 0..4 {
                            let d = srow[j] - (dc + dcrr[ry * 4 + j]).clamp(0, maxv);
                            sse_dc += (d * d) as i64;
                        }
                    }
                    dc_rd += rd_cost_i64(sse_dc, mlam_c, block_rate_bits(&ccf44[ci], &SCAN_4X4));

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
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &sv_preds_420[ci],
                        &self.src[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                    );
                    let (q, qt) = fwd_chroma_4x4(sv_tx, &resid, &self.cquant);
                    sv_ccf44_2[ci] = q;
                    trellis_optimize(&mut sv_ccf44_2[ci], &qt, dcq2, acq2, &SCAN_4X4, lam2);
                    let svrr = inv_chroma_4x4(sv_tx, &sv_ccf44_2[ci], &self.cquant);
                    let mut sse_sv = 0i64;
                    for ry in 0..4 {
                        let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                        let prow = &sv_preds_420[ci][ry * 4..];
                        for j in 0..4 {
                            let d = srow[j] - (prow[j] + svrr[ry * 4 + j]).clamp(0, maxv);
                            sse_sv += (d * d) as i64;
                        }
                    }
                    sv_rd +=
                        rd_cost_i64(sse_sv, mlam_c, block_rate_bits(&sv_ccf44_2[ci], &SCAN_4X4));
                }
                // SMOOTH_V also costs a non-DC uv_mode symbol.
                if sv_rd + rate_cost(mlam_c, SMOOTH_V_UV_SIGNAL_BITS) < dc_rd {
                    ccf44[..2].copy_from_slice(&sv_ccf44_2[..2]);
                    chosen_uv_block = SMOOTH_V_PRED;
                }
            }
        });
        // Note: SMOOTH_V for 4:4:4 8x8 (code_block small-block path) is intentionally
        // not added here — it introduces too many DC↔SV mode transitions at 8-row
        // boundaries that are visible as faint lines at quality 50-75.
        // 4:2:0 chroma-from-luma: predict the 4x4 U/V from the 2x2-subsampled
        // reconstructed luma of this 8x8 block (dav1d cfl_ac, ss_hor=ss_ver=1).
        // Competes with the current DC/SMOOTH_V choice on rate-distortion.
        outline_block8(|| {
            if !self.mono && self.ss420 && ru.is_none() {
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
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr, (1 << self.bd) - 1);
                let mut ac = [0i32; 16];
                cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
                let mut cfl_ccf = [[0i32; 16]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
                let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0i32; 16];
                    for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        drow.copy_from_slice(&self.src[plane][(cy + ry) * self.cw + cx..][..4]);
                    }
                    let curr = if chosen_uv_block == SMOOTH_V_PRED {
                        inv_chroma_4x4(chroma_tx_for_mode(SMOOTH_V_PRED), &ccf44[ci], &self.cquant)
                    } else {
                        idct_dequant_4x4(&ccf44[ci], &self.cquant)
                    };
                    let cur_pred = if chosen_uv_block == SMOOTH_V_PRED {
                        sv_preds_420[ci]
                    } else {
                        [dc; 16]
                    };
                    cur_sse += sse_recon::<16, 4>(&cur_pred, &curr, &src, 4, 0, 0, self.bd);
                    cur_bits += block_rate_bits(&ccf44[ci], &SCAN_4X4);
                    let a = cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 16];
                    for i in 0..16 {
                        cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                    }
                    let mut resid = [0i32; 16];
                    crate::rd_sse::residual_pred(&mut resid, &cpr, &src, 4, 0, 0, 4, 4);
                    let (mut q, qt) = forward_dct_quant_4x4_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X4, lam2);
                    let rr = idct_dequant_4x4(&q, &self.cquant);
                    cfl_sse += sse_recon::<16, 4>(&cpr, &rr, &src, 4, 0, 0, self.bd);
                    cfl_bits += block_rate_bits(&q, &SCAN_4X4);
                    cfl_ccf[ci] = q;
                    cpred420[ci] = cpr;
                }
                let sig = 4.0f32
                    + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                    + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
                let cur_total = rd_cost_i64(cur_sse, mlam, cur_bits);
                let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
                if ru.is_some() || (cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf44[..2].copy_from_slice(&cfl_ccf[..2]);
                }
            }
        });
        // 4:2:0 directional chroma: PAETH_PRED / SMOOTH_PRED (both -> ADST_ADST,
        // now available at 4x4). Same rationale and structure as the 4:4:4 path:
        // tracks chroma edges/gradients that plain DC over-smooths. Considered
        // only when CfL did not win; chosen on a real RD margin over DC.
        outline_block8(|| {
            if !self.mono && self.ss420 && !use_cfl && self.cquant.ac_q() < 120 && ru.is_none() {
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
                    let sse = sse_recon::<16, 4>(&[cpred[ci]; 16], &dcrr, &src, 4, 0, 0, self.bd);
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
                let directional_top = if ru.is_none() {
                    self.rank_chroma_directionals::<16>(candidates, px, py, cx, cy, 4, 4)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in candidates {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && is_directional_mode(cand) && !directional_top.contains(cand)
                    {
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
                        crate::rd_sse::residual_pred(
                            &mut resid,
                            &pp,
                            &src_planes[ci],
                            4,
                            0,
                            0,
                            4,
                            4,
                        );
                        let (mut q, qt) = fwd_chroma_4x4(tx, &resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X4, lam);
                        let rr = inv_chroma_4x4(tx, &q, &self.cquant);
                        let sse = sse_recon::<16, 4>(&pp, &rr, &src_planes[ci], 4, 0, 0, self.bd);
                        cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_4X4));
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
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
        });
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0).
        outline_block8(|| {
            if !self.mono && self.ss422 && ru.is_none() {
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
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr, (1 << self.bd) - 1);
                let mut ac = [0i32; 32];
                cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
                let mut cfl_ccf = [[0i32; 32]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
                let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0i32; 32];
                    for (ry, drow) in src.as_chunks_mut::<4>().0.iter_mut().enumerate() {
                        drow.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..4]);
                    }
                    let curr = idct_dequant_4x8(&ccf48[ci], &self.cquant);
                    cur_sse +=
                        crate::rd_sse::sse_recon(&[dc; 32], &curr, &src, 4, 0, 0, 4, 8, self.bd);
                    cur_bits += block_rate_bits(&ccf48[ci], &SCAN_4X8);
                    let a = cfl_best_alpha(&ac, &src, dc, 32, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 32];
                    for (cpr, &ac) in cpr[..32].iter_mut().zip(ac[..32].iter()) {
                        *cpr = cfl_pred_pixel(dc, ac, a, self.bd);
                    }
                    let mut resid = [0i32; 32];
                    crate::rd_sse::residual_pred(&mut resid, &cpr, &src, 4, 0, 0, 4, 8);
                    let (mut q, qt) = forward_dct_quant_4x8_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X8, lam2);
                    let rr = idct_dequant_4x8(&q, &self.cquant);
                    cfl_sse += crate::rd_sse::sse_recon(&cpr, &rr, &src, 4, 0, 0, 4, 8, self.bd);
                    cfl_bits += block_rate_bits(&q, &SCAN_4X8);
                    cfl_ccf[ci] = q;
                    cpred422[ci] = cpr;
                }
                let sig = 4.0f32
                    + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                    + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
                let cur_total = rd_cost_i64(cur_sse, mlam, cur_bits);
                let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
                if ru.is_some() || (cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf48[..2].copy_from_slice(&cfl_ccf[..2]);
                }
            }
        });
        // 4:2:2 directional chroma: PAETH_PRED / SMOOTH_PRED (-> ADST_ADST 4x8).
        // Same rationale/structure as the 4:2:0 path; block is 4 wide x 8 tall at
        // chroma coords (cx, py). Gated to higher quality (chroma ac_q < 120) and
        // only when CfL did not win; chosen on a real RD margin over DC.
        outline_block8(|| {
            if !self.mono && self.ss422 && !use_cfl && self.cquant.ac_q() < 120 && ru.is_none() {
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
                    let sse = crate::rd_sse::sse_recon(
                        &[cpred[ci]; 32],
                        &dcrr,
                        &src,
                        4,
                        0,
                        0,
                        4,
                        8,
                        self.bd,
                    );
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
                let directional_top = if ru.is_none() {
                    self.rank_chroma_directionals::<32>(candidates, px, py, cx, py, 4, 8)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in candidates {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && is_directional_mode(cand) && !directional_top.contains(cand)
                    {
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
                        crate::rd_sse::residual_pred(
                            &mut resid,
                            &pp,
                            &src_planes[ci],
                            4,
                            0,
                            0,
                            4,
                            8,
                        );
                        let (mut q, qt) = fwd_chroma_4x8(tx, &resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X8, lam);
                        let rr = inv_chroma_4x8(tx, &q, &self.cquant);
                        let sse = crate::rd_sse::sse_recon(
                            &pp,
                            &rr,
                            &src_planes[ci],
                            4,
                            0,
                            0,
                            4,
                            8,
                            self.bd,
                        );
                        cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_4X8));
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
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
        });
        // Pure-emit replay (4:2:0/4:2:2): install the captured chroma winner.
        // This lands AFTER `block_skip` was derived from the DC baseline above,
        // matching the Off ordering where the searches overwrite ccf44/ccf48
        // only after the skip flag is already coded.
        if !self.mono
            && (self.ss420 || self.ss422)
            && let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            if self.ss420 {
                for (dst, src) in ccf44.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            } else {
                for (dst, src) in ccf48.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            }
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
            if !use_cfl && r.uv != DC_PRED as u8 {
                if self.ss420 {
                    chosen_uv_420 = r.uv as usize;
                } else {
                    chosen_uv_422 = r.uv as usize;
                }
            }
        }
        // Capture the final chroma winner (CfL folded in as CFL_PRED; mono
        // pushes a DC dummy so the record cursor stays aligned per call).
        // NB the dead SMOOTH_V path (`chosen_uv_block`, gated off) would need
        // its own record entry if ever re-enabled — its coeffs come from a
        // different evaluation than the directional loop's.
        {
            let uv_final = if self.mono {
                DC_PRED
            } else if use_cfl {
                CFL_PRED
            } else if !self.ss420 && !self.ss422 {
                chosen_uv_444
            } else if self.ss420 {
                if chosen_uv_420 != DC_PRED {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                }
            } else if chosen_uv_422 != DC_PRED {
                chosen_uv_422
            } else {
                chosen_uv_block
            };
            self.push_uv_sel(UvSel { uv: uv_final as u8 });
            let cfl_rec = if use_cfl { cfl_alpha_uv } else { [0, 0] };
            if self.mono {
                self.push_uv_cf(&[], &[], [0, 0]);
            } else if self.ss420 {
                self.push_uv_cf(&ccf44[0], &ccf44[1], cfl_rec);
            } else if self.ss422 {
                self.push_uv_cf(&ccf48[0], &ccf48[1], cfl_rec);
            } else {
                self.push_uv_cf(&ccf8[0], &ccf8[1], cfl_rec);
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
        self.emit_palette_mode_info(px, py, 8, 8, best_mode, !self.mono, best_palette.as_ref());
        if best_palette.is_none() {
            self.emit_filter_intra(best_mode, 8, 8, best_filter_intra);
        }
        if let Some(palette) = best_palette.as_ref() {
            self.emit_palette_map(palette);
        }
        self.code_tx_depth(px, py, 8, 8, 0);
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
        // Pure-emit replay: recon is preinstalled from the record; the writes
        // below would need the prediction we no longer compute.
        if self.sb_mode != SbMode::Replay {
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
                recon_add_pred(drow, prow, rrow, (1 << self.bd) - 1);
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
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth420 = chosen_uv_420 != DC_PRED;
                // Derive the inverse from the uv_mode that is ACTUALLY SIGNALLED,
                // in the same precedence the emitter uses. Selecting on
                // `chosen_uv_block` alone desyncs whenever CfL subsequently wins:
                // the stream then says CFL_PRED (decoder derives DCT_DCT) while
                // the encoder reconstructed with ADST_DCT. That decodes without
                // error but collapses quality, and CfL wins more often at high
                // quality — which is why this only showed above q60.
                let uv_eff = if use_cfl {
                    CFL_PRED
                } else if paeth420 {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                };
                let rr = if block_skip {
                    [0i32; 16]
                } else {
                    inv_chroma_4x4(chroma_tx_for_mode(uv_eff), &ccf44[ci], &self.cquant)
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    // Precedence MUST match `uv_final` / `uv_eff` above:
                    // CfL > directional(chosen_uv_420) > SMOOTH_V > DC. Ranking
                    // SMOOTH_V ahead of the directional mode reconstructs with a
                    // different predictor than the one signalled whenever both
                    // fire — silent at low quality, but the directional chroma
                    // search fires far more above q60, which is where this
                    // showed up as a quality collapse at unchanged size.
                    if use_cfl {
                        recon_add_pred(&mut drow[..4], &cpred420[ci][ry * 4..], rrow, max);
                    } else if paeth420 {
                        recon_add_pred(&mut drow[..4], &paeth_pred420[ci][ry * 4..], rrow, max);
                    } else if chosen_uv_block == SMOOTH_V_PRED {
                        recon_add_pred(&mut drow[..4], &sv_preds_420[ci][ry * 4..], rrow, max);
                    } else {
                        recon_add_dc(drow, cpred[ci], rrow, max);
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
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth422 = chosen_uv_422 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 32]
                } else if paeth422 {
                    inv_chroma_4x8(chroma_tx_for_mode(chosen_uv_422), &ccf48[ci], &self.cquant)
                } else {
                    idct_dequant_4x8(&ccf48[ci], &self.cquant)
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    if use_cfl {
                        recon_add_pred(drow, &cpred422[ci][ry * 4..], rrow, max);
                    } else if paeth422 {
                        recon_add_pred(drow, &paeth_pred422[ci][ry * 4..], rrow, max);
                    } else {
                        recon_add_dc(drow, cpred[ci], rrow, max);
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
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth = chosen_uv_444 != DC_PRED && chosen_uv_444 != CFL_PRED;
                let rr = if block_skip {
                    [0i32; 64]
                } else if paeth {
                    // Directional chroma: tx derived from uv_mode (Mode_To_Txfm).
                    inv_chroma_8x8(chroma_tx_for_mode(chosen_uv_444), &ccf8[ci], &self.cquant)
                } else {
                    idct_dequant_8x8(&ccf8[ci], &self.cquant)
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        recon_add_pred(drow, &cpred444[ci][ry * 8..], rrow, max);
                    } else if paeth {
                        recon_add_pred(drow, &paeth_pred444[ci][ry * 8..], rrow, max);
                    } else {
                        // Plain DC chroma: use the scalar predictor directly so the
                        // reconstruction never depends on the CfL evaluation block
                        // having populated `cpred444`.
                        recon_add_dc(drow, cpred[ci], rrow, max);
                    }
                }
            }
        }
    }
}
