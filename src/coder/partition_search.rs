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
/// TEMPORARY A/B switch for the rectangular-leaf luma MODE search.
fn rect_leaf_mode_search_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MT_NO_RECT_MODE").is_err())
}

/// TEMPORARY A/B switch for the rectangular-leaf transform-type search.
fn rect_leaf_tx_search_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MT_NO_RECT_TX").is_err())
}

/// Filter-intra trial in the partition proxy: implemented and measured, but
/// **default OFF**. It is worth -0.08% on the tuning corpus and nothing on
/// holdout (-0.27% -> -0.25%, i.e. within noise) while costing real time, so it
/// does not meet the bar the lambda retune taught us to apply. `MT_PROXY_FI=1`
/// enables it for further evaluation on a larger corpus.
fn proxy_filter_intra_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MT_PROXY_FI").is_ok())
}

/// TEMPORARY A/B switch for the partition proxy's transform refinement.
fn proxy_tx_refine_enabled() -> bool {
    static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *E.get_or_init(|| std::env::var("MT_NO_PROXY_TX").is_err())
}

/// Mode set used by the partition-decision proxy (`rd_cost_square`).
///
/// The proxy prices both legs of every NONE-vs-SPLIT comparison, so a mode set
/// narrower than the final block's under-credits both — but not equally, since
/// the SPLIT leg gets four independent mode choices in the real encode against
/// the proxy's shared candidates. At `Speed::Slow` (the quality tier) it now
/// searches the same non-directional+directional set the final block does;
/// faster tiers keep the cheap three. Measured -0.16% BD-rate on the tuning
/// corpus and -0.07% on holdout for +10-16% encode time.
///
/// `MT_PROXY_MODES=3|seven|full` overrides for A/B.
fn proxy_modes(reduced: bool) -> &'static [usize] {
    static M: std::sync::OnceLock<(&'static [usize], &'static [usize])> =
        std::sync::OnceLock::new();
    let (fast, slow) = *M.get_or_init(|| {
        static THREE: [usize; 3] = [DC_PRED, SMOOTH_PRED, PAETH_PRED];
        // The full set costs 13 leaf encodes per proxy call. This 7-mode set
        // drops the six diagonals (the expensive half) but keeps the smooth
        // variants and V/H, which is where most of the shortfall is.
        static SEVEN: [usize; 7] = [
            DC_PRED,
            SMOOTH_PRED,
            SMOOTH_V_PRED,
            SMOOTH_H_PRED,
            PAETH_PRED,
            V_PRED,
            H_PRED,
        ];
        match std::env::var("MT_PROXY_MODES").ok().as_deref() {
            Some("3") => (&THREE, &THREE),
            Some("seven") => (&SEVEN, &SEVEN),
            Some("full") => (nd_modes(), nd_modes()),
            // Default: cheap set for the fast tiers, full set for Slow.
            _ => (&THREE, nd_modes()),
        }
    });
    if reduced { fast } else { slow }
}

impl<'a> LossyTile<'a> {
    fn rd_cost_square(
        &self,
        px: usize,
        py: usize,
        dim: usize,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) -> f32 {
        let acq = self.quant.ac_q() as f32;
        let dcq = self.quant.dc_q() as f32;
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // Mode set the partition proxy prices a leaf with.
        //
        // Historically hard-wired to DC/SMOOTH/PAETH while the FINAL block
        // searches 13 modes plus angle deltas, filter intra and a transform
        // search. That under-credits both legs of a NONE-vs-SPLIT comparison,
        // but not equally: the SPLIT leg gets four independent mode choices in
        // the real encode and only three shared candidates in the proxy, so the
        // shortfall is larger on the split side.
        let modes: &[usize] = proxy_modes(self.speed.reduced_modes());
        let mut best = f32::INFINITY;
        // Winning (mode, prediction, residual) per size, kept so the transform
        // refinement below can re-price the winner without re-predicting.
        let mut win: Option<(usize, [i32; 64], [i32; 64])> = None;
        let mut win16: Option<(usize, [i32; 256], [i32; 256])> = None;
        match dim {
            8 => {
                let scan = &SCAN_8X8;
                for &m in modes {
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
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        8,
                        8,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits =
                        self.luma_bits(&cf, scan, 8, px, py, m, 1) + self.mode_bits(px, py, m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                        win = Some((m, pred, resid));
                    }
                }
                // Transform refinement on the winner, mirroring the final
                // block's winner-only ADST/IDTX pass. Without it the proxy
                // prices every leaf as DCT-only while the real block may code it
                // far cheaper, so leaves that respond well to ADST are
                // systematically undervalued in the partition decision.
                if proxy_tx_refine_enabled()
                    && let Some((m, pred, resid)) = win
                {
                    for (txtp, fwd, inv) in [
                        (
                            ADST_ADST_TX8_IDX,
                            adst8x8_t as fn(&[i32; 64], &Quant) -> ([i32; 64], [f32; 64]),
                            iadst_dequant_8x8 as fn(&[i32; 64], &Quant) -> [i32; 64],
                        ),
                        (0, fidentity8x8_t, iidentity_dequant_8x8),
                    ] {
                        let (mut cf, tf) = fwd(&resid, &self.quant);
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = inv(&cf, &self.quant);
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.quant.ac_q() as f32,
                            |i| pred[i] + rr[i],
                        );
                        let b = self.luma_bits(&cf, scan, 8, px, py, m, txtp)
                            + self.mode_bits(px, py, m);
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
                // Filter intra, mirroring the final block (DC_PRED only,
                // max(w,h) <= 32). The proxy could not see it at all, so any
                // leaf whose real encode wins with filter intra was priced as
                // if that tool did not exist.
                if proxy_filter_intra_enabled() {
                    let fi_bits = cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(8, 8)], 1)
                        + cdf_cost(&self.dcdf().filter_intra_mode, 0)
                        + self.mode_bits(px, py, DC_PRED);
                    for fm in FILTER_INTRA_MODES {
                        let mut pred = [0i32; 64];
                        filter_intra_predict(
                            fm,
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
                            &mut resid, &pred, &self.src[0], self.w, px, py, 8, 8,
                        );
                        let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = idct_dequant_8x8(&cf, &self.quant);
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.quant.ac_q() as f32,
                            |i| pred[i] + rr[i],
                        );
                        let b = self.luma_bits(&cf, scan, 8, px, py, DC_PRED, 1) + fi_bits;
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
            }
            16 => {
                let scan = &SCAN_16X16;
                for &m in modes {
                    let mut pred = [0i32; 256];
                    if m == DC_PRED {
                        let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 256];
                    } else {
                        intra_predict_nd(
                            m,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            16,
                            16,
                            have_tr,
                            have_bl,
                            self.w,
                            self.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.bd,
                        );
                    }
                    let mut resid = [0i32; 256];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &pred,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        16,
                        16,
                    );
                    let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_16x16(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits =
                        self.luma_bits(&cf, scan, 16, px, py, m, 1) + self.mode_bits(px, py, m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                        win16 = Some((m, pred, resid));
                    }
                }
                if proxy_tx_refine_enabled()
                    && let Some((m, pred, resid)) = win16
                {
                    let (mut cf, tf) = adst16x16_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = iadst_dequant_16x16(&cf, &self.quant);
                    let d = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let b = self.luma_bits(&cf, scan, 16, px, py, m, ADST_ADST_TX16_IDX)
                        + self.mode_bits(px, py, m);
                    best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                }
                // Filter intra, mirroring the final block (DC_PRED only,
                // max(w,h) <= 32). The proxy could not see it at all, so any
                // leaf whose real encode wins with filter intra was priced as
                // if that tool did not exist.
                if proxy_filter_intra_enabled() {
                    let fi_bits = cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(16, 16)], 1)
                        + cdf_cost(&self.dcdf().filter_intra_mode, 0)
                        + self.mode_bits(px, py, DC_PRED);
                    for fm in FILTER_INTRA_MODES {
                        let mut pred = [0i32; 256];
                        filter_intra_predict(
                            fm,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            16,
                            16,
                            &mut pred,
                            self.bd,
                        );
                        let mut resid = [0i32; 256];
                        crate::rd_sse::residual_pred(
                            &mut resid, &pred, &self.src[0], self.w, px, py, 16, 16,
                        );
                        let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = idct_dequant_16x16(&cf, &self.quant);
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            16,
                            16,
                            self.quant.ac_q() as f32,
                            |i| pred[i] + rr[i],
                        );
                        let b = self.luma_bits(&cf, scan, 16, px, py, DC_PRED, 1) + fi_bits;
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
            }
            32 => {
                let scan = &SCAN_32X32;
                for &m in modes {
                    let mut pred = [0i32; 1024];
                    if m == DC_PRED {
                        let d = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 1024];
                    } else {
                        intra_predict_nd(
                            m,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            32,
                            32,
                            have_tr,
                            have_bl,
                            self.w,
                            self.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.bd,
                        );
                    }
                    let mut resid = [0i32; 1024];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &pred,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_32x32(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        32,
                        32,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits =
                        self.luma_bits(&cf, scan, 32, px, py, m, 0) + self.mode_bits(px, py, m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                    }
                }
            }
            _ => unreachable!("rd_cost_square dim {}", dim),
        }
        best
    }

    fn rd_cost_rect16_leaf_with_dc(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        prdo: f32,
        dc: i32,
    ) -> f32 {
        let acq = self.quant.ac_q() as f32;
        let dcq = self.quant.dc_q() as f32;
        let lam = trellis_lambda();
        let (lam, mlam) = (lam * prdo, self.mlam() * prdo);
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let mut resid = [0i32; 128];
        crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, px, py, w, h, dc);
        let (mut cf, tf) = if vert {
            dct8x16_t(&resid, &self.quant)
        } else {
            let (cf, tf) = dct16x8_t(&resid, &self.quant);
            (cf, tf)
        };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
        let rr = if vert {
            idct_dequant_8x16(&cf, &self.quant)
        } else {
            idct_dequant_16x8(&cf, &self.quant)
        };
        let distortion =
            self.luma_partition_distortion(px, py, w, h, self.quant.ac_q() as f32, |i| dc + rr[i]);
        crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, scan))
    }

    fn rd_cost_rect16_leaf(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let dc = if vert {
            dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc)
    }

    /// Reprice a rectangular leaf whose top or left edge belongs to an earlier
    /// sibling in the same asymmetric partition. The normal RDO view still has
    /// stale pixels there; source samples are a close proxy for the sibling's
    /// eventual reconstruction and, by taking the worse cost, prevent a stale
    /// zero edge from making the partition look artificially cheap.
    fn rd_cost_rect16_dependent(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        source_above: bool,
        source_left: bool,
        prdo: f32,
    ) -> f32 {
        let base = self.rd_cost_rect16_leaf(px, py, vert, prdo);
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let above = py > 0;
        let left = px > 0;
        let mut sum = 0i32;
        let mut count = 0i32;
        if above {
            let plane = if source_above {
                &self.src[0]
            } else {
                &self.recon[0]
            };
            sum += plane[(py - 1) * self.w + px..][..w].iter().sum::<i32>();
            count += w as i32;
        }
        if left {
            let plane = if source_left {
                &self.src[0]
            } else {
                &self.recon[0]
            };
            sum += plane[py * self.w + px - 1..]
                .iter()
                .step_by(self.w)
                .take(h)
                .sum::<i32>();
            count += h as i32;
        }
        let dc = if count == 0 {
            1 << (self.bd - 1)
        } else {
            (sum + count / 2) / count
        };
        base.max(self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc))
    }

    /// Luma intra modes a 16x8 / 8x16 rect leaf may search. Restricted to modes
    /// reading only the above row, left column and corner — the six diagonals
    /// additionally need top-right / bottom-left availability, which this path
    /// does not track, and a wrong edge flag desyncs the decoder.
    const RECT_LEAF_MODES: [usize; 7] = [
        DC_PRED,
        V_PRED,
        H_PRED,
        SMOOTH_PRED,
        SMOOTH_V_PRED,
        SMOOTH_H_PRED,
        PAETH_PRED,
    ];

    /// Mode search for a rectangular luma leaf. Returns the winning mode, its
    /// prediction plane, the residual against it and its post-trellis DCT
    /// coefficients. `dc` is the caller's DC value, used verbatim for the DC
    /// candidate so a DC win reproduces the previous output exactly.
    #[allow(clippy::too_many_arguments)]
    fn rect16_luma_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        dc: i32,
    ) -> (usize, [i32; 128], [i32; 128], [i32; 128]) {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (lam, mlam) = (trellis_lambda(), self.mlam());
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let kf = &self.dcdf().kf_y[yctx];
        let modes: &[usize] = if !rect_leaf_mode_search_enabled() {
            &[DC_PRED]
        } else if self.speed.reduced_modes() {
            &[DC_PRED, SMOOTH_PRED, PAETH_PRED]
        } else {
            &Self::RECT_LEAF_MODES
        };

        let mut best = (f32::INFINITY, DC_PRED, [0i32; 128], [0i32; 128], [0i32; 128]);
        for &m in modes {
            let mut pred = [0i32; 128];
            if m == DC_PRED {
                pred = [dc; 128];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    false,
                    false,
                    self.w,
                    self.h,
                    ftype,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 128];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, w, h);
            let (mut cf, tf) = if vert {
                dct8x16_t(&resid, &self.quant)
            } else {
                dct16x8_t(&resid, &self.quant)
            };
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            let mean = resid.iter().sum::<i32>() / 128;
            if cf[0] == 0 && mean.abs() >= 8 {
                cf[0] = if mean > 0 { 1 } else { -1 };
            }
            let rr = if vert {
                idct_dequant_8x16(&cf, &self.quant)
            } else {
                idct_dequant_16x8(&cf, &self.quant)
            };
            let sse =
                crate::rd_sse::sse_recon(&pred, &rr, &self.src[0], self.w, px, py, w, h, self.bd);
            let bits = block_rate_bits(&cf, scan) + cdf_cost(kf, m);
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best.0 {
                best = (cost, m, pred, resid, cf);
            }
        }
        (best.1, best.2, best.3, best.4)
    }

    /// Transform-type trial for a rectangular luma leaf: ADST_ADST against the
    /// committed DCT_DCT candidate.
    ///
    /// Rect leaves have never had a transform-type search — until now ADST did
    /// not even exist for RTX_8X16 / RTX_16X8. This is the first of the tools a
    /// PARTITION_NONE block has and a rect leaf does not (the others are filter
    /// intra, angle delta and TX split), which is why the 16x16 partition R-D
    /// systematically overrates rectangles. Returns the winning txtp symbol
    /// (1 = DCT_DCT, `ADST_ADST_TX8_IDX` = ADST_ADST) and its coefficients.
    fn rect_leaf_tx_trial(
        &self,
        resid: &[i32; 128],
        dct_cf: &[i32; 128],
        pred: &[i32; 128],
        px: usize,
        py: usize,
        vert: bool,
        y_mode: usize,
    ) -> (usize, [i32; 128]) {
        if !rect_leaf_tx_search_enabled() {
            return (1, *dct_cf);
        }
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (lam, mlam) = (trellis_lambda(), self.mlam());
        let txtp_cdf = &self.dcdf().txtp[y_mode];

        let cost_of = |cf: &[i32; 128], rr: &[i32; 128], txtp: usize| -> f32 {
            let sse =
                crate::rd_sse::sse_recon(pred, rr, &self.src[0], self.w, px, py, w, h, self.bd);
            let bits = block_rate_bits(cf, scan) + cdf_cost(txtp_cdf, txtp);
            rd_cost_i64(sse, mlam, bits)
        };

        let dct_rr = if vert {
            idct_dequant_8x16(dct_cf, &self.quant)
        } else {
            idct_dequant_16x8(dct_cf, &self.quant)
        };
        let dct_cost = cost_of(dct_cf, &dct_rr, 1);

        let (mut acf, atf) = if vert {
            adst8x16_t(resid, &self.quant)
        } else {
            adst16x8_t(resid, &self.quant)
        };
        trellis_optimize(&mut acf, &atf, dcq, acq, scan, lam);
        // Same DC-preservation snap the DCT candidate gets.
        let mean = resid.iter().sum::<i32>() / 128;
        if acf[0] == 0 && mean.abs() >= 8 {
            acf[0] = if mean > 0 { 1 } else { -1 };
        }
        let adst_rr = if vert {
            iadst_dequant_8x16(&acf, &self.quant)
        } else {
            iadst_dequant_16x8(&acf, &self.quant)
        };
        if cost_of(&acf, &adst_rr, ADST_ADST_TX8_IDX) < dct_cost {
            (ADST_ADST_TX8_IDX, acf)
        } else {
            (1, *dct_cf)
        }
    }

    fn rd_cost_horz(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, SPLIT_SIGNAL_BITS)
            + self.rd_cost_rect16_leaf(px, py, false, prdo)
            + self.rd_cost_rect16_leaf(px, py + 8, false, prdo)
    }

    fn rd_cost_vert(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, SPLIT_SIGNAL_BITS)
            + self.rd_cost_rect16_leaf(px, py, true, prdo)
            + self.rd_cost_rect16_leaf(px + 8, py, true, prdo)
    }

    /// Code a 16x16 luma region as PARTITION_H: two stacked 16x8 sub-blocks.
    /// 4:4:4 only, DC luma prediction + DCT_DCT, with a CfL chroma trial per
    /// sub-block (RD-compared against plain DC). A non-directional LUMA mode
    /// search was implemented and measured at exactly 0.00% BD-rate — the flat
    /// `mode_signal_bits` DC bias makes extra luma modes a wash here — so the
    /// luma side stays DC. No SMOOTH_V (needs the derived ADST_DCT chroma tx). Each 16x8 sub-block is a full intra block (own skip,
    /// y_mode, uv_mode, luma RTX_16X8 coeffs + chroma RTX_16X8 coeffs). Mirrors the
    /// decoder's two `decode_b(PARTITION_H)` calls.
    /// 4:4:4 VERT: two side-by-side 8x16 blocks (luma + chroma), DC intra.
    fn code_block16_vert_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8 + half * 8, y8 * 8);
            let (bx4, by4) = (px / 4, py / 4);
            let dc_l = dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let (y_mode, lpred_arr, lresid, lcf) =
                self.rect16_luma_mode_search(px, py, true, dc_l);
            let (ltxtp, lcf) =
                self.rect_leaf_tx_trial(&lresid, &lcf, &lpred_arr, px, py, true, y_mode);
            let inv8x16 = |cf: &[i32; 128], q: &Quant| {
                if ltxtp == 1 {
                    idct_dequant_8x16(cf, q)
                } else {
                    iadst_dequant_8x16(cf, q)
                }
            };
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_8x16(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.w, px, py, 8, 16, dc);
                let (mut q, qt) = dct8x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_8X16, lam);
                let mean_c = resid.iter().sum::<i32>() / 128;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            // --- CfL: predict chroma from this sub-block's reconstructed luma.
            // Allowed here (8x16 <= 32x32); at 4:4:4 chroma is full resolution
            // so chroma-from-luma is worth far more than a luma mode search.
            let mlam = self.mlam();
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_pred = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
            {
                let lrr_cfl = inv8x16(&lcf, &self.quant);
                let mut luma_rec = [0i32; 128];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
                let mut ac = [0i32; 128];
                cfl_ac_444(&luma_rec, 8, 16, &mut ac);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut csrc = [0i32; 128];
                    for ry in 0..16 {
                        csrc[ry * 8..ry * 8 + 8]
                            .copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..8]);
                    }
                    let dcrr = idct_dequant_8x16(&ccf[ci], &self.cquant);
                    let s = crate::rd_sse::sse_recon(
                        &[dc; 128], &dcrr, &csrc, 8, 0, 0, 8, 16, self.bd,
                    );
                    dc_cost[ci] = rd_cost_i64(s, mlam, block_rate_bits(&ccf[ci], &SCAN_8X16));
                    let a = cfl_best_alpha(&ac, &csrc, dc, 128, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 128];
                    for k in 0..128 {
                        cpr[k] = cfl_pred_pixel(dc, ac[k], a, self.bd);
                    }
                    let mut resid = [0i32; 128];
                    crate::rd_sse::residual_pred(&mut resid, &cpr, &csrc, 8, 0, 0, 8, 16);
                    let (mut q, qt) = dct8x16_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_8X16, lam);
                    let rr2 = idct_dequant_8x16(&q, &self.cquant);
                    let s2 = crate::rd_sse::sse_recon(
                        &cpr, &rr2, &csrc, 8, 0, 0, 8, 16, self.bd,
                    );
                    cfl_ccf[ci] = q;
                    cfl_pred[ci] = cpr;
                    cfl_cost[ci] = rd_cost_i64(s2, mlam, block_rate_bits(&q, &SCAN_8X16));
                }
            }
            // CFL_PRED uv_mode symbol + one alpha symbol per non-zero plane.
            let cfl_sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                    < dc_cost[0] + dc_cost[1];
            if use_cfl {
                ccf = cfl_ccf;
            }
            let cfl_opt = if use_cfl { Some(cfl_a) } else { None };
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(x8 + half, y8, 2, 4);
            self.mark_skip8_rect(x8 + half, y8, 1, 2, block_skip);
            self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
            // Directional modes carry an angle_delta symbol (`use_angle_delta` is
            // true for BLOCK_8X8 and larger); omitting it yields an invalid
            // bitstream. This search only offers delta 0.
            if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
                self.enc
                    .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
            }
            self.emit_uv_mode(y_mode, DC_PRED, cfl_opt, px, py, 8, 16);
            self.emit_palette_mode_info(px, py, 8, 16, y_mode, !self.mono, None);
            self.emit_filter_intra(y_mode, 8, 16, None);
            self.code_tx_depth(px, py, 8, 16, 0);
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 2].fill(sv);
            self.l_skip[by4..by4 + 4].fill(sv);
            self.a_mode[bx4..bx4 + 2].fill(y_mode as u8);
            self.l_mode[by4..by4 + 4].fill(y_mode as u8);
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_luma();
                let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
                encode_8x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, y_mode, ltxtp)
            };
            self.a_coef[0][bx4..bx4 + 2].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv8x16(&lcf, &self.quant)
            };
            for ry in 0..16 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_pred(&mut drow[..8], &lpred_arr[ry * 8..], &lrr[ry * 8..], maxval);
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_8x16_chroma(plane, bx4, by4);
                    let ds = self.dc_sign_ctx_8x16_chroma(plane, bx4, by4);
                    encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][bx4..bx4 + 2].fill(cres_ctx);
                self.l_coef[plane][by4..by4 + 4].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 128]
                } else {
                    idct_dequant_8x16(&ccf[ci], &self.cquant)
                };
                for ry in 0..16 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        recon_add_pred(
                            &mut drow[..8], &cfl_pred[ci][ry * 8..], &rr[ry * 8..], maxval,
                        );
                    } else {
                        recon_add_dc(&mut drow[..8], cpred[ci], &rr[ry * 8..], maxval);
                    }
                }
            }
        }
    }

    /// 4:2:0 rect: HORZ = two 16x8 luma + 8x4 chroma; VERT = two 8x16 + 4x8.
    fn code_block16_rect_420(&mut self, x8: usize, y8: usize, vert: bool) {
        for half in 0..2 {
            let (sx8, sy8) = if vert {
                (x8 + half, y8)
            } else {
                (x8, y8 + half)
            };
            self.code_block16_rect_leaf_420(sx8, sy8, vert);
        }
    }

    /// Emit one 16x8/8x16 leaf. Used by binary and asymmetric partitions.
    fn code_block16_rect_leaf_420(&mut self, x8: usize, y8: usize, vert: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let lpred = if vert {
            dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        let mut lresid = [0i32; 128];
        crate::rd_sse::residual_dc(&mut lresid, &self.src[0], self.w, px, py, lw, lh, lpred);
        let (mut lcf, ltf) = if vert {
            dct8x16_t(&lresid, &self.quant)
        } else {
            dct16x8_t(&lresid, &self.quant)
        };
        let lscan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
        let mean_l = lresid[..lw * lh].iter().sum::<i32>() / (lw * lh) as i32;
        if lcf[0] == 0 && mean_l.abs() >= 8 {
            lcf[0] = if mean_l > 0 { 1 } else { -1 };
        }
        // chroma 8x4 (horz) or 4x8 (vert) at chroma coords.
        let (cx, cy) = (px / 2, py / 2);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let (cw, ch) = if vert { (4usize, 8usize) } else { (8, 4) };
        let mut ccf = [[0i32; 32]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = if vert {
                dc_pred_4x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            } else {
                dc_pred_8x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            };
            cpred[ci] = dc;
            let mut resid = [0i32; 32];
            crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, cw, ch, dc);
            let (mut q, qt) = if vert {
                dct4x8_t(&resid, &self.cquant)
            } else {
                dct8x4_t(&resid, &self.cquant)
            };
            let cscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
            let mean_c = resid[..cw * ch].iter().sum::<i32>() / (cw * ch) as i32;
            if q[0] == 0 && mean_c.abs() >= 8 {
                q[0] = if mean_c > 0 { 1 } else { -1 };
            }
            ccf[ci] = q;
        }
        let luma_zero = lcf.iter().all(|&v| v == 0);
        let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
        let block_skip = luma_zero && chroma_zero;
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        if vert {
            self.record_blk_rect(x8, y8, 2, 4);
            self.mark_skip8_rect(x8, y8, 1, 2, block_skip);
        } else {
            self.record_blk_rect(x8, y8, 4, 2);
            self.mark_skip8_rect(x8, y8, 2, 1, block_skip);
        }
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
        self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, lw, lh);
        self.emit_palette_mode_info(px, py, lw, lh, DC_PRED, !self.mono, None);
        self.emit_filter_intra(DC_PRED, lw, lh, None);
        self.code_tx_depth(px, py, lw, lh, 0);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, lh / 4);
        self.a_skip[bx4..bx4 + aw].fill(sv);
        self.l_skip[by4..by4 + ah].fill(sv);
        self.a_mode[bx4..bx4 + aw].fill(DC_PRED as u8);
        self.l_mode[by4..by4 + ah].fill(DC_PRED as u8);
        let lres_ctx = if block_skip {
            0x40
        } else if vert {
            let sk = self.skip_ctx_8x16_luma();
            let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
            encode_8x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
        } else {
            let sk = self.skip_ctx_16x8_luma();
            let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
            encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
        };
        self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 128]
        } else if vert {
            idct_dequant_8x16(&lcf, &self.quant)
        } else {
            idct_dequant_16x8(&lcf, &self.quant)
        };
        for ry in 0..lh {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_dc(&mut drow[..lw], lpred, &lrr[ry * lw..], maxval);
        }
        let (caw, cah) = (cw / 4, (ch / 4).max(1));
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else if vert {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            } else {
                let sk = self.skip_ctx_8x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x4_chroma(plane, cbx4, cby4);
                encode_8x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            let fillw = caw.max(1);
            self.a_coef[plane][cbx4..cbx4 + fillw].fill(cres_ctx);
            self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 32]
            } else if vert {
                idct_dequant_4x8(&ccf[ci], &self.cquant)
            } else {
                idct_dequant_8x4(&ccf[ci], &self.cquant)
            };
            for ry in 0..ch {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
            }
        }
    }

    /// 4:2:2 HORZ: two 16x8 luma + 8x8 chroma (h-subsampled, v-full). V forbidden in 422.
    fn code_block16_horz_422(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4);
            let lpred = dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            crate::rd_sse::residual_dc(&mut lresid, &self.src[0], self.w, px, py, 16, 8, lpred);
            let (mut lcf, ltf) = dct16x8_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_16X8, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            let (cx, cy) = (px / 2, py);
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let mut ccf = [[0i32; 64]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 64];
                crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, 8, 8, dc);
                let (mut q, qt) = dct8x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_8X8, lam);
                let mean_c = resid.iter().sum::<i32>() / 64;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, 16, 8);
            self.emit_palette_mode_info(px, py, 16, 8, DC_PRED, !self.mono, None);
            self.emit_filter_intra(DC_PRED, 16, 8, None);
            self.code_tx_depth(px, py, 16, 8, 0);
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 2].fill(DC_PRED as u8);
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..16], lpred, &lrr[ry * 16..], maxval);
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_8x8_chroma(plane, cbx4, cby4);
                    let ds = self.dc_sign_ctx_8x8_chroma(plane, cbx4, cby4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf[ci],
                        true,
                        sk,
                        ds,
                        DC_PRED,
                        0,
                    )
                };
                self.a_coef[plane][cbx4..cbx4 + 2].fill(cres_ctx);
                self.l_coef[plane][cby4..cby4 + 2].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    idct_dequant_8x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    recon_add_dc(&mut drow[..8], cpred[ci], &rr[ry * 8..], maxval);
                }
            }
        }
    }

    fn code_block16_horz_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        // Two sub-blocks: half = 0 (top, py), half = 1 (bottom, py+8).
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4); // luma 4-unit coords
            // --- luma 16x8: DC predict, residual, forward, trellis, dc-snap ---
            let lpred = dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            crate::rd_sse::residual_dc(&mut lresid, &self.src[0], self.w, px, py, 16, 8, lpred);
            let (mut lcf, ltf) = dct16x8_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_16X8, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            // --- chroma 16x8 (4:4:4): DC predict each plane ---
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_16x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.w, px, py, 16, 8, dc);
                let (mut q, qt) = dct16x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_16X8, lam);
                let mean_c = resid.iter().sum::<i32>() / 128;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            // block_skip iff all planes have no coefficients.
            // --- CfL: predict chroma from this sub-block's reconstructed luma.
            // Allowed here (16x8 <= 32x32); at 4:4:4 chroma is full resolution
            // so chroma-from-luma is worth far more than a luma mode search.
            let mlam = self.mlam();
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_pred = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
            {
                let lrr_cfl = idct_dequant_16x8(&lcf, &self.quant);
                let mut luma_rec = [0i32; 128];
                recon_add_dc(&mut luma_rec, lpred, &lrr_cfl, maxval);
                let mut ac = [0i32; 128];
                cfl_ac_444(&luma_rec, 16, 8, &mut ac);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut csrc = [0i32; 128];
                    for ry in 0..8 {
                        csrc[ry * 16..ry * 16 + 16]
                            .copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..16]);
                    }
                    let dcrr = idct_dequant_16x8(&ccf[ci], &self.cquant);
                    let s = crate::rd_sse::sse_recon(
                        &[dc; 128], &dcrr, &csrc, 16, 0, 0, 16, 8, self.bd,
                    );
                    dc_cost[ci] = rd_cost_i64(s, mlam, block_rate_bits(&ccf[ci], &SCAN_16X8));
                    let a = cfl_best_alpha(&ac, &csrc, dc, 128, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 128];
                    for k in 0..128 {
                        cpr[k] = cfl_pred_pixel(dc, ac[k], a, self.bd);
                    }
                    let mut resid = [0i32; 128];
                    crate::rd_sse::residual_pred(&mut resid, &cpr, &csrc, 16, 0, 0, 16, 8);
                    let (mut q, qt) = dct16x8_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_16X8, lam);
                    let rr2 = idct_dequant_16x8(&q, &self.cquant);
                    let s2 = crate::rd_sse::sse_recon(
                        &cpr, &rr2, &csrc, 16, 0, 0, 16, 8, self.bd,
                    );
                    cfl_ccf[ci] = q;
                    cfl_pred[ci] = cpr;
                    cfl_cost[ci] = rd_cost_i64(s2, mlam, block_rate_bits(&q, &SCAN_16X8));
                }
            }
            // CFL_PRED uv_mode symbol + one alpha symbol per non-zero plane.
            let cfl_sig = 4.0f32
                + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
                + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
            let use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                    < dc_cost[0] + dc_cost[1];
            if use_cfl {
                ccf = cfl_ccf;
            }
            let cfl_opt = if use_cfl { Some(cfl_a) } else { None };
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            // --- header: skip, delta-q (once), y_mode (DC), uv_mode (DC) ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            // record the 16x8 footprint for the deblock filter: width 4 units,
            // height 2 units (vertical edges every 16, horizontal every 8).
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, cfl_opt, px, py, 16, 8);
            self.emit_palette_mode_info(px, py, 16, 8, DC_PRED, !self.mono, None);
            self.emit_filter_intra(DC_PRED, 16, 8, None);
            self.code_tx_depth(px, py, 16, 8, 0);
            // footprint update: skip/mode over 4 wide x 2 tall units.
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 2].fill(DC_PRED as u8);
            // --- luma coeffs (RTX_16X8) ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            // reconstruct luma
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..16], lpred, &lrr[ry * 16..], maxval);
            }
            // --- chroma coeffs + reconstruct (4:4:4, both planes RTX_16X8) ---
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_16x8_chroma(plane, bx4, by4);
                    let ds = self.dc_sign_ctx_16x8_chroma(plane, bx4, by4);
                    encode_16x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][bx4..bx4 + 4].fill(cres_ctx);
                self.l_coef[plane][by4..by4 + 2].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 128]
                } else {
                    idct_dequant_16x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        recon_add_pred(
                            &mut drow[..16], &cfl_pred[ci][ry * 16..], &rr[ry * 16..], maxval,
                        );
                    } else {
                        recon_add_dc(&mut drow[..16], cpred[ci], &rr[ry * 16..], maxval);
                    }
                }
            }
        }
    }

    /// Trial-code a 16x16 luma block as four TX_8X8 (`tx_depth = 1`), raster
    /// order, DCT only. Per the spec, intra prediction runs per TRANSFORM
    /// block: each 8x8 predicts from the running reconstruction (including the
    /// previous quadrants of this block), which is exactly what lets a smooth
    /// gradient ride the four sub-TX DCs instead of dying in one 16x16 AC
    /// (DC-granularity banding). Temporarily writes candidate recon into
    /// `self.recon[0]` for the sequential prediction and restores it before
    /// returning. Returns (packed quadrant-major coefficients, 16x16 recon
    /// row-major, summed SSE, summed proxy coefficient bits).
    ///
    /// Per-quadrant edge availability mirrors dav1d's per-TX edge flags for a
    /// 16x16 block: q(0,0) sees the block's outer edges; q(1,0)'s top-right is
    /// the block's `have_tr`; q(0,1)'s top-right is inside the block (always
    /// coded) and its bottom-left is the block's `have_bl`; q(1,1) has neither.
    #[allow(clippy::too_many_arguments)]
    fn split16_luma_try(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
    ) -> ([i32; 256], [i32; 256], i64, f32) {
        let mut saved = [0i32; 256];
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let maxv = (1i32 << self.bd) - 1;
        let mut cf4 = [0i32; 256];
        let mut rec = [0i32; 256];
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let quads = [(0usize, 0usize), (8, 0), (0, 8), (8, 8)];
        for (qi, &(sx, sy)) in quads.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (8, 0) => (have_tr, false),
                (0, 8) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = [0i32; 64];
            if mode == DC_PRED && delta == 0 {
                let d = dc_pred_8x8(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd_ad(
                    mode,
                    delta,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    8,
                    8,
                    tr,
                    bl,
                    self.w,
                    self.h,
                    block_ftype,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 64];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 8, 8);
            let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
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
                0,
            );
            let rr = idct_dequant_8x8(&cf, &self.quant);
            sse_sum += sse_recon::<64, 8>(&pred, &rr, &self.src[0], self.w, bx, by, self.bd);
            bits_sum += self.luma_bits(&cf, &SCAN_8X8, 8, bx, by, mode, 1);
            // Write the quadrant's candidate recon so later quadrants predict
            // from it (restored below).
            for ry in 0..8 {
                let rrow = &mut self.recon[0][(by + ry) * self.w + bx..];
                for rx in 0..8 {
                    let v = (pred[ry * 8 + rx] + rr[ry * 8 + rx]).clamp(0, maxv);
                    rrow[rx] = v;
                    rec[(sy + ry) * 16 + (sx + rx)] = v;
                }
            }
            cf4[qi * 64..qi * 64 + 64].copy_from_slice(&cf);
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        (cf4, rec, sse_sum, bits_sum)
    }

    /// Reconstruct a TX-split 16x16 luma block from its (packed quadrant-major)
    /// committed coefficients — the exact reconstruction the decoder performs
    /// (per-quadrant prediction from the running recon, DCT sub-TX). Used by
    /// the 4:4:4 CfL evaluation, which needs the decoder-side luma before the
    /// block is emitted. Restores `self.recon` before returning.
    #[allow(clippy::too_many_arguments)]
    fn split16_luma_recon_from_cf(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lcf: &[i32; 256],
    ) -> [i32; 256] {
        let mut saved = [0i32; 256];
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let maxv = (1i32 << self.bd) - 1;
        let block_ftype = self.luma_filter_type(px, py);
        let mut rec = [0i32; 256];
        for (qi, &(sx, sy)) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)].iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (8, 0) => (have_tr, false),
                (0, 8) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = [0i32; 64];
            if mode == DC_PRED {
                let d = dc_pred_8x8(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 64];
            } else {
                intra_predict_nd_ad(
                    mode,
                    delta,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    8,
                    8,
                    tr,
                    bl,
                    self.w,
                    self.h,
                    block_ftype,
                    &mut pred,
                    self.bd,
                );
            }
            let mut cfq = [0i32; 64];
            cfq.copy_from_slice(&lcf[qi * 64..qi * 64 + 64]);
            let rr = idct_dequant_8x8(&cfq, &self.quant);
            for ry in 0..8 {
                let rrow = &mut self.recon[0][(by + ry) * self.w + bx..];
                for rx in 0..8 {
                    let v = (pred[ry * 8 + rx] + rr[ry * 8 + rx]).clamp(0, maxv);
                    rrow[rx] = v;
                    rec[(sy + ry) * 16 + (sx + rx)] = v;
                }
            }
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        rec
    }

    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best_mode = DC_PRED;
        let mut txtp16: u8 = 0; // 0=DCT_DCT 1=ADST_ADST 2=ADST_DCT 3=DCT_ADST
        let mut lpred_arr = [0i32; 256];
        let mut lcf = [0i32; 256];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        let mut ltf = [0f32; 256]; // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Pure-emit replay: the recorded winner + its captured coefficients
        // replace every sub-search below — no candidate is evaluated at all
        // (see code_block in block8.rs for the pattern).
        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let directional_top = if rl.is_none() {
            self.rank_luma_directionals::<256>(modes, px, py, 16, 16, have_tr, have_bl)
        } else {
            DirectionalTopK::new()
        };
        for &m in modes {
            if rl.is_some() {
                break;
            }
            if is_directional_mode(m) && !directional_top.contains(m) {
                continue;
            }
            let mut pred = [0i32; 256];
            if m == DC_PRED {
                let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 256];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 16, 16);
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                sse_recon::<256, 16>(&pred, rr, &self.src[0], self.w, px, py, self.bd)
            };
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
            }
            let sse = blk_sse16(&idct_dequant_16x16(&cf, &self.quant));
            let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, m, 1);
            let filter_bits = if m == DC_PRED {
                cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(16, 16)], 0)
            } else {
                0.0
            };
            let cost = rd_cost_i64(sse, mlam, bits + self.mode_bits(px, py, m) + filter_bits);
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
        if rl.is_none() && self.speed == Speed::Slow {
            let bsize = av1_block_size_index(16, 16);
            for filter_mode in FILTER_INTRA_MODES {
                let mut pred = [0i32; 256];
                filter_intra_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
                let rr = idct_dequant_16x16(&cf, &self.quant);
                let sse =
                    sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, DC_PRED, 1);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if rl.is_some() || (filter_intra_sse_allowed(sse, best_dct_sse) && cost < best_eff)
                {
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
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
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
                let mut pred = [0i32; 256];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        self.dcdf(),
                        2,
                        0,
                        &self.dcdf().eob_bin_256_l,
                        dcs16,
                    );
                }
                let rr = idct_dequant_16x16(&cf, &self.quant);
                let sse = sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, best_mode, 1);
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
        // Fast path: run RDOQ once, on the winning mode only (libaom
        // winner-mode coeff opt). The decision above used un-trellised costs.
        if rl.is_none() && !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
            );
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style).
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_pred(
                &mut resid,
                &lpred_arr,
                &self.src[0],
                self.w,
                px,
                py,
                16,
                16,
            );
            let (mut acf, atf) = adst16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
            );
            let rr = iadst_dequant_16x16(&acf, &self.quant);
            let asse = sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let abits =
                self.luma_bits(&acf, &SCAN_16X16, 16, px, py, best_mode, ADST_ADST_TX16_IDX);
            // Quality guard: only accept ADST if it does not meaningfully worsen
            // SSE. At low quality lambda (~quantizer^2) is enormous, so a pure
            // RD test would pick ADST whenever it shaves a few bits even while
            // inflating distortion ~2x; that tanks perceptual quality (SSIMULACRA2)
            // for a trivial rate gain. Requiring SSE-non-worsening keeps the
            // genuine high-quality ADST wins (where it lowers SSE) and blocks the
            // low-quality "trade quality for bits" pathology.
            if rl.is_some()
                || (asse <= best_dct_sse + (best_dct_sse >> 5)
                    && rd_cost_i64(asse, mlam, abits)
                        < rd_cost_i64(best_dct_sse, mlam, best_dct_bits))
            {
                lcf = acf;
                txtp16 = 1;
            }
        }
        // Asymmetric-ADST refinement (ADST_DCT / DCT_ADST) for TX_16X16, same
        // rationale as the 8x8 path. Competes with the running tx winner.
        if rl.is_none() && self.speed.try_adst() && asym_adst_enabled() {
            let mut best_txtp16_sse = if txtp16 == 1 { i64::MAX } else { best_dct_sse };
            let mut best_txtp16_bits = best_dct_bits;
            if txtp16 == 1 {
                // recompute the ADST_ADST winner cost as the bar to beat
                let rr = iadst_dequant_16x16(&lcf, &self.quant);
                best_txtp16_sse =
                    sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                best_txtp16_bits =
                    self.luma_bits(&lcf, &SCAN_16X16, 16, px, py, best_mode, ADST_ADST_TX16_IDX);
            }
            for (fwd_dctadst, inv_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &lpred_arr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut acf, atf) = if fwd_dctadst {
                    dctadst16x16_t(&resid, &self.quant)
                } else {
                    adstdct16x16_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
                let rr = if inv_dctadst {
                    idctadst_dequant_16x16(&acf, &self.quant)
                } else {
                    iadstdct_dequant_16x16(&acf, &self.quant)
                };
                let asse =
                    sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                let abits = self.luma_bits(
                    &acf,
                    &SCAN_16X16,
                    16,
                    px,
                    py,
                    best_mode,
                    if inv_dctadst { DCT_ADST_TX16_IDX } else { ADST_DCT_TX16_IDX },
                );
                if rl.is_some()
                    || (asse <= best_dct_sse + (best_dct_sse >> 5)
                        && rd_cost_i64(asse, mlam, abits)
                            < rd_cost_i64(best_txtp16_sse, mlam, best_txtp16_bits))
                {
                    lcf = acf;
                    txtp16 = if inv_dctadst { 3 } else { 2 };
                    best_txtp16_sse = asse;
                    best_txtp16_bits = abits;
                }
            }
        }
        // TX split (tx_depth = 1): trial-code the winner mode as four TX_8X8
        // with per-sub-TX prediction. On smooth gradients the 16x16's low-freq
        // AC steps round to zero at the forward quantizer and the block bands;
        // the four sub-TX (whose predictions run from 8 px away and whose DCs
        // always survive) carry the ramp. Plain R-D decides — detail blocks
        // keep the single TX_16X16 when its rate saving is genuine.
        // `best_delta == 0`: our angle-delta predictor reuses the BASE angle's
        // edge filter/upsample setup (see `intra_predict_nd_ad`); at 16x16
        // (w+h=32, never upsampled) that is exact, but at 8x8 the delta-adjusted
        // angle can cross the upsample/filter-strength thresholds and diverge
        // from the decoder. Split therefore only offers itself for delta-0
        // winners (banding lives in DC/SMOOTH gradients anyway).
        if rl.is_none() && best_filter_intra.is_none() && best_delta == 0 {
            // Final (distortion, rate) of the whole-TX_16X16 winner, from its
            // committed coefficients (the sub-search locals are scope-bound).
            let rr16 = match txtp16 {
                1 => iadst_dequant_16x16(&lcf, &self.quant),
                2 => iadstdct_dequant_16x16(&lcf, &self.quant),
                3 => idctadst_dequant_16x16(&lcf, &self.quant),
                _ => idct_dequant_16x16(&lcf, &self.quant),
            };
            let none_sse =
                sse_recon::<256, 16>(&lpred_arr, &rr16, &self.src[0], self.w, px, py, self.bd);
            let none_bits = self.luma_bits(
                &lcf,
                &SCAN_16X16,
                16,
                px,
                py,
                best_mode,
                match txtp16 {
                    1 => ADST_ADST_TX16_IDX,
                    2 => ADST_DCT_TX16_IDX,
                    3 => DCT_ADST_TX16_IDX,
                    _ => 1, // DCT_DCT
                },
            );
            let (cf4, _rec, sse_s, bits_s) =
                self.split16_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam);
            // Signaling delta: tx_depth=1 instead of 0 (~1.5 bits) plus four
            // TX_8X8 txtp symbols (DCT, ~2 bits each) instead of one TX_16X16
            // txtp (~2 bits): net ~7.5 extra proxy bits.
            const SPLIT16_SIGNAL_BITS: f32 = 7.5;
            // On banding-risk regions plain SSE undervalues the split (the harm
            // is ±1-level band STRUCTURE, largely invisible to SSE): take the
            // split whenever its distortion is not meaningfully worse. On all
            // other content the plain R-D decides.
            let take = if self.banding_risk(px, py, 16) {
                sse_s <= none_sse + (none_sse >> 2)
            } else {
                rd_cost_i64(sse_s, mlam, bits_s + SPLIT16_SIGNAL_BITS)
                    < rd_cost_i64(none_sse, mlam, none_bits)
            };
            if take {
                txtp16 = 4;
                lcf = cf4;
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
            txtp16 = match r.tx {
                TxSel::Adst => 1,
                TxSel::AdstDct => 2,
                TxSel::DctAdst => 3,
                TxSel::SplitDct => 4,
                _ => 0,
            };
        }
        if let Some(cf) = rl_cf {
            lcf.copy_from_slice(&cf);
        }
        if txtp16 == 4 {
            // The in-loop deblock filter operates on TRANSFORM edges: re-record
            // the 16x16 as a grid of four 8x8 TXs so the filter masks (blk4/
            // blk4h and the edge-start flags) match the decoder's, which
            // filters the interior sub-TX boundaries too.
            self.record_tx_blk(x8, y8, 2);
            self.record_tx_blk(x8 + 1, y8, 2);
            self.record_tx_blk(x8, y8 + 1, 2);
            self.record_tx_blk(x8 + 1, y8 + 1, 2);
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            palette: 0,
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            // No IDTX sub-search at 16x16; txtp16 covers DCT/ADST/asym/split.
            tx: if txtp16 == 4 {
                TxSel::SplitDct
            } else {
                TxSel::from_flags(txtp16 == 1, false, txtp16 == 2, txtp16 == 3)
            },
        });
        self.push_luma_cf(&lcf);
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block16_420(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra, have_tr, have_bl,
            );
        } else if self.ss422 {
            self.code_block16_422(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra, have_tr, have_bl,
            );
        } else {
            self.code_block16_444(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra, have_tr, have_bl,
            );
        }
    }
}
