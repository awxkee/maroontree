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
    true
}

/// TEMPORARY A/B switch for the rectangular-leaf transform-type search.
fn rect_leaf_tx_search_enabled() -> bool {
    true
}

/// Filter-intra trial in the partition proxy: implemented and measured, but
/// **default OFF**. It is worth -0.08% on the tuning corpus and nothing on
/// holdout (-0.27% -> -0.25%, i.e. within noise) while costing real time.
fn proxy_filter_intra_enabled() -> bool {
    false
}

/// Inverse + dequant for a rect16 luma transform by 7-type txtp symbol
/// (0 = IDTX, 1 = DCT_DCT, 2 = V_DCT, 3 = H_DCT, 4 = ADST_ADST,
/// 5 = ADST_DCT, 6 = DCT_ADST). `vert` selects RTX_8X16 vs RTX_16X8.
fn inv_rect_luma_128(
    idct: &IdctDispatch,
    cf: &[i32; 128],
    q: &Quant,
    vert: bool,
    txtp: usize,
) -> [i32; 128] {
    if vert {
        match txtp {
            0 => idct.iidtx_dequant_8x16(cf, q),
            1 => idct.idct_dequant_8x16(cf, q),
            2 => idct.ivdct_dequant_8x16(cf, q),
            3 => idct.ihdct_dequant_8x16(cf, q),
            4 => idct.iadst_dequant_8x16(cf, q),
            5 => idct.iadstdct_dequant_8x16(cf, q),
            6 => idct.idctadst_dequant_8x16(cf, q),
            _ => unreachable!("rect16 txtp {txtp}"),
        }
    } else {
        match txtp {
            0 => idct.iidtx_dequant_16x8(cf, q),
            1 => idct.idct_dequant_16x8(cf, q),
            2 => idct.ivdct_dequant_16x8(cf, q),
            3 => idct.ihdct_dequant_16x8(cf, q),
            4 => idct.iadst_dequant_16x8(cf, q),
            5 => idct.iadstdct_dequant_16x8(cf, q),
            6 => idct.idctadst_dequant_16x8(cf, q),
            _ => unreachable!("rect16 txtp {txtp}"),
        }
    }
}

/// TEMPORARY A/B switch for the partition proxy's transform refinement.
fn proxy_tx_refine_enabled() -> bool {
    true
}

/// Trellis calibration for the compact partition-proxy beam. NOT a speed
/// knob (2026-07-24 ladder): gating it off at Medium collapsed synthetic
/// content (x_screen +46% / x_fractal +31% vs Slow — un-RDOQ'd proxy prices
/// overprice screen coefficients so badly the partition trees go wrong),
/// worth -3.68% avg on the 420 tuning corpus. The time cost (~+27%) is the
/// BETTER DECISIONS' coding work, not the DP: a one-pass lite variant saved
/// only 0.05s of 0.40 and gave back 0.6%.
fn proxy_candidate_rdoq(speed: Speed) -> bool {
    !matches!(speed, Speed::Fast)
}

/// Mode set used by the partition-decision proxy (`rd_cost_square`).
/// Threshold for [`LossyTile::hog_directional_scores`]: a directional mode
/// whose score is at or below this is never predicted. libaom uses
/// `{-1.2, -1.2, -0.6, 0.4}` for its four `intra_pruning_with_hog` levels.
/// `NEG_INFINITY` prunes nothing (bit-identical to no pruning at all).
///
/// MEASURED 2026-07-24, DEFAULT OFF. All three libaom thresholds are
/// BD-rate-neutral to three decimals — in fact byte-identical output, because
/// the modes this drops are exactly the ones the SATD beam was already going
/// to reject — but every one of them is SLOWER on photographs:
/// kodak20 +1.4..+2.8%, x_oahu +3.3..+5.5%; only x_screen at 0.4 wins
/// (-5.6%), where strong directional structure prunes many modes.
///
/// The reason is structural: libaom gains from HOG because it prunes full RD
/// evaluations, whereas `proxy_mode_beam` already narrows 13 modes to 2 with
/// SATD, so HOG can only save the cheaper *ranking* stage — and recomputing
/// Sobel + bin bisection per block costs more than it saves. libaom avoids
/// that by computing per-pixel gradients ONCE per superblock
/// (`lowbd_compute_gradient_info_sb`) and having each block merely accumulate
/// them. Retry from there, not from a per-block Sobel.
const HOG_PRUNE_THRESH: f32 = f32::NEG_INFINITY;

/// Bin index for a Sobel gradient, by bisection on `(dy << 16) / dx`.
/// Direct port of libaom `get_hist_bin_idx`.
#[inline]
fn hog_bin_idx(dx: i32, dy: i32) -> usize {
    let ratio = (dy * (1 << 16)) / dx;
    let th = &HOG_BIN_THRESHOLDS;
    let (lo, hi) = if ratio <= th[7] {
        (0, 7)
    } else if ratio <= th[15] {
        (8, 15)
    } else if ratio <= th[23] {
        (16, 23)
    } else {
        (24, 31)
    };
    for (idx, &t) in th.iter().enumerate().take(hi + 1).skip(lo) {
        if ratio <= t {
            return idx;
        }
    }
    31
}

/// Margin for the 8x8 SPLIT4 breakout, the [`split_breakout_k`] analogue one
/// level down. `rd_cost_split4_luma` prices an 8x8 as four 4x4s and is the
/// single largest source of 4x4 transform+trellis work (4x4 is ~60% of all
/// trellis calls), so skipping it when the source says the split cannot pay
/// removes real work. `INFINITY` disables it.
///
/// Measured 2026-07-25 (420, Slow), BD-rate tuning / holdout, wall:
///   K=1.0  +0.11% / -0.01% (worst image +0.12%)   -6.8..-11.7%
///   K=1.5  +0.02% / -0.01%                        -4.8..-6.3%
///   K=2.0  -0.00% / -0.02%                        -2.8..-6.0%
/// Holdout is neutral at every margin, so Slow takes the middle setting and
/// the speed tiers take the aggressive one — as with `split_breakout_k`.
fn split4_breakout_k(speed: Speed) -> f32 {
    match speed {
        Speed::Slow => crate::tuning::get().split4_breakout_slow,
        Speed::Medium | Speed::Fast => 1.0,
    }
}

/// Margin for the 16x16 SPLIT breakout in [`LossyTile::rd_choice_16_inner`]:
/// skip pricing the four 8x8 children (and the A/B legs that consume them)
/// when the source-domain model puts SPLIT worse than NONE by this factor.
/// Whether the rectangle DECISION evaluates the same leaf its emitter will.
///
/// Shipped false: the decision prices a DC/DCT-only leaf while the emitter uses
/// the full mode set + 7-type transform trial, so rectangles are judged as
/// their weakest form. Deliberate for speed -- turning it on is a diagnostic.
fn rect_dec_refine() -> bool {
    crate::tuning::get().rect_decision_refine
}

fn split_breakout_k(speed: Speed) -> f32 {
    match speed {
        Speed::Slow => crate::tuning::get().split_breakout_slow,
        Speed::Medium | Speed::Fast => 1.0,
    }
}

const PROXY_RDOQ_MSE_T: f32 = f32::INFINITY;

#[inline]
fn proxy_rdoq_pays(resid: &[i32], ac_q: f32) -> bool {
    if !PROXY_RDOQ_MSE_T.is_finite() {
        return true;
    }
    let sum_sq: i64 = resid.iter().map(|&v| (v as i64) * (v as i64)).sum();
    let mse = sum_sq as f32 / resid.len() as f32;
    mse <= PROXY_RDOQ_MSE_T * ac_q * ac_q
}

fn proxy_modes(reduced: bool) -> &'static [usize] {
    if reduced { fast_nd_modes() } else { nd_modes() }
}

/// Use cheap prediction-domain ranking before the transform/trellis/rate stage.
fn proxy_mode_beam_enabled() -> bool {
    true
}

fn proxy_mode_beam_len(speed: Speed, _dim: usize) -> usize {
    match speed {
        Speed::Fast => 1,
        Speed::Medium => 2,
        Speed::Slow => 2,
    }
}

struct Luma16BeamCandidate {
    luma_cost: f32,
    mode: usize,
    pred: SBuf<[i32; 256]>,
    cf: SBuf<[i32; 256]>,
    tf: SBuf<[f32; 256]>,
    sse: i64,
    bits: f32,
    palette: Option<LossyLumaPalette>,
}

fn block_hog_kernel(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    dim: usize,
) -> [f32; 32] {
    let mut hist = [0.0f32; 32];
    let mut total = 0.1f32;
    for r in 1..dim - 1 {
        let row = (py + r) * stride + px;
        for c in 1..dim - 1 {
            let i = row + c;
            let (up, dn) = (i - stride, i + stride);
            let dx = (src[up + 1] as i32 + 2 * src[i + 1] as i32 + src[dn + 1] as i32)
                - (src[up - 1] as i32 + 2 * src[i - 1] as i32 + src[dn - 1] as i32);
            let dy = (src[dn - 1] as i32 + 2 * src[dn] as i32 + src[dn + 1] as i32)
                - (src[up - 1] as i32 + 2 * src[up] as i32 + src[up + 1] as i32);
            let magnitude = dx.abs() + dy.abs();
            if magnitude == 0 {
                continue;
            }
            total += magnitude as f32;
            if dx == 0 {
                hist[0] += (magnitude / 2) as f32;
                hist[31] += (magnitude / 2) as f32;
            } else {
                hist[hog_bin_idx(dx, dy)] += magnitude as f32;
            }
        }
    }
    for bin in &mut hist {
        *bin /= total;
    }
    hist
}

fn hog_directional_scores_kernel(hist: &[f32; 32]) -> [f32; 8] {
    std::array::from_fn(|mode| {
        let mut dot = 0.0f32;
        for (&feature, &weight) in hist.iter().zip(&INTRA_HOG_WEIGHTS[mode]) {
            dot += feature * weight;
        }
        INTRA_HOG_BIAS[mode] + dot
    })
}

impl<'a> LossyTile<'a> {
    /// 32-bin histogram of Sobel gradient orientations over the SOURCE pixels
    /// of a `dim`x`dim` block, weighted by gradient magnitude and normalized.
    /// Direct port of libaom `lowbd_generate_hog` (interior pixels only, so
    /// no neighbour access is needed).
    fn block_hog(&self, px: usize, py: usize, dim: usize) -> [f32; 32] {
        block_hog_kernel(&self.src[0], self.w, px, py, dim)
    }

    /// Scores for the eight directional modes, indexed by `mode - V_PRED`
    /// (V, H, D45, D135, D113, D157, D203, D67 — our constants are in that
    /// same order). One 8x32 dot product; the model has no hidden layer.
    fn hog_directional_scores(&self, px: usize, py: usize, dim: usize) -> [f32; 8] {
        let hist = self.block_hog(px, py, dim);
        hog_directional_scores_kernel(&hist)
    }

    fn proxy_mode_beam<const N: usize>(
        &self,
        px: usize,
        py: usize,
        dim: usize,
        have_tr: bool,
        have_bl: bool,
        pred: &mut [i32; N],
    ) -> FixedList<usize, 13> {
        let mut ranked = FixedList::<(u64, usize), 13>::new((0, DC_PRED));
        // Gradient-histogram pruning: score the eight directional modes from
        // the source's edge orientations and drop the ones that disagree with
        // it BEFORE predicting them. Non-directional modes are always kept.
        let hog = (HOG_PRUNE_THRESH > f32::NEG_INFINITY)
            .then(|| self.hog_directional_scores(px, py, dim));
        for &m in nd_modes() {
            if let Some(scores) = hog.as_ref()
                && (V_PRED..=VERT_LEFT_PRED).contains(&m)
                && scores[m - V_PRED] <= HOG_PRUNE_THRESH
            {
                continue;
            }
            if m == DC_PRED {
                let d = self.intrapred.dc_pred(&self.recon[0], self.w, px, py, dim, dim, self.bd as i32);
                pred.fill(d);
            } else {
                self.intrapred.predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    dim,
                    dim,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    pred,
                    self.bd,
                );
            }
            let score = self.rd.satd_sad_proxy(
                &self.src[0][py * self.w + px..],
                self.w,
                pred,
                dim,
                dim,
                dim,
            );
            ranked.push((score, m));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, mode)| (score, mode));
        let beam_len = proxy_mode_beam_len(self.speed, dim);
        // Relative SATD cutoff (see `rank_luma_modes`): drop dead-end
        // candidates immediately instead of running the full pipeline.
        let t = 0;
        let best = ranked.first().map_or(0, |&(c, _)| c);
        let alive = |score: u64| t == 0 || score * 100 <= best * t;
        if beam_len < 3 {
            // The Slow branch used to return a fixed {DC, PAETH} pair for
            // every format, discarding the SATD ranking above (external
            // review 2026-07-27, finding 2). Ranked keep measured: 420 mid
            // -0.23 / 422 -0.11..-0.14, but 444 +0.39/+0.41 — the fixed
            // anchors are load-bearing at 444 and stay.
            if self.speed == Speed::Slow && !self.ss420 && !self.ss422 {
                let mut keep = FixedList::new(DC_PRED);
                keep.push(DC_PRED);
                keep.push(PAETH_PRED);
                return keep;
            }
            let mut keep = FixedList::new(DC_PRED);
            for (i, &(score, mode)) in ranked.iter().enumerate() {
                if i != 0 && !alive(score) {
                    break;
                }
                keep.push(mode);
                if keep.len() == beam_len {
                    break;
                }
            }
            return keep;
        }
        let mut keep = FixedList::new(DC_PRED);
        keep.push(DC_PRED);
        keep.push(SMOOTH_PRED);
        keep.push(PAETH_PRED);
        if beam_len == keep.len() {
            return keep;
        }
        for &(score, mode) in ranked.iter() {
            if !alive(score) {
                break;
            }
            if !keep.contains(&mode) {
                keep.push(mode);
                if keep.len() == beam_len {
                    break;
                }
            }
        }
        keep
    }

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
        let reduced = self.speed.reduced_modes();
        let configured_modes: &[usize] = proxy_modes(reduced);
        let mut best = f32::INFINITY;
        if self.try_palette() {
            let key = ((px as u64) << 34) | ((py as u64) << 8) | dim as u64;
            let cached = self.pal_est_cache.borrow().get(&key).copied();
            let pairs = if let Some(pairs) = cached {
                pairs
            } else {
                let mut pairs = [(f32::INFINITY, 0.0f32); 3];
                let mut tried = 0usize;
                let hist = block_color_histogram(&self.src[0], self.w, px, py, dim, dim);
                for n in [8usize, 4, 2] {
                    if tried >= 3 {
                        break;
                    }
                    // Proxy stays Lloyd-only: it prices an upper bound and the
                    // emitters try both families.
                    let Some(p) = hist.as_deref().and_then(|hh| {
                        lossy_luma_palette_from(
                            &self.kmeans,
                            hh,
                            &self.src[0],
                            self.w,
                            px,
                            py,
                            dim,
                            dim,
                            n,
                            false,
                        )
                    }) else {
                        continue;
                    };
                    let mut pred = self.sbuf_i1024();
                    palette_pred(
                        &mut pred[..dim * dim],
                        dim,
                        &p.colors,
                        &p.packed_map,
                        dim,
                        dim,
                    );
                    // Residual left uncoded: an upper bound on the emitter's
                    // cost (which additionally codes the residual when it pays).
                    let dist = self.luma_partition_distortion(
                        px,
                        py,
                        dim,
                        dim,
                        self.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &[],
                    );
                    let bits = self.mode_bits(px, py, DC_PRED) + self.palette_rate_bits(px, py, &p);
                    pairs[tried] = (dist, bits);
                    tried += 1;
                }
                self.pal_est_cache.borrow_mut().insert(key, pairs);
                pairs
            };
            for &(dist, bits) in &pairs {
                if dist.is_finite() {
                    let c = crate::partition_rd::rd_cost(dist, mlam, bits);
                    if c < best {
                        best = c;
                    }
                }
            }
        }
        // Winning (mode, prediction, residual) per size, kept so the transform
        // refinement below can re-price the winner without re-predicting.
        let mut win: Option<(usize, [i32; 64], [i32; 64])> = None;
        let mut win16: Option<(usize, [i32; 256], [i32; 256])> = None;
        match dim {
            8 => {
                let beam = proxy_mode_beam_enabled().then(|| {
                    let mut pred = self.sbuf_i64();
                    self.proxy_mode_beam::<64>(px, py, 8, have_tr, have_bl, &mut pred)
                });
                let modes = beam.as_deref().unwrap_or(configured_modes);
                let scan = &SCAN_8X8;
                for &m in modes {
                    let mut pred = [0i32; 64];
                    if m == DC_PRED {
                        let d = self.intrapred.dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 64];
                    } else {
                        self.intrapred.predict_nd(
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
                    self.rd.residual_pred(
                        &mut resid,
                        &pred[..],
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        8,
                        8,
                    );
                    let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                    if proxy_candidate_rdoq(self.speed) && proxy_rdoq_pays(&resid[..], acq) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        8,
                        8,
                        self.quant.ac_q() as f32,
                        &pred,
                        0,
                        &rr[..],
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
                if self.speed.try_adst()
                    && proxy_tx_refine_enabled()
                    && let Some((m, pred, resid)) = win
                {
                    for txtp in [ADST_ADST_TX8_IDX, 0] {
                        let (mut cf, tf) = if txtp == ADST_ADST_TX8_IDX {
                            self.dct.adst8x8_t(&resid, &self.quant)
                        } else {
                            self.dct.idtx8x8_t(&resid, &self.quant)
                        };
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = if txtp == ADST_ADST_TX8_IDX {
                            self.idct.iadst_dequant_8x8(&cf, &self.quant)
                        } else {
                            self.idct.iidentity_dequant_8x8(&cf, &self.quant)
                        };
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.quant.ac_q() as f32,
                            &pred[..],
                            0,
                            &rr[..],
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
                    let fi_bits =
                        cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(8, 8)], 1)
                            + cdf_cost(&self.dcdf().filter_intra_mode, 0)
                            + self.mode_bits(px, py, DC_PRED);
                    for fm in FILTER_INTRA_MODES {
                        let mut pred = [0i32; 64];
                        self.intrapred.filter_predict(
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
                        self.rd.residual_pred(
                            &mut resid,
                            &pred[..],
                            &self.src[0],
                            self.w,
                            px,
                            py,
                            8,
                            8,
                        );
                        let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.quant.ac_q() as f32,
                            &pred,
                            0,
                            &rr[..],
                        );
                        let b = self.luma_bits(&cf, scan, 8, px, py, DC_PRED, 1) + fi_bits;
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
            }
            16 => {
                let coupled =
                    !self.mono && self.speed == Speed::Slow && joint_luma_uv_proxy_enabled();
                let beam = proxy_mode_beam_enabled().then(|| {
                    let mut pred = self.sbuf_i256();
                    self.proxy_mode_beam::<256>(px, py, 16, have_tr, have_bl, &mut pred)
                });
                let modes = beam.as_deref().unwrap_or(configured_modes);
                let scan = &SCAN_16X16;
                #[allow(clippy::type_complexity)]
                let mut joint_beam: [Option<(
                    f32,
                    usize,
                    SBuf<[i32; 256]>,
                    SBuf<[i32; 256]>,
                    SBuf<[i32; 256]>,
                )>; JOINT_LARGE_BEAM] = std::array::from_fn(|_| None);
                for &m in modes {
                    let mut pred = self.sbuf_i256();
                    if m == DC_PRED {
                        let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                        *pred = [d; 256];
                    } else {
                        self.intrapred.predict_nd(
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
                            &mut pred[..],
                            self.bd,
                        );
                    }
                    let mut resid = self.sbuf_i256();
                    self.rd.residual_pred(
                        &mut resid[..],
                        &pred[..],
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        16,
                        16,
                    );
                    let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                    if proxy_candidate_rdoq(self.speed) && proxy_rdoq_pays(&resid[..], acq) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &rr[..],
                    );
                    let bits =
                        self.luma_bits(&cf, scan, 16, px, py, m, 1) + self.mode_bits(px, py, m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                        if !coupled {
                            win16 = Some((m, *pred, *resid));
                        }
                    }
                    if coupled {
                        let mut pos = JOINT_LARGE_BEAM;
                        for (i, slot) in joint_beam.iter().enumerate() {
                            if slot.as_ref().is_none_or(|old| cost < old.0) {
                                pos = i;
                                break;
                            }
                        }
                        if pos < JOINT_LARGE_BEAM {
                            let mut beam_cf = self.sbuf_i256();
                            *beam_cf = cf;
                            for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                                joint_beam[i] = joint_beam[i - 1].take();
                            }
                            joint_beam[pos] = Some((cost, m, pred, resid, beam_cf));
                        }
                    }
                }
                if coupled {
                    let mut joint_best = f32::INFINITY;
                    for candidate in joint_beam.into_iter().flatten() {
                        let score = candidate.0
                            + self.joint_uv_cost16(
                                &candidate.2,
                                &candidate.4,
                                candidate.1,
                                px,
                                py,
                                prdo,
                            );
                        if !joint_best.is_finite() || score < joint_best * crate::tuning::get().joint_large_gain {
                            joint_best = score;
                        }
                    }
                    best = joint_best;
                }
                if self.speed.try_adst()
                    && proxy_tx_refine_enabled()
                    && !coupled
                    && let Some((m, pred, resid)) = win16
                {
                    let (mut cf, tf) = self.dct.adst16x16_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = self.idct.iadst_dequant_16x16(&cf, &self.quant);
                    let d = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &rr[..],
                    );
                    let b = self.luma_bits(&cf, scan, 16, px, py, m, ADST_ADST_TX16_IDX)
                        + self.mode_bits(px, py, m);
                    best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                }
                // Filter intra, mirroring the final block (DC_PRED only,
                // max(w,h) <= 32). The proxy could not see it at all, so any
                // leaf whose real encode wins with filter intra was priced as
                // if that tool did not exist.
                if proxy_filter_intra_enabled() && !coupled {
                    let fi_bits =
                        cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(16, 16)], 1)
                            + cdf_cost(&self.dcdf().filter_intra_mode, 0)
                            + self.mode_bits(px, py, DC_PRED);
                    for fm in FILTER_INTRA_MODES {
                        let mut pred = self.sbuf_i256();
                        self.intrapred.filter_predict(
                            fm,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            16,
                            16,
                            &mut pred[..],
                            self.bd,
                        );
                        let mut resid = self.sbuf_i256();
                        self.rd.residual_pred(
                            &mut resid[..],
                            &pred[..],
                            &self.src[0],
                            self.w,
                            px,
                            py,
                            16,
                            16,
                        );
                        let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            16,
                            16,
                            self.quant.ac_q() as f32,
                            &pred[..],
                            0,
                            &rr[..],
                        );
                        let b = self.luma_bits(&cf, scan, 16, px, py, DC_PRED, 1) + fi_bits;
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
            }
            32 => {
                let coupled =
                    !self.mono && self.speed == Speed::Slow && joint_luma_uv_proxy_enabled();
                let beam = proxy_mode_beam_enabled().then(|| {
                    let mut pred = self.sbuf_i1024();
                    self.proxy_mode_beam::<1024>(px, py, 32, have_tr, have_bl, &mut pred)
                });
                let modes = beam.as_deref().unwrap_or(configured_modes);
                let scan = &SCAN_32X32;
                #[allow(clippy::type_complexity)]
                let mut joint_beam: [Option<(
                    f32,
                    usize,
                    SBuf<[i32; 1024]>,
                    SBuf<[i32; 1024]>,
                )>; JOINT_LARGE_BEAM] = std::array::from_fn(|_| None);
                for &m in modes {
                    let mut pred = self.sbuf_i1024();
                    if m == DC_PRED {
                        let d = self.intrapred.dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                        *pred = [d; 1024];
                    } else {
                        self.intrapred.predict_nd(
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
                            &mut pred[..],
                            self.bd,
                        );
                    }
                    let mut resid = self.sbuf_i1024();
                    self.rd.residual_pred(
                        &mut resid[..],
                        &pred[..],
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
                    if proxy_candidate_rdoq(self.speed) && proxy_rdoq_pays(&resid[..], acq) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        32,
                        32,
                        self.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &rr[..],
                    );
                    let bits =
                        self.luma_bits(&cf, scan, 32, px, py, m, 0) + self.mode_bits(px, py, m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                    }
                    if coupled {
                        let mut pos = JOINT_LARGE_BEAM;
                        for (i, slot) in joint_beam.iter().enumerate() {
                            if slot.as_ref().is_none_or(|old| cost < old.0) {
                                pos = i;
                                break;
                            }
                        }
                        if pos < JOINT_LARGE_BEAM {
                            let mut beam_cf = self.sbuf_i1024();
                            *beam_cf = cf;
                            for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                                joint_beam[i] = joint_beam[i - 1].take();
                            }
                            joint_beam[pos] = Some((cost, m, pred, beam_cf));
                        }
                    }
                }
                if coupled {
                    let mut joint_best = f32::INFINITY;
                    for (cost, mode, pred, cf) in joint_beam.into_iter().flatten() {
                        let score = cost + self.joint_uv_cost32(&pred, &cf, mode, px, py, prdo);
                        if !joint_best.is_finite() || score < joint_best * crate::tuning::get().joint_large_gain {
                            joint_best = score;
                        }
                    }
                    best = joint_best;
                }
            }
            _ => unreachable!("rd_cost_square dim {}", dim),
        }
        best
            + rate_cost(
                mlam,
                match crate::tuning::get().block_skip_price {
                    1 => self.block_skip_bits(px, py, true),
                    2 => 0.0,
                    _ => self.block_skip_bits(px, py, false),
                },
            )
    }

    fn rd_cost_rect16_leaf_with_dc(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        prdo: f32,
        dc: i32,
    ) -> f32 {
        let mlam = self.mlam() * prdo;
        // Epoch-stamped memo: the same leaf is re-priced by HORZ/VERT, the
        // A/B legs and the x3 bottom-up recompute. The (distortion, bits) pair
        // is NOT lambda-free — the cached coefficients were trellised at
        // `trellis_lambda() * prdo` — so prdo is part of the key (external
        // review round 2, finding 7).
        let key = (
            ((px as u64) << 40) | ((py as u64) << 20) | ((dc as u64) << 1) | vert as u64,
            prdo.to_bits(),
        );
        let epoch = self.emit_epoch.get();
        if let Some(&(e, dist, bits)) = self.rect_leaf_cache.borrow().get(&key)
            && e == epoch
        {
            return crate::partition_rd::rd_cost(dist, mlam, bits);
        }
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        // Use the baseline DC/DCT representation for partition staging. The
        // selected partition receives the unchanged full mode/transform
        // refinement in its emitter; doing both here duplicated winner work.
        let dlam = trellis_lambda() * prdo;
        let (y_mode, pred, resid, cf) =
            self.rect16_luma_mode_search(px, py, vert, dc, dlam, mlam, rect_dec_refine());
        let (txtp, cf) =
            self.rect_leaf_tx_trial(&resid, &cf, &pred, px, py, vert, y_mode, dlam, mlam, rect_dec_refine());
        let rr = inv_rect_luma_128(&self.idct, &cf, &self.quant, vert, txtp);
        let distortion = self.luma_partition_distortion(
            px,
            py,
            w,
            h,
            self.quant.ac_q() as f32,
            &pred[..],
            0,
            &rr[..],
        );
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let mut bits = cdf_cost(&self.dcdf().kf_y[yctx], y_mode);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            bits += cdf_cost(&self.dcdf().angle_delta[y_mode - V_PRED], 3);
        }
        bits += if txtp == 2 || txtp == 3 {
            self.luma_rect_bits_1d(&cf, w, txtp == 2, px, py, y_mode)
        } else {
            self.luma_rect_bits(&cf, scan, w, h, px, py, y_mode, txtp)
        };
        self.rect_leaf_cache
            .borrow_mut()
            .insert(key, (epoch, distortion, bits));
        crate::partition_rd::rd_cost(distortion, mlam, bits)
    }

    fn rd_cost_rect16_leaf(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let dc = if vert {
            self.intrapred.dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            self.intrapred.dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc)
    }

    /// Reprice a rectangular leaf whose top or left edge belongs to an earlier
    /// sibling in the same asymmetric partition. The normal RDO view still has
    /// stale pixels there; source samples are a close proxy for the sibling's
    /// eventual reconstruction and, by taking the worst cost, prevent a stale
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
            sum += self
                .rd
                .sum_u16(&plane[(py - 1) * self.w + px..][..w]);
            count += w as i32;
        }
        if left {
            let plane = if source_left {
                &self.src[0]
            } else {
                &self.recon[0]
            };
            sum +=
                self.rd
                    .sum_u16_strided(&plane[py * self.w + px - 1..], self.w, h);
            count += h as i32;
        }
        let dc = if count == 0 {
            1 << (self.bd - 1)
        } else {
            (sum + count / 2) / count
        };
        base.max(self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc))
    }

    const RECT_LEAF_MODES: [usize; 7] = [
        DC_PRED,
        V_PRED,
        H_PRED,
        SMOOTH_PRED,
        SMOOTH_V_PRED,
        SMOOTH_H_PRED,
        PAETH_PRED,
    ];

    /// Mode search for a rectangular luma leaf. `refine=false` selects the
    /// decision-stage DC/DCT proxy; emitters pass `true` and retain the full
    /// mode beam. Returns mode, prediction, residual, and DCT coefficients.
    #[allow(clippy::too_many_arguments)]
    fn rect16_luma_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        dc: i32,
        lam: f32,
        mlam: f32,
        refine: bool,
    ) -> (usize, [i32; 128], [i32; 128], [i32; 128]) {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let kf = &self.dcdf().kf_y[yctx];
        // The equipped leaf (mode search + 7-type tx trial) is a Slow tool:
        // at Medium it grew the tier ~60% for decision work Medium's budget
        // cannot carry — Medium keeps the pre-equip DC leaf.
        let modes: &[usize] =
            if !refine || !rect_leaf_mode_search_enabled() || self.speed != Speed::Slow {
                &[DC_PRED]
            } else {
                &Self::RECT_LEAF_MODES
            };

        // Prediction-domain SATD rerank: predictions are cheap next to the
        // DCT + trellis + entropy-rate pipeline, so score every candidate on
        // SATD and run the full pipeline on the top RECT_LEAF_BEAM only (DC
        // always retained as the safe fallback). Same pruning the square
        // proxy beam and the AV2 intra search use.
        let mut cands = FixedList::<(u64, usize), 7>::new((0, DC_PRED));
        for &m in modes {
            let mut pred = [0i32; 128];
            if m == DC_PRED {
                pred = [dc; 128];
            } else {
                self.intrapred.predict_nd(
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
            let score = self.rd.satd_sad_proxy(&self.src[0][py * self.w + px..], self.w, &pred, w, w, h);
            cands.push((score, m));
        }
        const RECT_LEAF_BEAM: usize = 3;
        if cands.len() > RECT_LEAF_BEAM {
            cands
                .as_mut_slice()
                .sort_unstable_by_key(|&(score, mode)| (score, mode));
            let dc_pos = cands.iter().position(|&(_, m)| m == DC_PRED).unwrap();
            if dc_pos >= RECT_LEAF_BEAM {
                cands.as_mut_slice().swap(RECT_LEAF_BEAM - 1, dc_pos);
            }
            cands.truncate(RECT_LEAF_BEAM);
        }

        let mut best = (
            f32::INFINITY,
            DC_PRED,
            [0i32; 128],
            [0i32; 128],
            [0i32; 128],
        );
        for &(_, m) in &cands {
            let mut pred = [0i32; 128];
            if m == DC_PRED {
                pred.fill(dc);
            } else {
                self.intrapred.predict_nd(
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
            self.rd.residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, w, h);
            let (mut cf, tf) = if vert {
                self.dct.dct8x16_t(&resid, &self.quant)
            } else {
                self.dct.dct16x8_t(&resid, &self.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, w, h, px, py);
            self.rd.preserve_dc(&mut cf[0], &resid[..]);
            let rr = if vert {
                self.idct.idct_dequant_8x16(&cf, &self.quant)
            } else {
                self.idct.idct_dequant_16x8(&cf, &self.quant)
            };
            let sse =
                self.rd.sse_recon(&pred, &rr, &self.src[0], self.w, px, py, w, h, self.bd);
            let bits = self.luma_rect_bits(&cf, scan, w, h, px, py, m, 1) + cdf_cost(kf, m);
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best.0 {
                best = (cost, m, pred, resid, cf);
            }
        }
        (best.1, best.2, best.3, best.4)
    }

    /// Transform-type trial for a rectangular luma leaf: ADST_ADST against the
    /// committed DCT_DCT candidate.
    #[allow(clippy::too_many_arguments)]
    fn rect_leaf_tx_trial(
        &self,
        resid: &[i32; 128],
        dct_cf: &[i32; 128],
        pred: &[i32; 128],
        px: usize,
        py: usize,
        vert: bool,
        y_mode: usize,
        lam: f32,
        mlam: f32,
        refine: bool,
    ) -> (usize, [i32; 128]) {
        if !refine || !rect_leaf_tx_search_enabled() || self.speed != Speed::Slow {
            return (1, *dct_cf);
        }
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let sse_of = |rr: &[i32; 128]| -> i64 {
            self.rd.sse_recon(pred, rr, &self.src[0], self.w, px, py, w, h, self.bd)
        };

        let dct_rr = if vert {
            self.idct.idct_dequant_8x16(dct_cf, &self.quant)
        } else {
            self.idct.idct_dequant_16x8(dct_cf, &self.quant)
        };
        let mut best_sse = sse_of(&dct_rr);
        let mut best = (
            rd_cost_i64(
                best_sse,
                mlam,
                self.luma_rect_bits(dct_cf, scan, w, h, px, py, y_mode, 1),
            ),
            1usize,
            *dct_cf,
        );

        // Trellis'd 2-D candidates: ADST_ADST (4), ADST_DCT (5), DCT_ADST (6).
        for txtp in [ADST_ADST_TX8_IDX, 5, 6] {
            let (mut acf, atf) = match (vert, txtp) {
                (true, ADST_ADST_TX8_IDX) => self.dct.adst8x16_t(resid, &self.quant),
                (true, 5) => self.dct.adstdct8x16_t(resid, &self.quant),
                (true, _) => self.dct.dctadst8x16_t(resid, &self.quant),
                (false, ADST_ADST_TX8_IDX) => self.dct.adst16x8_t(resid, &self.quant),
                (false, 5) => self.dct.adstdct16x8_t(resid, &self.quant),
                (false, _) => self.dct.dctadst16x8_t(resid, &self.quant),
            };
            self.luma_rect_trellis(&mut acf, &atf, dcq, acq, scan, lam, w, h, px, py);
            // Same DC-preservation snap the DCT candidate gets.
            self.rd.preserve_dc(&mut acf[0], &resid[..]);
            let rr = inv_rect_luma_128(&self.idct, &acf, &self.quant, vert, txtp);
            let sse = sse_of(&rr);
            let c = rd_cost_i64(
                sse,
                mlam,
                self.luma_rect_bits(&acf, scan, w, h, px, py, y_mode, txtp),
            );
            if sse <= best_sse + (best_sse >> 5) && c < best.0 {
                best = (c, txtp, acf);
                best_sse = sse;
            }
        }
        // IDTX (0) and the 1-D classes V_DCT (2) / H_DCT (3): no RDOQ (see
        // the 8x8 IDTX note — the 2026-07-22 full-RDOQ retry measured flat
        // to slightly negative on holdout at every format) and the STRICT
        // SSE gate.
        {
            let (icf, _itf) = if w == 8 {
                self.dct.idtx8x16_t(resid, &self.quant)
            } else {
                self.dct.idtx16x8_t(resid, &self.quant)
            };
            let rr = inv_rect_luma_128(&self.idct, &icf, &self.quant, vert, 0);
            let sse = sse_of(&rr);
            let c = rd_cost_i64(
                sse,
                mlam,
                self.luma_rect_bits(&icf, scan, w, h, px, py, y_mode, 0),
            );
            if sse <= best_sse && c < best.0 {
                best = (c, 0, icf);
                best_sse = sse;
            }
        }
        for (txtp, one_d_vertical) in [(2usize, true), (3usize, false)] {
            let (vcf, _vtf) = match (vert, one_d_vertical) {
                (true, true) => self.dct.fvdct8x16_t(resid, &self.quant),
                (true, false) => self.dct.fhdct8x16_t(resid, &self.quant),
                (false, true) => self.dct.fvdct16x8_t(resid, &self.quant),
                (false, false) => self.dct.fhdct16x8_t(resid, &self.quant),
            };
            let rr = inv_rect_luma_128(&self.idct, &vcf, &self.quant, vert, txtp);
            let sse = sse_of(&rr);
            let bits = self.luma_rect_bits_1d(&vcf, w, one_d_vertical, px, py, y_mode);
            let c = rd_cost_i64(sse, mlam, bits);
            if sse <= best_sse && c < best.0 {
                best = (c, txtp, vcf);
                best_sse = sse;
            }
        }
        (best.1, best.2)
    }

    fn rd_cost_horz(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, self.part16_rate(px / 8, py / 8, 1))
            + self.rd_cost_rect16_leaf(px, py, false, prdo)
            + self.rd_cost_rect16_leaf(px, py + 8, false, prdo)
    }

    fn rd_cost_vert(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, self.part16_rate(px / 8, py / 8, 2))
            + self.rd_cost_rect16_leaf(px, py, true, prdo)
            + self.rd_cost_rect16_leaf(px + 8, py, true, prdo)
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn rect444_uv_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        y_mode: usize,
        cur_total: f32,
        lam: f32,
        mlam: f32,
    ) -> Option<(usize, [[i32; 128]; 2], [[i32; 128]; 2])> {
        let (w, h) = if vert {
            (8usize, 16usize)
        } else {
            (16usize, 8usize)
        };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
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
        let directional_top = self.rank_chroma_modes::<128>(candidates, px, py, px, py, w, h);
        let mut best_total = cur_total;
        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, [[i32; 128]; 2], [[i32; 128]; 2])> = None;
        for &cand in candidates {
            if cand != V_PRED
                && cand != H_PRED
                && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                && !self.speed.chroma_angle_directional()
            {
                continue;
            }
            if !directional_top.contains(cand) {
                continue;
            }
            let tx = chroma_tx_for_mode(cand);
            let mut cand_ccf = [[0i32; 128]; 2];
            let mut cand_pred = [[0i32; 128]; 2];
            let mut cand_total = rate_cost(mlam, self.uv_mode_bits(y_mode, cand, None));
            for ci in 0..2 {
                let plane = ci + 1;
                self.intrapred.predict_nd(
                    cand,
                    &self.recon[plane],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    false,
                    false,
                    self.w,
                    self.h,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci],
                    self.bd,
                );
                let mut resid = [0i32; 128];
                self.rd.residual_pred(
                    &mut resid,
                    &cand_pred[ci],
                    &self.src[plane],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                );
                let (mut q, qt) = if vert {
                    fwd_chroma_8x16(&self.dct, tx, &resid, &self.cquant)
                } else {
                    fwd_chroma_16x8(&self.dct, tx, &resid, &self.cquant)
                };
                trellis_optimize(&mut q, &qt, cdcq, cacq, scan, lam);
                self.rd.preserve_dc(&mut q[0], &resid[..]);
                let rr = if vert {
                    inv_chroma_8x16(&self.idct, tx, &q, &self.cquant)
                } else {
                    inv_chroma_16x8(&self.idct, tx, &q, &self.cquant)
                };
                let sse = self.rd.sse_recon(
                    &cand_pred[ci],
                    &rr,
                    &self.src[plane],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    self.bd,
                );
                cand_ccf[ci] = q;
                cand_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_rect_bits(&q, scan, w, h, plane, px, py),
                );
            }
            if cand_total < best_total {
                best_total = cand_total;
                best = Some((cand, cand_ccf, cand_pred));
            }
        }
        best
    }

    fn code_block16_vert_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8 + half * 8, y8 * 8);
            let (bx4, by4) = (px / 4, py / 4);
            let dc_l = self.intrapred.dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let (y_mode, lpred_arr, lresid, lcf) =
                self.rect16_luma_mode_search(px, py, true, dc_l, lam, emlam, true);
            let (ltxtp, lcf) = self.rect_leaf_tx_trial(
                &lresid, &lcf, &lpred_arr, px, py, true, y_mode, lam, emlam, true,
            );
            let idct = self.idct;
            let inv8x16 = |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, true, ltxtp);
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = self.intrapred.dc_pred_8x16(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                self.rd.residual_dc(&mut resid, &self.src[plane], self.w, px, py, 8, 16, dc);
                let (mut q, qt) = self.dct.dct8x16_t(&resid, &self.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_8X16, lam, 8, 16, plane, px, py,
                );
                self.rd.preserve_dc(&mut q[0], &resid[..]);
                ccf[ci] = q;
            }
            // --- CfL: predict chroma from this sub-block's reconstructed luma.
            // Allowed here (8x16 <= 32x32); at 4:4:4 chroma is full resolution
            // so chroma-from-luma is worth far more than a luma mode search.
            let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_pred = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
            {
                let lrr_cfl = inv8x16(&lcf, &self.quant);
                let mut luma_rec = [0u16; 128];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
                let mut ac = [0i32; 128];
                self.intrapred.cfl_ac_444(&luma_rec, 8, 16, &mut ac);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut csrc = [0u16; 128];
                    self.rd
                        .copy_block_u16(&mut csrc, &self.src[plane], self.w, px, py, 8, 16);
                    let dcrr = self.idct.idct_dequant_8x16(&ccf[ci], &self.cquant);
                    let s =
                        self.rd.sse_recon(&[dc; 128], &dcrr, &csrc, 8, 0, 0, 8, 16, self.bd);
                    dc_cost[ci] = rd_cost_i64(
                        s,
                        mlam,
                        self.chroma_rect_bits(&ccf[ci], &SCAN_8X16, 8, 16, plane, px, py),
                    );
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &csrc, dc, 128, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 128];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..128], dc, a, self.bd);
                    let mut resid = [0i32; 128];
                    self.rd.residual_pred(&mut resid, &cpr, &csrc, 8, 0, 0, 8, 16);
                    let (mut q, qt) = self.dct.dct8x16_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, cdcq, cacq, &SCAN_8X16, lam, 8, 16, plane, px, py,
                    );
                    let rr2 = self.idct.idct_dequant_8x16(&q, &self.cquant);
                    let s2 = self.rd.sse_recon(&cpr, &rr2, &csrc, 8, 0, 0, 8, 16, self.bd);
                    cfl_ccf[ci] = q;
                    cfl_pred[ci] = cpr;
                    cfl_cost[ci] = rd_cost_i64(
                        s2,
                        mlam,
                        self.chroma_rect_bits(&q, &SCAN_8X16, 8, 16, plane, px, py),
                    );
                }
            }
            // CFL_PRED uv_mode symbol + one alpha symbol per non-zero plane.
            let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
            let mut use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                    < dc_cost[0]
                        + dc_cost[1]
                        + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None));
            if use_cfl {
                ccf = cfl_ccf;
            }
            // Rect-chroma equip: directional/smooth UV candidates compete with
            // the DC/CfL winner on the same R-D bar (mirrors NONE-16's search).
            let mut chosen_uv = DC_PRED;
            let mut uv_pred: Option<[[i32; 128]; 2]> = None;
            if self.speed.full_chroma_rdo() {
                let cur_total = if use_cfl {
                    cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                } else {
                    dc_cost[0]
                        + dc_cost[1]
                        + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None))
                };
                if let Some((m, c2, p2)) =
                    self.rect444_uv_mode_search(px, py, true, y_mode, cur_total, lam, mlam)
                {
                    chosen_uv = m;
                    ccf = c2;
                    uv_pred = Some(p2);
                    use_cfl = false;
                }
            }
            let cfl_opt = if use_cfl { Some(cfl_a) } else { None };
            let luma_zero = self.rd.all_zero_i32(&lcf);
            let chroma_zero =
                self.rd.all_zero_i32(&ccf[0]) && self.rd.all_zero_i32(&ccf[1]);
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
            self.emit_uv_mode(y_mode, chosen_uv, cfl_opt, px, py, 8, 16);
            self.emit_palette_mode_info(px, py, 8, 16, y_mode, !self.mono, None, None);
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
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        8,
                        ltxtp == 2,
                        sk,
                        ds,
                        y_mode,
                    )
                } else {
                    encode_8x16_luma_coeffs(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        sk,
                        ds,
                        y_mode,
                        ltxtp,
                    )
                }
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
                } else if chosen_uv != DC_PRED {
                    // Directional/smooth uv mode: decoder-derived chroma tx.
                    inv_chroma_8x16(
                        &self.idct,
                        chroma_tx_for_mode(chosen_uv),
                        &ccf[ci],
                        &self.cquant,
                    )
                } else {
                    self.idct.idct_dequant_8x16(&ccf[ci], &self.cquant)
                };
                for ry in 0..16 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if let Some(up) = &uv_pred {
                        recon_add_pred(&mut drow[..8], &up[ci][ry * 8..], &rr[ry * 8..], maxval);
                    } else if use_cfl {
                        recon_add_pred(
                            &mut drow[..8],
                            &cfl_pred[ci][ry * 8..],
                            &rr[ry * 8..],
                            maxval,
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
}

/// Scan table for the chroma transform shapes an H4/V4 strip can carry.
fn chroma_scan_for(cw: usize, ch: usize) -> &'static [u32] {
    match (cw, ch) {
        (16, 4) => &SCAN_16X4,
        (4, 16) => &SCAN_4X16,
        (8, 4) => &SCAN_8X4,
        (4, 8) => &SCAN_4X8,
        _ => unreachable!("strip chroma shape {cw}x{ch}"),
    }
}

/// Forward DCT + quant target for a strip's chroma transform. All shapes have
/// <= 64 coefficients; smaller shapes leave the array tail zero.
fn fwd_chroma_strip(
    dct: &DctDispatch,
    cw: usize,
    ch: usize,
    resid: &[i32; 64],
    q: &Quant,
) -> ([i32; 64], [f32; 64]) {
    match (cw, ch) {
        (16, 4) => dct.dct16x4_t(resid, q),
        (4, 16) => dct.dct4x16_t(resid, q),
        (8, 4) => {
            let (cf, tf) = dct.dct8x4_t(resid.first_chunk::<32>().unwrap(), q);
            let mut cf64 = [0i32; 64];
            let mut tf64 = [0f32; 64];
            cf64[..32].copy_from_slice(&cf);
            tf64[..32].copy_from_slice(&tf);
            (cf64, tf64)
        }
        (4, 8) => {
            let (cf, tf) = dct.dct4x8_t(resid.first_chunk::<32>().unwrap(), q);
            let mut cf64 = [0i32; 64];
            let mut tf64 = [0f32; 64];
            cf64[..32].copy_from_slice(&cf);
            tf64[..32].copy_from_slice(&tf);
            (cf64, tf64)
        }
        _ => unreachable!(),
    }
}

/// Inverse + dequant for a strip's chroma transform, row-major `cw` wide.
fn inv_chroma_strip(
    idct: &IdctDispatch,
    cw: usize,
    ch: usize,
    cf: &[i32; 64],
    q: &Quant,
) -> [i32; 64] {
    match (cw, ch) {
        (16, 4) => idct.idct_dequant_16x4(cf, q),
        (4, 16) => idct.idct_dequant_4x16(cf, q),
        (8, 4) => {
            let r = idct.idct_dequant_8x4(cf.first_chunk::<32>().unwrap(), q);
            let mut out = [0i32; 64];
            out[..32].copy_from_slice(&r);
            out
        }
        (4, 8) => {
            let r = idct.idct_dequant_4x8(cf.first_chunk::<32>().unwrap(), q);
            let mut out = [0i32; 64];
            out[..32].copy_from_slice(&r);
            out
        }
        _ => unreachable!(),
    }
}

/// Chroma coefficient coder dispatch for a strip's transform.
fn encode_chroma_strip(
    enc: &mut OdEcEncoder,
    cdfs: &mut Cdfs,
    cw: usize,
    ch: usize,
    cf: &[i32; 64],
    sk: usize,
    ds: usize,
) -> u8 {
    match (cw, ch) {
        (16, 4) => encode_16x4_chroma_coeffs(enc, cdfs, cf, sk, ds),
        (4, 16) => encode_4x16_chroma_coeffs(enc, cdfs, cf, sk, ds),
        (8, 4) => encode_8x4_chroma_coeffs(enc, cdfs, cf.first_chunk::<32>().unwrap(), sk, ds),
        (4, 8) => encode_4x8_chroma_coeffs(enc, cdfs, cf.first_chunk::<32>().unwrap(), sk, ds),
        _ => unreachable!(),
    }
}

impl<'a> LossyTile<'a> {
    /// Code a 16x16 region as PARTITION_HORZ_4 (four stacked 16x4 strips) or
    /// PARTITION_VERT_4 (four side-by-side 4x16). DC luma prediction + DCT
    /// (RTX_16X4/RTX_4X16), DC chroma. Chroma per format:
    /// - 4:4:4: every strip carries its own 16x4/4x16 chroma transform;
    /// - 4:2:2: HORZ_4 strips carry 8x4 chroma each (VERT_4 not selected at
    ///   4:2:2, matching the VERT rule);
    /// - 4:2:0: the subsampled dimension pairs strips — chroma is coded once
    ///   per pair, on strips 1 and 3, covering the pair's 8x4 / 4x8 region;
    /// - mono: luma only.
    fn code_block16_quad(&mut self, x8: usize, y8: usize, vert: bool) {
        for i in 0..4 {
            let (px, py) = if vert {
                (x8 * 8 + 4 * i, y8 * 8)
            } else {
                (x8 * 8, y8 * 8 + 4 * i)
            };
            self.code_strip16(px, py, vert, i);
        }
    }

    fn code_strip16(&mut self, px: usize, py: usize, vert: bool, idx: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(px, py, 16);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (lw, lh) = if vert { (4usize, 16usize) } else { (16, 4) };
        let (bx4, by4) = (px / 4, py / 4);
        let lscan: &[u32] = if vert { &SCAN_4X16 } else { &SCAN_16X4 };

        // --- luma: DC predict, forward, trellis, dc-preservation nudge ---
        let lpred = self.intrapred.dc_pred(&self.recon[0], self.w, px, py, lw, lh, self.bd as i32);
        let mut lresid = [0i32; 64];
        self.rd.residual_dc(&mut lresid, &self.src[0], self.w, px, py, lw, lh, lpred);
        let (mut lcf, ltf) = if vert {
            self.dct.dct4x16_t(&lresid, &self.quant)
        } else {
            self.dct.dct16x4_t(&lresid, &self.quant)
        };
        trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
        self.rd.preserve_dc(&mut lcf[0], &lresid[..]);

        // --- chroma geometry per format ---
        // (cx, cy, cw, ch) of the chroma transform this strip carries, if any.
        let chroma: Option<(usize, usize, usize, usize)> = if self.mono {
            None
        } else if self.ss420 {
            // Strips pair in the subsampled dimension; the odd strip codes the
            // pair's chroma (spec sub-8 rule: chroma belongs to the last block
            // covering the chroma unit).
            if idx % 2 == 1 {
                if vert {
                    Some(((px - 4) / 2, py / 2, 4, 8))
                } else {
                    Some((px / 2, (py - 4) / 2, 8, 4))
                }
            } else {
                None
            }
        } else if self.ss422 {
            // 4:2:2, HORZ_4 only: each 16x4 strip has 8x4 chroma.
            Some((px / 2, py, 8, 4))
        } else {
            Some((px, py, lw, lh))
        };

        let mut ccf = [[0i32; 64]; 2];
        let mut cn = 0usize; // chroma coeff count (cw*ch) when present
        let mut cpred = [0i32; 2];
        if let Some((cx, cy, cw, ch)) = chroma {
            cn = cw * ch;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = self.intrapred.dc_pred(&self.recon[plane], self.cw, cx, cy, cw, ch, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 64];
                self.rd.residual_dc(
                    &mut resid,
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    cw,
                    ch,
                    dc,
                );
                let cscan = chroma_scan_for(cw, ch);
                let (mut q, qt) = fwd_chroma_strip(&self.dct, cw, ch, &resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                self.rd.preserve_dc(&mut q[0], &resid[..cn]);
                ccf[ci] = q;
            }
        }

        let luma_zero = self.rd.all_zero_i32(&lcf[..]);
        let chroma_zero = chroma.is_none()
            || (self.rd.all_zero_i32(&ccf[0][..cn])
                && self.rd.all_zero_i32(&ccf[1][..cn]));
        let block_skip = luma_zero && chroma_zero;

        // --- header syntax, decoder order ---
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.record_blk_rect4(bx4, by4, (lw / 4) as u8, (lh / 4).max(1) as u8);
        // skip8 is 8px-granular (CDEF only); the second strip of each 8px band
        // overwrites the first — acceptable while CDEF search reads it only as
        // a hint.
        if vert {
            self.mark_skip8_rect(px / 8, py / 8, 1, 2, block_skip);
        } else {
            self.mark_skip8_rect(px / 8, py / 8, 2, 1, block_skip);
        }
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
        if chroma.is_some() {
            self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, lw, lh);
        }
        self.emit_palette_mode_info(px, py, lw, lh, DC_PRED, chroma.is_some(), None, None);
        self.emit_filter_intra(DC_PRED, lw, lh, None);
        self.code_tx_depth(px, py, lw, lh, 0);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, (lh / 4).max(1));
        self.a_skip[bx4..bx4 + aw].fill(sv);
        self.l_skip[by4..by4 + ah].fill(sv);
        self.a_mode[bx4..bx4 + aw].fill(DC_PRED as u8);
        self.l_mode[by4..by4 + ah].fill(DC_PRED as u8);

        // --- luma coefficients + recon ---
        let lres_ctx = if block_skip {
            0x40
        } else {
            let ds = self.dc_sign_ctx_span(0, bx4, by4, aw, ah);
            if vert {
                encode_4x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, DC_PRED, 1)
            } else {
                encode_16x4_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, DC_PRED, 1)
            }
        };
        self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 64]
        } else if vert {
            self.idct.idct_dequant_4x16(&lcf, &self.quant)
        } else {
            self.idct.idct_dequant_16x4(&lcf, &self.quant)
        };
        for ry in 0..lh {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_dc(&mut drow[..lw], lpred, &lrr[ry * lw..], maxval);
        }

        // --- chroma coefficients + recon ---
        if let Some((cx, cy, cw, ch)) = chroma {
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let (caw, cah) = ((cw / 4).max(1), (ch / 4).max(1));
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let a = &self.a_coef[plane];
                    let l = &self.l_coef[plane];
                    let ca = a[cbx4..(cbx4 + caw).min(a.len())]
                        .iter()
                        .any(|&x| x != 0x40) as usize;
                    let cl = l[cby4..(cby4 + cah).min(l.len())]
                        .iter()
                        .any(|&x| x != 0x40) as usize;
                    let sk = 7 + ca + cl;
                    let ds = self.dc_sign_ctx_span(plane, cbx4, cby4, caw, cah);
                    encode_chroma_strip(&mut self.enc, &mut self.cdfs, cw, ch, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    inv_chroma_strip(&self.idct, cw, ch, &ccf[ci], &self.cquant)
                };
                for ry in 0..ch {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                }
            }
        }
    }

    /// Emit one 16x8/8x16 leaf. Used by binary and asymmetric partitions.
    fn code_block16_rect_leaf_420(&mut self, x8: usize, y8: usize, vert: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        // Luma: mode search + 7-type transform trial (shared with 444).
        let dc_l = if vert {
            self.intrapred.dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            self.intrapred.dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let (y_mode, lpred_arr, lresid, lcf) =
            self.rect16_luma_mode_search(px, py, vert, dc_l, lam, emlam, true);
        let (ltxtp, lcf) = self.rect_leaf_tx_trial(
            &lresid, &lcf, &lpred_arr, px, py, vert, y_mode, lam, emlam, true,
        );
        let idct = self.idct;
        let inv_luma = |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, vert, ltxtp);
        // chroma 8x4 (horz) or 4x8 (vert) at chroma coords.
        let (cx, cy) = (px / 2, py / 2);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let (cw, ch) = if vert { (4usize, 8usize) } else { (8, 4) };
        let mut ccf = [[0i32; 32]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = if vert {
                self.intrapred.dc_pred_4x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            } else {
                self.intrapred.dc_pred_8x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            };
            cpred[ci] = dc;
            let mut resid = [0i32; 32];
            self.rd.residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, cw, ch, dc);
            let (mut q, qt) = if vert {
                self.dct.dct4x8_t(&resid, &self.cquant)
            } else {
                self.dct.dct8x4_t(&resid, &self.cquant)
            };
            let cscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
            self.rd.preserve_dc(&mut q[0], &resid[..cw * ch]);
            ccf[ci] = q;
        }
        let mut use_cfl = false;
        let mut cfl_alpha = [0i32; 2];
        let mut cfl_px = [[0i32; 32]; 2];
        if !self.mono {
            let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let lrr_cfl = inv_luma(&lcf, &self.quant);
            let mut luma_rec = [0u16; 128];
            recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
            let mut ac = [0i32; 32];
            self.intrapred
                .cfl_ac_sub(&luma_rec, lw, cw, ch, true, true, &mut ac);
            let cscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            let n = cw * ch;
            let mut cfl_ccf = [[0i32; 32]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = (0f32, 0f32);
            let mut cand_px = [[0i32; 32]; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let mut src = [0u16; 32];
                self.rd
                    .copy_block_u16(&mut src[..n], &self.src[plane], self.cw, cx, cy, cw, ch);
                let dcrr = if vert {
                    self.idct.idct_dequant_4x8(&ccf[ci], &self.cquant)
                } else {
                    self.idct.idct_dequant_8x4(&ccf[ci], &self.cquant)
                };
                let s0 = self.rd.sse_recon(
                    &[dc; 32][..n],
                    &dcrr[..n],
                    &src,
                    cw,
                    0,
                    0,
                    cw,
                    ch,
                    self.bd,
                );
                dc_cost += rd_cost_i64(
                    s0,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci], cscan, cw, ch, plane, cx, cy),
                );
                let a = self
                    .intrapred
                    .cfl_best_alpha(&ac, &src, dc, n, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 32];
                self.intrapred
                    .cfl_pred(&mut cpr[..n], &ac[..n], dc, a, self.bd);
                let mut resid = [0i32; 32];
                self.rd.residual_pred(&mut resid, &cpr, &src, cw, 0, 0, cw, ch);
                let (mut q, qt) = if vert {
                    self.dct.dct4x8_t(&resid, &self.cquant)
                } else {
                    self.dct.dct8x4_t(&resid, &self.cquant)
                };
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                let rr2 = if vert {
                    self.idct.idct_dequant_4x8(&q, &self.cquant)
                } else {
                    self.idct.idct_dequant_8x4(&q, &self.cquant)
                };
                let s2 =
                    self.rd.sse_recon(&cpr[..n], &rr2[..n], &src, cw, 0, 0, cw, ch, self.bd);
                cfl_ccf[ci] = q;
                cand_px[ci] = cpr;
                cfl_cost += rd_cost_i64(
                    s2,
                    mlam,
                    self.chroma_rect_bits(&q, cscan, cw, ch, plane, cx, cy),
                );
            }
            let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
            let dc_sig = self.uv_mode_bits(y_mode, DC_PRED, None);
            if (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost + rate_cost(mlam, cfl_sig) < dc_cost + rate_cost(mlam, dc_sig)
            {
                use_cfl = true;
                cfl_alpha = cfl_a;
                cfl_px = cand_px;
                ccf = cfl_ccf;
            }
        }
        let luma_zero = self.rd.all_zero_i32(&lcf);
        let chroma_zero =
            self.rd.all_zero_i32(&ccf[0]) && self.rd.all_zero_i32(&ccf[1]);
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
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(
            y_mode,
            DC_PRED,
            if use_cfl { Some(cfl_alpha) } else { None },
            px,
            py,
            lw,
            lh,
        );
        self.emit_palette_mode_info(px, py, lw, lh, y_mode, !self.mono, None, None);
        self.emit_filter_intra(y_mode, lw, lh, None);
        self.code_tx_depth(px, py, lw, lh, 0);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, lh / 4);
        self.a_skip[bx4..bx4 + aw].fill(sv);
        self.l_skip[by4..by4 + ah].fill(sv);
        self.a_mode[bx4..bx4 + aw].fill(y_mode as u8);
        self.l_mode[by4..by4 + ah].fill(y_mode as u8);
        let lres_ctx = if block_skip {
            0x40
        } else if ltxtp == 2 || ltxtp == 3 {
            let (sk, ds) = if vert {
                (
                    self.skip_ctx_8x16_luma(),
                    self.dc_sign_ctx_8x16_luma(bx4, by4),
                )
            } else {
                (
                    self.skip_ctx_16x8_luma(),
                    self.dc_sign_ctx_16x8_luma(bx4, by4),
                )
            };
            encode_rect_coeffs_1d(
                &mut self.enc,
                &mut self.cdfs,
                &lcf,
                lw,
                ltxtp == 2,
                sk,
                ds,
                y_mode,
            )
        } else if vert {
            let sk = self.skip_ctx_8x16_luma();
            let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
            encode_8x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, y_mode, ltxtp)
        } else {
            let sk = self.skip_ctx_16x8_luma();
            let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
            encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, y_mode, ltxtp)
        };
        self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 128]
        } else {
            inv_luma(&lcf, &self.quant)
        };
        for ry in 0..lh {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_pred(
                &mut drow[..lw],
                &lpred_arr[ry * lw..],
                &lrr[ry * lw..],
                maxval,
            );
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
                self.idct.idct_dequant_4x8(&ccf[ci], &self.cquant)
            } else {
                self.idct.idct_dequant_8x4(&ccf[ci], &self.cquant)
            };
            for ry in 0..ch {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_cfl {
                    recon_add_pred(
                        &mut drow[..cw],
                        &cfl_px[ci][ry * cw..],
                        &rr[ry * cw..],
                        maxval,
                    );
                } else {
                    recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                }
            }
        }
    }

    /// 4:2:2 HORZ: two 16x8 luma + 8x8 chroma (h-subsampled, v-full). V forbidden in 422.
    fn code_block16_horz_422(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4);
            // Luma 16x8: mode search + 7-type transform trial (shared with 444).
            let dc_l = self.intrapred.dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let (y_mode, lpred_arr, lresid, lcf) =
                self.rect16_luma_mode_search(px, py, false, dc_l, lam, emlam, true);
            let (ltxtp, lcf) = self.rect_leaf_tx_trial(
                &lresid, &lcf, &lpred_arr, px, py, false, y_mode, lam, emlam, true,
            );
            let idct = self.idct;
            let inv16x8 =
                |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, false, ltxtp);
            let (cx, cy) = (px / 2, py);
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let mut ccf = [[0i32; 64]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = self.intrapred.dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 64];
                self.rd.residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, 8, 8, dc);
                let (mut q, qt) = self.dct.dct8x8_t(&resid, &self.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
                );
                self.rd.preserve_dc(&mut q[0], &resid[..]);
                ccf[ci] = q;
            }
            // CfL candidate (2026-07-23 rect-CfL audit: the 4:2:2 HORZ leg —
            // a large share of 422 blocks — never offered CfL). AC reference
            // from this leaf's 16x8 reconstruction, horizontally subsampled.
            let mut use_cfl = false;
            let mut cfl_alpha = [0i32; 2];
            let mut cfl_px = [[0i32; 64]; 2];
            if !self.mono {
                let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
                let lrr_cfl = inv16x8(&lcf, &self.quant);
                let mut luma_rec = [0u16; 128];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
                let mut ac = [0i32; 64];
                self.intrapred
                    .cfl_ac_sub(&luma_rec, 16, 8, 8, true, false, &mut ac);
                let mut cfl_ccf = [[0i32; 64]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut dc_cost, mut cfl_cost) = (0f32, 0f32);
                let mut cand_px = [[0i32; 64]; 2];
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0u16; 64];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, cx, cy, 8, 8);
                    let dcrr = self.idct.idct_dequant_8x8(&ccf[ci], &self.cquant);
                    let s0 = sse_recon::<64, 8>(&self.rd, &[dc; 64], &dcrr, &src, 8, 0, 0, self.bd);
                    dc_cost += rd_cost_i64(
                        s0,
                        mlam,
                        self.chroma_bits(&ccf[ci], &SCAN_8X8, 8, plane, cx, cy),
                    );
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 64];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..64], dc, a, self.bd);
                    let mut resid = [0i32; 64];
                    self.rd.residual_pred(&mut resid, &cpr, &src, 8, 0, 0, 8, 8);
                    let (mut q, qt) = self.dct.dct8x8_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, cdcq, cacq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
                    );
                    let rr2 = self.idct.idct_dequant_8x8(&q, &self.cquant);
                    let s2 = sse_recon::<64, 8>(&self.rd, &cpr, &rr2, &src, 8, 0, 0, self.bd);
                    cfl_ccf[ci] = q;
                    cand_px[ci] = cpr;
                    cfl_cost +=
                        rd_cost_i64(s2, mlam, self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy));
                }
                let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
                let dc_sig = self.uv_mode_bits(y_mode, DC_PRED, None);
                if (cfl_a[0] != 0 || cfl_a[1] != 0)
                    && cfl_cost + rate_cost(mlam, cfl_sig) < dc_cost + rate_cost(mlam, dc_sig)
                {
                    use_cfl = true;
                    cfl_alpha = cfl_a;
                    cfl_px = cand_px;
                    ccf = cfl_ccf;
                }
            }
            let luma_zero = self.rd.all_zero_i32(&lcf);
            let chroma_zero =
                self.rd.all_zero_i32(&ccf[0]) && self.rd.all_zero_i32(&ccf[1]);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
            if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
                self.enc
                    .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
            }
            self.emit_uv_mode(
                y_mode,
                DC_PRED,
                if use_cfl { Some(cfl_alpha) } else { None },
                px,
                py,
                16,
                8,
            );
            self.emit_palette_mode_info(px, py, 16, 8, y_mode, !self.mono, None, None);
            self.emit_filter_intra(y_mode, 16, 8, None);
            self.code_tx_depth(px, py, 16, 8, 0);
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(y_mode as u8);
            self.l_mode[by4..by4 + 2].fill(y_mode as u8);
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        16,
                        ltxtp == 2,
                        sk,
                        ds,
                        y_mode,
                    )
                } else {
                    encode_16x8_luma_coeffs(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        sk,
                        ds,
                        y_mode,
                        ltxtp,
                    )
                }
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_pred(
                    &mut drow[..16],
                    &lpred_arr[ry * 16..],
                    &lrr[ry * 16..],
                    maxval,
                );
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
                    self.idct.idct_dequant_8x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    if use_cfl {
                        recon_add_pred(
                            &mut drow[..8],
                            &cfl_px[ci][ry * 8..],
                            &rr[ry * 8..],
                            maxval,
                        );
                    } else {
                        recon_add_dc(&mut drow[..8], cpred[ci], &rr[ry * 8..], maxval);
                    }
                }
            }
        }
    }

    fn code_block16_horz_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        // Two sub-blocks: half = 0 (top, py), half = 1 (bottom, py+8).
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4); // luma 4-unit coords
            // --- luma 16x8: mode search + 7-type transform trial ---
            let dc_l = self.intrapred.dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let (y_mode, lpred_arr, lresid, lcf) =
                self.rect16_luma_mode_search(px, py, false, dc_l, lam, emlam, true);
            let (ltxtp, lcf) = self.rect_leaf_tx_trial(
                &lresid, &lcf, &lpred_arr, px, py, false, y_mode, lam, emlam, true,
            );
            let idct = self.idct;
            let inv16x8 =
                |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, false, ltxtp);
            // --- chroma 16x8 (4:4:4): DC predict each plane ---
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = self.intrapred.dc_pred_16x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                self.rd.residual_dc(&mut resid, &self.src[plane], self.w, px, py, 16, 8, dc);
                let (mut q, qt) = self.dct.dct16x8_t(&resid, &self.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_16X8, lam, 16, 8, plane, px, py,
                );
                self.rd.preserve_dc(&mut q[0], &resid[..]);
                ccf[ci] = q;
            }
            // block_skip iff all planes have no coefficients.
            // --- CfL: predict chroma from this sub-block's reconstructed luma.
            // Allowed here (16x8 <= 32x32); at 4:4:4 chroma is full resolution
            // so chroma-from-luma is worth far more than a luma mode search.
            let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
            let mut cfl_ccf = [[0i32; 128]; 2];
            let mut cfl_pred = [[0i32; 128]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
            {
                let lrr_cfl = inv16x8(&lcf, &self.quant);
                let mut luma_rec = [0u16; 128];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
                let mut ac = [0i32; 128];
                self.intrapred.cfl_ac_444(&luma_rec, 16, 8, &mut ac);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut csrc = [0u16; 128];
                    self.rd
                        .copy_block_u16(&mut csrc, &self.src[plane], self.w, px, py, 16, 8);
                    let dcrr = self.idct.idct_dequant_16x8(&ccf[ci], &self.cquant);
                    let s = self.rd.sse_recon(
                        &[dc; 128], &dcrr, &csrc, 16, 0, 0, 16, 8, self.bd,
                    );
                    dc_cost[ci] = rd_cost_i64(
                        s,
                        mlam,
                        self.chroma_rect_bits(&ccf[ci], &SCAN_16X8, 16, 8, plane, px, py),
                    );
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &csrc, dc, 128, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 128];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..128], dc, a, self.bd);
                    let mut resid = [0i32; 128];
                    self.rd.residual_pred(&mut resid, &cpr, &csrc, 16, 0, 0, 16, 8);
                    let (mut q, qt) = self.dct.dct16x8_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, cdcq, cacq, &SCAN_16X8, lam, 16, 8, plane, px, py,
                    );
                    let rr2 = self.idct.idct_dequant_16x8(&q, &self.cquant);
                    let s2 = self.rd.sse_recon(&cpr, &rr2, &csrc, 16, 0, 0, 16, 8, self.bd);
                    cfl_ccf[ci] = q;
                    cfl_pred[ci] = cpr;
                    cfl_cost[ci] = rd_cost_i64(
                        s2,
                        mlam,
                        self.chroma_rect_bits(&q, &SCAN_16X8, 16, 8, plane, px, py),
                    );
                }
            }
            // CFL_PRED uv_mode symbol + one alpha symbol per non-zero plane.
            let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
            let mut use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                    < dc_cost[0]
                        + dc_cost[1]
                        + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None));
            if use_cfl {
                ccf = cfl_ccf;
            }
            // Rect-chroma equip: directional/smooth UV candidates compete with
            // the DC/CfL winner on the same R-D bar (mirrors NONE-16's search).
            let mut chosen_uv = DC_PRED;
            let mut uv_pred: Option<[[i32; 128]; 2]> = None;
            if self.speed.full_chroma_rdo() {
                let cur_total = if use_cfl {
                    cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                } else {
                    dc_cost[0]
                        + dc_cost[1]
                        + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None))
                };
                if let Some((m, c2, p2)) =
                    self.rect444_uv_mode_search(px, py, false, y_mode, cur_total, lam, mlam)
                {
                    chosen_uv = m;
                    ccf = c2;
                    uv_pred = Some(p2);
                    use_cfl = false;
                }
            }
            let cfl_opt = if use_cfl { Some(cfl_a) } else { None };
            let luma_zero = self.rd.all_zero_i32(&lcf);
            let chroma_zero =
                self.rd.all_zero_i32(&ccf[0]) && self.rd.all_zero_i32(&ccf[1]);
            let block_skip = luma_zero && chroma_zero;
            // --- header: skip, delta-q (once), y_mode (DC), uv_mode (DC) ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            // record the 16x8 footprint for the deblock filter: width 4 units,
            // height 2 units (vertical edges every 16, horizontal every 8).
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
            // Directional modes carry an angle_delta symbol (this search only
            // offers delta 0).
            if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
                self.enc
                    .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
            }
            self.emit_uv_mode(y_mode, chosen_uv, cfl_opt, px, py, 16, 8);
            self.emit_palette_mode_info(px, py, 16, 8, y_mode, !self.mono, None, None);
            self.emit_filter_intra(y_mode, 16, 8, None);
            self.code_tx_depth(px, py, 16, 8, 0);
            // footprint update: skip/mode over 4 wide x 2 tall units.
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(y_mode as u8);
            self.l_mode[by4..by4 + 2].fill(y_mode as u8);
            // --- luma coeffs (RTX_16X8) ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        16,
                        ltxtp == 2,
                        sk,
                        ds,
                        y_mode,
                    )
                } else {
                    encode_16x8_luma_coeffs(
                        &mut self.enc,
                        &mut self.cdfs,
                        &lcf,
                        sk,
                        ds,
                        y_mode,
                        ltxtp,
                    )
                }
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            // reconstruct luma
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_pred(
                    &mut drow[..16],
                    &lpred_arr[ry * 16..],
                    &lrr[ry * 16..],
                    maxval,
                );
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
                } else if chosen_uv != DC_PRED {
                    // Directional/smooth uv mode: decoder-derived chroma tx.
                    inv_chroma_16x8(
                        &self.idct,
                        chroma_tx_for_mode(chosen_uv),
                        &ccf[ci],
                        &self.cquant,
                    )
                } else {
                    self.idct.idct_dequant_16x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if let Some(up) = &uv_pred {
                        recon_add_pred(&mut drow[..16], &up[ci][ry * 16..], &rr[ry * 16..], maxval);
                    } else if use_cfl {
                        recon_add_pred(
                            &mut drow[..16],
                            &cfl_pred[ci][ry * 16..],
                            &rr[ry * 16..],
                            maxval,
                        );
                    } else {
                        recon_add_dc(&mut drow[..16], cpred[ci], &rr[ry * 16..], maxval);
                    }
                }
            }
        }
    }

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
        rd_lam: f32,
    ) -> ([i32; 256], [u16; 256], i64, f32, [u8; 4]) {
        let mut saved = self.sbuf_u256();
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        // Sub-transforms are coded in sequence, each seeing the previous one's
        // coefficient result context. Simulate that progression here and
        // restore it before returning (external review round 2, finding 3).
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 4] = self.a_coef[0][bx4_0..bx4_0 + 4].try_into().unwrap();
        let saved_l: [u8; 4] = self.l_coef[0][by4_0..by4_0 + 4].try_into().unwrap();
        let mut cf4 = self.sbuf_i256();
        let mut rec = self.sbuf_u256();
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let mut txtps = [1u8; 4];
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
                let d = self.intrapred.dc_pred_8x8(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 64];
            } else {
                self.intrapred.predict_nd_ad(
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
            self.rd.residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 8, 8);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 2, 2);
            let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut cf,
                &tf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                8,
                self.dcdf(),
                1,
                0,
                &self.dcdf().eob_bin_64_l,
                ds,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
            let dct_sse = sse_recon::<64, 8>(&self.rd, &pred, &rr, &self.src[0], self.w, bx, by, self.bd);
            let dct_bits =
                self.luma_bits_ctx_bounded(&cf, &SCAN_8X8, 8, bx, by, mode, 1, sk, ds, f32::INFINITY);
            let mut best = (cf, rr, dct_sse, dct_bits, 1u8);
            let quad_types: &[u8] = if self.ss420 || self.mono {
                &[]
            } else {
                &[4, 5, 6, 0, 2, 3]
            };
            for &txtp in quad_types {
                let (mut acf, atf) = match txtp {
                    4 => self.dct.adst8x8_t(&resid, &self.quant),
                    5 => self.dct.adstdct8x8_t(&resid, &self.quant),
                    6 => self.dct.dctadst8x8_t(&resid, &self.quant),
                    0 => self.dct.idtx8x8_t(&resid, &self.quant),
                    2 => self.dct.fvdct8x8_t(&resid, &self.quant),
                    _ => self.dct.fhdct8x8_t(&resid, &self.quant),
                };
                if txtp >= 4 {
                    trellis_optimize_ctx(
                        &mut acf,
                        &atf,
                        dcq,
                        acq,
                        &SCAN_8X8,
                        lam,
                        8,
                        8,
                        self.dcdf(),
                        1,
                        0,
                        &self.dcdf().eob_bin_64_l,
                        ds,
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let arr = match txtp {
                    4 => self.idct.iadst_dequant_8x8(&acf, &self.quant),
                    5 => self.idct.iadstdct_dequant_8x8(&acf, &self.quant),
                    6 => self.idct.idctadst_dequant_8x8(&acf, &self.quant),
                    0 => self.idct.iidentity_dequant_8x8(&acf, &self.quant),
                    2 => self.idct.ivdct_dequant_8x8(&acf, &self.quant),
                    _ => self.idct.ihdct_dequant_8x8(&acf, &self.quant),
                };
                let asse = sse_recon::<64, 8>(&self.rd, &pred, &arr, &self.src[0], self.w, bx, by, self.bd);
                // Gate BEFORE pricing: `abits` is only consumed by the
                // comparison below, so a failed SSE gate never needs the
                // rate. The exact-abort bound then caps the CDF walk at the
                // highest bit count that could still win (byte-identical —
                // see real_block_bits_bounded).
                let gate = if txtp >= 4 {
                    asse <= dct_sse + (dct_sse >> 5)
                } else {
                    asse <= best.2
                };
                if !gate {
                    continue;
                }
                let bits_bound = (rd_cost_i64(best.2, rd_lam, best.3) - asse as f32) / rd_lam;
                let abits = if txtp == 2 || txtp == 3 {
                    self.luma_bits_1d_8x8(&acf, txtp == 2, bx, by, mode)
                } else {
                    self.luma_bits_ctx_bounded(
                        &acf,
                        &SCAN_8X8,
                        8,
                        bx,
                        by,
                        mode,
                        txtp as usize,
                        sk,
                        ds,
                        bits_bound,
                    )
                };
                if rd_cost_i64(asse, rd_lam, abits) < rd_cost_i64(best.2, rd_lam, best.3) {
                    best = (acf, arr, asse, abits, txtp);
                }
            }
            let (bcf, brr, bsse, bbits, btxtp) = best;
            txtps[qi] = btxtp;
            sse_sum += bsse;
            bits_sum += bbits;
            let res_ctx = Self::coef_res_ctx(&bcf, &SCAN_8X8);
            self.a_coef[0][qbx4..qbx4 + 2].fill(res_ctx);
            self.l_coef[0][qby4..qby4 + 2].fill(res_ctx);
            // Write the quadrant's candidate recon so later quadrants predict
            // from it (restored below).
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                Some((&mut rec[sy * 16 + sx..], 16)),
                &pred,
                &brr,
                8,
                8,
                self.bd,
            );
            cf4[qi * 64..qi * 64 + 64].copy_from_slice(&bcf);
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        self.a_coef[0][bx4_0..bx4_0 + 4].copy_from_slice(&saved_a);
        self.l_coef[0][by4_0..by4_0 + 4].copy_from_slice(&saved_l);
        (*cf4, *rec, sse_sum, bits_sum, txtps)
    }

    /// Decoder-exact reconstruction of a depth-2 (sixteen TX_4X4) 16x16 from
    /// committed coefficients: raster per-TX prediction from the running
    /// recon with the grid edge-flag rule. Restores `self.recon`.
    #[allow(clippy::too_many_arguments)]
    fn split16_depth2_recon_from_cf(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lcf: &[i32; 256],
        txtps: [u8; 16],
    ) -> [u16; 256] {
        let mut saved = self.sbuf_u256();
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let block_ftype = self.luma_filter_type(px, py);
        let mut rec = self.sbuf_u256();
        for j in 0..4usize {
            for i in 0..4usize {
                let ti = j * 4 + i;
                let (bx, by) = (px + i * 4, py + j * 4);
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
                if mode == DC_PRED && delta == 0 {
                    let d = self.intrapred.dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    self.intrapred.predict_nd_ad(
                        mode,
                        delta,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        block_ftype,
                        &mut pred,
                        self.bd,
                    );
                }
                let cfq = lcf[ti * 16..ti * 16 + 16].first_chunk::<16>().unwrap();
                let rr = match txtps[ti] {
                    4 => self.idct.iadst_dequant_4x4(cfq, &self.quant),
                    0 => self.idct.iidentity_dequant_4x4(cfq, &self.quant),
                    2 => self.idct.ivdct_dequant_4x4(cfq, &self.quant),
                    3 => self.idct.ihdct_dequant_4x4(cfq, &self.quant),
                    _ => self.idct.idct_dequant_4x4(cfq, &self.quant),
                };
                self.rd.reconstruct(
                    &mut self.recon[0][by * self.w + bx..],
                    self.w,
                    Some((&mut rec[j * 4 * 16 + i * 4..], 16)),
                    &pred,
                    &rr,
                    4,
                    4,
                    self.bd,
                );
            }
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        *rec
    }

    /// Trial-code a 16x16 luma block as SIXTEEN TX_4X4 (`tx_depth = 2`),
    /// RASTER order (the decoder iterates the uniform TX grid row-major; the
    /// order matters for sequential prediction and coefficient contexts).
    /// Per-TX 4x4 tx-type candidates with the SPLIT4 gates. Edge flags per
    /// TX (i, j) on the 4-wide grid generalize the validated 2x2 rule.
    /// Temporarily writes candidate recon into `self.recon[0]`; restores.
    #[allow(clippy::too_many_arguments)]
    fn split16_depth2_try(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
        rd_lam: f32,
    ) -> ([i32; 256], [u16; 256], i64, f32, [u8; 16]) {
        let mut saved = self.sbuf_u256();
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let mut cf16 = self.sbuf_i256();
        let mut rec = self.sbuf_u256();
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let mut txtps = [1u8; 16];
        for j in 0..4usize {
            for i in 0..4usize {
                let ti = j * 4 + i;
                let (bx, by) = (px + i * 4, py + j * 4);
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
                if mode == DC_PRED && delta == 0 {
                    let d = self.intrapred.dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 16];
                } else {
                    self.intrapred.predict_nd_ad(
                        mode,
                        delta,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        4,
                        4,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        block_ftype,
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 16];
                self.rd.residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 4, 4);
                let (mut dcf, dtf) = self.dct.dct4x4_t(&resid, &self.quant);
                trellis_optimize(&mut dcf, &dtf, dcq, acq, &SCAN_4X4, lam);
                let drr = self.idct.idct_dequant_4x4(&dcf, &self.quant);
                let dct_sse =
                    sse_recon::<16, 4>(&self.rd, &pred, &drr, &self.src[0], self.w, bx, by, self.bd);
                let dct_bits = self.luma_bits(&dcf, &SCAN_4X4, 4, bx, by, mode, 1);
                let mut best = (dcf, drr, dct_sse, dct_bits, 1u8);
                for txtp in [4u8, 0, 2, 3] {
                    let (mut acf, atf) = match txtp {
                        4 => self.dct.adst4x4_t(&resid, &self.quant),
                        0 => self.dct.idtx4x4_t(&resid, &self.quant),
                        2 => self.dct.fvdct4x4_t(&resid, &self.quant),
                        _ => self.dct.fhdct4x4_t(&resid, &self.quant),
                    };
                    if txtp == 4 {
                        trellis_optimize(&mut acf, &atf, dcq, acq, &SCAN_4X4, lam);
                    }
                    let arr = match txtp {
                        4 => self.idct.iadst_dequant_4x4(&acf, &self.quant),
                        0 => self.idct.iidentity_dequant_4x4(&acf, &self.quant),
                        2 => self.idct.ivdct_dequant_4x4(&acf, &self.quant),
                        _ => self.idct.ihdct_dequant_4x4(&acf, &self.quant),
                    };
                    let asse =
                        sse_recon::<16, 4>(&self.rd, &pred, &arr, &self.src[0], self.w, bx, by, self.bd);
                    let bits_bound = (rd_cost_i64(best.2, rd_lam, best.3) - asse as f32) / rd_lam;
                    let abits = if txtp == 2 || txtp == 3 {
                        self.luma_bits_1d_4x4(&acf, txtp == 2, bx, by, mode)
                    } else {
                        self.luma_bits_bounded(
                            &acf,
                            &SCAN_4X4,
                            4,
                            bx,
                            by,
                            mode,
                            txtp as usize,
                            bits_bound,
                        )
                    };
                    if asse <= dct_sse + (dct_sse >> 5)
                        && rd_cost_i64(asse, rd_lam, abits) < rd_cost_i64(best.2, rd_lam, best.3)
                    {
                        best = (acf, arr, asse, abits, txtp);
                    }
                }
                let (bcf, brr, bsse, bbits, btxtp) = best;
                txtps[ti] = btxtp;
                sse_sum += bsse;
                bits_sum += bbits;
                self.rd.reconstruct(
                    &mut self.recon[0][by * self.w + bx..],
                    self.w,
                    Some((&mut rec[j * 4 * 16 + i * 4..], 16)),
                    &pred,
                    &brr,
                    4,
                    4,
                    self.bd,
                );
                cf16[ti * 16..ti * 16 + 16].copy_from_slice(&bcf);
            }
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        (*cf16, *rec, sse_sum, bits_sum, txtps)
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
        txtps: [u8; 4],
        d2_txtps: [u8; 16],
        depth2: bool,
    ) -> [u16; 256] {
        if depth2 {
            return self.split16_depth2_recon_from_cf(
                px, py, mode, delta, have_tr, have_bl, lcf, d2_txtps,
            );
        }
        let mut saved = self.sbuf_u256();
        for ry in 0..16 {
            saved[ry * 16..ry * 16 + 16]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..16]);
        }
        let block_ftype = self.luma_filter_type(px, py);
        let mut rec = self.sbuf_u256();
        for (qi, &(sx, sy)) in [(0usize, 0usize), (8, 0), (0, 8), (8, 8)]
            .iter()
            .enumerate()
        {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (8, 0) => (have_tr, false),
                (0, 8) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = [0i32; 64];
            if mode == DC_PRED {
                let d = self.intrapred.dc_pred_8x8(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 64];
            } else {
                self.intrapred.predict_nd_ad(
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
            let rr = match txtps[qi] {
                4 => self.idct.iadst_dequant_8x8(&cfq, &self.quant),
                5 => self.idct.iadstdct_dequant_8x8(&cfq, &self.quant),
                6 => self.idct.idctadst_dequant_8x8(&cfq, &self.quant),
                0 => self.idct.iidentity_dequant_8x8(&cfq, &self.quant),
                2 => self.idct.ivdct_dequant_8x8(&cfq, &self.quant),
                3 => self.idct.ihdct_dequant_8x8(&cfq, &self.quant),
                _ => self.idct.idct_dequant_8x8(&cfq, &self.quant),
            };
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                Some((&mut rec[sy * 16 + sx..], 16)),
                &pred,
                &rr,
                8,
                8,
                self.bd,
            );
        }
        for ry in 0..16 {
            self.recon[0][(py + ry) * self.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        *rec
    }

    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        // emit_prdo/emit_mlam already carry the perceptual scale; the extra
        // multiplication here ran square leaves at scale^2 while rect leaves,
        // 64s and the partition decision used scale^1 (external review
        // 2026-07-27, finding 3). `prdo` stays for the coupled chroma path.
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let mut best_mode = DC_PRED;
        let mut txtp16: u8 = 0; // 0=DCT_DCT 1=ADST_ADST 2=ADST_DCT 3=DCT_ADST
        let mut s8_txtps = [1u8; 4];
        let mut s16_txtps = [1u8; 16];
        let mut lpred_arr = self.sbuf_i256();
        let mut lcf = self.sbuf_i256();
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        let mut luma_beam: [Option<Luma16BeamCandidate>; JOINT_LARGE_BEAM] =
            std::array::from_fn(|_| None);
        let mut ltf = self.sbuf_f256(); // winner transform coeffs (f32, for winner-only RDOQ)
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
        let joint_large =
            rl.is_none() && !self.mono && self.speed == Speed::Slow && joint_luma_uv_large_enabled();
        let mode_shortlist = if rl.is_none() {
            self.rank_luma_modes::<256>(
                modes,
                px,
                py,
                16,
                16,
                have_tr,
                have_bl,
                self.luma_mode_budget_eff(),
            )
        } else {
            FixedList::new(DC_PRED)
        };
        for &m in modes {
            if rl.is_some() {
                break;
            }
            if !mode_shortlist.contains(&m) {
                continue;
            }
            let mut pred = self.sbuf_i256();
            if m == DC_PRED {
                let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                *pred = [d; 256];
            } else {
                self.intrapred.predict_nd(
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
                    &mut pred[..],
                    self.bd,
                );
            }
            let mut resid = self.sbuf_i256();
            self.rd.residual_pred(
                &mut resid[..],
                &pred[..],
                &self.src[0],
                self.w,
                px,
                py,
                16,
                16,
            );
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                sse_recon::<256, 16>(&self.rd, &pred, rr, &self.src[0], self.w, px, py, self.bd)
            };
            let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq_av1() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
            }
            let sse = blk_sse16(&self.idct.idct_dequant_16x16(&cf, &self.quant));
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
                *lpred_arr = *pred;
                *lcf = cf;
                *ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
                best_filter_intra = None;
            }
            if joint_large {
                let mut pos = JOINT_LARGE_BEAM;
                for (i, slot) in luma_beam.iter().enumerate() {
                    if slot.as_ref().is_none_or(|old| cost < old.luma_cost) {
                        pos = i;
                        break;
                    }
                }
                if pos < JOINT_LARGE_BEAM {
                    let mut beam_cf = self.sbuf_i256();
                    let mut beam_tf = self.sbuf_f256();
                    *beam_cf = cf;
                    *beam_tf = tf;
                    for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                        luma_beam[i] = luma_beam[i - 1].take();
                    }
                    luma_beam[pos] = Some(Luma16BeamCandidate {
                        luma_cost: cost,
                        mode: m,
                        pred,
                        cf: beam_cf,
                        tf: beam_tf,
                        sse,
                        bits,
                        palette: None,
                    });
                }
            }
        }
        // Exact-palette candidate: a 16x16 block holding <= 8 distinct sample
        // values is reproduced EXACTLY (zero residual, zero ringing) for just
        // the palette signaling cost — the decisive tool on synthetic/flat
        // content.
        let mut best_palette16: Option<LossyLumaPalette> = None;
        if rl.is_none()
            && self.try_palette()
            && let Some(hist) = block_color_histogram(&self.src[0], self.w, px, py, 16, 16)
        {
            for (palette, pred) in
                self.rank_luma_palette_candidates::<256>(&hist, px, py, 16, 16, mlam)
            {
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(
                    &mut resid[..],
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq_av1() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        16,
                        self.dcdf(),
                        2,
                        0,
                        &self.dcdf().eob_bin_256_l,
                        dcs16,
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
                let sse = sse_recon::<256, 16>(&self.rd, &pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let coeff_bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, DC_PRED, 1);
                let bits = coeff_bits
                    + self.mode_bits(px, py, DC_PRED)
                    + self.palette_rate_bits(px, py, &palette);
                let cost = rd_cost_i64(sse, mlam, bits);
                if cost < best_eff {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    best_filter_intra = None;
                    best_palette16 = Some(palette.clone());
                    *lpred_arr = pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = coeff_bits;
                }
                if joint_large {
                    let mut pos = JOINT_LARGE_BEAM;
                    for (i, slot) in luma_beam.iter().enumerate() {
                        if slot.as_ref().is_none_or(|old| cost < old.luma_cost) {
                            pos = i;
                            break;
                        }
                    }
                    if pos < JOINT_LARGE_BEAM {
                        let mut beam_pred = self.sbuf_i256();
                        let mut beam_cf = self.sbuf_i256();
                        let mut beam_tf = self.sbuf_f256();
                        *beam_pred = pred;
                        *beam_cf = cf;
                        *beam_tf = tf;
                        for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                            luma_beam[i] = luma_beam[i - 1].take();
                        }
                        luma_beam[pos] = Some(Luma16BeamCandidate {
                            luma_cost: cost,
                            mode: DC_PRED,
                            pred: beam_pred,
                            cf: beam_cf,
                            tf: beam_tf,
                            sse,
                            bits: coeff_bits,
                            palette: Some(palette.clone()),
                        });
                    }
                }
            }
        }
        if joint_large {
            let mut joint_best = f32::INFINITY;
            let mut selected = None;
            for candidate in luma_beam.into_iter().flatten() {
                let cost = candidate.luma_cost
                    + self.joint_uv_cost16(
                        &candidate.pred,
                        &candidate.cf,
                        candidate.mode,
                        px,
                        py,
                        prdo,
                    );
                if selected.is_none() || cost < joint_best * crate::tuning::get().joint_large_gain {
                    joint_best = cost;
                    selected = Some(candidate);
                }
            }
            if let Some(candidate) = selected {
                best_eff = candidate.luma_cost;
                best_mode = candidate.mode;
                *lpred_arr = *candidate.pred;
                *lcf = *candidate.cf;
                *ltf = *candidate.tf;
                best_dct_sse = candidate.sse;
                best_dct_bits = candidate.bits;
                best_filter_intra = None;
                best_palette16 = candidate.palette;
            }
        }
        if rl.is_none() && self.speed == Speed::Slow {
            let bsize = av1_block_size_index(16, 16);
            for &filter_mode in self
                .rank_filter_intra_modes::<256>(
                    px,
                    py,
                    16,
                    16,
                    self.speed.filter_intra_refine_budget(),
                )
                .iter()
            {
                let mut pred = self.sbuf_i256();
                self.intrapred.filter_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    &mut pred[..],
                    self.bd,
                );
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(
                    &mut resid[..],
                    &pred[..],
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
                let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
                let sse = sse_recon::<256, 16>(&self.rd, &pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, DC_PRED, 1);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if rl.is_some()
                    || raw_sse_guard_choice(
                        "filter16",
                        RawSseGuard::FilterIntra,
                        best_dct_sse,
                        sse,
                        best_eff,
                        cost,
                        sse <= best_dct_sse && cost < best_eff,
                    )
                {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    *lpred_arr = *pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                    best_filter_intra = Some(filter_mode);
                    best_palette16 = None;
                }
            }
        }
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
        if rl.is_none()
            && angle_delta_enabled()
            && self.speed.try_angle_deltas_av1(16, self.base_q_idx)
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
            let mut best_ad_cost =
                rd_cost_i64(best_dct_sse, mlam, best_dct_bits + cdf_cost(&ad_cdf, 3));
            let mut ad_pred0 = self.sbuf_i256();
            let mut ad_pred1 = self.sbuf_i256();
            let mut ad_scratch = self.sbuf_i256();
            let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
            for (di, &d) in self
                .rank_angle_deltas::<256>(
                    best_mode, px, py, 16, 16, have_tr, have_bl, 2, &mut ad_preds,
                )
                .iter()
                .enumerate()
            {
                let pred: &[i32; 256] = &*ad_preds[di];
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(
                    &mut resid[..],
                    &pred[..],
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq_av1() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        16,
                        self.dcdf(),
                        2,
                        0,
                        &self.dcdf().eob_bin_256_l,
                        dcs16,
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
                let sse = sse_recon::<256, 16>(&self.rd, pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, best_mode, 1);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if rl.is_some() || cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    *lpred_arr = *pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                }
            }
        }
        // Fast path: run RDOQ once, on the winning mode only (libaom
        // winner-mode coeff opt). The decision above used un-trellised costs.
        if rl.is_none() && !self.speed.per_candidate_rdoq_av1() {
            trellis_optimize_ctx(
                &mut lcf[..],
                &ltf[..],
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style).
        if rl.is_none() && best_palette16.is_none() && self.speed.try_adst() {
            let mut resid = self.sbuf_i256();
            self.rd.residual_pred(
                &mut resid[..],
                &lpred_arr[..],
                &self.src[0],
                self.w,
                px,
                py,
                16,
                16,
            );
            let (mut acf, atf) = self.dct.adst16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.iadst_dequant_16x16(&acf, &self.quant);
            let asse = sse_recon::<256, 16>(&self.rd, &lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let base_rd = rd_cost_i64(best_dct_sse, mlam, best_dct_bits);
            let bits_bound = if rl.is_some() {
                f32::INFINITY
            } else {
                (base_rd - asse as f32) / mlam
            };
            let abits = self.luma_bits_bounded(
                &acf,
                &SCAN_16X16,
                16,
                px,
                py,
                best_mode,
                ADST_ADST_TX16_IDX,
                bits_bound,
            );
            let candidate_rd = rd_cost_i64(asse, mlam, abits);
            if rl.is_some()
                || raw_sse_guard_choice(
                    "adst16",
                    RawSseGuard::TxType,
                    best_dct_sse,
                    asse,
                    base_rd,
                    candidate_rd,
                    asse <= best_dct_sse + (best_dct_sse >> 5) && candidate_rd < base_rd,
                )
            {
                *lcf = acf;
                txtp16 = 1;
            }
        }
        // Asymmetric-ADST refinement (ADST_DCT / DCT_ADST) for TX_16X16, same
        // rationale as the 8x8 path. Competes with the running tx winner.
        if rl.is_none() && best_palette16.is_none() && self.speed.try_adst() && asym_adst_enabled()
        {
            let mut best_txtp16_sse = if txtp16 == 1 { i64::MAX } else { best_dct_sse };
            let mut best_txtp16_bits = best_dct_bits;
            if txtp16 == 1 {
                // recompute the ADST_ADST winner cost as the bar to beat
                let rr = self.idct.iadst_dequant_16x16(&lcf, &self.quant);
                best_txtp16_sse =
                    sse_recon::<256, 16>(&self.rd, &lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                best_txtp16_bits = self.luma_bits(
                    &lcf[..],
                    &SCAN_16X16,
                    16,
                    px,
                    py,
                    best_mode,
                    ADST_ADST_TX16_IDX,
                );
            }
            for (fwd_dctadst, inv_dctadst) in [(false, false), (true, true)] {
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(
                    &mut resid[..],
                    &lpred_arr[..],
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut acf, atf) = if fwd_dctadst {
                    self.dct.dctadst16x16_t(&resid, &self.quant)
                } else {
                    self.dct.adstdct16x16_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
                let rr = if inv_dctadst {
                    self.idct.idctadst_dequant_16x16(&acf, &self.quant)
                } else {
                    self.idct.iadstdct_dequant_16x16(&acf, &self.quant)
                };
                let asse =
                    sse_recon::<256, 16>(&self.rd, &lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                let abits = self.luma_bits(
                    &acf,
                    &SCAN_16X16,
                    16,
                    px,
                    py,
                    best_mode,
                    if inv_dctadst {
                        DCT_ADST_TX16_IDX
                    } else {
                        ADST_DCT_TX16_IDX
                    },
                );
                let base_rd = rd_cost_i64(best_txtp16_sse, mlam, best_txtp16_bits);
                let candidate_rd = rd_cost_i64(asse, mlam, abits);
                if rl.is_some()
                    || raw_sse_guard_choice(
                        "asym-adst16",
                        RawSseGuard::TxType,
                        best_txtp16_sse,
                        asse,
                        base_rd,
                        candidate_rd,
                        asse <= best_dct_sse + (best_dct_sse >> 5) && candidate_rd < base_rd,
                    )
                {
                    *lcf = acf;
                    txtp16 = if inv_dctadst { 3 } else { 2 };
                    best_txtp16_sse = asse;
                    best_txtp16_bits = abits;
                }
            }
        }
        // Per-block IDTX refinement (mirror of the TX_8X8 one): the identity
        // transform wins on sharp edges / screen-content residuals that DCT
        // and ADST smear across many coefficients. One extra forward+inverse
        // on the winning prediction; no RDOQ (see the 8x8 IDTX note — the
        // full-RDOQ retry measured flat/negative), and the same
        // SSE-non-worsening guard as ADST so low-q lambda cannot trade real
        // detail for the cheap coefficients. IDTX = symbol 0 in the 5-type
        // DTT4_IDTX intra set at TX_16X16.
        if rl.is_none() && best_palette16.is_none() && self.speed.try_adst() {
            let mut resid = self.sbuf_i256();
            self.rd.residual_pred(
                &mut resid[..],
                &lpred_arr[..],
                &self.src[0],
                self.w,
                px,
                py,
                16,
                16,
            );
            let (icf, _itf) = self.dct.idtx16x16_t(&resid, &self.quant);
            let rr = self.idct.iidentity_dequant_16x16(&icf, &self.quant);
            let isse = sse_recon::<256, 16>(&self.rd, &lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let ibits = self.luma_bits(&icf, &SCAN_16X16, 16, px, py, best_mode, IDTX_TX16_IDX);
            // Current winner's (sse, bits) under its committed tx.
            let (cur_sse, cur_bits) = {
                let rrw = match txtp16 {
                    1 => self.idct.iadst_dequant_16x16(&lcf, &self.quant),
                    2 => self.idct.iadstdct_dequant_16x16(&lcf, &self.quant),
                    3 => self.idct.idctadst_dequant_16x16(&lcf, &self.quant),
                    _ => self.idct.idct_dequant_16x16(&lcf, &self.quant),
                };
                let sse =
                    sse_recon::<256, 16>(&self.rd, &lpred_arr, &rrw, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(
                    &lcf[..],
                    &SCAN_16X16,
                    16,
                    px,
                    py,
                    best_mode,
                    match txtp16 {
                        1 => ADST_ADST_TX16_IDX,
                        2 => ADST_DCT_TX16_IDX,
                        3 => DCT_ADST_TX16_IDX,
                        _ => 1,
                    },
                );
                (sse, bits)
            };
            let base_rd = rd_cost_i64(cur_sse, mlam, cur_bits);
            let candidate_rd = rd_cost_i64(isse, mlam, ibits);
            if raw_sse_guard_choice(
                "idtx16",
                RawSseGuard::TxType,
                cur_sse,
                isse,
                base_rd,
                candidate_rd,
                isse <= cur_sse + (cur_sse >> 5) && candidate_rd < base_rd,
            ) {
                *lcf = icf;
                txtp16 = 5;
            }
        }
        // TX split (tx_depth = 1): trial-code the winner mode as four TX_8X8
        // with per-sub-TX prediction. The old `best_delta == 0` gate denied
        // TX splitting to every directional winner carrying an angle delta,
        // though the helper handles deltas fine (external review round 2,
        // finding 8a): removing it measured holdout 420 -0.32 / 444 -0.09 /
        // 422 -0.04 for +3.5% Slow time.
        if rl.is_none() && best_filter_intra.is_none() && best_palette16.is_none() {
            // Final (distortion, rate) of the whole-TX_16X16 winner, from its
            // committed coefficients (the sub-search locals are scope-bound).
            let rr16 = match txtp16 {
                1 => self.idct.iadst_dequant_16x16(&lcf, &self.quant),
                2 => self.idct.iadstdct_dequant_16x16(&lcf, &self.quant),
                3 => self.idct.idctadst_dequant_16x16(&lcf, &self.quant),
                5 => self.idct.iidentity_dequant_16x16(&lcf, &self.quant),
                _ => self.idct.idct_dequant_16x16(&lcf, &self.quant),
            };
            let none_sse =
                sse_recon::<256, 16>(&self.rd, &lpred_arr, &rr16, &self.src[0], self.w, px, py, self.bd);
            let none_bits = self.luma_bits(
                &lcf[..],
                &SCAN_16X16,
                16,
                px,
                py,
                best_mode,
                match txtp16 {
                    1 => ADST_ADST_TX16_IDX,
                    2 => ADST_DCT_TX16_IDX,
                    3 => DCT_ADST_TX16_IDX,
                    5 => IDTX_TX16_IDX,
                    _ => 1, // DCT_DCT
                },
            );
            let (cf4, _rec, sse_s, bits_s, split_txtps) =
                self.split16_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam, mlam);
            let d0_bits = self.tx_depth_bits(px, py, 16, 16, 0);
            let d1_bits = self.tx_depth_bits(px, py, 16, 16, 1);
            let base_rd = rd_cost_i64(none_sse, mlam, none_bits + d0_bits);
            let candidate_rd = rd_cost_i64(sse_s, mlam, bits_s + d1_bits);
            let guarded_take = if self.banding_risk(px, py, 16) {
                sse_s <= none_sse + (none_sse >> 2)
            } else {
                candidate_rd < base_rd
            };
            let take = raw_sse_guard_choice(
                "split-tx16",
                RawSseGuard::TxSplit,
                none_sse,
                sse_s,
                base_rd,
                candidate_rd,
                guarded_take,
            );
            if take {
                txtp16 = 4;
                s8_txtps = split_txtps;
                *lcf = cf4;
            }
            // Depth-2 candidate: sixteen TX_4X4. Competes against the current
            // winner (whole or depth-1) on plain RD; ~2.5 bit allowance for
            // the deeper tx_depth symbol.
            let (cf16, _rec2, sse_2, bits_2, d2_txtps) =
                self.split16_depth2_try(px, py, best_mode, best_delta, have_tr, have_bl, lam, mlam);
            let cur_rd = if txtp16 == 4 {
                rd_cost_i64(sse_s, mlam, bits_s + d1_bits)
            } else {
                base_rd
            };
            let d2_rd = rd_cost_i64(
                sse_2,
                mlam,
                bits_2 + self.tx_depth_bits(px, py, 16, 16, 2),
            );
            if d2_rd < cur_rd {
                txtp16 = 6;
                s16_txtps = d2_txtps;
                *lcf = cf16;
            }
        }
        // Pure-emit replay: install the recorded winner and its captured
        // post-trellis coefficients (every luma sub-search above was skipped).
        if let Some(r) = rl {
            best_mode = r.mode as usize;
            best_delta = r.delta as i32;
            if r.palette > 0 {
                let p = lossy_luma_palette(
                    &self.kmeans,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    r.palette as usize,
                )
                .expect("16x16 palette replay: candidate no longer derivable");
                debug_assert_eq!(
                    p.colors.len() + if p.top { 8 } else { 0 },
                    r.palette as usize
                );
                palette_pred(&mut lpred_arr[..], 16, &p.colors, &p.packed_map, 16, 16);
                best_palette16 = Some(p);
            }
            best_filter_intra = FILTER_INTRA_MODES
                .iter()
                .copied()
                .find(|&f| f as u8 == r.filter);
            txtp16 = match r.tx {
                TxSel::Adst => 1,
                TxSel::AdstDct => 2,
                TxSel::DctAdst => 3,
                TxSel::SplitDct(t) => {
                    s8_txtps = t;
                    4
                }
                TxSel::Split16Tx(t) => {
                    s16_txtps = t;
                    6
                }
                TxSel::Idtx => 5,
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
        } else if txtp16 == 6 {
            // Depth-2: sixteen 4x4 TX cells (SPLIT4 map pattern).
            let nc4 = self.w / 4;
            let (bx4, by4) = (x8 * 2, y8 * 2);
            for uy in 0..4 {
                for ux in 0..4 {
                    let cell = (by4 + uy) * nc4 + (bx4 + ux);
                    self.blk4[cell] = 1;
                    self.blk4h[cell] = 1;
                    self.blk4v[cell] = true;
                    self.blk4t[cell] = true;
                }
            }
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            palette: best_palette16
                .as_ref()
                .map_or(0, |p| (p.colors.len() + if p.top { 8 } else { 0 }) as u8),
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            tx: if txtp16 == 6 {
                TxSel::Split16Tx(s16_txtps)
            } else if txtp16 == 4 {
                TxSel::SplitDct(s8_txtps)
            } else {
                TxSel::from_flags(txtp16 == 1, txtp16 == 5, txtp16 == 2, txtp16 == 3)
            },
        });
        self.push_luma_cf(&lcf[..]);
        let luma_zero = self.rd.all_zero_i32(&lcf[..]);
        if self.ss420 {
            self.code_block16_420(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                txtp16,
                s8_txtps,
                s16_txtps,
                best_delta,
                best_filter_intra,
                best_palette16.as_ref(),
                have_tr,
                have_bl,
            );
        } else if self.ss422 {
            self.code_block16_422(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                txtp16,
                s8_txtps,
                s16_txtps,
                best_delta,
                best_filter_intra,
                best_palette16.as_ref(),
                have_tr,
                have_bl,
            );
        } else {
            self.code_block16_444(
                x8,
                y8,
                &lcf,
                &lpred_arr,
                best_mode,
                luma_zero,
                txtp16,
                s8_txtps,
                s16_txtps,
                best_delta,
                best_filter_intra,
                best_palette16.as_ref(),
                have_tr,
                have_bl,
            );
        }
    }
}
