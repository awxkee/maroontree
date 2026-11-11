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
/// Pseudo txtp for a rect16 luma leaf coded as two TX_8X8 (tx_depth 1);
/// `cf` then holds the two 8x8 coefficient blocks half-major (first half in
/// `cf[..64]`, second in `cf[64..]`), see [`rect_half_offset`].
const RECT_SPLIT_TXTP: usize = 7;

/// A rect16 leaf's luma winner: (mode, angle delta, prediction, residual,
/// coefficients, palette, filter-intra mode).
type RectLumaWin = (
    usize,
    i32,
    [i32; 128],
    [i32; 128],
    [i32; 128],
    Option<LossyLumaPalette>,
    Option<FilterIntraMode>,
);

/// Luma offset of half `i` of a rect16 leaf: HORZ halves sit side by side,
/// VERT halves are stacked. Decode order is half 0 then half 1.
fn rect_half_offset(vert: bool, i: usize) -> (usize, usize) {
    if vert {
        (0, 8 * i)
    } else {
        (8 * i, 0)
    }
}

/// Extract half `i` (8x8, row-major) of a row-major rect16 luma block.
fn rect_half_i32(src: &[i32; 128], vert: bool, i: usize) -> [i32; 64] {
    let mut out = [0i32; 64];
    if vert {
        out.copy_from_slice(&src[64 * i..64 * i + 64]);
    } else {
        for r in 0..8 {
            out[r * 8..r * 8 + 8].copy_from_slice(&src[r * 16 + 8 * i..r * 16 + 8 * i + 8]);
        }
    }
    out
}

/// Scatter an 8x8 half back into a row-major rect16 block.
fn rect_half_place(dst: &mut [i32; 128], half: &[i32; 64], vert: bool, i: usize) {
    if vert {
        dst[64 * i..64 * i + 64].copy_from_slice(half);
    } else {
        for r in 0..8 {
            dst[r * 16 + 8 * i..r * 16 + 8 * i + 8].copy_from_slice(&half[r * 8..r * 8 + 8]);
        }
    }
}

/// Inverse + dequant for a rect16 luma transform by 7-type txtp symbol
/// (0 = IDTX, 1 = DCT_DCT, 2 = V_DCT, 3 = H_DCT, 4 = ADST_ADST,
/// 5 = ADST_DCT, 6 = DCT_ADST; [`RECT_SPLIT_TXTP`] = two TX_8X8 DCT halves,
/// an ESTIMATE only — the emitter predicts each half from the running
/// reconstruction). `vert` selects RTX_8X16 vs RTX_16X8.
fn inv_rect_luma_128(
    idct: &IdctDispatch,
    cf: &[i32; 128],
    q: &Quant,
    vert: bool,
    txtp: usize,
) -> [i32; 128] {
    if txtp == RECT_SPLIT_TXTP {
        let mut out = [0i32; 128];
        for i in 0..2 {
            let half = rect_half_i32(cf, true, i);
            let rr = idct.idct_dequant_8x8(&half, q);
            rect_half_place(&mut out, &rr, vert, i);
        }
        return out;
    }
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

fn proxy_candidate_rdoq(speed: Speed) -> bool {
    !matches!(speed, Speed::Fast)
}

/// Margin for the 8x8 SPLIT4 breakout, the [`split_breakout_k`] analogue one
/// level down. `rd_cost_split4_luma` prices an 8x8 as four 4x4s and is the
/// single largest source of 4x4 transform+trellis work (4x4 is ~60% of all
/// trellis calls), so skipping it when the source says the split cannot pay
/// removes real work. `INFINITY` disables it.
fn split4_breakout_k(speed: Speed) -> f32 {
    match speed {
        Speed::Slow | Speed::ExtraSlow => crate::tuning::get().split4_breakout_slow,
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
        Speed::Slow | Speed::ExtraSlow => crate::tuning::get().split_breakout_slow,
        Speed::Medium | Speed::Fast => 1.0,
    }
}

/// Ranked palette candidates the proxy fully prices (capped at 3).
fn palette_proxy_finalists() -> usize {
    crate::tuning::get().palette_proxy_finalists.min(3)
}

fn proxy_mode_beam_len(speed: Speed, _dim: usize) -> usize {
    match speed {
        Speed::Fast => 1,
        Speed::Medium => 2,
        Speed::Slow | Speed::ExtraSlow => 2,
    }
}

struct Luma16BeamCandidate {
    luma_cost: f32,
    mode: usize,
    delta: i32,
    pred: SBuf<[i32; 256]>,
    cf: SBuf<[i32; 256]>,
    tf: SBuf<[f32; 256]>,
    sse: i64,
    bits: f32,
    palette: Option<LossyLumaPalette>,
}

impl<'a> LossyTile<'a> {
    fn proxy_mode_beam<const N: usize>(
        &self,
        px: usize,
        py: usize,
        dim: usize,
        have_tr: bool,
        have_bl: bool,
        pred: &mut [i32; N],
    ) -> FixedList<usize, 13> {
        // Slow 4:4:4 keeps a FIXED {DC, PAETH} pair (see the note below), so
        // the SATD ranking would only be thrown away; return before paying
        // for eleven predictions + proxies. Callers regenerate the winners'
        // predictions themselves, so `pred` need not be filled.
        if proxy_mode_beam_len(self.cfg.speed, dim) < 3
            && self.cfg.speed.at_least_slow()
            && !self.cfg.is_420()
            && !self.cfg.is_422()
        {
            let mut keep = FixedList::new(DC_PRED);
            keep.push(DC_PRED);
            keep.push(PAETH_PRED);
            return keep;
        }
        let mut ranked = FixedList::<(u64, usize), 13>::new((0, DC_PRED));
        for &m in nd_modes() {
            if m == DC_PRED {
                let d = self.k.intrapred.dc_pred(&self.ctx.recon[0], self.cfg.w, px, py, dim, dim, self.cfg.bd as i32);
                pred.fill(d);
            } else {
                self.k.intrapred.predict_nd(
                    m,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    dim,
                    dim,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(px, py),
                    pred,
                    self.cfg.bd,
                );
            }
            let score = self.k.rd.satd_sad_proxy(self.src_blk(0, px, py, dim, dim), pred, dim);
            ranked.push((score, m));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, mode)| (score, mode));
        let beam_len = proxy_mode_beam_len(self.cfg.speed, dim);
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
            if self.cfg.speed.at_least_slow() && !self.cfg.is_420() && !self.cfg.is_422() {
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

    /// 16x16 angle-delta candidates: the two SATD-ranked deltas (shipped) or
    /// all six under `ad16_full`, each with its prediction.
    fn ad16_candidates(
        &self,
        mode: usize,
        px: usize,
        py: usize,
        have_tr: bool,
        have_bl: bool,
        ad_preds: &mut [&mut [i32; 256]; 3],
    ) -> Vec<(i32, SBuf<[i32; 256]>)> {
        let mut out = Vec::with_capacity(6);
        if crate::tuning::get().ad16_full {
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut b = self.sbuf_i256();
                self.k.intrapred.predict_nd_ad(
                    mode,
                    d,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(px, py),
                    &mut b[..],
                    self.cfg.bd,
                );
                out.push((d, b));
            }
            return out;
        }
        for (di, &d) in self
            .rank_angle_deltas::<256>(mode, px, py, 16, 16, have_tr, have_bl, 2, ad_preds)
            .iter()
            .enumerate()
        {
            let mut b = self.sbuf_i256();
            *b = *ad_preds[di];
            out.push((d, b));
        }
        out
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
        let acq = self.ctx.quant.ac_q() as f32;
        let dcq = self.ctx.quant.dc_q() as f32;
        let lam = self.trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best = f32::INFINITY;
        if self.try_palette() {
            let key = ((px as u64) << 34) | ((py as u64) << 8) | dim as u64;
            let context = self.palette_estimate_context(px, py, mlam);
            let cached = if let Some(context) = context {
                self.memo.screen_pal_est_cache
                    .borrow()
                    .get(&key)
                    .filter(|(saved, _)| *saved == context)
                    .map(|&(_, pairs)| pairs)
            } else {
                self.memo.pal_est_cache.borrow().get(&key).copied()
            };
            let pairs = if let Some(pairs) = cached {
                pairs
            } else {
                let mut pairs = [(f32::INFINITY, 0.0f32); 3];
                let mut tried = 0usize;
                let hist = block_color_histogram(&self.cfg.src[0], self.cfg.w, px, py, dim, dim);
                let ranked = if crate::tuning::get().palette_proxy_ranked
                    && (self.cfg.is_420() || self.cfg.is_422())
                {
                    hist.as_deref()
                        .map(|hh| self.rank_palette_centers(hh, mlam, (px, py, dim, dim)))
                } else {
                    None
                };
                let plan: Vec<(usize, bool, Option<FixedList<i32, 8>>)> = match &ranked {
                    Some(r) => r
                        .iter()
                        .take(palette_proxy_finalists())
                        .map(|&(_, order, centers, top)| (order / 2 + 2, top, Some(centers)))
                        .collect(),
                    None => vec![(8, false, None), (4, false, None), (2, false, None)],
                };
                for (n, top, centers) in plan {
                    if tried >= 3 {
                        break;
                    }
                    let Some(p) = hist.as_deref().and_then(|hh| match &centers {
                        Some(c) => Some(lossy_luma_palette_from_centers(
                            &self.k.kmeans,
                            &self.cfg.src[0],
                            self.cfg.w,
                            px,
                            py,
                            dim,
                            dim,
                            c.as_slice().to_vec(),
                            n + if top { 8 } else { 0 },
                            &self.palette_map_opts_proxy(),
                        )),
                        None => lossy_luma_palette_from(
                            &self.k.kmeans,
                            hh,
                            &self.cfg.src[0],
                            self.cfg.w,
                            px,
                            py,
                            dim,
                            dim,
                            n,
                            top,
                            &self.palette_map_opts_proxy(),
                        ),
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
                        self.ctx.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &[],
                    );
                    let bits = self.mode_bits(px, py, DC_PRED) + self.palette_rate_bits(px, py, &p);
                    pairs[tried] = (dist, bits);
                    tried += 1;
                }
                if let Some(context) = context {
                    self.memo.screen_pal_est_cache
                        .borrow_mut()
                        .insert(key, (context, pairs));
                } else {
                    self.memo.pal_est_cache.borrow_mut().insert(key, pairs);
                }
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
                let beam = {
                    let mut pred = self.sbuf_i64();
                    self.proxy_mode_beam::<64>(px, py, 8, have_tr, have_bl, &mut pred)
                };
                let modes = &beam;
                let scan = &SCAN_8X8;
                for &m in modes {
                    let mut pred = [0i32; 64];
                    if m == DC_PRED {
                        let d = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                        pred = [d; 64];
                    } else {
                        self.k.intrapred.predict_nd(
                            m,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            px,
                            py,
                            8,
                            8,
                            have_tr,
                            have_bl,
                            self.cfg.w,
                            self.cfg.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.cfg.bd,
                        );
                    }
                    let mut resid = [0i32; 64];
                    self.k.rd.residual_pred(
                        &mut resid,
                        &pred[..],
                        &self.cfg.src[0],
                        self.cfg.w,
                        px,
                        py,
                        8,
                        8,
                    );
                    let (mut cf, tf) = self.k.dct.dct8x8_t(&resid, &self.ctx.quant);
                    if proxy_candidate_rdoq(self.cfg.speed) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.k.idct.idct_dequant_8x8(&cf, &self.ctx.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        8,
                        8,
                        self.ctx.quant.ac_q() as f32,
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
                // `proxy_ad`: price the winner's angle deltas too. The emitted
                // 8x8 leaf refines deltas, so a delta-less proxy undervalues
                // the 8x8 children of every edge 16x16 (same hole as split4).
                if crate::tuning::get().proxy_ad
                    && let Some((m, _, _)) = win
                    && self.angle_delta_refines(m)
                {
                    let ad_cdf = &self.dcdf().angle_delta[m - V_PRED];
                    for d in [-3i32, -2, -1, 1, 2, 3] {
                        let mut pred = [0i32; 64];
                        self.k.intrapred.predict_nd_ad(
                            m,
                            d,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            px,
                            py,
                            8,
                            8,
                            have_tr,
                            have_bl,
                            self.cfg.w,
                            self.cfg.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.cfg.bd,
                        );
                        let mut resid = [0i32; 64];
                        self.k.rd.residual_pred(&mut resid, &pred[..], &self.cfg.src[0], self.cfg.w, px, py, 8, 8);
                        let (mut cf, tf) = self.k.dct.dct8x8_t(&resid, &self.ctx.quant);
                        if proxy_candidate_rdoq(self.cfg.speed) {
                            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        }
                        let rr = self.k.idct.idct_dequant_8x8(&cf, &self.ctx.quant);
                        let distortion = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.ctx.quant.ac_q() as f32,
                            &pred,
                            0,
                            &rr[..],
                        );
                        let bits = self.luma_bits(&cf, scan, 8, px, py, m, 1)
                            + self.mode_bits(px, py, m)
                            + cdf_cost(ad_cdf, (d + 3) as usize);
                        let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                        if cost < best {
                            best = cost;
                            win = Some((m, pred, resid));
                        }
                    }
                }
                // Transform refinement on the winner, mirroring the final
                // block's winner-only ADST/IDTX pass. Without it the proxy
                // prices every leaf as DCT-only while the real block may code it
                // far cheaper, so leaves that respond well to ADST are
                // systematically undervalued in the partition decision.
                if self.cfg.speed.try_adst() && let Some((m, pred, resid)) = win
                {
                    for txtp in [ADST_ADST_TX8_IDX, 0] {
                        let (mut cf, tf) = if txtp == ADST_ADST_TX8_IDX {
                            self.k.dct.adst8x8_t(&resid, &self.ctx.quant)
                        } else {
                            self.k.dct.idtx8x8_t(&resid, &self.ctx.quant)
                        };
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        let rr = if txtp == ADST_ADST_TX8_IDX {
                            self.k.idct.iadst_dequant_8x8(&cf, &self.ctx.quant)
                        } else {
                            self.k.idct.iidentity_dequant_8x8(&cf, &self.ctx.quant)
                        };
                        let d = self.luma_partition_distortion(
                            px,
                            py,
                            8,
                            8,
                            self.ctx.quant.ac_q() as f32,
                            &pred[..],
                            0,
                            &rr[..],
                        );
                        let b = self.luma_bits(&cf, scan, 8, px, py, m, txtp)
                            + self.mode_bits(px, py, m);
                        best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                    }
                }
            }
            16 => {
                let coupled =
                    !self.cfg.is_mono() && self.cfg.speed.at_least_slow() && joint_luma_uv_proxy_enabled();
                let beam = {
                    let mut pred = self.sbuf_i256();
                    self.proxy_mode_beam::<256>(px, py, 16, have_tr, have_bl, &mut pred)
                };
                let modes = &beam;
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
                        let d = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                        *pred = [d; 256];
                    } else {
                        self.k.intrapred.predict_nd(
                            m,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            px,
                            py,
                            16,
                            16,
                            have_tr,
                            have_bl,
                            self.cfg.w,
                            self.cfg.h,
                            self.luma_filter_type(px, py),
                            &mut pred[..],
                            self.cfg.bd,
                        );
                    }
                    let mut resid = self.sbuf_i256();
                    self.k.rd.residual_pred(
                        &mut resid[..],
                        &pred[..],
                        &self.cfg.src[0],
                        self.cfg.w,
                        px,
                        py,
                        16,
                        16,
                    );
                    let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
                    if proxy_candidate_rdoq(self.cfg.speed) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.ctx.quant.ac_q() as f32,
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
                // `proxy_ad` (16x16): the winner's six angle deltas, priced
                // like the loop above (plus the joint chroma term when coupled).
                // (joint beam is const-off; when coupled the delta pass is skipped.)
                let mut ad16: Option<(f32, usize, [i32; 256], [i32; 256])> = None;
                if crate::tuning::get().proxy_ad
                    && !coupled
                    && let Some((m, _, _)) = win16
                    && self.angle_delta_refines(m)
                {
                    let ad_cdf = &self.dcdf().angle_delta[m - V_PRED];
                    for d in [-3i32, -2, -1, 1, 2, 3] {
                        let mut pred = [0i32; 256];
                        self.k.intrapred.predict_nd_ad(
                            m,
                            d,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            px,
                            py,
                            16,
                            16,
                            have_tr,
                            have_bl,
                            self.cfg.w,
                            self.cfg.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.cfg.bd,
                        );
                        let mut resid = [0i32; 256];
                        self.k.rd.residual_pred(&mut resid, &pred[..], &self.cfg.src[0], self.cfg.w, px, py, 16, 16);
                        let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
                        if proxy_candidate_rdoq(self.cfg.speed) {
                            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                        }
                        let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                        let distortion = self.luma_partition_distortion(
                            px,
                            py,
                            16,
                            16,
                            self.ctx.quant.ac_q() as f32,
                            &pred[..],
                            0,
                            &rr[..],
                        );
                        let bits = self.luma_bits(&cf, scan, 16, px, py, m, 1)
                            + self.mode_bits(px, py, m)
                            + cdf_cost(ad_cdf, (d + 3) as usize);
                        let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                        if ad16.as_ref().is_none_or(|a| cost < a.0) {
                            ad16 = Some((cost, m, pred, resid));
                        }
                    }
                    if let Some((cost, m, pred, resid)) = ad16.as_ref()
                        && *cost < best
                    {
                        best = *cost;
                        win16 = Some((*m, *pred, *resid));
                    }
                }
                if coupled {
                    let mut joint_best = f32::INFINITY;
                    for candidate in joint_beam.into_iter().flatten() {
                        let score = candidate.0
                            + self.proxy_wsse(self.joint_uv_cost16(
                                &candidate.2,
                                &candidate.4,
                                candidate.1,
                                px,
                                py,
                                prdo,
                            ));
                        if !joint_best.is_finite() || score < joint_best * crate::tuning::get().joint_large_gain {
                            joint_best = score;
                        }
                    }
                    best = joint_best;
                }
                if self.cfg.speed.try_adst() && !coupled && let Some((m, pred, resid)) = win16
                {
                    let (mut cf, tf) = self.k.dct.adst16x16_t(&resid, &self.ctx.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = self.k.idct.iadst_dequant_16x16(&cf, &self.ctx.quant);
                    let d = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.ctx.quant.ac_q() as f32,
                        &pred[..],
                        0,
                        &rr[..],
                    );
                    let b = self.luma_bits(&cf, scan, 16, px, py, m, ADST_ADST_TX16_IDX)
                        + self.mode_bits(px, py, m);
                    best = best.min(crate::partition_rd::rd_cost(d, mlam, b));
                }
            }
            32 => {
                let coupled =
                    !self.cfg.is_mono() && self.cfg.speed.at_least_slow() && joint_luma_uv_proxy_enabled();
                let beam = {
                    let mut pred = self.sbuf_i1024();
                    self.proxy_mode_beam::<1024>(px, py, 32, have_tr, have_bl, &mut pred)
                };
                let modes = &beam;
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
                        let d = self.k.intrapred.dc_pred_32x32(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                        *pred = [d; 1024];
                    } else {
                        self.k.intrapred.predict_nd(
                            m,
                            &self.ctx.recon[0],
                            self.cfg.w,
                            px,
                            py,
                            32,
                            32,
                            have_tr,
                            have_bl,
                            self.cfg.w,
                            self.cfg.h,
                            self.luma_filter_type(px, py),
                            &mut pred[..],
                            self.cfg.bd,
                        );
                    }
                    let mut resid = self.sbuf_i1024();
                    self.k.rd.residual_pred(
                        &mut resid[..],
                        &pred[..],
                        &self.cfg.src[0],
                        self.cfg.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    let (mut cf, tf) = self.k.dct.dct32x32_t(&resid, &self.ctx.quant);
                    if proxy_candidate_rdoq(self.cfg.speed) {
                        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    }
                    let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        32,
                        32,
                        self.ctx.quant.ac_q() as f32,
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
                        let score =
                            cost + self.proxy_wsse(self.joint_uv_cost32(&pred, &cf, mode, px, py, prdo));
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
                self.block_skip_bits(px, py, false),
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn rd_cost_rect16_leaf_with_dc(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
        dc: i32,
    ) -> f32 {
        let mlam = self.mlam() * prdo;
        // Epoch-stamped memo: the same leaf is re-priced by HORZ/VERT, the
        // A/B legs and the x3 bottom-up recompute. The (distortion, bits) pair
        // is NOT lambda-free — the cached coefficients were trellised at
        // `self.trellis_lambda() * prdo` — so prdo is part of the key (external
        // review round 2, finding 7).
        let key = (
            ((px as u64) << 40)
                | ((py as u64) << 20)
                | ((dc as u64) << 1)
                | vert as u64
                | ((have_tr as u64) << 62)
                | ((have_bl as u64) << 63),
            prdo.to_bits(),
        );
        let epoch = self.memo.emit_epoch.get();
        if let Some(&(e, dist, bits)) = self.memo.rect_leaf_cache.borrow().get(&key)
            && e == epoch
        {
            return crate::partition_rd::rd_cost(dist, mlam, bits);
        }
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        // Use the baseline DC/DCT representation for partition staging. The
        // selected partition receives the unchanged full mode/transform
        // refinement in its emitter; doing both here duplicated winner work.
        let dlam = self.trellis_lambda() * prdo;
        // The mode / TX selection inside the leaf is a wSSE decision
        // (`lambda_d`); only the returned leaf is priced on SATD with `mlam`.
        let lam_d = self.lambda_d() * prdo;
        let (y_mode, y_delta, pred, resid, cf, rpal, rfi) = self.rect16_luma_mode_search(
            px,
            py,
            vert,
            have_tr,
            have_bl,
            dc,
            dlam,
            lam_d,
            rect_dec_refine(),
        );
        let tx_mode = filter_intra_tx_mode(rfi, y_mode);
        let (txtp, cf) = self.rect_leaf_tx_trial(
            &resid,
            &cf,
            &pred,
            px,
            py,
            vert,
            tx_mode,
            dlam,
            lam_d,
            rect_dec_refine(),
            rpal.is_some(),
            self.rect_split_allowed() && rfi.is_none() && (crate::tuning::get().rect_full_modes || Self::rect_tbfree(y_mode, y_delta)),
        );
        let rr = inv_rect_luma_128(&self.k.idct, &cf, &self.ctx.quant, vert, txtp);
        let distortion = self.luma_partition_distortion(
            px,
            py,
            w,
            h,
            self.ctx.quant.ac_q() as f32,
            &pred[..],
            0,
            &rr[..],
        );
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let mut bits = cdf_cost(&self.dcdf().kf_y[yctx], y_mode);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            bits += cdf_cost(&self.dcdf().angle_delta[y_mode - V_PRED], 3);
        }
        if let Some(p) = rpal.as_ref() {
            bits += self.palette_rate_bits(px, py, p);
        }
        if let Some(f) = rfi {
            bits += cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(w, h)], 1)
                + cdf_cost(&self.dcdf().filter_intra_mode, f as usize);
        }
        bits += if txtp == 2 || txtp == 3 {
            self.luma_rect_bits_1d(&cf, w, txtp == 2, px, py, tx_mode)
        } else if txtp == RECT_SPLIT_TXTP {
            self.rect_split_bits(&cf, vert, px, py, y_mode)
        } else {
            self.luma_rect_bits(&cf, scan, w, h, px, py, tx_mode, txtp)
        };
        self.memo.rect_leaf_cache
            .borrow_mut()
            .insert(key, (epoch, distortion, bits));
        crate::partition_rd::rd_cost(distortion, mlam, bits)
    }

    fn rd_cost_rect16_leaf(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) -> f32 {
        let dc = if vert {
            self.k.intrapred.dc_pred_8x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
        } else {
            self.k.intrapred.dc_pred_16x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
        };
        self.rd_cost_rect16_leaf_with_dc(px, py, vert, have_tr, have_bl, prdo, dc)
    }

    /// Reprice a rectangular leaf whose top or left edge belongs to an earlier
    /// sibling in the same asymmetric partition. The normal RDO view still has
    /// stale pixels there; source samples are a close proxy for the sibling's
    /// eventual reconstruction and, by taking the worst cost, prevent a stale
    /// zero edge from making the partition look artificially cheap.
    #[allow(clippy::too_many_arguments)]
    fn rd_cost_rect16_dependent(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        source_above: bool,
        source_left: bool,
        prdo: f32,
    ) -> f32 {
        let base = self.rd_cost_rect16_leaf(px, py, vert, have_tr, have_bl, prdo);
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let above = py > 0;
        let left = px > 0;
        let mut sum = 0i32;
        let mut count = 0i32;
        if above {
            let plane = if source_above {
                &self.cfg.src[0]
            } else {
                &self.ctx.recon[0]
            };
            sum += self
                .k.rd
                .sum_u16(&plane[(py - 1) * self.cfg.w + px..][..w]);
            count += w as i32;
        }
        if left {
            let plane = if source_left {
                &self.cfg.src[0]
            } else {
                &self.ctx.recon[0]
            };
            sum +=
                self.k.rd
                    .sum_u16_strided(&plane[py * self.cfg.w + px - 1..], self.cfg.w, h);
            count += h as i32;
        }
        let dc = if count == 0 {
            1 << (self.cfg.bd - 1)
        } else {
            (sum + count / 2) / count
        };
        base.max(self.rd_cost_rect16_leaf_with_dc(px, py, vert, have_tr, have_bl, prdo, dc))
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
    /// Decoder-exact (`have_tr`, `have_bl`) of a rect16 leaf, dav1d
    /// `init_edges` / `decode_sb`: HORZ top = (node tr, true), HORZ bottom =
    /// (false, node bl), VERT left = (true, node bl), VERT right = (node tr,
    /// false); the T-shapes' rect part carries the same half's flags
    /// (HORZ_A bottom = h[1], HORZ_B top = h[0], VERT_A right = v[1], VERT_B
    /// left = v[0]). Then the frame-bounds clip the square leaves get.
    fn rect_edge_flags(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        second: bool,
        thr: bool,
        lhb: bool,
    ) -> (bool, bool) {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let (tr, bl) = match (vert, second) {
            (false, false) => (thr, true),
            (false, true) => (false, lhb),
            (true, false) => (true, lhb),
            (true, true) => (thr, false),
        };
        (tr && py > 0 && px + w < self.cfg.w, bl && px > 0 && py + h < self.cfg.h)
    }

    /// A (mode, delta) whose prediction never reads above-right / below-left
    /// (zone 2 and the non-directional modes): exact under any edge flags,
    /// so the per-half re-prediction of the split-TX path stays exact.
    fn rect_tbfree(m: usize, d: i32) -> bool {
        !is_directional_mode(m)
            || (m == V_PRED && d >= 0)
            || (m == H_PRED && d <= 0)
            || m == D113_PRED
            || m == D135_PRED
            || m == D157_PRED
    }

    #[allow(clippy::too_many_arguments)]
    fn rect16_luma_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        dc: i32,
        lam: f32,
        mlam: f32,
        refine: bool,
    ) -> RectLumaWin {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let kf = &self.dcdf().kf_y[yctx];
        // The equipped leaf (mode search + 7-type tx trial) is a Slow tool:
        // at Medium it grew the tier ~60% for decision work Medium's budget
        // cannot carry — Medium keeps the pre-equip DC leaf.
        let modes: &[usize] =
            if !refine || !self.cfg.speed.at_least_slow() {
                &[DC_PRED]
            } else {
                &Self::RECT_LEAF_MODES
            };

        // Prediction-domain SATD rerank: predictions are cheap next to the
        // DCT + trellis + entropy-rate pipeline, so score every candidate on
        // SATD and run the full pipeline on the top RECT_LEAF_BEAM only (DC
        // always retained as the safe fallback). Same pruning the square
        // proxy beam and the AV2 intra search use.
        // `rect_ad`: V_PRED with positive and H_PRED with negative angle
        // deltas join the candidate set — zone-2 angles that read neither the
        // top-right nor the bottom-left edge, so the leaf's tr/bl-free edge
        // setup stays exact.
        let rect_ad = refine && self.cfg.speed.at_least_slow() && crate::tuning::get().rect_ad;
        // `rect_full_modes`: the leaf predicts with its decoder-exact edge
        // flags and offers every directional mode and delta (zone 1/3 too);
        // off, the tr/bl-free set predicted without the edges (exact).
        let full = refine && self.cfg.speed.at_least_slow() && crate::tuning::get().rect_full_modes;
        let (ptr, pbl) = if full { (have_tr, have_bl) } else { (false, false) };
        // Filter-intra on the rect leaf (`filter_intra_rect`): the square
        // leaves' SATD-ranked shortlist, priced like every other candidate;
        // the DC candidates then also carry the filter_intra=0 flag they
        // signal, so the comparison is fair.
        let fi_budget = if refine && self.cfg.speed.at_least_slow() && crate::tuning::get().filter_intra_rect {
            self.cfg.speed.filter_intra_refine_budget(self.ss444())
        } else {
            0
        };
        let fi_bsize = av1_block_size_index(w, h);
        let mut all = FixedList::<(usize, i32), 64>::new((DC_PRED, 0));
        for &m in modes {
            all.push((m, 0));
        }
        if rect_ad {
            for d in 1..=3 {
                all.push((V_PRED, d));
                all.push((H_PRED, -d));
            }
        }
        if full {
            for m in [D45_PRED, D135_PRED, D113_PRED, D157_PRED, D203_PRED, VERT_LEFT_PRED] {
                all.push((m, 0));
            }
            if rect_ad {
                for d in 1..=3 {
                    all.push((V_PRED, -d));
                    all.push((H_PRED, d));
                }
                for m in [D45_PRED, D203_PRED, VERT_LEFT_PRED] {
                    for d in [-3, -2, -1, 1, 2, 3] {
                        all.push((m, d));
                    }
                }
            }
        }
        let mut all_diag = FixedList::<(usize, i32), 21>::new((DC_PRED, 0));
        if rect_ad && crate::tuning::get().rect_diag {
            for m in [D113_PRED, D135_PRED, D157_PRED] {
                for d in -3..=3 {
                    all_diag.push((m, d));
                }
            }
        }
        let predict = |m: usize, d: i32, dc: i32, pred: &mut [i32; 128]| {
            if m == DC_PRED {
                pred.fill(dc);
            } else {
                self.k.intrapred.predict_nd_ad(
                    m,
                    d,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                    ptr,
                    pbl,
                    self.cfg.w,
                    self.cfg.h,
                    ftype,
                    &mut pred[..],
                    self.cfg.bd,
                );
            }
        };
        let mut cands = FixedList::<(u64, usize, i32), 90>::new((0, DC_PRED, 0));
        for &(m, d) in all.iter().chain(all_diag.iter()) {
            let mut pred = [0i32; 128];
            predict(m, d, dc, &mut pred);
            let score = self.k.rd.satd_sad_proxy(self.src_blk(0, px, py, w, h), &pred, w);
            cands.push((score, m, d));
        }
        const RECT_LEAF_BEAM: usize = 3;
        if cands.len() > RECT_LEAF_BEAM {
            cands
                .as_mut_slice()
                .sort_unstable_by_key(|&(score, mode, d)| (score, mode, d));
            let dc_pos = cands.iter().position(|&(_, m, _)| m == DC_PRED).unwrap();
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
            0i32,
        );
        let mut best_sse = i64::MAX;
        for &(_, m, d) in &cands {
            let mut pred = [0i32; 128];
            predict(m, d, dc, &mut pred);
            let mut resid = [0i32; 128];
            self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, px, py, w, h);
            let (mut cf, tf) = if vert {
                self.k.dct.dct8x16_t(&resid, &self.ctx.quant)
            } else {
                self.k.dct.dct16x8_t(&resid, &self.ctx.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, w, h, px, py);
            self.k.rd.preserve_dc(&mut cf[0], &resid[..]);
            let rr = if vert {
                self.k.idct.idct_dequant_8x16(&cf, &self.ctx.quant)
            } else {
                self.k.idct.idct_dequant_16x8(&cf, &self.ctx.quant)
            };
            let sse =
                self.k.rd.sse_recon(&pred, &rr, self.src_blk(0, px, py, w, h), self.cfg.bd);
            let mut bits = self.luma_rect_bits(&cf, scan, w, h, px, py, m, 1) + cdf_cost(kf, m);
            if rect_ad && (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                bits += cdf_cost(&self.dcdf().angle_delta[m - V_PRED], (d + 3) as usize);
            }
            if fi_budget > 0 && m == DC_PRED {
                bits += cdf_cost(&self.dcdf().filter_intra[fi_bsize], 0);
            }
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best.0 {
                best = (cost, m, pred, resid, cf, d);
                best_sse = sse;
            }
        }
        let mut best_fi: Option<FilterIntraMode> = None;
        if fi_budget > 0 {
            let mut fi_preds = [[0i32; 128]; 5];
            for (fi_si, &filter_mode) in self
                .rank_filter_intra_modes::<128>(px, py, w, h, fi_budget, &mut fi_preds)
                .iter()
                .enumerate()
            {
                let pred = fi_preds[fi_si];
                let mut resid = [0i32; 128];
                self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, px, py, w, h);
                let (mut cf, tf) = if vert {
                    self.k.dct.dct8x16_t(&resid, &self.ctx.quant)
                } else {
                    self.k.dct.dct16x8_t(&resid, &self.ctx.quant)
                };
                self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, w, h, px, py);
                self.k.rd.preserve_dc(&mut cf[0], &resid[..]);
                let rr = if vert {
                    self.k.idct.idct_dequant_8x16(&cf, &self.ctx.quant)
                } else {
                    self.k.idct.idct_dequant_16x8(&cf, &self.ctx.quant)
                };
                let sse =
                    self.k.rd.sse_recon(&pred, &rr, self.src_blk(0, px, py, w, h), self.cfg.bd);
                // Admission needs `sse <= best_sse`: skip the rate pass otherwise.
                if sse > best_sse {
                    continue;
                }
                // The tx-type set is keyed on the filter mode's intra direction.
                let tx_mode = filter_intra_tx_mode(Some(filter_mode), DC_PRED);
                let bits = self.luma_rect_bits(&cf, scan, w, h, px, py, tx_mode, 1)
                    + cdf_cost(kf, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[fi_bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits);
                // Same admission as the square leaves: must not lose SSE.
                if sse <= best_sse && cost < best.0 {
                    best = (cost, DC_PRED, pred, resid, cf, 0);
                    best_sse = sse;
                    best_fi = Some(filter_mode);
                }
            }
        }
        // Palette leaf: 16x8 / 8x16 is the dominant palette shape on text
        // (one glyph-row band), and the only way a palette reaches it —
        // SPLIT to 8x8 pays the header twice, a 16x16 spans two text bands.
        let mut best_pal: Option<LossyLumaPalette> = None;
        if self.cfg.screen_frame
            && self.try_palette()
            && let Some(hist) = block_color_histogram(&self.cfg.src[0], self.cfg.w, px, py, w, h)
        {
            for (palette, pred) in
                self.rank_luma_palette_candidates::<128>(&hist, px, py, w, h, mlam * K_PART_LAMBDA)
            {
                // Exact bound: distortion and coefficient bits are >= 0, so a
                // finalist whose signalling alone loses cannot win.
                let pal_bits = self.palette_rate_bits(px, py, &palette);
                if rate_cost(mlam, cdf_cost(kf, DC_PRED) + pal_bits) >= best.0 {
                    continue;
                }
                let mut resid = [0i32; 128];
                self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, px, py, w, h);
                let (mut cf, tf) = if vert {
                    self.k.dct.dct8x16_t(&resid, &self.ctx.quant)
                } else {
                    self.k.dct.dct16x8_t(&resid, &self.ctx.quant)
                };
                self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, w, h, px, py);
                let rr = if vert {
                    self.k.idct.idct_dequant_8x16(&cf, &self.ctx.quant)
                } else {
                    self.k.idct.idct_dequant_16x8(&cf, &self.ctx.quant)
                };
                let sse = self.k.rd.sse_recon(&pred, &rr, self.src_blk(0, px, py, w, h), self.cfg.bd);
                let bits = self.luma_rect_bits(&cf, scan, w, h, px, py, DC_PRED, 1)
                    + cdf_cost(kf, DC_PRED)
                    + pal_bits;
                let cost = rd_cost_i64(sse, mlam, bits);
                if cost < best.0 {
                    best = (cost, DC_PRED, pred, resid, cf, 0);
                    best_pal = Some(palette);
                    best_fi = None;
                }
            }
        }
        (best.1, best.5, best.2, best.3, best.4, best_pal, best_fi)
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
        palette: bool,
        allow_split: bool,
    ) -> (usize, [i32; 128]) {
        if !palette && (!refine || !self.cfg.speed.at_least_slow()) {
            return (1, *dct_cf);
        }
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let sse_of = |rr: &[i32; 128]| -> i64 {
            self.k.rd.sse_recon(pred, rr, self.src_blk(0, px, py, w, h), self.cfg.bd)
        };

        let dct_rr = if vert {
            self.k.idct.idct_dequant_8x16(dct_cf, &self.ctx.quant)
        } else {
            self.k.idct.idct_dequant_16x8(dct_cf, &self.ctx.quant)
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

        // Palette leaves: DCT vs IDTX only (see the 16x16 IDTX note).
        let adst_types: &[usize] = if palette { &[] } else { &[ADST_ADST_TX8_IDX, 5, 6] };
        // Trellis'd 2-D candidates: ADST_ADST (4), ADST_DCT (5), DCT_ADST (6).
        for &txtp in adst_types {
            let (mut acf, atf) = match (vert, txtp) {
                (true, ADST_ADST_TX8_IDX) => self.k.dct.adst8x16_t(resid, &self.ctx.quant),
                (true, 5) => self.k.dct.adstdct8x16_t(resid, &self.ctx.quant),
                (true, _) => self.k.dct.dctadst8x16_t(resid, &self.ctx.quant),
                (false, ADST_ADST_TX8_IDX) => self.k.dct.adst16x8_t(resid, &self.ctx.quant),
                (false, 5) => self.k.dct.adstdct16x8_t(resid, &self.ctx.quant),
                (false, _) => self.k.dct.dctadst16x8_t(resid, &self.ctx.quant),
            };
            self.luma_rect_trellis(&mut acf, &atf, dcq, acq, scan, lam, w, h, px, py);
            // Same DC-preservation snap the DCT candidate gets.
            self.k.rd.preserve_dc(&mut acf[0], &resid[..]);
            let rr = inv_rect_luma_128(&self.k.idct, &acf, &self.ctx.quant, vert, txtp);
            let sse = sse_of(&rr);
            let c = rd_cost_i64(
                sse,
                mlam,
                self.luma_rect_bits(&acf, scan, w, h, px, py, y_mode, txtp),
            );
            if sse <= best_sse + self.tx_guard(best_sse) && c < best.0 {
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
                self.k.dct.idtx8x16_t(resid, &self.ctx.quant)
            } else {
                self.k.dct.idtx16x8_t(resid, &self.ctx.quant)
            };
            let rr = inv_rect_luma_128(&self.k.idct, &icf, &self.ctx.quant, vert, 0);
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
        for (txtp, one_d_vertical) in if palette { &[][..] } else { &[(2usize, true), (3usize, false)][..] } {
            let (txtp, one_d_vertical) = (*txtp, *one_d_vertical);
            let (vcf, _vtf) = match (vert, one_d_vertical) {
                (true, true) => self.k.dct.fvdct8x16_t(resid, &self.ctx.quant),
                (true, false) => self.k.dct.fhdct8x16_t(resid, &self.ctx.quant),
                (false, true) => self.k.dct.fvdct16x8_t(resid, &self.ctx.quant),
                (false, false) => self.k.dct.fhdct16x8_t(resid, &self.ctx.quant),
            };
            let rr = inv_rect_luma_128(&self.k.idct, &vcf, &self.ctx.quant, vert, txtp);
            let sse = sse_of(&rr);
            let bits = self.luma_rect_bits_1d(&vcf, w, one_d_vertical, px, py, y_mode);
            let c = rd_cost_i64(sse, mlam, bits);
            if sse <= best_sse && c < best.0 {
                best = (c, txtp, vcf);
                best_sse = sse;
            }
        }
        // tx_depth 1: two TX_8X8 (DCT, the 4:2:0 sub-TX type set) predicted
        // here from the WHOLE-leaf prediction (the emitter re-predicts each
        // half from the running reconstruction). Plain RD, no SSE gate —
        // the 16x16 block's split decision is priced the same way.
        if allow_split && !palette {
            let (split_cf, sse, bits) = self.rect_split_try(resid, pred, px, py, vert, y_mode, lam);
            let c = rd_cost_i64(sse, mlam, bits);
            if c < best.0 {
                best = (c, RECT_SPLIT_TXTP, split_cf);
            }
        }
        (best.1, best.2)
    }

    /// Commit a rect16 leaf's tx_depth 1 (two TX_8X8 DCT): the decoder
    /// predicts each TX_8X8 from the running reconstruction (half 1 sees
    /// half 0), so re-predict, re-transform and reconstruct the halves now,
    /// in decode order. The luma reconstruction is final afterwards; returns
    /// the per-half coefficients (the caller emits them, see
    /// [`Self::rect_split_emit`]) and the leaf's reconstruction (the CfL
    /// reference). Edge flags: the rect leaf's mode set never reads
    /// above-right / below-left (see `rect16_luma_mode_search`), same as the
    /// whole leaf.
    /// Per-transform edge flags of the two TX_8X8 inside a rect16 leaf whose
    /// block flags are (`have_tr`, `have_bl`), dav1d `recon_tmpl.c`: a TX has
    /// above-right if it is in the first TX row and the block has it, or it
    /// is not the rightmost TX column; below-left if it is in the leftmost TX
    /// column and (the block has it, or it is not the last TX row). The
    /// in-block edges exist iff the block's top row / left column do.
    fn rect_tx_flags(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        i: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> (bool, bool) {
        match (vert, i) {
            (false, 0) => (py > 0, have_bl),
            (false, _) => (have_tr, false),
            (true, 0) => (have_tr, px > 0),
            (true, _) => (false, have_bl),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rect_split_commit(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        y_mode: usize,
        y_delta: i32,
        lam: f32,
    ) -> ([[i32; 64]; 2], [u16; 128]) {
        let maxval = (1 << self.cfg.bd) - 1;
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let mut split_cf = [[0i32; 64]; 2];
        let mut luma_rec = [0u16; 128];
        let block_ftype = self.luma_filter_type(px, py);
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        for (i, half_cf) in split_cf.iter_mut().enumerate() {
            let (dx, dy) = rect_half_offset(vert, i);
            let (bx, by) = (px + dx, py + dy);
            let (tx_tr, tx_bl) = if crate::tuning::get().rect_full_modes {
                self.rect_tx_flags(px, py, vert, i, have_tr, have_bl)
            } else {
                (false, false)
            };
            let mut pred8 = [0i32; 64];
            if y_mode == DC_PRED && y_delta == 0 {
                let d = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                pred8 = [d; 64];
            } else {
                self.k.intrapred.predict_nd_ad(
                    y_mode,
                    y_delta,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    bx,
                    by,
                    8,
                    8,
                    tx_tr,
                    tx_bl,
                    self.cfg.w,
                    self.cfg.h,
                    block_ftype,
                    &mut pred8,
                    self.cfg.bd,
                );
            }
            let mut r8 = [0i32; 64];
            self.k.rd.residual_pred(&mut r8, &pred8, &self.cfg.src[0], self.cfg.w, bx, by, 8, 8);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 2, 2);
            let (mut cf, tf) = self.k.dct.dct8x8_t(&r8, &self.ctx.quant);
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
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            let rr = self.k.idct.idct_dequant_8x8(&cf, &self.ctx.quant);
            for ry in 0..8 {
                let drow = &mut self.ctx.recon[0][(by + ry) * self.cfg.w + bx..];
                recon_add_pred(&mut drow[..8], &pred8[ry * 8..], &rr[ry * 8..], maxval);
            }
            *half_cf = cf;
        }
        for ry in 0..lh {
            luma_rec[ry * lw..ry * lw + lw]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..lw]);
        }
        (split_cf, luma_rec)
    }

    /// `rect_split_seq`: the decoder-faithful split-TX estimate the leaf's
    /// transform trial cannot make from `&self`: run [`Self::rect_split_commit`]
    /// (half 1 predicted from the RECONSTRUCTED half 0, per-TX edge flags),
    /// read the leaf's SSE and bits, and put the reconstruction back. The
    /// `&self` estimator (`rect_split_try`) predicts both halves from the
    /// whole-leaf prediction and under-reads the split's distortion gain:
    /// under aom's tree the blocks aom splits and we kept whole cost x1.31
    /// bits at x1.49 SSE, the same-TX blocks x1.09 at x1.10.
    /// `rect_split_seq`: re-decide a rect leaf's whole-TX winner `ltxtp`
    /// against two TX_8X8 coded the way the decoder sees them (half 1
    /// predicted from the reconstructed half 0, [`Self::rect_split_probe`]).
    /// Returns the transform type to code; the coefficients are unchanged.
    #[allow(clippy::too_many_arguments)]
    fn rect_split_seq_txtp(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        y_mode: usize,
        y_delta: i32,
        tx_mode: usize,
        lam: f32,
        emlam: f32,
        ltxtp: usize,
        lcf: &[i32; 128],
        lpred: &[i32; 128],
    ) -> usize {
        if ltxtp == RECT_SPLIT_TXTP || !crate::tuning::get().rect_split_seq {
            return ltxtp;
        }
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        let rr_w = inv_rect_luma_128(&self.k.idct, lcf, &self.ctx.quant, vert, ltxtp);
        let sse_w = self.k.rd.sse_recon(lpred, &rr_w, self.src_blk(0, px, py, lw, lh), self.cfg.bd);
        let bits_w = if ltxtp == 2 || ltxtp == 3 {
            self.luma_rect_bits_1d(lcf, lw, ltxtp == 2, px, py, tx_mode)
        } else {
            self.luma_rect_bits(lcf, scan, lw, lh, px, py, tx_mode, ltxtp)
        };
        let (sse_s, bits_s) =
            self.rect_split_probe(px, py, vert, have_tr, have_bl, y_mode, y_delta, tx_mode, lam);
        if rd_cost_i64(sse_s, emlam, bits_s) < rd_cost_i64(sse_w, emlam, bits_w) {
            RECT_SPLIT_TXTP
        } else {
            ltxtp
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn rect_split_probe(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        y_mode: usize,
        y_delta: i32,
        tx_mode: usize,
        lam: f32,
    ) -> (i64, f32) {
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let mut saved = [0u16; 128];
        for ry in 0..lh {
            saved[ry * lw..ry * lw + lw]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..lw]);
        }
        let (split_cf, luma_rec) =
            self.rect_split_commit(px, py, vert, have_tr, have_bl, y_mode, y_delta, lam);
        for ry in 0..lh {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..lw]
                .copy_from_slice(&saved[ry * lw..ry * lw + lw]);
        }
        let mut rec = [0i32; 128];
        for (r, &v) in rec.iter_mut().zip(luma_rec.iter()) {
            *r = v as i32;
        }
        let sse = self.k.rd.sse_recon(&rec, &[0i32; 128], self.src_blk(0, px, py, lw, lh), self.cfg.bd);
        let mut cf128 = [0i32; 128];
        cf128[..64].copy_from_slice(&split_cf[0]);
        cf128[64..].copy_from_slice(&split_cf[1]);
        let bits = self.rect_split_bits(&cf128, vert, px, py, tx_mode);
        (sse, bits)
    }

    /// Emit a split rect16 leaf's two TX_8X8 coefficient blocks in decode
    /// order and update the coefficient contexts; the luma was reconstructed
    /// by [`Self::rect_split_commit`].
    fn rect_split_emit(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        y_mode: usize,
        block_skip: bool,
        split_cf: &[[i32; 64]; 2],
    ) {
        for (i, half_cf) in split_cf.iter().enumerate() {
            let (dx, dy) = rect_half_offset(vert, i);
            let (qbx4, qby4) = ((px + dx) / 4, (py + dy) / 4);
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
                let ds = self.dc_sign_ctx(0, qbx4, qby4);
                encode_tx8_coeffs_adapt(
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    half_cf,
                    false,
                    sk,
                    ds,
                    y_mode,
                    1,
                )
            };
            self.ctx.a_coef[0][qbx4..qbx4 + 2].fill(res_ctx);
            self.ctx.l_coef[0][qby4..qby4 + 2].fill(res_ctx);
        }
    }

    /// Deblock footprint of a split rect16 leaf: the two TX_8X8 (the
    /// prediction map keeps the 16x8/8x16 block), exactly as the 16x16 leaf
    /// re-records its four 8x8 transforms.
    fn rect_split_record_tx(&mut self, px: usize, py: usize, vert: bool) {
        for i in 0..2 {
            let (dx, dy) = rect_half_offset(vert, i);
            self.record_tx_blk((px + dx) / 8, (py + dy) / 8, 2);
        }
    }

    /// Whether the rect16 leaf may code tx_depth 1 (two TX_8X8), at every
    /// chroma format (`rect_tx_split`).
    fn rect_split_allowed(&self) -> bool {
        crate::tuning::get().rect_tx_split
    }

    /// Estimate of a rect16 leaf coded as two TX_8X8: (half-major coefficients,
    /// SSE, bits incl. the tx_depth delta vs depth 0). Both halves use the
    /// given whole-leaf prediction and the pre-leaf coefficient contexts.
    #[allow(clippy::too_many_arguments)]
    fn rect_split_try(
        &self,
        resid: &[i32; 128],
        pred: &[i32; 128],
        px: usize,
        py: usize,
        vert: bool,
        y_mode: usize,
        lam: f32,
    ) -> ([i32; 128], i64, f32) {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let mut cf128 = [0i32; 128];
        let mut sse_sum = 0i64;
        let mut bits = self.tx_depth_bits(px, py, w, h, 1) - self.tx_depth_bits(px, py, w, h, 0);
        for i in 0..2 {
            let (dx, dy) = rect_half_offset(vert, i);
            let (bx, by) = (px + dx, py + dy);
            let r8 = rect_half_i32(resid, vert, i);
            let p8 = rect_half_i32(pred, vert, i);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 2, 2);
            let (mut cf, tf) = self.k.dct.dct8x8_t(&r8, &self.ctx.quant);
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
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            let rr = self.k.idct.idct_dequant_8x8(&cf, &self.ctx.quant);
            sse_sum += sse_recon::<64, 8>(&self.k.rd, &p8, &rr, self.src_blk(0, bx, by, 8, 8), self.cfg.bd);
            bits += self.luma_bits_ctx_bounded(&cf, &SCAN_8X8, 8, bx, by, y_mode, 1, sk, ds, f32::INFINITY);
            cf128[64 * i..64 * i + 64].copy_from_slice(&cf);
        }
        (cf128, sse_sum, bits)
    }

    /// Bits of a [`RECT_SPLIT_TXTP`] leaf: both TX_8X8 halves plus the
    /// tx_depth delta, from the pre-leaf contexts (estimate).
    fn rect_split_bits(&self, cf: &[i32; 128], vert: bool, px: usize, py: usize, y_mode: usize) -> f32 {
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let mut bits = self.tx_depth_bits(px, py, w, h, 1) - self.tx_depth_bits(px, py, w, h, 0);
        for i in 0..2 {
            let (dx, dy) = rect_half_offset(vert, i);
            let (bx, by) = (px + dx, py + dy);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 2, 2);
            let half = rect_half_i32(cf, true, i);
            bits += self.luma_bits_ctx_bounded(&half, &SCAN_8X8, 8, bx, by, y_mode, 1, sk, ds, f32::INFINITY);
        }
        bits
    }

    fn rd_cost_horz(&self, px: usize, py: usize, thr: bool, lhb: bool, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        let (t0, b0) = self.rect_edge_flags(px, py, false, false, thr, lhb);
        let (t1, b1) = self.rect_edge_flags(px, py + 8, false, true, thr, lhb);
        rate_cost(mlam, self.part16_rate(px / 8, py / 8, 1))
            + self.rd_cost_rect16_leaf(px, py, false, t0, b0, prdo)
            + self.rd_cost_rect16_leaf(px, py + 8, false, t1, b1, prdo)
    }

    fn rd_cost_vert(&self, px: usize, py: usize, thr: bool, lhb: bool, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        let (t0, b0) = self.rect_edge_flags(px, py, true, false, thr, lhb);
        let (t1, b1) = self.rect_edge_flags(px + 8, py, true, true, thr, lhb);
        rate_cost(mlam, self.part16_rate(px / 8, py / 8, 2))
            + self.rd_cost_rect16_leaf(px, py, true, t0, b0, prdo)
            + self.rd_cost_rect16_leaf(px + 8, py, true, t1, b1, prdo)
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
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
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
            self.rank_chroma_modes::<128>(candidates, px, py, px, py, w, h, false, false);
        let mut best_total = cur_total;
        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, [[i32; 128]; 2], [[i32; 128]; 2])> = None;
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
            let tx = chroma_tx_for_mode(cand);
            let mut cand_ccf = [[0i32; 128]; 2];
            let mut cand_pred = [[0i32; 128]; 2];
            let mut cand_total = rate_cost(mlam, self.uv_mode_bits(y_mode, cand, None));
            for ci in 0..2 {
                let plane = ci + 1;
                self.k.intrapred.predict_nd(
                    cand,
                    &self.ctx.recon[plane],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                    false,
                    false,
                    self.cfg.w,
                    self.cfg.h,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci],
                    self.cfg.bd,
                );
                let mut resid = [0i32; 128];
                self.k.rd.residual_pred(
                    &mut resid,
                    &cand_pred[ci],
                    &self.cfg.src[plane],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                );
                let (mut q, qt) = if vert {
                    fwd_chroma_8x16(&self.k.dct, tx, &resid, &self.ctx.cquant)
                } else {
                    fwd_chroma_16x8(&self.k.dct, tx, &resid, &self.ctx.cquant)
                };
                trellis_optimize(&mut q, &qt, cdcq, cacq, scan, lam);
                self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                let rr = if vert {
                    inv_chroma_8x16(&self.k.idct, tx, &q, &self.ctx.cquant)
                } else {
                    inv_chroma_16x8(&self.k.idct, tx, &q, &self.ctx.cquant)
                };
                let sse = self.k.rd.sse_recon(&cand_pred[ci], &rr, self.src_blk(plane, px, py, w, h), self.cfg.bd);
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

    /// Code both halves of the partition (or only the first, at a frame edge).
    fn code_block16_vert_444(&mut self, x8: usize, y8: usize, halves: usize, thr: bool, lhb: bool) {
        for half in 0..halves {
            self.code_block16_vert_444_half(x8, y8, half, thr, lhb);
        }
    }

    /// Search of one 8x16 half of a 16x16 VERT partition (4:4:4): luma mode and transform, then
    /// chroma. NOT state-neutral when the two-square TX split wins:
    /// `rect_split_commit` then writes the final luma reconstruction into the
    /// tile (each half predicts from the other), before the chroma search.
    fn pick_rect16_v444(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) -> Rect16PickV444 {
        let maxval = (1 << self.cfg.bd) - 1;
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8 + half * 8, y8 * 8);
        let (tr_h, bl_h) = self.rect_edge_flags(px, py, true, half == 1, thr, lhb);
        let (bx4, by4) = (px / 4, py / 4);
        let dc_l = self.k.intrapred.dc_pred_8x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let (y_mode, y_delta, lpred_arr, lresid, lcf, rpal, rfi) =
            self.rect16_luma_mode_search(px, py, true, tr_h, bl_h, dc_l, lam, emlam, true);
        let tx_mode = filter_intra_tx_mode(rfi, y_mode);
        let allow_split = self.rect_split_allowed() && rpal.is_none() && rfi.is_none() && (crate::tuning::get().rect_full_modes || Self::rect_tbfree(y_mode, y_delta));
        let (ltxtp, lcf) = self.rect_leaf_tx_trial(
            &lresid, &lcf, &lpred_arr, px, py, true, tx_mode, lam, emlam, true, rpal.is_some(),
            allow_split);
        // Decoder-faithful split-TX check (`rect_split_seq`): re-decide whole vs
        // two TX_8X8 with half 1 predicted from the reconstructed half 0.
        let ltxtp = if allow_split {
            self.rect_split_seq_txtp(
                px, py, true, tr_h, bl_h, y_mode, y_delta, tx_mode, lam, emlam, ltxtp, &lcf, &lpred_arr,
            )
        } else {
            ltxtp
        };
        let idct = self.k.idct;
        let inv8x16 = |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, true, ltxtp);
        let split = ltxtp == RECT_SPLIT_TXTP;
        let mut split_cf = [[0i32; 64]; 2];
        let mut split_rec = [0u16; 128];
        if split {
            (split_cf, split_rec) = self.rect_split_commit(px, py, true, tr_h, bl_h, y_mode, y_delta, lam);
        }
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_8x16(&self.ctx.recon[plane], self.cfg.w, px, py, self.cfg.bd as i32);
            cpred[ci] = dc;
            let mut resid = [0i32; 128];
            self.k.rd.residual_dc(&mut resid, &self.cfg.src[plane], self.cfg.w, px, py, 8, 16, dc);
            let (mut q, qt) = self.k.dct.dct8x16_t(&resid, &self.ctx.cquant);
            self.chroma_rect_trellis(
                &mut q, &qt, cdcq, cacq, &SCAN_8X16, lam, 8, 16, plane, px, py,
            );
            self.k.rd.preserve_dc(&mut q[0], &resid[..]);
            ccf[ci] = q;
        }
        // --- CfL: predict chroma from this sub-block's reconstructed luma.
        // Allowed here (8x16 <= 32x32); at 4:4:4 chroma is full resolution
        // so chroma-from-luma is worth far more than a luma mode search.
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
        let mut cfl_ccf = [[0i32; 128]; 2];
        let mut cfl_pred = [[0i32; 128]; 2];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
        {
            let mut luma_rec = [0u16; 128];
            if split {
                luma_rec = split_rec;
            } else {
                let lrr_cfl = inv8x16(&lcf, &self.ctx.quant);
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
            }
            let mut ac = [0i32; 128];
            self.k.intrapred.cfl_ac_444(&luma_rec, 8, 16, &mut ac);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let csrc = self.src_blk(plane, px, py, 8, 16);
                let dcrr = self.k.idct.idct_dequant_8x16(&ccf[ci], &self.ctx.cquant);
                let s =
                    self.k.rd.sse_recon(&[dc; 128], &dcrr, csrc, self.cfg.bd);
                dc_cost[ci] = rd_cost_i64(
                    s,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci], &SCAN_8X16, 8, 16, plane, px, py),
                );
                let a = self
                    .k.intrapred
                    .cfl_best_alpha(&ac, csrc, dc, self.cfg.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 128];
                self.k.intrapred.cfl_pred(&mut cpr, &ac[..128], dc, a, self.cfg.bd);
                let mut resid = [0i32; 128];
                self.k.rd.residual_pred_blk(&mut resid, &cpr, csrc);
                let (mut q, qt) = self.k.dct.dct8x16_t(&resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_8X16, lam, 8, 16, plane, px, py,
                );
                let rr2 = self.k.idct.idct_dequant_8x16(&q, &self.ctx.cquant);
                let s2 = self.k.rd.sse_recon(&cpr, &rr2, csrc, self.cfg.bd);
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
        if self.cfg.speed.full_chroma_rdo() {
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
        let luma_zero = if split {
            split_cf.iter().all(|half| self.k.rd.all_zero_i32(half))
        } else {
            self.k.rd.all_zero_i32(&lcf)
        };
        let chroma_zero =
            self.k.rd.all_zero_i32(&ccf[0]) && self.k.rd.all_zero_i32(&ccf[1]);
        let block_skip = luma_zero && chroma_zero && rpal.is_none();
        Rect16PickV444 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            chosen_uv,
            cfl_opt,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_pred,
            use_cfl,
            cfl_pred,
            cpred,
        }
    }

    /// Emit one 8x16 half of a 16x16 VERT partition (4:4:4) from its search result.
    fn emit_rect16_v444(&mut self, x8: usize, y8: usize, half: usize, pick: Rect16PickV444) {
        let Rect16PickV444 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            chosen_uv,
            cfl_opt,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_pred,
            use_cfl,
            cfl_pred,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (px, py) = (x8 * 8 + half * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.record_blk_rect(x8 + half, y8, 2, 4);
        self.mark_skip8_rect(x8 + half, y8, 1, 2, block_skip);
        if split {
            self.rect_split_record_tx(px, py, true);
        }
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        // Directional modes carry an angle_delta symbol (`use_angle_delta` is
        // true for BLOCK_8X8 and larger); omitting it yields an invalid
        // bitstream. This search only offers delta 0.
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc
                .encode_symbol((y_delta + 3) as usize, &mut self.ent.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, chosen_uv, cfl_opt, 0, px, py, 8, 16);
        self.emit_palette_mode_info(px, py, 8, 16, y_mode, !self.cfg.is_mono(), rpal.as_ref(), None);
        if rpal.is_none() {
            self.emit_filter_intra(y_mode, 8, 16, rfi);
        }
        if let Some(p) = rpal.as_ref() {
            self.emit_palette_map(p);
        }
        self.code_tx_depth(px, py, 8, 16, split as usize);
        let sv = block_skip as u8;
        self.ctx.a_skip[bx4..bx4 + 2].fill(sv);
        self.ctx.l_skip[by4..by4 + 4].fill(sv);
        self.ctx.a_mode[bx4..bx4 + 2].fill(y_mode as u8);
        self.ctx.l_mode[by4..by4 + 4].fill(y_mode as u8);
        if split {
            // Two TX_8X8 in decode order; luma was reconstructed above.
            self.rect_split_emit(px, py, true, y_mode, block_skip, &split_cf);
        } else {
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_luma();
                let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        8,
                        ltxtp == 2,
                        sk,
                        ds,
                        tx_mode,
                    )
                } else {
                    encode_8x16_luma_coeffs(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        sk,
                        ds,
                        tx_mode,
                        ltxtp,
                    )
                }
            };
            self.ctx.a_coef[0][bx4..bx4 + 2].fill(lres_ctx);
            self.ctx.l_coef[0][by4..by4 + 4].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv_rect_luma_128(&self.k.idct, &lcf, &self.ctx.quant, true, ltxtp)
            };
            for ry in 0..16 {
                let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
                recon_add_pred(&mut drow[..8], &lpred_arr[ry * 8..], &lrr[ry * 8..], maxval);
            }
        }
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_chroma(plane, bx4, by4);
                let ds = self.dc_sign_ctx_8x16_chroma(plane, bx4, by4);
                encode_8x16_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            };
            self.ctx.a_coef[plane][bx4..bx4 + 2].fill(cres_ctx);
            self.ctx.l_coef[plane][by4..by4 + 4].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 128]
            } else if chosen_uv != DC_PRED {
                // Directional/smooth uv mode: decoder-derived chroma tx.
                inv_chroma_8x16(
                    &self.k.idct,
                    chroma_tx_for_mode(chosen_uv),
                    &ccf[ci],
                    &self.ctx.cquant,
                )
            } else {
                self.k.idct.idct_dequant_8x16(&ccf[ci], &self.ctx.cquant)
            };
            for ry in 0..16 {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.w + px..];
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

    /// Code one 8x16 half of a 16x16 VERT partition (4:4:4): search, then emit.
    fn code_block16_vert_444_half(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) {
        let pick = self.pick_rect16_v444(x8, y8, half, thr, lhb);
        self.emit_rect16_v444(x8, y8, half, pick);
    }

    /// 4:2:0 rect: HORZ = two 16x8 luma + 8x4 chroma; VERT = two 8x16 + 4x8.
    fn code_block16_rect_420(&mut self, x8: usize, y8: usize, vert: bool, thr: bool, lhb: bool) {
        for half in 0..2 {
            let (sx8, sy8) = if vert {
                (x8 + half, y8)
            } else {
                (x8, y8 + half)
            };
            let (tr, bl) = self.rect_edge_flags(sx8 * 8, sy8 * 8, vert, half == 1, thr, lhb);
            self.code_block16_rect_leaf_420(sx8, sy8, vert, tr, bl);
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
    /// Decoder-exact (`have_tr`, `have_bl`) of strip `i` of a HORZ_4 / VERT_4
    /// 16x16 node (dav1d `init_edges` + `decode_sb`): HORZ_4 strips take
    /// h[0] = (node tr, true), h4 = (false, true), (false, true), h[1] =
    /// (false, node bl); VERT_4 strips v[0] = (true, node bl), v4 = (true,
    /// false), (true, false), v[1] = (node tr, false). Frame-bounds clipped.
    fn strip_edge_flags(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        i: usize,
        thr: bool,
        lhb: bool,
    ) -> (bool, bool) {
        let (w, h) = if vert { (4usize, 16usize) } else { (16, 4) };
        let (tr, bl) = match (vert, i) {
            (false, 0) => (thr, true),
            (false, 3) => (false, lhb),
            (false, _) => (false, true),
            (true, 0) => (true, lhb),
            (true, 3) => (thr, false),
            (true, _) => (true, false),
        };
        (tr && py > 0 && px + w < self.cfg.w, bl && px > 0 && py + h < self.cfg.h)
    }

    /// `strip_modes`: luma mode search of a 16x4 / 4x16 strip (every intra
    /// mode, angle deltas, decoder-exact edge flags), SATD beam then the full
    /// pipeline; returns (mode, delta, prediction, residual, coefficients).
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn strip16_luma_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
        mlam: f32,
    ) -> (usize, i32, [i32; 64], [i32; 64], [i32; 64]) {
        let (lw, lh) = if vert { (4usize, 16usize) } else { (16, 4) };
        let scan: &[u32] = if vert { &SCAN_4X16 } else { &SCAN_16X4 };
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let kf = &self.dcdf().kf_y[yctx];
        let ad_ok = self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx);
        const STRIP_MODES: [usize; 13] = [
            DC_PRED,
            V_PRED,
            H_PRED,
            D45_PRED,
            D135_PRED,
            D113_PRED,
            D157_PRED,
            D203_PRED,
            VERT_LEFT_PRED,
            SMOOTH_PRED,
            SMOOTH_V_PRED,
            SMOOTH_H_PRED,
            PAETH_PRED,
        ];
        let mut all = FixedList::<(usize, i32), 64>::new((DC_PRED, 0));
        for &m in &STRIP_MODES {
            all.push((m, 0));
            if ad_ok && (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                for d in [-3, -2, -1, 1, 2, 3] {
                    all.push((m, d));
                }
            }
        }
        let predict = |m: usize, d: i32, pred: &mut [i32; 64]| {
            if m == DC_PRED {
                let dc = self.k.intrapred.dc_pred(&self.ctx.recon[0], self.cfg.w, px, py, lw, lh, self.cfg.bd as i32);
                pred.fill(dc);
            } else {
                self.k.intrapred.predict_nd_ad(
                    m,
                    d,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    lw,
                    lh,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    ftype,
                    &mut pred[..],
                    self.cfg.bd,
                );
            }
        };
        let mut cands = FixedList::<(u64, usize, i32), 64>::new((0, DC_PRED, 0));
        for &(m, d) in all.iter() {
            let mut pred = [0i32; 64];
            predict(m, d, &mut pred);
            let score = self.k.rd.satd_sad_proxy(self.src_blk(0, px, py, lw, lh), &pred, lw);
            cands.push((score, m, d));
        }
        const STRIP_BEAM: usize = 3;
        if cands.len() > STRIP_BEAM {
            cands
                .as_mut_slice()
                .sort_unstable_by_key(|&(score, mode, d)| (score, mode, d));
            let dc_pos = cands.iter().position(|&(_, m, _)| m == DC_PRED).unwrap();
            if dc_pos >= STRIP_BEAM {
                cands.as_mut_slice().swap(STRIP_BEAM - 1, dc_pos);
            }
            cands.truncate(STRIP_BEAM);
        }
        let mut best = (f32::INFINITY, DC_PRED, 0i32, [0i32; 64], [0i32; 64], [0i32; 64]);
        for &(_, m, d) in &cands {
            let mut pred = [0i32; 64];
            predict(m, d, &mut pred);
            let mut resid = [0i32; 64];
            self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, px, py, lw, lh);
            let (mut cf, tf) = if vert {
                self.k.dct.dct4x16_t(&resid, &self.ctx.quant)
            } else {
                self.k.dct.dct16x4_t(&resid, &self.ctx.quant)
            };
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            self.k.rd.preserve_dc(&mut cf[0], &resid[..]);
            let rr = if vert {
                self.k.idct.idct_dequant_4x16(&cf, &self.ctx.quant)
            } else {
                self.k.idct.idct_dequant_16x4(&cf, &self.ctx.quant)
            };
            let sse = self.k.rd.sse_recon(&pred, &rr, self.src_blk(0, px, py, lw, lh), self.cfg.bd);
            let mut bits = self.luma_rect_bits(&cf, scan, lw, lh, px, py, m, 1) + cdf_cost(kf, m);
            if (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                bits += cdf_cost(&self.dcdf().angle_delta[m - V_PRED], (d + 3) as usize);
            }
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best.0 {
                best = (cost, m, d, pred, resid, cf);
            }
        }
        (best.1, best.2, best.3, best.4, best.5)
    }

    /// Code a 16x16 region as PARTITION_HORZ_4 (four stacked 16x4 strips) or
    /// PARTITION_VERT_4 (four side-by-side 4x16), each strip with its own
    /// mode search and an RTX_16X4 / RTX_4X16 luma transform. Chroma per format:
    /// - 4:4:4: every strip carries its own 16x4/4x16 chroma transform;
    /// - 4:2:2: HORZ_4 strips carry 8x4 chroma each (VERT_4 not selected at
    ///   4:2:2, matching the VERT rule);
    /// - 4:2:0: the subsampled dimension pairs strips — chroma is coded once
    ///   per pair, on strips 1 and 3, covering the pair's 8x4 / 4x8 region;
    /// - mono: luma only.
    fn code_block16_quad(&mut self, x8: usize, y8: usize, vert: bool, thr: bool, lhb: bool) {
        for i in 0..4 {
            let (px, py) = if vert {
                (x8 * 8 + 4 * i, y8 * 8)
            } else {
                (x8 * 8, y8 * 8 + 4 * i)
            };
            let (tr, bl) = self.strip_edge_flags(px, py, vert, i, thr, lhb);
            self.code_strip16(px, py, vert, i, tr, bl);
        }
    }

    #[allow(clippy::too_many_arguments)]
    /// Search of one 16x4 / 4x16 strip (index `idx` of the HORZ_4 / VERT_4
    /// partition): luma mode, then chroma where the strip carries it.
    /// Read-only.
    fn pick_strip16(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        idx: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> Strip16Pick {
        let lam = self.trellis_lambda() * self.emit_prdo(px, py, 16);
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (lw, lh) = if vert { (4usize, 16usize) } else { (16, 4) };
        let lscan: &[u32] = if vert { &SCAN_4X16 } else { &SCAN_16X4 };

        // --- luma: mode search (`strip_modes`) or the DC-only leaf ---
        let strip_modes = crate::tuning::get().strip_modes && self.cfg.speed.at_least_slow();
        let (y_mode, y_delta, lpred_arr, _lresid, lcf) = if strip_modes {
            let emlam = self.emit_mlam(px, py, 16);
            self.strip16_luma_mode_search(px, py, vert, have_tr, have_bl, lam, emlam)
        } else {
            let lpred = self.k.intrapred.dc_pred(&self.ctx.recon[0], self.cfg.w, px, py, lw, lh, self.cfg.bd as i32);
            let mut lresid = [0i32; 64];
            self.k.rd.residual_dc(&mut lresid, &self.cfg.src[0], self.cfg.w, px, py, lw, lh, lpred);
            let (mut lcf, ltf) = if vert {
                self.k.dct.dct4x16_t(&lresid, &self.ctx.quant)
            } else {
                self.k.dct.dct16x4_t(&lresid, &self.ctx.quant)
            };
            trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
            self.k.rd.preserve_dc(&mut lcf[0], &lresid[..]);
            (DC_PRED, 0i32, [lpred; 64], lresid, lcf)
        };

        // --- chroma geometry per format ---
        // (cx, cy, cw, ch) of the chroma transform this strip carries, if any.
        let chroma: Option<(usize, usize, usize, usize)> = if self.cfg.is_mono() {
            None
        } else if self.cfg.is_420() {
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
        } else if self.cfg.is_422() {
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
                let dc = self.k.intrapred.dc_pred(&self.ctx.recon[plane], self.cfg.cw, cx, cy, cw, ch, self.cfg.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 64];
                self.k.rd.residual_dc(
                    &mut resid,
                    &self.cfg.src[plane],
                    self.cfg.cw,
                    cx,
                    cy,
                    cw,
                    ch,
                    dc,
                );
                let cscan = chroma_scan_for(cw, ch);
                let (mut q, qt) = fwd_chroma_strip(&self.k.dct, cw, ch, &resid, &self.ctx.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                self.k.rd.preserve_dc(&mut q[0], &resid[..cn]);
                ccf[ci] = q;
            }
        }
        // `strip_uv_joint`: the rect leaf's joint (mode, delta) chroma
        // shortlist (section 18 of the synthetic-gap doc) on the strip pair's
        // 8x4 / 4x8 chroma, tr/bl-free set so the prediction is exact under
        // any edge flags; full RD per kept pair against the DC leaf.
        let mut chosen_uv = DC_PRED;
        let mut uv_delta_strip = 0i32;
        let mut uv_tx = ChromaTx::DctDct;
        let mut uv_pred = [[0i32; 32]; 2];
        if let Some((cx, cy, cw, ch)) = chroma
            && cw * ch == 32
            && crate::tuning::get().strip_uv_joint
            && self.cfg.speed.at_least_slow()
        {
            let tune = crate::tuning::get();
            let vert = cw == 4;
            let n = 32usize;
            let cscan = chroma_scan_for(cw, ch);
            let mlam = self.emit_mlam_c(px, py, 16);
            let mut dc_cost = 0f32;
            for ci in 0..2 {
                let plane = ci + 1;
                let src = self.src_blk(plane, cx, cy, cw, ch);
                let rr = inv_chroma_strip(&self.k.idct, cw, ch, &ccf[ci], &self.ctx.cquant);
                let s0 = self.k.rd.sse_recon(&[cpred[ci]; 32][..n], &rr[..n], src, self.cfg.bd);
                dc_cost += rd_cost_i64(
                    s0,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci][..n], cscan, cw, ch, plane, cx, cy),
                );
            }
            let mut best_total = dc_cost + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None));
            const STRIP_JOINT_MODES: [usize; 9] = [
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
            let k = (tune.rect_uv_joint_k as usize).clamp(1, 12);
            let ad_ok = self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx);
            let joint_ch = if self.cfg.is_420() { self.cfg.h.div_ceil(2) } else { self.cfg.h };
            let ftype = self.chroma_filter_type(px, py);
            let mut all = FixedList::<(u64, usize, i32), 45>::new((0, DC_PRED, 0));
            for &m in &STRIP_JOINT_MODES {
                let deltas: &[i32] = if !ad_ok || !(V_PRED..=VERT_LEFT_PRED).contains(&m) {
                    &[0]
                } else if m == V_PRED {
                    &[0, 1, 2, 3]
                } else if m == H_PRED {
                    &[-3, -2, -1, 0]
                } else {
                    &[-3, -2, -1, 0, 1, 2, 3]
                };
                for &d in deltas {
                    let mut cost = 0u64;
                    for plane in 1..=2 {
                        let mut pp = [0i32; 32];
                        self.k.intrapred.predict_nd_ad(
                            m, d, &self.ctx.recon[plane], self.cfg.cw, cx, cy, cw, ch, false, false,
                            self.cfg.cw, joint_ch, ftype, &mut pp, self.cfg.bd,
                        );
                        cost += self.k.rd.satd_sad_proxy(self.src_blk(plane, cx, cy, cw, ch), &pp, cw);
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
                let mut cand_ccf = [[0i32; 32]; 2];
                let mut cand_pred = [[0i32; 32]; 2];
                let mut cand_total =
                    rate_cost(mlam, self.uv_mode_bits(y_mode, cand, None) + ad_bits);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let src = self.src_blk(plane, cx, cy, cw, ch);
                    let mut pp = [0i32; 32];
                    self.k.intrapred.predict_nd_ad(
                        cand, jd, &self.ctx.recon[plane], self.cfg.cw, cx, cy, cw, ch, false, false,
                        self.cfg.cw, joint_ch, ftype, &mut pp, self.cfg.bd,
                    );
                    let mut resid = [0i32; 32];
                    self.k.rd.residual_pred_blk(&mut resid, &pp, src);
                    let (mut q, qt) = if vert {
                        fwd_chroma_4x8(&self.k.dct, tx, &resid, &self.ctx.cquant)
                    } else {
                        fwd_chroma_8x4(&self.k.dct, tx, &resid, &self.ctx.cquant)
                    };
                    trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                    self.k.rd.preserve_dc(&mut q[0], &resid[..n]);
                    let rr = if vert {
                        inv_chroma_4x8(&self.k.idct, tx, &q, &self.ctx.cquant)
                    } else {
                        inv_chroma_8x4(&self.k.idct, tx, &q, &self.ctx.cquant)
                    };
                    let sse = self.k.rd.sse_recon(&pp[..n], &rr[..n], src, self.cfg.bd);
                    cand_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_rect_bits(&q, cscan, cw, ch, plane, cx, cy),
                    );
                    cand_ccf[ci] = q;
                    cand_pred[ci] = pp;
                    if ci == 0 && cand_total >= best_total {
                        break; // cannot win: skip the second plane (non-negative costs)
                    }
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    chosen_uv = cand;
                    uv_delta_strip = jd;
                    uv_tx = tx;
                    uv_pred = cand_pred;
                    for ci in 0..2 {
                        ccf[ci] = [0i32; 64];
                        ccf[ci][..n].copy_from_slice(&cand_ccf[ci]);
                    }
                }
            }
        }

        let luma_zero = self.k.rd.all_zero_i32(&lcf[..]);
        let chroma_zero = chroma.is_none()
            || (self.k.rd.all_zero_i32(&ccf[0][..cn])
                && self.k.rd.all_zero_i32(&ccf[1][..cn]));
        let block_skip = luma_zero && chroma_zero;

        Strip16Pick {
            block_skip,
            y_mode,
            y_delta,
            chroma,
            chosen_uv,
            uv_delta_strip,
            lcf,
            lpred_arr,
            ccf,
            uv_tx,
            uv_pred,
            cpred,
        }
    }

    /// Emit one 16x4 / 4x16 strip from its search result.
    fn emit_strip16(&mut self, px: usize, py: usize, vert: bool, idx: usize, pick: Strip16Pick) {
        let Strip16Pick {
            block_skip,
            y_mode,
            y_delta,
            chroma,
            chosen_uv,
            uv_delta_strip,
            lcf,
            lpred_arr,
            ccf,
            uv_tx,
            uv_pred,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (lw, lh) = if vert { (4usize, 16usize) } else { (16, 4) };
        let (bx4, by4) = (px / 4, py / 4);
        // ---- Emit. Every search above is done; nothing below decides.
        // --- header syntax, decoder order ---
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.record_blk_rect4(bx4, by4, (lw / 4) as u8, (lh / 4).max(1) as u8);
        // skip8 is 8px-granular and is NOT a hint: it decides which 8x8s CDEF
        // filters and which 64x64 units count as signaled. Two strips share
        // each 8px band, and the decoder skips CDEF on an 8x8 only when ALL its
        // 4x4s are skip, so the odd (closing) strip ANDs into the even strip's
        // mark instead of overwriting it (a coded strip followed by a skipped
        // one used to suppress CDEF that the decoder applies — external review
        // round 3; same bug class as block32's 8x4 pairs).
        let band_skip = if idx % 2 == 1 {
            let sb8w = self.cfg.w.div_ceil(8);
            block_skip && self.ctx.maps.skip8.get((py / 8) * sb8w + px / 8).copied().unwrap_or(true)
        } else {
            block_skip
        };
        if vert {
            self.mark_skip8_rect(px / 8, py / 8, 1, 2, band_skip);
        } else {
            self.mark_skip8_rect(px / 8, py / 8, 2, 1, band_skip);
        }
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        // BLOCK_16X4 / BLOCK_4X16 sort above BLOCK_8X8, so a directional
        // strip codes its angle_delta symbol.
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc
                .encode_symbol((y_delta + 3) as usize, &mut self.ent.cdfs.angle_delta[y_mode - V_PRED]);
        }
        if chroma.is_some() {
            self.emit_uv_mode(y_mode, chosen_uv, None, uv_delta_strip, px, py, lw, lh);
        }
        self.emit_palette_mode_info(px, py, lw, lh, y_mode, chroma.is_some(), None, None);
        self.emit_filter_intra(y_mode, lw, lh, None);
        self.code_tx_depth(px, py, lw, lh, 0);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, (lh / 4).max(1));
        self.ctx.a_skip[bx4..bx4 + aw].fill(sv);
        self.ctx.l_skip[by4..by4 + ah].fill(sv);
        self.ctx.a_mode[bx4..bx4 + aw].fill(y_mode as u8);
        self.ctx.l_mode[by4..by4 + ah].fill(y_mode as u8);

        // --- luma coefficients + recon ---
        let lres_ctx = if block_skip {
            0x40
        } else {
            let ds = self.dc_sign_ctx_span(0, bx4, by4, aw, ah);
            if vert {
                encode_4x16_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, 0, ds, y_mode, 1)
            } else {
                encode_16x4_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, 0, ds, y_mode, 1)
            }
        };
        self.ctx.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.ctx.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 64]
        } else if vert {
            self.k.idct.idct_dequant_4x16(&lcf, &self.ctx.quant)
        } else {
            self.k.idct.idct_dequant_16x4(&lcf, &self.ctx.quant)
        };
        for ry in 0..lh {
            let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
            recon_add_pred(&mut drow[..lw], &lpred_arr[ry * lw..], &lrr[ry * lw..], maxval);
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
                    let a = &self.ctx.a_coef[plane];
                    let l = &self.ctx.l_coef[plane];
                    let ca = a[cbx4..(cbx4 + caw).min(a.len())]
                        .iter()
                        .any(|&x| x != 0x40) as usize;
                    let cl = l[cby4..(cby4 + cah).min(l.len())]
                        .iter()
                        .any(|&x| x != 0x40) as usize;
                    let sk = 7 + ca + cl;
                    let ds = self.dc_sign_ctx_span(plane, cbx4, cby4, caw, cah);
                    encode_chroma_strip(&mut self.ent.enc, &mut self.ent.cdfs, cw, ch, &ccf[ci], sk, ds)
                };
                self.ctx.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                self.ctx.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 64]
                } else if chosen_uv != DC_PRED {
                    let q: &[i32; 32] = ccf[ci][..32].try_into().unwrap();
                    let r = if cw == 4 {
                        inv_chroma_4x8(&self.k.idct, uv_tx, q, &self.ctx.cquant)
                    } else {
                        inv_chroma_8x4(&self.k.idct, uv_tx, q, &self.ctx.cquant)
                    };
                    let mut out = [0i32; 64];
                    out[..32].copy_from_slice(&r);
                    out
                } else {
                    inv_chroma_strip(&self.k.idct, cw, ch, &ccf[ci], &self.ctx.cquant)
                };
                for ry in 0..ch {
                    let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
                    if chosen_uv != DC_PRED {
                        recon_add_pred(&mut drow[..cw], &uv_pred[ci][ry * cw..], &rr[ry * cw..], maxval);
                    } else {
                        recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                    }
                }
            }
        }
    }

    /// Code one 16x4 / 4x16 strip: search, then emit.
    fn code_strip16(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        idx: usize,
        have_tr: bool,
        have_bl: bool,
    ) {
        let pick = self.pick_strip16(px, py, vert, idx, have_tr, have_bl);
        self.emit_strip16(px, py, vert, idx, pick);
    }

    /// Emit one 16x8/8x16 leaf. Used by binary and asymmetric partitions.
    /// Search of a 16x8 / 8x16 leaf at 4:2:0: luma mode and transform, then
    /// chroma. NOT state-neutral when the two-square TX split wins:
    /// `rect_split_commit` then writes the final luma reconstruction into the
    /// tile (each half predicts from the other), before the chroma search.
    fn pick_rect16_leaf_420(
        &mut self,
        x8: usize,
        y8: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
    ) -> Rect16Pick420 {
        let maxval = (1 << self.cfg.bd) - 1;
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8);
        let (lw, _) = if vert { (8usize, 16usize) } else { (16, 8) };
        // Luma: mode search + 7-type transform trial (shared with 444).
        let dc_l = if vert {
            self.k.intrapred.dc_pred_8x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
        } else {
            self.k.intrapred.dc_pred_16x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
        };
        let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let (y_mode, y_delta, lpred_arr, lresid, lcf, rpal, rfi) =
            self.rect16_luma_mode_search(px, py, vert, have_tr, have_bl, dc_l, lam, emlam, true);
        let tx_mode = filter_intra_tx_mode(rfi, y_mode);
        let allow_split = self.rect_split_allowed() && rpal.is_none() && rfi.is_none() && (crate::tuning::get().rect_full_modes || Self::rect_tbfree(y_mode, y_delta));
        let (ltxtp, lcf) = self.rect_leaf_tx_trial(
            &lresid,
            &lcf,
            &lpred_arr,
            px,
            py,
            vert,
            tx_mode,
            lam,
            emlam,
            true,
            rpal.is_some(),
            allow_split,
        );
        // Decoder-faithful split-TX check (`rect_split_seq`): re-decide whole vs
        // two TX_8X8 with half 1 predicted from the reconstructed half 0.
        let ltxtp = if allow_split {
            self.rect_split_seq_txtp(
                px, py, vert, have_tr, have_bl, y_mode, y_delta, tx_mode, lam, emlam, ltxtp, &lcf, &lpred_arr,
            )
        } else {
            ltxtp
        };
        let idct = self.k.idct;
        let inv_luma = |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, vert, ltxtp);
        let split = ltxtp == RECT_SPLIT_TXTP;
        // tx_depth 1: the decoder predicts each TX_8X8 from the running
        // reconstruction (half 1 sees half 0), so re-predict, re-transform and
        // reconstruct the halves now, in decode order. The luma reconstruction
        // is final after this; the coefficient symbols are emitted below.
        // Edge flags: the rect leaf's mode set never reads above-right /
        // below-left (see `rect16_luma_mode_search`), same as the whole leaf.
        let mut split_cf = [[0i32; 64]; 2];
        let mut luma_rec = [0u16; 128];
        if split {
            (split_cf, luma_rec) = self.rect_split_commit(px, py, vert, have_tr, have_bl, y_mode, y_delta, lam);
        } else {
            let lrr_cfl = inv_luma(&lcf, &self.ctx.quant);
            recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
        }
        // chroma 8x4 (horz) or 4x8 (vert) at chroma coords.
        let (cx, cy) = (px / 2, py / 2);
        let (cw, ch) = if vert { (4usize, 8usize) } else { (8, 4) };
        let mut ccf = [[0i32; 32]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = if vert {
                self.k.intrapred.dc_pred_4x8(&self.ctx.recon[plane], self.cfg.cw, cx, cy, self.cfg.bd as i32)
            } else {
                self.k.intrapred.dc_pred_8x4(&self.ctx.recon[plane], self.cfg.cw, cx, cy, self.cfg.bd as i32)
            };
            cpred[ci] = dc;
            let mut resid = [0i32; 32];
            self.k.rd.residual_dc(&mut resid, &self.cfg.src[plane], self.cfg.cw, cx, cy, cw, ch, dc);
            let (mut q, qt) = if vert {
                self.k.dct.dct4x8_t(&resid, &self.ctx.cquant)
            } else {
                self.k.dct.dct8x4_t(&resid, &self.ctx.cquant)
            };
            let cscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
            self.k.rd.preserve_dc(&mut q[0], &resid[..cw * ch]);
            ccf[ci] = q;
        }
        let mut use_cfl = false;
        let mut cfl_alpha = [0i32; 2];
        let mut cfl_px = [[0i32; 32]; 2];
        // Non-DC/CfL chroma winner (rect_uv_search): mode, its prediction and
        // the spec-derived transform type.
        let mut chosen_uv = DC_PRED;
        let mut uv_pred = [[0i32; 32]; 2];
        let mut uv_tx = ChromaTx::DctDct;
        let mut uv_delta_rect = 0i32;
        if !self.cfg.is_mono() {
            let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
            let mut ac = [0i32; 32];
            self.k.intrapred
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
                let src = self.src_blk(plane, cx, cy, cw, ch);
                let dcrr = if vert {
                    self.k.idct.idct_dequant_4x8(&ccf[ci], &self.ctx.cquant)
                } else {
                    self.k.idct.idct_dequant_8x4(&ccf[ci], &self.ctx.cquant)
                };
                let s0 = self.k.rd.sse_recon(&[dc; 32][..n], &dcrr[..n], src, self.cfg.bd);
                dc_cost += rd_cost_i64(
                    s0,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci], cscan, cw, ch, plane, cx, cy),
                );
                let a = self
                    .k.intrapred
                    .cfl_best_alpha(&ac, src, dc, self.cfg.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 32];
                self.k.intrapred
                    .cfl_pred(&mut cpr[..n], &ac[..n], dc, a, self.cfg.bd);
                let mut resid = [0i32; 32];
                self.k.rd.residual_pred_blk(&mut resid, &cpr, src);
                let (mut q, qt) = if vert {
                    self.k.dct.dct4x8_t(&resid, &self.ctx.cquant)
                } else {
                    self.k.dct.dct8x4_t(&resid, &self.ctx.cquant)
                };
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                let rr2 = if vert {
                    self.k.idct.idct_dequant_4x8(&q, &self.ctx.cquant)
                } else {
                    self.k.idct.idct_dequant_8x4(&q, &self.ctx.cquant)
                };
                let s2 =
                    self.k.rd.sse_recon(&cpr[..n], &rr2[..n], src, self.cfg.bd);
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
            let mut best_total = dc_cost + rate_cost(mlam, dc_sig);
            if (cfl_a[0] != 0 || cfl_a[1] != 0) && cfl_cost + rate_cost(mlam, cfl_sig) < best_total {
                best_total = cfl_cost + rate_cost(mlam, cfl_sig);
                use_cfl = true;
                cfl_alpha = cfl_a;
                cfl_px = cand_px;
                ccf = cfl_ccf;
            }
            // Full chroma mode search on the 8x4/4x8 chroma, mirroring the
            // 16x16 block's 8x8 chroma path (SATD top-K shortlist, spec
            // transform type per uv_mode). Without it the rect leaf's chroma
            // was DC/CfL-only while the squares it competes with are not.
            let tune = crate::tuning::get();
            // `rect_uv_joint`: (mode, delta) pairs ranked jointly on SATD, the
            // shipped 8x8/16x16 chroma shortlist carried to the rect leaf. Only
            // the above-right / below-left-free set (SMOOTH*, PAETH, V with
            // delta >= 0, H with delta <= 0, D113 / D135 / D157 at any delta),
            // so the tr/bl-free prediction is exact whatever the decoder's
            // edge flags are, same as the leaf's luma modes. aom codes these
            // blocks' chroma with D135 / D113 / D157 / H / SMOOTH far more
            // often than with CfL or DC (synthetic-gap coverage leg 3).
            let joint = tune.rect_uv_joint;
            let mut joint_list = FixedList::<(u64, usize, i32), 12>::new((0, DC_PRED, 0));
            if joint {
                const RECT_JOINT_MODES: [usize; 9] = [
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
                let k = (tune.rect_uv_joint_k as usize).clamp(1, 12);
                let ad_ok = self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx);
                let joint_ch = self.cfg.h.div_ceil(2);
                let ftype = self.chroma_filter_type(px, py);
                let mut all = FixedList::<(u64, usize, i32), 45>::new((0, DC_PRED, 0));
                for &m in &RECT_JOINT_MODES {
                    let deltas: &[i32] = if !ad_ok || !(V_PRED..=VERT_LEFT_PRED).contains(&m) {
                        &[0]
                    } else if m == V_PRED {
                        &[0, 1, 2, 3]
                    } else if m == H_PRED {
                        &[-3, -2, -1, 0]
                    } else {
                        &[-3, -2, -1, 0, 1, 2, 3]
                    };
                    for &d in deltas {
                        let mut cost = 0u64;
                        for plane in 1..=2 {
                            let mut pp = [0i32; 32];
                            self.k.intrapred.predict_nd_ad(
                                m,
                                d,
                                &self.ctx.recon[plane],
                                self.cfg.cw,
                                cx,
                                cy,
                                cw,
                                ch,
                                false,
                                false,
                                self.cfg.cw,
                                joint_ch,
                                ftype,
                                &mut pp,
                                self.cfg.bd,
                            );
                            cost += self
                                .k.rd
                                .satd_sad_proxy(self.src_blk(plane, cx, cy, cw, ch), &pp, cw);
                        }
                        all.push((cost, m, d));
                    }
                }
                all.as_mut_slice().sort_unstable_by_key(|&(c, m, d)| (c, m, d));
                for &e in all.as_slice().iter().take(k) {
                    joint_list.push(e);
                }
            }
            if tune.rect_uv_search || joint {
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
                let directional_top = if joint {
                    DirectionalTopK::new()
                } else {
                    self.rank_chroma_modes::<32>(candidates, px, py, cx, cy, cw, ch, false, false)
                };
                let mut plain_list = FixedList::<(u64, usize, i32), 12>::new((0, DC_PRED, 0));
                if !joint {
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
                        plain_list.push((0, cand, 0));
                    }
                }
                let list = if joint { joint_list.as_slice() } else { plain_list.as_slice() };
                for &(_, cand, jd) in list {
                    let tx = chroma_tx_for_mode(cand);
                    let ad_bits = if (V_PRED..=VERT_LEFT_PRED).contains(&cand) {
                        let ad = &self.dcdf().angle_delta[cand - V_PRED];
                        cdf_cost(ad, (jd + 3) as usize) - cdf_cost(ad, 3)
                    } else {
                        0.0
                    };
                    let mut cand_ccf = [[0i32; 32]; 2];
                    let mut cand_pred = [[0i32; 32]; 2];
                    let mut cand_total =
                        rate_cost(mlam, self.uv_mode_bits(y_mode, cand, None) + ad_bits);
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let src = self.src_blk(plane, cx, cy, cw, ch);
                        let mut pp = [0i32; 32];
                        if joint {
                            self.k.intrapred.predict_nd_ad(
                                cand,
                                jd,
                                &self.ctx.recon[plane],
                                self.cfg.cw,
                                cx,
                                cy,
                                cw,
                                ch,
                                false,
                                false,
                                self.cfg.cw,
                                self.cfg.h.div_ceil(2),
                                self.chroma_filter_type(px, py),
                                &mut pp,
                                self.cfg.bd,
                            );
                        } else {
                            self.k.intrapred.predict_nd(
                                cand,
                                &self.ctx.recon[plane],
                                self.cfg.cw,
                                cx,
                                cy,
                                cw,
                                ch,
                                false,
                                false,
                                self.cfg.cw,
                                self.cfg.h,
                                self.chroma_filter_type(px, py),
                                &mut pp,
                                self.cfg.bd,
                            );
                        }
                        let mut resid = [0i32; 32];
                        self.k.rd.residual_pred_blk(&mut resid, &pp, src);
                        let (mut q, qt) = if vert {
                            fwd_chroma_4x8(&self.k.dct, tx, &resid, &self.ctx.cquant)
                        } else {
                            fwd_chroma_8x4(&self.k.dct, tx, &resid, &self.ctx.cquant)
                        };
                        self.chroma_rect_trellis(
                            &mut q, &qt, cdcq, cacq, cscan, lam, cw, ch, plane, cx, cy,
                        );
                        self.k.rd.preserve_dc(&mut q[0], &resid[..n]);
                        let rr = if vert {
                            inv_chroma_4x8(&self.k.idct, tx, &q, &self.ctx.cquant)
                        } else {
                            inv_chroma_8x4(&self.k.idct, tx, &q, &self.ctx.cquant)
                        };
                        let sse = self.k.rd.sse_recon(&pp[..n], &rr[..n], src, self.cfg.bd);
                        cand_total += rd_cost_i64(
                            sse,
                            mlam,
                            self.chroma_rect_bits(&q, cscan, cw, ch, plane, cx, cy),
                        );
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                        if ci == 0 && cand_total >= best_total {
                            break; // cannot win: skip the second plane (non-negative costs)
                        }
                    }
                    if cand_total < best_total {
                        best_total = cand_total;
                        use_cfl = false;
                        chosen_uv = cand;
                        uv_delta_rect = jd;
                        uv_tx = tx;
                        uv_pred = cand_pred;
                        ccf = cand_ccf;
                    }
                }
            }
        }
        // tx_depth 1 re-derived each half's coefficients above from the
        // running reconstruction, so the skip flag must come from THOSE
        // (`lcf` is the whole-leaf estimate): an all-zero estimate with a
        // non-zero re-coded half used to signal skip=1 while the encoder's
        // reconstruction kept the residual -> decoder mismatch.
        let luma_zero = if split {
            split_cf.iter().all(|half| self.k.rd.all_zero_i32(half))
        } else {
            self.k.rd.all_zero_i32(&lcf)
        };
        let chroma_zero =
            self.k.rd.all_zero_i32(&ccf[0]) && self.k.rd.all_zero_i32(&ccf[1]);
        let block_skip = luma_zero && chroma_zero && rpal.is_none();
        Rect16Pick420 {
            block_skip,
            split,
            y_mode,
            y_delta,
            use_cfl,
            uv_delta_rect,
            chosen_uv,
            cfl_alpha,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_tx,
            cfl_px,
            uv_pred,
            cpred,
        }
    }

    /// Emit a 16x8 / 8x16 leaf at 4:2:0 from its search result.
    fn emit_rect16_leaf_420(&mut self, x8: usize, y8: usize, vert: bool, pick: Rect16Pick420) {
        let Rect16Pick420 {
            block_skip,
            split,
            y_mode,
            y_delta,
            use_cfl,
            uv_delta_rect,
            chosen_uv,
            cfl_alpha,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_tx,
            cfl_px,
            uv_pred,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let (cx, cy) = (px / 2, py / 2);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let (cw, ch) = if vert { (4usize, 8usize) } else { (8, 4) };
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        if vert {
            self.record_blk_rect(x8, y8, 2, 4);
            self.mark_skip8_rect(x8, y8, 1, 2, block_skip);
        } else {
            self.record_blk_rect(x8, y8, 4, 2);
            self.mark_skip8_rect(x8, y8, 2, 1, block_skip);
        }
        if split {
            self.rect_split_record_tx(px, py, vert);
        }
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc
                .encode_symbol((y_delta + 3) as usize, &mut self.ent.cdfs.angle_delta[y_mode - V_PRED]);
        }
        let uv_delta_sig = if use_cfl { 0 } else { uv_delta_rect };
        self.emit_uv_mode(
            y_mode,
            chosen_uv,
            if use_cfl { Some(cfl_alpha) } else { None },
            uv_delta_sig,
            px,
            py,
            lw,
            lh,
        );
        self.emit_palette_mode_info(px, py, lw, lh, y_mode, !self.cfg.is_mono(), rpal.as_ref(), None);
        if rpal.is_none() {
            self.emit_filter_intra(y_mode, lw, lh, rfi);
        }
        if let Some(p) = rpal.as_ref() {
            self.emit_palette_map(p);
        }
        self.code_tx_depth(px, py, lw, lh, split as usize);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, lh / 4);
        self.ctx.a_skip[bx4..bx4 + aw].fill(sv);
        self.ctx.l_skip[by4..by4 + ah].fill(sv);
        self.ctx.a_mode[bx4..bx4 + aw].fill(y_mode as u8);
        self.ctx.l_mode[by4..by4 + ah].fill(y_mode as u8);
        if split {
            // Two TX_8X8 in decode order; luma was reconstructed above.
            self.rect_split_emit(px, py, vert, y_mode, block_skip, &split_cf);
        } else {
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
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    &lcf,
                    lw,
                    ltxtp == 2,
                    sk,
                    ds,
                    tx_mode,
                )
            } else if vert {
                let sk = self.skip_ctx_8x16_luma();
                let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
                encode_8x16_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, sk, ds, tx_mode, ltxtp)
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, sk, ds, tx_mode, ltxtp)
            };
            self.ctx.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
            self.ctx.l_coef[0][by4..by4 + ah].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv_rect_luma_128(&self.k.idct, &lcf, &self.ctx.quant, vert, ltxtp)
            };
            for ry in 0..lh {
                let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
                recon_add_pred(
                    &mut drow[..lw],
                    &lpred_arr[ry * lw..],
                    &lrr[ry * lw..],
                    maxval,
                );
            }
        }
        let (caw, cah) = (cw / 4, (ch / 4).max(1));
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else if vert {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                encode_4x8_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            } else {
                let sk = self.skip_ctx_8x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x4_chroma(plane, cbx4, cby4);
                encode_8x4_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            };
            let fillw = caw.max(1);
            self.ctx.a_coef[plane][cbx4..cbx4 + fillw].fill(cres_ctx);
            self.ctx.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 32]
            } else if vert {
                inv_chroma_4x8(&self.k.idct, uv_tx, &ccf[ci], &self.ctx.cquant)
            } else {
                inv_chroma_8x4(&self.k.idct, uv_tx, &ccf[ci], &self.ctx.cquant)
            };
            for ry in 0..ch {
                let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
                if use_cfl {
                    recon_add_pred(
                        &mut drow[..cw],
                        &cfl_px[ci][ry * cw..],
                        &rr[ry * cw..],
                        maxval,
                    );
                } else if chosen_uv != DC_PRED {
                    recon_add_pred(
                        &mut drow[..cw],
                        &uv_pred[ci][ry * cw..],
                        &rr[ry * cw..],
                        maxval,
                    );
                } else {
                    recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                }
            }
        }
    }

    /// Code a 16x8 / 8x16 leaf at 4:2:0: search, then emit.
    fn code_block16_rect_leaf_420(
        &mut self,
        x8: usize,
        y8: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
    ) {
        let pick = self.pick_rect16_leaf_420(x8, y8, vert, have_tr, have_bl);
        self.emit_rect16_leaf_420(x8, y8, vert, pick);
    }

    /// 4:2:2 HORZ: two 16x8 luma + 8x8 chroma (h-subsampled, v-full). V forbidden in 422.
    /// Code both halves of the partition (or only the first, at a frame edge).
    fn code_block16_horz_422(&mut self, x8: usize, y8: usize, halves: usize, thr: bool, lhb: bool) {
        for half in 0..halves {
            self.code_block16_horz_422_half(x8, y8, half, thr, lhb);
        }
    }

    /// Search of one 16x8 half of a 16x16 HORZ partition (4:2:2): luma mode and transform, then
    /// chroma. NOT state-neutral when the two-square TX split wins:
    /// `rect_split_commit` then writes the final luma reconstruction into the
    /// tile (each half predicts from the other), before the chroma search.
    fn pick_rect16_h422(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) -> Rect16PickH422 {
        let maxval = (1 << self.cfg.bd) - 1;
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8 + half * 8);
        let (tr_h, bl_h) = self.rect_edge_flags(px, py, false, half == 1, thr, lhb);
        let (bx4, by4) = (px / 4, py / 4);
        // Luma 16x8: mode search + 7-type transform trial (shared with 444).
        let dc_l = self.k.intrapred.dc_pred_16x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let (y_mode, y_delta, lpred_arr, lresid, lcf, rpal, rfi) =
            self.rect16_luma_mode_search(px, py, false, tr_h, bl_h, dc_l, lam, emlam, true);
        let tx_mode = filter_intra_tx_mode(rfi, y_mode);
        let allow_split = self.rect_split_allowed() && rpal.is_none() && rfi.is_none() && (crate::tuning::get().rect_full_modes || Self::rect_tbfree(y_mode, y_delta));
        let (ltxtp, lcf) = self.rect_leaf_tx_trial(
            &lresid, &lcf, &lpred_arr, px, py, false, tx_mode, lam, emlam, true, rpal.is_some(),
            allow_split);
        // Decoder-faithful split-TX check (`rect_split_seq`): re-decide whole vs
        // two TX_8X8 with half 1 predicted from the reconstructed half 0.
        let ltxtp = if allow_split {
            self.rect_split_seq_txtp(
                px, py, false, tr_h, bl_h, y_mode, y_delta, tx_mode, lam, emlam, ltxtp, &lcf, &lpred_arr,
            )
        } else {
            ltxtp
        };
        let idct = self.k.idct;
        let inv16x8 =
            |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, false, ltxtp);
        let split = ltxtp == RECT_SPLIT_TXTP;
        let mut split_cf = [[0i32; 64]; 2];
        let mut split_rec = [0u16; 128];
        if split {
            (split_cf, split_rec) = self.rect_split_commit(px, py, false, tr_h, bl_h, y_mode, y_delta, lam);
        }
        let (cx, cy) = (px / 2, py);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let mut ccf = [[0i32; 64]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[plane], self.cfg.cw, cx, cy, self.cfg.bd as i32);
            cpred[ci] = dc;
            let mut resid = [0i32; 64];
            self.k.rd.residual_dc(&mut resid, &self.cfg.src[plane], self.cfg.cw, cx, cy, 8, 8, dc);
            let (mut q, qt) = self.k.dct.dct8x8_t(&resid, &self.ctx.cquant);
            self.chroma_rect_trellis(
                &mut q, &qt, cdcq, cacq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
            );
            self.k.rd.preserve_dc(&mut q[0], &resid[..]);
            ccf[ci] = q;
        }
        // CfL candidate (2026-07-23 rect-CfL audit: the 4:2:2 HORZ leg —
        // a large share of 422 blocks — never offered CfL). AC reference
        // from this leaf's 16x8 reconstruction, horizontally subsampled.
        let mut use_cfl = false;
        let mut cfl_alpha = [0i32; 2];
        let mut cfl_px = [[0i32; 64]; 2];
        if !self.cfg.is_mono() {
            let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
            let mut luma_rec = [0u16; 128];
            if split {
                luma_rec = split_rec;
            } else {
                let lrr_cfl = inv16x8(&lcf, &self.ctx.quant);
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
            }
            let mut ac = [0i32; 64];
            self.k.intrapred
                .cfl_ac_sub(&luma_rec, 16, 8, 8, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 64]; 2];
            let mut cfl_a = [0i32; 2];
            let (mut dc_cost, mut cfl_cost) = (0f32, 0f32);
            let mut cand_px = [[0i32; 64]; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let src = self.src_blk(plane, cx, cy, 8, 8);
                let dcrr = self.k.idct.idct_dequant_8x8(&ccf[ci], &self.ctx.cquant);
                let s0 = sse_recon::<64, 8>(&self.k.rd, &[dc; 64], &dcrr, src, self.cfg.bd);
                dc_cost += rd_cost_i64(
                    s0,
                    mlam,
                    self.chroma_bits(&ccf[ci], &SCAN_8X8, 8, plane, cx, cy),
                );
                let a = self
                    .k.intrapred
                    .cfl_best_alpha(&ac, src, dc, self.cfg.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 64];
                self.k.intrapred.cfl_pred(&mut cpr, &ac[..64], dc, a, self.cfg.bd);
                let mut resid = [0i32; 64];
                self.k.rd.residual_pred_blk(&mut resid, &cpr, src);
                let (mut q, qt) = self.k.dct.dct8x8_t(&resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_8X8, lam, 8, 8, plane, cx, cy,
                );
                let rr2 = self.k.idct.idct_dequant_8x8(&q, &self.ctx.cquant);
                let s2 = sse_recon::<64, 8>(&self.k.rd, &cpr, &rr2, src, self.cfg.bd);
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
        let luma_zero = if split {
            split_cf.iter().all(|half| self.k.rd.all_zero_i32(half))
        } else {
            self.k.rd.all_zero_i32(&lcf)
        };
        let chroma_zero =
            self.k.rd.all_zero_i32(&ccf[0]) && self.k.rd.all_zero_i32(&ccf[1]);
        let block_skip = luma_zero && chroma_zero && rpal.is_none();
        Rect16PickH422 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            use_cfl,
            cfl_alpha,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            cbx4,
            cby4,
            ccf,
            cy,
            cx,
            cfl_px,
            cpred,
        }
    }

    /// Emit one 16x8 half of a 16x16 HORZ partition (4:2:2) from its search result.
    fn emit_rect16_h422(&mut self, x8: usize, y8: usize, half: usize, pick: Rect16PickH422) {
        let Rect16PickH422 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            use_cfl,
            cfl_alpha,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            cbx4,
            cby4,
            ccf,
            cy,
            cx,
            cfl_px,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (px, py) = (x8 * 8, y8 * 8 + half * 8);
        let (bx4, by4) = (px / 4, py / 4);
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.record_blk_rect(x8, y8 + half, 4, 2);
        self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
        if split {
            self.rect_split_record_tx(px, py, false);
        }
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc
                .encode_symbol((y_delta + 3) as usize, &mut self.ent.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(
            y_mode,
            DC_PRED,
            if use_cfl { Some(cfl_alpha) } else { None },
            0,
            px,
            py,
            16,
            8,
        );
        self.emit_palette_mode_info(px, py, 16, 8, y_mode, !self.cfg.is_mono(), rpal.as_ref(), None);
        if rpal.is_none() {
            self.emit_filter_intra(y_mode, 16, 8, rfi);
        }
        if let Some(p) = rpal.as_ref() {
            self.emit_palette_map(p);
        }
        self.code_tx_depth(px, py, 16, 8, split as usize);
        let sv = block_skip as u8;
        self.ctx.a_skip[bx4..bx4 + 4].fill(sv);
        self.ctx.l_skip[by4..by4 + 2].fill(sv);
        self.ctx.a_mode[bx4..bx4 + 4].fill(y_mode as u8);
        self.ctx.l_mode[by4..by4 + 2].fill(y_mode as u8);
        if split {
            // Two TX_8X8 in decode order; luma was reconstructed above.
            self.rect_split_emit(px, py, false, y_mode, block_skip, &split_cf);
        } else {
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        16,
                        ltxtp == 2,
                        sk,
                        ds,
                        tx_mode,
                    )
                } else {
                    encode_16x8_luma_coeffs(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        sk,
                        ds,
                        tx_mode,
                        ltxtp,
                    )
                }
            };
            self.ctx.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.ctx.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv_rect_luma_128(&self.k.idct, &lcf, &self.ctx.quant, false, ltxtp)
            };
            for ry in 0..8 {
                let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
                recon_add_pred(
                    &mut drow[..16],
                    &lpred_arr[ry * 16..],
                    &lrr[ry * 16..],
                    maxval,
                );
            }
        }
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x8_chroma(plane, cbx4, cby4);
                encode_tx8_coeffs_adapt(
                    &mut self.ent.enc,
                    &mut self.ent.cdfs,
                    &ccf[ci],
                    true,
                    sk,
                    ds,
                    DC_PRED,
                    0,
                )
            };
            self.ctx.a_coef[plane][cbx4..cbx4 + 2].fill(cres_ctx);
            self.ctx.l_coef[plane][cby4..cby4 + 2].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 64]
            } else {
                self.k.idct.idct_dequant_8x8(&ccf[ci], &self.ctx.cquant)
            };
            for ry in 0..8 {
                let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
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

    /// Code one 16x8 half of a 16x16 HORZ partition (4:2:2): search, then emit.
    fn code_block16_horz_422_half(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) {
        let pick = self.pick_rect16_h422(x8, y8, half, thr, lhb);
        self.emit_rect16_h422(x8, y8, half, pick);
    }

    /// Code both halves of the partition (or only the first, at a frame edge).
    fn code_block16_horz_444(&mut self, x8: usize, y8: usize, halves: usize, thr: bool, lhb: bool) {
        for half in 0..halves {
            self.code_block16_horz_444_half(x8, y8, half, thr, lhb);
        }
    }

    /// Search of one 16x8 half of a 16x16 HORZ partition (4:4:4): luma mode and transform, then
    /// chroma. NOT state-neutral when the two-square TX split wins:
    /// `rect_split_commit` then writes the final luma reconstruction into the
    /// tile (each half predicts from the other), before the chroma search.
    fn pick_rect16_h444(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) -> Rect16PickH444 {
        let maxval = (1 << self.cfg.bd) - 1;
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8 + half * 8);
        let (tr_h, bl_h) = self.rect_edge_flags(px, py, false, half == 1, thr, lhb);
        let (bx4, by4) = (px / 4, py / 4); // luma 4-unit coords
        // --- luma 16x8: mode search + 7-type transform trial ---
        let dc_l = self.k.intrapred.dc_pred_16x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let emlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let (y_mode, y_delta, lpred_arr, lresid, lcf, rpal, rfi) =
            self.rect16_luma_mode_search(px, py, false, tr_h, bl_h, dc_l, lam, emlam, true);
        let tx_mode = filter_intra_tx_mode(rfi, y_mode);
        let allow_split = self.rect_split_allowed() && rpal.is_none() && rfi.is_none() && (crate::tuning::get().rect_full_modes || Self::rect_tbfree(y_mode, y_delta));
        let (ltxtp, lcf) = self.rect_leaf_tx_trial(
            &lresid, &lcf, &lpred_arr, px, py, false, tx_mode, lam, emlam, true, rpal.is_some(),
            allow_split);
        // Decoder-faithful split-TX check (`rect_split_seq`): re-decide whole vs
        // two TX_8X8 with half 1 predicted from the reconstructed half 0.
        let ltxtp = if allow_split {
            self.rect_split_seq_txtp(
                px, py, false, tr_h, bl_h, y_mode, y_delta, tx_mode, lam, emlam, ltxtp, &lcf, &lpred_arr,
            )
        } else {
            ltxtp
        };
        let idct = self.k.idct;
        let inv16x8 =
            |cf: &[i32; 128], q: &Quant| inv_rect_luma_128(&idct, cf, q, false, ltxtp);
        let split = ltxtp == RECT_SPLIT_TXTP;
        let mut split_cf = [[0i32; 64]; 2];
        let mut split_rec = [0u16; 128];
        if split {
            (split_cf, split_rec) = self.rect_split_commit(px, py, false, tr_h, bl_h, y_mode, y_delta, lam);
        }
        // --- chroma 16x8 (4:4:4): DC predict each plane ---
        let mut ccf = [[0i32; 128]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_16x8(&self.ctx.recon[plane], self.cfg.w, px, py, self.cfg.bd as i32);
            cpred[ci] = dc;
            let mut resid = [0i32; 128];
            self.k.rd.residual_dc(&mut resid, &self.cfg.src[plane], self.cfg.w, px, py, 16, 8, dc);
            let (mut q, qt) = self.k.dct.dct16x8_t(&resid, &self.ctx.cquant);
            self.chroma_rect_trellis(
                &mut q, &qt, cdcq, cacq, &SCAN_16X8, lam, 16, 8, plane, px, py,
            );
            self.k.rd.preserve_dc(&mut q[0], &resid[..]);
            ccf[ci] = q;
        }
        // block_skip iff all planes have no coefficients.
        // --- CfL: predict chroma from this sub-block's reconstructed luma.
        // Allowed here (16x8 <= 32x32); at 4:4:4 chroma is full resolution
        // so chroma-from-luma is worth far more than a luma mode search.
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 16);
        let mut cfl_ccf = [[0i32; 128]; 2];
        let mut cfl_pred = [[0i32; 128]; 2];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
        {
            let mut luma_rec = [0u16; 128];
            if split {
                luma_rec = split_rec;
            } else {
                let lrr_cfl = inv16x8(&lcf, &self.ctx.quant);
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, maxval);
            }
            let mut ac = [0i32; 128];
            self.k.intrapred.cfl_ac_444(&luma_rec, 16, 8, &mut ac);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cpred[ci];
                let csrc = self.src_blk(plane, px, py, 16, 8);
                let dcrr = self.k.idct.idct_dequant_16x8(&ccf[ci], &self.ctx.cquant);
                let s = self.k.rd.sse_recon(&[dc; 128], &dcrr, csrc, self.cfg.bd);
                dc_cost[ci] = rd_cost_i64(
                    s,
                    mlam,
                    self.chroma_rect_bits(&ccf[ci], &SCAN_16X8, 16, 8, plane, px, py),
                );
                let a = self
                    .k.intrapred
                    .cfl_best_alpha(&ac, csrc, dc, self.cfg.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 128];
                self.k.intrapred.cfl_pred(&mut cpr, &ac[..128], dc, a, self.cfg.bd);
                let mut resid = [0i32; 128];
                self.k.rd.residual_pred_blk(&mut resid, &cpr, csrc);
                let (mut q, qt) = self.k.dct.dct16x8_t(&resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q, &qt, cdcq, cacq, &SCAN_16X8, lam, 16, 8, plane, px, py,
                );
                let rr2 = self.k.idct.idct_dequant_16x8(&q, &self.ctx.cquant);
                let s2 = self.k.rd.sse_recon(&cpr, &rr2, csrc, self.cfg.bd);
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
        if self.cfg.speed.full_chroma_rdo() {
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
        let luma_zero = if split {
            split_cf.iter().all(|half| self.k.rd.all_zero_i32(half))
        } else {
            self.k.rd.all_zero_i32(&lcf)
        };
        let chroma_zero =
            self.k.rd.all_zero_i32(&ccf[0]) && self.k.rd.all_zero_i32(&ccf[1]);
        let block_skip = luma_zero && chroma_zero && rpal.is_none();
        // --- header: skip, delta-q (once), y_mode (DC), uv_mode (DC) ---
        Rect16PickH444 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            chosen_uv,
            cfl_opt,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_pred,
            use_cfl,
            cfl_pred,
            cpred,
        }
    }

    /// Emit one 16x8 half of a 16x16 HORZ partition (4:4:4) from its search result.
    fn emit_rect16_h444(&mut self, x8: usize, y8: usize, half: usize, pick: Rect16PickH444) {
        let Rect16PickH444 {
            block_skip,
            split,
            y_mode,
            yctx,
            y_delta,
            chosen_uv,
            cfl_opt,
            rpal,
            rfi,
            split_cf,
            ltxtp,
            lcf,
            tx_mode,
            lpred_arr,
            ccf,
            uv_pred,
            use_cfl,
            cfl_pred,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (px, py) = (x8 * 8, y8 * 8 + half * 8);
        let (bx4, by4) = (px / 4, py / 4);
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        // record the 16x8 footprint for the deblock filter: width 4 units,
        // height 2 units (vertical edges every 16, horizontal every 8).
        self.record_blk_rect(x8, y8 + half, 4, 2);
        self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
        if split {
            self.rect_split_record_tx(px, py, false);
        }
        self.ent.enc.encode_symbol(y_mode, &mut self.ent.cdfs.kf_y[yctx]);
        // Directional modes carry an angle_delta symbol (this search only
        // offers delta 0).
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.ent.enc
                .encode_symbol((y_delta + 3) as usize, &mut self.ent.cdfs.angle_delta[y_mode - V_PRED]);
        }
        self.emit_uv_mode(y_mode, chosen_uv, cfl_opt, 0, px, py, 16, 8);
        self.emit_palette_mode_info(px, py, 16, 8, y_mode, !self.cfg.is_mono(), rpal.as_ref(), None);
        if rpal.is_none() {
            self.emit_filter_intra(y_mode, 16, 8, rfi);
        }
        if let Some(p) = rpal.as_ref() {
            self.emit_palette_map(p);
        }
        self.code_tx_depth(px, py, 16, 8, split as usize);
        // footprint update: skip/mode over 4 wide x 2 tall units.
        let sv = block_skip as u8;
        self.ctx.a_skip[bx4..bx4 + 4].fill(sv);
        self.ctx.l_skip[by4..by4 + 2].fill(sv);
        self.ctx.a_mode[bx4..bx4 + 4].fill(y_mode as u8);
        self.ctx.l_mode[by4..by4 + 2].fill(y_mode as u8);
        // --- luma coeffs (RTX_16X8) ---
        if split {
            // Two TX_8X8 in decode order; luma was reconstructed above.
            self.rect_split_emit(px, py, false, y_mode, block_skip, &split_cf);
        } else {
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                if ltxtp == 2 || ltxtp == 3 {
                    encode_rect_coeffs_1d(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        16,
                        ltxtp == 2,
                        sk,
                        ds,
                        tx_mode,
                    )
                } else {
                    encode_16x8_luma_coeffs(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &lcf,
                        sk,
                        ds,
                        tx_mode,
                        ltxtp,
                    )
                }
            };
            self.ctx.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.ctx.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            // reconstruct luma
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                inv_rect_luma_128(&self.k.idct, &lcf, &self.ctx.quant, false, ltxtp)
            };
            for ry in 0..8 {
                let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
                recon_add_pred(
                    &mut drow[..16],
                    &lpred_arr[ry * 16..],
                    &lrr[ry * 16..],
                    maxval,
                );
            }
        }
        // --- chroma coeffs + reconstruct (4:4:4, both planes RTX_16X8) ---
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_chroma(plane, bx4, by4);
                let ds = self.dc_sign_ctx_16x8_chroma(plane, bx4, by4);
                encode_16x8_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            };
            self.ctx.a_coef[plane][bx4..bx4 + 4].fill(cres_ctx);
            self.ctx.l_coef[plane][by4..by4 + 2].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 128]
            } else if chosen_uv != DC_PRED {
                // Directional/smooth uv mode: decoder-derived chroma tx.
                inv_chroma_16x8(
                    &self.k.idct,
                    chroma_tx_for_mode(chosen_uv),
                    &ccf[ci],
                    &self.ctx.cquant,
                )
            } else {
                self.k.idct.idct_dequant_16x8(&ccf[ci], &self.ctx.cquant)
            };
            for ry in 0..8 {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.w + px..];
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

    /// Code one 16x8 half of a 16x16 HORZ partition (4:4:4): search, then emit.
    fn code_block16_horz_444_half(&mut self, x8: usize, y8: usize, half: usize, thr: bool, lhb: bool) {
        let pick = self.pick_rect16_h444(x8, y8, half, thr, lhb);
        self.emit_rect16_h444(x8, y8, half, pick);
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
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]);
        }
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        // Sub-transforms are coded in sequence, each seeing the previous one's
        // coefficient result context. Simulate that progression here and
        // restore it before returning (external review round 2, finding 3).
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 4] = *self.ctx.a_coef[0][bx4_0..].first_chunk().unwrap();
        let saved_l: [u8; 4] = *self.ctx.l_coef[0][by4_0..].first_chunk().unwrap();
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
                let d = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                pred = [d; 64];
            } else {
                self.k.intrapred.predict_nd_ad(
                    mode,
                    delta,
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
            let mut resid = [0i32; 64];
            self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, bx, by, 8, 8);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 2, 2);
            let (mut cf, tf) = self.k.dct.dct8x8_t(&resid, &self.ctx.quant);
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
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            let rr = self.k.idct.idct_dequant_8x8(&cf, &self.ctx.quant);
            let dct_sse = sse_recon::<64, 8>(&self.k.rd, &pred, &rr, self.src_blk(0, bx, by, 8, 8), self.cfg.bd);
            let dct_bits =
                self.luma_bits_ctx_bounded(&cf, &SCAN_8X8, 8, bx, by, mode, 1, sk, ds, f32::INFINITY);
            let mut best = (cf, rr, dct_sse, dct_bits, 1u8);
            let quad_types: &[u8] = if self.cfg.is_420() || self.cfg.is_mono() {
                &[]
            } else {
                &[4, 5, 6, 0, 2, 3]
            };
            for &txtp in quad_types {
                let (mut acf, atf) = match txtp {
                    4 => self.k.dct.adst8x8_t(&resid, &self.ctx.quant),
                    5 => self.k.dct.adstdct8x8_t(&resid, &self.ctx.quant),
                    6 => self.k.dct.dctadst8x8_t(&resid, &self.ctx.quant),
                    0 => self.k.dct.idtx8x8_t(&resid, &self.ctx.quant),
                    2 => self.k.dct.fvdct8x8_t(&resid, &self.ctx.quant),
                    _ => self.k.dct.fhdct8x8_t(&resid, &self.ctx.quant),
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
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let arr = match txtp {
                    4 => self.k.idct.iadst_dequant_8x8(&acf, &self.ctx.quant),
                    5 => self.k.idct.iadstdct_dequant_8x8(&acf, &self.ctx.quant),
                    6 => self.k.idct.idctadst_dequant_8x8(&acf, &self.ctx.quant),
                    0 => self.k.idct.iidentity_dequant_8x8(&acf, &self.ctx.quant),
                    2 => self.k.idct.ivdct_dequant_8x8(&acf, &self.ctx.quant),
                    _ => self.k.idct.ihdct_dequant_8x8(&acf, &self.ctx.quant),
                };
                let asse = sse_recon::<64, 8>(&self.k.rd, &pred, &arr, self.src_blk(0, bx, by, 8, 8), self.cfg.bd);
                // Gate BEFORE pricing: `abits` is only consumed by the
                // comparison below, so a failed SSE gate never needs the
                // rate. The exact-abort bound then caps the CDF walk at the
                // highest bit count that could still win (byte-identical —
                // see real_block_bits_bounded).
                let gate = if txtp >= 4 {
                    asse <= dct_sse + self.tx_guard(dct_sse)
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
            self.ctx.a_coef[0][qbx4..qbx4 + 2].fill(res_ctx);
            self.ctx.l_coef[0][qby4..qby4 + 2].fill(res_ctx);
            // Write the quadrant's candidate recon so later quadrants predict
            // from it (restored below).
            self.k.rd.reconstruct(
                &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                self.cfg.w,
                Some((&mut rec[sy * 16 + sx..], 16)),
                &pred,
                &brr,
                8,
                8,
                self.cfg.bd,
            );
            cf4[qi * 64..qi * 64 + 64].copy_from_slice(&bcf);
        }
        for ry in 0..16 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        self.ctx.a_coef[0][bx4_0..bx4_0 + 4].copy_from_slice(&saved_a);
        self.ctx.l_coef[0][by4_0..by4_0 + 4].copy_from_slice(&saved_l);
        (*cf4, *rec, sse_sum, bits_sum, txtps)
    }

    #[cfg(debug_assertions)]
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
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]);
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
                    let d = self.k.intrapred.dc_pred_4x4(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                    pred = [d; 16];
                } else {
                    self.k.intrapred.predict_nd_ad(
                        mode,
                        delta,
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
                let cfq = lcf[ti * 16..ti * 16 + 16].first_chunk::<16>().unwrap();
                let rr = match txtps[ti] {
                    4 => self.k.idct.iadst_dequant_4x4(cfq, &self.ctx.quant),
                    0 => self.k.idct.iidentity_dequant_4x4(cfq, &self.ctx.quant),
                    2 => self.k.idct.ivdct_dequant_4x4(cfq, &self.ctx.quant),
                    3 => self.k.idct.ihdct_dequant_4x4(cfq, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_4x4(cfq, &self.ctx.quant),
                };
                self.k.rd.reconstruct(
                    &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                    self.cfg.w,
                    Some((&mut rec[j * 4 * 16 + i * 4..], 16)),
                    &pred,
                    &rr,
                    4,
                    4,
                    self.cfg.bd,
                );
            }
        }
        for ry in 0..16 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]
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
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]);
        }
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
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
                    let d = self.k.intrapred.dc_pred_4x4(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                    pred = [d; 16];
                } else {
                    self.k.intrapred.predict_nd_ad(
                        mode,
                        delta,
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
                let mut resid = [0i32; 16];
                self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, bx, by, 4, 4);
                let (mut dcf, dtf) = self.k.dct.dct4x4_t(&resid, &self.ctx.quant);
                trellis_optimize(&mut dcf, &dtf, dcq, acq, &SCAN_4X4, lam);
                let drr = self.k.idct.idct_dequant_4x4(&dcf, &self.ctx.quant);
                let dct_sse =
                    sse_recon::<16, 4>(&self.k.rd, &pred, &drr, self.src_blk(0, bx, by, 4, 4), self.cfg.bd);
                let dct_bits = self.luma_bits(&dcf, &SCAN_4X4, 4, bx, by, mode, 1);
                let mut best = (dcf, drr, dct_sse, dct_bits, 1u8);
                for txtp in [4u8, 0, 2, 3] {
                    let (mut acf, atf) = match txtp {
                        4 => self.k.dct.adst4x4_t(&resid, &self.ctx.quant),
                        0 => self.k.dct.idtx4x4_t(&resid, &self.ctx.quant),
                        2 => self.k.dct.fvdct4x4_t(&resid, &self.ctx.quant),
                        _ => self.k.dct.fhdct4x4_t(&resid, &self.ctx.quant),
                    };
                    if txtp == 4 {
                        trellis_optimize(&mut acf, &atf, dcq, acq, &SCAN_4X4, lam);
                    }
                    let arr = match txtp {
                        4 => self.k.idct.iadst_dequant_4x4(&acf, &self.ctx.quant),
                        0 => self.k.idct.iidentity_dequant_4x4(&acf, &self.ctx.quant),
                        2 => self.k.idct.ivdct_dequant_4x4(&acf, &self.ctx.quant),
                        _ => self.k.idct.ihdct_dequant_4x4(&acf, &self.ctx.quant),
                    };
                    let asse =
                        sse_recon::<16, 4>(&self.k.rd, &pred, &arr, self.src_blk(0, bx, by, 4, 4), self.cfg.bd);
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
                    if asse <= dct_sse + self.tx_guard(dct_sse)
                        && rd_cost_i64(asse, rd_lam, abits) < rd_cost_i64(best.2, rd_lam, best.3)
                    {
                        best = (acf, arr, asse, abits, txtp);
                    }
                }
                let (bcf, brr, bsse, bbits, btxtp) = best;
                txtps[ti] = btxtp;
                sse_sum += bsse;
                bits_sum += bbits;
                self.k.rd.reconstruct(
                    &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                    self.cfg.w,
                    Some((&mut rec[j * 4 * 16 + i * 4..], 16)),
                    &pred,
                    &brr,
                    4,
                    4,
                    self.cfg.bd,
                );
                cf16[ti * 16..ti * 16 + 16].copy_from_slice(&bcf);
            }
        }
        for ry in 0..16 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        (*cf16, *rec, sse_sum, bits_sum, txtps)
    }

    #[cfg(debug_assertions)]
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
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]);
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
                let d = self.k.intrapred.dc_pred_8x8(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                pred = [d; 64];
            } else {
                self.k.intrapred.predict_nd_ad(
                    mode,
                    delta,
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
            let mut cfq = [0i32; 64];
            cfq.copy_from_slice(&lcf[qi * 64..qi * 64 + 64]);
            let rr = match txtps[qi] {
                4 => self.k.idct.iadst_dequant_8x8(&cfq, &self.ctx.quant),
                5 => self.k.idct.iadstdct_dequant_8x8(&cfq, &self.ctx.quant),
                6 => self.k.idct.idctadst_dequant_8x8(&cfq, &self.ctx.quant),
                0 => self.k.idct.iidentity_dequant_8x8(&cfq, &self.ctx.quant),
                2 => self.k.idct.ivdct_dequant_8x8(&cfq, &self.ctx.quant),
                3 => self.k.idct.ihdct_dequant_8x8(&cfq, &self.ctx.quant),
                _ => self.k.idct.idct_dequant_8x8(&cfq, &self.ctx.quant),
            };
            self.k.rd.reconstruct(
                &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                self.cfg.w,
                Some((&mut rec[sy * 16 + sx..], 16)),
                &pred,
                &rr,
                8,
                8,
                self.cfg.bd,
            );
        }
        for ry in 0..16 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..16]
                .copy_from_slice(&saved[ry * 16..ry * 16 + 16]);
        }
        *rec
    }

    /// Luma search of a 16x16 leaf: prediction mode (with angle delta, filter
    /// intra or palette), transform type and transform split. Leaves the tile
    /// unchanged: `&mut` only because the TX-split trials rebuild the block
    /// in the tile reconstruction and restore it.
    fn pick_luma16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) -> Luma16Pick {
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let lam = self.trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 16);
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 16);
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        // emit_prdo/emit_mlam already carry the perceptual scale; the extra
        // multiplication here ran square leaves at scale^2 while rect leaves,
        // 64s and the partition decision used scale^1 (external review
        // 2026-07-27, finding 3). `prdo` stays for the coupled chroma path.
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let mut best_mode = DC_PRED;
        let mut best_delta: i32 = 0;
        let mut txtp16: u8 = 0; // 0=DCT_DCT 1=ADST_ADST 2=ADST_DCT 3=DCT_ADST
        let mut s8_txtps = [1u8; 4];
        let mut s16_txtps = [1u8; 16];
        let mut lpred_arr = self.sbuf_i256();
        let mut lcf = self.sbuf_i256();
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        // Decoder-exact reconstruction of a TX-split winner (`txtp16` 4 / 6).
        let mut split_rec: Option<[u16; 256]> = None;
        let mut luma_beam: [Option<Luma16BeamCandidate>; JOINT_LARGE_BEAM] =
            std::array::from_fn(|_| None);
        let mut ltf = self.sbuf_f256(); // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.cfg.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        let joint_large =
            !self.cfg.is_mono() && self.cfg.speed.at_least_slow() && joint_luma_uv_large_enabled();
        let mut dir_top = [(f32::INFINITY, DC_PRED); 2];
        let mode_shortlist = self.rank_luma_candidates::<256>(
                modes,
                px,
                py,
                16,
                16,
                have_tr,
                have_bl,
                self.luma_mode_budget_eff(),
            );
        for &(m, delta) in &mode_shortlist {
            let mut pred = self.sbuf_i256();
            if m == DC_PRED {
                let d = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                *pred = [d; 256];
            } else {
                self.k.intrapred.predict_nd_ad(
                    m,
                    delta,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(px, py),
                    &mut pred[..],
                    self.cfg.bd,
                );
            }
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_pred(
                &mut resid[..],
                &pred[..],
                &self.cfg.src[0],
                self.cfg.w,
                px,
                py,
                16,
                16,
            );
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                sse_recon::<256, 16>(&self.k.rd, &pred, rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd)
            };
            let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
            if self.cfg.speed.per_candidate_rdoq_av1() {
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
                    self.ctx.quant.qm_level(),
                    self.ctx.quant.qidx() as i32,
                );
            }
            let sse = blk_sse16(&self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant));
            let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, m, 1);
            let syntax_bits = self.luma_intra_syntax_bits(px, py, 16, 16, m, delta);
            let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
            if delta == 0 && (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                if cost < dir_top[0].0 {
                    dir_top[1] = dir_top[0];
                    dir_top[0] = (cost, m);
                } else if cost < dir_top[1].0 {
                    dir_top[1] = (cost, m);
                }
            }
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                best_delta = delta;
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
                        delta,
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
        if self.try_palette()
            && let Some(hist) = block_color_histogram(&self.cfg.src[0], self.cfg.w, px, py, 16, 16)
        {
            for (palette, pred) in
                self.rank_luma_palette_candidates::<256>(&hist, px, py, 16, 16, mlam * K_PART_LAMBDA)
            {
                let mut resid = self.sbuf_i256();
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &pred,
                    &self.cfg.src[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
                if self.cfg.speed.per_candidate_rdoq_av1() {
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
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                let sse = sse_recon::<256, 16>(&self.k.rd, &pred, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
                let coeff_bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, DC_PRED, 1);
                let bits = coeff_bits
                    + self.mode_bits(px, py, DC_PRED)
                    + self.palette_rate_bits(px, py, &palette);
                let cost = rd_cost_i64(sse, mlam, bits);
                if cost < best_eff {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    best_delta = 0;
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
                            delta: 0,
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
                best_delta = candidate.delta;
                *lpred_arr = *candidate.pred;
                *lcf = *candidate.cf;
                *ltf = *candidate.tf;
                best_dct_sse = candidate.sse;
                best_dct_bits = candidate.bits;
                best_filter_intra = None;
                best_palette16 = candidate.palette;
            }
        }
        if self.cfg.speed.at_least_slow() {
            let bsize = av1_block_size_index(16, 16);
            let mut fi_preds = [[0i32; 256]; 5];
            for (fi_si, &filter_mode) in self
                .rank_filter_intra_modes::<256>(px, py, 16, 16, self.cfg.speed.filter_intra_refine_budget(self.ss444()), &mut fi_preds)
                .iter()
                .enumerate()
            {
                let mut pred = self.sbuf_i256();
                pred[..].copy_from_slice(&fi_preds[fi_si]);
                let mut resid = self.sbuf_i256();
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &pred[..],
                    &self.cfg.src[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
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
                    self.ctx.quant.qm_level(),
                    self.ctx.quant.qidx() as i32,
                );
                let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                let sse = sse_recon::<256, 16>(&self.k.rd, &pred, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
                // The admission below needs `sse <= best_dct_sse` (unless the RD-only
                // study knob is on): a candidate that
                // already lost on SSE skips the entropy-accurate rate pass.
                if !crate::tuning::get().filter_intra_rd_only && sse > best_dct_sse {
                    continue;
                }
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, DC_PRED, 1);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if (crate::tuning::get().filter_intra_rd_only || sse <= best_dct_sse) && cost < best_eff
                {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    best_delta = 0;
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
        // Refine the directional winner, retaining its current angle cost if
        // it came from the global angular shortlist.
        if self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx)
            && self.angle_delta_refines(best_mode)
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
            let mut best_ad_cost = rd_cost_i64(
                best_dct_sse,
                mlam,
                best_dct_bits + cdf_cost(&ad_cdf, (best_delta + 3) as usize),
            );
            let mut ad_pred0 = self.sbuf_i256();
            let mut ad_pred1 = self.sbuf_i256();
            let mut ad_scratch = self.sbuf_i256();
            let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
            let ad_cands = self.ad16_candidates(best_mode, px, py, have_tr, have_bl, &mut ad_preds);
            for (d, pred) in &ad_cands {
                let d = *d;
                let pred: &[i32; 256] = pred;
                let mut resid = self.sbuf_i256();
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &pred[..],
                    &self.cfg.src[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
                if self.cfg.speed.per_candidate_rdoq_av1() {
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
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                let sse = sse_recon::<256, 16>(&self.k.rd, pred, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, best_mode, 1);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
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
        // Runner-up angle-delta refinement (`ad_runner`): the best directional
        // mode that did NOT win (the winner may be non-directional) tries its
        // two SATD-ranked deltas against the winner's full cost.
        let runner = if dir_top[0].1 == best_mode { dir_top[1] } else { dir_top[0] };
        if crate::tuning::get().ad_runner
            && self.cfg.speed.try_angle_deltas_av1(16, self.cfg.base_q_idx)
            && runner.0.is_finite()
            && self.angle_delta_refines(runner.1)
            && best_filter_intra.is_none()
            && best_palette16.is_none()
        {
            let m2 = runner.1;
            let cur_extra = if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
                cdf_cost(&self.dcdf().angle_delta[best_mode - V_PRED], (best_delta + 3) as usize)
            } else if best_mode == DC_PRED {
                cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(16, 16)], 0)
            } else {
                0.0
            };
            let mut best_cost = rd_cost_i64(
                best_dct_sse,
                mlam,
                best_dct_bits + self.mode_bits(px, py, best_mode) + cur_extra,
            );
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[m2 - V_PRED]);
            let m2_bits = self.mode_bits(px, py, m2);
            let mut ad_pred0 = self.sbuf_i256();
            let mut ad_pred1 = self.sbuf_i256();
            let mut ad_scratch = self.sbuf_i256();
            let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
            let ad_cands = self.ad16_candidates(m2, px, py, have_tr, have_bl, &mut ad_preds);
            for (d, pred) in &ad_cands {
                let d = *d;
                let pred: &[i32; 256] = pred;
                let mut resid = self.sbuf_i256();
                self.k.rd
                    .residual_pred(&mut resid[..], &pred[..], &self.cfg.src[0], self.cfg.w, px, py, 16, 16);
                let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
                if self.cfg.speed.per_candidate_rdoq_av1() {
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
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
                let sse = sse_recon::<256, 16>(&self.k.rd, pred, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
                let bits = self.luma_bits(&cf, &SCAN_16X16, 16, px, py, m2, 1);
                let cost =
                    rd_cost_i64(sse, mlam, bits + m2_bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_cost {
                    best_cost = cost;
                    best_mode = m2;
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
        if !self.cfg.speed.per_candidate_rdoq_av1() {
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
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            // Re-price the DCT baseline after RDOQ: the ADST/IDTX legs below
            // compare trellised costs against it (external review round 3).
            let rr = self.k.idct.idct_dequant_16x16(&lcf, &self.ctx.quant);
            best_dct_sse =
                sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
            best_dct_bits = self.luma_bits(&lcf[..], &SCAN_16X16, 16, px, py, best_mode, 1);
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style). Palette
        // winners skip it (ADST on palette residuals measured 4:2:0 holdout
        // +0.13% for nothing on screenshots); they take IDTX below instead.
        if best_palette16.is_none() && self.cfg.speed.try_adst() {
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_pred(
                &mut resid[..],
                &lpred_arr[..],
                &self.cfg.src[0],
                self.cfg.w,
                px,
                py,
                16,
                16,
            );
            let (mut acf, atf) = self.k.dct.adst16x16_t(&resid, &self.ctx.quant);
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
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            let rr = self.k.idct.iadst_dequant_16x16(&acf, &self.ctx.quant);
            let asse = sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
            let base_rd = rd_cost_i64(best_dct_sse, mlam, best_dct_bits);
            let bits_bound = (base_rd - asse as f32) / mlam;
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
            if asse <= best_dct_sse + self.tx_guard(best_dct_sse) && candidate_rd < base_rd
            {
                *lcf = acf;
                txtp16 = 1;
            }
        }
        // Asymmetric-ADST refinement (ADST_DCT / DCT_ADST) for TX_16X16, same
        // rationale as the 8x8 path. Competes with the running tx winner.
        if best_palette16.is_none() && self.cfg.speed.try_adst() {
            let mut best_txtp16_sse = if txtp16 == 1 { i64::MAX } else { best_dct_sse };
            let mut best_txtp16_bits = best_dct_bits;
            if txtp16 == 1 {
                // recompute the ADST_ADST winner cost as the bar to beat
                let rr = self.k.idct.iadst_dequant_16x16(&lcf, &self.ctx.quant);
                best_txtp16_sse =
                    sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
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
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &lpred_arr[..],
                    &self.cfg.src[0],
                    self.cfg.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut acf, atf) = if fwd_dctadst {
                    self.k.dct.dctadst16x16_t(&resid, &self.ctx.quant)
                } else {
                    self.k.dct.adstdct16x16_t(&resid, &self.ctx.quant)
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
                    self.ctx.quant.qm_level(),
                    self.ctx.quant.qidx() as i32,
                );
                let rr = if inv_dctadst {
                    self.k.idct.idctadst_dequant_16x16(&acf, &self.ctx.quant)
                } else {
                    self.k.idct.iadstdct_dequant_16x16(&acf, &self.ctx.quant)
                };
                let asse =
                    sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
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
                if asse <= best_dct_sse + self.tx_guard(best_dct_sse) && candidate_rd < base_rd
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
        //
        // Palette winners take it too (aom searches the tx type over palette
        // residuals, and most of its palette blocks end up residual-free). Their residual is a sparse set of anti-aliasing spikes
        // the palette missed; under DCT it rings across the flat background
        // the palette had reproduced exactly — the grey smudges around text
        // that made 4:2:0 screenshots look worse WITH screen-content tools
        // (t_dark crop: 4:2:0 -5.1%, high band now beats --no-screen-content).
        if self.cfg.speed.try_adst() {
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_pred(
                &mut resid[..],
                &lpred_arr[..],
                &self.cfg.src[0],
                self.cfg.w,
                px,
                py,
                16,
                16,
            );
            let (icf, _itf) = self.k.dct.idtx16x16_t(&resid, &self.ctx.quant);
            let rr = self.k.idct.iidentity_dequant_16x16(&icf, &self.ctx.quant);
            let isse = sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
            let ibits = self.luma_bits(&icf, &SCAN_16X16, 16, px, py, best_mode, IDTX_TX16_IDX);
            // Current winner's (sse, bits) under its committed tx.
            let (cur_sse, cur_bits) = {
                let rrw = match txtp16 {
                    1 => self.k.idct.iadst_dequant_16x16(&lcf, &self.ctx.quant),
                    2 => self.k.idct.iadstdct_dequant_16x16(&lcf, &self.ctx.quant),
                    3 => self.k.idct.idctadst_dequant_16x16(&lcf, &self.ctx.quant),
                    _ => self.k.idct.idct_dequant_16x16(&lcf, &self.ctx.quant),
                };
                let sse =
                    sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rrw, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
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
            if isse <= cur_sse + self.tx_guard(cur_sse) && candidate_rd < base_rd {
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
        if best_filter_intra.is_none() && best_palette16.is_none() {
            // Final (distortion, rate) of the whole-TX_16X16 winner, from its
            // committed coefficients (the sub-search locals are scope-bound).
            let rr16 = match txtp16 {
                1 => self.k.idct.iadst_dequant_16x16(&lcf, &self.ctx.quant),
                2 => self.k.idct.iadstdct_dequant_16x16(&lcf, &self.ctx.quant),
                3 => self.k.idct.idctadst_dequant_16x16(&lcf, &self.ctx.quant),
                5 => self.k.idct.iidentity_dequant_16x16(&lcf, &self.ctx.quant),
                _ => self.k.idct.idct_dequant_16x16(&lcf, &self.ctx.quant),
            };
            let none_sse =
                sse_recon::<256, 16>(&self.k.rd, &lpred_arr, &rr16, self.src_blk(0, px, py, 16, 16), self.cfg.bd);
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
            let (cf4, rec4, mut sse_s, mut bits_s, split_txtps) =
                self.split16_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam, mlam);
            let d0_bits = self.tx_depth_bits(px, py, 16, 16, 0);
            let d1_bits = self.tx_depth_bits(px, py, 16, 16, 1);
            let base_rd = rd_cost_i64(none_sse, mlam, none_bits + d0_bits);
            let candidate_rd = rd_cost_i64(sse_s, mlam, bits_s + d1_bits);
            let banding = self.banding_risk(px, py, 16);
            let guarded_take = if banding {
                sse_s <= none_sse + (none_sse >> 2)
            } else {
                candidate_rd < base_rd
            };
            if guarded_take {
                txtp16 = 4;
                s8_txtps = split_txtps;
                *lcf = cf4;
                split_rec = Some(rec4);
            }
            // DC re-decision under the split. The mode above was chosen by its
            // whole-TX_16X16 cost, but the block may be coded as four TX_8X8,
            // each re-predicted from its own neighbours — which helps DC (a
            // per-8x8 DC) far more than a directional mode. Medium cannot fall
            // back to 8x8 partitions (`min_size_16`), so on grainy full-res
            // content diagonals won at TX_16X16 and were then split anyway:
            // digital_art_portrait 444 q80 Medium coded +9.6% bytes at -0.25
            // SS2 vs Fast. Compared with mode syntax on both sides.
            if !banding
                && self.cfg.speed.split16_dc_redecide()
                && (best_mode, best_delta) != (DC_PRED, 0)
            {
                let cur_syn = self.luma_intra_syntax_bits(px, py, 16, 16, best_mode, best_delta);
                let cur_rd = if txtp16 == 4 {
                    rd_cost_i64(sse_s, mlam, bits_s + d1_bits + cur_syn)
                } else {
                    rd_cost_i64(none_sse, mlam, none_bits + d0_bits + cur_syn)
                };
                let (dcf, drec, dsse, dbits, dtxtps) =
                    self.split16_luma_try(px, py, DC_PRED, 0, have_tr, have_bl, lam, mlam);
                let dc_syn = self.luma_intra_syntax_bits(px, py, 16, 16, DC_PRED, 0);
                if rd_cost_i64(dsse, mlam, dbits + d1_bits + dc_syn) < cur_rd {
                    best_mode = DC_PRED;
                    best_delta = 0;
                    txtp16 = 4;
                    s8_txtps = dtxtps;
                    *lcf = dcf;
                    split_rec = Some(drec);
                    (sse_s, bits_s) = (dsse, dbits);
                    let d = self
                        .k.intrapred
                        .dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                    *lpred_arr = [d; 256];
                }
            }
            // Depth-2 candidate: sixteen TX_4X4. Competes against the current
            // winner (whole or depth-1) on plain RD; ~2.5 bit allowance for
            // the deeper tx_depth symbol.
            let (cf16, rec2, sse_2, bits_2, d2_txtps) =
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
                split_rec = Some(rec2);
            }
        }
        Luma16Pick {
            cf: lcf,
            pred: lpred_arr,
            y_mode: best_mode,
            txtp16,
            s8_txtps,
            s16_txtps,
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette16,
            split_rec,
        }
    }

    /// Code a 16x16 leaf: search luma, then chroma for that luma, then emit.
    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let Luma16Pick {
            cf: lcf,
            pred: lpred_arr,
            y_mode: best_mode,
            txtp16,
            s8_txtps,
            s16_txtps,
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette16,
            split_rec,
        } = self.pick_luma16(x8, y8, have_tr, have_bl);
        #[cfg(debug_assertions)]
        if let Some(rec) = &split_rec {
            let again = self.split16_luma_recon_from_cf(
                x8 * 8,
                y8 * 8,
                best_mode,
                best_delta,
                have_tr,
                have_bl,
                &lcf,
                s8_txtps,
                s16_txtps,
                txtp16 == 6,
            );
            debug_assert!(*rec == again, "split-TX trial recon differs from the committed-coefficient recon");
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
            let nc4 = self.cfg.w / 4;
            let (bx4, by4) = (x8 * 2, y8 * 2);
            for uy in 0..4 {
                for ux in 0..4 {
                    let cell = (by4 + uy) * nc4 + (bx4 + ux);
                    self.ctx.maps.blk4[cell] = 1;
                    self.ctx.maps.blk4h[cell] = 1;
                    self.ctx.maps.blk4v[cell] = true;
                    self.ctx.maps.blk4t[cell] = true;
                }
            }
        }
        let l = Luma16 {
            cf: &lcf,
            pred: &lpred_arr,
            y_mode: best_mode,
            luma_zero: self.k.rd.all_zero_i32(&lcf[..]),
            txtp16,
            s8_txtps,
            s16_txtps,
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette16.as_ref(),
            split_rec: split_rec.as_ref(),
            have_tr,
            have_bl,
        };
        match self.cfg.chroma {
            Chroma::Mono => {
                let skip = l.palette.is_none() && l.luma_zero;
                self.code_header_luma16(x8, y8, &l, skip, DC_PRED, None, 0, None);
            }
            Chroma::I420 => {
                let uv = self.pick_uv16_420(x8, y8, &l);
                self.emit_block16_420(x8, y8, &l, uv);
            }
            Chroma::I422 => {
                let uv = self.pick_uv16_422(x8, y8, &l);
                self.emit_block16_422(x8, y8, &l, uv);
            }
            Chroma::I444 => {
                let uv = self.pick_uv16_444(x8, y8, &l);
                self.emit_block16_444(x8, y8, &l, uv);
            }
        }
    }
}
