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

/// TEMPORARY diagnostic knob for the 32x32 split-signal multiplier (1.0 = the
/// corrected pricing, 4.0 = the historical double-count).
fn split32_signal_mult() -> f32 {
    static M: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("MT_SPLIT32_MULT")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1.0)
    })
}

impl<'a> LossyTile<'a> {
    /// Raster quadrants of a 32x32 block as (dx, dy) pixel offsets (TX_16X16).
    const Q32: [(usize, usize); 4] = [(0, 0), (16, 0), (0, 16), (16, 16)];

    /// 4:2:2 chroma txb_skip (all_zero) context for an RTX_4X8 block (1 unit
    /// wide, 2 units tall; `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40) as usize;
        let cl = (l[by4c] != 0x40 || l[by4c + 1] != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_4X8: 1 unit wide, 2 tall, baseline -3.
    fn dc_sign_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 + (l[by4c + 1] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_8X16 block (2 units wide, 4 tall;
    /// chroma tx == chroma block so ctx_offset = 7): `7 + a_nz + l_nz`, where each
    /// term ORs over the units the block spans.
    fn skip_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = (a[bx4c] != 0x40 || a[bx4c + 1] != 0x40) as usize;
        let cl =
            (l[by4c] != 0x40 || l[by4c + 1] != 0x40 || l[by4c + 2] != 0x40 || l[by4c + 3] != 0x40)
                as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_8X16: 2 units wide, 4 tall, baseline -6.
    fn dc_sign_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32
            + (a[bx4c + 1] >> 6) as i32
            + (l[by4c] >> 6) as i32
            + (l[by4c + 1] >> 6) as i32
            + (l[by4c + 2] >> 6) as i32
            + (l[by4c + 3] >> 6) as i32
            - 6;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_16X32 block (4 units wide, 8 tall).
    fn skip_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4c..bx4c + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4c..by4c + 8].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_16X32: 4 units wide, 8 tall, baseline -12.
    fn dc_sign_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4c..bx4c + 4].iter().map(|x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4c..by4c + 8].iter().map(|x| (x >> 6) as i32).sum();
        let s = suma + suml - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:0 chroma txb_skip context for a TX_4X4 block (1 unit wide and tall;
    /// `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        7 + (a[bx4c] != 0x40) as usize + (l[by4c] != 0x40) as usize
    }

    /// 4:2:0 chroma dc_sign context for TX_4X4: 1 unit each side, baseline -2.
    fn dc_sign_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a TX_32X32 (8-unit footprint, baseline -16).
    fn dc_sign_ctx_32(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let suma: i32 = a[bx4..bx4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let suml: i32 = l[by4..by4 + 8].iter().map(|&x| (x >> 6) as i32).sum();
        let s = suma + suml - 16;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a TX_32X32 (8-unit footprint). Luma (max tx in a
    /// 32x32 block) is always ctx 0; chroma uses `7 + above_nz + left_nz`.
    fn skip_ctx_32(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = a[bx4..bx4 + 8].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 8].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn choose_rect8(&self, _x8: usize, _y8: usize) -> Part16 {
        Part16::None
    }

    fn rd_cost_rect32(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let (lw, lh) = if vert { (16usize, 32usize) } else { (32, 16) };
        let mut total = rate_cost(mlam, SPLIT_SIGNAL_BITS);
        for half in 0..2 {
            let (sx, sy) = if vert {
                (px + half * 16, py)
            } else {
                (px, py + half * 16)
            };
            let dc = if vert {
                dc_pred_16x32(&self.recon[0], self.w, sx, sy, self.bd as i32)
            } else {
                dc_pred_32x16(&self.recon[0], self.w, sx, sy, self.bd as i32)
            };
            let mut resid = [0i32; 512];
            crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, sx, sy, lw, lh, dc);
            let (mut cf, tf) = if vert {
                dct16x32_t(&resid, &self.quant)
            } else {
                dct32x16_t(&resid, &self.quant)
            };
            let scan: &[u32] = if vert { &SCAN_16X32 } else { &SCAN_32X16 };
            trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
            let rr = if vert {
                idct_dequant_16x32(&cf, &self.quant)
            } else {
                idct_dequant_32x16(&cf, &self.quant)
            };
            let distortion =
                self.luma_partition_distortion(sx, sy, lw, lh, self.quant.ac_q() as f32, |i| {
                    dc + rr[i]
                });
            total += crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, scan));
        }
        total
    }

    /// 32x32 shared-partition choice. Keep the established luma preselection
    /// between NONE and SPLIT, then include derived chroma R-D when comparing
    /// that square candidate with HORZ/VERT.
    fn choose_rect32(&self, x8: usize, y8: usize) -> Part16 {
        // Monochrome has no whole-32 emitter (`code_block32` dispatches to the
        // 4:4:4 helper, which indexes the absent chroma planes), so 4:0:0 always
        // splits — as it did before the NONE-vs-SPLIT R-D rework.
        if self.mono {
            return Part16::Split;
        }
        let unpruned_rect32 = self.base_q_idx >= UNPRUNED_RECT32_MIN_QINDEX;
        let (px, py) = (x8 * 8, y8 * 8);
        let prdo = self.perceptual_rd_scale(px, py, 32);
        let block_var = self.luma_variance(px, py, 32, 32);
        let chroma_none = if self.mono {
            0.0
        } else {
            self.rd_cost_chroma_partition(px, py, 32, Part16::None, prdo)
        };
        let chroma_split = if self.mono {
            0.0
        } else {
            self.rd_cost_chroma_partition(px, py, 32, Part16::Split, prdo)
        };
        // Small split-favoring bias: the SATD distortion proxy undervalues the
        // detail a single 32x32 loses when it merges four busier 16x16s, so a
        // pure comparison over-merges on textured content (SSIMULACRA2 penalizes
        // it). This is a mild, symmetric thumb (not the old prefilter that
        // skipped the comparison entirely).
        let rd_none = (self.rd_cost_none32(px, py, prdo) + chroma_none) * NONE32_SPLIT_BIAS;
        let rd_split = self.rd_cost_split32(px, py, prdo) + chroma_split;
        // Rectangular candidates stay pruned at high quality (their DC-only 16x8
        // sub-blocks lose to the square path's full mode search there).
        if self.mono || (!unpruned_rect32 && self.quant.ac_q() < AC_Q_HORZ_MIN) {
            return if rd_none <= rd_split {
                Part16::None
            } else {
                Part16::Split
            };
        }
        let horz_on = HORZ_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let vert_on = !self.ss422 && VERT_ENABLED.load(std::sync::atomic::Ordering::Relaxed);
        let mut rd_h = f32::INFINITY;
        if horz_on {
            let v0 = self.luma_variance(px, py, 32, 16);
            let v1 = self.luma_variance(px, py + 16, 32, 16);
            if unpruned_rect32 || (block_var > 1.0 && 0.5 * (v0 + v1) < 0.85 * block_var) {
                rd_h = self.rd_cost_rect32(px, py, false, prdo)
                    + self.rd_cost_chroma_partition(px, py, 32, Part16::Horz, prdo);
            }
        }
        let mut rd_v = f32::INFINITY;
        if vert_on {
            let v0 = self.luma_variance(px, py, 16, 32);
            let v1 = self.luma_variance(px + 16, py, 16, 32);
            if unpruned_rect32 || (block_var > 1.0 && 0.5 * (v0 + v1) < 0.85 * block_var) {
                rd_v = self.rd_cost_rect32(px, py, true, prdo)
                    + self.rd_cost_chroma_partition(px, py, 32, Part16::Vert, prdo);
            }
        }
        let cands = [
            (rd_none, Part16::None),
            (rd_split, Part16::Split),
            (rd_h, Part16::Horz),
            (rd_v, Part16::Vert),
        ];
        cands
            .into_iter()
            .fold((f32::INFINITY, Part16::Split), |b, c| {
                if c.0 < b.0 { c } else { b }
            })
            .1
    }

    fn rd_cost_none32(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let dc = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
        let mut resid = [0i32; 1024];
        crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, px, py, 32, 32, dc);
        let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
        trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
        let rr = idct_dequant_32x32(&cf, &self.quant);
        let distortion =
            self.luma_partition_distortion(px, py, 32, 32, self.quant.ac_q() as f32, |i| {
                dc + rr[i]
            });
        crate::partition_rd::rd_cost(
            distortion,
            mlam,
            self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0),
        )
    }

    fn rd_cost_split32(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // `SPLIT_SIGNAL_BITS` already accounts for the parent partition symbol,
        // the four child symbols AND four sets of per-block headers (see its
        // definition), so it is charged ONCE — as every other partition site
        // does (block32.rs rect leg, lossy_state 16x16, block64). The previous
        // `* 4.0` double-counted it at 96 bits and biased 32x32 toward merging.
        let mut total = rate_cost(mlam, SPLIT_SIGNAL_BITS * split32_signal_mult());
        for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
            let dc = dc_pred_16x16(&self.recon[0], self.w, px + sx, py + sy, self.bd as i32);
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, px + sx, py + sy, 16, 16, dc);
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_16X16, lam);
            let rr = idct_dequant_16x16(&cf, &self.quant);
            let distortion = self.luma_partition_distortion(
                px + sx,
                py + sy,
                16,
                16,
                self.quant.ac_q() as f32,
                |i| dc + rr[i],
            );
            total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&cf, &SCAN_16X16, 16, px + sx, py + sy, DC_PRED, 1),
            );
        }
        total
    }

    /// Code a 32x32 region (4:4:4 only) as a single TX_32X32 block: DC-pred luma
    /// and both chroma planes, forward DCT32 + quant + trellis, the TX_32X32
    /// coefficient coder, and reconstruction via the exact integer inverse.
    /// Updates the 8-unit (32-sample) skip / mode / coef neighbor footprint.
    /// (DC-only for now; SMOOTH/PAETH/directional and CfL at 32x32 come next.)
    /// Trial-code a 32x32 luma block as four TX_16X16 (`tx_depth = 1`) with
    /// per-sub-transform intra prediction, as the decoder reconstructs it: each
    /// quadrant predicts from the RUNNING reconstruction. The candidate recon is
    /// written into `self.recon` so later quadrants see it, then restored — the
    /// caller commits only if the split is selected. Mirrors `split16_luma_try`.
    #[allow(clippy::too_many_arguments)]
    fn split32_luma_try(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
    ) -> ([i32; 1024], i64) {
        let mut saved = [0i32; 1024];
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..32]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let maxv = (1i32 << self.bd) - 1;
        let mut cf4 = [0i32; 1024];
        let mut sse_sum = 0i64;
        for (qi, &(sx, sy)) in Self::Q32.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (16, 0) => (have_tr, false),
                (0, 16) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = [0i32; 256];
            if mode == DC_PRED && delta == 0 {
                let d = dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 256];
            } else {
                intra_predict_nd_ad(
                    mode, delta, &self.recon[0], self.w, bx, by, 16, 16, tr, bl, self.w,
                    self.h, block_ftype, &mut pred, self.bd,
                );
            }
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 16, 16);
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut cf, &tf, dcq, acq, &SCAN_16X16, lam, 16, self.dcdf(), 2, 0,
                &self.dcdf().eob_bin_256_l, 0,
            );
            let rr = idct_dequant_16x16(&cf, &self.quant);
            sse_sum += sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, bx, by, self.bd);
            for ry in 0..16 {
                let rrow = &mut self.recon[0][(by + ry) * self.w + bx..];
                for rx in 0..16 {
                    rrow[rx] = (pred[ry * 16 + rx] + rr[ry * 16 + rx]).clamp(0, maxv);
                }
            }
            cf4[qi * 256..qi * 256 + 256].copy_from_slice(&cf);
        }
        for ry in 0..32 {
            self.recon[0][(py + ry) * self.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        (cf4, sse_sum)
    }

    fn code_block32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 8);
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let prdo = self.perceptual_rd_scale(px, py, 32);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // luma intra mode search (non-directional + directional; the TX_32X32
        // residual transform is always DCT_DCT, so the mode affects prediction
        // only). Mirrors the 16x16 search.
        let mut best_mode = DC_PRED;
        let mut lpred = [0i32; 1024];
        let mut lcf = [0i32; 1024];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_filter_intra = None;
        let mut ltf = [0f32; 1024]; // winner transform coeffs (f32, for winner-only RDOQ)
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
            self.rank_luma_directionals::<1024>(modes, px, py, 32, 32, have_tr, have_bl)
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
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 32, 32);
            let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    self.dcdf(),
                    3,
                    0,
                    &self.dcdf().eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                );
            }
            let rr = idct_dequant_32x32(&cf, &self.quant);
            let sse = sse_recon::<1024, 32>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
            let filter_bits = if m == DC_PRED {
                cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(32, 32)], 0)
            } else {
                0.0
            };
            let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, m, 0)
                + self.mode_bits(px, py, m)
                + filter_bits;
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_filter_intra = None;
            }
        }
        if rl.is_none() && self.speed == Speed::Slow {
            let bsize = av1_block_size_index(32, 32);
            for filter_mode in FILTER_INTRA_MODES {
                let mut pred = [0i32; 1024];
                filter_intra_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    &mut pred,
                    self.bd,
                );
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
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    self.dcdf(),
                    3,
                    0,
                    &self.dcdf().eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                );
                let rr = idct_dequant_32x32(&cf, &self.quant);
                let sse = sse_recon::<1024, 32>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if rl.is_some() || (filter_intra_sse_allowed(sse, best_dct_sse) && cost < best_eff)
                {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    lpred = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
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
            let ds = self.dc_sign_ctx_32(0, px / 4, py / 4);
            let wrr = idct_dequant_32x32(&lcf, &self.quant);
            let wsse = sse_recon::<1024, 32>(&lpred, &wrr, &self.src[0], self.w, px, py, self.bd);
            let wbits = self.luma_bits(&lcf, &SCAN_32X32, 32, px, py, best_mode, 0);
            let mut best_ad_cost = rd_cost_i64(wsse, mlam, wbits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 1024];
                intra_predict_nd_ad(
                    best_mode,
                    d,
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
                let mut resid = [0i32; 1024];
                crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 32, 32);
                let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_32X32,
                        lam,
                        32,
                        self.dcdf(),
                        3,
                        0,
                        &self.dcdf().eob_bin_1024_l,
                        ds,
                    );
                }
                let rr = idct_dequant_32x32(&cf, &self.quant);
                let sse = sse_recon::<1024, 32>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, best_mode, 0);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if rl.is_some() || cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred = pred;
                    lcf = cf;
                    ltf = tf;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if rl.is_none() && !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                self.dcdf(),
                3,
                0,
                &self.dcdf().eob_bin_1024_l,
                self.dc_sign_ctx_32(0, px / 4, py / 4),
            );
        }
        // TX split (tx_depth = 1): four TX_16X16 instead of one TX_32X32.
        // NOT a plain-RD choice — a straight SSE R-D trial picks this on detailed
        // blocks where the four transforms' extra txb_skip/EOB/txtp symbols cost
        // more than the better compaction saves (measured +2.2% BD-rate). The
        // split's real benefit is BANDING: on a smooth ramp the 32x32's low-freq
        // AC dies at the forward quantizer and the block reconstructs flat, while
        // four sub-transform DCs carry the ramp. SSE cannot see that, so gate on
        // the same `banding_risk` trigger block16 uses and accept whenever the
        // distortion is not meaningfully worse.
        let mut tx_split = false;
        if rl.is_none() && best_filter_intra.is_none() && self.banding_risk(px, py, 32) {
            let none_sse = sse_recon::<1024, 32>(
                &lpred, &idct_dequant_32x32(&lcf, &self.quant),
                &self.src[0], self.w, px, py, self.bd,
            );
            let (cf4, sse_s) =
                self.split32_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam);
            // Acceptance: the split must genuinely improve SSE. block16 uses a
            // permissive +25% tolerance at 16x16, but four TX_16X16 cost far more
            // syntax than four TX_8X8 do, so copying that tolerance here lets in
            // bad splits — measured +6.7% BD-rate on detailed content. Requiring
            // a real improvement keeps the banding win and removes the loss.
            if (sse_s as i128) * 1024
                <= (none_sse as i128) * (1024 - SPLIT32_SSE_MARGIN as i128)
            {
                tx_split = true;
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
            tx_split = r.tx == TxSel::SplitDct;
        }
        if let Some(cf) = rl_cf {
            lcf.copy_from_slice(&cf);
        }
        if tx_split {
            // Deblock works on TRANSFORM edges: re-record as four TX_16X16 tiles.
            self.record_tx_blk(x8, y8, 4);
            self.record_tx_blk(x8 + 2, y8, 4);
            self.record_tx_blk(x8, y8 + 2, 4);
            self.record_tx_blk(x8 + 2, y8 + 2, 4);
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            palette: 0,
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            // `SplitDct` marks the tx_depth=1 grid of four TX_16X16
            // (coefficients packed quadrant-major); otherwise DCT_DCT.
            tx: if tx_split {
                TxSel::SplitDct
            } else {
                TxSel::from_flags(false, false, false, false)
            },
        });
        self.push_luma_cf(&lcf);
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block32_420(
                x8,
                y8,
                &lcf,
                &lpred,
                best_mode,
                luma_zero,
                best_delta,
                best_filter_intra,
                tx_split,
                have_tr,
                have_bl,
            );
        } else if self.ss422 {
            self.code_block32_422(
                x8,
                y8,
                &lcf,
                &lpred,
                best_mode,
                luma_zero,
                best_delta,
                best_filter_intra,
                tx_split,
                have_tr,
                have_bl,
            );
        } else {
            self.code_block32_444(
                x8,
                y8,
                &lcf,
                &lpred,
                best_mode,
                luma_zero,
                best_delta,
                best_filter_intra,
                tx_split,
                have_tr,
                have_bl,
            );
        }
    }

    /// Shared header + luma for a TX_32X32 block: block skip flag, y/uv modes
    /// (uv via `emit_uv_mode`, plain DC or CfL), `angle_delta` for directional
    /// luma modes, the TX_32X32 luma coefficients (no tx-type symbol), the
    /// 8-unit (32-sample) skip/mode/coef footprint, and luma reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn code_header_luma32(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
        tx_split: bool,
        have_tr: bool,
        have_bl: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 4, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (angle_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        self.emit_uv_mode(y_mode, uv_mode, cfl, px, py, 32, 32);
        self.emit_palette_mode_info(px, py, 32, 32, y_mode, !self.mono, None);
        self.emit_filter_intra(y_mode, 32, 32, filter_intra);
        self.code_tx_depth(px, py, 32, 32, tx_split as usize);
        // Derived ONCE at the block origin (dav1d), before a_mode/l_mode are filled.
        let block_ftype = self.luma_filter_type(px, py);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 8].fill(sv);
        self.l_skip[by4..by4 + 8].fill(sv);
        self.a_mode[bx4..bx4 + 8].fill(mv);
        self.l_mode[by4..by4 + 8].fill(mv);
        if tx_split {
            let maxv = (1i32 << self.bd) - 1;
            for (qi, &(sx, sy)) in Self::Q32.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let mut cfq = [0i32; 256];
                cfq.copy_from_slice(&lcf[qi * 256..qi * 256 + 256]);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
                    let ds = self.dc_sign_ctx_16(0, qbx4, qby4);
                    encode_tx16_coeffs_adapt(
                        &mut self.enc, &mut self.cdfs, &cfq, false, sk, ds,
                        filter_intra_tx_mode(None, y_mode), 1, // DCT_DCT
                    )
                };
                self.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
                self.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
                if self.sb_mode == SbMode::Replay {
                    continue;
                }
                let (tr, bl) = match (sx, sy) {
                    (0, 0) => (py > 0, px > 0),
                    (16, 0) => (have_tr, false),
                    (0, 16) => (true, have_bl),
                    _ => (false, false),
                };
                let mut pred = [0i32; 256];
                if y_mode == DC_PRED && angle_delta == 0 {
                    let d = dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                    pred = [d; 256];
                } else {
                    intra_predict_nd_ad(
                        y_mode, angle_delta, &self.recon[0], self.w, bx, by, 16, 16, tr, bl,
                        self.w, self.h, block_ftype, &mut pred, self.bd,
                    );
                }
                let rr = if block_skip {
                    [0i32; 256]
                } else {
                    idct_dequant_16x16(&cfq, &self.quant)
                };
                for ry in 0..16 {
                    let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                    recon_add_pred(&mut drow[..16], &pred[ry * 16..], &rr[ry * 16..], maxv);
                }
            }
            return;
        }
        let lres = if block_skip {
            0x40
        } else {
            let sk = self.skip_ctx_32(0, bx4, by4, false);
            let ds = self.dc_sign_ctx_32(0, bx4, by4);
            encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, lcf, false, sk, ds)
        };
        self.a_coef[0][bx4..bx4 + 8].fill(lres);
        self.l_coef[0][by4..by4 + 8].fill(lres);
        // Pure-emit replay: recon is preinstalled from the record; the write
        // below would need the prediction the caller no longer computes.
        if self.sb_mode == SbMode::Replay {
            return;
        }
        let lrr = if block_skip {
            [0i32; 1024]
        } else {
            idct_dequant_32x32(lcf, &self.quant)
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<32>()
            .0
            .iter()
            .zip(lrr.as_chunks::<32>().0.iter())
            .enumerate()
        {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_pred(drow, prow, rrow, (1 << self.bd) - 1);
        }
    }

    /// 8x8 rect: HORZ = two 8x4, VERT = two 4x8. Shared 4x4 chroma in 4:2:0
    /// (coded on 2nd sub-block); per-sub chroma in 4:4:4/4:2:2. V forbidden in 4:2:2.
    fn code_block8_rect(&mut self, x8: usize, y8: usize, vert: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (lw, lh) = if vert { (4usize, 8usize) } else { (8, 4) };
        for half in 0..2 {
            let (px, py) = if vert {
                (x8 * 8 + half * 4, y8 * 8)
            } else {
                (x8 * 8, y8 * 8 + half * 4)
            };
            let (bx4, by4) = (px / 4, py / 4);
            let lpred = if vert {
                dc_pred_4x8(&self.recon[0], self.w, px, py, self.bd as i32)
            } else {
                dc_pred_8x4(&self.recon[0], self.w, px, py, self.bd as i32)
            };
            let mut lresid = [0i32; 32];
            crate::rd_sse::residual_dc(&mut lresid, &self.src[0], self.w, px, py, lw, lh, lpred);
            let (mut lcf, ltf) = if vert {
                dct4x8_t(&lresid, &self.quant)
            } else {
                dct8x4_t(&lresid, &self.quant)
            };
            let lscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
            let mean_l = lresid[..lw * lh].iter().sum::<i32>() / (lw * lh) as i32;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            let luma_zero = lcf.iter().all(|&v| v == 0);
            // chroma present on this sub-block?
            let has_chroma = if self.ss420 {
                if vert { px % 8 != 0 } else { py % 8 != 0 } // 2nd sub-block only
            } else {
                true
            };
            let (cx, cy, cw, ch) = if self.ss420 {
                (x8 * 4, y8 * 4, 4usize, 4usize) // 4x4 over the 8x8 luma area
            } else if self.ss422 {
                (px / 2, py, lw / 2, lh)
            } else {
                (px, py, lw, lh)
            };
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let cn = cw * ch;
            let mut ccf = [[0i32; 64]; 2];
            let mut cpred = [0i32; 2];
            if has_chroma {
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = chroma_dc_rect8(
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        cw,
                        ch,
                        self.bd as i32,
                    );
                    cpred[ci] = dc;
                    let mut resid = [0i32; 64];
                    crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, cw, ch, dc);
                    let (mut q, qt) = fwd_chroma_rect8(cw, ch, &resid, &self.cquant);
                    let cscan = scan_rect8(cw, ch);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                    let mean_c = resid[..cn].iter().sum::<i32>() / cn as i32;
                    if q[0] == 0 && mean_c.abs() >= 8 {
                        q[0] = if mean_c > 0 { 1 } else { -1 };
                    }
                    ccf[ci] = q;
                }
            }
            let chroma_zero =
                !has_chroma || (ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0));
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(px / 8, py / 8, (lw / 4).max(1) as u8, (lh / 4).max(1) as u8);
            self.mark_skip8_rect(px / 8, py / 8, 1, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            if has_chroma {
                self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, lw, lh);
            }
            self.emit_palette_mode_info(px, py, lw, lh, DC_PRED, has_chroma, None);
            self.emit_filter_intra(DC_PRED, lw, lh, None);
            self.code_tx_depth(px, py, lw, lh, 0);
            let sv = block_skip as u8;
            let (aw, ah) = ((lw / 4).max(1), (lh / 4).max(1));
            self.a_skip[bx4..bx4 + aw].fill(sv);
            self.l_skip[by4..by4 + ah].fill(sv);
            self.a_mode[bx4..bx4 + aw].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + ah].fill(DC_PRED as u8);
            let lres_ctx = if block_skip {
                0x40
            } else if vert {
                let ds = self.dc_sign_ctx_4x8_luma(bx4, by4);
                encode_4x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, DC_PRED, 1)
            } else {
                let ds = self.dc_sign_ctx_8x4_luma(bx4, by4);
                encode_8x4_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
            self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 32]
            } else if vert {
                idct_dequant_4x8(&lcf, &self.quant)
            } else {
                idct_dequant_8x4(&lcf, &self.quant)
            };
            for ry in 0..lh {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..lw], lpred, &lrr[ry * lw..], maxval);
            }
            if has_chroma {
                let (caw, cah) = ((cw / 4).max(1), (ch / 4).max(1));
                for ci in 0..2 {
                    let plane = ci + 1;
                    let cres_ctx = if block_skip {
                        0x40
                    } else {
                        self.emit_chroma_rect8(plane, cbx4, cby4, cw, ch, &ccf[ci])
                    };
                    self.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                    self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                    let rr = if block_skip {
                        [0i32; 64]
                    } else {
                        inv_chroma_rect8(cw, ch, &ccf[ci], &self.cquant)
                    };
                    for ry in 0..ch {
                        let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                        recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                    }
                }
            }
        }
    }

    fn emit_chroma_rect8(
        &mut self,
        plane: usize,
        cbx4: usize,
        cby4: usize,
        cw: usize,
        ch: usize,
        cf: &[i32; 64],
    ) -> u8 {
        match (cw, ch) {
            (8, 4) => {
                let sk = self.skip_ctx_8x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x4_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 32];
                a.copy_from_slice(&cf[..32]);
                encode_8x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            (4, 8) => {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 32];
                a.copy_from_slice(&cf[..32]);
                encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            _ => {
                let sk = self.skip_ctx_4x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x4_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 16];
                a.copy_from_slice(&cf[..16]);
                encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
        }
    }

    /// 32x32 rect: HORZ = two 32x16, VERT = two 16x32. Chroma per format. DC intra.
    /// V forbidden in 4:2:2.
    fn code_block32_rect(&mut self, x8: usize, y8: usize, vert: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = if vert {
                (x8 * 8 + half * 16, y8 * 8)
            } else {
                (x8 * 8, y8 * 8 + half * 16)
            };
            let (bx4, by4) = (px / 4, py / 4);
            let (lw, lh) = if vert { (16usize, 32usize) } else { (32, 16) };
            let lpred = if vert {
                dc_pred_16x32(&self.recon[0], self.w, px, py, self.bd as i32)
            } else {
                dc_pred_32x16(&self.recon[0], self.w, px, py, self.bd as i32)
            };
            let mut lresid = [0i32; 512];
            crate::rd_sse::residual_dc(&mut lresid, &self.src[0], self.w, px, py, lw, lh, lpred);
            let (mut lcf, ltf) = if vert {
                dct16x32_t(&lresid, &self.quant)
            } else {
                dct32x16_t(&lresid, &self.quant)
            };
            let lscan: &[u32] = if vert { &SCAN_16X32 } else { &SCAN_32X16 };
            trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
            let mean_l = lresid[..lw * lh].iter().sum::<i32>() / (lw * lh) as i32;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            let luma_zero = lcf.iter().all(|&v| v == 0);
            // chroma dims per format
            let (cx, cy, cw, ch) = if self.ss420 {
                (px / 2, py / 2, lw / 2, lh / 2)
            } else if self.ss422 {
                (px / 2, py, lw / 2, lh)
            } else {
                (px, py, lw, lh)
            };
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let cn = cw * ch;
            let mut ccf = [[0i32; 512]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc =
                    chroma_dc_rect(&self.recon[plane], self.cw, cx, cy, cw, ch, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 512];
                crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, cw, ch, dc);
                let (mut q, qt) = fwd_chroma_rect(cw, ch, &resid, &self.cquant);
                let cscan = scan_rect(cw, ch);
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                let mean_c = resid[..cn].iter().sum::<i32>() / cn as i32;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(px / 8, py / 8, (lw / 4) as u8, (lh / 4) as u8);
            self.mark_skip8_rect(px / 8, py / 8, lw / 8, lh / 8, block_skip);
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
                let ds = self.dc_sign_ctx_16x32_luma(bx4, by4);
                encode_16x32_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds)
            } else {
                let ds = self.dc_sign_ctx_32x16_luma(bx4, by4);
                encode_32x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds)
            };
            self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
            self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 512]
            } else if vert {
                idct_dequant_16x32(&lcf, &self.quant)
            } else {
                idct_dequant_32x16(&lcf, &self.quant)
            };
            for ry in 0..lh {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..lw], lpred, &lrr[ry * lw..], maxval);
            }
            let (caw, cah) = (cw / 4, ch / 4);
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    self.emit_chroma_rect(plane, cbx4, cby4, cw, ch, &ccf[ci])
                };
                self.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 512]
                } else {
                    inv_chroma_rect(cw, ch, &ccf[ci], &self.cquant)
                };
                for ry in 0..ch {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                }
            }
        }
    }

    fn emit_chroma_rect(
        &mut self,
        plane: usize,
        cbx4: usize,
        cby4: usize,
        cw: usize,
        ch: usize,
        cf: &[i32; 512],
    ) -> u8 {
        match (cw, ch) {
            (32, 16) => {
                let sk = self.skip_ctx_32x16_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_32x16_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 512];
                a.copy_from_slice(cf);
                encode_32x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            (16, 32) => {
                let sk = self.skip_ctx_16x32_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_16x32_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 512];
                a.copy_from_slice(cf);
                encode_16x32_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            (16, 8) => {
                let sk = self.skip_ctx_16x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_16x8_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 128];
                a.copy_from_slice(&cf[..128]);
                encode_16x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            (8, 16) => {
                let sk = self.skip_ctx_8x16_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x16_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 128];
                a.copy_from_slice(&cf[..128]);
                encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            _ => {
                let sk = self.skip_ctx_16x16_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_16x16_chroma(plane, cbx4, cby4);
                let mut a = [0i32; 256];
                a.copy_from_slice(&cf[..256]);
                encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &a,
                    true,
                    sk,
                    ds,
                    DC_PRED,
                    0,
                )
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block32_444(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
        tx_split: bool,
        have_tr: bool,
        have_bl: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f32,
            self.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        // plain-DC chroma (skipped in pure-emit replay: the captured winner
        // coeffs install below and the recon is preinstalled).
        let mut ccf = [[0i32; 1024]; 2];
        let mut cdc = [0i32; 2];
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let dc = dc_pred_32x32(&self.recon[plane], self.w, px, py, self.bd as i32);
            cdc[ci] = dc;
            let mut cresid = [0i32; 1024];
            crate::rd_sse::residual_dc(&mut cresid, &self.src[plane], self.w, px, py, 32, 32, dc);
            let (q, qt) = forward_dct_quant_32x32_t(&cresid, &self.cquant);
            ccf[ci] = q;
            trellis_optimize(&mut ccf[ci], &qt, dcq, acq, &SCAN_32X32, lam);
            let mean_resid_dc = cresid.iter().sum::<i32>() / 1024;
            if ccf[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // CfL: predict chroma from the reconstructed luma AC.
        let mut cfl_ccf = [[0i32; 1024]; 2];
        let mut cfl_pred = [[0i32; 1024]; 2];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
        let mlam = self.mlam();
        // Pure-emit replay never evaluates CfL; the use_cfl decision below
        // replays from the record and the winner state installs after this.
        if ru.is_none() {
            let lrr_cfl = idct_dequant_32x32(lcf, &self.quant);
            let mut luma_rec = [0i32; 1024];
            recon_add_pred(&mut luma_rec, lpred, &lrr_cfl, (1 << self.bd) - 1);
            let mut ac = [0i32; 1024];
            cfl_ac_444(&luma_rec, 32, 32, &mut ac);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cdc[ci];
                let mut src = [0i32; 1024];
                for (ry, drow) in src.as_chunks_mut::<32>().0.iter_mut().enumerate() {
                    drow.copy_from_slice(&self.src[plane][(py + ry) * self.w + px..][..32]);
                }
                let dcrr = idct_dequant_32x32(&ccf[ci], &self.cquant);
                let s = sse_recon::<1024, 32>(&[dc; 1024], &dcrr, &src, 32, 0, 0, self.bd);
                dc_cost[ci] =
                    rd_cost_i64(s, mlam, self.chroma_bits(&ccf[ci], &SCAN_32X32, 32, plane, px, py));
                let a = cfl_best_alpha(&ac, &src, dc, 1024, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 1024];
                for i in 0..1024 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                }
                let mut resid = [0i32; 1024];
                crate::rd_sse::residual_pred(&mut resid, &cpr, &src, 32, 0, 0, 32, 32);
                let (mut q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_32X32, lam);
                let rr = idct_dequant_32x32(&q, &self.cquant);
                let s2 = sse_recon::<1024, 32>(&cpr, &rr, &src, 32, 0, 0, self.bd);
                cfl_ccf[ci] = q;
                cfl_pred[ci] = cpr;
                cfl_cost[ci] =
                    rd_cost_i64(s2, mlam, self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py));
            }
        }
        // Pure-emit replay: install the captured winner state before the
        // cf_use/cfl_opt bindings below read it (CfL coeffs+alphas go to
        // cfl_ccf/cfl_a, DC/directional coeffs to ccf).
        if let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            if r.uv == CFL_PRED as u8 {
                cfl_a = *al;
                for (dst, src) in cfl_ccf.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            } else {
                for (dst, src) in ccf.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            }
        }
        // CfL signaling costs extra (sign + per-plane alpha); only use it when
        // it beats plain DC on both planes' summed cost by that overhead.
        let cfl_sig = 4.0f32
            + if cfl_a[0] != 0 { 4.0f32 } else { 0.0f32 }
            + if cfl_a[1] != 0 { 4.0f32 } else { 0.0f32 };
        // Let the RD comparison decide DC-vs-CfL across the whole quality range;
        // the old `acq > 300` gate suppressed CfL exactly where it helps most
        // (high quality). block8/block16 already dropped it — this path was the
        // last one still gated, which mattered most at 4:4:4 where chroma is
        // full resolution and 32x32 blocks are common.
        let use_cfl = ru.map_or(
            (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig) < dc_cost[0] + dc_cost[1],
            |r| r.uv == CFL_PRED as u8,
        );
        #[allow(unused_mut)] // cfl_opt mutated in 'sv block when SMOOTH_V wins
        let (cf_use, pred_dc, mut cfl_opt): (
            &[[i32; 1024]; 2],
            [i32; 2],
            Option<[i32; 2]>,
        ) = if use_cfl {
            (&cfl_ccf, cdc, Some(cfl_a))
        } else {
            (&ccf, cdc, None)
        };
        // Directional / smooth chroma on the 32x32 chroma block. Per the AV1 spec
        // (compute_tx_type), intra blocks whose square transform size is >= TX_32X32
        // always use DCT_DCT, so every non-DC uv_mode here codes its residual with
        // the plain 32x32 DCT — only the prediction differs. PAETH/SMOOTH/SMOOTH_V/
        // SMOOTH_H are searched against the current DC/CfL winner on the libaom-style
        // R-D cost (SSE + mlam*(coeff_bits + mode_signal_bits)).
        #[allow(unused_mut)] // assigned via break in 'sv labeled block
        let mut cf_use_owned: [[i32; 1024]; 2];
        let mut sv_preds32 = [[0i32; 1024]; 2];
        let (final_cf, chosen_uv_32) = 'sv: {
            // Pure-emit replay: the captured coefficients were installed above
            // (into cfl_ccf for CfL, ccf otherwise); no search runs at all.
            if let Some(r) = ru {
                if r.uv == DC_PRED as u8 || r.uv == CFL_PRED as u8 {
                    break 'sv (cf_use, DC_PRED);
                }
                cfl_opt = None;
                break 'sv (cf_use, r.uv as usize);
            }
            let dcq2 = self.cquant.dc_q() as f32;
            let acq2 = self.cquant.ac_q() as f32;
            let lam2 = trellis_lambda();
            let mlam = self.mlam_c();
            // R-D of the current winner (DC or CfL), residual already in `cf_use`.
            let mut cur_total = 0f32;
            if use_cfl {
                let a = cfl_a;
                cur_total += rate_cost(
                    mlam,
                    4.0f32
                        + if a[0] != 0 { 4.0f32 } else { 0.0f32 }
                        + if a[1] != 0 { 4.0f32 } else { 0.0f32 },
                );
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let rr = idct_dequant_32x32(&cf_use[ci], &self.cquant);
                let cur_pred = if use_cfl {
                    cfl_pred[ci]
                } else {
                    [pred_dc[ci]; 1024]
                };
                let sse = sse_recon::<1024, 32>(
                    &cur_pred,
                    &rr,
                    &self.src[plane],
                    self.w,
                    px,
                    py,
                    self.bd,
                );
                cur_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&cf_use[ci], &SCAN_32X32, 32, plane, px, py),
                );
            }

            let mut best_total = cur_total;
            let mut best_mode = DC_PRED;
            let mut best_ccf = [[0i32; 1024]; 2];
            let mut best_pred = [[0i32; 1024]; 2];
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
                self.rank_chroma_directionals::<1024>(candidates, px, py, px, py, 32, 32)
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
                let mut cand_ccf = [[0i32; 1024]; 2];
                let mut cand_pred = [[0i32; 1024]; 2];
                // V/H also emit a chroma angle_delta symbol (~3 bits); transform stays
                // DCT_DCT here (spec forces it at Tx_Size_Sqr >= TX_32X32).
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
                        32,
                        32,
                        false,
                        false,
                        self.w,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        self.bd,
                    );
                    let mut resid = [0i32; 1024];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &cand_pred[ci],
                        &self.src[plane],
                        self.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    // Forced DCT_DCT at 32x32 (spec), regardless of uv_mode.
                    let (mut q, qt) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_32X32, lam2);
                    let mean_resid = resid.iter().sum::<i32>() / 1024;
                    if q[0] == 0 && mean_resid.abs() >= 8 {
                        q[0] = if mean_resid > 0 { 1 } else { -1 };
                    }
                    cand_ccf[ci] = q;
                    let rr = idct_dequant_32x32(&q, &self.cquant);
                    let sse = sse_recon::<1024, 32>(
                        &cand_pred[ci],
                        &rr,
                        &self.src[plane],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    cand_total +=
                        rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py));
                }
                if ru.is_some() || cand_total < best_total {
                    best_total = cand_total;
                    best_mode = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            if best_mode != DC_PRED {
                cfl_opt = None; // a non-DC chroma mode overrides CfL if it wins
                cf_use_owned = best_ccf;
                sv_preds32 = best_pred;
                break 'sv (&cf_use_owned, best_mode);
            }
            (cf_use, DC_PRED)
        };
        // Capture the final chroma winner (CfL folded in as CFL_PRED).
        self.push_uv_sel(UvSel {
            uv: if chosen_uv_32 != DC_PRED {
                chosen_uv_32 as u8
            } else if cfl_opt.is_some() {
                CFL_PRED as u8
            } else {
                DC_PRED as u8
            },
        });
        self.push_uv_cf(&final_cf[0], &final_cf[1], cfl_opt.unwrap_or([0, 0]));
        let block_skip =
            luma_zero && final_cf[0].iter().all(|&c| c == 0) && final_cf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv_32,
            cfl_opt,
            angle_delta,
            filter_intra,
            tx_split,
            have_tr,
            have_bl,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_32(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_32(plane, bx4, by4);
                encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &final_cf[ci], true, sk, ds)
            };
            self.a_coef[plane][bx4..bx4 + 8].fill(cres);
            self.l_coef[plane][by4..by4 + 8].fill(cres);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let crr = if block_skip {
                [0i32; 1024]
            } else {
                idct_dequant_32x32(&final_cf[ci], &self.cquant)
            };
            let max = (1 << self.bd) - 1;
            for (ry, rrow) in crr.as_chunks::<32>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if chosen_uv_32 != DC_PRED {
                    recon_add_pred(&mut drow[..32], &sv_preds32[ci][ry * 32..], rrow, max);
                } else if use_cfl {
                    recon_add_pred(&mut drow[..32], &cfl_pred[ci][ry * 32..], rrow, max);
                } else {
                    recon_add_dc(&mut drow[..32], pred_dc[ci], rrow, max);
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn code_block32_420(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
        tx_split: bool,
        have_tr: bool,
        have_bl: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (bx4c, by4c) = (cx / 4, cy / 4);
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f32,
            self.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let maxval = (1 << self.bd) - 1;
        // DC path (skipped in pure-emit replay: block_skip below reads the
        // FINAL coeffs, installed from the record).
        let mut ccf_dc = [[0i32; 256]; 2];
        let mut dc_preds = [0i32; 2];
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let dc = dc_pred_16x16(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, 16, 16, dc);
            let (q, qt) = forward_dct_quant_16x16_t(&resid, &self.cquant);
            ccf_dc[ci] = q;
            trellis_optimize(&mut ccf_dc[ci], &qt, dcq, acq, &SCAN_16X16, lam);
            let mean_resid_dc = resid.iter().sum::<i32>() / 256;
            if ccf_dc[ci][0] == 0 && mean_resid_dc.abs() >= 8 {
                ccf_dc[ci][0] = if mean_resid_dc > 0 { 1 } else { -1 };
            }
        }
        // SMOOTH_V chroma derives ADST_DCT (a 2D tx -> default scan and coef
        // contexts identical to DCT_DCT; only the transform differs). Forward with
        // adstdct16x16_t and reconstruct with iadstdct_dequant_16x16 to match the
        // decoder's derived chroma txtp. Offered at every quality; the Lagrangian
        // R-D decision below selects it only when it truly wins.
        // DC baseline R-D (libaom-style: SSE + mlam*coeff_bits over U+V).
        let mlam = self.mlam();
        let mut rr_dc = [[0i32; 256]; 2];
        let mut dc_total = 0f32;
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            rr_dc[ci] = idct_dequant_16x16(&ccf_dc[ci], &self.cquant);
            let dc = dc_preds[ci];
            let sse = sse_recon::<256, 16>(
                &[dc; 256],
                &rr_dc[ci],
                &self.src[plane],
                self.cw,
                cx,
                cy,
                self.bd,
            );
            dc_total += rd_cost_i64(
                sse,
                mlam,
                self.chroma_bits(&ccf_dc[ci], &SCAN_16X16, 16, plane, cx, cy),
            );
        }
        // Directional / smooth chroma modes (PAETH/SMOOTH/SMOOTH_V/SMOOTH_H), each
        // with its decoder-derived chroma tx. Winner must beat DC on the R-D metric.
        let mut best_total = dc_total;
        let mut chosen_uv = DC_PRED;
        let mut best_ccf = ccf_dc;
        let mut best_rr = rr_dc;
        let mut sv_preds = [[0i32; 256]; 2];
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
            self.rank_chroma_directionals::<256>(candidates, px, py, cx, cy, 16, 16)
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
            let mut cand_ccf = [[0i32; 256]; 2];
            let mut cand_rr = [[0i32; 256]; 2];
            let mut cand_pred = [[0i32; 256]; 2];
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
                    16,
                    16,
                    false,
                    false,
                    self.cw,
                    self.h,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci],
                    self.bd,
                );
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &cand_pred[ci],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    16,
                    16,
                );
                let (mut q, qt) = fwd_chroma_16x16(tx, &resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X16, lam);
                let mean_resid = resid.iter().sum::<i32>() / 256;
                if q[0] == 0 && mean_resid.abs() >= 8 {
                    q[0] = if mean_resid > 0 { 1 } else { -1 };
                }
                cand_ccf[ci] = q;
                cand_rr[ci] = inv_chroma_16x16(tx, &q, &self.cquant);
                let sse = sse_recon::<256, 16>(
                    &cand_pred[ci],
                    &cand_rr[ci],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    self.bd,
                );
                cand_total +=
                    rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy));
            }
            if ru.is_some() || cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                best_ccf = cand_ccf;
                best_rr = cand_rr;
                sv_preds = cand_pred;
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
        let use_sv = chosen_uv != DC_PRED;
        let (ccf, rr_cache) = (best_ccf, best_rr);
        self.push_uv_sel(UvSel {
            uv: chosen_uv as u8,
        });
        self.push_uv_cf(&ccf[0], &ccf[1], [0, 0]);
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            None,
            angle_delta,
            filter_intra,
            tx_split,
            have_tr,
            have_bl,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx_16(plane, bx4c, by4c);
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
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let rr = if block_skip {
                [0i32; 256]
            } else {
                rr_cache[ci]
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                if use_sv {
                    recon_add_pred(&mut drow[..16], &sv_preds[ci][ry * 16..], rrow, maxval);
                } else {
                    recon_add_dc(&mut drow[..16], dc_preds[ci], rrow, maxval);
                }
            }
        }
    }

    /// 4:2:2: a 32x32 luma region maps to a 16-wide x 32-tall chroma block per
    /// plane (`RTX_16X32`, coef-CDF class 3). DC-pred chroma.
    #[allow(clippy::too_many_arguments)]
    fn code_block32_422(
        &mut self,
        x8: usize,
        y8: usize,
        lcf: &[i32; 1024],
        lpred: &[i32; 1024],
        y_mode: usize,
        luma_zero: bool,
        angle_delta: i32,
        filter_intra: Option<FilterIntraMode>,
        tx_split: bool,
        have_tr: bool,
        have_bl: bool,
    ) {
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        let (bx4c, by4c) = (cx / 4, py / 4);
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let maxv = (1 << self.bd) - 1;
        let (dcq, acq, lam) = (
            self.cquant.dc_q() as f32,
            self.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        let mlam = self.mlam();
        let mut ccf = [[0i32; 512]; 2];
        let mut cpred = [0i32; 2];
        let mut cpred_px = [[0i32; 512]; 2];
        let mut src_planes = [[0i32; 512]; 2];
        let mut dc_ccf = [[0i32; 512]; 2];
        let mut dc_sse = [0i64; 2];
        let mut dc_bits = [0f32; 2];
        // DC option (skipped in pure-emit replay: the captured winner installs
        // below and block_skip reads the FINAL coeffs, matching Off).
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let pred = dc_pred_16x32(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut src = [0i32; 512];
            for (ry, srow_dst) in src.as_chunks_mut::<16>().0.iter_mut().enumerate() {
                srow_dst.copy_from_slice(&self.src[plane][(py + ry) * self.cw + cx..][..16]);
            }
            src_planes[ci] = src;
            let mut resid = [0i32; 512];
            crate::rd_sse::residual_dc(&mut resid, &src, 16, 0, 0, 16, 32, pred);
            let (mut q, qt) = forward_dct_quant_16x32_t(&resid, &self.cquant);
            trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X32, lam);
            let rr = idct_dequant_16x32(&q, &self.cquant);
            dc_ccf[ci] = q;
            dc_sse[ci] =
                crate::rd_sse::sse_recon(&[pred; 512], &rr, &src, 16, 0, 0, 16, 32, self.bd);
            dc_bits[ci] = block_rate_bits(&q, &SCAN_16X32);
        }

        // CfL: predict the 16x32 U/V from the horizontally-subsampled 32x32
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0). 32x32 luma is
        // always DCT_DCT here, so the AC reference inverts with idct_dequant_32x32.
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // below.
        if ru.is_none() {
            let lrr_cfl = idct_dequant_32x32(lcf, &self.quant);
            let mut luma_rec = [0i32; 1024];
            recon_add_pred(&mut luma_rec, lpred, &lrr_cfl, maxv);
            let mut ac = [0i32; 512];
            cfl_ac_sub(&luma_rec, 32, 16, 32, true, false, &mut ac);
            let mut cfl_ccf = [[0i32; 512]; 2];
            let mut cfl_a = [0i32; 2];
            let mut cfl_sse = [0i64; 2];
            let mut cfl_bits = [0f32; 2];
            for ci in 0..2 {
                let dc = cpred[ci];
                let src = src_planes[ci];
                let a = cfl_best_alpha(&ac, &src, dc, 512, self.bd);
                cfl_a[ci] = a;
                let mut cpr = [0i32; 512];
                for i in 0..512 {
                    cpr[i] = cfl_pred_pixel(dc, ac[i], a, self.bd);
                }
                let mut resid = [0i32; 512];
                crate::rd_sse::residual_pred(&mut resid, &cpr, &src, 16, 0, 0, 16, 32);
                let (mut q, qt) = forward_dct_quant_16x32_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X32, lam);
                let rr = idct_dequant_16x32(&q, &self.cquant);
                cfl_ccf[ci] = q;
                cfl_sse[ci] = crate::rd_sse::sse_recon(&cpr, &rr, &src, 16, 0, 0, 16, 32, self.bd);
                cfl_bits[ci] = block_rate_bits(&q, &SCAN_16X32);
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
        // Directional / smooth chroma on the 16x32 block. The AV1 spec forces
        // DCT_DCT for any intra block whose square transform size is >= TX_32X32,
        // and 16x32 (Tx_Size_Sqr = TX_32X32) hits that rule -- so the residual is
        // always the plain 16x32 DCT and only the prediction changes (PAETH/SMOOTH/
        // SMOOTH_V/SMOOTH_H). Searched against the DC/CfL winner on the R-D metric.
        let mut chosen_uv = if use_cfl { CFL_PRED } else { DC_PRED };
        // Pure-emit replay never runs the directional search; the captured
        // winner installs below.
        if ru.is_none() {
            // R-D of the current winner (DC or CfL), from the committed ccf/cpred.
            let mut best_total = 0f32;
            if use_cfl {
                let a = cfl_alpha_uv;
                best_total += rate_cost(
                    mlam,
                    4.0f32
                        + if a[0] != 0 { 4.0f32 } else { 0.0f32 }
                        + if a[1] != 0 { 4.0f32 } else { 0.0f32 },
                );
            }
            for ci in 0..2 {
                let cur_ccf = if use_cfl { ccf[ci] } else { dc_ccf[ci] };
                let rr = idct_dequant_16x32(&cur_ccf, &self.cquant);
                let cur_pred = if use_cfl {
                    cpred_px[ci]
                } else {
                    [cpred[ci]; 512]
                };
                let sse = crate::rd_sse::sse_recon(
                    &cur_pred,
                    &rr,
                    &src_planes[ci],
                    16,
                    0,
                    0,
                    16,
                    32,
                    self.bd,
                );
                best_total += rd_cost_i64(sse, mlam, block_rate_bits(&cur_ccf, &SCAN_16X32));
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
            let directional_top = if ru.is_none() {
                self.rank_chroma_directionals::<512>(candidates, px, py, cx, py, 16, 32)
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
                let mut cand_ccf = [[0i32; 512]; 2];
                let mut cand_pred = [[0i32; 512]; 2];
                // V/H also emit a chroma angle_delta symbol (~3 bits); transform stays
                // DCT_DCT here (spec forces it at Tx_Size_Sqr >= TX_32X32).
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
                        py,
                        16,
                        32,
                        false,
                        false,
                        self.cw,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci],
                        self.bd,
                    );
                    let src = src_planes[ci];
                    let mut resid = [0i32; 512];
                    crate::rd_sse::residual_pred(&mut resid, &cand_pred[ci], &src, 16, 0, 0, 16, 32);
                    // Forced DCT_DCT at 16x32 (spec), regardless of uv_mode.
                    let (mut q, qt) = forward_dct_quant_16x32_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_16X32, lam);
                    let mean_resid = resid.iter().sum::<i32>() / 512;
                    if q[0] == 0 && mean_resid.abs() >= 8 {
                        q[0] = if mean_resid > 0 { 1 } else { -1 };
                    }
                    cand_ccf[ci] = q;
                    let rr = idct_dequant_16x32(&q, &self.cquant);
                    let sse = crate::rd_sse::sse_recon(
                        &cand_pred[ci],
                        &rr,
                        &src,
                        16,
                        0,
                        0,
                        16,
                        32,
                        self.bd,
                    );
                    cand_total += rd_cost_i64(sse, mlam, block_rate_bits(&q, &SCAN_16X32));
                }
                if ru.is_some() || cand_total < best_total {
                    best_total = cand_total;
                    chosen_uv = cand;
                    use_cfl = false;
                    ccf[..2].copy_from_slice(&cand_ccf[..2]);
                    cpred_px[..2].copy_from_slice(&cand_pred[..2]);
                }
            }
        }
        if ru.is_none() && chosen_uv == DC_PRED {
            for ci in 0..2 {
                ccf[ci] = dc_ccf[ci];
                cpred_px[ci] = [cpred[ci]; 512];
            }
        }
        // Pure-emit replay: install the captured chroma winner (mode, coeffs,
        // CfL alphas; recon is preinstalled from the record).
        if let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            chosen_uv = r.uv as usize;
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
            for (dst, src) in ccf.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
        }
        self.push_uv_sel(UvSel {
            uv: chosen_uv as u8,
        });
        self.push_uv_cf(
            &ccf[0],
            &ccf[1],
            if use_cfl { cfl_alpha_uv } else { [0, 0] },
        );
        let block_skip =
            luma_zero && ccf[0].iter().all(|&c| c == 0) && ccf[1].iter().all(|&c| c == 0);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            if use_cfl { Some(cfl_alpha_uv) } else { None },
            angle_delta,
            filter_intra,
            tx_split,
            have_tr,
            have_bl,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x32_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_16x32_422(plane, bx4c, by4c);
                encode_16x32_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            self.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.l_coef[plane][by4c..by4c + 8].fill(res_ctx);
            if self.sb_mode == SbMode::Replay {
                continue; // recon preinstalled
            }
            let rr = if block_skip {
                [0i32; 512]
            } else {
                idct_dequant_16x32(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                recon_add_pred(drow, &cpred_px[ci][ry * 16..], rrow, maxv);
            }
        }
    }
}
