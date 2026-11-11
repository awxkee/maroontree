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

const fn bottomup_split32() -> bool {
    true
}

const fn full_partition_proxy32() -> bool {
    true
}

struct Luma32BeamCandidate {
    luma_cost: f32,
    mode: usize,
    delta: i32,
    pred: SBuf<[i32; 1024]>,
    cf: SBuf<[i32; 1024]>,
    tf: SBuf<[f32; 1024]>,
    sse: i64,
    palette: Option<LossyLumaPalette>,
}

/// TEMPORARY diagnostic knob for the 32x32 split-signal multiplier (1.0 = the
/// corrected pricing, 4.0 = the historical double-count).
fn split32_signal_mult() -> f32 {
    crate::tuning::get().split32_signal_mult
}

impl<'a> LossyTile<'a> {
    /// Raster quadrants of a 32x32 block as (dx, dy) pixel offsets (TX_16X16).
    const Q32: [(usize, usize); 4] = [(0, 0), (16, 0), (0, 16), (16, 16)];

    /// 4:2:2 chroma txb_skip (all_zero) context for an RTX_4X8 block (1 unit
    /// wide, 2 units tall; `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let ca = (a[bx4c] != 0x40) as usize;
        let cl = (l[by4c] != 0x40 || l[by4c + 1] != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_4X8: 1 unit wide, 2 tall, baseline -3.
    fn dc_sign_ctx_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 + (l[by4c + 1] >> 6) as i32 - 3;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:2 chroma txb_skip context for an RTX_8X16 block (2 units wide, 4 tall;
    /// chroma tx == chroma block so ctx_offset = 7): `7 + a_nz + l_nz`, where each
    /// term ORs over the units the block spans.
    fn skip_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let ca = (a[bx4c] != 0x40 || a[bx4c + 1] != 0x40) as usize;
        let cl =
            (l[by4c] != 0x40 || l[by4c + 1] != 0x40 || l[by4c + 2] != 0x40 || l[by4c + 3] != 0x40)
                as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_8X16: 2 units wide, 4 tall, baseline -6.
    fn dc_sign_ctx_8x16_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
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
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let ca = a[bx4c..bx4c + 4].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4c..by4c + 8].iter().any(|&x| x != 0x40) as usize;
        7 + ca + cl
    }

    /// 4:2:2 chroma dc_sign context for RTX_16X32: 4 units wide, 8 tall, baseline -12.
    fn dc_sign_ctx_16x32_422(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let suma: i32 = sum_coef_sign(&a[bx4c..bx4c + 4]);
        let suml: i32 = sum_coef_sign(&l[by4c..by4c + 8]);
        let s = suma + suml - 12;
        (s != 0) as usize + (s > 0) as usize
    }

    /// 4:2:0 chroma txb_skip context for a TX_4X4 block (1 unit wide and tall;
    /// `not_one_blk`=0): `7 + a_nz + l_nz`.
    fn skip_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        7 + (a[bx4c] != 0x40) as usize + (l[by4c] != 0x40) as usize
    }

    /// 4:2:0 chroma dc_sign context for TX_4X4: 1 unit each side, baseline -2.
    fn dc_sign_ctx_420(&self, plane: usize, bx4c: usize, by4c: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let s = (a[bx4c] >> 6) as i32 + (l[by4c] >> 6) as i32 - 2;
        (s != 0) as usize + (s > 0) as usize
    }

    /// dc_sign context for a TX_32X32 (8-unit footprint, baseline -16).
    fn dc_sign_ctx_32(&self, plane: usize, bx4: usize, by4: usize) -> usize {
        let a = &self.ctx.a_coef[plane];
        let l = &self.ctx.l_coef[plane];
        let suma: i32 = sum_coef_sign(&a[bx4..bx4 + 8]);
        let suml: i32 = sum_coef_sign(&l[by4..by4 + 8]);
        let s = suma + suml - 16;
        (s != 0) as usize + (s > 0) as usize
    }

    /// txb_skip context for a TX_32X32 (8-unit footprint). Luma (max tx in a
    /// 32x32 block) is always ctx 0; chroma uses `7 + above_nz + left_nz`.
    fn skip_ctx_32(&self, plane: usize, bx4: usize, by4: usize, chroma: bool) -> usize {
        if !chroma {
            0
        } else {
            let a = &self.ctx.a_coef[plane];
            let l = &self.ctx.l_coef[plane];
            let ca = a[bx4..bx4 + 8].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 8].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn choose_rect8(&self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) -> Part16 {
        if !crate::tuning::get().rect8_enabled || !self.cfg.speed.at_least_slow() || !self.cfg.is_420() {
            return Part16::None;
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let prdo = self.perceptual_rd_scale(px, py, 8);
        let mlam = self.mlam() * prdo;
        // Exact frozen-CDF partition symbols for all four candidates. The old
        // code charged NONE nothing and HORZ/VERT the flat ~24-bit
        // `partition_signal_bits()` — its own comment said that overprices the
        // 8x8 rect signal ~12x, but it kept using it. That is the same defect
        // whose fix at the 16 level is what finally made rect16 win.
        let none = self.rd_cost_square(px, py, 8, false, false, prdo)
            + rate_cost(mlam, self.part_bl8_rate(x8, y8, 0));
        let sig_h = rate_cost(mlam, self.part_bl8_rate(x8, y8, 1));
        let sig_v = rate_cost(mlam, self.part_bl8_rate(x8, y8, 2));
        let rd_h = sig_h
            + self.rd_cost_rect8_leaf(px, py, false, have_tr, px > 0, prdo)
            + self.rd_cost_rect8_leaf(px, py + 4, false, false, have_bl, prdo);
        // VERT forbidden at 4:2:2 (same rule as every level: a 4-wide luma
        // block yields 2-wide chroma, which needs the pairing machinery the
        // 422 branch of the emitter does not have).
        let rd_v = if self.cfg.is_422() {
            f32::INFINITY
        } else {
            sig_v
                + self.rd_cost_rect8_leaf(px, py, true, py > 0, have_bl, prdo)
                + self.rd_cost_rect8_leaf(px + 4, py, true, have_tr, false, prdo)
        };
        // Rectangle-vs-square balance at the 8x8 node, the analogue of
        // `rect16_bias`. Without it the optimizer can only turn rect8 on or
        // off, never price it — and pricing is what made rect16 work.
        let rb = crate::tuning::get().rect8_bias;
        let (rd_h, rd_v) = (rd_h * rb, rd_v * rb);
        if rd_h.min(rd_v) < none {
            if rd_h <= rd_v {
                Part16::Horz
            } else {
                Part16::Vert
            }
        } else {
            Part16::None
        }
    }

    /// Equipped price of one 8x4/4x8 luma leaf: the SAME 13-mode search the
    /// emitter runs (2026-07-24 — the DC/DCT-only leaves measured dead
    /// neutral, the rect16 under-tooled-leaf lesson; aom's hump-band rect8
    /// usage is directional thin blocks along edges).
    fn rd_cost_rect8_leaf(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        sub_tr: bool,
        sub_bl: bool,
        prdo: f32,
    ) -> f32 {
        let (acq, dcq) = (self.ctx.quant.ac_q() as f32, self.ctx.quant.dc_q() as f32);
        let (lam, mlam) = (trellis_lambda() * prdo, self.mlam() * prdo);
        let (lw, lh) = if vert { (4usize, 8) } else { (8, 4) };
        let scan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let _yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let mut best = f32::INFINITY;
        let mut pred = [0i32; 32];
        for m in 0..13usize {
            if m == DC_PRED {
                let dc = if vert {
                    self.k.intrapred.dc_pred_4x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
                } else {
                    self.k.intrapred.dc_pred_8x4(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
                };
                pred[..lw * lh].fill(dc);
            } else {
                self.k.intrapred.predict_nd(
                    m,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    lw,
                    lh,
                    sub_tr,
                    sub_bl,
                    self.cfg.w,
                    self.cfg.h,
                    ftype,
                    &mut pred,
                    self.cfg.bd,
                );
            }
            let mut resid = [0i32; 32];
            self.k.rd.residual_pred(&mut resid, &pred, &self.cfg.src[0], self.cfg.w, px, py, lw, lh);
            let (mut cf, tf) = if vert {
                self.k.dct.dct4x8_t(&resid, &self.ctx.quant)
            } else {
                self.k.dct.dct8x4_t(&resid, &self.ctx.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, lw, lh, px, py);
            let rr = if vert {
                self.k.idct.idct_dequant_4x8(&cf, &self.ctx.quant)
            } else {
                self.k.idct.idct_dequant_8x4(&cf, &self.ctx.quant)
            };
            let distortion = self.luma_partition_distortion(
                px,
                py,
                lw,
                lh,
                self.ctx.quant.ac_q() as f32,
                &pred[..],
                0,
                &rr[..],
            );
            // PRICE PARITY with the NONE leg (rd_cost_square dim=8): real
            // coefficient bits and the same flat mode-cost model. The old
            // proxy rate + ~5-bit kf_y cost subsidized each leaf by ~25
            // bits, which is why rect8 only ever "won" falsely.
            let bits =
                self.luma_rect_bits(&cf, scan, lw, lh, px, py, m, 1) + self.mode_bits(px, py, m);
            let c = crate::partition_rd::rd_cost(distortion, mlam, bits);
            if c < best {
                best = c;
            }
        }
        best
    }
    fn rd_choice_rect32(
        &self,
        x8: usize,
        y8: usize,
        prdo: f32,
        known_none: Option<f32>,
        thr: bool,
        lhb: bool,
    ) -> PartVerdict {
        // Grey 4:0:0 keeps 8x8 leaves; the alpha profile (`mono_large`) runs a
        // luma-only NONE vs SPLIT here (the rect32 legs have no mono emitter).
        if self.cfg.is_mono() && !self.cfg.mono_large {
            let (px, py) = (x8 * 8, y8 * 8);
            return PartVerdict::unmeasured(Part16::Split, self.rd_cost_split32(px, py, prdo, thr, lhb));
        }
        if self.cfg.is_mono() && crate::tuning::fixed_size(self.cfg.speed) == 0 {
            let (px, py) = (x8 * 8, y8 * 8);
            let split = self.rd_cost_split32(px, py, prdo, thr, lhb);
            let none = known_none.unwrap_or_else(|| {
                let (htr32, hbl32) = self.leaf_edge_flags(px, py, 32, thr, lhb);
                self.rd_cost_none32(px, py, prdo, htr32, hbl32)
            });
            return if none < split {
                PartVerdict::unmeasured(Part16::None, none)
            } else {
                PartVerdict::unmeasured(Part16::Split, split)
            };
        }
        let (px, py) = (x8 * 8, y8 * 8);
        // Fixed-partition mode: no R-D at all at this level.
        //
        // The returned cost is a placeholder. In fixed mode EVERY level
        // short-circuits (choose_64, here, and rd_choice_16_inner), so no
        // caller ever compares one of these costs against an alternative --
        // `choose_rect32` and `partition_choice_16` keep only the partition, and
        // `choose_64`'s call site is unreachable because 64 short-circuits
        // first. Skipping the pricing is the entire speed win; if a future
        // caller starts consuming the cost, this must change with it.
        match crate::tuning::fixed_size(self.cfg.speed) {
            0 => {}
            32 => return PartVerdict::unmeasured(Part16::None, 0.0),
            n if n < 32 => return PartVerdict::unmeasured(Part16::Split, 0.0),
            _ => {}
        }
        let coupled_square =
            !self.cfg.is_mono() && self.cfg.speed.at_least_slow() && joint_luma_uv_proxy_enabled();
        let chroma_split = if self.cfg.is_mono()
            || !self.cfg.speed.full_partition_rdo()
            || coupled_square
            || bottomup_split32()
        {
            // Bottom-up: the 16-level child totals already include chroma.
            0.0
        } else {
            self.rd_cost_chroma_partition(px, py, 32, Part16::Split, prdo, false)
        };
        // Small split-favoring bias: the SATD distortion proxy undervalues the
        // detail a single 32x32 loses when it merges four busier 16x16s, so a
        // pure comparison over-merges on textured content (SSIMULACRA2 penalizes
        // it). This is a mild, symmetric thumb (not the old prefilter that
        // skipped the comparison entirely).
        let rd_none = known_none.unwrap_or_else(|| {
            let (htr32, hbl32) = self.leaf_edge_flags(px, py, 32, thr, lhb);
            self.rd_cost_none32(px, py, prdo, htr32, hbl32)
                + if self.cfg.speed.full_partition_rdo() && !coupled_square {
                    self.rd_cost_chroma_partition(px, py, 32, Part16::None, prdo, false)
                } else {
                    0.0
                }
        }) * if self.top_band() && self.cfg.is_420() && !self.palette_exact(px, py, 32) {
            top_none_bias_420(self.ctx.aq.base_q)
        } else {
            self.none32_split_bias_at()
        } + rate_cost(self.mlam() * prdo, self.part_rate_bl(2, x8, y8, 0));
        // Flat-block SPLIT skip (aom VAR_BASED_PARTITION / SVT depth-refinement
        // analog): when the source detail in this 32x32 is small relative to the
        // AC quantizer, the four-child search cannot recover enough distortion
        // to beat NONE, so skip it outright. Our branch-and-bound only stops
        // once the running child total exceeds the bound; this never enters the
        // child search at all.
        let vbp_on = if self.cfg.is_420() {
            self.cfg.speed != Speed::Medium || crate::tuning::get().vbp_medium_420
        } else if self.cfg.is_422() {
            crate::tuning::get().vbp_422
        } else if !self.cfg.is_mono() {
            crate::tuning::get().vbp_444
        } else {
            false
        };
        let vbp_thresh = if self.cfg.is_420() {
            vbp_thresh_420()
        } else {
            crate::tuning::get().vbp_thresh_hi
        };
        let rd_split = if vbp_on && self.var_over_qstep2(px, py, 32) < vbp_thresh {
            f32::INFINITY
        } else {
            self.rd_cost_split32_bounded(px, py, prdo, rd_none, thr, lhb) + chroma_split
        };
        // IntraBC candidate: whole-32 exact-copy, all planes priced inside.
        let rd_ibc = if self.cfg.allow_intrabc {
            self.rd_cost_intrabc(px, py, 32, thr, prdo)
                .unwrap_or(f32::INFINITY)
        } else {
            f32::INFINITY
        };
        if rd_ibc < rd_none.min(rd_split) {
            return PartVerdict::unmeasured(Part16::Intrabc, rd_ibc);
        }
        let margin = Self::proxy_margin(rd_none, rd_split);
        if !self.cfg.speed.full_partition_rdo() {
            return if rd_none <= rd_split {
                PartVerdict { part: Part16::None, cost: rd_none, margin }
            } else {
                PartVerdict { part: Part16::Split, cost: rd_split, margin }
            };
        }
        // (previous state, kept for the record:)
        // 32-level HORZ/VERT candidates were DISABLED (2026-07-21): with
        // bottom-up best-cost children pricing SPLIT honestly, the DC-only /
        // DCT-only 32x16 rect leaves are net-harmful — removing them is
        // -0.79% tuning / **-1.93% HOLDOUT** at 4:2:0 and -2.03% / **-2.71%**
        // at 4:4:4, EVERY holdout image improving (same root cause as the
        // 16x16 rect finding: under-tooled rect leaves vs fully-equipped
        // square alternatives). Re-enabling requires equipping the rect
        // leaves (mode + tx-type search) first. The old gates
        // (UNPRUNED_RECT32_MIN_QINDEX / AC_Q_HORZ_MIN) went with them; the
        // partial prunes measured strictly between off and on.
        if rd_none <= rd_split {
            PartVerdict { part: Part16::None, cost: rd_none, margin }
        } else {
            PartVerdict { part: Part16::Split, cost: rd_split, margin }
        }
    }

    /// 32x32 shared-partition choice: NONE vs bottom-up SPLIT (32-level
    /// HORZ/VERT candidates were removed — see `rd_choice_rect32`).
    fn choose_rect32(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> PartVerdict {
        let (px, py) = (x8 * 8, y8 * 8);
        self.rd_choice_rect32(x8, y8, self.perceptual_rd_scale(px, py, 32), None, thr, lhb)
    }

    fn rd_cost_none32(
        &self,
        px: usize,
        py: usize,
        prdo: f32,
        have_tr: bool,
        have_bl: bool,
    ) -> f32 {
        // The whole-32 leg is compared against four 16x16 children; pricing
        // either side DC-only makes the comparison depend on which side happens
        // to suit DC, not on what the encoder will actually code. `rd_cost_square`
        // carries the same mode set and transform refinement the final block
        // gets, so both legs are now estimated the same way.
        if full_partition_proxy32() {
            return self.rd_cost_square(px, py, 32, have_tr, have_bl, prdo);
        }
        let (acq, dcq) = (self.ctx.quant.ac_q() as f32, self.ctx.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let dc = self.k.intrapred.dc_pred_32x32(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
        let mut resid = self.sbuf_i1024();
        self.k.rd.residual_dc(&mut resid[..], &self.cfg.src[0], self.cfg.w, px, py, 32, 32, dc);
        let (mut cf, tf) = self.k.dct.dct32x32_t(&resid, &self.ctx.quant);
        trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
        let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
        let distortion = self.luma_partition_distortion(
            px,
            py,
            32,
            32,
            self.ctx.quant.ac_q() as f32,
            &[],
            dc,
            &rr[..],
        );
        crate::partition_rd::rd_cost(
            distortion,
            mlam,
            self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0),
        )
    }

    fn rd_cost_split32(&self, px: usize, py: usize, prdo: f32, thr: bool, lhb: bool) -> f32 {
        self.rd_cost_split32_bounded(px, py, prdo, f32::INFINITY, thr, lhb)
    }

    /// As [`Self::rd_cost_split32`], but may stop summing children once the
    /// running total exceeds `bound` and return that partial sum.
    fn rd_cost_split32_bounded(
        &self,
        px: usize,
        py: usize,
        prdo: f32,
        bound: f32,
        thr: bool,
        lhb: bool,
    ) -> f32 {
        // Bottom-up child costing: charge each 16x16 child its BEST achievable
        // total (the full 16-level candidate min, luma + chroma), not its
        // forced-NONE cost. Forced-NONE is an upper bound on the child, so the
        // old pricing systematically overestimated SPLIT and over-merged
        // whenever a child would itself split or go rect. The 16-level totals
        // already include their own chroma partition costs, so the caller must
        // NOT add a 32-level chroma term on top (see `rd_choice_rect32`).
        if bottomup_split32() && (!self.cfg.is_mono() || self.cfg.mono_large) && self.cfg.speed.full_partition_rdo() {
            let mlam = self.mlam() * prdo;
            let mut total =
                rate_cost(mlam, self.part_rate_bl(2, px / 8, py / 8, 3) * split32_signal_mult());
            for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
                let (cthr, clhb) = Self::child_edge_flags(sx, sy, thr, lhb);
                total += self
                    .rd_choice_16((px + sx) / 8, (py + sy) / 8, cthr, clhb)
                    .1;
                if total > bound {
                    return total; // SPLIT already lost; skip the rest
                }
            }
            return total;
        }
        if full_partition_proxy32() {
            let mlam = self.mlam() * prdo;
            let mut total = rate_cost(mlam, self.part_rate_bl(2, px / 8, py / 8, 3) * split32_signal_mult());
            for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
                let (cthr, clhb) = Self::child_edge_flags(sx, sy, thr, lhb);
                let (chtr, chbl) = self.leaf_edge_flags(px + sx, py + sy, 16, cthr, clhb);
                total += self.rd_cost_square(px + sx, py + sy, 16, chtr, chbl, prdo);
                if total > bound {
                    return total; // SPLIT already lost; skip the rest
                }
            }
            return total;
        }
        let (acq, dcq) = (self.ctx.quant.ac_q() as f32, self.ctx.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // This is an empirical proxy-gap allowance, not literal partition
        // entropy. Charge it once; the historical `* 4` double-counted it.
        let mut total = rate_cost(mlam, self.part_rate_bl(2, px / 8, py / 8, 3) * split32_signal_mult());
        for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
            let dc = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, px + sx, py + sy, self.cfg.bd as i32);
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_dc(
                &mut resid[..],
                &self.cfg.src[0],
                self.cfg.w,
                px + sx,
                py + sy,
                16,
                16,
                dc,
            );
            let (mut cf, tf) = self.k.dct.dct16x16_t(&resid, &self.ctx.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_16X16, lam);
            let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
            let distortion = self.luma_partition_distortion(
                px + sx,
                py + sy,
                16,
                16,
                self.ctx.quant.ac_q() as f32,
                &[],
                dc,
                &rr[..],
            );
            total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&cf, &SCAN_16X16, 16, px + sx, py + sy, DC_PRED, 1),
            );
            if total > bound {
                return total; // SPLIT already lost; skip the rest
            }
        }
        total
    }

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
    ) -> ([i32; 1024], i64, f32, [u16; 1024]) {
        let mut saved = self.sbuf_u1024();
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        // Progressive sub-transform contexts (external review round 2,
        // finding 3): each TX_16X16 sees the previous quadrant's result.
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 8] = *self.ctx.a_coef[0][bx4_0..].first_chunk().unwrap();
        let saved_l: [u8; 8] = *self.ctx.l_coef[0][by4_0..].first_chunk().unwrap();
        let mut cf4 = self.sbuf_i1024();
        let mut sse_sum = 0i64;
        let mut bits_sum = 0.0f32;
        for (qi, &(sx, sy)) in Self::Q32.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (16, 0) => (have_tr, false),
                (0, 16) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = self.sbuf_i256();
            if mode == DC_PRED && delta == 0 {
                let d = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                *pred = [d; 256];
            } else {
                self.k.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    bx,
                    by,
                    16,
                    16,
                    tr,
                    bl,
                    self.cfg.w,
                    self.cfg.h,
                    block_ftype,
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
                bx,
                by,
                16,
                16,
            );
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 4, 4);
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
                ds,
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
            );
            let rr = self.k.idct.idct_dequant_16x16(&cf, &self.ctx.quant);
            sse_sum += sse_recon::<256, 16>(&self.k.rd, &pred, &rr, self.src_blk(0, bx, by, 16, 16), self.cfg.bd);
            bits_sum += self.luma_bits_ctx_bounded(
                &cf,
                &SCAN_16X16,
                16,
                bx,
                by,
                mode,
                1,
                sk,
                ds,
                f32::INFINITY,
            );
            let res_ctx = Self::coef_res_ctx(&cf, &SCAN_16X16);
            self.ctx.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
            self.ctx.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
            self.k.rd.reconstruct(
                &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                self.cfg.w,
                None,
                &pred[..],
                &rr,
                16,
                16,
                self.cfg.bd,
            );
            cf4[qi * 256..qi * 256 + 256].copy_from_slice(&cf);
        }
        // The decoder-exact reconstruction of this trial, for the caller.
        let mut rec = [0u16; 1024];
        for ry in 0..32 {
            rec[ry * 32..ry * 32 + 32].copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        for ry in 0..32 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.ctx.a_coef[0][bx4_0..bx4_0 + 8].copy_from_slice(&saved_a);
        self.ctx.l_coef[0][by4_0..by4_0 + 8].copy_from_slice(&saved_l);
        (*cf4, sse_sum, bits_sum, rec)
    }

    /// Luma search of a 32x32 leaf: prediction mode (with angle delta, filter
    /// intra or palette) and transform split. Leaves the tile unchanged:
    /// `&mut` only because the TX-split trials rebuild the block in the tile
    /// reconstruction and restore it.
    fn pick_luma32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) -> Luma32Pick {
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 32);
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 32);
        // emit_prdo/emit_mlam already carry the perceptual scale; the extra
        // multiplication here ran square leaves at scale^2 while rect leaves,
        // 64s and the partition decision used scale^1 (external review
        // 2026-07-27, finding 3). `prdo` stays for the coupled chroma path.
        let prdo = self.perceptual_rd_scale(px, py, 32);
        // luma intra mode search (non-directional + directional; the TX_32X32
        // residual transform is always DCT_DCT, so the mode affects prediction
        // only). Mirrors the 16x16 search.
        let mut best_mode = DC_PRED;
        let mut best_delta: i32 = 0;
        let mut lpred = self.sbuf_i1024();
        let mut lcf = self.sbuf_i1024();
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_filter_intra = None;
        // Decoder-exact reconstruction of a TX-split winner.
        let mut split_rec: Option<[u16; 1024]> = None;
        let mut luma_beam: [Option<Luma32BeamCandidate>; JOINT_LARGE_BEAM] =
            std::array::from_fn(|_| None);
        let mut ltf = self.sbuf_f1024(); // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.cfg.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        let joint_large =
            !self.cfg.is_mono() && self.cfg.speed.at_least_slow() && joint_luma_uv_large_enabled();
        let mode_shortlist = self.rank_luma_candidates::<1024>(
            modes,
            px,
            py,
            32,
            32,
            have_tr,
            have_bl,
            // 32-blocks are rare enough (40-550/frame) that the 444-class
            // finalist budget is ~free here.
            if self.cfg.speed.at_least_slow() {
                5
            } else {
                self.luma_mode_budget_eff()
            },
        );
        for &(m, delta) in &mode_shortlist {
            let mut pred = self.sbuf_i1024();
            if m == DC_PRED {
                let d = self.k.intrapred.dc_pred_32x32(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32);
                *pred = [d; 1024];
            } else {
                self.k.intrapred.predict_nd_ad(
                    m,
                    delta,
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
            if self.cfg.speed.per_candidate_rdoq_av1() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    32,
                    self.dcdf(),
                    3,
                    0,
                    &self.dcdf().eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                    self.ctx.quant.qm_level(),
                    self.ctx.quant.qidx() as i32,
                );
            }
            let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
            let sse = sse_recon::<1024, 32>(&self.k.rd, &pred, &rr, self.src_blk(0, px, py, 32, 32), self.cfg.bd);
            let syntax_bits = self.luma_intra_syntax_bits(px, py, 32, 32, m, delta);
            let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, m, 0) + syntax_bits;
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                best_delta = delta;
                *lpred = *pred;
                *lcf = cf;
                *ltf = tf;
                best_dct_sse = sse;
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
                    let mut beam_cf = self.sbuf_i1024();
                    let mut beam_tf = self.sbuf_f1024();
                    *beam_cf = cf;
                    *beam_tf = tf;
                    for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                        luma_beam[i] = luma_beam[i - 1].take();
                    }
                    luma_beam[pos] = Some(Luma32BeamCandidate {
                        luma_cost: cost,
                        mode: m,
                        delta,
                        pred,
                        cf: beam_cf,
                        tf: beam_tf,
                        sse,
                        palette: None,
                    });
                }
            }
        }
        // Palette candidate at 32x32 (mirror of the 16x16 one): x_screen is
        // 86% palette-eligible at this size, and without a 32 candidate the
        // partition pays the palette header 4x through a forced SPLIT.
        let mut best_palette32: Option<LossyLumaPalette> = None;
        if self.try_palette()
            && let Some(hist) = block_color_histogram(&self.cfg.src[0], self.cfg.w, px, py, 32, 32)
        {
            for (palette, pred) in
                self.rank_luma_palette_candidates::<1024>(&hist, px, py, 32, 32, mlam * K_PART_LAMBDA)
            {
                let mut resid = self.sbuf_i1024();
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &pred,
                    &self.cfg.src[0],
                    self.cfg.w,
                    px,
                    py,
                    32,
                    32,
                );
                let (mut cf, tf) = self.k.dct.dct32x32_t(&resid, &self.ctx.quant);
                if self.cfg.speed.per_candidate_rdoq_av1() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_32X32,
                        lam,
                        32,
                        32,
                        self.dcdf(),
                        3,
                        0,
                        &self.dcdf().eob_bin_1024_l,
                        self.dc_sign_ctx_32(0, px / 4, py / 4),
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
                let sse = sse_recon::<1024, 32>(&self.k.rd, &pred, &rr, self.src_blk(0, px, py, 32, 32), self.cfg.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0)
                    + self.mode_bits(px, py, DC_PRED)
                    + self.palette_rate_bits(px, py, &palette);
                let cost = rd_cost_i64(sse, mlam, bits);
                if cost < best_eff {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    best_delta = 0;
                    best_filter_intra = None;
                    best_palette32 = Some(palette.clone());
                    *lpred = pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
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
                        let mut beam_pred = self.sbuf_i1024();
                        let mut beam_cf = self.sbuf_i1024();
                        let mut beam_tf = self.sbuf_f1024();
                        *beam_pred = pred;
                        *beam_cf = cf;
                        *beam_tf = tf;
                        for i in (pos + 1..JOINT_LARGE_BEAM).rev() {
                            luma_beam[i] = luma_beam[i - 1].take();
                        }
                        luma_beam[pos] = Some(Luma32BeamCandidate {
                            luma_cost: cost,
                            mode: DC_PRED,
                            delta: 0,
                            pred: beam_pred,
                            cf: beam_cf,
                            tf: beam_tf,
                            sse,
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
                    + self.joint_uv_cost32(
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
                *lpred = *candidate.pred;
                *lcf = *candidate.cf;
                *ltf = *candidate.tf;
                best_dct_sse = candidate.sse;
                best_filter_intra = None;
                best_palette32 = candidate.palette;
            }
        }
        if self.cfg.speed.at_least_slow() {
            let bsize = av1_block_size_index(32, 32);
            let mut fi_preds = [[0i32; 1024]; 5];
            for (fi_si, &filter_mode) in self
                .rank_filter_intra_modes::<1024>(px, py, 32, 32, self.cfg.speed.filter_intra_refine_budget(self.ss444()), &mut fi_preds)
                .iter()
                .enumerate()
            {
                let mut pred = self.sbuf_i1024();
                pred[..].copy_from_slice(&fi_preds[fi_si]);
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
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    32,
                    self.dcdf(),
                    3,
                    0,
                    &self.dcdf().eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, px / 4, py / 4),
                    self.ctx.quant.qm_level(),
                    self.ctx.quant.qidx() as i32,
                );
                let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
                let sse = sse_recon::<1024, 32>(&self.k.rd, &pred, &rr, self.src_blk(0, px, py, 32, 32), self.cfg.bd);
                // The admission below needs `sse <= best_dct_sse` (unless the RD-only
                // study knob is on): a candidate that
                // already lost on SSE skips the entropy-accurate rate pass.
                if !crate::tuning::get().filter_intra_rd_only && sse > best_dct_sse {
                    continue;
                }
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if (crate::tuning::get().filter_intra_rd_only || sse <= best_dct_sse) && cost < best_eff
                {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    best_delta = 0;
                    *lpred = *pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
                    best_filter_intra = Some(filter_mode);
                    best_palette32 = None;
                }
            }
        }
        // Refine the directional winner, retaining its current angle cost if
        // it came from the global angular shortlist.
        if self.cfg.speed.try_angle_deltas_av1(32, self.cfg.base_q_idx)
            && self.angle_delta_refines(best_mode)
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
            let ds = self.dc_sign_ctx_32(0, px / 4, py / 4);
            let wrr = self.k.idct.idct_dequant_32x32(&lcf, &self.ctx.quant);
            let wsse = sse_recon::<1024, 32>(&self.k.rd, &lpred, &wrr, self.src_blk(0, px, py, 32, 32), self.cfg.bd);
            let wbits = self.luma_bits(&lcf[..], &SCAN_32X32, 32, px, py, best_mode, 0);
            let mut best_ad_cost = rd_cost_i64(
                wsse,
                mlam,
                wbits + cdf_cost(&ad_cdf, (best_delta + 3) as usize),
            );
            let mut ad_pred0 = self.sbuf_i1024();
            let mut ad_pred1 = self.sbuf_i1024();
            let mut ad_scratch = self.sbuf_i1024();
            let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
            for (di, &d) in self.rank_angle_deltas::<1024>(
                    best_mode, px, py, 32, 32, have_tr, have_bl, 2, &mut ad_preds,
                )
                .iter()
                .enumerate()
            {
                let pred: &[i32; 1024] = &*ad_preds[di];
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
                if self.cfg.speed.per_candidate_rdoq_av1() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_32X32,
                        lam,
                        32,
                        32,
                        self.dcdf(),
                        3,
                        0,
                        &self.dcdf().eob_bin_1024_l,
                        ds,
                        self.ctx.quant.qm_level(),
                        self.ctx.quant.qidx() as i32,
                    );
                }
                let rr = self.k.idct.idct_dequant_32x32(&cf, &self.ctx.quant);
                let sse = sse_recon::<1024, 32>(&self.k.rd, pred, &rr, self.src_blk(0, px, py, 32, 32), self.cfg.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, best_mode, 0);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    *lpred = *pred;
                    *lcf = cf;
                    *ltf = tf;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if !self.cfg.speed.per_candidate_rdoq_av1() {
            trellis_optimize_ctx(
                &mut lcf[..],
                &ltf[..],
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                32,
                self.dcdf(),
                3,
                0,
                &self.dcdf().eob_bin_1024_l,
                self.dc_sign_ctx_32(0, px / 4, py / 4),
                self.ctx.quant.qm_level(),
                self.ctx.quant.qidx() as i32,
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
        let mut d1_rd: Option<f32> = None;
        let banding32 = self.banding_risk(px, py, 32);
        if best_filter_intra.is_none()
            && best_palette32.is_none()
            && (banding32 || crate::tuning::get().tx32_split_rd)
        {
            let none_sse = sse_recon::<1024, 32>(&self.k.rd, &lpred, &self.k.idct.idct_dequant_32x32(&lcf, &self.ctx.quant), self.src_blk(0, px, py, 32, 32), self.cfg.bd);
            let (cf4, sse_s, bits_s, rec4) =
                self.split32_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam);
            // Acceptance: the split must genuinely improve SSE. block16 uses a
            // permissive +25% tolerance at 16x16, but four TX_16X16 cost far more
            // syntax than four TX_8X8 do, so copying that tolerance here lets in
            // bad splits — measured +6.7% BD-rate on detailed content. Requiring
            // a real improvement keeps the banding win and removes the loss.
            // `tx32_split_rd`: outside the banding trigger, plain RD decides
            // (the synthetic-gap transplant found 32x32 gradient+edge blocks
            // coded whole at 2x the bytes and 3x the error of aom's four TX16).
            let guarded_take = if banding32 {
                (sse_s as i128) * 1024 <= (none_sse as i128) * (1024 - SPLIT32_SSE_MARGIN as i128)
            } else {
                let none_bits = self.luma_bits(&lcf[..], &SCAN_32X32, 32, px, py, best_mode, 0);
                rd_cost_i64(sse_s, mlam, bits_s + self.tx_depth_bits(px, py, 32, 32, 1))
                    < rd_cost_i64(none_sse, mlam, none_bits + self.tx_depth_bits(px, py, 32, 32, 0))
            };
            if guarded_take {
                tx_split = true;
                d1_rd = Some(rd_cost_i64(
                    sse_s,
                    mlam,
                    bits_s + self.tx_depth_bits(px, py, 32, 32, 1),
                ));
                *lcf = cf4;
                split_rec = Some(rec4);
            }
        }
        // tx_depth = 2 (`tx32_depth2`): sixteen TX_8X8 with per-TX txtp,
        // against the current winner (whole or depth-1) on plain RD. A 32x32
        // whose only detail is one hard edge in a corner codes that edge in
        // one or two 8x8 transforms instead of ringing a TX_32X32 across the
        // whole block (illustrations: our non-skipped 32x32s cost ~45 B each).
        let mut tx_d2: Option<[u8; 16]> = None;
        if crate::tuning::get().tx32_depth2
            && best_filter_intra.is_none()
            && best_palette32.is_none()
        {
            let cur_rd = match d1_rd {
                Some(r) => r,
                None => {
                    let whole_sse = sse_recon::<1024, 32>(
                        &self.k.rd,
                        &lpred,
                        &self.k.idct.idct_dequant_32x32(&lcf, &self.ctx.quant),
                        self.src_blk(0, px, py, 32, 32),
                        self.cfg.bd,
                    );
                    let whole_bits = self.luma_bits(&lcf[..], &SCAN_32X32, 32, px, py, best_mode, 0);
                    rd_cost_i64(whole_sse, mlam, whole_bits + self.tx_depth_bits(px, py, 32, 32, 0))
                }
            };
            let (cf16, sse_2, bits_2, d2_txtps, rec2) =
                self.split32_depth2_try(px, py, best_mode, best_delta, have_tr, have_bl, lam, mlam);
            let d2_rd = rd_cost_i64(sse_2, mlam, bits_2 + self.tx_depth_bits(px, py, 32, 32, 2));
            if d2_rd < cur_rd {
                tx_d2 = Some(d2_txtps);
                tx_split = false;
                *lcf = cf16;
                split_rec = Some(rec2);
            }
        }
        Luma32Pick {
            cf: lcf,
            pred: lpred,
            y_mode: best_mode,
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette32,
            tx_split,
            tx_d2,
            split_rec,
        }
    }

    /// Code a 32x32 leaf: search luma, then chroma for that luma, then emit.
    fn code_block32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 8);
        let Luma32Pick {
            cf: lcf,
            pred: lpred,
            y_mode: best_mode,
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette32,
            tx_split,
            tx_d2,
            split_rec,
        } = self.pick_luma32(x8, y8, have_tr, have_bl);
        #[cfg(debug_assertions)]
        if let Some(rec) = &split_rec {
            let (px, py) = (x8 * 8, y8 * 8);
            let again = match tx_d2 {
                Some(t) => self.split32_depth2_recon_from_cf(px, py, best_mode, best_delta, have_tr, have_bl, &lcf, t),
                None => self.split32_luma_recon_from_cf(px, py, best_mode, best_delta, have_tr, have_bl, &lcf),
            };
            debug_assert!(rec[..] == again[..], "split-TX trial recon differs from the committed-coefficient recon");
            self.sc().put_u1024(again);
        }
        if tx_d2.is_some() {
            // Deblock works on TRANSFORM edges: sixteen TX_8X8 tiles.
            for j in 0..4 {
                for i in 0..4 {
                    self.record_tx_blk(x8 + i, y8 + j, 2);
                }
            }
        } else if tx_split {
            // Deblock works on TRANSFORM edges: re-record as four TX_16X16 tiles.
            self.record_tx_blk(x8, y8, 4);
            self.record_tx_blk(x8 + 2, y8, 4);
            self.record_tx_blk(x8, y8 + 2, 4);
            self.record_tx_blk(x8 + 2, y8 + 2, 4);
        }
        let l = Luma32 {
            cf: &lcf,
            pred: &lpred,
            y_mode: best_mode,
            luma_zero: self.k.rd.all_zero_i32(&lcf[..]),
            angle_delta: best_delta,
            filter_intra: best_filter_intra,
            palette: best_palette32.as_ref(),
            tx_split,
            tx_d2,
            split_rec: split_rec.as_ref(),
            have_tr,
            have_bl,
        };
        match self.cfg.chroma {
            Chroma::Mono => {
                let skip = l.palette.is_none() && l.luma_zero;
                self.code_header_luma32(x8, y8, &l, skip, DC_PRED, None, 0, None);
            }
            Chroma::I420 => {
                let uv = self.pick_uv32_420(x8, y8, &l);
                self.emit_block32_420(x8, y8, &l, uv);
            }
            Chroma::I422 => {
                let uv = self.pick_uv32_422(x8, y8, &l);
                self.emit_block32_422(x8, y8, &l, uv);
            }
            Chroma::I444 => {
                let uv = self.pick_uv32_444(x8, y8, &l);
                self.emit_block32_444(x8, y8, &l, uv);
            }
        }
    }

    /// Emit a 32x32 block's header and luma: skip flag, modes, palette and
    /// filter-intra syntax, transform depth, then the luma coefficients and
    /// reconstruction of the winner `l`.
    #[allow(clippy::too_many_arguments)]
    fn code_header_luma32(
        &mut self,
        x8: usize,
        y8: usize,
        l: &Luma32<'_>,
        block_skip: bool,
        uv_mode: usize,
        cfl: Option<[i32; 2]>,
        uv_delta: i32,
        uv_palette: Option<&LossyUvPalette>,
    ) {
        let Luma32 {
            cf: lcf,
            pred: lpred,
            y_mode,
            angle_delta,
            filter_intra,
            palette,
            tx_split,
            tx_d2,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 4, block_skip);
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
            self.emit_uv_mode(y_mode, uv_mode, cfl, uv_delta, px, py, 32, 32);
        }
        self.emit_palette_mode_info(px, py, 32, 32, y_mode, !self.cfg.is_mono(), palette, uv_palette);
        if palette.is_none() {
            self.emit_filter_intra(y_mode, 32, 32, filter_intra);
        }
        if let Some(p) = palette {
            self.emit_palette_map(p);
        }
        if let Some(up) = uv_palette {
            self.emit_palette_uv_map(up);
        }
        self.code_tx_depth(px, py, 32, 32, if tx_d2.is_some() { 2 } else { tx_split as usize });
        // Derived ONCE at the block origin (dav1d), before a_mode/l_mode are filled.
        let block_ftype = self.luma_filter_type(px, py);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.ctx.a_skip[bx4..bx4 + 8].fill(sv);
        self.ctx.l_skip[by4..by4 + 8].fill(sv);
        self.ctx.a_mode[bx4..bx4 + 8].fill(mv);
        self.ctx.l_mode[by4..by4 + 8].fill(mv);
        if let Some(txtps) = tx_d2 {
            // tx_depth = 2: SIXTEEN TX_8X8 in raster order, per-TX prediction
            // from the running recon (grid edge-flag rule, as the 16x16 depth-2
            // path) and per-TX txtp from the TX_8X8 intra set.
            for j in 0..4usize {
                for i in 0..4usize {
                    let ti = j * 4 + i;
                    let (bx, by) = (px + i * 8, py + j * 8);
                    let (qbx4, qby4) = (bx / 4, by / 4);
                    let mut cfq = [0i32; 64];
                    cfq.copy_from_slice(&lcf[ti * 64..ti * 64 + 64]);
                    let res_ctx = if block_skip {
                        0x40
                    } else {
                        let sk = self.skip_ctx_split(qbx4, qby4, 2, 2);
                        let ds = self.dc_sign_ctx(0, qbx4, qby4);
                        let t = txtps[ti];
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
                    self.ctx.a_coef[0][qbx4..qbx4 + 2].fill(res_ctx);
                    self.ctx.l_coef[0][qby4..qby4 + 2].fill(res_ctx);
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
                    let mut pred = [0i32; 64];
                    if y_mode == DC_PRED && angle_delta == 0 {
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
                        match txtps[ti] {
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
            }
            return;
        }
        if tx_split {
            let maxv = (1i32 << self.cfg.bd) - 1;
            for (qi, &(sx, sy)) in Self::Q32.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let mut cfq = self.sbuf_i256();
                cfq.copy_from_slice(&lcf[qi * 256..qi * 256 + 256]);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
                    let ds = self.dc_sign_ctx_16(0, qbx4, qby4);
                    encode_tx16_coeffs_adapt(
                        &mut self.ent.enc,
                        &mut self.ent.cdfs,
                        &cfq,
                        false,
                        sk,
                        ds,
                        filter_intra_tx_mode(None, y_mode),
                        1, // DCT_DCT
                    )
                };
                self.ctx.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
                self.ctx.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
                let (tr, bl) = match (sx, sy) {
                    (0, 0) => (py > 0, px > 0),
                    (16, 0) => (have_tr, false),
                    (0, 16) => (true, have_bl),
                    _ => (false, false),
                };
                let mut pred = self.sbuf_i256();
                if y_mode == DC_PRED && angle_delta == 0 {
                    let d = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                    *pred = [d; 256];
                } else {
                    self.k.intrapred.predict_nd_ad(
                        y_mode,
                        angle_delta,
                        &self.ctx.recon[0],
                        self.cfg.w,
                        bx,
                        by,
                        16,
                        16,
                        tr,
                        bl,
                        self.cfg.w,
                        self.cfg.h,
                        block_ftype,
                        &mut pred[..],
                        self.cfg.bd,
                    );
                }
                let rr = if block_skip {
                    [0i32; 256]
                } else {
                    self.k.idct.idct_dequant_16x16(&cfq, &self.ctx.quant)
                };
                for ry in 0..16 {
                    let drow = &mut self.ctx.recon[0][(by + ry) * self.cfg.w + bx..];
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
            encode_tx32_coeffs_adapt(&mut self.ent.enc, &mut self.ent.cdfs, lcf, false, sk, ds)
        };
        self.ctx.a_coef[0][bx4..bx4 + 8].fill(lres);
        self.ctx.l_coef[0][by4..by4 + 8].fill(lres);
        let lrr = if block_skip {
            [0i32; 1024]
        } else {
            self.k.idct.idct_dequant_32x32(lcf, &self.ctx.quant)
        };
        for (ry, (prow, rrow)) in lpred
            .as_chunks::<32>()
            .0
            .iter()
            .zip(lrr.as_chunks::<32>().0.iter())
            .enumerate()
        {
            let drow = &mut self.ctx.recon[0][(py + ry) * self.cfg.w + px..];
            recon_add_pred(drow, prow, rrow, (1 << self.cfg.bd) - 1);
        }
    }

    /// Search of one 8x4 / 4x8 leaf (`half` of an 8x8 HORZ / VERT partition):
    /// luma mode, then chroma where this half carries it (at 4:2:0 the shared
    /// 4x4 chroma rides on the second half).
    fn pick_rect8(
        &self,
        x8: usize,
        y8: usize,
        vert: bool,
        half: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> Rect8Pick {
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 8);
        let sel_mlam = self.emit_mlam(x8 * 8, y8 * 8, 8);
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
        let (lw, lh) = if vert { (4usize, 8usize) } else { (8, 4) };
        let (px, py) = if vert {
            (x8 * 8 + half * 4, y8 * 8)
        } else {
            (x8 * 8, y8 * 8 + half * 4)
        };
        // Spec z-order edge availability per half (mirrors the SPLIT4
        // table; must match the decoder exactly or directional predictions
        // drift).
        let (sub_tr, sub_bl) = if vert {
            if half == 0 {
                (py > 0, have_bl)
            } else {
                (have_tr, false)
            }
        } else if half == 0 {
            (have_tr, px > 0)
        } else {
            (false, have_bl)
        };
        let (bx4, by4) = (px / 4, py / 4);
        let lscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
        let ftype = self.luma_filter_type(px, py);
        let yctx_sel = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        let mut best_cost = f32::INFINITY;
        let mut best_mode = DC_PRED;
        let mut lpred_arr = [0i32; 32];
        let mut lcf = [0i32; 32];
        let mut cand_pred = [0i32; 32];
        for m in 0..13usize {
            if m == DC_PRED {
                let dc = if vert {
                    self.k.intrapred.dc_pred_4x8(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
                } else {
                    self.k.intrapred.dc_pred_8x4(&self.ctx.recon[0], self.cfg.w, px, py, self.cfg.bd as i32)
                };
                cand_pred[..lw * lh].fill(dc);
            } else {
                self.k.intrapred.predict_nd(
                    m,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    lw,
                    lh,
                    sub_tr,
                    sub_bl,
                    self.cfg.w,
                    self.cfg.h,
                    ftype,
                    &mut cand_pred,
                    self.cfg.bd,
                );
            }
            let mut resid = [0i32; 32];
            self.k.rd.residual_pred(
                &mut resid,
                &cand_pred,
                &self.cfg.src[0],
                self.cfg.w,
                px,
                py,
                lw,
                lh,
            );
            let (mut cf, tf) = if vert {
                self.k.dct.dct4x8_t(&resid, &self.ctx.quant)
            } else {
                self.k.dct.dct8x4_t(&resid, &self.ctx.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, lscan, lam, lw, lh, px, py);
            self.k.rd.preserve_dc(&mut cf[0], &resid[..lw * lh]);
            let rr = if vert {
                self.k.idct.idct_dequant_4x8(&cf, &self.ctx.quant)
            } else {
                self.k.idct.idct_dequant_8x4(&cf, &self.ctx.quant)
            };
            let sse = self.k.rd.sse_recon(&cand_pred, &rr, self.src_blk(0, px, py, lw, lh), self.cfg.bd);
            let bits = self.luma_rect_bits(&cf, lscan, lw, lh, px, py, m, 1)
                + cdf_cost(&self.dcdf().kf_y[yctx_sel], m);
            let cost = rd_cost_i64(sse, sel_mlam, bits);
            if cost < best_cost {
                best_cost = cost;
                best_mode = m;
                lpred_arr = cand_pred;
                lcf = cf;
            }
        }
        let luma_zero = self.k.rd.all_zero_i32(&lcf);
        // chroma present on this sub-block?
        let has_chroma = if self.cfg.is_420() {
            if vert { px % 8 != 0 } else { py % 8 != 0 } // 2nd sub-block only
        } else {
            true
        };
        let (cx, cy, cw, ch) = if self.cfg.is_420() {
            (x8 * 4, y8 * 4, 4usize, 4usize) // 4x4 over the 8x8 luma area
        } else if self.cfg.is_422() {
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
                    &self.k.intrapred,
                    &self.ctx.recon[plane],
                    self.cfg.cw,
                    cx,
                    cy,
                    cw,
                    ch,
                    self.cfg.bd as i32,
                );
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
                let (mut q, qt) = fwd_chroma_rect8(&self.k.dct, cw, ch, &resid, &self.ctx.cquant);
                let cscan = scan_rect8(cw, ch);
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                self.k.rd.preserve_dc(&mut q[0], &resid[..cn]);
                ccf[ci] = q;
            }
        }
        let chroma_zero = !has_chroma
            || (self.k.rd.all_zero_i32(&ccf[0]) && self.k.rd.all_zero_i32(&ccf[1]));
        let block_skip = luma_zero && chroma_zero;
        Rect8Pick {
            bx4,
            by4,
            block_skip,
            best_mode,
            has_chroma,
            lcf,
            lpred_arr,
            cw,
            ch,
            cbx4,
            cby4,
            ccf,
            cy,
            cx,
            cpred,
        }
    }

    /// Emit one 8x4 / 4x8 leaf from its search result.
    fn emit_rect8(&mut self, x8: usize, y8: usize, vert: bool, half: usize, pick: Rect8Pick) {
        let Rect8Pick {
            bx4,
            by4,
            block_skip,
            best_mode,
            has_chroma,
            lcf,
            lpred_arr,
            cw,
            ch,
            cbx4,
            cby4,
            ccf,
            cy,
            cx,
            cpred,
        } = pick;
        let maxval = (1 << self.cfg.bd) - 1;
        let (lw, lh) = if vert { (4usize, 8usize) } else { (8, 4) };
        let (px, py) = if vert {
            (x8 * 8 + half * 4, y8 * 8)
        } else {
            (x8 * 8, y8 * 8 + half * 4)
        };
        // ---- Emit. Every search above is done; nothing below decides.
        let sctx = (self.ctx.a_skip[bx4] + self.ctx.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        // 4-unit-origin record: the second half of the pair starts at a
        // 4px offset, which the 8px-unit record helper truncates away —
        // both halves then stamped the SAME origin and the internal edge
        // deblocked wrong (latent while rect8 only fired at frame edges,
        // where the filter clip masked it; exposed by the 2026-07-23
        // mid-frame enable).
        self.record_blk_rect4(px / 4, py / 4, (lw / 4).max(1) as u8, (lh / 4).max(1) as u8);
        // CDEF skip map: BOTH halves of an 8x4/4x8 pair live in the SAME
        // 8x8 cell (px/8 and py/8 are identical for py and py+4), so
        // marking it per half let the second half overwrite the first. A
        // coefficient-carrying first half followed by a skipped second half
        // marked the whole 8x8 as skipped and suppressed CDEF on real
        // residual. Accumulate and mark once after the loop, exactly as
        // SPLIT4 does with `all4_skip` (block16.rs). Same bug class as the
        // 4-unit-origin record fix noted above, missed for this map.
        let yctx = INTRA_MODE_CTX[self.ctx.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.ctx.l_mode[by4] as usize];
        self.ent.enc.encode_symbol(best_mode, &mut self.ent.cdfs.kf_y[yctx]);
        if has_chroma {
            self.emit_uv_mode(best_mode, DC_PRED, None, 0, px, py, lw, lh);
        }
        self.emit_palette_mode_info(px, py, lw, lh, best_mode, has_chroma, None, None);
        self.emit_filter_intra(best_mode, lw, lh, None);
        self.code_tx_depth(px, py, lw, lh, 0);
        let sv = block_skip as u8;
        let (aw, ah) = ((lw / 4).max(1), (lh / 4).max(1));
        self.ctx.a_skip[bx4..bx4 + aw].fill(sv);
        self.ctx.l_skip[by4..by4 + ah].fill(sv);
        self.ctx.a_mode[bx4..bx4 + aw].fill(best_mode as u8);
        self.ctx.l_mode[by4..by4 + ah].fill(best_mode as u8);
        let lres_ctx = if block_skip {
            0x40
        } else if vert {
            let ds = self.dc_sign_ctx_4x8_luma(bx4, by4);
            encode_4x8_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, 0, ds, best_mode, 1)
        } else {
            let ds = self.dc_sign_ctx_8x4_luma(bx4, by4);
            encode_8x4_luma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &lcf, 0, ds, best_mode, 1)
        };
        self.ctx.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.ctx.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 32]
        } else if vert {
            self.k.idct.idct_dequant_4x8(&lcf, &self.ctx.quant)
        } else {
            self.k.idct.idct_dequant_8x4(&lcf, &self.ctx.quant)
        };
        for ry in 0..lh {
            for rx in 0..lw {
                let i = ry * lw + rx;
                self.ctx.recon[0][(py + ry) * self.cfg.w + px + rx] =
                    (lpred_arr[i] + lrr[i]).clamp(0, maxval) as u16;
            }
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
                self.ctx.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                self.ctx.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    inv_chroma_rect8(&self.k.idct, cw, ch, &ccf[ci], &self.ctx.cquant)
                };
                for ry in 0..ch {
                    let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
                    recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                }
            }
        }
    }

    /// 8x8 rect: HORZ = two 8x4, VERT = two 4x8. Shared 4x4 chroma in 4:2:0
    /// (coded on 2nd sub-block); per-sub chroma in 4:4:4/4:2:2. V forbidden in 4:2:2.
    fn code_block8_rect(&mut self, x8: usize, y8: usize, vert: bool, have_tr: bool, have_bl: bool) {
        // skip8 is per 8x8: the unit is skip only when both halves are.
        let mut all2_skip = true;
        for half in 0..2 {
            let pick = self.pick_rect8(x8, y8, vert, half, have_tr, have_bl);
            all2_skip &= pick.block_skip;
            self.emit_rect8(x8, y8, vert, half, pick);
        }
        self.mark_skip8(x8, y8, 1, all2_skip);
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
                encode_8x4_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &cf.as_chunks::<32>().0[0], sk, ds)
            }
            (4, 8) => {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                encode_4x8_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &cf.as_chunks::<32>().0[0], sk, ds)
            }
            _ => {
                let sk = self.skip_ctx_4x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x4_chroma(plane, cbx4, cby4);
                encode_4x4_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &cf.as_chunks::<16>().0[0], sk, ds)
            }
        }
    }

    /// Trial-code the 32x32 as SIXTEEN TX_8X8 (`tx_depth = 2`, raster order):
    /// per-TX prediction from the running recon with the grid edge-flag rule,
    /// per-TX txtp from the TX_8X8 intra set (DCT, ADST_ADST, ADST_DCT,
    /// DCT_ADST, IDTX, V_DCT, H_DCT), progressive coefficient contexts.
    /// Returns (coefficients packed TX-raster-major, SSE, bits, per-TX txtps).
    /// Restores `self.recon` and the coefficient contexts.
    #[allow(clippy::too_many_arguments)]
    fn split32_depth2_try(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
        rd_lam: f32,
    ) -> ([i32; 1024], i64, f32, [u8; 16], [u16; 1024]) {
        let mut saved = self.sbuf_u1024();
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        let (dcq, acq) = (self.ctx.quant.dc_q() as f32, self.ctx.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 8] = *self.ctx.a_coef[0][bx4_0..].first_chunk().unwrap();
        let saved_l: [u8; 8] = *self.ctx.l_coef[0][by4_0..].first_chunk().unwrap();
        let mut cf16 = self.sbuf_i1024();
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let mut txtps = [1u8; 16];
        for j in 0..4usize {
            for i in 0..4usize {
                let ti = j * 4 + i;
                let (bx, by) = (px + i * 8, py + j * 8);
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
                for txtp in [4u8, 5, 6, 0, 2, 3] {
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
                txtps[ti] = btxtp;
                sse_sum += bsse;
                bits_sum += bbits;
                let res_ctx = Self::coef_res_ctx(&bcf, &SCAN_8X8);
                self.ctx.a_coef[0][qbx4..qbx4 + 2].fill(res_ctx);
                self.ctx.l_coef[0][qby4..qby4 + 2].fill(res_ctx);
                self.k.rd.reconstruct(
                    &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                    self.cfg.w,
                    None,
                    &pred,
                    &brr,
                    8,
                    8,
                    self.cfg.bd,
                );
                cf16[ti * 64..ti * 64 + 64].copy_from_slice(&bcf);
            }
        }
        // The decoder-exact reconstruction of this trial, for the caller.
        let mut rec = [0u16; 1024];
        for ry in 0..32 {
            rec[ry * 32..ry * 32 + 32].copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        for ry in 0..32 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.ctx.a_coef[0][bx4_0..bx4_0 + 8].copy_from_slice(&saved_a);
        self.ctx.l_coef[0][by4_0..by4_0 + 8].copy_from_slice(&saved_l);
        (*cf16, sse_sum, bits_sum, txtps, rec)
    }

    #[cfg(debug_assertions)]
    /// Decoder-exact reconstruction of a depth-2 (sixteen TX_8X8) 32x32 from
    /// committed coefficients: raster per-TX prediction from the running recon
    /// with the grid edge-flag rule and the per-TX inverse. Used by the CfL
    /// luma reference. Restores `self.recon`.
    #[allow(clippy::too_many_arguments)]
    fn split32_depth2_recon_from_cf(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lcf: &[i32; 1024],
        txtps: [u8; 16],
    ) -> Box<[u16; 1024]> {
        let mut saved = self.sc().take_u1024();
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        let block_ftype = self.luma_filter_type(px, py);
        let mut rec = self.sc().take_u1024();
        for j in 0..4usize {
            for i in 0..4usize {
                let ti = j * 4 + i;
                let (bx, by) = (px + i * 8, py + j * 8);
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
                let mut cfq = [0i32; 64];
                cfq.copy_from_slice(&lcf[ti * 64..ti * 64 + 64]);
                let rr = match txtps[ti] {
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
                    Some((&mut rec[j * 8 * 32 + i * 8..], 32)),
                    &pred,
                    &rr,
                    8,
                    8,
                    self.cfg.bd,
                );
            }
        }
        for ry in 0..32 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.sc().put_u1024(saved);
        rec
    }

    #[cfg(debug_assertions)]
    #[allow(clippy::too_many_arguments)]
    fn split32_luma_recon_from_cf(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lcf: &[i32; 1024],
    ) -> Box<[u16; 1024]> {
        let mut saved = self.sc().take_u1024();
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]);
        }
        let block_ftype = self.luma_filter_type(px, py);
        let mut rec = self.sc().take_u1024();
        for (qi, &(sx, sy)) in Self::Q32.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (16, 0) => (have_tr, false),
                (0, 16) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = self.sbuf_i256();
            if mode == DC_PRED && delta == 0 {
                let d = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[0], self.cfg.w, bx, by, self.cfg.bd as i32);
                *pred = [d; 256];
            } else {
                self.k.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    bx,
                    by,
                    16,
                    16,
                    tr,
                    bl,
                    self.cfg.w,
                    self.cfg.h,
                    block_ftype,
                    &mut pred[..],
                    self.cfg.bd,
                );
            }
            let mut cfq = self.sbuf_i256();
            cfq.copy_from_slice(&lcf[qi * 256..qi * 256 + 256]);
            let rr = self.k.idct.idct_dequant_16x16(&cfq, &self.ctx.quant);
            self.k.rd.reconstruct(
                &mut self.ctx.recon[0][by * self.cfg.w + bx..],
                self.cfg.w,
                Some((&mut rec[sy * 32 + sx..], 32)),
                &pred[..],
                &rr,
                16,
                16,
                self.cfg.bd,
            );
        }
        for ry in 0..32 {
            self.ctx.recon[0][(py + ry) * self.cfg.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.sc().put_u1024(saved);
        rec
    }

    /// Chroma search of a 32x32 block (444) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block32_444`.
    fn pick_uv32_444(&self, x8: usize, y8: usize, l: &Luma32<'_>) -> UvPick<SBuf<[i32; 1024]>, SBuf<[i32; 1024]>> {
        let Luma32 {
            cf: lcf,
            pred: lpred,
            y_mode,
            palette,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq, lam) = (
            self.ctx.cquant.dc_q() as f32,
            self.ctx.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        // plain-DC chroma.
        let mut ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cdc = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_32x32(&self.ctx.recon[plane], self.cfg.w, px, py, self.cfg.bd as i32);
            cdc[ci] = dc;
            let mut cresid = self.sbuf_i1024();
            self.k.rd.residual_dc(
                &mut cresid[..],
                &self.cfg.src[plane],
                self.cfg.w,
                px,
                py,
                32,
                32,
                dc,
            );
            let (q, qt) = self.k.dct.dct32x32_t(&cresid, &self.ctx.cquant);
            *ccf[ci] = q;
            self.chroma_rect_trellis(
                &mut ccf[ci][..],
                &qt,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                32,
                plane,
                px,
                py,
            );
            self.k.rd.preserve_dc(&mut ccf[ci][0], &cresid[..]);
        }
        // CfL: predict chroma from the reconstructed luma AC.
        let mut cfl_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cfl_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 32);
        if self.cfg.speed.full_chroma_rdo() {
            let mut luma_rec = self.sbuf_u1024();
            if let Some(rec) = l.split_rec {
                luma_rec.copy_from_slice(&rec[..]);
            } else {
                let lrr_cfl = self.k.idct.idct_dequant_32x32(lcf, &self.ctx.quant);
                recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, (1 << self.cfg.bd) - 1);
            }
            let mut ac = self.sbuf_i1024();
            self.k.intrapred
                .cfl_ac_444(&luma_rec[..], 32, 32, &mut ac[..]);
            let steps = crate::tuning::get().cfl_alpha_rd_steps;
            let a_e = [0, 1].map(|ci| {
                self.k.intrapred.cfl_best_alpha(&ac[..], self.src_blk(ci + 1, px, py, 32, 32), cdc[ci], self.cfg.bd)
            });
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cdc[ci];
                let src = self.src_blk(plane, px, py, 32, 32);
                let dcrr = self.k.idct.idct_dequant_32x32(&ccf[ci], &self.ctx.cquant);
                let s = sse_recon::<1024, 32>(&self.k.rd, &[dc; 1024], &dcrr, src, self.cfg.bd);
                dc_cost[ci] = rd_cost_i64(
                    s,
                    mlam,
                    self.chroma_bits(&ccf[ci][..], &SCAN_32X32, 32, plane, px, py),
                );
                let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
                let realise = |a: i32| {
                    let mut cpr = self.sbuf_i1024();
                    self.k.intrapred
                        .cfl_pred(&mut cpr[..], &ac[..1024], dc, a, self.cfg.bd);
                    let mut resid = self.sbuf_i1024();
                    self.k.rd.residual_pred_blk(&mut resid[..], &cpr[..], src);
                    let (mut q, qt) = self.k.dct.dct32x32_t(&resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q,
                        &qt,
                        dcq,
                        acq,
                        &SCAN_32X32,
                        lam,
                        32,
                        32,
                        plane,
                        px,
                        py,
                    );
                    let rr = self.k.idct.idct_dequant_32x32(&q, &self.ctx.cquant);
                    let dist = sse_recon::<1024, 32>(&self.k.rd, &cpr, &rr, src, self.cfg.bd);
                    let bits = self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py);
                    let body = rd_cost_i64(dist, mlam, bits);
                    let pair = if ci == 0 { [a, other] } else { [other, a] };
                    let cost = cfl_alpha_walk_cost(
                        dist,
                        mlam,
                        bits,
                        self.uv_mode_bits(y_mode, CFL_PRED, Some(pair)),
                    );
                    (cost, (q, cpr, body))
                };
                let (c0, p0) = realise(a_e[ci]);
                let (a, (q, cpr, body)) =
                    cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                cfl_a[ci] = a;
                *cfl_ccf[ci] = q;
                *cfl_pred[ci] = *cpr;
                cfl_cost[ci] = body;
            }
        }
        // CfL signaling costs extra (sign + per-plane alpha); only use it when
        // it beats plain DC on both planes' summed cost by that overhead.
        let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
        // Let the RD comparison decide DC-vs-CfL across the whole quality range;
        // the old `acq > 300` gate suppressed CfL exactly where it helps most
        // (high quality). block8/block16 already dropped it — this path was the
        // last one still gated, which mattered most at 4:4:4 where chroma is
        // full resolution and 32x32 blocks are common.
        let use_cfl = (cfl_a[0] != 0 || cfl_a[1] != 0)
            && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                < dc_cost[0] + dc_cost[1] + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None));
        #[allow(unused_mut)] // cfl_opt mutated in 'sv block when SMOOTH_V wins
        #[allow(clippy::type_complexity)]
        let (cf_use, pred_dc, mut cfl_opt): (
            &[SBuf<[i32; 1024]>; 2],
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
        let mut cf_use_owned: [SBuf<[i32; 1024]>; 2];
        let mut sv_preds32 = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut uv_pal32: Option<LossyUvPalette> = None;
        let mut uv_delta32 = 0i32;
        let (final_cf, chosen_uv_32) = 'sv: {
            if !self.cfg.speed.full_chroma_rdo() {
                break 'sv (cf_use, DC_PRED);
            }
            let dcq2 = self.ctx.cquant.dc_q() as f32;
            let acq2 = self.ctx.cquant.ac_q() as f32;
            let lam2 = trellis_lambda();
            let mlam = self.mlam_c() * (self.emit_mlam(x8 * 8, y8 * 8, 32) / self.mlam());
            // R-D of the current winner (DC or CfL), residual already in `cf_use`.
            let mut cur_total = 0f32;
            cur_total += rate_cost(
                mlam,
                if use_cfl {
                    self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a))
                } else {
                    self.uv_mode_bits(y_mode, DC_PRED, None)
                },
            );
            for ci in 0..2 {
                let plane = ci + 1;
                let rr = self.k.idct.idct_dequant_32x32(&cf_use[ci], &self.ctx.cquant);
                let cur_pred = if use_cfl {
                    *cfl_pred[ci]
                } else {
                    [pred_dc[ci]; 1024]
                };
                let sse = sse_recon::<1024, 32>(&self.k.rd, &cur_pred, &rr, self.src_blk(plane, px, py, 32, 32), self.cfg.bd);
                cur_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&cf_use[ci][..], &SCAN_32X32, 32, plane, px, py),
                );
            }

            let mut best_total = cur_total;
            let mut best_mode = DC_PRED;
            let mut best_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
            let mut best_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
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
            let directional_top = self.rank_chroma_modes::<1024>(candidates, px, py, px, py, 32, 32, have_tr, have_bl);
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
                let mut cand_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
                let mut cand_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
                // V/H also emit a chroma angle_delta symbol (~3 bits); transform stays
                // DCT_DCT here (spec forces it at Tx_Size_Sqr >= TX_32X32).
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
                        32,
                        32,
                        have_tr,
                        have_bl,
                        self.cfg.w,
                        self.cfg.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci][..],
                        self.cfg.bd,
                    );
                    let mut resid = self.sbuf_i1024();
                    self.k.rd.residual_pred(
                        &mut resid[..],
                        &cand_pred[ci][..],
                        &self.cfg.src[plane],
                        self.cfg.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    // Forced DCT_DCT at 32x32 (spec), regardless of uv_mode.
                    let (mut q, qt) = self.k.dct.dct32x32_t(&resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(
                        &mut q,
                        &qt,
                        dcq2,
                        acq2,
                        &SCAN_32X32,
                        lam2,
                        32,
                        32,
                        plane,
                        px,
                        py,
                    );
                    self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    *cand_ccf[ci] = q;
                    let rr = self.k.idct.idct_dequant_32x32(&q, &self.ctx.cquant);
                    let sse = sse_recon::<1024, 32>(&self.k.rd, &cand_pred[ci], &rr, self.src_blk(plane, px, py, 32, 32), self.cfg.bd);
                    cand_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py),
                    );
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    best_mode = cand;
                    best_ccf = cand_ccf;
                    best_pred = cand_pred;
                }
            }
            // Chroma angle-delta refinement (`uv_ad`, zone-2; see code_block16).
            if crate::tuning::get().uv_ad
                && crate::tuning::get().uv_ad32
                && self.cfg.speed.try_angle_deltas_av1(32, self.cfg.base_q_idx)
                && (V_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            {
                let cand = best_mode;
                let ad = &self.dcdf().angle_delta[cand - V_PRED];
                let base_sig = self.uv_mode_bits(y_mode, cand, None) - cdf_cost(ad, 3);
                for &d in uv_zone2_deltas(cand) {
                    let mut cand_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
                    let mut cand_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
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
                            32,
                            32,
                            false,
                            false,
                            self.cfg.w,
                            self.cfg.h,
                            self.chroma_filter_type(px, py),
                            &mut cand_pred[ci][..],
                            self.cfg.bd,
                        );
                        let mut resid = self.sbuf_i1024();
                        self.k.rd.residual_pred(&mut resid[..], &cand_pred[ci][..], &self.cfg.src[plane], self.cfg.w, px, py, 32, 32);
                        let (mut q, qt) = self.k.dct.dct32x32_t(&resid, &self.ctx.cquant);
                        self.chroma_rect_trellis(&mut q, &qt, dcq2, acq2, &SCAN_32X32, lam2, 32, 32, plane, px, py);
                        self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                        let rr = self.k.idct.idct_dequant_32x32(&q, &self.ctx.cquant);
                        let sse = sse_recon::<1024, 32>(&self.k.rd, &cand_pred[ci], &rr, self.src_blk(plane, px, py, 32, 32), self.cfg.bd);
                        *cand_ccf[ci] = q;
                        cand_total += rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py));
                    }
                    if cand_total < best_total {
                        best_total = cand_total;
                        uv_delta32 = d;
                        best_ccf = cand_ccf;
                        best_pred = cand_pred;
                    }
                }
            }
            if best_mode != DC_PRED {
                cfl_opt = None; // a non-DC chroma mode overrides CfL if it wins
                cf_use_owned = best_ccf;
                sv_preds32 = best_pred;
                break 'sv (&cf_use_owned, best_mode);
            }
            // UV palette candidates (4:4:4): exact, else lossy k-means
            // clusterings priced with real chroma SSE (see the 16x16 twin).
            if self.try_palette() {
                let exact = exact_uv_palette(&self.cfg.src[1], &self.cfg.src[2], self.cfg.w, px, py, 32, 32);
                let pcands: Vec<LossyUvPalette> = if let Some(up) = exact {
                    vec![up]
                } else {
                    // Lossy candidates re-enabled 2026-07-23 WITH residual
                    // coding (below): the palette is now a PREDICTOR — the
                    // coded residual repairs the chroma flattening that made
                    // zero-residual lossy palettes SS2-fatal in both prior
                    // attempts. This is aom's mechanism (490 UV palettes on
                    // kodak20 q93, all carrying coefficients).
                    let uv_hist = uv_pair_histogram(&self.cfg.src[1], &self.cfg.src[2], self.cfg.w, px, py, 32, 32);
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
                                32,
                                32,
                                k,
                                top,
                                &uv_map_opts,
                            )
                        })
                        .collect()
                };
                #[allow(clippy::type_complexity)]
                let mut win: Option<(
                    f32,
                    LossyUvPalette,
                    [SBuf<[i32; 1024]>; 2],
                    [SBuf<[i32; 1024]>; 2],
                )> = None;
                for up in pcands {
                    // Residual-over-palette (see the 16x16 twin).
                    let (dcq2, acq2) = (self.ctx.cquant.dc_q() as f32, self.ctx.cquant.ac_q() as f32);
                    let mut pal_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
                    let [pred_u, pred_v] = &mut pal_pred;
                    palette_uv_pred(&mut pred_u[..], &mut pred_v[..], &up.map, &up.u, &up.v);
                    let mut bits = self.uv_mode_bits(y_mode, DC_PRED, None)
                        + self.palette_uv_rate_bits(px, py, palette.is_some(), &up);
                    let mut sse = 0i64;
                    let mut pal_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
                    let mut resids = [self.sbuf_i1024(), self.sbuf_i1024()];
                    for ci in 0..2 {
                        self.k.rd.residual_pred(
                            &mut resids[ci][..],
                            &pal_pred[ci][..],
                            &self.cfg.src[ci + 1],
                            self.cfg.w,
                            px,
                            py,
                            32,
                            32,
                        );
                    }
                    let raw_sse = self.uv_pal_gate_dist(&resids[0][..], &resids[1][..], 32, 32);
                    let gate_ref = win.as_ref().map_or(best_total, |w| w.0.min(best_total));
                    if self.uv_palette_gate(raw_sse, bits, mlam, gate_ref) {
                        continue;
                    }
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let resid = &resids[ci];
                        let (mut q, qt) = self.k.dct.dct32x32_t(resid, &self.ctx.cquant);
                        trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_32X32, trellis_lambda());
                        let rr = self.k.idct.idct_dequant_32x32(&q, &self.ctx.cquant);
                        sse += self.k.rd.sse_recon(&pal_pred[ci][..], &rr, self.src_blk(plane, px, py, 32, 32), self.cfg.bd);
                        *pal_ccf[ci] = q;
                        bits += self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py);
                    }
                    let cand_total = rd_cost_i64(sse, mlam, bits);
                    if cand_total < best_total && win.as_ref().is_none_or(|w| cand_total < w.0) {
                        win = Some((cand_total, up, pal_ccf, pal_pred));
                    }
                }
                if let Some((_, up, pal_ccf, pal_pred)) = win {
                    cfl_opt = None;
                    cf_use_owned = pal_ccf;
                    sv_preds32 = pal_pred;
                    uv_pal32 = Some(up);
                    break 'sv (&cf_use_owned, DC_PRED);
                }
            }
            (cf_use, DC_PRED)
        };
        // The winner's coefficients live in one of several candidate buffers;
        // hand the emitter its own copy, and the per-pixel prediction that
        // goes with it (directional / palette, else CfL).
        let mut coeffs = [self.sbuf_i1024(), self.sbuf_i1024()];
        for ci in 0..2 {
            coeffs[ci].copy_from_slice(&final_cf[ci][..]);
        }
        let pred = if chosen_uv_32 != DC_PRED || uv_pal32.is_some() {
            sv_preds32
        } else {
            cfl_pred
        };
        UvPick {
            coeffs,
            pred,
            resid: None,
            dc: pred_dc,
            mode: chosen_uv_32,
            cfl: cfl_opt,
            delta: uv_delta32,
            palette: uv_pal32,
        }
    }

    /// Emit a 32x32 block (444) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block32_444(&mut self, x8: usize, y8: usize, l: &Luma32<'_>, uv: UvPick<SBuf<[i32; 1024]>, SBuf<[i32; 1024]>>) {
        let (palette, luma_zero) = (l.palette, l.luma_zero);
        let UvPick {
            coeffs: final_cf,
            pred: uv_pred,
            dc: pred_dc,
            mode: chosen_uv_32,
            cfl: cfl_opt,
            delta: uv_delta32,
            palette: uv_pal32,
            ..
        } = uv;
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let block_skip = palette.is_none()
            && uv_pal32.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&final_cf[0][..])
            && self.k.rd.all_zero_i32(&final_cf[1][..]);
        let uv_delta_sig = if chosen_uv_32 != DC_PRED { uv_delta32 } else { 0 };
        self.code_header_luma32(
            x8,
            y8,
            l,
            block_skip,
            chosen_uv_32,
            cfl_opt,
            uv_delta_sig,
            uv_pal32.as_ref(),
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_32(plane, bx4, by4, true);
                let ds = self.dc_sign_ctx_32(plane, bx4, by4);
                encode_tx32_coeffs_adapt(&mut self.ent.enc, &mut self.ent.cdfs, &final_cf[ci], true, sk, ds)
            };
            self.ctx.a_coef[plane][bx4..bx4 + 8].fill(cres);
            self.ctx.l_coef[plane][by4..by4 + 8].fill(cres);
            let crr = if block_skip {
                [0i32; 1024]
            } else {
                self.k.idct.idct_dequant_32x32(&final_cf[ci], &self.ctx.cquant)
            };
            let max = (1 << self.cfg.bd) - 1;
            for (ry, rrow) in crr.as_chunks::<32>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.w + px..];
                if chosen_uv_32 != DC_PRED || uv_pal32.is_some() || cfl_opt.is_some() {
                    recon_add_pred(&mut drow[..32], &uv_pred[ci][ry * 32..], rrow, max);
                } else {
                    recon_add_dc(&mut drow[..32], pred_dc[ci], rrow, max);
                }
            }
        }
    }

    /// Chroma search of a 32x32 block (420) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block32_420`.
    fn pick_uv32_420(&self, x8: usize, y8: usize, l: &Luma32<'_>) -> UvPick<SBuf<[i32; 256]>, SBuf<[i32; 256]>> {
        let Luma32 {
            cf: lcf,
            pred: lpred,
            y_mode,
            have_tr,
            have_bl,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let (cx, cy) = (px / 2, py / 2);
        let (dcq, acq, lam) = (
            self.ctx.cquant.dc_q() as f32,
            self.ctx.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        let maxval = (1 << self.cfg.bd) - 1;
        // DC path.
        let mut ccf_dc = [self.sbuf_i256(), self.sbuf_i256()];
        let mut dc_preds = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = self.k.intrapred.dc_pred_16x16(&self.ctx.recon[plane], self.cfg.cw, cx, cy, self.cfg.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = self.sbuf_i256();
            self.k.rd.residual_dc(
                &mut resid[..],
                &self.cfg.src[plane],
                self.cfg.cw,
                cx,
                cy,
                16,
                16,
                dc,
            );
            let (q, qt) = self.k.dct.dct16x16_t(&resid, &self.ctx.cquant);
            *ccf_dc[ci] = q;
            self.chroma_rect_trellis(
                &mut ccf_dc[ci][..],
                &qt,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                16,
                plane,
                cx,
                cy,
            );
            self.k.rd.preserve_dc(&mut ccf_dc[ci][0], &resid[..]);
        }
        // SMOOTH_V chroma derives ADST_DCT (a 2D tx -> default scan and coef
        // contexts identical to DCT_DCT; only the transform differs). Forward with
        // adstdct16x16_t and reconstruct with iadstdct_dequant_16x16 to match the
        // decoder's derived chroma txtp. Offered at every quality; the Lagrangian
        // R-D decision below selects it only when it truly wins.
        // DC baseline R-D (libaom-style: SSE + mlam*coeff_bits over U+V).
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 32);
        let mut rr_dc = [self.sbuf_i256(), self.sbuf_i256()];
        let mut dc_total = 0f32;
        for ci in 0..2 {
            let plane = ci + 1;
            *rr_dc[ci] = self.k.idct.idct_dequant_16x16(&ccf_dc[ci], &self.ctx.cquant);
            let dc = dc_preds[ci];
            let sse = sse_recon::<256, 16>(&self.k.rd, &[dc; 256], &rr_dc[ci], self.src_blk(plane, cx, cy, 16, 16), self.cfg.bd);
            dc_total += rd_cost_i64(
                sse,
                mlam,
                self.chroma_bits(&ccf_dc[ci][..], &SCAN_16X16, 16, plane, cx, cy),
            );
        }
        // Directional / smooth chroma modes (PAETH/SMOOTH/SMOOTH_V/SMOOTH_H), each
        // with its decoder-derived chroma tx. Winner must beat DC on the R-D metric.
        let mut best_total = dc_total;
        let mut chosen_uv = DC_PRED;
        let mut best_ccf = ccf_dc;
        let mut best_rr = rr_dc;
        let mut sv_preds = [self.sbuf_i256(), self.sbuf_i256()];
        let mut use_cfl = false;
        let mut cfl_alpha = [0i32; 2];
        if !self.cfg.is_mono() {
            let mut luma_rec = self.sbuf_u1024();
            if let Some(rec) = l.split_rec {
                luma_rec.copy_from_slice(&rec[..]);
            } else {
                let lrr = self.k.idct.idct_dequant_32x32(lcf, &self.ctx.quant);
                recon_add_pred(&mut luma_rec[..], lpred, &lrr, maxval);
            }
            let mut ac = self.sbuf_i256();
            self.k.intrapred
                .cfl_ac_sub(&luma_rec[..], 32, 16, 16, true, true, &mut ac[..]);
            let mut cfl_ccf = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_rr = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_px = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_a = [0i32; 2];
            let mut cfl_body = 0f32;
            let steps = crate::tuning::get().cfl_alpha_rd_steps;
            let a_e = [0, 1].map(|ci| {
                self.k.intrapred.cfl_best_alpha(&ac[..], self.src_blk(ci + 1, cx, cy, 16, 16), dc_preds[ci], self.cfg.bd)
            });
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_preds[ci];
                let src = self.src_blk(plane, cx, cy, 16, 16);
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
                        cx,
                        cy,
                    );
                    let rr = self.k.idct.idct_dequant_16x16(&q, &self.ctx.cquant);
                    let dist = sse_recon::<256, 16>(&self.k.rd, &cpr, &rr, src, self.cfg.bd);
                    let bits = self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy);
                    let body = rd_cost_i64(dist, mlam, bits);
                    let pair = if ci == 0 { [a, other] } else { [other, a] };
                    let cost = cfl_alpha_walk_cost(
                        dist,
                        mlam,
                        bits,
                        self.uv_mode_bits(y_mode, CFL_PRED, Some(pair)),
                    );
                    (cost, (q, rr, cpr, body))
                };
                let (c0, p0) = realise(a_e[ci]);
                let (a, (q, rr, cpr, body)) =
                    cfl_alpha_rd_refine(a_e[ci], c0, p0, steps, realise);
                cfl_a[ci] = a;
                *cfl_ccf[ci] = q;
                *cfl_rr[ci] = rr;
                *cfl_px[ci] = *cpr;
                cfl_body += body;
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
                sv_preds = cfl_px;
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
        let directional_top = self.rank_chroma_modes::<256>(candidates, px, py, cx, cy, 16, 16, have_tr, have_bl);
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
            let mut cand_ccf = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cand_rr = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cand_pred = [self.sbuf_i256(), self.sbuf_i256()];
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
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.cfg.cw,
                    self.cfg.h / 2,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci][..],
                    self.cfg.bd,
                );
                let mut resid = self.sbuf_i256();
                self.k.rd.residual_pred(
                    &mut resid[..],
                    &cand_pred[ci][..],
                    &self.cfg.src[plane],
                    self.cfg.cw,
                    cx,
                    cy,
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
                    cx,
                    cy,
                );
                self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                *cand_ccf[ci] = q;
                *cand_rr[ci] = inv_chroma_16x16(&self.k.idct, tx, &q, &self.ctx.cquant);
                let sse = sse_recon::<256, 16>(&self.k.rd, &cand_pred[ci], &cand_rr[ci], self.src_blk(plane, cx, cy, 16, 16), self.cfg.bd);
                cand_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy),
                );
            }
            if cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                use_cfl = false;
                best_ccf = cand_ccf;
                best_rr = cand_rr;
                sv_preds = cand_pred;
            }
        }
        // Chroma angle-delta refinement (`uv_ad`, zone-2 deltas; see code_block16).
        let mut uv_delta = 0i32;
        if !use_cfl
            && crate::tuning::get().uv_ad
            && crate::tuning::get().uv_ad32
            && self.cfg.speed.try_angle_deltas_av1(32, self.cfg.base_q_idx)
            && (V_PRED..=VERT_LEFT_PRED).contains(&chosen_uv)
        {
            let cand = chosen_uv;
            let tx = chroma_tx_for_mode(cand);
            let ad = &self.dcdf().angle_delta[cand - V_PRED];
            let base_sig = self.uv_mode_bits(y_mode, cand, None) - cdf_cost(ad, 3);
            for &d in uv_zone2_deltas(cand) {
                let mut cand_ccf = [self.sbuf_i256(), self.sbuf_i256()];
                let mut cand_rr = [self.sbuf_i256(), self.sbuf_i256()];
                let mut cand_pred = [self.sbuf_i256(), self.sbuf_i256()];
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
                        16,
                        16,
                        false,
                        false,
                        self.cfg.cw,
                        self.cfg.h,
                        self.chroma_filter_type(px, py),
                        &mut cand_pred[ci][..],
                        self.cfg.bd,
                    );
                    let mut resid = self.sbuf_i256();
                    self.k.rd.residual_pred(&mut resid[..], &cand_pred[ci][..], &self.cfg.src[plane], self.cfg.cw, cx, cy, 16, 16);
                    let (mut q, qt) = fwd_chroma_16x16(&self.k.dct, tx, &resid, &self.ctx.cquant);
                    self.chroma_rect_trellis(&mut q, &qt, dcq, acq, &SCAN_16X16, lam, 16, 16, plane, cx, cy);
                    self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                    *cand_ccf[ci] = q;
                    *cand_rr[ci] = inv_chroma_16x16(&self.k.idct, tx, &q, &self.ctx.cquant);
                    let sse = sse_recon::<256, 16>(&self.k.rd, &cand_pred[ci], &cand_rr[ci], self.src_blk(plane, cx, cy, 16, 16), self.cfg.bd);
                    cand_total += rd_cost_i64(sse, mlam, self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy));
                }
                if cand_total < best_total {
                    best_total = cand_total;
                    uv_delta = d;
                    best_ccf = cand_ccf;
                    best_rr = cand_rr;
                    sv_preds = cand_pred;
                }
            }
        }
        UvPick {
            coeffs: best_ccf,
            pred: sv_preds,
            resid: Some(best_rr),
            dc: dc_preds,
            mode: chosen_uv,
            cfl: use_cfl.then_some(cfl_alpha),
            delta: uv_delta,
            palette: None,
        }
    }

    /// Emit a 32x32 block (420) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block32_420(&mut self, x8: usize, y8: usize, l: &Luma32<'_>, uv: UvPick<SBuf<[i32; 256]>, SBuf<[i32; 256]>>) {
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
        let use_sv = chosen_uv != DC_PRED || use_cfl;
        let block_skip = palette.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&ccf[0][..])
            && self.k.rd.all_zero_i32(&ccf[1][..]);
        self.code_header_luma32(
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
                let sk = self.skip_ctx_16(plane, bx4c, by4c, true);
                let ds = self.dc_sign_ctx_16(plane, bx4c, by4c);
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
            self.ctx.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.ctx.l_coef[plane][by4c..by4c + 4].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 256]
            } else {
                *rr_cache[ci]
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(cy + ry) * self.cfg.cw + cx..];
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
    /// Chroma search of a 32x32 block (422) for the luma winner `l`.
    /// Read-only: returns the winner for `emit_block32_422`.
    fn pick_uv32_422(&self, x8: usize, y8: usize, l: &Luma32<'_>) -> UvPick<SBuf<[i32; 512]>, SBuf<[i32; 512]>> {
        let Luma32 {
            cf: lcf,
            pred: lpred,
            y_mode,
            ..
        } = *l;
        let (px, py) = (x8 * 8, y8 * 8);
        let cx = px / 2;
        // Chroma winner (popped here, pushed before the emit below; exactly one
        // per code_block32 call — this helper is its only chroma path).
        let maxv = (1 << self.cfg.bd) - 1;
        let (dcq, acq, lam) = (
            self.ctx.cquant.dc_q() as f32,
            self.ctx.cquant.ac_q() as f32,
            trellis_lambda(),
        );
        let mlam = self.emit_mlam_c(x8 * 8, y8 * 8, 32);
        let mut ccf = [self.sbuf_i512(), self.sbuf_i512()];
        let mut cpred = [0i32; 2];
        let mut cpred_px = [self.sbuf_i512(), self.sbuf_i512()];
        let src_planes = [1, 2].map(|plane| self.src_blk(plane, cx, py, 16, 32));
        let mut dc_ccf = [self.sbuf_i512(), self.sbuf_i512()];
        let mut dc_sse = [0i64; 2];
        let mut dc_bits = [0f32; 2];
        // DC option.
        for ci in 0..2 {
            let plane = ci + 1;
            let pred = self.k.intrapred.dc_pred_16x32(&self.ctx.recon[plane], self.cfg.cw, cx, py, self.cfg.bd as i32);
            cpred[ci] = pred;
            let src = src_planes[ci];
            let mut resid = self.sbuf_i512();
            self.k.rd.residual_dc_blk(&mut resid[..], src, pred);
            let (mut q, qt) = self.k.dct.dct16x32_t(&resid, &self.ctx.cquant);
            self.chroma_rect_trellis(
                &mut q,
                &qt,
                dcq,
                acq,
                &SCAN_16X32,
                lam,
                16,
                32,
                ci + 1,
                cx,
                py,
            );
            let rr = self.k.idct.idct_dequant_16x32(&q, &self.ctx.cquant);
            *dc_ccf[ci] = q;
            dc_sse[ci] =
                self.k.rd.sse_recon(&[pred; 512], &rr, src, self.cfg.bd);
            dc_bits[ci] = self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, plane, cx, py);
        }

        // CfL: predict the 16x32 U/V from the horizontally-subsampled 32x32
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0). 32x32 luma is
        // always DCT_DCT here, so the AC reference inverts with idct_dequant_32x32.
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        let mut luma_rec = self.sbuf_u1024();
        if let Some(rec) = l.split_rec {
            luma_rec.copy_from_slice(&rec[..]);
        } else {
            let lrr_cfl = self.k.idct.idct_dequant_32x32(lcf, &self.ctx.quant);
            recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, maxv);
        }
        let mut ac = self.sbuf_i512();
        self.k.intrapred
            .cfl_ac_sub(&luma_rec[..], 32, 16, 32, true, false, &mut ac[..]);
        let mut cfl_ccf = [self.sbuf_i512(), self.sbuf_i512()];
        let mut cfl_a = [0i32; 2];
        let mut cfl_sse = [0i64; 2];
        let mut cfl_bits = [0f32; 2];
        let steps = crate::tuning::get().cfl_alpha_rd_steps;
        let a_e = [0, 1].map(|ci| {
            self.k.intrapred.cfl_best_alpha(&ac[..], src_planes[ci], cpred[ci], self.cfg.bd)
        });
        for ci in 0..2 {
            let dc = cpred[ci];
            let src = src_planes[ci];
            let other = if ci == 0 { a_e[1] } else { cfl_a[0] };
            let realise = |a: i32| {
                let mut cpr = self.sbuf_i512();
                self.k.intrapred.cfl_pred(&mut cpr[..], &ac[..512], dc, a, self.cfg.bd);
                let mut resid = self.sbuf_i512();
                self.k.rd.residual_pred_blk(&mut resid[..], &cpr[..], src);
                let (mut q, qt) = self.k.dct.dct16x32_t(&resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q,
                    &qt,
                    dcq,
                    acq,
                    &SCAN_16X32,
                    lam,
                    16,
                    32,
                    ci + 1,
                    cx,
                    py,
                );
                let rr = self.k.idct.idct_dequant_16x32(&q, &self.ctx.cquant);
                let sse = self.k.rd.sse_recon(&cpr[..], &rr, src, self.cfg.bd);
                let bits = self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, ci + 1, cx, py);
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
            *cpred_px[ci] = *cpr;
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
            for ci in 0..2 {
                *ccf[ci] = *cfl_ccf[ci];
            }
        }
        // Directional / smooth chroma on the 16x32 block. The AV1 spec forces
        // DCT_DCT for any intra block whose square transform size is >= TX_32X32,
        // and 16x32 (Tx_Size_Sqr = TX_32X32) hits that rule -- so the residual is
        // always the plain 16x32 DCT and only the prediction changes (PAETH/SMOOTH/
        // SMOOTH_V/SMOOTH_H). Searched against the DC/CfL winner on the R-D metric.
        let mut chosen_uv = if use_cfl { CFL_PRED } else { DC_PRED };
        let mut best_total = 0f32;
        best_total += rate_cost(
            mlam,
            if use_cfl {
                self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_alpha_uv))
            } else {
                self.uv_mode_bits(y_mode, DC_PRED, None)
            },
        );
        for ci in 0..2 {
            let cur_ccf = if use_cfl { *ccf[ci] } else { *dc_ccf[ci] };
            let rr = self.k.idct.idct_dequant_16x32(&cur_ccf, &self.ctx.cquant);
            let cur_pred = if use_cfl {
                *cpred_px[ci]
            } else {
                [cpred[ci]; 512]
            };
            let sse = self.k.rd.sse_recon(&cur_pred, &rr, src_planes[ci], self.cfg.bd);
            best_total += rd_cost_i64(
                sse,
                mlam,
                self.chroma_rect_bits(&cur_ccf[..], &SCAN_16X32, 16, 32, ci + 1, cx, py),
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
        let directional_top = self.rank_chroma_modes::<512>(candidates, px, py, cx, py, 16, 32, false, false);
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
            let mut cand_ccf = [self.sbuf_i512(), self.sbuf_i512()];
            let mut cand_pred = [self.sbuf_i512(), self.sbuf_i512()];
            // V/H also emit a chroma angle_delta symbol (~3 bits); transform stays
            // DCT_DCT here (spec forces it at Tx_Size_Sqr >= TX_32X32).
            let sig_bits = self.uv_mode_bits(y_mode, cand, None);
            let mut cand_total = rate_cost(mlam, sig_bits);
            for ci in 0..2 {
                let plane = ci + 1;
                self.k.intrapred.predict_nd(
                    cand,
                    &self.ctx.recon[plane],
                    self.cfg.cw,
                    cx,
                    py,
                    16,
                    32,
                    false,
                    false,
                    self.cfg.cw,
                    self.cfg.h,
                    self.chroma_filter_type(px, py),
                    &mut cand_pred[ci][..],
                    self.cfg.bd,
                );
                let src = src_planes[ci];
                let mut resid = self.sbuf_i512();
                self.k.rd.residual_pred_blk(&mut resid[..], &cand_pred[ci][..], src);
                // Forced DCT_DCT at 16x32 (spec), regardless of uv_mode.
                let (mut q, qt) = self.k.dct.dct16x32_t(&resid, &self.ctx.cquant);
                self.chroma_rect_trellis(
                    &mut q,
                    &qt,
                    dcq,
                    acq,
                    &SCAN_16X32,
                    lam,
                    16,
                    32,
                    ci + 1,
                    cx,
                    py,
                );
                self.k.rd.preserve_dc(&mut q[0], &resid[..]);
                *cand_ccf[ci] = q;
                let rr = self.k.idct.idct_dequant_16x32(&q, &self.ctx.cquant);
                let sse = self.k.rd.sse_recon(&cand_pred[ci][..], &rr, src, self.cfg.bd);
                cand_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, plane, cx, py),
                );
            }
            if cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                use_cfl = false;
                for ci in 0..2 {
                    *ccf[ci] = *cand_ccf[ci];
                }
                for ci in 0..2 {
                    *cpred_px[ci] = *cand_pred[ci];
                }
            }
        }
        if chosen_uv == DC_PRED {
            for ci in 0..2 {
                *ccf[ci] = *dc_ccf[ci];
                *cpred_px[ci] = [cpred[ci]; 512];
            }
        }
        UvPick {
            coeffs: ccf,
            pred: cpred_px,
            resid: None,
            dc: cpred,
            mode: chosen_uv,
            cfl: use_cfl.then_some(cfl_alpha_uv),
            delta: 0,
            palette: None,
        }
    }

    /// Emit a 32x32 block (422) from its luma and chroma winners: the block
    /// header, luma, then the chroma coefficients and reconstruction.
    fn emit_block32_422(&mut self, x8: usize, y8: usize, l: &Luma32<'_>, uv: UvPick<SBuf<[i32; 512]>, SBuf<[i32; 512]>>) {
        let (palette, luma_zero) = (l.palette, l.luma_zero);
        let UvPick {
            coeffs: ccf,
            pred: cpred_px,
            mode: chosen_uv,
            cfl,
            ..
        } = uv;
        let py = y8 * 8;
        let cx = x8 * 4;
        let (bx4c, by4c) = (cx / 4, py / 4);
        let maxv = (1 << self.cfg.bd) - 1;
        let block_skip = palette.is_none()
            && luma_zero
            && self.k.rd.all_zero_i32(&ccf[0][..])
            && self.k.rd.all_zero_i32(&ccf[1][..]);
        self.code_header_luma32(
            x8,
            y8,
            l,
            block_skip,
            chosen_uv,
            cfl,
            0,
            None,
        );
        for ci in 0..2 {
            let plane = ci + 1;
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x32_422(plane, bx4c, by4c);
                let ds = self.dc_sign_ctx_16x32_422(plane, bx4c, by4c);
                encode_16x32_chroma_coeffs(&mut self.ent.enc, &mut self.ent.cdfs, &ccf[ci], sk, ds)
            };
            self.ctx.a_coef[plane][bx4c..bx4c + 4].fill(res_ctx);
            self.ctx.l_coef[plane][by4c..by4c + 8].fill(res_ctx);
            let rr = if block_skip {
                [0i32; 512]
            } else {
                self.k.idct.idct_dequant_16x32(&ccf[ci], &self.ctx.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.ctx.recon[plane][(py + ry) * self.cfg.cw + cx..];
                recon_add_pred(drow, &cpred_px[ci][ry * 16..], rrow, maxv);
            }
        }
    }
}
