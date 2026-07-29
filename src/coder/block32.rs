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

const RECT32_ENABLED: bool = false;

const AB32_ENABLED: bool = false;

const fn ab32_bias() -> f32 {
    1.0
}

const fn rect32_bias() -> f32 {
    1.1
}

const fn full_partition_proxy32() -> bool {
    true
}

struct Luma32BeamCandidate {
    luma_cost: f32,
    mode: usize,
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
        let suma: i32 = sum_coef_sign(&a[bx4c..bx4c + 4]);
        let suml: i32 = sum_coef_sign(&l[by4c..by4c + 8]);
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
            let a = &self.a_coef[plane];
            let l = &self.l_coef[plane];
            let ca = a[bx4..bx4 + 8].iter().any(|&x| x != 0x40) as usize;
            let cl = l[by4..by4 + 8].iter().any(|&x| x != 0x40) as usize;
            7 + ca + cl
        }
    }

    fn choose_rect8(&self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) -> Part16 {
        if !crate::tuning::get().rect8_enabled || self.speed != Speed::Slow || !self.ss420 {
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
        let rd_v = if self.ss422 {
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
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let (lam, mlam) = (trellis_lambda() * prdo, self.mlam() * prdo);
        let (lw, lh) = if vert { (4usize, 8) } else { (8, 4) };
        let scan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let _yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let mut best = f32::INFINITY;
        let mut pred = [0i32; 32];
        for m in 0..13usize {
            if m == DC_PRED {
                let dc = if vert {
                    self.intrapred.dc_pred_4x8(&self.recon[0], self.w, px, py, self.bd as i32)
                } else {
                    self.intrapred.dc_pred_8x4(&self.recon[0], self.w, px, py, self.bd as i32)
                };
                pred[..lw * lh].fill(dc);
            } else {
                self.intrapred.predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    lw,
                    lh,
                    sub_tr,
                    sub_bl,
                    self.w,
                    self.h,
                    ftype,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 32];
            self.rd.residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, lw, lh);
            let (mut cf, tf) = if vert {
                self.dct.dct4x8_t(&resid, &self.quant)
            } else {
                self.dct.dct8x4_t(&resid, &self.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, lw, lh, px, py);
            let rr = if vert {
                self.idct.idct_dequant_4x8(&cf, &self.quant)
            } else {
                self.idct.idct_dequant_8x4(&cf, &self.quant)
            };
            let distortion = self.luma_partition_distortion(
                px,
                py,
                lw,
                lh,
                self.quant.ac_q() as f32,
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
    /// Decision-side price of one equipped 32-level rect leaf: the emitter's
    /// mode search re-run under the caller's prdo-scaled lambdas, distortion
    /// via the masking-weighted partition metric, epoch-memoized (the two
    /// rect orientations and the 64-level SPLIT recompute share leaves).
    fn rd_cost_rect32_leaf(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        let dc = if vert {
            self.intrapred.dc_pred_16x32(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            self.intrapred.dc_pred_32x16(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        let key = (
            (1u64 << 62)
                | ((px as u64) << 40)
                | ((py as u64) << 20)
                | ((dc as u64) << 1)
                | vert as u64,
            prdo.to_bits(),
        );
        let epoch = self.emit_epoch.get();
        if let Some(&(e, dist, bits)) = self.rect_leaf_cache.borrow().get(&key)
            && e == epoch
        {
            return crate::partition_rd::rd_cost(dist, mlam, bits);
        }
        let (w, h) = if vert { (16usize, 32usize) } else { (32, 16) };
        let scan: &[u32] = if vert { &SCAN_16X32 } else { &SCAN_32X16 };
        let dlam = trellis_lambda() * prdo;
        let (y_mode, pred, resid_scratch, cf) =
            self.rect32_luma_mode_search(px, py, vert, dc, dlam, mlam);
        self.sc().put_i512(resid_scratch);
        let rr = if vert {
            self.idct.idct_dequant_16x32(&cf, &self.quant)
        } else {
            self.idct.idct_dequant_32x16(&cf, &self.quant)
        };
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
        bits += self.luma_rect_bits(&cf[..], scan, w, h, px, py, y_mode, 1);
        self.rect_leaf_cache
            .borrow_mut()
            .insert(key, (epoch, distortion, bits));
        {
            let mut sc = self.sc();
            sc.put_i512(pred);
            sc.put_i512(cf);
        }
        crate::partition_rd::rd_cost(distortion, mlam, bits)
    }

    fn rd_choice_rect32(
        &self,
        x8: usize,
        y8: usize,
        prdo: f32,
        known_none: Option<f32>,
        thr: bool,
        lhb: bool,
    ) -> (Part16, f32) {
        // Monochrome has no whole-32 emitter (`code_block32` dispatches to the
        // 4:4:4 helper, which indexes the absent chroma planes), so 4:0:0 always
        // splits — as it did before the NONE-vs-SPLIT R-D rework.
        if self.mono {
            let (px, py) = (x8 * 8, y8 * 8);
            return (Part16::Split, self.rd_cost_split32(px, py, prdo, thr, lhb));
        }
        let (px, py) = (x8 * 8, y8 * 8);
        // Fixed-partition mode: no R-D at all at this level.
        //
        // The returned cost is a placeholder. In fixed mode EVERY level
        // short-circuits (choose_64, here, and rd_choice_16_inner), so no
        // caller ever compares one of these costs against an alternative --
        // `choose_rect32` and `partition_choice_16` take `.0` and drop it, and
        // `choose_64`'s call site is unreachable because 64 short-circuits
        // first. Skipping the pricing is the entire speed win; if a future
        // caller starts consuming `.1`, this must change with it.
        match crate::tuning::fixed_size(self.speed) {
            0 => {}
            32 => return (Part16::None, 0.0),
            n if n < 32 => return (Part16::Split, 0.0),
            _ => {}
        }
        let coupled_square =
            !self.mono && self.speed == Speed::Slow && joint_luma_uv_proxy_enabled();
        let chroma_split = if self.mono
            || !self.speed.full_partition_rdo()
            || coupled_square
            || bottomup_split32()
        {
            // Bottom-up: the 16-level child totals already include chroma.
            0.0
        } else {
            self.rd_cost_chroma_partition(px, py, 32, Part16::Split, prdo)
        };
        // Small split-favoring bias: the SATD distortion proxy undervalues the
        // detail a single 32x32 loses when it merges four busier 16x16s, so a
        // pure comparison over-merges on textured content (SSIMULACRA2 penalizes
        // it). This is a mild, symmetric thumb (not the old prefilter that
        // skipped the comparison entirely).
        let rd_none = known_none.unwrap_or_else(|| {
            let (htr32, hbl32) = self.leaf_edge_flags(px, py, 32, thr, lhb);
            self.rd_cost_none32(px, py, prdo, htr32, hbl32)
                + if self.speed.full_partition_rdo() && !coupled_square {
                    self.rd_cost_chroma_partition(px, py, 32, Part16::None, prdo)
                } else {
                    0.0
                }
        }) * if self.top_band() && self.ss420 {
            top_none_bias_420(self.aq.base_q)
        } else {
            none32_split_bias()
        } + rate_cost(self.mlam() * prdo, self.part_rate_bl(2, x8, y8, 0));
        // Flat-block SPLIT skip (aom VAR_BASED_PARTITION / SVT depth-refinement
        // analog): when the source detail in this 32x32 is small relative to the
        // AC quantizer, the four-child search cannot recover enough distortion
        // to beat NONE, so skip it outright. Our branch-and-bound only stops
        // once the running child total exceeds the bound; this never enters the
        // child search at all.
        let vbp_on = if self.ss420 {
            self.speed != Speed::Medium || crate::tuning::get().vbp_medium_420
        } else if self.ss422 {
            crate::tuning::get().vbp_422
        } else if !self.mono {
            crate::tuning::get().vbp_444
        } else {
            false
        };
        let vbp_thresh = if self.ss420 {
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
        let rd_ibc = if self.allow_intrabc {
            self.rd_cost_intrabc(px, py, 32, prdo)
                .unwrap_or(f32::INFINITY)
        } else {
            f32::INFINITY
        };
        if rd_ibc < rd_none.min(rd_split) {
            return (Part16::Intrabc, rd_ibc);
        }
        if !self.speed.full_partition_rdo() {
            return if rd_none <= rd_split {
                (Part16::None, rd_none)
            } else {
                (Part16::Split, rd_split)
            };
        }
        let rect32_on = self.speed == Speed::Slow
            && self.speed.full_partition_rdo()
            && !self.mono
            && RECT32_ENABLED;
        if rect32_on {
            let mlam = self.mlam() * prdo;
            let sig = rate_cost(mlam, self.partition_signal_bits());
            let rd_horz = (sig
                + self.rd_cost_rect32_leaf(px, py, false, prdo)
                + self.rd_cost_rect32_leaf(px, py + 16, false, prdo)
                + self.rd_cost_chroma_partition(px, py, 32, Part16::Horz, prdo))
                * rect32_bias();
            let rd_vert = (sig
                + self.rd_cost_rect32_leaf(px, py, true, prdo)
                + self.rd_cost_rect32_leaf(px + 16, py, true, prdo)
                + self.rd_cost_chroma_partition(px, py, 32, Part16::Vert, prdo))
                * rect32_bias();
            let best_rect = rd_horz.min(rd_vert);
            if best_rect < rd_none.min(rd_split) {
                return if rd_horz <= rd_vert {
                    (Part16::Horz, rd_horz)
                } else {
                    (Part16::Vert, rd_vert)
                };
            }
        }
        // A/B T-shapes at the 32 parent: one equipped 32x16/16x32 trailing or
        // leading leaf + two 16x16 squares. Unlike pure HORZ/VERT (closed as
        // net-negative), only ONE half is a rect; the squares keep the full
        // 16-level tooling. The 16x16 children of a T-shape are forced
        // BLOCK_16X16 leaves (no nested partition), so they are priced with
        // `rd_cost_square` (the NONE-16 estimator), NOT the 16-level best
        // total. Chroma for all three pieces is priced by
        // `rd_cost_chroma_partition` at luma_dim 32.
        let ab32_on = self.speed == Speed::Slow
            && self.speed.full_partition_rdo()
            && self.ss420
            && AB32_ENABLED;
        if ab32_on {
            let mlam = self.mlam() * prdo;
            let asym_sig = rate_cost(mlam, self.partition_signal_bits());
            // Children on the same axis as SPLIT's: `rd_cost_square` at Slow
            // is the chroma-coupled joint proxy (the same estimator behind the
            // 16-level totals), so NO separate chroma term for them — only the
            // rect piece's chroma is added (the rect leaf pricer is luma-only).
            let sq = |sx: usize, sy: usize| {
                self.rd_cost_square(px + sx, py + sy, 16, false, false, prdo)
            };
            let rect_chroma = |ox: usize, oy: usize, lw: usize, lh: usize| {
                if coupled_square {
                    let sub_x = (self.ss420 || self.ss422) as usize;
                    let sub_y = self.ss420 as usize;
                    self.chroma_partition_weight_at(px + ox, py + oy, lw, lh)
                        * self.rd_cost_chroma_block(
                            (px + ox) >> sub_x,
                            (py + oy) >> sub_y,
                            lw >> sub_x,
                            lh >> sub_y,
                            prdo,
                        )
                } else {
                    0.0 // uncoupled: the full-partition chroma term is added below
                }
            };
            let full_chroma = |part: Part16| {
                if coupled_square {
                    0.0
                } else {
                    self.rd_cost_chroma_partition(px, py, 32, part, prdo)
                }
            };
            let bias = ab32_bias();
            let cands = [
                (
                    Part16::HorzA,
                    asym_sig
                        + sq(0, 0)
                        + sq(16, 0)
                        + self.rd_cost_rect32_leaf(px, py + 16, false, prdo)
                        + rect_chroma(0, 16, 32, 16)
                        + full_chroma(Part16::HorzA),
                ),
                (
                    Part16::HorzB,
                    asym_sig
                        + self.rd_cost_rect32_leaf(px, py, false, prdo)
                        + rect_chroma(0, 0, 32, 16)
                        + sq(0, 16)
                        + sq(16, 16)
                        + full_chroma(Part16::HorzB),
                ),
                (
                    Part16::VertA,
                    asym_sig
                        + sq(0, 0)
                        + sq(0, 16)
                        + self.rd_cost_rect32_leaf(px + 16, py, true, prdo)
                        + rect_chroma(16, 0, 16, 32)
                        + full_chroma(Part16::VertA),
                ),
                (
                    Part16::VertB,
                    asym_sig
                        + self.rd_cost_rect32_leaf(px, py, true, prdo)
                        + rect_chroma(0, 0, 16, 32)
                        + sq(16, 0)
                        + sq(16, 16)
                        + full_chroma(Part16::VertB),
                ),
            ];
            let (ab_part, ab_cost) = cands
                .into_iter()
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();
            let ab_cost = ab_cost * bias;
            if ab_cost < rd_none.min(rd_split) {
                return (ab_part, ab_cost);
            }
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
            (Part16::None, rd_none)
        } else {
            (Part16::Split, rd_split)
        }
    }

    /// 32x32 shared-partition choice: NONE vs bottom-up SPLIT (32-level
    /// HORZ/VERT candidates were removed — see `rd_choice_rect32`).
    fn choose_rect32(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> Part16 {
        let (px, py) = (x8 * 8, y8 * 8);
        self.rd_choice_rect32(x8, y8, self.perceptual_rd_scale(px, py, 32), None, thr, lhb)
            .0
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
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let dc = self.intrapred.dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
        let mut resid = self.sbuf_i1024();
        self.rd.residual_dc(&mut resid[..], &self.src[0], self.w, px, py, 32, 32, dc);
        let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
        trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
        let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
        let distortion = self.luma_partition_distortion(
            px,
            py,
            32,
            32,
            self.quant.ac_q() as f32,
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
        if bottomup_split32() && !self.mono && self.speed.full_partition_rdo() {
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
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        // This is an empirical proxy-gap allowance, not literal partition
        // entropy. Charge it once; the historical `* 4` double-counted it.
        let mut total = rate_cost(mlam, self.part_rate_bl(2, px / 8, py / 8, 3) * split32_signal_mult());
        for (sx, sy) in [(0usize, 0usize), (16, 0), (0, 16), (16, 16)] {
            let dc = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, px + sx, py + sy, self.bd as i32);
            let mut resid = self.sbuf_i256();
            self.rd.residual_dc(
                &mut resid[..],
                &self.src[0],
                self.w,
                px + sx,
                py + sy,
                16,
                16,
                dc,
            );
            let (mut cf, tf) = self.dct.dct16x16_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_16X16, lam);
            let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
            let distortion = self.luma_partition_distortion(
                px + sx,
                py + sy,
                16,
                16,
                self.quant.ac_q() as f32,
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
    ) -> ([i32; 1024], i64, f32) {
        let mut saved = self.sbuf_u1024();
        for ry in 0..32 {
            saved[ry * 32..ry * 32 + 32]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..32]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        // Progressive sub-transform contexts (external review round 2,
        // finding 3): each TX_16X16 sees the previous quadrant's result.
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 8] = self.a_coef[0][bx4_0..bx4_0 + 8].try_into().unwrap();
        let saved_l: [u8; 8] = self.l_coef[0][by4_0..by4_0 + 8].try_into().unwrap();
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
                let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                *pred = [d; 256];
            } else {
                self.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    16,
                    16,
                    tr,
                    bl,
                    self.w,
                    self.h,
                    block_ftype,
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
                bx,
                by,
                16,
                16,
            );
            let (qbx4, qby4) = (bx / 4, by / 4);
            let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
            let ds = self.dc_sign_ctx_span(0, qbx4, qby4, 4, 4);
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
                ds,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
            sse_sum += sse_recon::<256, 16>(&self.rd, &pred, &rr, &self.src[0], self.w, bx, by, self.bd);
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
            self.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
            self.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                None,
                &pred[..],
                &rr,
                16,
                16,
                self.bd,
            );
            cf4[qi * 256..qi * 256 + 256].copy_from_slice(&cf);
        }
        for ry in 0..32 {
            self.recon[0][(py + ry) * self.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.a_coef[0][bx4_0..bx4_0 + 8].copy_from_slice(&saved_a);
        self.l_coef[0][by4_0..by4_0 + 8].copy_from_slice(&saved_l);
        (*cf4, sse_sum, bits_sum)
    }

    fn code_block32(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 8);
        let (px, py) = (x8 * 8, y8 * 8);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
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
        let mut lpred = self.sbuf_i1024();
        let mut lcf = self.sbuf_i1024();
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_filter_intra = None;
        let mut luma_beam: [Option<Luma32BeamCandidate>; JOINT_LARGE_BEAM] =
            std::array::from_fn(|_| None);
        let mut ltf = self.sbuf_f1024(); // winner transform coeffs (f32, for winner-only RDOQ)
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
            self.rank_luma_modes::<1024>(
                modes,
                px,
                py,
                32,
                32,
                have_tr,
                have_bl,
                // 32-blocks are rare enough (40-550/frame) that the 444-class
                // finalist budget is ~free here (-1.2% time noise). Swept
                // 2026-07-27: subsampled Slow 3 -> 5 = holdout 422 -0.22
                // (4/5 neg) / 420 -0.03, tuning mid -0.04; 444 identity
                // (budget already 5).
                if self.speed == Speed::Slow {
                    5
                } else {
                    self.luma_mode_budget_eff()
                },
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
            if self.speed.per_candidate_rdoq_av1() {
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
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
            }
            let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
            let sse = sse_recon::<1024, 32>(&self.rd, &pred, &rr, &self.src[0], self.w, px, py, self.bd);
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
        if rl.is_none()
            && self.try_palette()
            && let Some(hist) = block_color_histogram(&self.src[0], self.w, px, py, 32, 32)
        {
            for (palette, pred) in
                self.rank_luma_palette_candidates::<1024>(&hist, px, py, 32, 32, mlam)
            {
                let mut resid = self.sbuf_i1024();
                self.rd.residual_pred(
                    &mut resid[..],
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                );
                let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq_av1() {
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
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                let sse = sse_recon::<1024, 32>(&self.rd, &pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0)
                    + self.mode_bits(px, py, DC_PRED)
                    + self.palette_rate_bits(px, py, &palette);
                let cost = rd_cost_i64(sse, mlam, bits);
                if cost < best_eff {
                    best_eff = cost;
                    best_mode = DC_PRED;
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
                *lpred = *candidate.pred;
                *lcf = *candidate.cf;
                *ltf = *candidate.tf;
                best_dct_sse = candidate.sse;
                best_filter_intra = None;
                best_palette32 = candidate.palette;
            }
        }
        if rl.is_none() && self.speed == Speed::Slow {
            let bsize = av1_block_size_index(32, 32);
            for &filter_mode in self
                .rank_filter_intra_modes::<1024>(
                    px,
                    py,
                    32,
                    32,
                    self.speed.filter_intra_refine_budget(),
                )
                .iter()
            {
                let mut pred = self.sbuf_i1024();
                self.intrapred.filter_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    32,
                    32,
                    &mut pred[..],
                    self.bd,
                );
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
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                let sse = sse_recon::<1024, 32>(&self.rd, &pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, DC_PRED, 0);
                let syntax_bits = self.mode_bits(px, py, DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if rl.is_some()
                    || raw_sse_guard_choice(
                        "filter32",
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
                    *lpred = *pred;
                    *lcf = cf;
                    *ltf = tf;
                    best_dct_sse = sse;
                    best_filter_intra = Some(filter_mode);
                    best_palette32 = None;
                }
            }
        }
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
        if rl.is_none()
            && angle_delta_enabled()
            && self.speed.try_angle_deltas_av1(32, self.base_q_idx)
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
            let ds = self.dc_sign_ctx_32(0, px / 4, py / 4);
            let wrr = self.idct.idct_dequant_32x32(&lcf, &self.quant);
            let wsse = sse_recon::<1024, 32>(&self.rd, &lpred, &wrr, &self.src[0], self.w, px, py, self.bd);
            let wbits = self.luma_bits(&lcf[..], &SCAN_32X32, 32, px, py, best_mode, 0);
            let mut best_ad_cost = rd_cost_i64(wsse, mlam, wbits + cdf_cost(&ad_cdf, 3));
            let mut ad_pred0 = self.sbuf_i1024();
            let mut ad_pred1 = self.sbuf_i1024();
            let mut ad_scratch = self.sbuf_i1024();
            let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
            for (di, &d) in self
                .rank_angle_deltas::<1024>(
                    best_mode, px, py, 32, 32, have_tr, have_bl, 2, &mut ad_preds,
                )
                .iter()
                .enumerate()
            {
                let pred: &[i32; 1024] = &*ad_preds[di];
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
                if self.speed.per_candidate_rdoq_av1() {
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
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                let sse = sse_recon::<1024, 32>(&self.rd, pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = self.luma_bits(&cf, &SCAN_32X32, 32, px, py, best_mode, 0);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if rl.is_some() || cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    *lpred = *pred;
                    *lcf = cf;
                    *ltf = tf;
                }
            }
        }
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        if rl.is_none() && !self.speed.per_candidate_rdoq_av1() {
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
                self.quant.qm_level(),
                self.quant.qidx() as i32,
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
        if rl.is_none()
            && best_filter_intra.is_none()
            && best_palette32.is_none()
            && self.banding_risk(px, py, 32)
        {
            let none_sse = sse_recon::<1024, 32>(&self.rd,
                &lpred,
                &self.idct.idct_dequant_32x32(&lcf, &self.quant),
                &self.src[0],
                self.w,
                px,
                py,
                self.bd,
            );
            let none_bits = self.luma_bits(&lcf[..], &SCAN_32X32, 32, px, py, best_mode, 0);
            let (cf4, sse_s, bits_s) =
                self.split32_luma_try(px, py, best_mode, best_delta, have_tr, have_bl, lam);
            // Acceptance: the split must genuinely improve SSE. block16 uses a
            // permissive +25% tolerance at 16x16, but four TX_16X16 cost far more
            // syntax than four TX_8X8 do, so copying that tolerance here lets in
            // bad splits — measured +6.7% BD-rate on detailed content. Requiring
            // a real improvement keeps the banding win and removes the loss.
            let base_rd = rd_cost_i64(
                none_sse,
                mlam,
                none_bits + self.tx_depth_bits(px, py, 32, 32, 0),
            );
            let candidate_rd = rd_cost_i64(
                sse_s,
                mlam,
                bits_s + self.tx_depth_bits(px, py, 32, 32, 1),
            );
            let guarded_take =
                (sse_s as i128) * 1024 <= (none_sse as i128) * (1024 - SPLIT32_SSE_MARGIN as i128);
            if raw_sse_guard_choice(
                "split-tx32",
                RawSseGuard::TxSplit,
                none_sse,
                sse_s,
                base_rd,
                candidate_rd,
                guarded_take,
            ) {
                tx_split = true;
                *lcf = cf4;
            }
        }
        // Pure-emit replay: install the recorded winner and its captured
        // post-trellis coefficients (every luma sub-search above was skipped).
        if let Some(r) = rl {
            best_mode = r.mode as usize;
            best_delta = r.delta as i32;
            if r.palette > 0 {
                let p =
                    lossy_luma_palette(
                        &self.kmeans,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        32,
                        32,
                        r.palette as usize,
                    )
                        .expect("32x32 palette replay: candidate no longer derivable");
                palette_pred(&mut lpred[..], 32, &p.colors, &p.packed_map, 32, 32);
                best_palette32 = Some(p);
            }
            best_filter_intra = FILTER_INTRA_MODES
                .iter()
                .copied()
                .find(|&f| f as u8 == r.filter);
            tx_split = matches!(r.tx, TxSel::SplitDct(_));
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
            palette: best_palette32
                .as_ref()
                .map_or(0, |p| (p.colors.len() + if p.top { 8 } else { 0 }) as u8),
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            // `SplitDct` marks the tx_depth=1 grid of four TX_16X16
            // (coefficients packed quadrant-major); otherwise DCT_DCT.
            tx: if tx_split {
                TxSel::SplitDct([1; 4])
            } else {
                TxSel::from_flags(false, false, false, false)
            },
        });
        self.push_luma_cf(&lcf[..]);
        let luma_zero = self.rd.all_zero_i32(&lcf[..]);
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
                best_palette32.as_ref(),
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
                best_palette32.as_ref(),
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
                best_palette32.as_ref(),
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
        palette: Option<&LossyLumaPalette>,
        uv_palette: Option<&LossyUvPalette>,
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
        self.emit_palette_mode_info(px, py, 32, 32, y_mode, !self.mono, palette, uv_palette);
        if palette.is_none() {
            self.emit_filter_intra(y_mode, 32, 32, filter_intra);
        }
        if let Some(p) = palette {
            self.emit_palette_map(p);
        }
        if let Some(up) = uv_palette {
            self.emit_palette_uv_map(up);
        }
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
                let mut cfq = self.sbuf_i256();
                cfq.copy_from_slice(&lcf[qi * 256..qi * 256 + 256]);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
                    let ds = self.dc_sign_ctx_16(0, qbx4, qby4);
                    encode_tx16_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &cfq,
                        false,
                        sk,
                        ds,
                        filter_intra_tx_mode(None, y_mode),
                        1, // DCT_DCT
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
                let mut pred = self.sbuf_i256();
                if y_mode == DC_PRED && angle_delta == 0 {
                    let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                    *pred = [d; 256];
                } else {
                    self.intrapred.predict_nd_ad(
                        y_mode,
                        angle_delta,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        16,
                        16,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        block_ftype,
                        &mut pred[..],
                        self.bd,
                    );
                }
                let rr = if block_skip {
                    [0i32; 256]
                } else {
                    self.idct.idct_dequant_16x16(&cfq, &self.quant)
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
            self.idct.idct_dequant_32x32(lcf, &self.quant)
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
    fn code_block8_rect(&mut self, x8: usize, y8: usize, vert: bool, have_tr: bool, have_bl: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 8);
        let sel_mlam = self.emit_mlam(x8 * 8, y8 * 8, 8);
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (lw, lh) = if vert { (4usize, 8usize) } else { (8, 4) };
        let mut all2_skip = true;
        for half in 0..2 {
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
            let yctx_sel = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let mut best_cost = f32::INFINITY;
            let mut best_mode = DC_PRED;
            let mut lpred_arr = [0i32; 32];
            let mut lcf = [0i32; 32];
            let mut cand_pred = [0i32; 32];
            for m in 0..13usize {
                if m == DC_PRED {
                    let dc = if vert {
                        self.intrapred.dc_pred_4x8(&self.recon[0], self.w, px, py, self.bd as i32)
                    } else {
                        self.intrapred.dc_pred_8x4(&self.recon[0], self.w, px, py, self.bd as i32)
                    };
                    cand_pred[..lw * lh].fill(dc);
                } else {
                    self.intrapred.predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        px,
                        py,
                        lw,
                        lh,
                        sub_tr,
                        sub_bl,
                        self.w,
                        self.h,
                        ftype,
                        &mut cand_pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 32];
                self.rd.residual_pred(
                    &mut resid,
                    &cand_pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    lw,
                    lh,
                );
                let (mut cf, tf) = if vert {
                    self.dct.dct4x8_t(&resid, &self.quant)
                } else {
                    self.dct.dct8x4_t(&resid, &self.quant)
                };
                self.luma_rect_trellis(&mut cf, &tf, dcq, acq, lscan, lam, lw, lh, px, py);
                self.rd.preserve_dc(&mut cf[0], &resid[..lw * lh]);
                let rr = if vert {
                    self.idct.idct_dequant_4x8(&cf, &self.quant)
                } else {
                    self.idct.idct_dequant_8x4(&cf, &self.quant)
                };
                let sse = self.rd.sse_recon(
                    &cand_pred,
                    &rr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    lw,
                    lh,
                    self.bd,
                );
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
            let luma_zero = self.rd.all_zero_i32(&lcf);
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
                        &self.intrapred,
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
                    let (mut q, qt) = fwd_chroma_rect8(&self.dct, cw, ch, &resid, &self.cquant);
                    let cscan = scan_rect8(cw, ch);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                    self.rd.preserve_dc(&mut q[0], &resid[..cn]);
                    ccf[ci] = q;
                }
            }
            let chroma_zero = !has_chroma
                || (self.rd.all_zero_i32(&ccf[0]) && self.rd.all_zero_i32(&ccf[1]));
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
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
            all2_skip &= block_skip;
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
            if has_chroma {
                self.emit_uv_mode(best_mode, DC_PRED, None, px, py, lw, lh);
            }
            self.emit_palette_mode_info(px, py, lw, lh, best_mode, has_chroma, None, None);
            self.emit_filter_intra(best_mode, lw, lh, None);
            self.code_tx_depth(px, py, lw, lh, 0);
            let sv = block_skip as u8;
            let (aw, ah) = ((lw / 4).max(1), (lh / 4).max(1));
            self.a_skip[bx4..bx4 + aw].fill(sv);
            self.l_skip[by4..by4 + ah].fill(sv);
            self.a_mode[bx4..bx4 + aw].fill(best_mode as u8);
            self.l_mode[by4..by4 + ah].fill(best_mode as u8);
            let lres_ctx = if block_skip {
                0x40
            } else if vert {
                let ds = self.dc_sign_ctx_4x8_luma(bx4, by4);
                encode_4x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, best_mode, 1)
            } else {
                let ds = self.dc_sign_ctx_8x4_luma(bx4, by4);
                encode_8x4_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, 0, ds, best_mode, 1)
            };
            self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
            self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 32]
            } else if vert {
                self.idct.idct_dequant_4x8(&lcf, &self.quant)
            } else {
                self.idct.idct_dequant_8x4(&lcf, &self.quant)
            };
            for ry in 0..lh {
                for rx in 0..lw {
                    let i = ry * lw + rx;
                    self.recon[0][(py + ry) * self.w + px + rx] =
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
                    self.a_coef[plane][cbx4..cbx4 + caw].fill(cres_ctx);
                    self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
                    let rr = if block_skip {
                        [0i32; 64]
                    } else {
                        inv_chroma_rect8(&self.idct, cw, ch, &ccf[ci], &self.cquant)
                    };
                    for ry in 0..ch {
                        let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                        recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                    }
                }
            }
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
                encode_8x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &cf.as_chunks::<32>().0[0], sk, ds)
            }
            (4, 8) => {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &cf.as_chunks::<32>().0[0], sk, ds)
            }
            _ => {
                let sk = self.skip_ctx_4x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x4_chroma(plane, cbx4, cby4);
                encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &cf.as_chunks::<16>().0[0], sk, ds)
            }
        }
    }

    /// 32x32 rect: HORZ = two 32x16, VERT = two 16x32. Chroma per format. DC intra.
    /// V forbidden in 4:2:2.
    /// Mode search for a 32-level rect luma leaf (RTX_32X16 / RTX_16X32 —
    /// DCT-only by spec, so no tx trial). Mirrors `rect16_luma_mode_search`:
    /// prediction-domain SATD rerank to a 3-beam (DC always kept), then the
    /// full DCT + trellis + entropy-rate pipeline under the caller's lambdas.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    fn rect32_luma_mode_search(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        dc: i32,
        lam: f32,
        mlam: f32,
    ) -> (usize, Box<[i32; 512]>, Box<[i32; 512]>, Box<[i32; 512]>) {
        let (w, h) = if vert { (16usize, 32usize) } else { (32, 16) };
        let scan: &[u32] = if vert { &SCAN_16X32 } else { &SCAN_32X16 };
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let ftype = self.luma_filter_type(px, py);
        let (bx4, by4) = (px / 4, py / 4);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        let kf = &self.dcdf().kf_y[yctx];
        let modes: &[usize] = if self.speed != Speed::Slow {
            &[DC_PRED]
        } else {
            &Self::RECT_LEAF_MODES
        };
        let mut cands = FixedList::<(u64, usize), 7>::new((0, DC_PRED));
        for &m in modes {
            let mut pred = self.sbuf_i512();
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
                    &mut pred[..],
                    self.bd,
                );
            }
            let score =
                self.rd.satd_sad_proxy(&self.src[0][py * self.w + px..], self.w, &pred[..], w, w, h);
            cands.push((score, m));
        }
        const BEAM: usize = 3;
        if cands.len() > BEAM {
            cands
                .as_mut_slice()
                .sort_unstable_by_key(|&(score, mode)| (score, mode));
            let dc_pos = cands.iter().position(|c| c.1 == DC_PRED).unwrap();
            if dc_pos >= BEAM {
                cands.as_mut_slice().swap(BEAM - 1, dc_pos);
            }
            cands.truncate(BEAM);
        }
        // Take the three buffers in SEPARATE statements: inside one tuple
        // expression the `RefMut` temporaries from each `self.sc()` all live
        // to the end of the full expression, and the second borrow panics
        // ("RefCell already borrowed"). Latent since the CoderScratch
        // migration — this leg was dormant and never executed it.
        let b0 = self.sc().take_i512();
        let b1 = self.sc().take_i512();
        let b2 = self.sc().take_i512();
        #[allow(clippy::type_complexity)]
        let mut best: (
            f32,
            usize,
            Box<[i32; 512]>,
            Box<[i32; 512]>,
            Box<[i32; 512]>,
        ) = (f32::INFINITY, DC_PRED, b0, b1, b2);
        for &(_, m) in &cands {
            let mut pred = self.sbuf_i512();
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
                    &mut pred[..],
                    self.bd,
                );
            }
            let mut resid = self.sbuf_i512();
            self.rd.residual_pred(
                &mut resid[..],
                &pred[..],
                &self.src[0],
                self.w,
                px,
                py,
                w,
                h,
            );
            let (mut cf, tf) = if vert {
                self.dct.dct16x32_t(&resid, &self.quant)
            } else {
                self.dct.dct32x16_t(&resid, &self.quant)
            };
            self.luma_rect_trellis(&mut cf, &tf, dcq, acq, scan, lam, w, h, px, py);
            self.rd.preserve_dc(&mut cf[0], &resid[..]);
            let rr = if vert {
                self.idct.idct_dequant_16x32(&cf, &self.quant)
            } else {
                self.idct.idct_dequant_32x16(&cf, &self.quant)
            };
            let sse = self.rd.sse_recon(
                &pred[..],
                &rr,
                &self.src[0],
                self.w,
                px,
                py,
                w,
                h,
                self.bd,
            );
            let mut bits = self.luma_rect_bits(&cf, scan, w, h, px, py, m, 1) + cdf_cost(kf, m);
            if (V_PRED..=VERT_LEFT_PRED).contains(&m) {
                bits += cdf_cost(&self.dcdf().angle_delta[m - V_PRED], 3);
            }
            let cost = rd_cost_i64(sse, mlam, bits);
            if cost < best.0 {
                best.0 = cost;
                best.1 = m;
                best.2.copy_from_slice(&pred[..]);
                best.3.copy_from_slice(&resid[..]);
                best.4.copy_from_slice(&cf);
            }
        }
        (best.1, best.2, best.3, best.4)
    }

    fn code_block32_rect(&mut self, x8: usize, y8: usize, vert: bool) {
        self.code_block32_rect_halves(x8, y8, vert, 0..2);
    }

    /// Trial-code a 32x16 / 16x32 rect leaf as TWO TX_16X16 (`tx_depth = 1`,
    /// `t_dim.sub` of the rect TX). Per the spec, intra prediction runs per
    /// TRANSFORM block: the second TX predicts from the first's running
    /// reconstruction — the finer prediction granularity is the point (a
    /// whole-rect prediction from a 32-wide edge is the measured weakness of
    /// the rect leaves). Each sub-TX searches the TX_16X16 5-type set (DCT /
    /// ADST_ADST / ADST_DCT / DCT_ADST trellis'd, IDTX plain) under the
    /// 8x8-style SSE admission gates. The searched mode set is extension-free
    /// (DC/V/H/SMOOTH*/PAETH), so `false, false` edge flags match dav1d at
    /// every sub-TX position. Temporarily writes candidate recon into
    /// `self.recon[0]` and restores it before returning.
    /// Returns (packed cf [sub0|sub1], packed prediction, recon, sse, bits,
    /// per-sub txtp16 indices).
    #[allow(clippy::type_complexity)]
    fn rect32_split_try(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        mode: usize,
        lam: f32,
        rd_lam: f32,
    ) -> (
        Box<[i32; 512]>,
        Box<[i32; 512]>,
        Box<[u16; 512]>,
        i64,
        f32,
        [usize; 2],
    ) {
        let (lw, lh) = if vert { (16usize, 32usize) } else { (32, 16) };
        let mut saved = self.sc().take_u512();
        for ry in 0..lh {
            saved[ry * lw..ry * lw + lw]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..lw]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let mut cf2 = self.sc().take_i512();
        let mut pred2 = self.sc().take_i512();
        let mut rec = self.sc().take_u512();
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let mut txtps = [1usize; 2];
        let subs: [(usize, usize); 2] = if vert {
            [(0, 0), (0, 16)]
        } else {
            [(0, 0), (16, 0)]
        };
        for (si, &(sx, sy)) in subs.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let ds = self.dc_sign_ctx_span(0, bx / 4, by / 4, 4, 4);
            let mut pred = self.sc().take_i256();
            if mode == DC_PRED {
                let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred.fill(d);
            } else {
                self.intrapred.predict_nd(
                    mode,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    16,
                    16,
                    false,
                    false,
                    self.w,
                    self.h,
                    block_ftype,
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
                bx,
                by,
                16,
                16,
            );
            // DCT baseline.
            let (mut dcf, dtf) = self.dct.dct16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut dcf,
                &dtf,
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
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let drr = self.idct.idct_dequant_16x16(&dcf, &self.quant);
            let dct_sse = sse_recon::<256, 16>(&self.rd, &pred, &drr, &self.src[0], self.w, bx, by, self.bd);
            let dct_bits = self.luma_bits(&dcf, &SCAN_16X16, 16, bx, by, mode, 1);
            let mut best = (dcf, dct_sse, dct_bits, 1usize);
            // 2-D refinements (trellis'd) + IDTX (plain), 8x8-style gates.
            for txtp in [
                ADST_ADST_TX16_IDX,
                ADST_DCT_TX16_IDX,
                DCT_ADST_TX16_IDX,
                IDTX_TX16_IDX,
            ] {
                let (mut acf, atf, rdoq) = match txtp {
                    ADST_ADST_TX16_IDX => {
                        let (cf, tf) = self.dct.adst16x16_t(&resid, &self.quant);
                        (cf, tf, true)
                    }
                    ADST_DCT_TX16_IDX => {
                        let (cf, tf) = self.dct.adstdct16x16_t(&resid, &self.quant);
                        (cf, tf, true)
                    }
                    DCT_ADST_TX16_IDX => {
                        let (cf, tf) = self.dct.dctadst16x16_t(&resid, &self.quant);
                        (cf, tf, true)
                    }
                    _ => {
                        let (cf, tf) = self.dct.idtx16x16_t(&resid, &self.quant);
                        (cf, tf, false)
                    }
                };
                if rdoq {
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
                        ds,
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let arr = match txtp {
                    x if x == ADST_ADST_TX16_IDX => self.idct.iadst_dequant_16x16(&acf, &self.quant),
                    x if x == ADST_DCT_TX16_IDX => self.idct.iadstdct_dequant_16x16(&acf, &self.quant),
                    x if x == DCT_ADST_TX16_IDX => self.idct.idctadst_dequant_16x16(&acf, &self.quant),
                    _ => self.idct.iidentity_dequant_16x16(&acf, &self.quant),
                };
                let asse = sse_recon::<256, 16>(&self.rd, &pred, &arr, &self.src[0], self.w, bx, by, self.bd);
                let gate_sse = if rdoq {
                    // +3% tolerance for the trellis'd 2-D family.
                    asse <= dct_sse + (dct_sse >> 5)
                } else {
                    // STRICT for IDTX.
                    asse <= best.1
                };
                if !gate_sse {
                    continue;
                }
                let bits_bound = (rd_cost_i64(best.1, rd_lam, best.2) - asse as f32) / rd_lam;
                let abits =
                    self.luma_bits_bounded(&acf, &SCAN_16X16, 16, bx, by, mode, txtp, bits_bound);
                if rd_cost_i64(asse, rd_lam, abits) < rd_cost_i64(best.1, rd_lam, best.2) {
                    best = (acf, asse, abits, txtp);
                }
            }
            let (bcf, bsse, bbits, btxtp) = best;
            let brr = match btxtp {
                1 => self.idct.idct_dequant_16x16(&bcf, &self.quant),
                x if x == ADST_ADST_TX16_IDX => self.idct.iadst_dequant_16x16(&bcf, &self.quant),
                x if x == ADST_DCT_TX16_IDX => self.idct.iadstdct_dequant_16x16(&bcf, &self.quant),
                x if x == DCT_ADST_TX16_IDX => self.idct.idctadst_dequant_16x16(&bcf, &self.quant),
                _ => self.idct.iidentity_dequant_16x16(&bcf, &self.quant),
            };
            // Install this sub-TX's recon so the next sub predicts from it.
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                Some((&mut rec[sy * lw + sx..], lw)),
                &pred[..],
                &brr,
                16,
                16,
                self.bd,
            );
            for (src, dst) in pred
                .as_chunks::<16>().0.iter()
                .zip(pred2[sy * lw + sx..].chunks_exact_mut(lw))
                .take(16)
            {
                dst[..16].copy_from_slice(src);
            }
            cf2[si * 256..si * 256 + 256].copy_from_slice(&bcf);
            txtps[si] = btxtp;
            sse_sum += bsse;
            bits_sum += bbits;
            self.sc().put_i256(pred);
        }
        // Restore the caller's reconstruction state.
        for ry in 0..lh {
            self.recon[0][(py + ry) * self.w + px..][..lw]
                .copy_from_slice(&saved[ry * lw..ry * lw + lw]);
        }
        self.sc().put_u512(saved);
        (cf2, pred2, rec, sse_sum, bits_sum, txtps)
    }

    /// Code a subrange of the two 32x16 / 16x32 halves of a 32 parent.
    /// `0..2` is the full HORZ/VERT partition; the A/B T-shapes code exactly
    /// one half (`1..2` for the trailing leaf of HORZ_A/VERT_A, `0..1` for
    /// the leading leaf of HORZ_B/VERT_B) with the 16x16 squares coded by the
    /// caller. Lambdas stay anchored at the 32 parent either way.
    fn code_block32_rect_halves(
        &mut self,
        x8: usize,
        y8: usize,
        vert: bool,
        halves: std::ops::Range<usize>,
    ) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in halves {
            let (px, py) = if vert {
                (x8 * 8 + half * 16, y8 * 8)
            } else {
                (x8 * 8, y8 * 8 + half * 16)
            };
            let (bx4, by4) = (px / 4, py / 4);
            let (lw, lh) = if vert { (16usize, 32usize) } else { (32, 16) };
            // Equipped leaf: mode search (edge-safe set, SATD 3-beam) under
            // the emit lambdas; DCT-only by spec at this size.
            let dc_l = if vert {
                self.intrapred.dc_pred_16x32(&self.recon[0], self.w, px, py, self.bd as i32)
            } else {
                self.intrapred.dc_pred_32x16(&self.recon[0], self.w, px, py, self.bd as i32)
            };
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            let emlam = self.emit_mlam(x8 * 8, y8 * 8, 32);
            let (y_mode, lpred_arr, lresid_scratch, lcf_box) =
                self.rect32_luma_mode_search(px, py, vert, dc_l, lam, emlam);
            self.sc().put_i512(lresid_scratch);
            let lcf = *lcf_box;
            self.sc().put_i512(lcf_box);
            // Whole-TX (DCT by spec at 32-dim) vs tx_depth=1 (two TX_16X16
            // with per-TX prediction + the 5-type set): the TX split is what
            // re-opens transform types at this block size.
            let whole_rr = if vert {
                self.idct.idct_dequant_16x32(&lcf, &self.quant)
            } else {
                self.idct.idct_dequant_32x16(&lcf, &self.quant)
            };
            let whole_sse = self.rd.sse_recon(
                &lpred_arr[..],
                &whole_rr,
                &self.src[0],
                self.w,
                px,
                py,
                lw,
                lh,
                self.bd,
            );
            let whole_bits =
                self.luma_rect_bits(&lcf, scan_rect(lw, lh), lw, lh, px, py, y_mode, 1);
            let (s_cf, s_pred_scratch, s_rec, s_sse, s_bits, s_txtps) =
                self.rect32_split_try(px, py, vert, y_mode, lam, emlam);
            self.sc().put_i512(s_pred_scratch);
            // +1 bit allowance for the deeper tx_depth symbol.
            let tx_split = rd_cost_i64(s_sse, emlam, s_bits + self.tx_depth_bits(px, py, lw, lh, 1))
                < rd_cost_i64(
                    whole_sse,
                    emlam,
                    whole_bits + self.tx_depth_bits(px, py, lw, lh, 0),
                );
            let luma_zero = if tx_split {
                self.rd.all_zero_i32(&s_cf[..])
            } else {
                self.rd.all_zero_i32(&lcf)
            };
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
            let mut ccf = [self.sbuf_i512(), self.sbuf_i512()];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc =
                    chroma_dc_rect(
                        &self.intrapred,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        cw,
                        ch,
                        self.bd as i32,
                    );
                cpred[ci] = dc;
                let mut resid = self.sbuf_i512();
                self.rd.residual_dc(
                    &mut resid[..],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    cw,
                    ch,
                    dc,
                );
                let (mut q, qt) = fwd_chroma_rect(&self.dct, cw, ch, &resid, &self.cquant);
                let cscan = scan_rect(cw, ch);
                trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                self.rd.preserve_dc(&mut q[0], &resid[..cn]);
                *ccf[ci] = q;
            }
            // CfL trial: predict the rect chroma from this leaf's reconstructed
            // luma (legal here — the LUMA block is <= 32x32). The missing
            // chroma tooling was the measured reason equipped-luma rect32
            // still lost to the fully-armed 32-NONE.
            let mut use_cfl = false;
            let mut cfl_alpha = [0i32; 2];
            let mut cfl_pred_px: [Box<[i32; 512]>; 2] = {
                let mut sc = self.sc();
                [sc.take_i512(), sc.take_i512()]
            };
            if !self.mono {
                let mlam32 = self.emit_mlam(x8 * 8, y8 * 8, 32);
                let mut luma_rec = self.sc().take_u512();
                if tx_split {
                    luma_rec[..].copy_from_slice(&s_rec[..]);
                } else {
                    let lrr_cfl = if vert {
                        self.idct.idct_dequant_16x32(&lcf, &self.quant)
                    } else {
                        self.idct.idct_dequant_32x16(&lcf, &self.quant)
                    };
                    recon_add_pred(&mut luma_rec[..], &lpred_arr[..], &lrr_cfl, maxval);
                }
                let mut ac = self.sc().take_i512();
                if self.ss420 {
                    self.intrapred
                        .cfl_ac_sub(&luma_rec[..], lw, cw, ch, true, true, &mut ac[..]);
                } else if self.ss422 {
                    self.intrapred
                        .cfl_ac_sub(&luma_rec[..], lw, cw, ch, true, false, &mut ac[..]);
                } else {
                    self.intrapred
                        .cfl_ac_444(&luma_rec[..], lw, lh, &mut ac[..]);
                }
                self.sc().put_u512(luma_rec);
                let cscan = scan_rect(cw, ch);
                let mut cfl_ccf = [self.sbuf_i512(), self.sbuf_i512()];
                let mut cfl_a = [0i32; 2];
                let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut csrc = self.sc().take_u512();
                    self.rd.copy_block_u16(
                        &mut csrc[..cn],
                        &self.src[plane],
                        self.cw,
                        cx,
                        cy,
                        cw,
                        ch,
                    );
                    let dcrr = inv_chroma_rect(&self.idct, cw, ch, &ccf[ci], &self.cquant);
                    let s = self.rd.sse_recon(
                        &[dc; 512][..cn],
                        &dcrr[..cn],
                        &csrc[..],
                        cw,
                        0,
                        0,
                        cw,
                        ch,
                        self.bd,
                    );
                    dc_cost[ci] = rd_cost_i64(
                        s,
                        mlam32,
                        self.chroma_rect_bits(&ccf[ci][..cn], cscan, cw, ch, plane, cx, cy),
                    );
                    let a =
                        self.intrapred
                            .cfl_best_alpha(&ac[..cn], &csrc[..cn], dc, cn, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = self.sc().take_i512();
                    self.intrapred
                        .cfl_pred(&mut cpr[..cn], &ac[..cn], dc, a, self.bd);
                    let mut resid = self.sbuf_i512();
                    self.rd.residual_pred(
                        &mut resid[..],
                        &cpr[..],
                        &csrc[..],
                        cw,
                        0,
                        0,
                        cw,
                        ch,
                    );
                    let (mut q, qt) = fwd_chroma_rect(&self.dct, cw, ch, &resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
                    let rr2 = inv_chroma_rect(&self.idct, cw, ch, &q, &self.cquant);
                    let s2 = self.rd.sse_recon(
                        &cpr[..cn],
                        &rr2[..cn],
                        &csrc[..],
                        cw,
                        0,
                        0,
                        cw,
                        ch,
                        self.bd,
                    );
                    *cfl_ccf[ci] = q;
                    cfl_pred_px[ci][..cn].copy_from_slice(&cpr[..cn]);
                    cfl_cost[ci] = rd_cost_i64(
                        s2,
                        mlam32,
                        self.chroma_rect_bits(&q[..cn], cscan, cw, ch, plane, cx, cy),
                    );
                    let mut sc = self.sc();
                    sc.put_u512(csrc);
                    sc.put_i512(cpr);
                }
                let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
                if (cfl_a[0] != 0 || cfl_a[1] != 0)
                    && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam32, cfl_sig)
                        < dc_cost[0]
                            + dc_cost[1]
                            + rate_cost(mlam32, self.uv_mode_bits(y_mode, DC_PRED, None))
                {
                    use_cfl = true;
                    cfl_alpha = cfl_a;
                    ccf = cfl_ccf;
                }
                self.sc().put_i512(ac);
            }
            let chroma_zero =
                self.rd.all_zero_i32(&ccf[0][..]) && self.rd.all_zero_i32(&ccf[1][..]);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(px / 8, py / 8, (lw / 4) as u8, (lh / 4) as u8);
            if tx_split {
                // The deblock filter runs on TRANSFORM edges: override the tx
                // map with the two TX_16X16 subs (the prediction-block map
                // keeps the whole rect). Same pattern as the 16x16 and 32x32
                // TX-split paths.
                if vert {
                    self.record_tx_blk(px / 8, py / 8, 4);
                    self.record_tx_blk(px / 8, py / 8 + 2, 4);
                } else {
                    self.record_tx_blk(px / 8, py / 8, 4);
                    self.record_tx_blk(px / 8 + 2, py / 8, 4);
                }
            }
            self.mark_skip8_rect(px / 8, py / 8, lw / 8, lh / 8, block_skip);
            self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
            if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
                self.enc
                    .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
            }
            self.emit_uv_mode(
                y_mode,
                if use_cfl { CFL_PRED } else { DC_PRED },
                if use_cfl { Some(cfl_alpha) } else { None },
                px,
                py,
                lw,
                lh,
            );
            self.emit_palette_mode_info(px, py, lw, lh, y_mode, !self.mono, None, None);
            self.emit_filter_intra(y_mode, lw, lh, None);
            self.code_tx_depth(px, py, lw, lh, tx_split as usize);
            let sv = block_skip as u8;
            let (aw, ah) = (lw / 4, lh / 4);
            self.a_skip[bx4..bx4 + aw].fill(sv);
            self.l_skip[by4..by4 + ah].fill(sv);
            self.a_mode[bx4..bx4 + aw].fill(y_mode as u8);
            self.l_mode[by4..by4 + ah].fill(y_mode as u8);
            if tx_split && !block_skip {
                // Two TX_16X16 in coding order with progressive coef contexts
                // (sub1's skip/dc-sign ctx sees sub0's result, like dav1d).
                let subs: [(usize, usize); 2] = if vert {
                    [(0, 0), (0, 16)]
                } else {
                    [(0, 0), (16, 0)]
                };
                for (si, &(sx, sy)) in subs.iter().enumerate() {
                    let (sbx4, sby4) = ((px + sx) / 4, (py + sy) / 4);
                    let mut cfs = self.sbuf_i256();
                    cfs.copy_from_slice(&s_cf[si * 256..si * 256 + 256]);
                    let sk = self.skip_ctx_split(sbx4, sby4, 4, 4);
                    let ds = self.dc_sign_ctx_16(0, sbx4, sby4);
                    let res_ctx = encode_tx16_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &cfs,
                        false,
                        sk,
                        ds,
                        y_mode,
                        s_txtps[si],
                    );
                    self.a_coef[0][sbx4..sbx4 + 4].fill(res_ctx);
                    self.l_coef[0][sby4..sby4 + 4].fill(res_ctx);
                }
                for ry in 0..lh {
                    self.recon[0][(py + ry) * self.w + px..][..lw]
                        .copy_from_slice(&s_rec[ry * lw..ry * lw + lw]);
                }
            } else if tx_split {
                // Skipped split block: the decoder still predicts per TX, and
                // the second TX predicts from the FIRST'S PREDICTION-ONLY
                // reconstruction (zero residual) — NOT from the trial's
                // with-residual recon. Recompute the sequential prediction.
                self.a_coef[0][bx4..bx4 + aw].fill(0x40);
                self.l_coef[0][by4..by4 + ah].fill(0x40);
                let block_ftype = self.luma_filter_type(px, py);
                let subs: [(usize, usize); 2] = if vert {
                    [(0, 0), (0, 16)]
                } else {
                    [(0, 0), (16, 0)]
                };
                for &(sx, sy) in subs.iter() {
                    let (bx, by) = (px + sx, py + sy);
                    let mut pred = self.sc().take_i256();
                    if y_mode == DC_PRED {
                        let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                        pred.fill(d);
                    } else {
                        self.intrapred.predict_nd(
                            y_mode,
                            &self.recon[0],
                            self.w,
                            bx,
                            by,
                            16,
                            16,
                            false,
                            false,
                            self.w,
                            self.h,
                            block_ftype,
                            &mut pred[..],
                            self.bd,
                        );
                    }
                    self.rd.reconstruct(
                        &mut self.recon[0][by * self.w + bx..],
                        self.w,
                        None,
                        &pred[..],
                        &[],
                        16,
                        16,
                        self.bd,
                    );
                    self.sc().put_i256(pred);
                }
            } else {
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
                    self.idct.idct_dequant_16x32(&lcf, &self.quant)
                } else {
                    self.idct.idct_dequant_32x16(&lcf, &self.quant)
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
                    inv_chroma_rect(&self.idct, cw, ch, &ccf[ci], &self.cquant)
                };
                for ry in 0..ch {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    if use_cfl {
                        recon_add_pred(
                            &mut drow[..cw],
                            &cfl_pred_px[ci][ry * cw..],
                            &rr[ry * cw..],
                            maxval,
                        );
                    } else {
                        recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
                    }
                }
            }
            let mut sc = self.sc();
            sc.put_i512(lpred_arr);
            sc.put_i512(s_cf);
            sc.put_u512(s_rec);
            let [cp0, cp1] = cfl_pred_px;
            sc.put_i512(cp0);
            sc.put_i512(cp1);
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
                let mut a = self.sbuf_i512();
                a.copy_from_slice(cf);
                encode_32x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &a, sk, ds)
            }
            (16, 32) => {
                let sk = self.skip_ctx_16x32_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_16x32_chroma(plane, cbx4, cby4);
                let mut a = self.sbuf_i512();
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
                let mut a = self.sbuf_i256();
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
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..32]);
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
                let d = self.intrapred.dc_pred_16x16(&self.recon[0], self.w, bx, by, self.bd as i32);
                *pred = [d; 256];
            } else {
                self.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    16,
                    16,
                    tr,
                    bl,
                    self.w,
                    self.h,
                    block_ftype,
                    &mut pred[..],
                    self.bd,
                );
            }
            let mut cfq = self.sbuf_i256();
            cfq.copy_from_slice(&lcf[qi * 256..qi * 256 + 256]);
            let rr = self.idct.idct_dequant_16x16(&cfq, &self.quant);
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                Some((&mut rec[sy * 32 + sx..], 32)),
                &pred[..],
                &rr,
                16,
                16,
                self.bd,
            );
        }
        for ry in 0..32 {
            self.recon[0][(py + ry) * self.w + px..][..32]
                .copy_from_slice(&saved[ry * 32..ry * 32 + 32]);
        }
        self.sc().put_u1024(saved);
        rec
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
        palette: Option<&LossyLumaPalette>,
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
        let mut ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cdc = [0i32; 2];
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let dc = self.intrapred.dc_pred_32x32(&self.recon[plane], self.w, px, py, self.bd as i32);
            cdc[ci] = dc;
            let mut cresid = self.sbuf_i1024();
            self.rd.residual_dc(
                &mut cresid[..],
                &self.src[plane],
                self.w,
                px,
                py,
                32,
                32,
                dc,
            );
            let (q, qt) = self.dct.dct32x32_t(&cresid, &self.cquant);
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
            self.rd.preserve_dc(&mut ccf[ci][0], &cresid[..]);
        }
        // CfL: predict chroma from the reconstructed luma AC.
        let mut cfl_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cfl_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
        let mut cfl_a = [0i32; 2];
        let (mut dc_cost, mut cfl_cost) = ([0f32; 2], [0f32; 2]);
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 32);
        // Pure-emit replay never evaluates CfL; the use_cfl decision below
        // replays from the record and the winner state installs after this.
        if self.speed.full_chroma_rdo() && ru.is_none() {
            let mut luma_rec = self.sbuf_u1024();
            if tx_split {
                let r = self.split32_luma_recon_from_cf(
                    px,
                    py,
                    y_mode,
                    angle_delta,
                    have_tr,
                    have_bl,
                    lcf,
                );
                luma_rec.copy_from_slice(&r[..]);
                self.sc().put_u1024(r);
            } else {
                let lrr_cfl = self.idct.idct_dequant_32x32(lcf, &self.quant);
                recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, (1 << self.bd) - 1);
            }
            let mut ac = self.sbuf_i1024();
            self.intrapred
                .cfl_ac_444(&luma_rec[..], 32, 32, &mut ac[..]);
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = cdc[ci];
                let mut src = self.sbuf_u1024();
                self.rd
                    .copy_block_u16(&mut src[..], &self.src[plane], self.w, px, py, 32, 32);
                let dcrr = self.idct.idct_dequant_32x32(&ccf[ci], &self.cquant);
                let s = sse_recon::<1024, 32>(&self.rd, &[dc; 1024], &dcrr, &src[..], 32, 0, 0, self.bd);
                dc_cost[ci] = rd_cost_i64(
                    s,
                    mlam,
                    self.chroma_bits(&ccf[ci][..], &SCAN_32X32, 32, plane, px, py),
                );
                let a = self
                    .intrapred
                    .cfl_best_alpha(&ac[..], &src[..], dc, 1024, self.bd);
                cfl_a[ci] = a;
                let mut cpr = self.sbuf_i1024();
                self.intrapred
                    .cfl_pred(&mut cpr[..], &ac[..1024], dc, a, self.bd);
                let mut resid = self.sbuf_i1024();
                self.rd.residual_pred(&mut resid[..], &cpr[..], &src[..], 32, 0, 0, 32, 32);
                let (mut q, qt) = self.dct.dct32x32_t(&resid, &self.cquant);
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
                let rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
                let s2 = sse_recon::<1024, 32>(&self.rd, &cpr, &rr, &src[..], 32, 0, 0, self.bd);
                *cfl_ccf[ci] = q;
                *cfl_pred[ci] = *cpr;
                cfl_cost[ci] = rd_cost_i64(
                    s2,
                    mlam,
                    self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py),
                );
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
        let cfl_sig = self.uv_mode_bits(y_mode, CFL_PRED, Some(cfl_a));
        // Let the RD comparison decide DC-vs-CfL across the whole quality range;
        // the old `acq > 300` gate suppressed CfL exactly where it helps most
        // (high quality). block8/block16 already dropped it — this path was the
        // last one still gated, which mattered most at 4:4:4 where chroma is
        // full resolution and 32x32 blocks are common.
        let use_cfl = ru.map_or(
            (cfl_a[0] != 0 || cfl_a[1] != 0)
                && cfl_cost[0] + cfl_cost[1] + rate_cost(mlam, cfl_sig)
                    < dc_cost[0]
                        + dc_cost[1]
                        + rate_cost(mlam, self.uv_mode_bits(y_mode, DC_PRED, None)),
            |r| r.uv == CFL_PRED as u8,
        );
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
        let (final_cf, chosen_uv_32) = 'sv: {
            // Pure-emit replay: the captured coefficients were installed above
            // (into cfl_ccf for CfL, ccf otherwise); no search runs at all.
            if let Some(r) = ru {
                if r.uv == DC_PRED as u8 || r.uv == CFL_PRED as u8 {
                    if r.palette > 0 {
                        uv_pal32 = Some(uv_palette_rederive(
                            &self.kmeans,
                            &self.src[1],
                            &self.src[2],
                            self.w,
                            px,
                            py,
                            32,
                            32,
                            r.palette as usize,
                        ));
                    }
                    break 'sv (cf_use, DC_PRED);
                }
                cfl_opt = None;
                break 'sv (cf_use, r.uv as usize);
            }
            if !self.speed.full_chroma_rdo() {
                break 'sv (cf_use, DC_PRED);
            }
            let dcq2 = self.cquant.dc_q() as f32;
            let acq2 = self.cquant.ac_q() as f32;
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
                let rr = self.idct.idct_dequant_32x32(&cf_use[ci], &self.cquant);
                let cur_pred = if use_cfl {
                    *cfl_pred[ci]
                } else {
                    [pred_dc[ci]; 1024]
                };
                let sse = sse_recon::<1024, 32>(&self.rd,
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
                    self.chroma_bits(&cf_use[ci][..], &SCAN_32X32, 32, plane, px, py),
                );
            }

            let mut best_total = cur_total;
            let mut best_mode = DC_PRED;
            let mut best_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
            let mut best_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
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
                self.rank_chroma_modes::<1024>(candidates, px, py, px, py, 32, 32)
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
                if ru.is_none() && !directional_top.contains(cand) {
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
                    self.intrapred.predict_nd(
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
                        &mut cand_pred[ci][..],
                        self.bd,
                    );
                    let mut resid = self.sbuf_i1024();
                    self.rd.residual_pred(
                        &mut resid[..],
                        &cand_pred[ci][..],
                        &self.src[plane],
                        self.w,
                        px,
                        py,
                        32,
                        32,
                    );
                    // Forced DCT_DCT at 32x32 (spec), regardless of uv_mode.
                    let (mut q, qt) = self.dct.dct32x32_t(&resid, &self.cquant);
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
                    self.rd.preserve_dc(&mut q[0], &resid[..]);
                    *cand_ccf[ci] = q;
                    let rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
                    let sse = sse_recon::<1024, 32>(&self.rd,
                        &cand_pred[ci],
                        &rr,
                        &self.src[plane],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    cand_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_bits(&q, &SCAN_32X32, 32, plane, px, py),
                    );
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
            // UV palette candidates (4:4:4): exact, else lossy k-means
            // clusterings priced with real chroma SSE (see the 16x16 twin).
            if self.try_palette() {
                let exact = exact_uv_palette(&self.src[1], &self.src[2], self.w, px, py, 32, 32);
                let pcands: Vec<LossyUvPalette> = if let Some(up) = exact {
                    vec![up]
                } else {
                    // Lossy candidates re-enabled 2026-07-23 WITH residual
                    // coding (below): the palette is now a PREDICTOR — the
                    // coded residual repairs the chroma flattening that made
                    // zero-residual lossy palettes SS2-fatal in both prior
                    // attempts. This is aom's mechanism (490 UV palettes on
                    // kodak20 q93, all carrying coefficients).
                    [(8usize, false), (4, false), (8, true), (4, true)]
                        .iter()
                        .filter_map(|&(k, top)| {
                            lossy_uv_palette(
                                &self.kmeans,
                                &self.src[1],
                                &self.src[2],
                                self.w,
                                px,
                                py,
                                32,
                                32,
                                k,
                                top,
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
                    let (dcq2, acq2) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
                    let mut pal_pred = [self.sbuf_i1024(), self.sbuf_i1024()];
                    let [pred_u, pred_v] = &mut pal_pred;
                    palette_uv_pred(&mut pred_u[..], &mut pred_v[..], &up.map, &up.u, &up.v);
                    let mut bits = self.uv_mode_bits(y_mode, DC_PRED, None)
                        + self.palette_uv_rate_bits(palette.is_some(), &up);
                    let mut sse = 0i64;
                    let mut pal_ccf = [self.sbuf_i1024(), self.sbuf_i1024()];
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let mut resid = self.sbuf_i1024();
                        self.rd.residual_pred(
                            &mut resid[..],
                            &pal_pred[ci][..],
                            &self.src[plane],
                            self.w,
                            px,
                            py,
                            32,
                            32,
                        );
                        let (mut q, qt) = self.dct.dct32x32_t(&resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_32X32, trellis_lambda());
                        let rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
                        sse += self.rd.sse_recon(
                            &pal_pred[ci][..],
                            &rr,
                            &self.src[plane],
                            self.w,
                            px,
                            py,
                            32,
                            32,
                            self.bd,
                        );
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
        // Capture the final chroma winner (CfL folded in as CFL_PRED).
        self.push_uv_sel(UvSel {
            uv: if chosen_uv_32 != DC_PRED {
                chosen_uv_32 as u8
            } else if cfl_opt.is_some() {
                CFL_PRED as u8
            } else {
                DC_PRED as u8
            },
            palette: uv_pal32
                .as_ref()
                .map_or(0, |p| (p.u.len() + if p.top { 8 } else { 0 }) as u8),
        });
        self.push_uv_cf(
            &final_cf[0][..],
            &final_cf[1][..],
            cfl_opt.unwrap_or([0, 0]),
        );
        let block_skip = palette.is_none()
            && uv_pal32.is_none()
            && luma_zero
            && self.rd.all_zero_i32(&final_cf[0][..])
            && self.rd.all_zero_i32(&final_cf[1][..]);
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
            palette,
            uv_pal32.as_ref(),
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
                self.idct.idct_dequant_32x32(&final_cf[ci], &self.cquant)
            };
            let max = (1 << self.bd) - 1;
            for (ry, rrow) in crr.as_chunks::<32>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                if chosen_uv_32 != DC_PRED || uv_pal32.is_some() {
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
        palette: Option<&LossyLumaPalette>,
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
        let mut ccf_dc = [self.sbuf_i256(), self.sbuf_i256()];
        let mut dc_preds = [0i32; 2];
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let dc = self.intrapred.dc_pred_16x16(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
            dc_preds[ci] = dc;
            let mut resid = self.sbuf_i256();
            self.rd.residual_dc(
                &mut resid[..],
                &self.src[plane],
                self.cw,
                cx,
                cy,
                16,
                16,
                dc,
            );
            let (q, qt) = self.dct.dct16x16_t(&resid, &self.cquant);
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
            self.rd.preserve_dc(&mut ccf_dc[ci][0], &resid[..]);
        }
        // SMOOTH_V chroma derives ADST_DCT (a 2D tx -> default scan and coef
        // contexts identical to DCT_DCT; only the transform differs). Forward with
        // adstdct16x16_t and reconstruct with iadstdct_dequant_16x16 to match the
        // decoder's derived chroma txtp. Offered at every quality; the Lagrangian
        // R-D decision below selects it only when it truly wins.
        // DC baseline R-D (libaom-style: SSE + mlam*coeff_bits over U+V).
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 32);
        let mut rr_dc = [self.sbuf_i256(), self.sbuf_i256()];
        let mut dc_total = 0f32;
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            *rr_dc[ci] = self.idct.idct_dequant_16x16(&ccf_dc[ci], &self.cquant);
            let dc = dc_preds[ci];
            let sse = sse_recon::<256, 16>(&self.rd,
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
        if ru.is_none() && !self.mono {
            let mut luma_rec = self.sbuf_u1024();
            if tx_split {
                let r = self.split32_luma_recon_from_cf(
                    px,
                    py,
                    y_mode,
                    angle_delta,
                    have_tr,
                    have_bl,
                    lcf,
                );
                luma_rec.copy_from_slice(&r[..]);
                self.sc().put_u1024(r);
            } else {
                let lrr = self.idct.idct_dequant_32x32(lcf, &self.quant);
                recon_add_pred(&mut luma_rec[..], lpred, &lrr, maxval);
            }
            let mut ac = self.sbuf_i256();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 32, 16, 16, true, true, &mut ac[..]);
            let mut cfl_ccf = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_rr = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_px = [self.sbuf_i256(), self.sbuf_i256()];
            let mut cfl_a = [0i32; 2];
            let mut cfl_body = 0f32;
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_preds[ci];
                let mut src = self.sbuf_u256();
                self.rd
                    .copy_block_u16(&mut src[..], &self.src[plane], self.cw, cx, cy, 16, 16);
                let a = self
                    .intrapred
                    .cfl_best_alpha(&ac[..], &src[..], dc, 256, self.bd);
                cfl_a[ci] = a;
                let mut cpr = self.sbuf_i256();
                self.intrapred.cfl_pred(&mut cpr[..], &ac[..256], dc, a, self.bd);
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(&mut resid[..], &cpr[..], &src[..], 16, 0, 0, 16, 16);
                let (mut q, qt) = self.dct.dct16x16_t(&resid, &self.cquant);
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
                let rr = self.idct.idct_dequant_16x16(&q, &self.cquant);
                let sse = sse_recon::<256, 16>(&self.rd, &cpr, &rr, &src[..], 16, 0, 0, self.bd);
                *cfl_ccf[ci] = q;
                *cfl_rr[ci] = rr;
                *cfl_px[ci] = *cpr;
                cfl_body += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy),
                );
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
            self.rank_chroma_modes::<256>(candidates, px, py, cx, cy, 16, 16)
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
                self.intrapred.predict_nd(
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
                    &mut cand_pred[ci][..],
                    self.bd,
                );
                let mut resid = self.sbuf_i256();
                self.rd.residual_pred(
                    &mut resid[..],
                    &cand_pred[ci][..],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    16,
                    16,
                );
                let (mut q, qt) = fwd_chroma_16x16(&self.dct, tx, &resid, &self.cquant);
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
                self.rd.preserve_dc(&mut q[0], &resid[..]);
                *cand_ccf[ci] = q;
                *cand_rr[ci] = inv_chroma_16x16(&self.idct, tx, &q, &self.cquant);
                let sse = sse_recon::<256, 16>(&self.rd,
                    &cand_pred[ci],
                    &cand_rr[ci],
                    &self.src[plane],
                    self.cw,
                    cx,
                    cy,
                    self.bd,
                );
                cand_total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy),
                );
            }
            if ru.is_some() || cand_total < best_total {
                best_total = cand_total;
                chosen_uv = cand;
                use_cfl = false;
                best_ccf = cand_ccf;
                best_rr = cand_rr;
                sv_preds = cand_pred;
            }
        }
        // Pure-emit replay: install the captured chroma winner (mode + coeffs).
        if let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            chosen_uv = r.uv as usize;
            if r.uv == CFL_PRED as u8 {
                chosen_uv = DC_PRED;
                use_cfl = true;
                cfl_alpha = *al;
            }
            for (dst, src) in best_ccf.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
        }
        let use_sv = chosen_uv != DC_PRED || use_cfl;
        let (ccf, rr_cache) = (best_ccf, best_rr);
        self.push_uv_sel(UvSel {
            uv: if use_cfl {
                CFL_PRED as u8
            } else {
                chosen_uv as u8
            },
            palette: 0,
        });
        self.push_uv_cf(
            &ccf[0][..],
            &ccf[1][..],
            if use_cfl { cfl_alpha } else { [0, 0] },
        );
        let block_skip = palette.is_none()
            && luma_zero
            && self.rd.all_zero_i32(&ccf[0][..])
            && self.rd.all_zero_i32(&ccf[1][..]);
        self.code_header_luma32(
            x8,
            y8,
            lcf,
            lpred,
            y_mode,
            block_skip,
            chosen_uv,
            if use_cfl { Some(cfl_alpha) } else { None },
            angle_delta,
            filter_intra,
            palette,
            None,
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
                *rr_cache[ci]
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
        palette: Option<&LossyLumaPalette>,
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
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 32);
        let mut ccf = [self.sbuf_i512(), self.sbuf_i512()];
        let mut cpred = [0i32; 2];
        let mut cpred_px = [self.sbuf_i512(), self.sbuf_i512()];
        let mut src_planes = [self.sbuf_u512(), self.sbuf_u512()];
        let mut dc_ccf = [self.sbuf_i512(), self.sbuf_i512()];
        let mut dc_sse = [0i64; 2];
        let mut dc_bits = [0f32; 2];
        // DC option (skipped in pure-emit replay: the captured winner installs
        // below and block_skip reads the FINAL coeffs, matching Off).
        for ci in 0..(if ru.is_some() { 0 } else { 2 }) {
            let plane = ci + 1;
            let pred = self.intrapred.dc_pred_16x32(&self.recon[plane], self.cw, cx, py, self.bd as i32);
            cpred[ci] = pred;
            let mut src = self.sbuf_u512();
            self.rd
                .copy_block_u16(&mut src[..], &self.src[plane], self.cw, cx, py, 16, 32);
            let mut resid = self.sbuf_i512();
            self.rd.residual_dc(&mut resid[..], &src[..], 16, 0, 0, 16, 32, pred);
            let (mut q, qt) = self.dct.dct16x32_t(&resid, &self.cquant);
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
            let rr = self.idct.idct_dequant_16x32(&q, &self.cquant);
            *dc_ccf[ci] = q;
            dc_sse[ci] =
                self.rd.sse_recon(&[pred; 512], &rr, &src[..], 16, 0, 0, 16, 32, self.bd);
            dc_bits[ci] = self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, plane, cx, py);
            src_planes[ci] = src;
        }

        // CfL: predict the 16x32 U/V from the horizontally-subsampled 32x32
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0). 32x32 luma is
        // always DCT_DCT here, so the AC reference inverts with idct_dequant_32x32.
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // below.
        if ru.is_none() {
            let mut luma_rec = self.sbuf_u1024();
            if tx_split {
                // Same root cause as the 4:4:4 fix: TX-split luma must be
                // reconstructed per sub-TX for the CfL AC reference.
                let r = self.split32_luma_recon_from_cf(
                    px,
                    py,
                    y_mode,
                    angle_delta,
                    have_tr,
                    have_bl,
                    lcf,
                );
                luma_rec.copy_from_slice(&r[..]);
                self.sc().put_u1024(r);
            } else {
                let lrr_cfl = self.idct.idct_dequant_32x32(lcf, &self.quant);
                recon_add_pred(&mut luma_rec[..], lpred, &lrr_cfl, maxv);
            }
            let mut ac = self.sbuf_i512();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 32, 16, 32, true, false, &mut ac[..]);
            let mut cfl_ccf = [self.sbuf_i512(), self.sbuf_i512()];
            let mut cfl_a = [0i32; 2];
            let mut cfl_sse = [0i64; 2];
            let mut cfl_bits = [0f32; 2];
            for ci in 0..2 {
                let dc = cpred[ci];
                let src = &src_planes[ci];
                let a = self
                    .intrapred
                    .cfl_best_alpha(&ac[..], &src[..], dc, 512, self.bd);
                cfl_a[ci] = a;
                let mut cpr = self.sbuf_i512();
                self.intrapred.cfl_pred(&mut cpr[..], &ac[..512], dc, a, self.bd);
                let mut resid = self.sbuf_i512();
                self.rd.residual_pred(&mut resid[..], &cpr[..], &src[..], 16, 0, 0, 16, 32);
                let (mut q, qt) = self.dct.dct16x32_t(&resid, &self.cquant);
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
                let rr = self.idct.idct_dequant_16x32(&q, &self.cquant);
                *cfl_ccf[ci] = q;
                cfl_sse[ci] =
                    self.rd.sse_recon(&cpr[..], &rr, &src[..], 16, 0, 0, 16, 32, self.bd);
                cfl_bits[ci] = self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, ci + 1, cx, py);
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
            if ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                use_cfl = true;
                cfl_alpha_uv = cfl_a;
                for ci in 0..2 {
                    *ccf[ci] = *cfl_ccf[ci];
                }
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
                let rr = self.idct.idct_dequant_16x32(&cur_ccf, &self.cquant);
                let cur_pred = if use_cfl {
                    *cpred_px[ci]
                } else {
                    [cpred[ci]; 512]
                };
                let sse = self.rd.sse_recon(
                    &cur_pred,
                    &rr,
                    &src_planes[ci][..],
                    16,
                    0,
                    0,
                    16,
                    32,
                    self.bd,
                );
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
            let directional_top = if ru.is_none() {
                self.rank_chroma_modes::<512>(candidates, px, py, cx, py, 16, 32)
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
                if ru.is_none() && !directional_top.contains(cand) {
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
                    self.intrapred.predict_nd(
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
                        &mut cand_pred[ci][..],
                        self.bd,
                    );
                    let src = &src_planes[ci];
                    let mut resid = self.sbuf_i512();
                    self.rd.residual_pred(
                        &mut resid[..],
                        &cand_pred[ci][..],
                        &src[..],
                        16,
                        0,
                        0,
                        16,
                        32,
                    );
                    // Forced DCT_DCT at 16x32 (spec), regardless of uv_mode.
                    let (mut q, qt) = self.dct.dct16x32_t(&resid, &self.cquant);
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
                    self.rd.preserve_dc(&mut q[0], &resid[..]);
                    *cand_ccf[ci] = q;
                    let rr = self.idct.idct_dequant_16x32(&q, &self.cquant);
                    let sse = self.rd.sse_recon(
                        &cand_pred[ci][..],
                        &rr,
                        &src[..],
                        16,
                        0,
                        0,
                        16,
                        32,
                        self.bd,
                    );
                    cand_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_rect_bits(&q, &SCAN_16X32, 16, 32, plane, cx, py),
                    );
                }
                if ru.is_some() || cand_total < best_total {
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
        }
        if ru.is_none() && chosen_uv == DC_PRED {
            for ci in 0..2 {
                *ccf[ci] = *dc_ccf[ci];
                *cpred_px[ci] = [cpred[ci]; 512];
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
            palette: 0,
        });
        self.push_uv_cf(
            &ccf[0][..],
            &ccf[1][..],
            if use_cfl { cfl_alpha_uv } else { [0, 0] },
        );
        let block_skip = palette.is_none()
            && luma_zero
            && self.rd.all_zero_i32(&ccf[0][..])
            && self.rd.all_zero_i32(&ccf[1][..]);
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
            palette,
            None,
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
                self.idct.idct_dequant_16x32(&ccf[ci], &self.cquant)
            };
            for (ry, rrow) in rr.as_chunks::<16>().0.iter().enumerate() {
                let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                recon_add_pred(drow, &cpred_px[ci][ry * 16..], rrow, maxv);
            }
        }
    }
}
