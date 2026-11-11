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
    fn emit_filter_intra(
        &mut self,
        y_mode: usize,
        width: usize,
        height: usize,
        choice: Option<FilterIntraMode>,
    ) {
        if !filter_intra_allowed(y_mode, width, height) {
            debug_assert!(choice.is_none());
            return;
        }
        let bsize = av1_block_size_index(width, height);
        self.ent.enc.encode_symbol(
            usize::from(choice.is_some()),
            &mut self.ent.cdfs.filter_intra[bsize],
        );
        if let Some(mode) = choice {
            self.ent.enc
                .encode_symbol(mode as usize, &mut self.ent.cdfs.filter_intra_mode);
        }
    }

    #[inline]
    fn luma_filter_type(&self, px: usize, py: usize) -> bool {
        let (bx4, by4) = (px / 4, py / 4);
        (py > 0 && is_smooth_mode(self.ctx.a_mode[bx4] as usize))
            || (px > 0 && is_smooth_mode(self.ctx.l_mode[by4] as usize))
    }

    /// AV1 7.11.2.8 `get_filter_type()` for a chroma transform whose containing
    /// block starts at the luma-sample position `(px, py)`.
    #[inline]
    fn chroma_filter_type(&self, px: usize, py: usize) -> bool {
        let (mi_col, mi_row) = (px / 4, py / 4);
        let mut above_smooth = false;
        if py > 0 {
            let mut c = mi_col;
            if (self.cfg.is_420() || self.cfg.is_422()) && mi_col & 1 == 0 {
                c += 1;
            }
            above_smooth = is_smooth_mode(self.ctx.a_uv_mode[c] as usize);
        }
        let mut left_smooth = false;
        if px > 0 {
            let mut r = mi_row;
            if self.cfg.is_420() && mi_row & 1 == 0 {
                r += 1;
            }
            left_smooth = is_smooth_mode(self.ctx.l_uv_mode[r] as usize);
        }
        above_smooth || left_smooth
    }

    #[inline]
    fn commit_uv_mode(&mut self, px: usize, py: usize, w: usize, h: usize, mode: usize) {
        let (bx4, by4) = (px / 4, py / 4);
        let (w4, h4) = (w.div_ceil(4), h.div_ceil(4));
        let a_end = (bx4 + w4).min(self.ctx.a_uv_mode.len());
        let l_end = (by4 + h4).min(self.ctx.l_uv_mode.len());
        self.ctx.a_uv_mode[bx4..a_end].fill(mode as u8);
        self.ctx.l_uv_mode[by4..l_end].fill(mode as u8);
    }

    /// AOM-style model-RD first stage: rank every permitted luma predictor in
    /// the prediction domain and send only a small speed-dependent shortlist to
    /// transform, inverse-transform, trellis and coefficient-rate evaluation.
    #[allow(clippy::too_many_arguments)]
    fn rank_luma_modes<const N: usize>(
        &self,
        modes: &[usize],
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        have_tr: bool,
        have_bl: bool,
        budget: usize,
    ) -> FixedList<usize, 13> {
        debug_assert_eq!(N, w * h);
        debug_assert!(modes.len() <= 13);
        let mut ranked = FixedList::<(u64, usize), 13>::new((0, DC_PRED));
        // Sparse-residual slot (`Tuning::beam_sparse_slot`): per mode, the
        // centred residual energy E_AC = S2 - S1^2/N is what remains after a
        // single DC coefficient corrects the mean offset. The SATD proxy
        // scores a constant offset c as N*|c| (only the 4x4 Hadamard DC term
        // survives), so a predictor whose error is "right shape, wrong level"
        // is pruned even though one coefficient codes it. The min-E_AC mode
        // takes the LAST shortlist slot when it was pruned.
        // Subsampled formats only.
        let sparse = crate::tuning::get().beam_sparse_slot && (self.cfg.is_420() || self.cfg.is_422());
        let mut best_eac: (i64, usize) = (i64::MAX, DC_PRED);
        for &mode in modes {
            let mut pred = [0i32; N];
            if mode == DC_PRED {
                pred.fill(self.k.intrapred.dc_pred(
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                    self.cfg.bd as i32,
                ));
            } else {
                self.k.intrapred.predict_nd(
                    mode,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.cfg.bd,
                );
            }
            let src = self.src_blk(0, px, py, w, h);
            ranked.push((self.k.rd.satd_sad_proxy(src, &pred, w), mode));
            if sparse {
                let (s1, s2, n) = self.k.rd.residual_moments(src, &pred, w);
                let eac = if n == 0 { 0 } else { s2 - (s1 * s1) / n };
                if (eac, mode) < best_eac {
                    best_eac = (eac, mode);
                }
            }
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(model_cost, mode)| (model_cost, mode));
        let t = 0u64;
        let best = ranked.first().map_or(0, |&(c, _)| c);
        let mut shortlist = FixedList::new(DC_PRED);
        let limit = budget.max(1).min(modes.len());
        for (i, &(cost, mode)) in ranked.iter().enumerate() {
            if i != 0 && t != 0 && cost * 100 > best * t {
                break;
            }
            shortlist.push(mode);
            if shortlist.len() == limit {
                break;
            }
        }
        if sparse && limit < modes.len() && !shortlist.contains(&best_eac.1) {
            let last = shortlist.len() - 1;
            shortlist.as_mut_slice()[last] = best_eac.1;
        }
        shortlist
    }

    /// Syntax emitted for an ordinary luma predictor (excluding coefficients).
    /// Directional delta zero still carries an angle-delta symbol at >=8x8.
    #[allow(clippy::too_many_arguments)]
    fn luma_intra_syntax_bits(
        &self,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        mode: usize,
        delta: i32,
    ) -> f32 {
        let mut bits = self.mode_bits(px, py, mode);
        if is_directional_mode(mode) && w >= 8 && h >= 8 {
            bits += cdf_cost(
                &self.dcdf().angle_delta[mode - V_PRED],
                (delta + 3) as usize,
            );
        } else if filter_intra_allowed(mode, w, h) {
            bits += cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(w, h)], 0);
        }
        bits
    }

    /// Dominant edge direction of the source block as an AV1 prediction angle
    /// (degrees in [0, 180)) and that direction's share of the gradient
    /// energy: a 3-degree orientation histogram of squared central-difference
    /// gradients over the block interior; the peak is the [1 2 1]-smoothed
    /// maximum and the share is its three bins over the total.
    fn luma_gradient_peak(&self, px: usize, py: usize, w: usize, h: usize) -> (f32, f32) {
        const BINS: usize = 60;
        let src = &self.cfg.src[0];
        let stride = self.cfg.w;
        let mut hist = [0f32; BINS];
        let mut total = 0f32;
        for y in py + 1..py + h - 1 {
            for x in px + 1..px + w - 1 {
                let i = y * stride + x;
                let dx = src[i + 1] as f32 - src[i - 1] as f32;
                let dy = src[i + stride] as f32 - src[i - stride] as f32;
                let mag = dx * dx + dy * dy;
                if mag == 0.0 {
                    continue;
                }
                // The edge line is perpendicular to the gradient; image y points
                // down while AV1 angles use y up, so line angle = atan2(-dy, dx) + 90.
                let a = ((-dy).atan2(dx).to_degrees() + 90.0).rem_euclid(180.0);
                let b = ((a / 3.0).round() as usize) % BINS;
                hist[b] += mag;
                total += mag;
            }
        }
        if total <= 0.0 {
            return (0.0, 0.0);
        }
        let mut best = (0usize, -1.0f32);
        for b in 0..BINS {
            let m = hist[(b + BINS - 1) % BINS] + 2.0 * hist[b] + hist[(b + 1) % BINS];
            if m > best.1 {
                best = (b, m);
            }
        }
        let share =
            (hist[(best.0 + BINS - 1) % BINS] + hist[best.0] + hist[(best.0 + 1) % BINS]) / total;
        (best.0 as f32 * 3.0, share)
    }

    /// Preserve the nominal shortlist and add two globally ranked nonzero
    /// angles. A family that loses at delta zero can now reach full RD and the
    /// luma/UV beam. Slow and ExtraSlow run this search (SHIPPED 2026-10-26 with
    /// the gradient gate: 4 photos/synth -1.05 (420) / -1.07 (444), illustrations
    /// -2.70 / -1.59, +4-6% instructions); Medium can probe it with
    /// `joint_angle_search`. The gradient-orientation gate (`angle_hog_gate`,
    /// `angle_hog_win`) keeps the same BD at a third of the ungated cost: an
    /// exhaustive all-pairs oracle measured WORSE than two finalists, so the
    /// gate only drops candidates that never paid.
    #[allow(clippy::too_many_arguments)]
    fn rank_luma_candidates<const N: usize>(
        &self,
        modes: &[usize],
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        have_tr: bool,
        have_bl: bool,
        budget: usize,
    ) -> FixedList<(usize, i32), 64> {
        let mut out = FixedList::new((DC_PRED, 0));
        let nominal = self.rank_luma_modes::<N>(modes, px, py, w, h, have_tr, have_bl, budget);
        for &mode in modes {
            if nominal.contains(&mode) {
                out.push((mode, 0));
            }
        }
        if w < 8
            || h < 8
            || !self.cfg.speed.try_angle_deltas_av1(w.max(h), self.cfg.base_q_idx)
            || !(self.cfg.speed.at_least_slow() || crate::tuning::get().joint_angle_search)
        {
            return out;
        }
        let tune = crate::tuning::get();
        let hog_peak = if tune.angle_hog_gate > 0.0 && !tune.joint_angle_all {
            let (angle, share) = self.luma_gradient_peak(px, py, w, h);
            if share < tune.angle_hog_gate {
                return out;
            }
            Some(angle)
        } else {
            None
        };
        let mut ranked = FixedList::<(u64, f32, usize, i32), 48>::new((0, 0.0, DC_PRED, 0));
        let mut pred = [0i32; N];
        for &mode in modes {
            if !self.angle_delta_refines(mode) {
                continue;
            }
            for delta in [-3, -2, -1, 1, 2, 3] {
                if let Some(peak) = hog_peak {
                    let a = (mode_base_angle(mode) + 3 * delta) as f32;
                    let d = (a - peak).rem_euclid(180.0);
                    if d.min(180.0 - d) > tune.angle_hog_win {
                        continue;
                    }
                }
                self.k.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.ctx.recon[0],
                    self.cfg.w,
                    px,
                    py,
                    w,
                    h,
                    have_tr,
                    have_bl,
                    self.cfg.w,
                    self.cfg.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.cfg.bd,
                );
                let score = self
                    .k.rd
                    .satd_sad_proxy(self.src_blk(0, px, py, w, h), &pred, w);
                let bits = self.luma_intra_syntax_bits(px, py, w, h, mode, delta);
                ranked.push((score, bits, mode, delta));
            }
        }
        // SATD is a screening proxy; exact syntax breaks equal-distortion ties.
        // Full reconstruction RD decides between these and the nominal modes.
        ranked.as_mut_slice().sort_unstable_by(|a, b| {
            a.0.cmp(&b.0)
                .then(a.1.total_cmp(&b.1))
                .then((a.2, a.3).cmp(&(b.2, b.3)))
        });
        let take = if tune.joint_angle_all {
            ranked.len()
        } else {
            tune.joint_angle_count as usize
        };
        for &(_, _, mode, delta) in ranked.iter().take(take) {
            out.push((mode, delta));
        }
        out
    }

    /// Prediction-only shortlist for the expensive Slow filter-intra trials.
    fn rank_filter_intra_modes<const N: usize>(
        &self,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        budget: usize,
        preds: &mut [[i32; N]; 5],
    ) -> FixedList<FilterIntraMode, 5> {
        debug_assert_eq!(N, w * h);
        if budget == 0 {
            return FixedList::new(FILTER_INTRA_MODES[0]);
        }
        // The five predictions are computed once here; the shortlist's are
        // handed back in shortlist order so the full-RD trial does not
        // predict them a second time.
        let mut all = [[0i32; N]; 5];
        let mut ranked =
            FixedList::<(u64, usize, FilterIntraMode), 5>::new((0, 0, FILTER_INTRA_MODES[0]));
        for (order, mode) in FILTER_INTRA_MODES.into_iter().enumerate() {
            let pred = &mut all[order];
            self.k.intrapred.filter_predict(
                mode,
                &self.ctx.recon[0],
                self.cfg.w,
                px,
                py,
                w,
                h,
                pred,
                self.cfg.bd,
            );
            let score = self.k.rd.satd_sad_proxy(self.src_blk(0, px, py, w, h), pred, w);
            ranked.push((score, order, mode));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, order, _)| (score, order));
        let mut shortlist = FixedList::new(FILTER_INTRA_MODES[0]);
        for (si, &(_, order, mode)) in ranked.iter().take(budget).enumerate() {
            shortlist.push(mode);
            preds[si] = all[order];
        }
        shortlist
    }

    /// Prediction-only shortlist for nonzero directional angle deltas. Delta
    /// zero has already been fully evaluated as the base directional mode.
    #[allow(clippy::too_many_arguments)]
    fn rank_angle_deltas<const N: usize>(
        &self,
        mode: usize,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        have_tr: bool,
        have_bl: bool,
        budget: usize,
        pred_bufs: &mut [&mut [i32; N]; 3],
    ) -> FixedList<i32, 6> {
        debug_assert_eq!(N, w * h);
        assert!(budget <= 2);
        let keep = budget;
        if keep == 0 {
            return FixedList::new(0);
        }
        let mut selected = [(u64::MAX, usize::MAX, 0i32); 2];
        for (order, delta) in [-3i32, -2, -1, 1, 2, 3].into_iter().enumerate() {
            self.k.intrapred.predict_nd_ad(
                mode,
                delta,
                &self.ctx.recon[0],
                self.cfg.w,
                px,
                py,
                w,
                h,
                have_tr,
                have_bl,
                self.cfg.w,
                self.cfg.h,
                self.luma_filter_type(px, py),
                pred_bufs[2],
                self.cfg.bd,
            );
            let score = self.k.rd.satd_sad_proxy(self.src_blk(0, px, py, w, h), pred_bufs[2], w);
            let candidate = (score, order, delta);
            if let Some(pos) = selected[..keep]
                .iter()
                .position(|&(old_score, old_order, _)| (score, order) < (old_score, old_order))
            {
                for i in (pos + 1..keep).rev() {
                    selected[i] = selected[i - 1];
                }
                selected[pos] = candidate;
                // Prediction scratch is always slot 2, including for top-1.
                pred_bufs.swap(keep, 2);
                for i in (pos + 1..=keep).rev() {
                    pred_bufs.swap(i - 1, i);
                }
            }
        }
        let mut shortlist = FixedList::new(0);
        for &(_, _, delta) in &selected[..keep] {
            shortlist.push(delta);
        }
        shortlist
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_chroma_modes<const N: usize>(
        &self,
        modes: &[usize],
        px: usize,
        py: usize,
        cx: usize,
        cy: usize,
        w: usize,
        h: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> DirectionalTopK {
        debug_assert_eq!(N, w * h);
        let ch = if self.cfg.is_420() { self.cfg.h / 2 } else { self.cfg.h };
        let mut top = DirectionalTopK::new();
        for &mode in modes {
            if mode != V_PRED && mode != H_PRED && !self.cfg.speed.chroma_angle_directional()
                && is_directional_mode(mode) {
                    continue;
                }
            let mut cost = 0u64;
            for plane in 1..=2 {
                let mut pred = [0i32; N];
                self.k.intrapred.predict_nd(
                    mode,
                    &self.ctx.recon[plane],
                    self.cfg.cw,
                    cx,
                    cy,
                    w,
                    h,
                    have_tr,
                    have_bl,
                    self.cfg.cw,
                    ch,
                    self.chroma_filter_type(px, py),
                    &mut pred,
                    self.cfg.bd,
                );
                cost += self.k.rd.satd_sad_proxy(self.src_blk(plane, cx, cy, w, h), &pred, w);
            }
            top.insert(mode, cost);
        }
        top
    }

    /// Record a square luma block's size (in 4-sample units) into the `blk4`
    /// grid for every 4x4 luma unit it covers. Used by the deblocking filter to
    /// locate block edges and pick filter widths. `(x8,y8)` is the 8-sample
    /// origin; `dim4` is the block size in 4-units (2=8px, 4=16px, 8=32px).
    /// Record a coding block's skip flag into the per-8x8 CDEF map. `dim8` is the
    /// block side in 8-pixel units (8x8 -> 1, 16x16 -> 2, 32x32 -> 4).
    fn mark_skip8(&mut self, x8: usize, y8: usize, dim8: usize, skip: bool) {
        let sb8w = self.cfg.w.div_ceil(8);
        let sb8h = self.cfg.h.div_ceil(8);
        for ry in 0..dim8 {
            for rx in 0..dim8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.ctx.maps.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    /// Rectangular per-8x8 skip mark: `w8` x `h8` 8x8 luma units.
    fn mark_skip8_rect(&mut self, x8: usize, y8: usize, w8: usize, h8: usize, skip: bool) {
        let sb8w = self.cfg.w.div_ceil(8);
        let sb8h = self.cfg.h.div_ceil(8);
        for ry in 0..h8 {
            for rx in 0..w8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.ctx.maps.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    fn record_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        self.memo.emit_epoch.set(self.memo.emit_epoch.get() + 1);
        self.record_tx_blk(x8, y8, dim4);
        let nc4 = self.cfg.w / 4;
        let (bx4, by4) = (x8 * 2, y8 * 2);
        let (d, nr4) = (dim4 as usize, self.cfg.h / 4);
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.ctx.maps.pblk4[r * nc4 + c] = dim4;
                self.ctx.maps.pblk4h[r * nc4 + c] = dim4;
                self.ctx.maps.pblk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Record PREDICTION geometry only, leaving the transform map intact. Used
    /// by BLOCK_64X64, whose four TX_32X32 quadrants are transform subdivisions
    /// of a single 64x64 prediction block.
    fn record_pred_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.cfg.w / 4;
        let (bx4, by4) = (x8 * 2, y8 * 2);
        let (d, nr4) = (dim4 as usize, self.cfg.h / 4);
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.ctx.maps.pblk4[r * nc4 + c] = dim4;
                self.ctx.maps.pblk4h[r * nc4 + c] = dim4;
                self.ctx.maps.pblk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Record TRANSFORM geometry only, leaving the prediction-block map intact.
    /// Used by the TX-split paths, which subdivide a block's transforms without
    /// changing the block itself — chroma still codes one transform over the
    /// whole block, so its deblock edges must not follow the luma subdivision.
    fn record_tx_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.cfg.w / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        let d = dim4 as usize;
        let nr4 = self.cfg.h / 4;
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.ctx.maps.blk4[r * nc4 + c] = dim4;
                self.ctx.maps.blk4h[r * nc4 + c] = dim4;
                self.ctx.maps.blk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.blk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Rectangular block record for deblocking. `w4`/`h4` are the block width and
    /// height in 4-sample units. The luma deblock derives vertical-edge spacing
    /// from the width map (`blk4`) and horizontal-edge spacing from the height
    /// map (`blk4h`), so a non-square block stores its true width and height
    /// separately — no longer the conservative min().
    fn record_blk_rect(&mut self, x8: usize, y8: usize, w4: u8, h4: u8) {
        // A rect emission changes recon, modes, skip and coefficient contexts
        // exactly like a square one: invalidate epoch-keyed partition caches.
        self.memo.emit_epoch.set(self.memo.emit_epoch.get() + 1);
        let nc4 = self.cfg.w / 4;
        let nr4 = self.cfg.h / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        for r in by4..(by4 + h4 as usize).min(nr4) {
            for c in bx4..(bx4 + w4 as usize).min(nc4) {
                self.ctx.maps.blk4[r * nc4 + c] = w4;
                self.ctx.maps.blk4h[r * nc4 + c] = h4;
                self.ctx.maps.blk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.blk4t[r * nc4 + c] = r == by4;
                self.ctx.maps.pblk4[r * nc4 + c] = w4;
                self.ctx.maps.pblk4h[r * nc4 + c] = h4;
                self.ctx.maps.pblk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// 4-sample-granular rectangular block record: H4/V4 strips are 4 px in one
    /// dimension, which the 8px-unit `record_blk_rect` cannot address. Writes
    /// BOTH the transform and prediction maps (strip TX spans the whole strip).
    fn record_blk_rect4(&mut self, bx4: usize, by4: usize, w4: u8, h4: u8) {
        self.memo.emit_epoch.set(self.memo.emit_epoch.get() + 1);
        let nc4 = self.cfg.w / 4;
        let nr4 = self.cfg.h / 4;
        for r in by4..(by4 + h4 as usize).min(nr4) {
            for c in bx4..(bx4 + w4 as usize).min(nc4) {
                self.ctx.maps.blk4[r * nc4 + c] = w4;
                self.ctx.maps.blk4h[r * nc4 + c] = h4;
                self.ctx.maps.blk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.blk4t[r * nc4 + c] = r == by4;
                self.ctx.maps.pblk4[r * nc4 + c] = w4;
                self.ctx.maps.pblk4h[r * nc4 + c] = h4;
                self.ctx.maps.pblk4v[r * nc4 + c] = c == bx4;
                self.ctx.maps.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// What a square node offers the partition decision.
    fn node_kind(&self, n: &Node) -> NodeKind {
        let (w, h) = (self.cfg.w, self.cfg.h);
        let fits = (n.x8 + n.sz8) * 8 <= w && (n.y8 + n.sz8) * 8 <= h;
        match n.sz8 {
            // BL_8X8: always fully in-frame (the coded frame is 8-aligned).
            1 => NodeKind::Full,
            // BL_16X16: content past the midpoint decides. Both ways = the
            // whole 16x16 is in-frame (8-aligned frame, strict test).
            2 => {
                let have_h = (n.x8 + 1) * 8 < w;
                let have_v = (n.y8 + 1) * 8 < h;
                if have_h && have_v {
                    NodeKind::Full
                } else if have_h != have_v {
                    NodeKind::Edge16 { horz: have_h }
                } else {
                    NodeKind::SplitOnly
                }
            }
            4 if fits => NodeKind::Full,
            // BL_64X64: 4:0:0 has no whole-64 chroma path and keeps splitting.
            8 if fits && !self.cfg.is_mono() => NodeKind::Full,
            _ => NodeKind::SplitOnly,
        }
    }

    /// Step 1 of a node: decide. Read-only: a proxy partition choice, a
    /// choice already `forced` on this node, or the verdict that the node
    /// must be trial-encoded.
    fn decide_node(&self, n: &Node, kind: NodeKind, forced: Option<Part16>) -> NodeChoice {
        let Node { x8, y8, sz8, thr, lhb, .. } = *n;
        let (px, py, dim) = (x8 * 8, y8 * 8, sz8 * 8);
        match kind {
            NodeKind::SplitOnly => {
                debug_assert!(forced.is_none(), "forced {forced:?} on a node with no choice");
                NodeChoice::Part(Part16::Split)
            }
            // Exactly one frame edge crosses this node: the spec codes a
            // boolean, SPLIT vs HORZ (edge below) / VERT (edge right).
            NodeKind::Edge16 { horz } => {
                debug_assert!(forced.is_none(), "forced {forced:?} on an edge node");
                NodeChoice::Part(self.edge_choice_16(x8, y8, thr, lhb, horz))
            }
            NodeKind::Full => {
                if let Some(part) = forced {
                    return NodeChoice::Part(part);
                }
                // Real-R-D (trial-encode) levels: the proxy still runs first,
                // and decides whether its verdict stands or both legs get coded.
                let trial = sz8 > 1 && self.trial_level_on(sz8) && self.trial_src_gate(px, py, dim);
                let gate = crate::tuning::get().trial_gate;
                match sz8 {
                    1 => NodeChoice::Part(self.choose_8(n)),
                    // NONE vs bottom-up SPLIT. Under trial the proxy only
                    // arbitrates the whole-32 IntraBC copy where it exists,
                    // unless `trial_gate` / `trial_margin` let a verdict stand.
                    4 => {
                        if trial {
                            let margin_on = crate::tuning::get().trial_margin > 0.0;
                            if !(self.cfg.allow_intrabc || gate != 0 || margin_on) {
                                return NodeChoice::Trial { skip_none: false };
                            }
                            let v = self.choose_rect32(x8, y8, thr, lhb);
                            let stands = !Self::trial_may_overturn(v.part, gate)
                                || !self.trial_margin_ok(v.margin, px, py, 32);
                            return if stands {
                                NodeChoice::Part(v.part)
                            } else {
                                NodeChoice::Trial { skip_none: false }
                            };
                        }
                        NodeChoice::Part(self.choose_rect32(x8, y8, thr, lhb).part)
                    }
                    // A non-square proxy winner (rects, quads, IntraBC) is
                    // kept as decided unless the rect-vs-split trial takes it.
                    2 => {
                        let proxy = self.partition_choice_16(x8, y8, thr, lhb);
                        if trial {
                            if Self::trial_may_overturn(proxy, gate)
                                && self.trial_margin_ok(self.proxy_margin_16(x8, y8, thr, lhb), px, py, 16)
                            {
                                return NodeChoice::Trial { skip_none: false };
                            }
                            if self.trial_rect_vs_split(proxy, gate, 2) {
                                return NodeChoice::Trial { skip_none: true };
                            }
                        }
                        NodeChoice::Part(proxy)
                    }
                    // Whole-64 vs SPLIT; rect64 / IntraBC verdicts stand.
                    _ => {
                        let proxy = self.choose_64(x8, y8, thr, lhb);
                        if trial && Self::trial_may_overturn(proxy, gate) {
                            return NodeChoice::Trial { skip_none: false };
                        }
                        NodeChoice::Part(proxy)
                    }
                }
            }
        }
    }

    /// BL_8X8 decision: an IntraBC copy, the 8x8 leaf, its two rects, or
    /// four 4x4 (`Part16::Split`).
    fn choose_8(&self, n: &Node) -> Part16 {
        let Node { x8, y8, thr, .. } = *n;
        let (px, py) = (x8 * 8, y8 * 8);
        let (tr, bl) = n.leaf_edges(self.cfg.w, self.cfg.h);
        // 8x8 IntraBC copy (glyph-sized repeats on text/UI): taken when it
        // beats both the equipped NONE leaf and split4.
        if self.cfg.allow_intrabc
            && let Some(ibc) = self.rd_cost_intrabc(px, py, 8, thr, self.perceptual_rd_scale(px, py, 8))
        {
            let prdo = self.perceptual_rd_scale(px, py, 8);
            let mut alt = self.rd_cost_square(px, py, 8, tr, bl, prdo);
            if !self.cfg.is_mono() && ibc < alt {
                alt = alt.min(self.rd_cost_split4_luma(px, py, self.trellis_lambda() * prdo, self.mlam() * prdo));
            }
            if ibc < alt {
                return Part16::Intrabc;
            }
        }
        let want_split = !self.cfg.is_mono() && !self.prefer_8x8_none(x8, y8, tr, bl);
        if want_split {
            Part16::Split
        } else {
            self.choose_rect8(x8, y8, tr, bl)
        }
    }

    /// The partition symbol's CDF row for a node: dav1d's context from the
    /// above/left partition bytes.
    #[inline]
    fn partition_ctx(&self, n: &Node) -> usize {
        get_partition_ctx(&self.ctx.a_part, &self.ctx.l_part, n.bl, n.x8, n.y8)
    }

    /// Emit an adapting partition symbol for the node.
    fn emit_partition(&mut self, n: &Node, symbol: usize) {
        let ctx = self.partition_ctx(n);
        let cdf = if n.sz8 == 1 {
            &mut self.ent.cdfs.part_bl8[ctx]
        } else {
            &mut self.ent.cdfs.part_split[n.bl - 1][ctx]
        };
        self.ent.enc.encode_symbol(symbol, cdf);
    }

    /// Store the partition context bytes a coded node leaves for its right
    /// and lower neighbours, clipped to the in-frame entries (the
    /// out-of-frame ones are never read).
    fn set_partition_ctx(&mut self, n: &Node, (above, left): (u8, u8)) {
        let ae = (n.x8 + n.sz8).min(self.ctx.a_part.len());
        let le = (n.y8 + n.sz8).min(self.ctx.l_part.len());
        self.ctx.a_part[n.x8..ae].fill(above);
        self.ctx.l_part[n.y8..le].fill(left);
    }

    /// Code the square node of `sz8` 8px units at (x8, y8): decide its
    /// partition, emit it, and code its leaves or recurse into its children.
    /// `forced` pins THIS node's partition (a trial-encode leg); children
    /// always decide for themselves.
    #[allow(clippy::too_many_arguments)]
    fn decode_sb(
        &mut self,
        bl: usize,
        x8: usize,
        y8: usize,
        sz8: usize,
        thr: bool,
        lhb: bool,
        forced: Option<Part16>,
    ) {
        let n = Node { bl, x8, y8, sz8, thr, lhb };
        if sz8 == 8 && crate::tuning::get().edge_fill && self.edge_in_flat(x8 * 8, y8 * 8, 64) {
            self.fill_recon_estimate_region(x8 * 8, y8 * 8, 64, 64);
        }
        let kind = self.node_kind(&n);
        match self.decide_node(&n, kind, forced) {
            // Codes every leg for real and commits the winner: fully emitted.
            NodeChoice::Trial { skip_none } => self.trial_partition(bl, x8, y8, sz8, thr, lhb, skip_none),
            // The 8x8 node's SPLIT is a leaf (four 4x4), not a recursion.
            NodeChoice::Part(Part16::Split) if sz8 > 1 => self.code_split(&n),
            NodeChoice::Part(part) => match kind {
                NodeKind::Edge16 { horz } => self.code_edge16(&n, horz),
                _ => self.code_partition(&n, part),
            },
        }
    }

    /// Step 2 of a decided node: the partition symbol, its leaves, and the
    /// partition context it leaves behind.
    fn code_partition(&mut self, n: &Node, part: Part16) {
        let Some(ctx_bytes) = part_ctx_bytes(n.sz8, part) else {
            // A partition this level has no emitter for. Only the 64-level
            // proxy can name one; it falls back to SPLIT as before.
            assert!(n.sz8 == 8, "disabled {}x{} partition {part:?}", n.sz8 * 8, n.sz8 * 8);
            return self.code_split(n);
        };
        self.emit_partition(n, part.symbol());
        self.code_leaves(n, part);
        self.set_partition_ctx(n, ctx_bytes);
    }

    /// The leaves of `part` at node `n`, in coding order.
    fn code_leaves(&mut self, n: &Node, part: Part16) {
        let Node { x8, y8, sz8, thr, lhb, .. } = *n;
        let (w, h) = (self.cfg.w, self.cfg.h);
        let (have_tr, have_bl) = n.leaf_edges(w, h);
        match (sz8, part) {
            // --- BL_8X8
            (1, Part16::Intrabc) => self.code_intrabc_block(x8, y8, 8, thr),
            (1, Part16::None) => self.code_block(x8, y8, have_tr, have_bl),
            (1, Part16::Split) => self.code_block_split4_dc(x8, y8, have_tr, have_bl),
            (1, Part16::Horz) => self.code_block8_rect(x8, y8, false, have_tr, have_bl),
            (1, Part16::Vert) => self.code_block8_rect(x8, y8, true, have_tr, have_bl),
            // --- BL_16X16: one TX_16X16 block, two rects, four strips, or a
            // T-shape (two 8x8 plus one 16x8 / 8x16). Chroma: 16x16 (4:4:4),
            // TX_8X8 (4:2:0), RTX_8X16 (4:2:2).
            (2, Part16::Intrabc) => self.code_intrabc_block(x8, y8, 16, thr),
            (2, Part16::None) => self.code_block16(x8, y8, have_tr, have_bl),
            (2, Part16::Horz) => {
                if self.cfg.is_420() {
                    self.code_block16_rect_420(x8, y8, false, thr, lhb);
                } else if self.cfg.is_422() {
                    self.code_block16_horz_422(x8, y8, 2, thr, lhb);
                } else {
                    self.code_block16_horz_444(x8, y8, 2, thr, lhb);
                }
            }
            (2, Part16::Vert) => {
                if self.cfg.is_420() {
                    self.code_block16_rect_420(x8, y8, true, thr, lhb);
                } else {
                    self.code_block16_vert_444(x8, y8, 2, thr, lhb);
                }
            }
            (2, Part16::Horz4) => self.code_block16_quad(x8, y8, false, thr, lhb),
            (2, Part16::Vert4) => self.code_block16_quad(x8, y8, true, thr, lhb),
            (2, Part16::HorzA) => {
                self.code_block(x8, y8, y8 > 0 && (x8 * 8 + 8) < w, x8 > 0 && (y8 * 8 + 8) < h);
                self.code_block(x8 + 1, y8, thr && y8 > 0 && (x8 * 8 + 16) < w, false);
                let (tr, bl) = self.rect_edge_flags(x8 * 8, y8 * 8 + 8, false, true, thr, lhb);
                self.code_block16_rect_leaf_420(x8, y8 + 1, false, tr, bl);
            }
            (2, Part16::HorzB) => {
                let (tr, bl) = self.rect_edge_flags(x8 * 8, y8 * 8, false, false, thr, lhb);
                self.code_block16_rect_leaf_420(x8, y8, false, tr, bl);
                self.code_block(x8, y8 + 1, (x8 * 8 + 8) < w, lhb && x8 > 0 && (y8 * 8 + 16) < h);
                self.code_block(x8 + 1, y8 + 1, false, false);
            }
            (2, Part16::VertA) => {
                self.code_block(x8, y8, y8 > 0 && (x8 * 8 + 8) < w, x8 > 0 && (y8 * 8 + 8) < h);
                self.code_block(x8, y8 + 1, false, lhb && x8 > 0 && (y8 * 8 + 16) < h);
                let (tr, bl) = self.rect_edge_flags(x8 * 8 + 8, y8 * 8, true, true, thr, lhb);
                self.code_block16_rect_leaf_420(x8 + 1, y8, true, tr, bl);
            }
            (2, Part16::VertB) => {
                let (tr, bl) = self.rect_edge_flags(x8 * 8, y8 * 8, true, false, thr, lhb);
                self.code_block16_rect_leaf_420(x8, y8, true, tr, bl);
                self.code_block(x8 + 1, y8, thr && y8 > 0 && (x8 * 8 + 16) < w, true);
                self.code_block(x8 + 1, y8 + 1, false, false);
            }
            // --- BL_32X32 (the 32-level rects are disabled).
            (4, Part16::Intrabc) => self.code_intrabc_block(x8, y8, 32, thr),
            (4, Part16::None) => self.code_block32(x8, y8, have_tr, have_bl),
            // --- BL_64X64
            (8, Part16::Intrabc) => self.code_block64_intrabc(x8, y8, thr),
            (8, Part16::None) => self.code_block64(x8, y8, have_tr, have_bl),
            (8, Part16::Horz) => self.code_block64_rect(x8, y8, false, thr, lhb),
            (8, Part16::Vert) => self.code_block64_rect(x8, y8, true, thr, lhb),
            _ => unreachable!("no {}x{} emitter for {part:?}", sz8 * 8, sz8 * 8),
        }
    }

    /// A 16x16 node crossed by one frame edge that takes its rect (`horz` =
    /// HORZ, edge below; else VERT, edge right) instead of SPLIT. The rect's
    /// first half is fully in-frame (8-aligned coded frame); its second half
    /// is never coded.
    fn code_edge16(&mut self, n: &Node, horz: bool) {
        let Node { bl, x8, y8, thr, lhb, .. } = *n;
        let ctx = self.partition_ctx(n);
        self.ent.enc
            .encode_gathered_partition(horz, false, &self.ent.cdfs.part_split[bl - 1][ctx]);
        if self.cfg.is_420() {
            let (tr, bl) = self.rect_edge_flags(x8 * 8, y8 * 8, !horz, false, thr, lhb);
            self.code_block16_rect_leaf_420(x8, y8, !horz, tr, bl);
        } else if !horz {
            self.code_block16_vert_444(x8, y8, 1, thr, lhb);
        } else if self.cfg.is_422() {
            self.code_block16_horz_422(x8, y8, 1, thr, lhb);
        } else {
            self.code_block16_horz_444(x8, y8, 1, thr, lhb);
        }
        // dav1d `al_part_ctx` bl16 h / v.
        self.set_partition_ctx(n, if horz { (0x1c, 0x1e) } else { (0x1e, 0x1c) });
    }

    /// PARTITION_SPLIT of a node above 8x8: the split symbol, then the four
    /// children in z-order.
    fn code_split(&mut self, n: &Node) {
        let Node { bl, x8, y8, sz8, thr, lhb } = *n;
        let hh = sz8 / 2;
        // content past the horizontal / vertical midpoint of this block?
        let have_h = (x8 + hh) * 8 < self.cfg.w;
        let have_v = (y8 + hh) * 8 < self.cfg.h;
        let ctx = self.partition_ctx(n);
        if have_h && have_v {
            // full PARTITION_SPLIT symbol -> adapt the partition CDF (dav1d uses
            // decode_symbol_adapt here).
            self.ent.enc
                .encode_symbol(3, &mut self.ent.cdfs.part_split[bl - 1][ctx]);
        } else if have_h {
            // edge: dav1d codes a NON-adapting bool from a gathered probability;
            // read the live (possibly already-adapted) icdf CDF, do not adapt it.
            self.ent.enc
                .encode_gathered_partition(true, true, &self.ent.cdfs.part_split[bl - 1][ctx]);
        } else if have_v {
            self.ent.enc
                .encode_gathered_partition(false, true, &self.ent.cdfs.part_split[bl - 1][ctx]);
        }
        // else: neither -> implicit split, no symbol
        // recurse the children whose top-left is in-frame, propagating the
        // intra-edge availability (dav1d intra-edge tree). z-order child index
        // n: 0=TL,1=TR,2=BL,3=BR -> (top_has_right, left_has_bottom):
        //   TL=(1,1)  TR=(parent_thr,0)  BL=(1,parent_lhb)  BR=(0,0)
        let children = [
            (x8, y8, true, true),
            (x8 + hh, y8, thr, false),
            (x8, y8 + hh, true, lhb),
            (x8 + hh, y8 + hh, false, false),
        ];
        for (cx, cy, cthr, clhb) in children {
            if cx * 8 < self.cfg.w && cy * 8 < self.cfg.h {
                self.decode_sb(bl + 1, cx, cy, hh, cthr, clhb, None);
            }
        }
    }
}

/// One square node of the superblock partition tree.
#[derive(Clone, Copy)]
struct Node {
    /// dav1d block level: 1 = 64x64 ... 4 = 8x8.
    bl: usize,
    /// Position and size in 8-pixel units.
    x8: usize,
    y8: usize,
    sz8: usize,
    /// Intra-edge availability inherited down the tree: the top row has
    /// samples to its right / the left column has samples below.
    thr: bool,
    lhb: bool,
}

impl Node {
    /// (top-right, bottom-left) edge availability of the leaf that covers the
    /// whole node, in a `w` x `h` tile.
    #[inline]
    fn leaf_edges(&self, w: usize, h: usize) -> (bool, bool) {
        let dim = self.sz8 * 8;
        (
            self.thr && self.y8 > 0 && (self.x8 * 8 + dim) < w,
            self.lhb && self.x8 > 0 && (self.y8 * 8 + dim) < h,
        )
    }
}

/// The decision a square node offers.
#[derive(Clone, Copy, PartialEq, Eq)]
enum NodeKind {
    /// Fully in-frame, with a real partition choice.
    Full,
    /// A 16x16 node crossed by exactly one frame edge: SPLIT or the one rect
    /// the edge allows.
    Edge16 { horz: bool },
    /// Crossed by the frame edge (or without a leaf at this size): must split.
    SplitOnly,
}

/// Outcome of a node's decision step.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum NodeChoice {
    /// Take this partition.
    Part(Part16),
    /// Trial-encode the node's legs for real; `skip_none` leaves NONE out.
    Trial { skip_none: bool },
}

impl Part16 {
    /// The AV1 `partition` symbol value.
    #[inline]
    fn symbol(self) -> usize {
        match self {
            // An IntraBC copy is a PARTITION_NONE block.
            Part16::Intrabc | Part16::None => 0,
            Part16::Horz => 1,
            Part16::Vert => 2,
            Part16::Split => 3,
            Part16::HorzA => 4,
            Part16::HorzB => 5,
            Part16::VertA => 6,
            Part16::VertB => 7,
            Part16::Horz4 => 8,
            Part16::Vert4 => 9,
        }
    }
}

/// dav1d `al_part_ctx`: the (above, left) partition-context bytes a node of
/// `sz8` 8px units leaves after coding `part`. `None` = this level has no
/// emitter for that partition.
fn part_ctx_bytes(sz8: usize, part: Part16) -> Option<(u8, u8)> {
    use Part16::*;
    Some(match (sz8, part) {
        (1, Intrabc | None) => (0x1e, 0x1e),
        (1, Horz) => (0x1e, 0x1f),
        (1, Vert) => (0x1f, 0x1e),
        (1, Split) => (0x1f, 0x1f),
        (2, Intrabc | None) => (0x1c, 0x1c),
        (2, Horz | HorzA) => (0x1c, 0x1e),
        (2, Vert | VertA) => (0x1e, 0x1c),
        (2, HorzB | VertB) => (0x1e, 0x1e),
        (2, Horz4) => (0x1c, 0x1f),
        (2, Vert4) => (0x1f, 0x1c),
        (4, Intrabc | None) => (0x18, 0x18),
        (8, Intrabc | None) => (0x10, 0x10),
        (8, Horz) => (0x10, 0x18),
        (8, Vert) => (0x18, 0x10),
        _ => return Option::None,
    })
}

/// Spec `Mode_To_Angle`: the nominal prediction angle of a directional mode.
fn mode_base_angle(mode: usize) -> i32 {
    match mode {
        V_PRED => 90,
        H_PRED => 180,
        D45_PRED => 45,
        D135_PRED => 135,
        D113_PRED => 113,
        D157_PRED => 157,
        D203_PRED => 203,
        VERT_LEFT_PRED => 67,
        _ => unreachable!("not a directional mode"),
    }
}
