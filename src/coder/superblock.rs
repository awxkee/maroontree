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
        self.enc.encode_symbol(
            usize::from(choice.is_some()),
            &mut self.cdfs.filter_intra[bsize],
        );
        if let Some(mode) = choice {
            self.enc
                .encode_symbol(mode as usize, &mut self.cdfs.filter_intra_mode);
        }
    }

    #[inline]
    fn luma_filter_type(&self, px: usize, py: usize) -> bool {
        let (bx4, by4) = (px / 4, py / 4);
        (py > 0 && is_smooth_mode(self.a_mode[bx4] as usize))
            || (px > 0 && is_smooth_mode(self.l_mode[by4] as usize))
    }

    /// AV1 7.11.2.8 `get_filter_type()` for a chroma transform whose containing
    /// block starts at the luma-sample position `(px, py)`.
    #[inline]
    fn chroma_filter_type(&self, px: usize, py: usize) -> bool {
        let (mi_col, mi_row) = (px / 4, py / 4);
        let mut above_smooth = false;
        if py > 0 {
            let mut c = mi_col;
            if (self.ss420 || self.ss422) && mi_col & 1 == 0 {
                c += 1;
            }
            above_smooth = is_smooth_mode(self.a_uv_mode[c] as usize);
        }
        let mut left_smooth = false;
        if px > 0 {
            let mut r = mi_row;
            if self.ss420 && mi_row & 1 == 0 {
                r += 1;
            }
            left_smooth = is_smooth_mode(self.l_uv_mode[r] as usize);
        }
        above_smooth || left_smooth
    }

    #[inline]
    fn commit_uv_mode(&mut self, px: usize, py: usize, w: usize, h: usize, mode: usize) {
        let (bx4, by4) = (px / 4, py / 4);
        let (w4, h4) = (w.div_ceil(4), h.div_ceil(4));
        let a_end = (bx4 + w4).min(self.a_uv_mode.len());
        let l_end = (by4 + h4).min(self.l_uv_mode.len());
        self.a_uv_mode[bx4..a_end].fill(mode as u8);
        self.l_uv_mode[by4..l_end].fill(mode as u8);
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
        const MID32_FLAT_ONLY: bool = false;
        let deny_dir_32 = MID32_FLAT_ONLY
            && N == 1024
            && (41..=143).contains(&(self.aq.base_q as i32));
        for &mode in modes {
            if deny_dir_32 && (1..=8).contains(&mode) {
                continue;
            }
            let mut pred = [0i32; N];
            if mode == DC_PRED {
                pred.fill(self.intrapred.dc_pred(
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    self.bd as i32,
                ));
            } else {
                self.intrapred.predict_nd(
                    mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    w,
                    h,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let src = &self.src[0][py * self.w + px..];
            ranked.push((self.rd.satd_sad_proxy(src, self.w, &pred, w, w, h), mode));
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
        shortlist
    }

    /// Prediction-only shortlist for the expensive Slow filter-intra trials.
    fn rank_filter_intra_modes<const N: usize>(
        &self,
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        budget: usize,
    ) -> FixedList<FilterIntraMode, 5> {
        debug_assert_eq!(N, w * h);
        let mut ranked =
            FixedList::<(u64, usize, FilterIntraMode), 5>::new((0, 0, FILTER_INTRA_MODES[0]));
        for (order, mode) in FILTER_INTRA_MODES.into_iter().enumerate() {
            let mut pred = [0i32; N];
            self.intrapred.filter_predict(
                mode,
                &self.recon[0],
                self.w,
                px,
                py,
                w,
                h,
                &mut pred,
                self.bd,
            );
            let score = self.rd.satd_sad_proxy(
                &self.src[0][py * self.w + px..],
                self.w,
                &pred,
                w,
                w,
                h,
            );
            ranked.push((score, order, mode));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, order, _)| (score, order));
        let mut shortlist = FixedList::new(FILTER_INTRA_MODES[0]);
        for &(_, _, mode) in ranked.iter().take(budget) {
            shortlist.push(mode);
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
        debug_assert!(budget <= 2);
        let keep = budget.max(1);
        let mut selected = [(u64::MAX, usize::MAX, 0i32); 2];
        for (order, delta) in [-3i32, -2, -1, 1, 2, 3].into_iter().enumerate() {
            self.intrapred.predict_nd_ad(
                mode,
                delta,
                &self.recon[0],
                self.w,
                px,
                py,
                w,
                h,
                have_tr,
                have_bl,
                self.w,
                self.h,
                self.luma_filter_type(px, py),
                pred_bufs[2],
                self.bd,
            );
            let score = self.rd.satd_sad_proxy(
                &self.src[0][py * self.w + px..],
                self.w,
                pred_bufs[2],
                w,
                w,
                h,
            );
            let candidate = (score, order, delta);
            if let Some(pos) = selected[..keep]
                .iter()
                .position(|&(old_score, old_order, _)| (score, order) < (old_score, old_order))
            {
                for i in (pos + 1..keep).rev() {
                    selected[i] = selected[i - 1];
                }
                selected[pos] = candidate;
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
    ) -> DirectionalTopK {
        debug_assert_eq!(N, w * h);
        let mut top = DirectionalTopK::new();
        for &mode in modes {
            if mode != V_PRED && mode != H_PRED && !self.speed.chroma_angle_directional()
                && is_directional_mode(mode) {
                    continue;
                }
            let mut cost = 0u64;
            for plane in 1..=2 {
                let mut pred = [0i32; N];
                self.intrapred.predict_nd(
                    mode,
                    &self.recon[plane],
                    self.cw,
                    cx,
                    cy,
                    w,
                    h,
                    false,
                    false,
                    self.cw,
                    self.h,
                    self.chroma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
                let src = &self.src[plane][cy * self.cw + cx..];
                cost += self.rd.satd_sad_proxy(src, self.cw, &pred, w, w, h);
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
        let sb8w = self.w.div_ceil(8);
        let sb8h = self.h.div_ceil(8);
        for ry in 0..dim8 {
            for rx in 0..dim8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    /// Rectangular per-8x8 skip mark: `w8` x `h8` 8x8 luma units.
    fn mark_skip8_rect(&mut self, x8: usize, y8: usize, w8: usize, h8: usize, skip: bool) {
        let sb8w = self.w.div_ceil(8);
        let sb8h = self.h.div_ceil(8);
        for ry in 0..h8 {
            for rx in 0..w8 {
                let (bx, by) = (x8 + rx, y8 + ry);
                if bx < sb8w && by < sb8h {
                    self.skip8[by * sb8w + bx] = skip;
                }
            }
        }
    }

    fn record_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        self.emit_epoch.set(self.emit_epoch.get() + 1);
        self.record_tx_blk(x8, y8, dim4);
        let nc4 = self.w / 4;
        let (bx4, by4) = (x8 * 2, y8 * 2);
        let (d, nr4) = (dim4 as usize, self.h / 4);
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.pblk4[r * nc4 + c] = dim4;
                self.pblk4h[r * nc4 + c] = dim4;
                self.pblk4v[r * nc4 + c] = c == bx4;
                self.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Record PREDICTION geometry only, leaving the transform map intact. Used
    /// by BLOCK_64X64, whose four TX_32X32 quadrants are transform subdivisions
    /// of a single 64x64 prediction block.
    fn record_pred_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.w / 4;
        let (bx4, by4) = (x8 * 2, y8 * 2);
        let (d, nr4) = (dim4 as usize, self.h / 4);
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.pblk4[r * nc4 + c] = dim4;
                self.pblk4h[r * nc4 + c] = dim4;
                self.pblk4v[r * nc4 + c] = c == bx4;
                self.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Record TRANSFORM geometry only, leaving the prediction-block map intact.
    /// Used by the TX-split paths, which subdivide a block's transforms without
    /// changing the block itself — chroma still codes one transform over the
    /// whole block, so its deblock edges must not follow the luma subdivision.
    fn record_tx_blk(&mut self, x8: usize, y8: usize, dim4: u8) {
        let nc4 = self.w / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        let d = dim4 as usize;
        let nr4 = self.h / 4;
        for r in by4..(by4 + d).min(nr4) {
            for c in bx4..(bx4 + d).min(nc4) {
                self.blk4[r * nc4 + c] = dim4;
                self.blk4h[r * nc4 + c] = dim4;
                self.blk4v[r * nc4 + c] = c == bx4;
                self.blk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// Rectangular block record for deblocking. `w4`/`h4` are the block width and
    /// height in 4-sample units. The luma deblock derives vertical-edge spacing
    /// from the width map (`blk4`) and horizontal-edge spacing from the height
    /// map (`blk4h`), so a non-square block stores its true width and height
    /// separately — no longer the conservative min().
    fn record_blk_rect(&mut self, x8: usize, y8: usize, w4: u8, h4: u8) {
        let nc4 = self.w / 4;
        let nr4 = self.h / 4;
        let bx4 = x8 * 2;
        let by4 = y8 * 2;
        for r in by4..(by4 + h4 as usize).min(nr4) {
            for c in bx4..(bx4 + w4 as usize).min(nc4) {
                self.blk4[r * nc4 + c] = w4;
                self.blk4h[r * nc4 + c] = h4;
                self.blk4v[r * nc4 + c] = c == bx4;
                self.blk4t[r * nc4 + c] = r == by4;
                self.pblk4[r * nc4 + c] = w4;
                self.pblk4h[r * nc4 + c] = h4;
                self.pblk4v[r * nc4 + c] = c == bx4;
                self.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    /// 4-sample-granular rectangular block record: H4/V4 strips are 4 px in one
    /// dimension, which the 8px-unit `record_blk_rect` cannot address. Writes
    /// BOTH the transform and prediction maps (strip TX spans the whole strip).
    fn record_blk_rect4(&mut self, bx4: usize, by4: usize, w4: u8, h4: u8) {
        self.emit_epoch.set(self.emit_epoch.get() + 1);
        let nc4 = self.w / 4;
        let nr4 = self.h / 4;
        for r in by4..(by4 + h4 as usize).min(nr4) {
            for c in bx4..(bx4 + w4 as usize).min(nc4) {
                self.blk4[r * nc4 + c] = w4;
                self.blk4h[r * nc4 + c] = h4;
                self.blk4v[r * nc4 + c] = c == bx4;
                self.blk4t[r * nc4 + c] = r == by4;
                self.pblk4[r * nc4 + c] = w4;
                self.pblk4h[r * nc4 + c] = h4;
                self.pblk4v[r * nc4 + c] = c == bx4;
                self.pblk4t[r * nc4 + c] = r == by4;
            }
        }
    }

    fn decode_sb(&mut self, bl: usize, x8: usize, y8: usize, sz8: usize, thr: bool, lhb: bool) {
        if sz8 == 1 {
            // BL_8X8 leaf (always fully in-frame for multiple-of-8 dimensions):
            // emit PARTITION_NONE, then the block. When the split scaffold is
            // forced (test-only), emit PARTITION_SPLIT and code four BLOCK_4X4.
            let ctx = get_partition_ctx(&self.a_part, &self.l_part, 4, x8, y8);
            let r8_tr = thr && y8 > 0 && (x8 * 8 + 8) < self.w;
            let r8_bl = lhb && x8 > 0 && (y8 * 8 + 8) < self.h;
            let r8 = self.part_decision(|t| {
                let split_eligible = !t.mono;
                let want_split = split_eligible
                    && (FORCE_SPLIT4.load(std::sync::atomic::Ordering::Relaxed)
                        || (SPLIT4_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
                            && !t.prefer_8x8_none(x8, y8)));
                if want_split {
                    Part16::Split
                } else {
                    t.choose_rect8(x8, y8, r8_tr, r8_bl)
                }
            });
            if r8 == Part16::Split {
                self.enc.encode_symbol(3, &mut self.cdfs.part_bl8[ctx]); // SPLIT
                let have_tr = thr && y8 > 0 && (x8 * 8 + 8) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 8) < self.h;
                self.code_block_split4_dc(x8, y8, have_tr, have_bl);
                self.a_part[x8] = 0x1f;
                self.l_part[y8] = 0x1f;
                return;
            }
            if r8 == Part16::Horz {
                self.enc.encode_symbol(1, &mut self.cdfs.part_bl8[ctx]);
                self.code_block8_rect(x8, y8, false, r8_tr, r8_bl);
                self.a_part[x8] = 0x1e;
                self.l_part[y8] = 0x1f;
                return;
            }
            if r8 == Part16::Vert {
                self.enc.encode_symbol(2, &mut self.cdfs.part_bl8[ctx]);
                self.code_block8_rect(x8, y8, true, r8_tr, r8_bl);
                self.a_part[x8] = 0x1f;
                self.l_part[y8] = 0x1e;
                return;
            }
            self.enc.encode_symbol(0, &mut self.cdfs.part_bl8[ctx]);
            let have_tr = thr && y8 > 0 && (x8 * 8 + 8) < self.w;
            let have_bl = lhb && x8 > 0 && (y8 * 8 + 8) < self.h;
            self.code_block(x8, y8, have_tr, have_bl);
            self.a_part[x8] = 0x1e;
            self.l_part[y8] = 0x1e;
            return;
        }
        // BL_32X32: choose the preselected square geometry or a legal
        // rectangular partition. Chroma geometry follows that shared choice.
        // Requires the full 32x32 in-frame.
        if sz8 == 4 {
            let full_h = (x8 + 4) * 8 <= self.w;
            let full_v = (y8 + 4) * 8 <= self.h;
            if full_h && full_v {
                let choice = self.part_decision(|t| t.choose_rect32(x8, y8, thr, lhb));
                let have_tr = thr && y8 > 0 && (x8 * 8 + 32) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 32) < self.h;
                match choice {
                    Part16::Intrabc => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_intrabc_block(x8, y8, 32);
                        self.a_part[x8..x8 + 4].fill(0x18);
                        self.l_part[y8..y8 + 4].fill(0x18);
                        return;
                    }
                    Part16::None => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block32(x8, y8, have_tr, have_bl);
                        self.a_part[x8..x8 + 4].fill(0x18);
                        self.l_part[y8..y8 + 4].fill(0x18);
                        return;
                    }
                    Part16::Horz => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(1, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block32_rect(x8, y8, false);
                        self.a_part[x8..x8 + 4].fill(0x18);
                        self.l_part[y8..y8 + 4].fill(0x1c);
                        return;
                    }
                    Part16::Vert => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(2, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block32_rect(x8, y8, true);
                        self.a_part[x8..x8 + 4].fill(0x1c);
                        self.l_part[y8..y8 + 4].fill(0x18);
                        return;
                    }
                    // A/B T-shapes: two BLOCK_16X16 leaves + one 32x16/16x32
                    // rect leaf, z-order. Child edge flags mirror the BL16
                    // A/B arms scaled to 16px children; al_part_ctx values are
                    // dav1d's bl32 tts/tbs/tls/trs rows.
                    Part16::HorzA => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(4, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block16(
                            x8,
                            y8,
                            y8 > 0 && (x8 * 8 + 16) < self.w,
                            x8 > 0 && (y8 * 8 + 16) < self.h,
                        );
                        self.code_block16(
                            x8 + 2,
                            y8,
                            thr && y8 > 0 && (x8 * 8 + 32) < self.w,
                            false,
                        );
                        self.code_block32_rect_halves(x8, y8, false, 1..2);
                        self.a_part[x8..x8 + 4].fill(0x18);
                        self.l_part[y8..y8 + 4].fill(0x1c);
                        return;
                    }
                    Part16::HorzB => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(5, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block32_rect_halves(x8, y8, false, 0..1);
                        self.code_block16(
                            x8,
                            y8 + 2,
                            (x8 * 8 + 16) < self.w,
                            lhb && x8 > 0 && (y8 * 8 + 32) < self.h,
                        );
                        self.code_block16(x8 + 2, y8 + 2, false, false);
                        self.a_part[x8..x8 + 4].fill(0x1c);
                        self.l_part[y8..y8 + 4].fill(0x1c);
                        return;
                    }
                    Part16::VertA => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(6, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block16(
                            x8,
                            y8,
                            y8 > 0 && (x8 * 8 + 16) < self.w,
                            x8 > 0 && (y8 * 8 + 16) < self.h,
                        );
                        self.code_block16(
                            x8,
                            y8 + 2,
                            false,
                            lhb && x8 > 0 && (y8 * 8 + 32) < self.h,
                        );
                        self.code_block32_rect_halves(x8, y8, true, 1..2);
                        self.a_part[x8..x8 + 4].fill(0x1c);
                        self.l_part[y8..y8 + 4].fill(0x18);
                        return;
                    }
                    Part16::VertB => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(7, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block32_rect_halves(x8, y8, true, 0..1);
                        self.code_block16(
                            x8 + 2,
                            y8,
                            thr && y8 > 0 && (x8 * 8 + 32) < self.w,
                            true,
                        );
                        self.code_block16(x8 + 2, y8 + 2, false, false);
                        self.a_part[x8..x8 + 4].fill(0x1c);
                        self.l_part[y8..y8 + 4].fill(0x1c);
                        return;
                    }
                    _ => {}
                }
            }
        }
        // BL_16X16: optionally code the whole 16x16 as one TX_16X16 block
        // (PARTITION_NONE) instead of splitting into four 8x8. Enabled for all
        // subsampling modes: 4:4:4 (chroma 16x16), 4:2:0 (chroma TX_8X8) and
        // 4:2:2 (chroma RTX_8X16). Requires the full 16x16 to be in-frame
        // (have_h && have_v at hh=1 guarantees it, since the coded frame is
        // 8-aligned and the test is strict).
        if sz8 == 2 {
            let have_h = (x8 + 1) * 8 < self.w;
            let have_v = (y8 + 1) * 8 < self.h;
            if have_h && have_v {
                // FORCE_HORZ (test) overrides the RD decision for 4:4:4.
                let choice = self.part_decision(|t| {
                    let forced_horz = !t.ss420
                        && !t.ss422
                        && !t.mono
                        && FORCE_HORZ.load(std::sync::atomic::Ordering::Relaxed);
                    if forced_horz {
                        Part16::Horz
                    } else {
                        t.partition_choice_16(x8, y8, thr, lhb)
                    }
                });
                match choice {
                    Part16::Horz => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(1, &mut self.cdfs.part_split[bl - 1][ctx]); // HORZ
                        if self.ss420 {
                            self.code_block16_rect_420(x8, y8, false);
                        } else if self.ss422 {
                            self.code_block16_horz_422(x8, y8);
                        } else {
                            self.code_block16_horz_444(x8, y8);
                        }
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1e);
                        return;
                    }
                    Part16::Vert => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(2, &mut self.cdfs.part_split[bl - 1][ctx]); // VERT
                        if self.ss420 {
                            self.code_block16_rect_420(x8, y8, true);
                        } else {
                            self.code_block16_vert_444(x8, y8);
                        }
                        self.a_part[x8..x8 + 2].fill(0x1e);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::Horz4 => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(8, &mut self.cdfs.part_split[bl - 1][ctx]); // HORZ_4
                        self.code_block16_quad(x8, y8, false);
                        // dav1d `al_part_ctx` bl16: h4 -> above 0x1c, left 0x1f.
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1f);
                        return;
                    }
                    Part16::Vert4 => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(9, &mut self.cdfs.part_split[bl - 1][ctx]); // VERT_4
                        self.code_block16_quad(x8, y8, true);
                        // dav1d `al_part_ctx` bl16: v4 -> above 0x1f, left 0x1c.
                        self.a_part[x8..x8 + 2].fill(0x1f);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::HorzA => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(4, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block(
                            x8,
                            y8,
                            y8 > 0 && (x8 * 8 + 8) < self.w,
                            x8 > 0 && (y8 * 8 + 8) < self.h,
                        );
                        self.code_block(x8 + 1, y8, thr && y8 > 0 && (x8 * 8 + 16) < self.w, false);
                        self.code_block16_rect_leaf_420(x8, y8 + 1, false);
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1e);
                        return;
                    }
                    Part16::HorzB => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(5, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block16_rect_leaf_420(x8, y8, false);
                        self.code_block(
                            x8,
                            y8 + 1,
                            (x8 * 8 + 8) < self.w,
                            lhb && x8 > 0 && (y8 * 8 + 16) < self.h,
                        );
                        self.code_block(x8 + 1, y8 + 1, false, false);
                        self.a_part[x8..x8 + 2].fill(0x1e);
                        self.l_part[y8..y8 + 2].fill(0x1e);
                        return;
                    }
                    Part16::VertA => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(6, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block(
                            x8,
                            y8,
                            y8 > 0 && (x8 * 8 + 8) < self.w,
                            x8 > 0 && (y8 * 8 + 8) < self.h,
                        );
                        self.code_block(x8, y8 + 1, false, lhb && x8 > 0 && (y8 * 8 + 16) < self.h);
                        self.code_block16_rect_leaf_420(x8 + 1, y8, true);
                        self.a_part[x8..x8 + 2].fill(0x1e);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::VertB => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(7, &mut self.cdfs.part_split[bl - 1][ctx]);
                        self.code_block16_rect_leaf_420(x8, y8, true);
                        self.code_block(x8 + 1, y8, thr && y8 > 0 && (x8 * 8 + 16) < self.w, true);
                        self.code_block(x8 + 1, y8 + 1, false, false);
                        self.a_part[x8..x8 + 2].fill(0x1e);
                        self.l_part[y8..y8 + 2].fill(0x1e);
                        return;
                    }
                    Part16::Intrabc => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                        self.code_intrabc_block(x8, y8, 16);
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::None => {
                        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                        self.enc
                            .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]); // NONE
                        let have_tr = thr && y8 > 0 && (x8 * 8 + 16) < self.w;
                        let have_bl = lhb && x8 > 0 && (y8 * 8 + 16) < self.h;
                        self.code_block16(x8, y8, have_tr, have_bl);
                        self.a_part[x8..x8 + 2].fill(0x1c);
                        self.l_part[y8..y8 + 2].fill(0x1c);
                        return;
                    }
                    Part16::Split => { /* fall through to the SPLIT path below */ }
                }
            }
        }
        // BL_64X64 whole-superblock PARTITION_NONE (all chroma formats; 4:0:0
        // has no whole-64 chroma path and keeps splitting), when the full 64x64
        // is in-frame. Compared against SPLIT by real R-D in `choose_64`.
        if sz8 == 8
            && !self.mono
            && BLOCK64_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
            && (x8 + 8) * 8 <= self.w
            && (y8 + 8) * 8 <= self.h
        {
            let choice = self.part_decision(|t| t.choose_64(x8, y8, thr, lhb));
            if choice == Part16::Intrabc {
                let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                self.enc
                    .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]);
                self.code_block64_intrabc(x8, y8);
                self.a_part[x8..x8 + 8].fill(0x10);
                self.l_part[y8..y8 + 8].fill(0x10);
                return;
            }
            if choice == Part16::None {
                let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
                self.enc
                    .encode_symbol(0, &mut self.cdfs.part_split[bl - 1][ctx]);
                let have_tr = thr && y8 > 0 && (x8 * 8 + 64) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 64) < self.h;
                self.code_block64(x8, y8, have_tr, have_bl);
                self.a_part[x8..x8 + 8].fill(0x10);
                self.l_part[y8..y8 + 8].fill(0x10);
                return;
            }
        }
        let hh = sz8 / 2;
        // content past the horizontal / vertical midpoint of this block?
        let have_h = (x8 + hh) * 8 < self.w;
        let have_v = (y8 + hh) * 8 < self.h;
        let ctx = get_partition_ctx(&self.a_part, &self.l_part, bl, x8, y8);
        if have_h && have_v {
            // full PARTITION_SPLIT symbol -> adapt the partition CDF (dav1d uses
            // decode_symbol_adapt here).
            self.enc
                .encode_symbol(3, &mut self.cdfs.part_split[bl - 1][ctx]);
        } else if have_h {
            // edge: dav1d codes a NON-adapting bool from a gathered probability;
            // read the live (possibly already-adapted) icdf CDF, do not adapt it.
            self.enc
                .encode_gathered_partition(true, &self.cdfs.part_split[bl - 1][ctx]);
        } else if have_v {
            self.enc
                .encode_gathered_partition(false, &self.cdfs.part_split[bl - 1][ctx]);
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
            if cx * 8 < self.w && cy * 8 < self.h {
                self.decode_sb(bl + 1, cx, cy, hh, cthr, clhb);
            }
        }
    }
}
