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

    #[allow(clippy::too_many_arguments)]
    fn rank_luma_directionals<const N: usize>(
        &self,
        modes: &[usize],
        px: usize,
        py: usize,
        w: usize,
        h: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> DirectionalTopK {
        debug_assert_eq!(N, w * h);
        let mut top = DirectionalTopK::new();
        for &mode in modes {
            if !is_directional_mode(mode) {
                continue;
            }
            let mut pred = [0i32; N];
            intra_predict_nd(
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
            let src = &self.src[0][py * self.w + px..];
            top.insert(mode, satd_sad_proxy(src, self.w, &pred, w, w, h));
        }
        top
    }

    #[allow(clippy::too_many_arguments)]
    fn rank_chroma_directionals<const N: usize>(
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
            if !is_directional_mode(mode) {
                continue;
            }
            if mode != V_PRED && mode != H_PRED && !self.speed.chroma_angle_directional() {
                continue;
            }
            let mut cost = 0u64;
            for plane in 1..=2 {
                let mut pred = [0i32; N];
                intra_predict_nd(
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
                cost += satd_sad_proxy(src, self.cw, &pred, w, w, h);
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
            }
        }
    }

    fn decode_sb(&mut self, bl: usize, x8: usize, y8: usize, sz8: usize, thr: bool, lhb: bool) {
        if sz8 == 1 {
            // BL_8X8 leaf (always fully in-frame for multiple-of-8 dimensions):
            // emit PARTITION_NONE, then the block. When the split scaffold is
            // forced (test-only), emit PARTITION_SPLIT and code four BLOCK_4X4.
            let ctx = get_partition_ctx(&self.a_part, &self.l_part, 4, x8, y8);
            let split_eligible = !self.ss422 && !self.mono;
            let want_split = split_eligible
                && (FORCE_SPLIT4.load(std::sync::atomic::Ordering::Relaxed)
                    || (SPLIT4_ENABLED.load(std::sync::atomic::Ordering::Relaxed)
                        && !self.prefer_8x8_none(x8, y8)));
            if want_split {
                self.enc.encode_symbol(3, &mut self.cdfs.part_bl8[ctx]); // SPLIT
                self.code_block_split4_dc(x8, y8);
                self.a_part[x8] = 0x1f;
                self.l_part[y8] = 0x1f;
                return;
            }
            let r8 = self.choose_rect8(x8, y8);
            if r8 == Part16::Horz {
                self.enc.encode_symbol(1, &mut self.cdfs.part_bl8[ctx]);
                self.code_block8_rect(x8, y8, false);
                self.a_part[x8] = 0x1e;
                self.l_part[y8] = 0x1f;
                return;
            }
            if r8 == Part16::Vert {
                self.enc.encode_symbol(2, &mut self.cdfs.part_bl8[ctx]);
                self.code_block8_rect(x8, y8, true);
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
                let prefer_none = self.prefer_32x32(x8, y8);
                let choice = self.choose_rect32(x8, y8, prefer_none);
                let have_tr = thr && y8 > 0 && (x8 * 8 + 32) < self.w;
                let have_bl = lhb && x8 > 0 && (y8 * 8 + 32) < self.h;
                match choice {
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
                let forced_horz = !self.ss420
                    && !self.ss422
                    && !self.mono
                    && FORCE_HORZ.load(std::sync::atomic::Ordering::Relaxed);
                let choice = if forced_horz {
                    Part16::Horz
                } else {
                    self.partition_choice_16(x8, y8)
                };
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
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], true);
            self.enc.encode_bool(true, p);
        } else if have_v {
            let p = gather_split_prob_icdf(&self.cdfs.part_split[bl - 1][ctx], false);
            self.enc.encode_bool(true, p);
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
