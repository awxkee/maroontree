/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

fn b64_refinement_window() -> f32 {
    crate::tuning::get().b64_refinement_window
}
fn b64_split_refinement() -> f32 {
    crate::tuning::get().b64_split_refinement
}

/// SSE against the wavefront's raw shared reconstruction
#[allow(clippy::too_many_arguments)]
unsafe fn sse_u16_raw_reference(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: *const u16,
    reference_len: usize,
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    debug_assert!(h == 0 || (ref_y + h - 1) * ref_stride + ref_x + w <= reference_len);
    let mut sse = 0i64;
    for row in 0..h {
        let src_row = &src[(src_y + row) * src_stride + src_x..][..w];
        let ref_offset = (ref_y + row) * ref_stride + ref_x;
        for (column, &src) in src_row.iter().enumerate() {
            // SAFETY: the caller guarantees that this finished reference
            // rectangle is initialized and inside the shared plane.
            let reference = unsafe { *reference.add(ref_offset + column) };
            let diff = i64::from(src) - i64::from(reference);
            sse += diff * diff;
        }
    }
    sse
}

/// Frame-wide 4x4-prefix hash index for the lossy IntraBC exact matcher.
struct LossyIbcIndex {
    step_x: usize,
    step_y: usize,
    mono: bool,
    entries: Vec<(u32, u32)>, // (full hash, packed origin)
    offsets: Vec<u32>,
}

impl LossyIbcIndex {
    fn fingerprint(
        src: &[Vec<u16>; 3],
        w: usize,
        cw: usize,
        mono: bool,
        sub: (usize, usize),
        x: usize,
        y: usize,
    ) -> u32 {
        let mut hash = 0x811c_9dc5u32;
        for (dx, dy) in [(0usize, 0usize), (3, 0), (0, 3), (3, 3)] {
            hash ^= src[0][(y + dy) * w + x + dx] as u32;
            hash = hash.wrapping_mul(0x0100_0193);
        }
        if !mono {
            for plane in &src[1..3] {
                hash ^= plane[(y >> sub.1) * cw + (x >> sub.0)] as u32;
                hash = hash.wrapping_mul(0x0100_0193);
            }
        }
        hash
    }

    fn build(
        src: &[Vec<u16>; 3],
        w: usize,
        h: usize,
        cw: usize,
        mono: bool,
        sub: (usize, usize),
    ) -> Self {
        const BUCKETS: usize = 1 << 16;
        let (step_x, step_y) = (1usize << sub.0, 1usize << sub.1);
        let mut offsets = vec![0u32; BUCKETS + 1];
        if w < 16 || h < 16 {
            return Self {
                step_x,
                step_y,
                mono,
                entries: Vec::new(),
                offsets,
            };
        }
        let xs: Vec<usize> = (0..=w - 4).step_by(step_x).collect();
        let ys: Vec<usize> = (0..=h - 4).step_by(step_y).collect();
        let mut hashes = Vec::with_capacity(xs.len() * ys.len());
        for &y in &ys {
            for &x in &xs {
                let hh = Self::fingerprint(src, w, cw, mono, sub, x, y);
                hashes.push((hh, ((y as u32) << 16) | x as u32));
                offsets[(hh >> 16) as usize + 1] += 1;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor: Vec<u32> = offsets[..BUCKETS].to_vec();
        let mut entries = vec![(0u32, 0u32); hashes.len()];
        for (hh, origin) in hashes {
            let b = (hh >> 16) as usize;
            entries[cursor[b] as usize] = (hh, origin);
            cursor[b] += 1;
        }
        Self {
            step_x,
            step_y,
            mono,
            entries,
            offsets,
        }
    }

    /// Candidate origins whose 4x4-prefix fingerprint matches (px, py)'s, in
    /// raster order, truncated at `max_y` — the caller's legality rule rejects
    /// every origin below that row, and the group is sorted by packed origin
    /// (y-major), so cutting there drops only candidates it would have skipped.
    fn candidates<'i>(
        &'i self,
        src: &[Vec<u16>; 3],
        w: usize,
        cw: usize,
        px: usize,
        py: usize,
        max_y: usize,
    ) -> impl Iterator<Item = (usize, usize)> + 'i {
        let sub = (
            self.step_x.trailing_zeros() as usize,
            self.step_y.trailing_zeros() as usize,
        );
        let want = Self::fingerprint(src, w, cw, self.mono, sub, px, py);
        let b = (want >> 16) as usize;
        let (s, e) = (self.offsets[b] as usize, self.offsets[b + 1] as usize);
        let bucket = &self.entries[s..e];
        let end = bucket.partition_point(|&(_, origin)| (origin >> 16) as usize <= max_y);
        bucket[..end]
            .iter()
            .filter(move |&&(hh, _)| hh == want)
            .map(|&(_, origin)| ((origin & 0xffff) as usize, (origin >> 16) as usize))
    }
}

impl<'a> LossyTile<'a> {
    /// Luma raster quadrants of a 64x64 block, as (dx, dy) pixel offsets.
    const Q64: [(usize, usize); 4] = [(0, 0), (32, 0), (0, 32), (32, 32)];

    /// Rank 64x64 shared luma modes without transform coding. A full candidate
    /// costs four progressive TX_32X32 trellis searches, so Slow protects the
    /// stable DC/SMOOTH/PAETH anchors and admits two additional modes selected
    /// by four-quadrant SATD+SAD.
    fn rank_luma64_modes(
        &self,
        px: usize,
        py: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> FixedList<usize, 13> {
        if self.speed != Speed::Slow {
            let mut keep = FixedList::new(DC_PRED);
            for &mode in fast_nd_modes() {
                keep.push(mode);
            }
            return keep;
        }
        let mut ranked = FixedList::<(u64, usize), 13>::new((0, DC_PRED));
        let ftype = self.luma_filter_type(px, py);
        for &mode in nd_modes() {
            let mut score = 0u64;
            for (sx, sy) in Self::Q64 {
                let (bx, by) = (px + sx, py + sy);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                if mode == DC_PRED {
                    pred.fill(self.intrapred.dc_pred_32x32(
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        self.bd as i32,
                    ));
                } else {
                    self.intrapred.predict_nd(
                        mode,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        32,
                        32,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        ftype,
                        &mut pred[..],
                        self.bd,
                    );
                }
                score += self.rd.satd_sad_proxy(
                    &self.src[0][by * self.w + bx..],
                    self.w,
                    &pred[..],
                    32,
                    32,
                    32,
                );
            }
            ranked.push((score, mode));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, mode)| (score, mode));
        let mut keep = FixedList::new(DC_PRED);
        keep.push(DC_PRED);
        keep.push(SMOOTH_PRED);
        keep.push(PAETH_PRED);
        for &(_, mode) in ranked.iter() {
            if !keep.contains(&mode) {
                keep.push(mode);
                if keep.len() == 5 {
                    break;
                }
            }
        }
        keep
    }

    /// Simplified dav1d dv-stack emulation for a `size`-px square: scan the
    /// neighbor 4x4 IntraBC-MV window; exactly one distinct MV -> that MV,
    /// none -> the spec default DV, several distinct -> None (the candidate is
    /// DROPPED — our scan cannot reproduce dav1d's stack ordering, and a
    /// predictor mismatch decodes a different MV = silent corruption).
    fn intrabc_predictor(&self, px: usize, py: usize, size: usize) -> Option<(i16, i16)> {
        let (x4, y4, n4, stride4) = (px / 4, py / 4, size / 4, self.w / 4);
        let x_start = x4.saturating_sub(5);
        let x_end = (x4 + n4).min(stride4 - 1);
        let y_start = y4.saturating_sub(5);
        let y_end = (y4 + n4).min(self.h / 4 - 1);
        let mut found = None;
        for y in y_start..y4 {
            for x in x4.saturating_sub(1)..=x_end {
                if let Some(mv) = self.ibc_mv[y * stride4 + x] {
                    if found.is_some_and(|old| old != mv) {
                        return None;
                    }
                    found = Some(mv);
                }
            }
        }
        for x in x_start..x4 {
            for y in y4.saturating_sub(1)..=y_end {
                if let Some(mv) = self.ibc_mv[y * stride4 + x] {
                    if found.is_some_and(|old| old != mv) {
                        return None;
                    }
                    found = Some(mv);
                }
            }
        }
        match found {
            None => Some(if py < 64 { (0, -2560) } else { (-512, 0) }),
            Some(mv) => {
                // dav1d's dv stack is built from ITS scan region (immediate
                // above row / left column, plus outer rows and the top-right
                // cell). Our window is a SUPERSET, so a lone MV seen only in
                // the outer band may be invisible to dav1d — its predictor
                // would fall back to the default and the decoded DV lands
                // elsewhere (stream error or silent corruption). Only trust
                // the MV when a guaranteed-scanned cell carries it.
                let mut confirmed = false;
                if y4 > 0 {
                    let row = y4 - 1;
                    for x in x4..(x4 + n4).min(stride4) {
                        if self.ibc_mv[row * stride4 + x] == Some(mv) {
                            confirmed = true;
                            break;
                        }
                    }
                }
                if !confirmed && x4 > 0 {
                    let col = x4 - 1;
                    for y in y4..(y4 + n4).min(self.h / 4) {
                        if self.ibc_mv[y * stride4 + col] == Some(mv) {
                            confirmed = true;
                            break;
                        }
                    }
                }
                if confirmed { Some(mv) } else { None }
            }
        }
    }

    #[allow(clippy::type_complexity)]
    fn find_intrabc(
        &self,
        px: usize,
        py: usize,
        size: usize,
    ) -> Option<(usize, usize, (i16, i16), (i16, i16))> {
        if !self.allow_intrabc {
            return None;
        }
        let pred = self.intrabc_predictor(px, py, size)?;
        let sbx = px / 64 * 64;
        let sby = py / 64 * 64;
        // Chroma-parity restriction (see doc comment).
        let (need_ex, need_ey) = if self.mono {
            (false, false)
        } else {
            (self.ss420 || self.ss422, self.ss420)
        };
        let exact = |rx: usize, ry: usize| {
            (0..if self.mono { 1 } else { 3 }).all(|plane| {
                let sx = usize::from(plane != 0 && (self.ss420 || self.ss422));
                let sy = usize::from(plane != 0 && self.ss420);
                let stride = if plane == 0 { self.w } else { self.cw };
                let (x, y, ref_x, ref_y) = (px >> sx, py >> sy, rx >> sx, ry >> sy);
                let (bw, bh) = (size >> sx, size >> sy);
                (0..bh).all(|row| {
                    self.src[plane][(y + row) * stride + x..][..bw]
                        == self.src[plane][(ref_y + row) * stride + ref_x..][..bw]
                })
            })
        };
        let legal = |rx: usize, ry: usize| {
            if rx + size > self.w
                || ry + size > self.h
                || ry + size > sby + 64
                || (ry + size > sby && rx + size > sbx)
                || (need_ex && rx & 1 != 0)
                || (need_ey && ry & 1 != 0)
            {
                return false;
            }
            // Wavefront restriction, applied at EVERY thread count so serial
            // and captured decisions stay byte-identical: under the capture
            // schedule (d = 2r + c, deps left/top/top-right) the finished set
            // when cell (r, c) starts is {r' < r: c' <= c + (r - r')} plus
            // {r' = r: c' < c}. Every superblock the reference spans must be
            // inside it. Costs only far above-right references (rarely the
            // nearest match on repetitive content).
            let (cr, cc) = (py / 64, px / 64);
            let c1 = (rx + size - 1) / 64;
            let (r0, r1) = (ry / 64, (ry + size - 1) / 64);
            (r0..=r1).all(|r| if r < cr { c1 <= cc + (cr - r) } else { c1 < cc })
        };
        let make = |rx: usize, ry: usize| {
            let dy = (ry as isize - py as isize) * 8;
            let dx = (rx as isize - px as isize) * 8;
            i16::try_from(dy)
                .ok()
                .zip(i16::try_from(dx).ok())
                .filter(|&(dy, dx)| {
                    (i32::from(dy) - i32::from(pred.0)).unsigned_abs() <= 16_384
                        && (i32::from(dx) - i32::from(pred.1)).unsigned_abs() <= 16_384
                })
                .map(|mv| (rx, ry, mv, pred))
        };
        let default = if py < 64 {
            px.checked_sub(320).map(|x| (x, py))
        } else {
            Some((px, py - 64))
        };
        if let Some((rx, ry)) = default
            && legal(rx, ry)
            && exact(rx, ry)
        {
            return make(rx, ry);
        }
        // Hash-index lookup on the block's 4x4 prefix (built once per tile);
        // every hit is verified over the complete block and all planes.
        let idx = self.ibc_index?.get_or_init(|| {
            let sub = if self.mono {
                (0, 0)
            } else {
                (
                    usize::from(self.ss420 || self.ss422),
                    usize::from(self.ss420),
                )
            };
            LossyIbcIndex::build(self.src, self.w, self.h, self.cw, self.mono, sub)
        });
        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, usize, (i16, i16), (i16, i16))> = None;
        let mut best_cost = u32::MAX;
        let mut verified = 0usize;
        // `legal` rejects every origin whose block leaves the current
        // superblock row (`ry + size > sby + 64`), so the index can stop the
        // raster-ordered group there instead of walking it to the frame bottom.
        let max_y = (sby + 64).saturating_sub(size);
        for (rx, ry) in idx.candidates(self.src, self.w, self.cw, px, py, max_y) {
            if (rx, ry) == (px, py) || !legal(rx, ry) {
                continue;
            }
            verified += 1;
            if exact(rx, ry)
                && let Some(found) = make(rx, ry)
            {
                let cost = (i32::from(found.2.0) - i32::from(found.3.0)).unsigned_abs()
                    + (i32::from(found.2.1) - i32::from(found.3.1)).unsigned_abs();
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(found);
                }
            }
            // Bound worst-case work on pathological repeat content.
            if verified >= 128 {
                break;
            }
        }
        best
    }

    /// R-D cost of coding a `size`-px square as a skip IntraBC copy: the
    /// residual is dropped (skip = 1), so distortion is the quantization
    /// drift already present in the reference reconstruction.
    fn rd_cost_intrabc(&self, px: usize, py: usize, size: usize, prdo: f32) -> Option<f32> {
        if size == 64 && self.aq.enabled && self.aq.pending != 0 {
            return None;
        }
        let (rx, ry, mv, pred) = self.find_intrabc(px, py, size)?;
        let mut distortion = 0i64;
        for plane in 0..1 {
            let sx = usize::from(plane != 0 && (self.ss420 || self.ss422));
            let sy = usize::from(plane != 0 && self.ss420);
            let stride = if plane == 0 { self.w } else { self.cw };
            let (x, y, ref_x, ref_y) = (px >> sx, py >> sy, rx >> sx, ry >> sy);
            let (bw, bh) = (size >> sx, size >> sy);
            // Capture workers read the reference from the shared finished
            // planes (the legality rule keeps it inside finished cells, whose
            // values equal the serial reconstruction); serial and replay read
            // the local reconstruction.
            if let Some(sh) = self.ibc_shared {
                let (ptr, len, _) = sh.planes[plane];
                // SAFETY: the plane allocation remains live for the tile and
                // the IntraBC legality rule admits only finished cells.
                distortion += unsafe {
                    sse_u16_raw_reference(
                        &self.src[plane],
                        stride,
                        x,
                        y,
                        ptr,
                        len,
                        stride,
                        ref_x,
                        ref_y,
                        bw,
                        bh,
                    )
                };
            } else {
                distortion += self.rd.sse_u16(
                    &self.src[plane],
                    stride,
                    x,
                    y,
                    &self.recon[plane],
                    stride,
                    ref_x,
                    ref_y,
                    bw,
                    bh,
                );
            }
        }
        let residual_pixels = ((i32::from(mv.0) - i32::from(pred.0)).unsigned_abs()
            + (i32::from(mv.1) - i32::from(pred.1)).unsigned_abs())
            as f32
            / 8.0;
        Some(rd_cost_i64(
            distortion,
            self.mlam() * prdo,
            8.0 + dirty_log2f(residual_pixels.max(1.0)) * 2.0,
        ))
    }

    fn encode_intrabc_mv_component(&mut self, comp: usize, diff: i32) {
        self.enc
            .encode_symbol(usize::from(diff < 0), &mut self.cdfs.mv_sign[comp]);
        let up = diff.unsigned_abs() as usize / 8 - 1;
        let class = if up <= 1 {
            0
        } else {
            usize::BITS as usize - 1 - up.leading_zeros() as usize
        };
        self.enc
            .encode_symbol(class, &mut self.cdfs.mv_classes[comp]);
        if class == 0 {
            self.enc.encode_symbol(up, &mut self.cdfs.mv_class0[comp]);
        } else {
            for n in 0..class {
                self.enc
                    .encode_symbol((up >> n) & 1, &mut self.cdfs.mv_class_n[comp][n]);
            }
        }
    }

    fn encode_intrabc_mv(&mut self, mv: (i16, i16), pred: (i16, i16)) {
        let dy = i32::from(mv.0) - i32::from(pred.0);
        let dx = i32::from(mv.1) - i32::from(pred.1);
        let joint = usize::from(dx != 0) | (usize::from(dy != 0) << 1);
        self.enc.encode_symbol(joint, &mut self.cdfs.mv_joint);
        if dy != 0 {
            self.encode_intrabc_mv_component(0, dy);
        }
        if dx != 0 {
            self.encode_intrabc_mv_component(1, dx);
        }
    }

    fn code_block64_intrabc(&mut self, x8: usize, y8: usize) {
        // Guarded by rd_cost_intrabc: a delta-carrying SB must never take the
        // whole-64 skip path (the decoder would not read the armed token).
        debug_assert!(!self.aq.enabled || self.aq.pending == 0);
        self.aq_cancel_skipped_sb();
        self.code_intrabc_block(x8, y8, 64);
    }

    /// Code a `size`-px square (16/32/64) as a skip IntraBC copy: skip = 1,
    /// use_intrabc = 1, DV residual, no coefficients. Reconstruction is an
    /// integer copy of all coded planes (candidates are chroma-parity-even).
    fn code_intrabc_block(&mut self, x8: usize, y8: usize, size: usize) {
        #[cfg(test)]
        LOSSY_INTRABC_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (px, py) = (x8 * 8, y8 * 8);
        let (rx, ry, mv, pred) = self
            .find_intrabc(px, py, size)
            .expect("legal IntraBC reference");
        let (bx4, by4, stride4, n4) = (px / 4, py / 4, self.w / 4, size / 4);
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc.encode_symbol(1, &mut self.cdfs.skip[sctx]);
        if size < 64 {
            // Sub-SB skip blocks still carry the per-SB delta-q symbol when
            // armed (the spec's early return is only MiSize==sbSize && skip;
            // the 64x64 caller handles that case via aq_cancel_skipped_sb).
            self.code_delta_q_if_armed();
        }
        self.enc.encode_symbol(1, &mut self.cdfs.intrabc);
        self.encode_intrabc_mv(mv, pred);

        self.record_blk(x8, y8, (size / 4) as u8);
        self.mark_skip8_rect(x8, y8, size / 8, size / 8, true);
        // Skip blocks carry the block's max TX dims in the tx-size ctx rows
        // (dav1d b->max_ytx): 16 -> 2, 32 -> 3, 64 -> 4.
        let txc = (size.trailing_zeros() - 2) as i8;
        self.a_skip[bx4..bx4 + n4].fill(1);
        self.l_skip[by4..by4 + n4].fill(1);
        self.a_mode[bx4..bx4 + n4].fill(DC_PRED as u8);
        self.l_mode[by4..by4 + n4].fill(DC_PRED as u8);
        self.a_tx[bx4..bx4 + n4].fill(txc);
        self.l_tx[by4..by4 + n4].fill(txc);
        self.commit_uv_mode(px, py, size, size, DC_PRED);
        for slot in &mut self.a_palette[bx4..bx4 + n4] {
            slot.clear();
        }
        for slot in &mut self.l_palette[by4..by4 + n4] {
            slot.clear();
        }
        // The UV palette state must clear too: the decoder zeroes pal_sz for
        // EVERY block, IntraBC included. Leaving it stale poisons the next UV
        // palette's color cache (cityscape 444 q100 multitile decode-fatal —
        // same family as the ineligible-size clear in emit_palette_mode_info).
        for slot in &mut self.a_palette_uv[bx4..bx4 + n4] {
            slot.clear();
        }
        for slot in &mut self.l_palette_uv[by4..by4 + n4] {
            slot.clear();
        }
        for y in by4..by4 + n4 {
            self.ibc_mv[y * stride4 + bx4..y * stride4 + bx4 + n4].fill(Some(mv));
        }

        for plane in 0..if self.mono { 1 } else { 3 } {
            let sx = usize::from(plane != 0 && (self.ss420 || self.ss422));
            let sy = usize::from(plane != 0 && self.ss420);
            let stride = if plane == 0 { self.w } else { self.cw };
            let (x, y, ref_x, ref_y) = (px >> sx, py >> sy, rx >> sx, ry >> sy);
            let (bw, bh) = (size >> sx, size >> sy);
            if let Some(sh) = self.ibc_shared {
                // Capture sink emission: the reference lies outside this
                // worker's halo, so copy it from the shared finished planes
                // (local recon there is stale — copying it would pollute this
                // cell's recon, poisoning both later same-cell predictions
                // and the streamed pure-emit recon).
                let (ptr, len, _) = sh.planes[plane];
                for row in 0..bh {
                    let off = (ref_y + row) * stride + ref_x;
                    debug_assert!(off + bw <= len);
                    // SAFETY: finished-cell read, see IbcSharedRecon.
                    let srcrow = unsafe { std::slice::from_raw_parts(ptr.add(off), bw) };
                    self.recon[plane][(y + row) * stride + x..][..bw].copy_from_slice(srcrow);
                }
            } else {
                for row in 0..bh {
                    let src = (ref_y + row) * stride + ref_x..(ref_y + row) * stride + ref_x + bw;
                    self.recon[plane].copy_within(src, (y + row) * stride + x);
                }
            }
            let (cx4, cy4, cw4, ch4) = (x / 4, y / 4, (bw / 4).max(1), (bh / 4).max(1));
            self.a_coef[plane][cx4..cx4 + cw4].fill(0x40);
            self.l_coef[plane][cy4..cy4 + ch4].fill(0x40);
        }
    }

    /// Trial the exact luma shape used by `code_block64`: one shared prediction
    /// mode and four raster-order TX_32X32 transforms. Each quadrant is
    /// reconstructed before the next prediction, then the 64x64 region is
    /// restored before returning. Slow fully codes a protected five-mode beam;
    /// Medium/Fast retain the reduced three-mode set.
    fn rd_pick_luma64(
        &mut self,
        px: usize,
        py: usize,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) -> (usize, f32) {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let maxv = (1i32 << self.bd) - 1;
        let block_ftype = self.luma_filter_type(px, py);
        let mut saved = self.sc().take_u4096();
        for row in 0..64 {
            saved[row * 64..row * 64 + 64]
                .copy_from_slice(&self.recon[0][(py + row) * self.w + px..][..64]);
        }
        let restore = |recon: &mut [u16]| {
            for row in 0..64 {
                recon[(py + row) * self.w + px..][..64]
                    .copy_from_slice(&saved[row * 64..row * 64 + 64]);
            }
        };
        let mut best = (DC_PRED, f32::INFINITY);
        for &mode in self.rank_luma64_modes(px, py, have_tr, have_bl).iter() {
            restore(&mut self.recon[0]);
            let mut total = rate_cost(mlam, self.mode_bits(px, py, mode));
            if (V_PRED..=VERT_LEFT_PRED).contains(&mode) {
                total += rate_cost(mlam, cdf_cost(&self.dcdf().angle_delta[mode - V_PRED], 3));
            }
            for (sx, sy) in Self::Q64 {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                if mode == DC_PRED {
                    *pred = [self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    self.intrapred.predict_nd(
                        mode,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        32,
                        32,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        block_ftype,
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
                    bx,
                    by,
                    32,
                    32,
                );
                let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
                // Candidate pricing: Fast uses the plain trellis (the ctx DP
                // here is per-mode cost shared across the whole candidate
                // loop; the winner is re-coded with full ctx trellis at emit
                // in code_block64).
                if self.speed == Speed::Fast {
                    trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
                } else {
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
                        self.dc_sign_ctx_32(0, qbx4, qby4),
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                let distortion =
                    self.luma_partition_distortion(bx, by, 32, 32, acq, &pred[..], 0, &rr[..]);
                total += crate::partition_rd::rd_cost(
                    distortion,
                    mlam,
                    self.luma_bits(&cf, &SCAN_32X32, 32, bx, by, mode, 0),
                );
                for row in 0..32 {
                    let dst = &mut self.recon[0][(by + row) * self.w + bx..][..32];
                    recon_add_pred(dst, &pred[row * 32..], &rr[row * 32..], maxv);
                }
            }
            if total < best.1 {
                best = (mode, total);
            }
        }
        restore(&mut self.recon[0]);
        self.sc().put_u4096(saved);
        best
    }

    fn rd_cost_none64_luma(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let modes: &[usize] = &[DC_PRED];
        let mut best = f32::INFINITY;
        for &m in modes {
            let mut total = 0.0f32;
            for (sx, sy) in Self::Q64 {
                let (bx, by) = (px + sx, py + sy);
                let mut pred = self.sbuf_i1024();
                if m == DC_PRED {
                    *pred = [self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    self.intrapred.predict_nd(
                        m,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        32,
                        32,
                        false,
                        false,
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
                    bx,
                    by,
                    32,
                    32,
                );
                let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                let distortion = self.luma_partition_distortion(
                    bx,
                    by,
                    32,
                    32,
                    self.quant.ac_q() as f32,
                    &pred[..],
                    0,
                    &rr[..],
                );
                total += crate::partition_rd::rd_cost(
                    distortion,
                    mlam,
                    self.luma_bits(&cf, &SCAN_32X32, 32, bx, by, m, 0),
                );
            }
            total += rate_cost(mlam, self.mode_bits(px, py, m));
            if total < best {
                best = total;
            }
        }
        best
    }

    fn rd_cost_chroma64(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let (cx, cy, cgrid, _) = self.chroma64_geom(px, py);
        let mut total = 0.0f32;
        for plane in 1..=2 {
            for &(gx, gy) in cgrid {
                let (tx0, ty0) = (cx + gx, cy + gy);
                let dc = self.intrapred.dc_pred_32x32(&self.recon[plane], self.cw, tx0, ty0, self.bd as i32);
                let mut resid = self.sbuf_i1024();
                self.rd.residual_dc(
                    &mut resid[..],
                    &self.src[plane],
                    self.cw,
                    tx0,
                    ty0,
                    32,
                    32,
                    dc,
                );
                let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.cquant);
                self.chroma_rect_trellis(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    32,
                    plane,
                    tx0,
                    ty0,
                );
                let rr = self.idct.idct_dequant_32x32(&cf, &self.cquant);
                let sse = sse_recon::<1024, 32>(&self.rd,
                    &[dc; 1024],
                    &rr,
                    &self.src[plane],
                    self.cw,
                    tx0,
                    ty0,
                    self.bd,
                );
                total += rd_cost_i64(
                    sse,
                    mlam,
                    self.chroma_bits(&cf, &SCAN_32X32, 32, plane, tx0, ty0),
                );
            }
        }
        self.chroma_partition_weight_at(px, py, 64, 64) * total
    }

    /// SB-level NONE-vs-SPLIT decision for a fully-in-frame 64x64.
    /// Returns `Part16::None` to code one BLOCK_64X64, else `Part16::Split`.
    fn choose_64(&self, x8: usize, y8: usize, thr: bool, lhb: bool) -> Part16 {
        // Fixed-partition mode: take the answer without pricing either leg.
        match crate::tuning::fixed_size(self.speed) {
            0 => {}
            64 => return Part16::None,
            _ => return Part16::Split,
        }
        let (px, py) = (x8 * 8, y8 * 8);
        let prdo = self.perceptual_rd_scale(px, py, 64);
        if self.prefer_split64_from_source(px, py, prdo) {
            return Part16::Split;
        }
        let part_lam = self.mlam() * prdo;
        // Whole-64: four TX_32X32 luma + one 32x32 chroma (per plane).
        // Both legs use the same DC chroma proxy and format-aware chroma scale.
        // The old 4:2:2/4:4:4 handicaps attempted to predict child mode/CfL
        // headroom with fixed constants; once sample density is normalized they
        // are neutral noise and make saturated-edge decisions less portable.
        let none_luma = self.rd_cost_none64_luma(px, py, prdo);
        let rd_none = (none_luma + self.rd_cost_chroma64(px, py, prdo))
            * if self.top_band() && self.ss420 {
                top_none_bias_420(self.aq.base_q)
            } else {
                none64_split_bias()
            }
            + rate_cost(part_lam, self.part_rate_bl(1, x8, y8, 0));
        // First price four forced-NONE 32x32 children. This is an upper bound on
        // the matching recursive 32x32 search, so it biases toward 64x64 NONE:
        // when NONE already loses to this bound, SPLIT is guaranteed cheaper and
        // we can avoid the more expensive child search. Otherwise refine the
        // bound below; using it as the final comparison would over-merge whenever
        // a child's SPLIT/HORZ/VERT candidate is cheaper than its NONE candidate.
        let split_signal = rate_cost(part_lam, self.part_rate_bl(1, x8, y8, 3));
        let mut rd_split_upper = split_signal;
        let mut child_none = [0.0f32; 4];
        let coupled_children =
            !self.mono && self.speed == Speed::Slow && joint_luma_uv_proxy_enabled();
        for (i, (sx, sy)) in [(0usize, 0usize), (32, 0), (0, 32), (32, 32)]
            .into_iter()
            .enumerate()
        {
            let (qx, qy) = (px + sx, py + sy);
            let (cthr, clhb) = Self::child_edge_flags(sx, sy, thr, lhb);
            let (chtr, chbl) = self.leaf_edge_flags(qx, qy, 32, cthr, clhb);
            child_none[i] = self.rd_cost_none32(qx, qy, prdo, chtr, chbl)
                + if coupled_children {
                    0.0
                } else {
                    self.rd_cost_chroma_partition(qx, qy, 32, Part16::None, prdo, false)
                };
            rd_split_upper += child_none[i];
        }
        let rd_ibc = self.rd_cost_intrabc(px, py, 64, prdo);
        let best_whole = rd_none.min(rd_ibc.unwrap_or(f32::INFINITY));
        if rd_split_upper < best_whole {
            return Part16::Split;
        }

        if best_whole <= rd_split_upper * b64_refinement_window() {
            if let Some(rd_ibc) = rd_ibc
                && rd_ibc < rd_none.min(rd_split_upper)
            {
                return Part16::Intrabc;
            }
            let keep = rd_none <= rd_split_upper;
            return if keep { Part16::None } else { Part16::Split };
        }

        // Ambiguous case: price the same legal 32x32 candidate set that
        // `decode_sb` can select after a 64x64 SPLIT. Keep the parent's
        // perceptual scale for all candidates so costs remain on one lambda
        // axis.
        let mut rd_split = split_signal;
        for (i, (sx, sy)) in [(0usize, 0usize), (32, 0), (0, 32), (32, 32)]
            .into_iter()
            .enumerate()
        {
            let (qx, qy) = (px + sx, py + sy);
            let (cthr, clhb) = Self::child_edge_flags(sx, sy, thr, lhb);
            rd_split += self
                .rd_choice_rect32(qx / 8, qy / 8, prdo, Some(child_none[i]), cthr, clhb)
                .1;
        }
        // The child estimator is deliberately lightweight and can exaggerate
        // the benefit of deeper partitions. Blend its saving against the legal
        // forced-NONE bound so confidence can be calibrated by measured RD.
        // A partial-sum abort bound was built and measured here 2026-07-24 and
        // does NOT pay: `b64_split_refinement()` halves the partial sum's weight,
        // so the bound only reaches `rd_none` once the sum is nearly complete.
        rd_split =
            rd_split_upper + b64_split_refinement() * (rd_split.min(rd_split_upper) - rd_split_upper);
        if let Some(rd_ibc) = rd_ibc
            && rd_ibc < rd_none.min(rd_split)
        {
            return Part16::Intrabc;
        }
        let keep = rd_none <= rd_split;
        if keep { Part16::None } else { Part16::Split }
    }

    /// Code a fully-in-frame 64x64 region as one BLOCK_64X64. Luma uses one
    /// shared intra mode reconstructed as four running-raster TX_32X32
    /// quadrants; chroma is a DC-predicted TX_32X32 grid whose shape follows
    /// the subsampling (4:4:4 2x2, 4:2:2 1x2, 4:2:0 single).
    fn code_block64(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (cx, cy, cgrid, csplit) = self.chroma64_geom(px, py);
        let maxv = (1i32 << self.bd) - 1;
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let prdo = self.perceptual_rd_scale(px, py, 64);
        let lam = trellis_lambda() * prdo;

        // Deblock footprint: four TX_32X32 tiles so the filter sees the interior
        // 32-sample transform edges (mirrors block16's tx-split re-record).
        for (sx, sy) in Self::Q64 {
            self.record_tx_blk((px + sx) / 8, (py + sy) / 8, 8);
        }

        // Intra-edge smooth-filter flag: dav1d derives it ONCE at the BLOCK
        // origin from the neighbor modes and reuses it for every sub-transform.
        // Deriving it per quadrant (or after a_mode/l_mode are overwritten)
        // desyncs the prediction from the decoder — the stream still decodes,
        // but the reconstruction diverges (severe on detail, invisible on flats).
        let block_ftype = self.luma_filter_type(px, py);

        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();

        // --- Luma: pick a mode, then real four-quadrant coding (running recon).
        let mut lcf = [
            self.sbuf_i1024(),
            self.sbuf_i1024(),
            self.sbuf_i1024(),
            self.sbuf_i1024(),
        ];
        let y_mode;
        if let Some(r) = rl {
            y_mode = r.mode as usize;
            if let Some(cf) = rl_cf {
                for qi in 0..4 {
                    lcf[qi].copy_from_slice(&cf[qi * 1024..qi * 1024 + 1024]);
                }
            }
        } else {
            y_mode = self.rd_pick_luma64(px, py, have_tr, have_bl, prdo).0;
        }
        self.record_pred_blk(x8, y8, 16);
        // Real coding of the winner: four TX_32X32, each predicted from the
        // running reconstruction, coefficients captured into `lcf`. Skipped in
        // Replay (recon preinstalled, coeffs loaded from the record above).
        if rl.is_none() {
            for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                if y_mode == DC_PRED {
                    *pred = [self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    self.intrapred.predict_nd(
                        y_mode,
                        &self.recon[0],
                        self.w,
                        bx,
                        by,
                        32,
                        32,
                        tr,
                        bl,
                        self.w,
                        self.h,
                        block_ftype,
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
                    bx,
                    by,
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
                    self.dc_sign_ctx_32(0, qbx4, qby4),
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
                let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
                for ry in 0..32 {
                    let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                    recon_add_pred(&mut drow[..32], &pred[ry * 32..], &rr[ry * 32..], maxv);
                }
                *lcf[qi] = cf;
            }
        }
        let luma_zero = lcf.iter().all(|q| self.rd.all_zero_i32(&q[..]));

        // --- Chroma: DC prediction, a TX_32X32 grid per plane (see
        // `chroma64_geom`). Each transform predicts from the RUNNING chroma
        // reconstruction, exactly as the decoder does per transform block.
        let ncg = cgrid.len();
        // Chroma coefficients land straight in the flat per-plane scratch the
        // capture push wants (`uflat`/`vflat` below) instead of a 32 KiB
        // zeroed stack array; only the `ncg` groups actually in the grid are
        // live. The scratch is not zeroed on reuse, so the one path that
        // writes neither branch (replay without coefficients) clears its own
        // live range.
        let mut uflat = self.sbuf_i4096();
        let mut vflat = self.sbuf_i4096();
        if let Some((cf, _)) = ru_cf.as_ref() {
            for (ci, dst) in [&mut uflat, &mut vflat].into_iter().enumerate() {
                dst[..ncg * 1024].copy_from_slice(&cf[ci][..ncg * 1024]);
            }
        } else if ru.is_some() {
            uflat[..ncg * 1024].fill(0);
            vflat[..ncg * 1024].fill(0);
        }
        if ru.is_none() {
            #[allow(clippy::needless_range_loop)]
            for ci in 0..2 {
                let plane = ci + 1;
                for (gi, &(gx, gy)) in cgrid.iter().enumerate() {
                    let (tx0, ty0) = (cx + gx, cy + gy);
                    let dc = self.intrapred.dc_pred_32x32(&self.recon[plane], self.cw, tx0, ty0, self.bd as i32);
                    let mut resid = self.sbuf_i1024();
                    self.rd.residual_dc(
                        &mut resid[..],
                        &self.src[plane],
                        self.cw,
                        tx0,
                        ty0,
                        32,
                        32,
                        dc,
                    );
                    let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut cf,
                        &tf,
                        cdcq,
                        cacq,
                        &SCAN_32X32,
                        lam,
                        32,
                        32,
                        plane,
                        tx0,
                        ty0,
                    );
                    self.rd.preserve_dc(&mut cf[0], &resid[..]);
                    // Reconstruct now so the next transform predicts off it.
                    let rr = self.idct.idct_dequant_32x32(&cf, &self.cquant);
                    for ry in 0..32 {
                        let drow = &mut self.recon[plane][(ty0 + ry) * self.cw + tx0..];
                        recon_add_dc(&mut drow[..32], dc, &rr[ry * 32..], maxv);
                    }
                    let dst = if ci == 0 { &mut uflat } else { &mut vflat };
                    dst[gi * 1024..gi * 1024 + 1024].copy_from_slice(&cf);
                }
            }
        }
        // AV1 5.11.6 `read_delta_qindex()` returns early when
        // `MiSize == sbSize && skip` — a BLOCK_64X64 IS the superblock size, so a
        // SKIPPED one codes no delta_q token while the encoder still advanced
        // `cur_qidx`, desyncing the quantizer for every later superblock (a
        // compounding DC error). Never emit a skipped 64-block: coding it
        // non-skip with all-zero transforms (`txb_skip = 1` each) is legal, costs
        // ~6 symbols, and keeps the delta_q token unconditional.
        let block_skip = false;
        let _ = luma_zero;

        // Record the winner for the wavefront (Capture only; no-ops otherwise).
        self.push_luma_sel(LumaSel {
            mode: y_mode as u8,
            delta: 0,
            palette: 0,
            filter: NO_FILTER,
            tx: TxSel::SplitDct([1; 4]),
        });
        let mut flat = self.sbuf_i4096();
        for qi in 0..4 {
            flat[qi * 1024..qi * 1024 + 1024].copy_from_slice(&lcf[qi][..]);
        }
        self.push_luma_cf(&flat[..]);
        self.push_uv_sel(UvSel {
            uv: DC_PRED as u8,
            palette: 0,
        });
        self.push_uv_cf(&uflat[..ncg * 1024], &vflat[..ncg * 1024], [0, 0]);

        // --- Header syntax (decoder order): skip, y_mode, uv_mode, tx_depth.
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens_64(block_skip, sctx);
        self.mark_skip8(x8, y8, 8, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        // Directional modes carry an angle_delta symbol (`use_angle_delta` is
        // true for BLOCK_8X8 and larger). The 64x64 search offers delta 0 only.
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[y_mode - V_PRED]);
        }
        // CfL is not allowed at 64x64, so uv_mode uses the NOCFL CDF (index m,
        // not 13+m) — emit it directly rather than via `emit_uv_mode`.
        self.enc
            .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[y_mode]);
        self.commit_uv_mode(px, py, 64, 64, DC_PRED);
        self.emit_palette_mode_info(px, py, 64, 64, y_mode, !self.mono, None, None);
        // filter_intra is disallowed for max(w,h) > 32, so no symbol here.
        self.code_tx_depth(px, py, 64, 64, 1);
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 16].fill(sv);
        self.l_skip[by4..by4 + 16].fill(sv);
        self.a_mode[bx4..bx4 + 16].fill(mv);
        self.l_mode[by4..by4 + 16].fill(mv);

        // --- Luma coefficients: four TX_32X32 in raster order (split contexts).
        for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
            let (qbx4, qby4) = ((px + sx) / 4, (py + sy) / 4);
            let res_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_split(qbx4, qby4, 8, 8);
                let ds = self.dc_sign_ctx_32(0, qbx4, qby4);
                encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &lcf[qi], false, sk, ds)
            };
            self.a_coef[0][qbx4..qbx4 + 8].fill(res_ctx);
            self.l_coef[0][qby4..qby4 + 8].fill(res_ctx);
        }
        // --- Chroma coefficients: the TX_32X32 grid, raster order per plane.
        // Reconstruction already happened during the compute pass above (the
        // running-recon prediction requires it), so this only emits + updates
        // the neighbor coefficient contexts.
        #[allow(clippy::needless_range_loop)]
        for ci in 0..2 {
            let plane = ci + 1;
            for (gi, &(gx, gy)) in cgrid.iter().enumerate() {
                let (gbx4, gby4) = ((cx + gx) / 4, (cy + gy) / 4);
                let cres = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_chroma32(plane, gbx4, gby4, csplit);
                    let ds = self.dc_sign_ctx_32(plane, gbx4, gby4);
                    encode_tx32_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        (if ci == 0 { &uflat } else { &vflat })[gi * 1024..]
                            .first_chunk::<1024>()
                            .unwrap(),
                        true,
                        sk,
                        ds,
                    )
                };
                self.a_coef[plane][gbx4..gbx4 + 8].fill(cres);
                self.l_coef[plane][gby4..gby4 + 8].fill(cres);
            }
        }
    }

    /// Chroma geometry for a 64x64 luma block: the chroma-plane origin and the
    /// grid of TX_32X32 transforms covering the chroma block, plus whether that
    /// grid is a true split. AV1 `get_tx_size()` clamps any chroma transform
    /// that would be 64 wide or tall down to TX_32X32, so the chroma block is
    /// tiled: 4:4:4 (64x64 chroma) needs a 2x2 grid, 4:2:2 (32x64) a vertical
    /// pair, and 4:2:0 (32x32) a single transform that covers the block exactly.
    #[inline]
    fn chroma64_geom(
        &self,
        px: usize,
        py: usize,
    ) -> (usize, usize, &'static [(usize, usize)], bool) {
        static G1: [(usize, usize); 1] = [(0, 0)];
        static G2: [(usize, usize); 2] = [(0, 0), (0, 32)];
        static G4: [(usize, usize); 4] = [(0, 0), (32, 0), (0, 32), (32, 32)];
        if self.ss420 {
            (px / 2, py / 2, &G1[..], false)
        } else if self.ss422 {
            (px / 2, py, &G2[..], true)
        } else {
            (px, py, &G4[..], true)
        }
    }

    /// `txb_skip` context for a chroma TX_32X32. `split` selects dav1d's
    /// `not_one_blk` bucket (+3), used when the transform does not cover the
    /// whole chroma plane block (4:4:4 / 4:2:2 at 64x64). 4:2:0's single
    /// block-sized transform keeps the plain `7 + above + left` form that
    /// `skip_ctx_32` already implements.
    #[inline]
    fn skip_ctx_chroma32(&self, plane: usize, bx4: usize, by4: usize, split: bool) -> usize {
        let a = &self.a_coef[plane];
        let l = &self.l_coef[plane];
        let ca = a[bx4..bx4 + 8].iter().any(|&x| x != 0x40) as usize;
        let cl = l[by4..by4 + 8].iter().any(|&x| x != 0x40) as usize;
        7 + if split { 3 } else { 0 } + ca + cl
    }

    /// Per-quadrant intra-edge availability, mirroring the block16 tx-split map.
    #[inline]
    fn quad_edges(
        sx: usize,
        sy: usize,
        px: usize,
        py: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> (bool, bool) {
        match (sx, sy) {
            (0, 0) => (py > 0, px > 0),
            (32, 0) => (have_tr, false),
            (0, 32) => (true, have_bl),
            _ => (false, false),
        }
    }
}
