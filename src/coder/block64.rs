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

/// Tile-wide IntraBC exact-match index over FULL 8x8 luma windows.
///
/// Every chroma-parity-grid position holds the polynomial hash of the whole
/// 8x8 luma window starting there. A query anchors on the block's most
/// textured 8x8 sub-block (see [`Self::anchor`]), so text and UI blocks whose
/// top-left corner is plain background still land in a small, exact bucket.
/// (The previous 4-corner 4x4-prefix fingerprint put every block with a flat
/// top-left 4x4 into ONE bucket — 58% of a screenshot — and the 128-candidate
/// verify cap expired long before reaching the real glyph match.)
struct LossyIbcIndex {
    entries: Vec<(u32, u32)>, // (mixed 8x8 hash, packed origin)
    offsets: Vec<u32>,
}

/// Memoized IntraBC match search for one block (see `find_intrabc`).
enum IbcMatches {
    /// The spec default DV is a legal exact copy (searched first).
    Default(usize, usize),
    /// Exact copies among the first 128 legal index candidates, in order.
    List(Vec<(u32, u32)>),
}

const IBC_HP: u32 = 0x9E37_79B1;
const IBC_HQ: u32 = 0x85EB_CA77;

#[inline]
fn ibc_mix(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x7feb_352d);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846c_a68b);
    h ^ (h >> 16)
}

impl LossyIbcIndex {
    /// Direct (non-rolling) form of the window hash; equals the rolling one.
    fn hash8(luma: &[u16], w: usize, x: usize, y: usize) -> u32 {
        let mut c = 0u32;
        for j in 0..8 {
            let row = luma[(y + j) * w + x..].first_chunk::<8>().unwrap();
            let mut r = 0u32;
            for &v in row.iter() {
                r = r.wrapping_mul(IBC_HP).wrapping_add(u32::from(v) + 1);
            }
            c = c.wrapping_mul(IBC_HQ).wrapping_add(r);
        }
        ibc_mix(c)
    }

    fn build(src: &[Vec<u16>; 3], w: usize, h: usize, sub: (usize, usize)) -> Self {
        const BUCKETS: usize = 1 << 16;
        let (step_x, step_y) = (1usize << sub.0, 1usize << sub.1);
        let mut offsets = vec![0u32; BUCKETS + 1];
        if w < 16 || h < 16 {
            return Self {
                entries: Vec::new(),
                offsets,
            };
        }
        let luma = &src[0];
        let nx = w - 7;
        // Row hashes of every horizontal 8-run, then a vertical roll over
        // them; wrapping u32 arithmetic is a ring, so the rolled value equals
        // `hash8` exactly.
        let p7 = (0..7).fold(1u32, |a, _| a.wrapping_mul(IBC_HP));
        let q7 = (0..7).fold(1u32, |a, _| a.wrapping_mul(IBC_HQ));
        let mut rows = vec![0u32; h * nx];
        for y in 0..h {
            let line = &luma[y * w..][..w];
            let out = &mut rows[y * nx..][..nx];
            let mut r = 0u32;
            for &v in &line[..8] {
                r = r.wrapping_mul(IBC_HP).wrapping_add(u32::from(v) + 1);
            }
            out[0] = r;
            for x in 1..nx {
                r = r
                    .wrapping_sub((u32::from(line[x - 1]) + 1).wrapping_mul(p7))
                    .wrapping_mul(IBC_HP)
                    .wrapping_add(u32::from(line[x + 7]) + 1);
                out[x] = r;
            }
        }
        let ny = h - 7;
        let mut col = vec![0u32; nx];
        for j in 0..8 {
            for (c, &r) in col.iter_mut().zip(&rows[j * nx..][..nx]) {
                *c = c.wrapping_mul(IBC_HQ).wrapping_add(r);
            }
        }
        let mut hashes = Vec::with_capacity((nx / step_x + 1) * (ny / step_y + 1));
        for y in 0..ny {
            if y > 0 {
                let (old, new) = (&rows[(y - 1) * nx..][..nx], &rows[(y + 7) * nx..][..nx]);
                for ((c, &o), &n) in col.iter_mut().zip(old).zip(new) {
                    *c = c.wrapping_sub(o.wrapping_mul(q7)).wrapping_mul(IBC_HQ).wrapping_add(n);
                }
            }
            if y % step_y != 0 {
                continue;
            }
            for x in (0..nx).step_by(step_x) {
                let hh = ibc_mix(col[x]);
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
        Self { entries, offsets }
    }

    /// The 8-aligned 8x8 sub-block of a `size` square with the widest luma
    /// range (first on ties; (0, 0) for a flat block). Offsets are multiples
    /// of 8, so anchored origins keep the index's chroma parity.
    fn anchor(luma: &[u16], w: usize, px: usize, py: usize, size: usize) -> (usize, usize) {
        let mut best = (0usize, 0usize);
        let mut best_range = 0u16;
        for oy in (0..size).step_by(8) {
            for ox in (0..size).step_by(8) {
                let (mut lo, mut hi) = (u16::MAX, 0u16);
                for j in 0..8 {
                    for &v in &luma[(py + oy + j) * w + px + ox..][..8] {
                        lo = lo.min(v);
                        hi = hi.max(v);
                    }
                }
                if hi - lo > best_range {
                    best_range = hi - lo;
                    best = (ox, oy);
                }
            }
        }
        best
    }

    /// Candidate origins whose anchored 8x8 window hashes like (px, py)'s,
    /// in raster order, truncated at `max_y` — the caller's legality rule
    /// rejects every origin below that row, and the bucket is sorted by
    /// packed origin (y-major), so cutting there drops only candidates it
    /// would have skipped.
    #[allow(clippy::too_many_arguments)]
    fn candidates<'i>(
        &'i self,
        luma: &[u16],
        w: usize,
        px: usize,
        py: usize,
        size: usize,
        max_y: usize,
    ) -> impl Iterator<Item = (usize, usize)> + 'i {
        let (ox, oy) = Self::anchor(luma, w, px, py, size);
        let want = Self::hash8(luma, w, px + ox, py + oy);
        let b = (want >> 16) as usize;
        let (s, e) = (self.offsets[b] as usize, self.offsets[b + 1] as usize);
        let bucket = &self.entries[s..e];
        let end = bucket.partition_point(|&(_, origin)| (origin >> 16) as usize <= max_y + oy);
        bucket[..end]
            .iter()
            .filter(move |&&(hh, _)| hh == want)
            .filter_map(move |&(_, origin)| {
                let (ax, ay) = ((origin & 0xffff) as usize, (origin >> 16) as usize);
                Some((ax.checked_sub(ox)?, ay.checked_sub(oy)?))
            })
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

    /// Luma prediction-block dims (in 4x4 units) and IntraBC DV of the
    /// decoded 4x4 cell at (x4, y4), as dav1d's `refmvs_block` holds them.
    #[inline]
    fn ibc_cell(&self, x4: usize, y4: usize) -> (Option<(i16, i16)>, usize, usize) {
        let i = y4 * (self.w / 4) + x4;
        (
            self.ibc_mv[i],
            usize::from(self.pblk4[i]).max(1),
            usize::from(self.pblk4h[i]).max(1),
        )
    }

    /// dav1d `add_spatial_candidate` for the IntraBC reference (ref 0,
    /// single): plain intra cells carry no MV and add nothing.
    fn ibc_add_cand(stack: &mut [((i16, i16), u32); 8], cnt: &mut usize, weight: u32, mv: Option<(i16, i16)>) {
        let Some(mv) = mv else {
            return;
        };
        for slot in stack[..*cnt].iter_mut() {
            if slot.0 == mv {
                slot.1 += weight;
                return;
            }
        }
        if *cnt < 8 {
            stack[*cnt] = (mv, weight);
            *cnt += 1;
        }
    }

    /// dav1d `scan_row`: candidates along luma row `y4`, starting at `x4`.
    #[allow(clippy::too_many_arguments)]
    fn ibc_scan_row(
        &self,
        stack: &mut [((i16, i16), u32); 8],
        cnt: &mut usize,
        x4: usize,
        y4: usize,
        bw4: usize,
        w4: usize,
        max_rows: usize,
        step: usize,
    ) -> usize {
        let (mv, cbw4, cbh4) = self.ibc_cell(x4, y4);
        let mut len = step.max(bw4.min(cbw4));
        if bw4 <= cbw4 {
            let weight = if bw4 == 1 { 2 } else { 2usize.max((2 * max_rows).min(cbh4)) };
            Self::ibc_add_cand(stack, cnt, (len * weight) as u32, mv);
            return weight >> 1;
        }
        let mut mv = mv;
        let mut x = 0usize;
        loop {
            Self::ibc_add_cand(stack, cnt, (len * 2) as u32, mv);
            x += len;
            if x >= w4 {
                return 1;
            }
            let (m, cw, _) = self.ibc_cell(x4 + x, y4);
            mv = m;
            len = step.max(cw);
        }
    }

    /// dav1d `scan_col`: candidates down luma column `x4`, starting at `y4`.
    #[allow(clippy::too_many_arguments)]
    fn ibc_scan_col(
        &self,
        stack: &mut [((i16, i16), u32); 8],
        cnt: &mut usize,
        x4: usize,
        y4: usize,
        bh4: usize,
        h4: usize,
        max_cols: usize,
        step: usize,
    ) -> usize {
        let (mv, cbw4, cbh4) = self.ibc_cell(x4, y4);
        let mut len = step.max(bh4.min(cbh4));
        if bh4 <= cbh4 {
            let weight = if bh4 == 1 { 2 } else { 2usize.max((2 * max_cols).min(cbw4)) };
            Self::ibc_add_cand(stack, cnt, (len * weight) as u32, mv);
            return weight >> 1;
        }
        let mut mv = mv;
        let mut y = 0usize;
        loop {
            Self::ibc_add_cand(stack, cnt, (len * 2) as u32, mv);
            y += len;
            if y >= h4 {
                return 1;
            }
            let (m, _, ch) = self.ibc_cell(x4, y4 + y);
            mv = m;
            len = step.max(ch);
        }
    }

    /// The decoder's IntraBC DV predictor for a `size`-px square at (px, py):
    /// an exact port of dav1d `dav1d_refmvs_find` for the IntraBC reference
    /// (no global/temporal/extended candidates exist in an intra frame, and
    /// its clamp cannot bind for a DV the legality rule admits), followed by
    /// the stack[0] -> stack[1] -> default choice in `decode_b`. Reads only
    /// decoded neighbors: `ibc_mv` plus the prediction-block dims `pblk4` /
    /// `pblk4h`, all of which cross the wavefront handoff. `thr` is the
    /// node's TOP_HAS_RIGHT edge flag.
    fn intrabc_predictor(&self, px: usize, py: usize, size: usize, thr: bool) -> (i16, i16) {
        let (bx4, by4, n4) = (px / 4, py / 4, size / 4);
        let (end_c, end_r) = self.ibc_end4;
        let (bw4, bh4) = (n4, n4);
        let w4 = bw4.min(16).min(end_c.saturating_sub(bx4));
        let h4 = bh4.min(16).min(end_r.saturating_sub(by4));
        let mut stack = [((0i16, 0i16), 0u32); 8];
        let mut cnt = 0usize;
        let mut max_rows = 0usize;
        let mut n_rows = usize::MAX;
        if by4 > 0 {
            max_rows = ((by4 + 1) >> 1).min(2 + usize::from(bh4 > 1));
            n_rows = self.ibc_scan_row(
                &mut stack,
                &mut cnt,
                bx4,
                by4 - 1,
                bw4,
                w4,
                max_rows,
                if bw4 >= 16 { 4 } else { 1 },
            );
        }
        let mut max_cols = 0usize;
        let mut n_cols = usize::MAX;
        if bx4 > 0 {
            max_cols = ((bx4 + 1) >> 1).min(2 + usize::from(bw4 > 1));
            n_cols = self.ibc_scan_col(
                &mut stack,
                &mut cnt,
                bx4 - 1,
                by4,
                bh4,
                h4,
                max_cols,
                if bh4 >= 16 { 4 } else { 1 },
            );
        }
        // top-right: TOP_HAS_RIGHT edge flag, block <= 64, inside the tile
        if n_rows != usize::MAX && thr && bw4.max(bh4) <= 16 && bw4 + bx4 < end_c {
            let (mv, _, _) = self.ibc_cell(bx4 + bw4, by4 - 1);
            Self::ibc_add_cand(&mut stack, &mut cnt, 4, mv);
        }
        let nearest = cnt;
        for slot in &mut stack[..nearest] {
            slot.1 += 640;
        }
        // top-left (needs both edges)
        if n_rows != usize::MAX && n_cols != usize::MAX {
            let (mv, _, _) = self.ibc_cell(bx4 - 1, by4 - 1);
            Self::ibc_add_cand(&mut stack, &mut cnt, 4, mv);
        }
        // secondary rows/cols at 8x8 resolution (odd 4x4 positions)
        for n in 2..=3usize {
            if n_rows != usize::MAX && n > n_rows && n <= max_rows {
                let row = (by4 + 1 - 2 * n) | 1;
                n_rows += self.ibc_scan_row(
                    &mut stack,
                    &mut cnt,
                    bx4 | 1,
                    row,
                    bw4,
                    w4,
                    1 + max_rows - n,
                    if bw4 >= 16 { 4 } else { 2 },
                );
            }
            if n_cols != usize::MAX && n > n_cols && n <= max_cols {
                let col = (bx4 + 1 - 2 * n) | 1;
                n_cols += self.ibc_scan_col(
                    &mut stack,
                    &mut cnt,
                    col,
                    by4 | 1,
                    bh4,
                    h4,
                    1 + max_cols - n,
                    if bh4 >= 16 { 4 } else { 2 },
                );
            }
        }
        // bubble sorts (nearest, then secondary), swapping only on strict <
        let mut len = nearest;
        while len > 0 {
            let mut last = 0;
            for n in 1..len {
                if stack[n - 1].1 < stack[n].1 {
                    stack.swap(n - 1, n);
                    last = n;
                }
            }
            len = last;
        }
        let mut len = cnt;
        while len > nearest {
            let mut last = nearest;
            for n in nearest + 1..len {
                if stack[n - 1].1 < stack[n].1 {
                    stack.swap(n - 1, n);
                    last = n;
                }
            }
            len = last;
        }
        // decode_b: stack[0], else stack[1] (zero-filled past cnt), else default
        for slot in stack.iter().take(cnt.min(2)) {
            if slot.0 != (0, 0) {
                return slot.0;
            }
        }
        if py < 64 { (0, -2560) } else { (-512, 0) }
    }

    #[allow(clippy::type_complexity)]
    fn find_intrabc(
        &self,
        px: usize,
        py: usize,
        size: usize,
        thr: bool,
    ) -> Option<(usize, usize, (i16, i16), (i16, i16))> {
        if !self.allow_intrabc {
            return None;
        }
        let pred = self.intrabc_predictor(px, py, size, thr);
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
            if !(r0..=r1).all(|r| if r < cr { c1 <= cc + (cr - r) } else { c1 < cc }) {
                return false;
            }
            crate::tile::intrabc_dv_in_range(px, py, rx, ry)
                && crate::tile::intrabc_dv_conformant(px, py, rx, ry, size, self.w)
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
        // The match search is a pure function of the source and the block
        // geometry (index, legality, exactness); only the DV choice below
        // depends on the neighbor-derived predictor. Bottom-up pricing, the
        // parent's SPLIT leg, the node decision and the emitter all re-ask
        // for the same block, so memoize the verified match set.
        let key = ((px as u64) << 34) | ((py as u64) << 8) | size as u64;
        let mut cache = self.ibc_match_cache.borrow_mut();
        let matches = match cache.entry(key) {
            hashbrown::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            hashbrown::hash_map::Entry::Vacant(entry) => {
                let m = if let Some((rx, ry)) = default
                    && legal(rx, ry)
                    && exact(rx, ry)
                {
                    IbcMatches::Default(rx, ry)
                } else {
                    // Hash-index lookup on the block's anchored 8x8 window
                    // (built once per tile); every hit is verified over the
                    // complete block and all planes.
                    let idx = self.ibc_index?.get_or_init(|| {
                        let sub = if self.mono {
                            (0, 0)
                        } else {
                            (
                                usize::from(self.ss420 || self.ss422),
                                usize::from(self.ss420),
                            )
                        };
                        LossyIbcIndex::build(self.src, self.w, self.h, sub)
                    });
                    let mut list = Vec::new();
                    let mut verified = 0usize;
                    // `legal` rejects every origin whose block leaves the
                    // current superblock row (`ry + size > sby + 64`), so the
                    // index can stop the raster-ordered group there.
                    let max_y = (sby + 64).saturating_sub(size);
                    for (rx, ry) in idx.candidates(&self.src[0], self.w, px, py, size, max_y) {
                        if (rx, ry) == (px, py) || !legal(rx, ry) {
                            continue;
                        }
                        verified += 1;
                        if exact(rx, ry) {
                            list.push((rx as u32, ry as u32));
                        }
                        // Bound worst-case work on pathological repeat content.
                        if verified >= 128 {
                            break;
                        }
                    }
                    IbcMatches::List(list)
                };
                entry.insert(m)
            }
        };
        let list = match matches {
            IbcMatches::Default(rx, ry) => return make(*rx, *ry),
            IbcMatches::List(list) => list,
        };
        #[allow(clippy::type_complexity)]
        let mut best: Option<(usize, usize, (i16, i16), (i16, i16))> = None;
        let mut best_cost = u32::MAX;
        for &(rx, ry) in list.iter() {
            if let Some(found) = make(rx as usize, ry as usize) {
                let cost = (i32::from(found.2.0) - i32::from(found.3.0)).unsigned_abs()
                    + (i32::from(found.2.1) - i32::from(found.3.1)).unsigned_abs();
                if cost < best_cost {
                    best_cost = cost;
                    best = Some(found);
                }
            }
        }
        best
    }

    /// R-D cost of coding a `size`-px square as a skip IntraBC copy: the
    /// residual is dropped (skip = 1), so distortion is the quantization
    /// drift already present in the reference reconstruction.
    fn rd_cost_intrabc(
        &self,
        px: usize,
        py: usize,
        size: usize,
        thr: bool,
        prdo: f32,
    ) -> Option<f32> {
        if size == 64 && self.aq.enabled && self.aq.pending != 0 {
            return None;
        }
        let (rx, ry, mv, pred) = self.find_intrabc(px, py, size, thr)?;
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

    fn code_block64_intrabc(&mut self, x8: usize, y8: usize, thr: bool) {
        // Guarded by rd_cost_intrabc: a delta-carrying SB must never take the
        // whole-64 skip path (the decoder would not read the armed token).
        debug_assert!(!self.aq.enabled || self.aq.pending == 0);
        self.aq_cancel_skipped_sb();
        self.code_intrabc_block(x8, y8, 64, thr);
    }

    /// Code a `size`-px square (16/32/64) as a skip IntraBC copy: skip = 1,
    /// use_intrabc = 1, DV residual, no coefficients. Reconstruction is an
    /// integer copy of all coded planes (candidates are chroma-parity-even).
    fn code_intrabc_block(&mut self, x8: usize, y8: usize, size: usize, thr: bool) {
        #[cfg(test)]
        LOSSY_INTRABC_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (px, py) = (x8 * 8, y8 * 8);
        let (rx, ry, mv, pred) = self
            .find_intrabc(px, py, size, thr)
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
                self.none64_split_bias_at()
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
        let rd_ibc = self.rd_cost_intrabc(px, py, 64, thr, prdo);
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

#[cfg(test)]
mod ibc_index_tests {
    use super::LossyIbcIndex;

    /// The index rolls the 8x8 window hash; queries use the direct form. They
    /// must agree at every indexed position, or no lookup can ever hit.
    #[test]
    fn rolling_hash_matches_direct() {
        let (w, h) = (37usize, 29usize);
        let mut seed = 0x1234_5678u32;
        let luma: Vec<u16> = (0..w * h)
            .map(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                (seed % 1024) as u16
            })
            .collect();
        let src = [luma.clone(), vec![0; w * h], vec![0; w * h]];
        for sub in [(0, 0), (1, 1), (1, 0)] {
            let idx = LossyIbcIndex::build(&src, w, h, sub);
            let mut n = 0;
            for &(hash, origin) in &idx.entries {
                let (x, y) = ((origin & 0xffff) as usize, (origin >> 16) as usize);
                assert_eq!(x % (1 << sub.0), 0);
                assert_eq!(y % (1 << sub.1), 0);
                assert_eq!(hash, LossyIbcIndex::hash8(&luma, w, x, y), "({x},{y})");
                n += 1;
            }
            assert_eq!(n, (w - 7).div_ceil(1 << sub.0) * (h - 7).div_ceil(1 << sub.1));
        }
    }

    /// A glyph block whose top-left 8x8 is flat background must still find its
    /// earlier copy through the textured anchor.
    #[test]
    fn anchor_finds_copy_behind_flat_corner() {
        let (w, h) = (128usize, 64usize);
        let mut luma = vec![200u16; w * h];
        // a 16x16 block at (8, 8) and its copy at (72, 40): flat top-left
        // 8x8, "ink" in the bottom-right 8x8
        for (bx, by) in [(8usize, 8usize), (72, 40)] {
            for j in 8..16 {
                for i in 8..16 {
                    luma[(by + j) * w + bx + i] = ((i * 7 + j * 13) % 50) as u16;
                }
            }
        }
        let src = [luma.clone(), vec![0; w * h], vec![0; w * h]];
        let idx = LossyIbcIndex::build(&src, w, h, (0, 0));
        assert_eq!(LossyIbcIndex::anchor(&luma, w, 72, 40, 16), (8, 8));
        let hits: Vec<_> = idx.candidates(&luma, w, 72, 40, 16, h).collect();
        assert!(hits.contains(&(8, 8)), "{hits:?}");
        assert!(hits.len() <= 2, "flat-free anchor must give a tiny bucket: {hits:?}");
    }
}
