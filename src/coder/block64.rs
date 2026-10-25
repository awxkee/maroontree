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

/// How a whole-64 luma is transformed: `tx_depth` 1 (four TX_32X32, the
/// default), 0 (one TX_64X64, `tx64`) or 2 (sixteen TX_16X16, `tx64_depth2`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Luma64Tx {
    Quad32,
    Tx64,
    Grid16,
}

/// SSE against the wavefront's raw shared reconstruction
#[allow(clippy::too_many_arguments)]
unsafe fn sse_u16_raw_reference(
    src: crate::rd_sse::SrcBlock<'_>,
    reference: *const u16,
    reference_len: usize,
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
) -> i64 {
    // Visible part only, like every other distortion (see `SrcBlock`).
    let (w, h) = src.vis();
    debug_assert!(h == 0 || (ref_y + h - 1) * ref_stride + ref_x + w <= reference_len);
    let mut sse = 0i64;
    for row in 0..h {
        let src_row = &src.row(row)[..w];
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

/// A residual-coded IntraBC leaf (`Tuning::ibc_residual`): an approximate copy
/// with the prediction error coded through one luma transform of block size
/// (DCT_DCT, inter tx-type set) and the decoder-derived chroma transform, as an
/// AV1 INTER block with skip = 0. Predictions, levels and dequantised residuals
/// are kept so the emitter writes exactly what the decision priced.
struct IbcResidPlan {
    mv: (i16, i16),
    pred: (i16, i16),
    cost: f32,
    lpred: Vec<i32>,
    lcf: Vec<i32>,
    lrr: Vec<i32>,
    cpred: [Vec<i32>; 2],
    ccf: [Vec<i32>; 2],
    crr: [Vec<i32>; 2],
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

/// IntraBC chroma prediction on a subsampled plane: `(kx, ky)` is the
/// integer reference origin (`luma_ref >> sub`), `(fx, fy)` whether the
/// luma DV is odd along that subsampled axis, which puts the chroma position
/// at a half sample. AV1 predicts IntraBC with the BILINEAR 8-tap filter
/// (7.11.3.4): at the half position the taps are {64, 64} on the floor and
/// floor+1 samples, and with InterRound0/InterRound1 = (3, 11) (12-bit:
/// (5, 9)) the two passes reduce exactly to `(a + b + 1) >> 1` for one
/// half-sample axis and `(a + b + c + d + 2) >> 2` for both — at every bit
/// depth (checked against the generic convolution in the tests). Integer
/// positions copy. `get(x, y)` reads the reference plane; `out` is `bw` x
/// `bh`, stride `bw`.
#[allow(clippy::too_many_arguments)]
fn ibc_chroma_pred(
    get: &dyn Fn(usize, usize) -> u32,
    kx: usize,
    ky: usize,
    fx: bool,
    fy: bool,
    bw: usize,
    bh: usize,
    out: &mut [u16],
) {
    for r in 0..bh {
        for c in 0..bw {
            let (x, y) = (kx + c, ky + r);
            let v = match (fx, fy) {
                (false, false) => get(x, y),
                (true, false) => (get(x, y) + get(x + 1, y) + 1) >> 1,
                (false, true) => (get(x, y) + get(x, y + 1) + 1) >> 1,
                (true, true) => (get(x, y) + get(x + 1, y) + get(x, y + 1) + get(x + 1, y + 1) + 2) >> 2,
            };
            out[r * bw + c] = v as u16;
        }
    }
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
        // Only eight horizontal rows feed the current vertical window. Keep
        // them in a ring instead of retaining row hashes for the whole tile.
        let mut rows = vec![0u32; 8 * nx];
        let hash_row = |y: usize, out: &mut [u32]| {
            let line = &luma[y * w..][..w];
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
        };
        for y in 0..8 {
            hash_row(y, &mut rows[y * nx..][..nx]);
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
                let row = &mut rows[((y - 1) & 7) * nx..][..nx];
                for (c, &old) in col.iter_mut().zip(row.iter()) {
                    *c = c.wrapping_sub(old.wrapping_mul(q7)).wrapping_mul(IBC_HQ);
                }
                hash_row(y + 7, row);
                for (c, &new) in col.iter_mut().zip(row.iter()) {
                    *c = c.wrapping_add(new);
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
        drop(rows);
        drop(col);
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
        if !self.speed.at_least_slow() {
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
                score += self.rd.satd_sad_proxy(self.src_blk(0, bx, by, 32, 32), &pred[..], 32);
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
        let sby = py / 64 * 64;
        // Legality (tile, odd-DV chroma taps, wavefront cone, DV range and
        // conformance) lives in `ibc_legal`; odd DVs (`ibc_odd_dv`) predict
        // the subsampled planes at the half sample with the BILINEAR filter
        // (`ibc_chroma_pred`), exactly as the decoders do.
        // Half-sample flags of a reference origin on the subsampled planes
        // (`px`, `py` are 8-aligned, so the DV parity is the origin parity).
        let fracs = |rx: usize, ry: usize| -> (bool, bool) {
            (
                (self.ss420 || self.ss422) && !self.mono && rx & 1 != 0,
                self.ss420 && !self.mono && ry & 1 != 0,
            )
        };
        let exact = |rx: usize, ry: usize| {
            let (fx, fy) = fracs(rx, ry);
            (0..if self.mono { 1 } else { 3 }).all(|plane| {
                let sx = usize::from(plane != 0 && (self.ss420 || self.ss422));
                let sy = usize::from(plane != 0 && self.ss420);
                let stride = if plane == 0 { self.w } else { self.cw };
                let (x, y, ref_x, ref_y) = (px >> sx, py >> sy, rx >> sx, ry >> sy);
                let (bw, bh) = (size >> sx, size >> sy);
                if plane != 0 && (fx || fy) {
                    // Exactness of the FILTERED prediction against the source.
                    let src = &self.src[plane];
                    let get = |xx: usize, yy: usize| u32::from(src[yy * stride + xx]);
                    let mut pred = [0u16; 32 * 32];
                    ibc_chroma_pred(&get, ref_x, ref_y, fx, fy, bw, bh, &mut pred[..bw * bh]);
                    return (0..bh).all(|row| {
                        src[(y + row) * stride + x..][..bw] == pred[row * bw..][..bw]
                    });
                }
                (0..bh).all(|row| {
                    self.src[plane][(y + row) * stride + x..][..bw]
                        == self.src[plane][(ref_y + row) * stride + ref_x..][..bw]
                })
            })
        };
        let legal = |rx: usize, ry: usize| self.ibc_legal(px, py, size, rx, ry);
        let make = |rx: usize, ry: usize| self.ibc_make_mv(px, py, rx, ry, pred);
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
                    let idx = self.ibc_index_ref()?;
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


    /// Whether (rx, ry) is a legal IntraBC reference origin for the `size`
    /// block at (px, py): inside the tile, chroma taps included for odd DVs,
    /// not in the current/later superblocks of this row, inside the capture
    /// wavefront's finished cone (`ibc_wavefront_reach`), within DV range and
    /// `av1_is_dv_valid`-conformant. Pure function of geometry and format, so
    /// every thread count agrees.
    fn ibc_legal(&self, px: usize, py: usize, size: usize, rx: usize, ry: usize) -> bool {
        let sbx = px / 64 * 64;
        let sby = py / 64 * 64;
        let odd_ok = self.ibc_odd_dv_allowed();
        let (need_ex, need_ey) = if self.mono || odd_ok {
            (false, false)
        } else {
            (self.ss420 || self.ss422, self.ss420)
        };
        let fracs = |rx: usize, ry: usize| -> (bool, bool) {
            (
                (self.ss420 || self.ss422) && !self.mono && rx & 1 != 0,
                self.ss420 && !self.mono && ry & 1 != 0,
            )
        };
        // A half-sample chroma position reads one more sample right/below
        // the block's chroma footprint, i.e. the luma column/row just past
        // the reference block: the footprint checked below grows by 2 luma
        // samples on that axis so every tap is in-tile and finished
        // (libaom does not extend borders for IntraBC; dav1d clamps).
        let (fx, fy) = fracs(rx, ry);
        let (ex, ey) = (size + 2 * usize::from(fx), size + 2 * usize::from(fy));
        if rx + ex > self.w
            || ry + ey > self.h
            || ry + ey > sby + 64
            || (ry + ey > sby && rx + ex > sbx)
            || (need_ex && rx & 1 != 0)
            || (need_ey && ry & 1 != 0)
        {
            return false;
        }
        // Wavefront restriction, applied at EVERY thread count so serial
        // and captured decisions stay byte-identical: under the capture
        // schedule (deps left/top and (r-1, c+reach)) the finished set
        // when cell (r, c) starts is {r' < r: c' <= c + reach*(r - r')}
        // plus {r' = r: c' < c}. Every superblock the reference spans must
        // be inside it. `ibc_wavefront_reach` 5 would contain every
        // `av1_is_dv_valid` source (gradient 1 + INTRABC_DELAY_SB64); the
        // shipped 2 covers the row above completely (where the copies on
        // repetitive content live) and nearly all of the second row. The
        // old reach 1 admitted 1% of s_manga's exact 8x8 sources vs 56%
        // legal; 2 recovers 98% of the resulting -10% on manga.
        let reach = (crate::tuning::get().ibc_wavefront_reach as usize).max(1);
        let (cr, cc) = (py / 64, px / 64);
        let c1 = (rx + ex - 1) / 64;
        let (r0, r1) = (ry / 64, (ry + ey - 1) / 64);
        if !(r0..=r1).all(|r| if r < cr { c1 <= cc + reach * (cr - r) } else { c1 < cc }) {
            return false;
        }
        crate::tile::intrabc_dv_in_range(px, py, rx, ry)
            && crate::tile::intrabc_dv_conformant_rect(px, py, rx, ry, ex, ey, self.w)
    }

    /// DV (and its predictor) of the reference origin (rx, ry) for the block
    /// at (px, py), or None when the DV residual leaves the codable range.
    #[allow(clippy::type_complexity)]
    fn ibc_make_mv(
        &self,
        px: usize,
        py: usize,
        rx: usize,
        ry: usize,
        pred: (i16, i16),
    ) -> Option<(usize, usize, (i16, i16), (i16, i16))> {
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
    }

    /// The tile's IntraBC hash index (built on first use; every position when
    /// odd DVs are allowed, else the chroma-parity grid).
    fn ibc_index_ref(&self) -> Option<&LossyIbcIndex> {
        let odd_ok = self.ibc_odd_dv_allowed();
        Some(self.ibc_index?.get_or_init(|| {
            let sub = if self.mono || odd_ok {
                (0, 0)
            } else {
                (
                    usize::from(self.ss420 || self.ss422),
                    usize::from(self.ss420),
                )
            };
            LossyIbcIndex::build(self.src, self.w, self.h, sub)
        }))
    }

    /// Reference block of `plane` for the copy (rx, ry) -> (px, py): integer
    /// copy, or the bilinear half-sample chroma of an odd DV; read from the
    /// shared finished planes under capture, the local recon otherwise.
    #[allow(clippy::too_many_arguments)]
    fn ibc_reference(&self, plane: usize, rx: usize, ry: usize, size: usize, out: &mut [i32]) {
        let sx = usize::from(plane != 0 && (self.ss420 || self.ss422));
        let sy = usize::from(plane != 0 && self.ss420);
        let stride = if plane == 0 { self.w } else { self.cw };
        let (ref_x, ref_y) = (rx >> sx, ry >> sy);
        let (bw, bh) = (size >> sx, size >> sy);
        let (fx, fy) = (sx == 1 && rx & 1 != 0, sy == 1 && ry & 1 != 0);
        let fill = |get: &dyn Fn(usize, usize) -> u32, out: &mut [i32]| {
            if fx || fy {
                let mut blk = [0u16; 32 * 32];
                ibc_chroma_pred(get, ref_x, ref_y, fx, fy, bw, bh, &mut blk[..bw * bh]);
                for (o, &v) in out[..bw * bh].iter_mut().zip(&blk[..bw * bh]) {
                    *o = i32::from(v);
                }
            } else {
                for r in 0..bh {
                    for c in 0..bw {
                        out[r * bw + c] = get(ref_x + c, ref_y + r) as i32;
                    }
                }
            }
        };
        if let Some(sh) = self.ibc_shared {
            let (ptr, len, _) = sh.planes[plane];
            let get = |xx: usize, yy: usize| {
                let off = yy * stride + xx;
                debug_assert!(off < len);
                // SAFETY: finished-cell read (legality keeps the footprint
                // plus taps inside finished cells).
                u32::from(unsafe { *ptr.add(off) })
            };
            fill(&get, out);
        } else {
            let rec = &self.recon[plane];
            let get = |xx: usize, yy: usize| u32::from(rec[yy * stride + xx]);
            fill(&get, out);
        }
    }

    /// Residual-coded IntraBC leaves (`Tuning::ibc_residual`): 16 / 32 on
    /// IntraBC frames at 4:2:0 / 4:4:4 (4:2:2 chroma would need the rect
    /// chroma transforms), Slow+.
    fn ibc_residual_allowed(&self, size: usize) -> bool {
        // The test build always exercises the path (knobs are const without
        // the `tuning` feature): the dav1d oracle and the t1 == tN tests then
        // cover the inter syntax even while the knob ships off.
        (crate::tuning::get().ibc_residual || cfg!(test))
            && self.allow_intrabc
            && !self.mono
            && !self.ss422
            && (size == 16 || size == 32)
            && self.speed.at_least_slow()
    }

    /// Source-domain luma SAD of the copy (rx, ry) -> (px, py).
    fn ibc_src_sad(&self, px: usize, py: usize, rx: usize, ry: usize, size: usize) -> u64 {
        let src = &self.src[0];
        let mut sad = 0u64;
        for r in 0..size {
            let a = &src[(py + r) * self.w + px..][..size];
            let b = &src[(ry + r) * self.w + rx..][..size];
            for (&x, &y) in a.iter().zip(b) {
                sad += u64::from(x.abs_diff(y));
            }
        }
        sad
    }

    /// Approximate-copy candidates for a residual IntraBC leaf: the DV
    /// predictor, the above/left neighbours' DVs and the exact 8x8 hash hits
    /// of every textured 8x8 sub-block (translated to the block origin), all
    /// legal; ranked by source luma SAD, the best `ibc_residual_cands` kept.
    fn ibc_residual_candidates(&self, px: usize, py: usize, size: usize, thr: bool) -> Vec<(usize, usize)> {
        let mut cands: Vec<(usize, usize)> = Vec::new();
        let consider = |rx: usize, ry: usize, cands: &mut Vec<(usize, usize)>| {
            if (rx, ry) != (px, py) && !cands.contains(&(rx, ry)) && self.ibc_legal(px, py, size, rx, ry) {
                cands.push((rx, ry));
            }
        };
        let from_mv = |mv: (i16, i16)| -> Option<(usize, usize)> {
            let (dy, dx) = (i32::from(mv.0) / 8, i32::from(mv.1) / 8);
            let rx = usize::try_from(px as i32 + dx).ok()?;
            let ry = usize::try_from(py as i32 + dy).ok()?;
            Some((rx, ry))
        };
        let pred = self.intrabc_predictor(px, py, size, thr);
        if let Some((rx, ry)) = from_mv(pred) {
            consider(rx, ry, &mut cands);
        }
        let (bx4, by4, stride4) = (px / 4, py / 4, self.w / 4);
        if by4 > 0 && let Some(mv) = self.ibc_mv[(by4 - 1) * stride4 + bx4] && let Some((rx, ry)) = from_mv(mv) {
            consider(rx, ry, &mut cands);
        }
        if bx4 > 0 && let Some(mv) = self.ibc_mv[by4 * stride4 + bx4 - 1] && let Some((rx, ry)) = from_mv(mv) {
            consider(rx, ry, &mut cands);
        }
        if let Some(idx) = self.ibc_index_ref() {
            let sby = py / 64 * 64;
            let max_y = (sby + 64).saturating_sub(8);
            let luma = &self.src[0];
            for oy in (0..size).step_by(8) {
                for ox in (0..size).step_by(8) {
                    // Flat sub-blocks hash like every other flat 8x8: no
                    // information about the copy, and a huge bucket.
                    let (sx0, sy0) = (px + ox, py + oy);
                    let (mut lo, mut hi) = (u16::MAX, 0u16);
                    for r in 0..8 {
                        for &v in &luma[(sy0 + r) * self.w + sx0..][..8] {
                            lo = lo.min(v);
                            hi = hi.max(v);
                        }
                    }
                    if hi == lo {
                        continue;
                    }
                    let (mut taken, mut scanned) = (0usize, 0usize);
                    for (ax, ay) in idx.candidates(luma, self.w, sx0, sy0, 8, max_y) {
                        scanned += 1;
                        if scanned > 64 || taken >= 4 {
                            break;
                        }
                        let (Some(rx), Some(ry)) = (ax.checked_sub(ox), ay.checked_sub(oy)) else {
                            continue;
                        };
                        let before = cands.len();
                        consider(rx, ry, &mut cands);
                        taken += usize::from(cands.len() > before);
                    }
                }
            }
        }
        let keep = crate::tuning::get().ibc_residual_cands as usize;
        let mut scored: Vec<(u64, usize, usize)> = cands
            .into_iter()
            .map(|(rx, ry)| (self.ibc_src_sad(px, py, rx, ry, size), rx, ry))
            .collect();
        scored.sort_unstable();
        scored.truncate(keep);
        scored.into_iter().map(|(_, rx, ry)| (rx, ry)).collect()
    }

    /// Full R-D of one residual IntraBC candidate: luma one TX of block size
    /// (DCT_DCT, ctx trellis, exact inter rate), chroma the derived DCT, the
    /// DV / flag / `txfm_split` header. Reference pixels come from the
    /// reconstruction (finished cells), so capture and serial agree.
    #[allow(clippy::too_many_arguments)]
    fn ibc_residual_rd(
        &self,
        px: usize,
        py: usize,
        size: usize,
        rx: usize,
        ry: usize,
        thr: bool,
        prdo: f32,
    ) -> Option<IbcResidPlan> {
        let pred_dv = self.intrabc_predictor(px, py, size, thr);
        let (rx, ry, mv, pred_mv) = self.ibc_make_mv(px, py, rx, ry, pred_dv)?;
        let n = size * size;
        let mlam = self.mlam() * prdo;
        let lam = trellis_lambda() * prdo;
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (bx4, by4) = (px / 4, py / 4);
        let mut lpred = vec![0i32; n];
        self.ibc_reference(0, rx, ry, size, &mut lpred);
        let mut resid = vec![0i32; n];
        self.rd.residual_pred(&mut resid, &lpred, &self.src[0], self.w, px, py, size, size);
        let c = self.dcdf();
        let (lcf, lrr, lbits): (Vec<i32>, Vec<i32>, f32) = if size == 16 {
            let r: &[i32; 256] = resid[..].try_into().unwrap();
            let (mut cf, tf) = self.dct.dct16x16_t(r, &self.quant);
            trellis_optimize_ctx(
                &mut cf,
                &tf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                16,
                c,
                2,
                0,
                &c.eob_bin_256_l,
                self.dc_sign_ctx_16(0, bx4, by4),
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.idct_dequant_16x16(&cf, &self.quant);
            let bits = self.luma_bits_inter(&cf, &SCAN_16X16, 16, px, py);
            (cf.to_vec(), rr.to_vec(), bits)
        } else {
            let r: &[i32; 1024] = resid[..].try_into().unwrap();
            let (mut cf, tf) = self.dct.dct32x32_t(r, &self.quant);
            trellis_optimize_ctx(
                &mut cf,
                &tf,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                32,
                c,
                3,
                0,
                &c.eob_bin_1024_l,
                self.dc_sign_ctx_32(0, bx4, by4),
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
            let bits = self.luma_bits_inter(&cf, &SCAN_32X32, 32, px, py);
            (cf.to_vec(), rr.to_vec(), bits)
        };
        // The node's other candidates (NONE / SPLIT / rect proxies) carry the
        // SATD-scaled partition distortion, not SSE — price this leaf in the
        // same units or an approximate copy gets an ~18x distortion discount
        // (measured: x_screen +33% bytes at SS2 -11 with SSE here).
        let dist_l = self.luma_partition_distortion(px, py, size, size, acq, &lpred, 0, &lrr);
        // Header: the skip variant's DV model + the `txfm_split` = 0 bool
        // (dav1d `read_tx_tree`: cat = 2 * (TX_64X64 - max), ctx = a + l with
        // a = above tx width < this; the initial neighbour value is TX_64X64).
        let dv_px = ((i32::from(mv.0) - i32::from(pred_mv.0)).unsigned_abs()
            + (i32::from(mv.1) - i32::from(pred_mv.1)).unsigned_abs()) as f32
            / 8.0;
        let lw = size.trailing_zeros() as usize - 2;
        let cat = 2 * (4 - lw);
        let a = usize::from(self.a_tx[bx4] >= 0 && (self.a_tx[bx4] as usize) < lw);
        let l = usize::from(self.l_tx[by4] >= 0 && (self.l_tx[by4] as usize) < lw);
        let hdr_bits = 8.0 + dirty_log2f(dv_px.max(1.0)) * 2.0 + cdf_cost(&c.txpart[cat][a + l], 0);
        let mut cost = crate::partition_rd::rd_cost(dist_l, mlam, lbits + hdr_bits);
        // Chroma (4:2:0 half dims, 4:4:4 full): the derived DCT transform.
        let (sx, sy) = (usize::from(self.ss420), usize::from(self.ss420));
        let (cw, chh) = (size >> sx, size >> sy);
        let cn = cw * chh;
        let (cx, cy) = (px >> sx, py >> sy);
        let mlam_c = self.emit_mlam_c(px, py, size);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let mut cpred = [vec![0i32; cn], vec![0i32; cn]];
        let mut ccf = [Vec::new(), Vec::new()];
        let mut crr = [Vec::new(), Vec::new()];
        for ci in 0..2 {
            let plane = ci + 1;
            self.ibc_reference(plane, rx, ry, size, &mut cpred[ci]);
            let mut cres = vec![0i32; cn];
            self.rd.residual_pred(&mut cres, &cpred[ci], &self.src[plane], self.cw, cx, cy, cw, chh);
            let (q, rr, bits): (Vec<i32>, Vec<i32>, f32) = match cw {
                8 => {
                    let r: &[i32; 64] = cres[..].try_into().unwrap();
                    let (mut q, qt) = self.dct.dct8x8_t(r, &self.cquant);
                    self.chroma_rect_trellis(&mut q, &qt, cdcq, cacq, &SCAN_8X8, lam, 8, 8, plane, cx, cy);
                    self.rd.preserve_dc(&mut q[0], &cres);
                    let rr = self.idct.idct_dequant_8x8(&q, &self.cquant);
                    let bits = self.chroma_bits(&q, &SCAN_8X8, 8, plane, cx, cy);
                    (q.to_vec(), rr.to_vec(), bits)
                }
                16 => {
                    let r: &[i32; 256] = cres[..].try_into().unwrap();
                    let (mut q, qt) = self.dct.dct16x16_t(r, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_16X16, lam);
                    self.rd.preserve_dc(&mut q[0], &cres);
                    let rr = self.idct.idct_dequant_16x16(&q, &self.cquant);
                    let bits = self.chroma_bits(&q, &SCAN_16X16, 16, plane, cx, cy);
                    (q.to_vec(), rr.to_vec(), bits)
                }
                _ => {
                    let r: &[i32; 1024] = cres[..].try_into().unwrap();
                    let (mut q, qt) = self.dct.dct32x32_t(r, &self.cquant);
                    trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_32X32, lam);
                    self.rd.preserve_dc(&mut q[0], &cres);
                    let rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
                    let bits = self.chroma_bits(&q, &SCAN_32X32, 32, plane, cx, cy);
                    (q.to_vec(), rr.to_vec(), bits)
                }
            };
            let d = self.rd.sse_recon(&cpred[ci], &rr, self.src_blk(plane, cx, cy, cw, chh), self.bd);
            cost += rd_cost_i64(d, mlam_c, bits);
            ccf[ci] = q;
            crr[ci] = rr;
        }
        let _ = (rx, ry);
        Some(IbcResidPlan {
            mv,
            pred: pred_mv,
            cost,
            lpred,
            lcf,
            lrr,
            cpred,
            ccf,
            crr,
        })
    }

    /// The IntraBC plan for the `size` block: the cheaper of the exact skip
    /// copy (`rd_cost_intrabc_skip`) and the best residual-coded approximate
    /// copy, with its cost. Decision and emitter call the same function.
    fn ibc_plan(&self, px: usize, py: usize, size: usize, thr: bool, prdo: f32) -> Option<(f32, Option<IbcResidPlan>)> {
        let mut best: Option<(f32, Option<IbcResidPlan>)> =
            self.rd_cost_intrabc_skip(px, py, size, thr, prdo).map(|c| (c, None));
        if self.ibc_residual_allowed(size) {
            for (rx, ry) in self.ibc_residual_candidates(px, py, size, thr) {
                if let Some(plan) = self.ibc_residual_rd(px, py, size, rx, ry, thr, prdo)
                    && best.as_ref().is_none_or(|b| plan.cost < b.0)
                {
                    best = Some((plan.cost, Some(plan)));
                }
            }
        }
        best
    }

    /// R-D cost of the IntraBC leaf at (px, py): exact skip copy or
    /// residual-coded approximate copy, whichever is cheaper.
    fn rd_cost_intrabc(&self, px: usize, py: usize, size: usize, thr: bool, prdo: f32) -> Option<f32> {
        self.ibc_plan(px, py, size, thr, prdo).map(|p| p.0)
    }

    /// Emit a residual-coded IntraBC block (AV1 inter syntax in an intra
    /// frame): skip = 0, delta-q token, use_intrabc = 1, DV, `txfm_split` = 0,
    /// luma coefficients with the inter tx-type symbol, chroma coefficients
    /// with the derived type; reconstruction = reference + dequantised
    /// residual.
    fn code_intrabc_residual_block(&mut self, x8: usize, y8: usize, size: usize, plan: IbcResidPlan) {
        #[cfg(test)]
        LOSSY_INTRABC_RESID_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4, stride4, n4) = (px / 4, py / 4, self.w / 4, size / 4);
        let maxv = (1i32 << self.bd) - 1;
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.enc.encode_symbol(0, &mut self.cdfs.skip[sctx]);
        if !self.cdef_point_marked {
            self.cdef_point_marked = true;
            self.enc.trace_cdef_mark();
        }
        self.code_delta_q_if_armed();
        self.enc.encode_symbol(1, &mut self.cdfs.intrabc);
        self.encode_intrabc_mv(plan.mv, plan.pred);
        // Var-tx tree: one TX of block size (no split).
        let lw = size.trailing_zeros() as usize - 2;
        let cat = 2 * (4 - lw);
        let a = usize::from(self.a_tx[bx4] >= 0 && (self.a_tx[bx4] as usize) < lw);
        let l = usize::from(self.l_tx[by4] >= 0 && (self.l_tx[by4] as usize) < lw);
        self.enc.encode_symbol(0, &mut self.cdfs.txpart[cat][a + l]);

        self.record_blk(x8, y8, n4 as u8);
        self.mark_skip8_rect(x8, y8, size / 8, size / 8, false);
        let txc = lw as i8;
        self.a_skip[bx4..bx4 + n4].fill(0);
        self.l_skip[by4..by4 + n4].fill(0);
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
        for slot in &mut self.a_palette_uv[bx4..bx4 + n4] {
            slot.clear();
        }
        for slot in &mut self.l_palette_uv[by4..by4 + n4] {
            slot.clear();
        }
        for y in by4..by4 + n4 {
            self.ibc_mv[y * stride4 + bx4..y * stride4 + bx4 + n4].fill(Some(plan.mv));
        }
        // Luma: coefficients (txb_skip ctx 0, block-span DC-sign ctx) and recon.
        let res_ctx = if size == 16 {
            let cf: &[i32; 256] = plan.lcf[..].try_into().unwrap();
            let ds = self.dc_sign_ctx_16(0, bx4, by4);
            encode_tx16_coeffs_inter_dct(&mut self.enc, &mut self.cdfs, cf, 0, ds)
        } else {
            let cf: &[i32; 1024] = plan.lcf[..].try_into().unwrap();
            let ds = self.dc_sign_ctx_32(0, bx4, by4);
            encode_tx32_coeffs_inter_dct(&mut self.enc, &mut self.cdfs, cf, 0, ds)
        };
        self.a_coef[0][bx4..bx4 + n4].fill(res_ctx);
        self.l_coef[0][by4..by4 + n4].fill(res_ctx);
        for r in 0..size {
            let dst = &mut self.recon[0][(py + r) * self.w + px..][..size];
            recon_add_pred(dst, &plan.lpred[r * size..], &plan.lrr[r * size..], maxv);
        }
        // Chroma planes: derived DCT, recon = reference + residual.
        let (sx, sy) = (usize::from(self.ss420), usize::from(self.ss420));
        let (cw, chh) = (size >> sx, size >> sy);
        let (cx, cy) = (px >> sx, py >> sy);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let cn4 = (cw / 4).max(1);
        for ci in 0..2 {
            let plane = ci + 1;
            let cres = match cw {
                8 => {
                    let q: &[i32; 64] = plan.ccf[ci][..].try_into().unwrap();
                    let sk = self.skip_ctx(plane, cbx4, cby4, true);
                    let ds = self.dc_sign_ctx(plane, cbx4, cby4);
                    encode_tx8_coeffs_adapt(&mut self.enc, &mut self.cdfs, q, true, sk, ds, 0, 1)
                }
                16 => {
                    let q: &[i32; 256] = plan.ccf[ci][..].try_into().unwrap();
                    let sk = self.skip_ctx_16(plane, cbx4, cby4, true);
                    let ds = self.dc_sign_ctx_16(plane, cbx4, cby4);
                    encode_tx16_coeffs_adapt(&mut self.enc, &mut self.cdfs, q, true, sk, ds, 0, 1)
                }
                _ => {
                    let q: &[i32; 1024] = plan.ccf[ci][..].try_into().unwrap();
                    let sk = self.skip_ctx_32(plane, cbx4, cby4, true);
                    let ds = self.dc_sign_ctx_32(plane, cbx4, cby4);
                    encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, q, true, sk, ds)
                }
            };
            self.a_coef[plane][cbx4..cbx4 + cn4].fill(cres);
            self.l_coef[plane][cby4..cby4 + cn4].fill(cres);
            for r in 0..chh {
                let dst = &mut self.recon[plane][(cy + r) * self.cw + cx..][..cw];
                recon_add_pred(dst, &plan.cpred[ci][r * cw..], &plan.crr[ci][r * cw..], maxv);
            }
        }
    }

    /// R-D cost of coding a `size`-px square as a skip IntraBC copy: the
    /// residual is dropped (skip = 1), so distortion is the quantization
    /// drift already present in the reference reconstruction.
    fn rd_cost_intrabc_skip(
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
                        self.src_blk(plane, x, y, bw, bh),
                        ptr,
                        len,
                        stride,
                        ref_x,
                        ref_y,
                    )
                };
            } else {
                distortion += self.rd.sse_u16(
                    self.src_blk(plane, x, y, bw, bh),
                    &self.recon[plane],
                    stride,
                    ref_x,
                    ref_y,
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

    /// Code a `size`-px square (8/16/32/64) as a skip IntraBC copy: skip = 1,
    /// use_intrabc = 1, DV residual, no coefficients. Reconstruction is an
    /// integer copy of every plane, except that an odd DV (`ibc_odd_dv`)
    /// predicts the subsampled planes at the half sample with the BILINEAR
    /// filter (`ibc_chroma_pred`).
    fn code_intrabc_block(&mut self, x8: usize, y8: usize, size: usize, thr: bool) {
        #[cfg(test)]
        LOSSY_INTRABC_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (px, py) = (x8 * 8, y8 * 8);
        if self.ibc_residual_allowed(size) {
            let prdo = self.perceptual_rd_scale(px, py, size);
            if let Some((_, Some(plan))) = self.ibc_plan(px, py, size, thr, prdo) {
                self.code_intrabc_residual_block(x8, y8, size, plan);
                return;
            }
        }
        let (rx, ry, mv, pred) = self
            .find_intrabc(px, py, size, thr)
            .expect("legal IntraBC reference");
        let (fx, fy) = (
            (self.ss420 || self.ss422) && !self.mono && rx & 1 != 0,
            self.ss420 && !self.mono && ry & 1 != 0,
        );
        #[cfg(test)]
        if fx || fy {
            LOSSY_INTRABC_ODD_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
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
            if plane != 0 && (fx || fy) {
                // Half-sample chroma: predict from the reference (shared
                // finished planes under capture, local recon otherwise) into
                // a scratch block first — the +1 taps never touch this block,
                // but the destination rows alias the recon being read.
                let mut blk = [0u16; 32 * 32];
                if let Some(sh) = self.ibc_shared {
                    let (ptr, len, _) = sh.planes[plane];
                    let get = |xx: usize, yy: usize| {
                        let off = yy * stride + xx;
                        debug_assert!(off < len);
                        // SAFETY: finished-cell read (legality keeps the
                        // footprint plus taps inside finished cells).
                        u32::from(unsafe { *ptr.add(off) })
                    };
                    ibc_chroma_pred(&get, ref_x, ref_y, fx, fy, bw, bh, &mut blk[..bw * bh]);
                } else {
                    let rec = &self.recon[plane];
                    let get = |xx: usize, yy: usize| u32::from(rec[yy * stride + xx]);
                    ibc_chroma_pred(&get, ref_x, ref_y, fx, fy, bw, bh, &mut blk[..bw * bh]);
                }
                for row in 0..bh {
                    self.recon[plane][(y + row) * stride + x..][..bw]
                        .copy_from_slice(&blk[row * bw..][..bw]);
                }
            } else if let Some(sh) = self.ibc_shared {
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

    /// Odd IntraBC DVs with bilinear half-sample chroma (`ibc_odd_dv`): only
    /// meaningful on subsampled formats.
    #[inline]
    fn ibc_odd_dv_allowed(&self) -> bool {
        crate::tuning::get().ibc_odd_dv && !self.mono && (self.ss420 || self.ss422)
    }

    /// TX_64X64 for whole-64 luma (`tx64` knob).
    fn tx64_allowed(&self) -> bool {
        crate::tuning::get().tx64
    }

    /// Bits of the whole-64 luma `tx_depth` symbol (`txsz[3]`, max TX_64X64):
    /// depth 0 = TX_64X64, depth 1 = four TX_32X32.
    fn tx_depth64_bits(&self, px: usize, py: usize, depth: usize) -> f32 {
        let ctx = (self.l_tx[py / 4] >= 4) as usize + (self.a_tx[px / 4] >= 4) as usize;
        cdf_cost(&self.dcdf().txsz[3][ctx], depth)
    }

    /// Code a whole-64 luma as one TX_64X64 in (`mode`, `delta`): a single
    /// 64x64 prediction off the block edges (dav1d predicts per transform
    /// block, and the transform IS the block), the forward 64 (coded 32x32),
    /// the class-4 context trellis and the exact inverse. Writes the
    /// reconstruction into `recon[0]`, the levels into `cf`, and returns
    /// (distortion, bits) in the units of the 4xTX_32X32 pass.
    #[allow(clippy::too_many_arguments)]
    fn tx64_luma_pass(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        ftype: bool,
        lam: f32,
        cf: &mut [i32; 1024],
    ) -> (f32, f32) {
        let maxv = (1i32 << self.bd) - 1;
        let mut pred = self.sbuf_i4096();
        let mut rr = self.sbuf_i4096();
        let plain = self.speed == Speed::Fast;
        let r = self.tx64_luma_eval(
            px, py, mode, delta, have_tr, have_bl, ftype, lam, plain, cf, &mut pred, &mut rr,
        );
        for row in 0..64 {
            let dst = &mut self.recon[0][(py + row) * self.w + px..][..64];
            recon_add_pred(dst, &pred[row * 64..], &rr[row * 64..], maxv);
        }
        r
    }

    /// The read-only core of [`Self::tx64_luma_pass`]: fills `pred`, the
    /// levels `cf` and the inverse residual `rr`; `plain` selects the
    /// context-free trellis (Fast, and the NONE-vs-SPLIT estimator).
    #[allow(clippy::too_many_arguments)]
    fn tx64_luma_eval(
        &self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        ftype: bool,
        lam: f32,
        plain: bool,
        cf: &mut [i32; 1024],
        pred: &mut [i32; 4096],
        rr: &mut [i32; 4096],
    ) -> (f32, f32) {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        if mode == DC_PRED {
            let dc = self.intrapred.dc_pred(&self.recon[0], self.w, px, py, 64, 64, self.bd as i32);
            pred.fill(dc);
        } else {
            self.intrapred.predict_nd_ad(
                mode,
                delta,
                &self.recon[0],
                self.w,
                px,
                py,
                64,
                64,
                have_tr,
                have_bl,
                self.w,
                self.h,
                ftype,
                &mut pred[..],
                self.bd,
            );
        }
        let mut resid = self.sbuf_i4096();
        self.rd
            .residual_pred(&mut resid[..], &pred[..], &self.src[0], self.w, px, py, 64, 64);
        let (mut q, tf) = self.dct.dct64x64_t(&resid, &self.quant);
        let dcs = self.dc_sign_ctx_span(0, px / 4, py / 4, 16, 16);
        if plain {
            trellis_optimize(&mut q, &tf, dcq, acq, &SCAN_32X32, lam);
        } else {
            trellis_optimize_ctx(
                &mut q,
                &tf,
                dcq,
                acq,
                &SCAN_32X32,
                lam,
                32,
                32,
                self.dcdf(),
                4,
                0,
                &self.dcdf().eob_bin_1024_l,
                dcs,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
        }
        crate::idct::idct_dequant_64x64(&q, &self.quant, rr);
        let dist = self.luma_partition_distortion(px, py, 64, 64, acq, &pred[..], 0, &rr[..]);
        let bits = self.luma_bits_tx64(&q, px, py);
        *cf = q;
        (dist, bits)
    }

    /// Code a whole-64 luma as sixteen raster-order TX_16X16 (`tx_depth` 2)
    /// in (`mode`, `delta`): each transform is predicted from the RUNNING
    /// reconstruction with dav1d's per-transform edge rule (the grid rule of
    /// `split32_depth2_try`), DCT_DCT, the class-2 context trellis with
    /// progressive coefficient contexts. Writes the reconstruction into
    /// `recon[0]` and the levels TX-raster-major (256 each) into `cf`;
    /// returns (distortion, bits) in the units of the 4xTX_32X32 pass.
    /// `commit` keeps the reconstruction (the emitter re-sets the coefficient
    /// contexts from the coded symbols); otherwise it is restored. The
    /// coefficient contexts are always restored.
    #[allow(clippy::too_many_arguments)]
    fn grid16_luma_pass(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        ftype: bool,
        lam: f32,
        cf: &mut [i32; 4096],
        commit: bool,
    ) -> (f32, f32) {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let maxv = (1i32 << self.bd) - 1;
        let (bx4_0, by4_0) = (px / 4, py / 4);
        let saved_a: [u8; 16] = *self.a_coef[0][bx4_0..].first_chunk().unwrap();
        let saved_l: [u8; 16] = *self.l_coef[0][by4_0..].first_chunk().unwrap();
        let mut saved = if commit {
            None
        } else {
            Some(self.sc().take_u4096())
        };
        if let Some(sv) = saved.as_mut() {
            for row in 0..64 {
                sv[row * 64..row * 64 + 64]
                    .copy_from_slice(&self.recon[0][(py + row) * self.w + px..][..64]);
            }
        }
        let mut dist = 0f32;
        let mut bits = 0f32;
        for j in 0..4usize {
            for i in 0..4usize {
                let ti = j * 4 + i;
                let (bx, by) = (px + i * 16, py + j * 16);
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
                        ftype,
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
                let (mut q, tf) = self.dct.dct16x16_t(&resid, &self.quant);
                trellis_optimize_ctx(
                    &mut q,
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
                let rr = self.idct.idct_dequant_16x16(&q, &self.quant);
                dist += self.luma_partition_distortion(bx, by, 16, 16, acq, &pred[..], 0, &rr[..]);
                bits += self.luma_bits_ctx_bounded(
                    &q,
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
                let res_ctx = Self::coef_res_ctx(&q, &SCAN_16X16);
                self.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
                self.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
                for ry in 0..16 {
                    let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                    recon_add_pred(&mut drow[..16], &pred[ry * 16..], &rr[ry * 16..], maxv);
                }
                cf[ti * 256..ti * 256 + 256].copy_from_slice(&q[..]);
            }
        }
        self.a_coef[0][bx4_0..bx4_0 + 16].copy_from_slice(&saved_a);
        self.l_coef[0][by4_0..by4_0 + 16].copy_from_slice(&saved_l);
        if let Some(sv) = saved {
            for row in 0..64 {
                self.recon[0][(py + row) * self.w + px..][..64]
                    .copy_from_slice(&sv[row * 64..row * 64 + 64]);
            }
            self.sc().put_u4096(sv);
        }
        (dist, bits)
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
    ) -> (usize, i32, f32, Luma64Tx, Option<LossyLumaPalette>) {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let tx64 = self.tx64_allowed();
        // tx_depth symbol difference (TX_64X64 = depth 0, 4xTX_32X32 = depth 1);
        // constant over modes, so only the TX64 leg carries it.
        let tx64_depth_rate = if tx64 {
            rate_cost(self.mlam() * prdo, self.tx_depth64_bits(px, py, 0) - self.tx_depth64_bits(px, py, 1))
        } else {
            0.0
        };
        let mlam = self.mlam() * prdo;
        let maxv = (1i32 << self.bd) - 1;
        let block_ftype = self.luma_filter_type(px, py);
        let mut saved = self.sc().take_u4096();
        for row in 0..64 {
            saved[row * 64..row * 64 + 64]
                .copy_from_slice(&self.recon[0][(py + row) * self.w + px..][..64]);
        }
        let stride = self.w;
        let restore = |recon: &mut [u16]| {
            for row in 0..64 {
                recon[(py + row) * stride + px..][..64]
                    .copy_from_slice(&saved[row * 64..row * 64 + 64]);
            }
        };
        let mut best = (DC_PRED, 0i32, f32::INFINITY, Luma64Tx::Quad32);
        // Stage 1: the ranked modes at angle delta 0. Stage 2: the six other
        // deltas of a directional winner (the quadrant edge flags are the
        // dav1d per-transform ones, so extension-reading angles stay exact).
        let mut cands: Vec<(usize, i32)> = self
            .rank_luma64_modes(px, py, have_tr, have_bl)
            .iter()
            .map(|&m| (m, 0))
            .collect();
        let mut stage = 0;
        while stage < 2 {
            for &(mode, delta) in &cands {
            restore(&mut self.recon[0]);
            let mut total = rate_cost(mlam, self.mode_bits(px, py, mode));
            if (V_PRED..=VERT_LEFT_PRED).contains(&mode) {
                total += rate_cost(
                    mlam,
                    cdf_cost(&self.dcdf().angle_delta[mode - V_PRED], (delta + 3) as usize),
                );
            }
            let mode_cost = total;
            if tx64 {
                let mut cf = self.sbuf_i1024();
                let (dist, bits) =
                    self.tx64_luma_pass(px, py, mode, delta, have_tr, have_bl, block_ftype, lam, &mut cf);
                let t64 = (mode_cost
                    + crate::partition_rd::rd_cost(dist, mlam, bits)
                    + tx64_depth_rate)
                    * crate::tuning::get().tx64_bias;
                if t64 < best.2 {
                    best = (mode, delta, t64, Luma64Tx::Tx64);
                }
                restore(&mut self.recon[0]);
            }
            for (sx, sy) in Self::Q64 {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                if mode == DC_PRED {
                    *pred = [self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    self.intrapred.predict_nd_ad(
                        mode,
                        delta,
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
            if total < best.2 {
                best = (mode, delta, total, Luma64Tx::Quad32);
            }
            }
            stage += 1;
            cands.clear();
            if stage == 1
                && (V_PRED..=VERT_LEFT_PRED).contains(&best.0)
                && self.speed.try_angle_deltas_av1(64, self.base_q_idx)
            {
                cands.extend([-3, -2, -1, 1, 2, 3].map(|d| (best.0, d)));
            }
        }
        // Palette leaf (`palette64`, screen frames): the block's own palette
        // header + map, residual through the four TX_32X32 (Quad32).
        let mut best_pal: Option<LossyLumaPalette> = None;
        for (p, pred64) in self.big_palette_finalists(px, py, 64, 64, mlam) {
            restore(&mut self.recon[0]);
            let total = rate_cost(
                mlam,
                self.mode_bits(px, py, DC_PRED) + self.palette_rate_bits(px, py, &p),
            ) + self.quad32_pass_pred(px, py, &pred64, lam, mlam);
            if total < best.2 {
                best = (DC_PRED, 0, total, Luma64Tx::Quad32);
                best_pal = Some(p);
            }
        }
        // tx_depth 2 (`tx64_depth2`): sixteen TX_16X16 on the winning mode,
        // against its four-TX_32X32 (or TX_64X64) total — a 64 whose detail
        // sits in a few corners codes it in a few 16x16s instead of ringing
        // four TX_32X32. Not at Fast: its candidate loop prices the quadrants
        // with the plain trellis, so the ctx-trellis leg would not compare.
        // A palette winner keeps Quad32 (its prediction is not an intra mode).
        if crate::tuning::get().tx64_depth2
            && self.speed != Speed::Fast
            && best.2.is_finite()
            && best_pal.is_none()
        {
            restore(&mut self.recon[0]);
            let (mode, delta) = (best.0, best.1);
            let mut mode_cost = rate_cost(mlam, self.mode_bits(px, py, mode));
            if (V_PRED..=VERT_LEFT_PRED).contains(&mode) {
                mode_cost += rate_cost(
                    mlam,
                    cdf_cost(&self.dcdf().angle_delta[mode - V_PRED], (delta + 3) as usize),
                );
            }
            let mut cf = self.sbuf_i4096();
            let (dist, bits) = self.grid16_luma_pass(
                px, py, mode, delta, have_tr, have_bl, block_ftype, lam, &mut cf, false,
            );
            let t2 = mode_cost
                + crate::partition_rd::rd_cost(dist, mlam, bits)
                + rate_cost(mlam, self.tx_depth64_bits(px, py, 2) - self.tx_depth64_bits(px, py, 1));
            if t2 < best.2 {
                best = (mode, delta, t2, Luma64Tx::Grid16);
            }
        }
        restore(&mut self.recon[0]);
        self.sc().put_u4096(saved);
        (best.0, best.1, best.2, best.3, best_pal)
    }

    /// Price the four TX_32X32 quadrants of the 64x64 at (px, py) under a
    /// given whole-block prediction (`pred64`, stride 64): running
    /// reconstruction, ctx trellis, the same distortion/rate terms as the
    /// candidate loop of [`Self::rd_pick_luma64`]. The prediction's own
    /// header bits are the caller's. Leaves the recon modified.
    fn quad32_pass_pred(&mut self, px: usize, py: usize, pred64: &[i32], lam: f32, mlam: f32) -> f32 {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let maxv = (1i32 << self.bd) - 1;
        let mut total = 0.0f32;
        for (sx, sy) in Self::Q64 {
            let (bx, by) = (px + sx, py + sy);
            let (qbx4, qby4) = (bx / 4, by / 4);
            let mut pred = self.sbuf_i1024();
            for ry in 0..32 {
                pred[ry * 32..ry * 32 + 32].copy_from_slice(&pred64[(sy + ry) * 64 + sx..][..32]);
            }
            let mut resid = self.sbuf_i1024();
            self.rd.residual_pred(&mut resid[..], &pred[..], &self.src[0], self.w, bx, by, 32, 32);
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
            let distortion =
                self.luma_partition_distortion(bx, by, 32, 32, acq, &pred[..], 0, &rr[..]);
            total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&cf, &SCAN_32X32, 32, bx, by, DC_PRED, 0),
            );
            for row in 0..32 {
                let dst = &mut self.recon[0][(by + row) * self.w + bx..][..32];
                recon_add_pred(dst, &pred[row * 32..], &rr[row * 32..], maxv);
            }
        }
        total
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
            if self.tx64_allowed() && crate::tuning::get().tx64_none {
                let (mut cf, mut pred, mut rr) =
                    (self.sbuf_i1024(), self.sbuf_i4096(), self.sbuf_i4096());
                let (dist, bits) = self.tx64_luma_eval(
                    px, py, m, 0, false, false, false, lam, true, &mut cf, &mut pred, &mut rr,
                );
                let t64 = (crate::partition_rd::rd_cost(dist, mlam, bits)
                    + rate_cost(
                        mlam,
                        self.mode_bits(px, py, m) + self.tx_depth64_bits(px, py, 0)
                            - self.tx_depth64_bits(px, py, 1),
                    ))
                    * crate::tuning::get().tx64_bias;
                best = best.min(t64);
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
                let sse = sse_recon::<1024, 32>(&self.rd, &[dc; 1024], &rr, self.src_blk(plane, tx0, ty0, 32, 32), self.bd);
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
        // Whole-64 palette leaf (`palette64`, screen frames): without its
        // bound the DC-only proxy sends every palette-exact superblock to
        // SPLIT, where the children pay four headers and four map borders.
        // A palette-driven NONE is never accepted on the forced-NONE child
        // bound alone (below): its bound is tiny against four child palettes,
        // yet the refined children (IntraBC copies, free DC flats) are often
        // cheaper still — measured +0.5..+1.2% on n_bbscreen without this.
        let none_luma_dc = self.rd_cost_none64_luma(px, py, prdo);
        let pal_bound = self.palette_region_bound(px, py, 64, 64, part_lam);
        let pal_leaf = pal_bound < none_luma_dc;
        let none_luma = none_luma_dc.min(pal_bound);
        let none_bias = if self.top_band() && self.ss420 && !self.palette_exact(px, py, 64) {
            top_none_bias_420(self.aq.base_q)
        } else {
            self.none64_split_bias_at()
        };
        let rd_none = (none_luma + self.rd_cost_chroma64(px, py, prdo)) * none_bias
            + rate_cost(part_lam, self.part_rate_bl(1, x8, y8, 0));
        // PARTITION_HORZ / PARTITION_VERT (BLOCK_64X32 / BLOCK_32X64). The three
        // leaf shapes are compared on ONE mode-aware estimator (the DC-only
        // whole-64 proxy cannot see a rect's per-half mode). A winning rect only
        // REPLACES NONE: the leaf-vs-SPLIT comparison below keeps the tuned
        // whole-64 cost. MEASURED: letting the rect compete with SPLIT at its own
        // (scaled) estimate stole superblocks the bottom-up 32 subtree codes far
        // better (+1.9% synth / +0.3% photos), while forced rect64 everywhere is
        // RD-equal-or-better than forced whole-64 (s_mascot 420 q60: 48.8KB/78.61
        // vs 50.3KB/78.54).
        let (rd_leaf, leaf) = if self.rect64_allowed(false) || self.rect64_allowed(true) {
            let (htr, hbl) = (
                thr && y8 > 0 && (px + 64) < self.w,
                lhb && x8 > 0 && (py + 64) < self.h,
            );
            let whole = self.rd_cost_whole64_modes(x8, y8, htr, hbl, prdo)
                + rate_cost(part_lam, self.part_rate_bl(1, x8, y8, 0));
            let bias = crate::tuning::get().rect64_bias;
            let mut best = (whole, Part16::None, 0usize);
            for (vert, sym, part) in [(false, 1usize, Part16::Horz), (true, 2, Part16::Vert)] {
                if self.rect64_allowed(vert) {
                    let c = self.rd_cost_rect64(x8, y8, vert, thr, lhb, prdo) * bias
                        + rate_cost(part_lam, self.part_rate_bl(1, x8, y8, sym));
                    if c < best.0 {
                        best = (c, part, sym);
                    }
                }
            }
            (rd_none, best.1)
        } else {
            (rd_none, Part16::None)
        };
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
            !self.mono && self.speed.at_least_slow() && joint_luma_uv_proxy_enabled();
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
        let best_whole = rd_leaf.min(rd_ibc.unwrap_or(f32::INFINITY));
        if rd_split_upper < best_whole {
            return Part16::Split;
        }

        if !pal_leaf && best_whole <= rd_split_upper * b64_refinement_window() {
            if let Some(rd_ibc) = rd_ibc
                && rd_ibc < rd_leaf.min(rd_split_upper)
            {
                return Part16::Intrabc;
            }
            let keep = rd_leaf <= rd_split_upper;
            return if keep { leaf } else { Part16::Split };
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
        // A palette-driven NONE competes against the children's refined
        // costs in full: the blend exists to temper the child estimator's
        // optimism about deeper intra partitions, not to let a zero-residual
        // palette bound outbid IntraBC copies and free DC flats.
        let refinement = if pal_leaf { 1.0 } else { b64_split_refinement() };
        rd_split = rd_split_upper + refinement * (rd_split.min(rd_split_upper) - rd_split_upper);
        if let Some(rd_ibc) = rd_ibc
            && rd_ibc < rd_leaf.min(rd_split)
        {
            return Part16::Intrabc;
        }
        let keep = rd_leaf <= rd_split;
        if keep { leaf } else { Part16::Split }
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
        let prdo = self.perceptual_rd_scale(px, py, 64);
        let lam = trellis_lambda() * prdo;

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
        let (y_mode, y_delta, ltx, palette);
        if let Some(r) = rl {
            y_mode = r.mode as usize;
            y_delta = r.delta as i32;
            ltx = match r.tx {
                TxSel::Tx64 => Luma64Tx::Tx64,
                TxSel::Split64D2 => Luma64Tx::Grid16,
                _ => Luma64Tx::Quad32,
            };
            palette = (r.palette > 0).then(|| {
                self.rederive_luma_palette(px, py, 64, 64, r.palette as usize)
                    .expect("64x64 palette replay: candidate no longer derivable")
            });
            if let Some(cf) = rl_cf {
                for qi in 0..4 {
                    lcf[qi].copy_from_slice(&cf[qi * 1024..qi * 1024 + 1024]);
                }
            }
        } else {
            let pick = self.rd_pick_luma64(px, py, have_tr, have_bl, prdo);
            (y_mode, y_delta, ltx, palette) = (pick.0, pick.1, pick.3, pick.4);
        }
        // Whole-block palette prediction (stride 64) for the quadrant pass.
        let pal_pred = palette.as_ref().map(|p| {
            let mut pred = vec![0i32; 4096];
            palette_pred(&mut pred, 64, &p.colors, &p.packed_map, 64, 64);
            pred
        });
        let tx64 = ltx == Luma64Tx::Tx64;
        let grid16 = ltx == Luma64Tx::Grid16;
        self.record_pred_blk(x8, y8, 16);
        // Deblock footprint: the luma transform grid — one TX_64X64, sixteen
        // TX_16X16, or four TX_32X32 so the filter sees the interior
        // transform edges (mirrors block16's tx-split re-record).
        if tx64 {
            self.record_tx_blk(x8, y8, 16);
        } else if grid16 {
            for j in 0..4 {
                for i in 0..4 {
                    self.record_tx_blk(x8 + i * 2, y8 + j * 2, 4);
                }
            }
        } else {
            for (sx, sy) in Self::Q64 {
                self.record_tx_blk((px + sx) / 8, (py + sy) / 8, 8);
            }
        }
        // Real coding of the winner: four TX_32X32, each predicted from the
        // running reconstruction, coefficients captured into `lcf`. Skipped in
        // Replay (recon preinstalled, coeffs loaded from the record above).
        if rl.is_none() && tx64 {
            let mut cf = self.sbuf_i1024();
            self.tx64_luma_pass(px, py, y_mode, y_delta, have_tr, have_bl, block_ftype, lam, &mut cf);
            *lcf[0] = *cf;
            for q in &mut lcf[1..] {
                q.fill(0);
            }
        } else if rl.is_none() && grid16 {
            // Sixteen TX_16X16 in raster order, packed 256 each across the
            // four 1024-entry records (TX `ti` = `lcf[ti / 4][(ti % 4) * 256..]`).
            let mut cf = self.sbuf_i4096();
            self.grid16_luma_pass(px, py, y_mode, y_delta, have_tr, have_bl, block_ftype, lam, &mut cf, true);
            for (qi, q) in lcf.iter_mut().enumerate() {
                q.copy_from_slice(&cf[qi * 1024..qi * 1024 + 1024]);
            }
        } else if rl.is_none() {
            for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                if let Some(region) = pal_pred.as_deref() {
                    for ry in 0..32 {
                        pred[ry * 32..ry * 32 + 32]
                            .copy_from_slice(&region[(sy + ry) * 64 + sx..][..32]);
                    }
                } else if y_mode == DC_PRED {
                    *pred = [self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    self.intrapred.predict_nd_ad(
                        y_mode,
                        y_delta,
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
        let mut uvf = [self.sbuf_i4096(), self.sbuf_i4096()];
        if let Some((cf, _)) = ru_cf.as_ref() {
            for (ci, dst) in uvf.iter_mut().enumerate() {
                dst[..ncg * 1024].copy_from_slice(&cf[ci][..ncg * 1024]);
            }
        } else if ru.is_some() {
            uvf[0][..ncg * 1024].fill(0);
            uvf[1][..ncg * 1024].fill(0);
        }
        // Chroma mode: searched over the non-CfL set (see `UV64_MODES`), each
        // transform predicted from the running recon.
        let (cbw, cbh) = if self.ss420 {
            (32, 32)
        } else if self.ss422 {
            (32, 64)
        } else {
            (64, 64)
        };
        let uv_mode = match ru {
            Some(r) => r.uv as usize,
            None => self.pick_uv64(px, py, cx, cy, cbw, cbh, cgrid, 32, 32, y_mode, prdo),
        };
        if ru.is_none() {
            let cftype = self.chroma_filter_type(px, py);
            let mlam_c = self.mlam_c() * prdo;
            self.uv64_pass(cx, cy, cgrid, 32, 32, uv_mode, cftype, lam, mlam_c, &mut uvf);
        }
        let [uflat, vflat] = uvf;
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
            delta: y_delta as i8,
            palette: palette.as_ref().map_or(0, |p| p.sel),
            filter: NO_FILTER,
            tx: if tx64 {
                TxSel::Tx64
            } else if grid16 {
                TxSel::Split64D2
            } else {
                TxSel::SplitDct([1; 4])
            },
        });
        let mut flat = self.sbuf_i4096();
        for qi in 0..4 {
            flat[qi * 1024..qi * 1024 + 1024].copy_from_slice(&lcf[qi][..]);
        }
        self.push_luma_cf(&flat[..]);
        self.push_uv_sel(UvSel {
            uv: uv_mode as u8,
            palette: 0,
            delta: 0,
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
        // true for BLOCK_8X8 and larger).
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (y_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        // CfL is not allowed at 64x64, so uv_mode uses the NOCFL CDF (index m,
        // not 13+m) — emit it directly rather than via `emit_uv_mode`.
        self.enc
            .encode_symbol(uv_mode, &mut self.cdfs.uv_mode[y_mode]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[uv_mode - V_PRED]);
        }
        self.commit_uv_mode(px, py, 64, 64, uv_mode);
        self.emit_palette_mode_info(px, py, 64, 64, y_mode, !self.mono, palette.as_ref(), None);
        // filter_intra is disallowed for max(w,h) > 32, so no symbol here.
        if let Some(p) = &palette {
            self.emit_palette_map(p);
        }
        self.code_tx_depth(px, py, 64, 64, if tx64 { 0 } else if grid16 { 2 } else { 1 });
        let sv = block_skip as u8;
        let mv = y_mode as u8;
        self.a_skip[bx4..bx4 + 16].fill(sv);
        self.l_skip[by4..by4 + 16].fill(sv);
        self.a_mode[bx4..bx4 + 16].fill(mv);
        self.l_mode[by4..by4 + 16].fill(mv);

        // --- Luma coefficients: one TX_64X64 (block == transform: txb_skip
        // ctx 0, 16-unit DC-sign span), or four TX_32X32 in raster order
        // (split contexts).
        if tx64 {
            let ds = self.dc_sign_ctx_span(0, bx4, by4, 16, 16);
            let res_ctx = crate::coeffs::encode_tx64_luma_coeffs_adapt(
                &mut self.enc,
                &mut self.cdfs,
                &lcf[0],
                0,
                ds,
            );
            self.a_coef[0][bx4..bx4 + 16].fill(res_ctx);
            self.l_coef[0][by4..by4 + 16].fill(res_ctx);
        }
        // Sixteen TX_16X16 in raster order (split contexts over the 4-unit
        // footprint, the intra TX_16X16 txtp symbol = DCT_DCT); the
        // reconstruction happened in `grid16_luma_pass` (or is preinstalled).
        if grid16 {
            for ti in 0..16usize {
                let (bx, by) = (px + (ti % 4) * 16, py + (ti / 4) * 16);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let cfq: &[i32; 256] = lcf[ti / 4][(ti % 4) * 256..].first_chunk::<256>().unwrap();
                let sk = self.skip_ctx_split(qbx4, qby4, 4, 4);
                let ds = self.dc_sign_ctx_16(0, qbx4, qby4);
                let res_ctx = encode_tx16_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    cfq,
                    false,
                    sk,
                    ds,
                    y_mode,
                    1, // DCT_DCT
                );
                self.a_coef[0][qbx4..qbx4 + 4].fill(res_ctx);
                self.l_coef[0][qby4..qby4 + 4].fill(res_ctx);
            }
        }
        for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
            if tx64 || grid16 {
                break;
            }
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

    use super::ibc_chroma_pred;

    /// The spec's two-pass 8-tap BILINEAR convolution (7.11.3.4) at the
    /// positions an odd luma DV produces (0 or 8 in 1/16 chroma samples).
    fn spec_bilinear(get: &dyn Fn(usize, usize) -> u32, kx: usize, ky: usize, fx: bool, fy: bool, bd: u32) -> u32 {
        let (r0, r1) = if bd == 12 { (5u32, 9u32) } else { (3, 11) };
        let taps = |p: u32| -> [i64; 8] { [0, 0, 0, 128 - 8 * p as i64, 8 * p as i64, 0, 0, 0] };
        let (px, py) = (if fx { 8 } else { 0 }, if fy { 8 } else { 0 });
        let round2 = |v: i64, n: u32| (v + (1i64 << (n - 1))) >> n;
        // Horizontal pass over the 8 rows the vertical taps can touch.
        let mut inter = [0i64; 8];
        for (r, slot) in inter.iter_mut().enumerate() {
            let yy = (ky + r).saturating_sub(3);
            let mut sum = 0i64;
            for (t, coef) in taps(px).iter().enumerate() {
                if *coef != 0 {
                    let xx = kx + t - 3;
                    sum += coef * i64::from(get(xx, yy));
                }
            }
            *slot = round2(sum, r0);
        }
        let mut sum = 0i64;
        for (t, coef) in taps(py).iter().enumerate() {
            if *coef != 0 {
                sum += coef * inter[t];
            }
        }
        round2(sum, r1) as u32
    }

    #[test]
    fn ibc_chroma_pred_matches_spec_bilinear_convolution() {
        let mut seed = 0x5eed_1234u32;
        for bd in [8u32, 10, 12] {
            let (w, h) = (48usize, 40usize);
            let plane: Vec<u32> = (0..w * h)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 17;
                    seed ^= seed << 5;
                    seed % (1 << bd)
                })
                .collect();
            let get = |x: usize, y: usize| plane[y * w + x];
            for fx in [false, true] {
                for fy in [false, true] {
                    let (kx, ky, bw, bh) = (5usize, 7usize, 16usize, 8usize);
                    let mut out = [0u16; 16 * 8];
                    ibc_chroma_pred(&get, kx, ky, fx, fy, bw, bh, &mut out);
                    for r in 0..bh {
                        for c in 0..bw {
                            assert_eq!(
                                u32::from(out[r * bw + c]),
                                spec_bilinear(&get, kx + c, ky + r, fx, fy, bd),
                                "bd {bd} fx {fx} fy {fy} at ({c},{r})"
                            );
                        }
                    }
                }
            }
        }
    }

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

    #[test]
    fn hash_bucket_preserves_raster_order_after_ring_wraps() {
        let (w, h) = (37usize, 29usize);
        let src = [vec![200u16; w * h], Vec::new(), Vec::new()];
        for sub in [(0, 0), (1, 0), (1, 1)] {
            let index = LossyIbcIndex::build(&src, w, h, sub);
            let expected: Vec<u32> = (0..h - 7)
                .step_by(1 << sub.1)
                .flat_map(|y| {
                    (0..w - 7)
                        .step_by(1 << sub.0)
                        .map(move |x| ((y as u32) << 16) | x as u32)
                })
                .collect();
            let hash = LossyIbcIndex::hash8(&src[0], w, 0, 0);
            assert_eq!(
                index.entries,
                expected
                    .into_iter()
                    .map(|origin| (hash, origin))
                    .collect::<Vec<_>>(),
            );
        }
    }
}
