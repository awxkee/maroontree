/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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

// Real-R-D (trial-encode) partition decision.
//
// The proxy partition decisions (`choose_rect32`, `partition_choice_16`)
// price a SPLIT candidate's later children from reconstruction that is still
// zero-filled, so one big DC block swallows the hard edges of illustrations
// (see the 2026-10 partition-proxy audits). This module instead CODES both
// legs of a NONE-vs-SPLIT node for real — children included, each with its
// own lower-level decision — into scratch state, measures the real symbol
// stream and the real reconstruction error, keeps the cheaper leg's state and
// commits its symbols to the live entropy coder.
//
// Determinism contract (`-t1 == -tN`): the legs encode into a semantic-token
// sink keyed by a slot-tagged CDF copy (exactly what the wavefront workers
// do), rate is priced against the immutable decision snapshot `dec_cdfs`,
// and the winner's tokens are replayed into whatever live encoder this tile
// has (range coder in the serial path, the worker's token sink in the
// wavefront). Nothing here reads adaptive CDF probabilities or encoder state.
//
// State contract: everything `decode_sb` mutates for one square node is
// region-local (the block's own recon rectangle, its own above/left context
// segments, its own map cells) plus a few scalars; [`TrialSnap`] captures
// exactly that and debug builds assert the restore against a full clone.
// The epoch-stamped R-D memos are kept valid by rewinding the emit epoch to
// its pre-trial value after a restore and purging every entry stamped later.

/// Region snapshot of the state one square node's coding mutates.
struct TrialSnap {
    recon: [Vec<u16>; 3],
    a_coef: [Vec<u8>; 3],
    l_coef: [Vec<u8>; 3],
    a_tx: Vec<i8>,
    l_tx: Vec<i8>,
    a_part: Vec<u8>,
    l_part: Vec<u8>,
    a_skip: Vec<u8>,
    l_skip: Vec<u8>,
    a_mode: Vec<u8>,
    l_mode: Vec<u8>,
    a_uv_mode: Vec<u8>,
    l_uv_mode: Vec<u8>,
    a_palette: Vec<Vec<i32>>,
    a_palette_uv: Vec<Vec<i32>>,
    l_palette: Vec<Vec<i32>>,
    l_palette_uv: Vec<Vec<i32>>,
    blk4: Vec<u8>,
    blk4h: Vec<u8>,
    blk4v: Vec<bool>,
    blk4t: Vec<bool>,
    pblk4: Vec<u8>,
    pblk4h: Vec<u8>,
    pblk4v: Vec<bool>,
    pblk4t: Vec<bool>,
    skip8: Vec<bool>,
    ibc_mv: Vec<Option<(i16, i16)>>,
    aq: AqCtx,
    quant: Quant,
    cquant: Quant,
    cdef_point_marked: bool,
    uv_delta: i32,
}

#[inline]
fn region_1d<T: Clone>(v: &[T], a: usize, n: usize) -> Vec<T> {
    let e = (a + n).min(v.len());
    if a >= e {
        Vec::new()
    } else {
        v[a..e].to_vec()
    }
}

#[inline]
fn restore_1d<T: Clone>(v: &mut [T], a: usize, saved: &[T]) {
    if !saved.is_empty() {
        v[a..a + saved.len()].clone_from_slice(saved);
    }
}

#[inline]
fn region_2d<T: Clone>(v: &[T], stride: usize, x: usize, y: usize, w: usize, h: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w * h);
    for r in 0..h {
        out.extend_from_slice(&v[(y + r) * stride + x..][..w]);
    }
    out
}

#[inline]
fn restore_2d<T: Clone>(v: &mut [T], stride: usize, x: usize, y: usize, w: usize, saved: &[T]) {
    if w == 0 {
        return;
    }
    for (r, row) in saved.chunks_exact(w).enumerate() {
        v[(y + r) * stride + x..][..w].clone_from_slice(row);
    }
}

/// Everything the region snapshot claims to cover, cloned whole (debug-only
/// proof that a restore is complete).
#[cfg(debug_assertions)]
#[derive(PartialEq)]
struct TrialFullState {
    recon: [Vec<u16>; 3],
    a_coef: [Vec<u8>; 3],
    l_coef: [Vec<u8>; 3],
    a_tx: Vec<i8>,
    l_tx: Vec<i8>,
    a_part: Vec<u8>,
    l_part: Vec<u8>,
    a_skip: Vec<u8>,
    l_skip: Vec<u8>,
    a_mode: Vec<u8>,
    l_mode: Vec<u8>,
    a_uv_mode: Vec<u8>,
    l_uv_mode: Vec<u8>,
    a_palette: Vec<Vec<i32>>,
    a_palette_uv: Vec<Vec<i32>>,
    l_palette: Vec<Vec<i32>>,
    l_palette_uv: Vec<Vec<i32>>,
    blk4: Vec<u8>,
    blk4h: Vec<u8>,
    blk4v: Vec<bool>,
    blk4t: Vec<bool>,
    pblk4: Vec<u8>,
    pblk4h: Vec<u8>,
    pblk4v: Vec<bool>,
    pblk4t: Vec<bool>,
    skip8: Vec<bool>,
    ibc_mv: Vec<Option<(i16, i16)>>,
    aq: (i32, i32, bool, i32),
    q: (u8, u8),
    cdef_point_marked: bool,
    uv_delta: i32,
}

impl<'a> LossyTile<'a> {
    /// Bitmask of square-node sizes (in 8px units: 8 = 64x64, 4 = 32x32,
    /// 2 = 16x16) whose NONE-vs-SPLIT choice is made by trial encoding.
    #[inline]
    fn trial_levels(&self) -> u32 {
        let t = crate::tuning::get();
        let mut m = t.trial_part;
        if self.speed == Speed::ExtraSlow {
            m |= t.trial_part_xslow;
        }
        m
    }

    /// Whether the node of size `sz8` at this call takes the trial path:
    /// enabled for the level, a plain decide+emit tile (no capture/replay),
    /// not already inside a forced leg at this node, chroma present (mono has
    /// no whole-32/16 emitters), and no fixed-size short circuit.
    #[inline]
    fn trial_level_on(&self, sz8: usize) -> bool {
        self.trial_levels() & sz8 as u32 != 0
            && !self.mono
            && self.sb_mode == SbMode::Off
            && self.force_part.is_none()
            && crate::tuning::fixed_size(self.speed) == 0
    }

    /// Whether a proxy verdict is one the trial re-decides under `trial_gate`
    /// (non-square and IntraBC verdicts always stand).
    #[inline]
    fn trial_may_overturn(proxy: Part16, gate: u32) -> bool {
        match proxy {
            Part16::None => gate != 2,
            Part16::Split => gate != 1,
            _ => false,
        }
    }

    /// Source gate (`trial_edge_hi` / `trial_edge_lo`): see the knobs.
    fn trial_src_gate(&self, px: usize, py: usize, dim: usize) -> bool {
        let t = crate::tuning::get();
        if t.trial_edge_hi <= 0.0 {
            return true;
        }
        let (hi, lo) = (t.trial_edge_hi * t.trial_edge_hi, t.trial_edge_lo * t.trial_edge_lo);
        let (bw, bh) = (dim.min(self.w.saturating_sub(px)), dim.min(self.h.saturating_sub(py)));
        let (mut any_hi, mut any_lo) = (false, t.trial_edge_lo <= 0.0);
        for y in (0..bh).step_by(8) {
            for x in (0..bw).step_by(8) {
                let (w, h) = ((bw - x).min(8), (bh - y).min(8));
                let (sum, sum_sq) = block_moments_i32(&self.src[0], self.w, px + x, py + y, w, h);
                let cnt = (w * h) as f32;
                let mean = sum as f32 / cnt;
                let var = (sum_sq as f32 / cnt - mean * mean).max(0.0);
                any_hi |= var >= hi;
                any_lo |= var <= lo;
            }
        }
        any_hi && any_lo
    }

    #[inline]
    fn plane_shift(&self, p: usize) -> (usize, usize) {
        if p == 0 {
            (0, 0)
        } else {
            (usize::from(self.ss420 || self.ss422), usize::from(self.ss420))
        }
    }

    fn trial_snapshot(&self, x8: usize, y8: usize, sz8: usize) -> TrialSnap {
        let (px, py, dim) = (x8 * 8, y8 * 8, sz8 * 8);
        let (x4, y4, n4) = (px / 4, py / 4, dim / 4);
        let mut recon: [Vec<u16>; 3] = Default::default();
        let mut a_coef: [Vec<u8>; 3] = Default::default();
        let mut l_coef: [Vec<u8>; 3] = Default::default();
        for p in 0..3 {
            let (sx, sy) = self.plane_shift(p);
            let (bx, by, bw, bh) = (px >> sx, py >> sy, dim >> sx, dim >> sy);
            if !self.recon[p].is_empty() {
                let pw = if p == 0 { self.w } else { self.cw };
                let ph = self.recon[p].len() / pw;
                recon[p] = region_2d(&self.recon[p], pw, bx, by, bw.min(pw - bx), bh.min(ph - by));
            }
            a_coef[p] = region_1d(&self.a_coef[p], bx / 4, bw / 4);
            l_coef[p] = region_1d(&self.l_coef[p], by / 4, bh / 4);
        }
        let s4 = self.w / 4;
        let s8 = self.w.div_ceil(8);
        TrialSnap {
            recon,
            a_coef,
            l_coef,
            a_tx: region_1d(&self.a_tx, x4, n4),
            l_tx: region_1d(&self.l_tx, y4, n4),
            a_part: region_1d(&self.a_part, x8, sz8),
            l_part: region_1d(&self.l_part, y8, sz8),
            a_skip: region_1d(&self.a_skip, x4, n4),
            l_skip: region_1d(&self.l_skip, y4, n4),
            a_mode: region_1d(&self.a_mode, x4, n4),
            l_mode: region_1d(&self.l_mode, y4, n4),
            a_uv_mode: region_1d(&self.a_uv_mode, x4, n4),
            l_uv_mode: region_1d(&self.l_uv_mode, y4, n4),
            a_palette: region_1d(&self.a_palette, x4, n4),
            a_palette_uv: region_1d(&self.a_palette_uv, x4, n4),
            l_palette: region_1d(&self.l_palette, y4, n4),
            l_palette_uv: region_1d(&self.l_palette_uv, y4, n4),
            blk4: region_2d(&self.blk4, s4, x4, y4, n4, n4),
            blk4h: region_2d(&self.blk4h, s4, x4, y4, n4, n4),
            blk4v: region_2d(&self.blk4v, s4, x4, y4, n4, n4),
            blk4t: region_2d(&self.blk4t, s4, x4, y4, n4, n4),
            pblk4: region_2d(&self.pblk4, s4, x4, y4, n4, n4),
            pblk4h: region_2d(&self.pblk4h, s4, x4, y4, n4, n4),
            pblk4v: region_2d(&self.pblk4v, s4, x4, y4, n4, n4),
            pblk4t: region_2d(&self.pblk4t, s4, x4, y4, n4, n4),
            skip8: region_2d(&self.skip8, s8, x8, y8, sz8, sz8),
            ibc_mv: if self.ibc_mv.is_empty() {
                Vec::new()
            } else {
                region_2d(&self.ibc_mv, s4, x4, y4, n4, n4)
            },
            aq: self.aq.clone(),
            quant: self.quant,
            cquant: self.cquant,
            cdef_point_marked: self.cdef_point_marked,
            uv_delta: self.uv_delta,
        }
    }

    fn trial_restore(&mut self, x8: usize, y8: usize, sz8: usize, s: &TrialSnap) {
        let (px, py, dim) = (x8 * 8, y8 * 8, sz8 * 8);
        let (x4, y4, n4) = (px / 4, py / 4, dim / 4);
        for p in 0..3 {
            let (sx, sy) = self.plane_shift(p);
            let (bx, by, bw) = (px >> sx, py >> sy, dim >> sx);
            if !self.recon[p].is_empty() {
                let pw = if p == 0 { self.w } else { self.cw };
                restore_2d(&mut self.recon[p], pw, bx, by, bw.min(pw - bx), &s.recon[p]);
            }
            restore_1d(&mut self.a_coef[p], bx / 4, &s.a_coef[p]);
            restore_1d(&mut self.l_coef[p], by / 4, &s.l_coef[p]);
        }
        let s4 = self.w / 4;
        let s8 = self.w.div_ceil(8);
        restore_1d(&mut self.a_tx, x4, &s.a_tx);
        restore_1d(&mut self.l_tx, y4, &s.l_tx);
        restore_1d(&mut self.a_part, x8, &s.a_part);
        restore_1d(&mut self.l_part, y8, &s.l_part);
        restore_1d(&mut self.a_skip, x4, &s.a_skip);
        restore_1d(&mut self.l_skip, y4, &s.l_skip);
        restore_1d(&mut self.a_mode, x4, &s.a_mode);
        restore_1d(&mut self.l_mode, y4, &s.l_mode);
        restore_1d(&mut self.a_uv_mode, x4, &s.a_uv_mode);
        restore_1d(&mut self.l_uv_mode, y4, &s.l_uv_mode);
        restore_1d(&mut self.a_palette, x4, &s.a_palette);
        restore_1d(&mut self.a_palette_uv, x4, &s.a_palette_uv);
        restore_1d(&mut self.l_palette, y4, &s.l_palette);
        restore_1d(&mut self.l_palette_uv, y4, &s.l_palette_uv);
        restore_2d(&mut self.blk4, s4, x4, y4, n4, &s.blk4);
        restore_2d(&mut self.blk4h, s4, x4, y4, n4, &s.blk4h);
        restore_2d(&mut self.blk4v, s4, x4, y4, n4, &s.blk4v);
        restore_2d(&mut self.blk4t, s4, x4, y4, n4, &s.blk4t);
        restore_2d(&mut self.pblk4, s4, x4, y4, n4, &s.pblk4);
        restore_2d(&mut self.pblk4h, s4, x4, y4, n4, &s.pblk4h);
        restore_2d(&mut self.pblk4v, s4, x4, y4, n4, &s.pblk4v);
        restore_2d(&mut self.pblk4t, s4, x4, y4, n4, &s.pblk4t);
        restore_2d(&mut self.skip8, s8, x8, y8, sz8, &s.skip8);
        if !self.ibc_mv.is_empty() {
            restore_2d(&mut self.ibc_mv, s4, x4, y4, n4, &s.ibc_mv);
        }
        self.aq = s.aq.clone();
        self.quant = s.quant;
        self.cquant = s.cquant;
        self.cdef_point_marked = s.cdef_point_marked;
        self.uv_delta = s.uv_delta;
    }

    #[cfg(debug_assertions)]
    fn trial_full_state(&self) -> TrialFullState {
        TrialFullState {
            recon: self.recon.clone(),
            a_coef: self.a_coef.clone(),
            l_coef: self.l_coef.clone(),
            a_tx: self.a_tx.clone(),
            l_tx: self.l_tx.clone(),
            a_part: self.a_part.clone(),
            l_part: self.l_part.clone(),
            a_skip: self.a_skip.clone(),
            l_skip: self.l_skip.clone(),
            a_mode: self.a_mode.clone(),
            l_mode: self.l_mode.clone(),
            a_uv_mode: self.a_uv_mode.clone(),
            l_uv_mode: self.l_uv_mode.clone(),
            a_palette: self.a_palette.clone(),
            a_palette_uv: self.a_palette_uv.clone(),
            l_palette: self.l_palette.clone(),
            l_palette_uv: self.l_palette_uv.clone(),
            blk4: self.blk4.clone(),
            blk4h: self.blk4h.clone(),
            blk4v: self.blk4v.clone(),
            blk4t: self.blk4t.clone(),
            pblk4: self.pblk4.clone(),
            pblk4h: self.pblk4h.clone(),
            pblk4v: self.pblk4v.clone(),
            pblk4t: self.pblk4t.clone(),
            skip8: self.skip8.clone(),
            ibc_mv: self.ibc_mv.clone(),
            aq: (
                self.aq.cur_qidx,
                self.aq.prev_qidx,
                self.aq.read_deltas,
                self.aq.pending,
            ),
            q: (self.quant.qidx(), self.cquant.qidx()),
            cdef_point_marked: self.cdef_point_marked,
            uv_delta: self.uv_delta,
        }
    }

    /// Drop every epoch-stamped R-D memo entry computed after `epoch`: after a
    /// restore to the state of `epoch`, those were computed against a
    /// reconstruction that no longer exists, while entries at or before it
    /// are exact for the restored state.
    fn trial_purge_caches(&self, epoch: u64) {
        self.chroma_rd_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.rd16_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.rect_leaf_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.split4_rd_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
    }

    /// Per-quadrant (luma, chroma) SSE of the current reconstruction against
    /// the source over the `dim`-square at (px, py), visible pixels only.
    fn trial_distortion(&self, px: usize, py: usize, dim: usize) -> [(i64, i64); 4] {
        let half = dim / 2;
        let mut out = [(0i64, 0i64); 4];
        for (qi, slot) in out.iter_mut().enumerate() {
            let (qx, qy) = (px + (qi & 1) * half, py + (qi >> 1) * half);
            let y = self.rd.sse_u16(
                self.src_blk(0, qx, qy, half, half),
                &self.recon[0],
                self.w,
                qx,
                qy,
            );
            let mut c = 0i64;
            if !self.mono {
                for p in 1..3 {
                    let (sx, sy) = self.plane_shift(p);
                    let (cx, cy, cw, ch) = (qx >> sx, qy >> sy, half >> sx, half >> sy);
                    c += self.rd.sse_u16(
                        self.src_blk(p, cx, cy, cw, ch),
                        &self.recon[p],
                        self.cw,
                        cx,
                        cy,
                    );
                }
            }
            *slot = (y, c);
        }
        out
    }

    /// Per-slot copies of the decision CDF tables (same order as
    /// `Cdfs::semantic_slots`), built lazily per decision snapshot.
    fn trial_cost_tables(&self) -> std::cell::Ref<'_, Vec<Vec<u16>>> {
        {
            let mut slot = self.trial_cost.borrow_mut();
            if slot.is_none() {
                let mut c = (*self.dec_cdfs).clone();
                let tables = c
                    .semantic_slots()
                    .iter()
                    .map(|&(ptr, len)| {
                        // SAFETY: `c` is alive and unresized for the whole
                        // mapping; the slots point into its own tables.
                        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
                    })
                    .collect();
                *slot = Some(tables);
            }
        }
        std::cell::Ref::map(self.trial_cost.borrow(), |o| {
            o.as_ref().expect("initialized above")
        })
    }

    /// Combined luma+chroma cost of one leg on the luma-SSE axis of this node:
    /// `sum_q lam * (Dy_q / lam_y(q) + Dc_q / lam_c(q)) + lam * bits`, where
    /// the per-quadrant lambdas are the node's own (default) or each 16x16
    /// quadrant's (`trial_child_prdo`, mirroring the bottom-up proxy sum).
    fn trial_cost(
        &self,
        d: &[(i64, i64); 4],
        bits: f32,
        px: usize,
        py: usize,
        dim: usize,
    ) -> f64 {
        let child = crate::tuning::get().trial_child_prdo;
        let lam = (self.mlam() * self.perceptual_rd_scale(px, py, dim)) as f64;
        let lam_c = self.emit_mlam_c(px, py, dim) as f64;
        let half = dim / 2;
        let mut j = 0f64;
        for (qi, &(dy, dc)) in d.iter().enumerate() {
            let (ly, lc) = if child {
                let (qx, qy) = (px + (qi & 1) * half, py + (qi >> 1) * half);
                (
                    (self.mlam() * self.perceptual_rd_scale(qx, qy, half)) as f64,
                    self.emit_mlam_c(qx, qy, half) as f64,
                )
            } else {
                (lam, lam_c)
            };
            j += lam * (dy as f64 / ly + if self.mono { 0.0 } else { dc as f64 / lc });
        }
        j + lam * bits as f64
    }

    /// Trial-encode NONE-vs-SPLIT decision for the square node of `sz8` 8px
    /// units at (x8, y8): codes both legs for real into scratch state, keeps
    /// the cheaper leg's reconstruction/contexts and commits its symbols to
    /// the live entropy coder. On return the node is fully emitted.
    fn trial_partition(&mut self, bl: usize, x8: usize, y8: usize, sz8: usize, thr: bool, lhb: bool) {
        let (px, py, dim) = (x8 * 8, y8 * 8, sz8 * 8);
        let e0 = self.emit_epoch.get();
        let snap0 = self.trial_snapshot(x8, y8, sz8);
        #[cfg(debug_assertions)]
        let full0 = self.trial_full_state();

        // Swap in the scratch entropy state: a token sink plus the slot-tagged
        // CDF copy whose tags key the tokens (a nested trial finds these taken
        // and builds its own; the outer one wins the slot on the way back).
        let mut enc = self.trial_enc.take().unwrap_or_default();
        let mut cdfs = match self.trial_cdfs.take() {
            Some(c) => c,
            None => {
                let mut c = Box::new(self.cdfs.clone());
                let slots = c.semantic_slots();
                enc.set_semantic_cdfs(&slots);
                c
            }
        };
        enc.begin_semantic_sink();
        std::mem::swap(&mut self.enc, &mut enc);
        std::mem::swap(&mut self.cdfs, &mut *cdfs);
        let stats0 = self.cdfs.skip_stats;

        // Leg A: PARTITION_NONE.
        self.force_part = Some(Part16::None);
        self.decode_sb(bl, x8, y8, sz8, thr, lhb);
        debug_assert!(self.force_part.is_none(), "forced NONE leg was not consumed");
        let tok_a = self.enc.take_semantic();
        let stats_a = skip_stats_sub(&self.cdfs.skip_stats, &stats0);
        let d_a = self.trial_distortion(px, py, dim);
        let snap_a = self.trial_snapshot(x8, y8, sz8);
        let e_a = self.emit_epoch.get();

        // Back to the pre-trial state, memos included.
        self.trial_restore(x8, y8, sz8, &snap0);
        #[cfg(debug_assertions)]
        debug_assert!(
            full0 == self.trial_full_state(),
            "trial snapshot does not cover everything the NONE leg mutated"
        );
        self.trial_purge_caches(e0);
        self.emit_epoch.set(e0);
        self.cdfs.skip_stats = stats0;
        self.enc.begin_semantic_sink();

        // Leg B: PARTITION_SPLIT (children decide themselves).
        self.force_part = Some(Part16::Split);
        self.decode_sb(bl, x8, y8, sz8, thr, lhb);
        debug_assert!(self.force_part.is_none(), "forced SPLIT leg was not consumed");
        let tok_b = self.enc.take_semantic();
        let stats_b = skip_stats_sub(&self.cdfs.skip_stats, &stats0);
        let d_b = self.trial_distortion(px, py, dim);
        let e_b = self.emit_epoch.get();

        let (bits_a, bits_b) = {
            let tables = self.trial_cost_tables();
            (tok_a.estimate_bits(&tables), tok_b.estimate_bits(&tables))
        };
        let j_a = self.trial_cost(&d_a, bits_a, px, py, dim);
        let j_b = self.trial_cost(&d_b, bits_b, px, py, dim)
            * crate::tuning::get().trial_split_bias as f64;
        let (tokens, stats) = if j_a <= j_b {
            self.trial_restore(x8, y8, sz8, &snap_a);
            (tok_a, stats_a)
        } else {
            (tok_b, stats_b)
        };
        // Every memo entry now refers to a superseded reconstruction, exactly
        // as after any emitted block.
        self.emit_epoch.set(e_a.max(e_b) + 1);

        // Live entropy state back; commit the winner's symbols through it.
        std::mem::swap(&mut self.enc, &mut enc);
        std::mem::swap(&mut self.cdfs, &mut *cdfs);
        self.trial_enc = Some(enc);
        self.trial_cdfs = Some(cdfs);
        let mut slots = self.cdfs.semantic_slots();
        self.enc.replay_semantic(&tokens, &mut slots);
        skip_stats_add(&mut self.cdfs.skip_stats, &stats);
    }
}
