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
// segments, its own map cells) plus a few scalars; `CodingCtx::snapshot_region` captures
// exactly that and debug builds assert the restore against a full clone.
// The epoch-stamped R-D memos are kept valid by rewinding the emit epoch to
// its pre-trial value after a restore and purging every entry stamped later.

/// Placement of the square node a trial snapshot covers.
#[derive(Clone, Copy)]
struct SnapRegion {
    x8: usize,
    y8: usize,
    sz8: usize,
    /// Luma plane width (stride of the luma-grid maps).
    w: usize,
    /// Chroma plane width.
    cw: usize,
    /// Chroma (horizontal, vertical) subsampling shifts.
    sub: (usize, usize),
}

impl SnapRegion {
    /// Plane `p`'s (x, y, width, height) of the node, in that plane's samples.
    #[inline]
    fn plane_rect(&self, p: usize) -> (usize, usize, usize, usize) {
        let (sx, sy) = if p == 0 { (0, 0) } else { self.sub };
        let (px, py, dim) = (self.x8 * 8, self.y8 * 8, self.sz8 * 8);
        (px >> sx, py >> sy, dim >> sx, dim >> sy)
    }
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

impl CodingCtx {
    /// Region snapshot of the state one square node's coding mutates: a
    /// `CodingCtx` whose arrays hold only the node's own segments. Built as a
    /// full struct literal, so a field added to `CodingCtx` does not compile
    /// until it is classified here and in [`CodingCtx::restore_region`].
    fn snapshot_region(&self, r: &SnapRegion) -> CodingCtx {
        let (x8, y8, sz8) = (r.x8, r.y8, r.sz8);
        let (x4, y4, n4) = (x8 * 2, y8 * 2, sz8 * 2);
        let mut recon: [Vec<u16>; 3] = Default::default();
        let mut a_coef: [Vec<u8>; 3] = Default::default();
        let mut l_coef: [Vec<u8>; 3] = Default::default();
        for p in 0..3 {
            let (bx, by, bw, bh) = r.plane_rect(p);
            if !self.recon[p].is_empty() {
                let pw = if p == 0 { r.w } else { r.cw };
                let ph = self.recon[p].len() / pw;
                recon[p] = region_2d(&self.recon[p], pw, bx, by, bw.min(pw - bx), bh.min(ph - by));
            }
            a_coef[p] = region_1d(&self.a_coef[p], bx / 4, bw / 4);
            l_coef[p] = region_1d(&self.l_coef[p], by / 4, bh / 4);
        }
        let s4 = r.w / 4;
        let s8 = r.w.div_ceil(8);
        let m = &self.maps;
        CodingCtx {
            // Scalars: copied whole.
            quant: self.quant,
            cquant: self.cquant,
            cdef_point_marked: self.cdef_point_marked,
            aq: self.aq.clone(),
            lr_ref_h: self.lr_ref_h,
            lr_ref_v: self.lr_ref_v,
            // Per-plane sample / coefficient-context segments.
            recon,
            a_coef,
            l_coef,
            // Above rows (per 4x4 column) and left columns (per 4x4 row).
            a_tx: region_1d(&self.a_tx, x4, n4),
            l_tx: region_1d(&self.l_tx, y4, n4),
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
            // Partition context (per 8x8).
            a_part: region_1d(&self.a_part, x8, sz8),
            l_part: region_1d(&self.l_part, y8, sz8),
            // 4x4-grid and 8x8-grid maps.
            ibc_mv: if self.ibc_mv.is_empty() {
                Vec::new()
            } else {
                region_2d(&self.ibc_mv, s4, x4, y4, n4, n4)
            },
            maps: BlockMaps {
                blk4: region_2d(&m.blk4, s4, x4, y4, n4, n4),
                blk4h: region_2d(&m.blk4h, s4, x4, y4, n4, n4),
                blk4v: region_2d(&m.blk4v, s4, x4, y4, n4, n4),
                blk4t: region_2d(&m.blk4t, s4, x4, y4, n4, n4),
                pblk4: region_2d(&m.pblk4, s4, x4, y4, n4, n4),
                pblk4h: region_2d(&m.pblk4h, s4, x4, y4, n4, n4),
                pblk4v: region_2d(&m.pblk4v, s4, x4, y4, n4, n4),
                pblk4t: region_2d(&m.pblk4t, s4, x4, y4, n4, n4),
                skip8: region_2d(&m.skip8, s8, x8, y8, sz8, sz8),
            },
        }
    }

    /// Write a [`CodingCtx::snapshot_region`] of the same region back.
    fn restore_region(&mut self, r: &SnapRegion, s: &CodingCtx) {
        // Exhaustive on purpose: see `snapshot_region`.
        let CodingCtx {
            quant,
            cquant,
            cdef_point_marked,
            aq,
            lr_ref_h,
            lr_ref_v,
            recon,
            a_coef,
            l_coef,
            a_tx,
            l_tx,
            a_skip,
            l_skip,
            a_mode,
            l_mode,
            a_uv_mode,
            l_uv_mode,
            a_palette,
            a_palette_uv,
            l_palette,
            l_palette_uv,
            a_part,
            l_part,
            ibc_mv,
            maps:
                BlockMaps {
                    blk4,
                    blk4h,
                    blk4v,
                    blk4t,
                    pblk4,
                    pblk4h,
                    pblk4v,
                    pblk4t,
                    skip8,
                },
        } = s;
        let (x8, y8, sz8) = (r.x8, r.y8, r.sz8);
        let (x4, y4, n4) = (x8 * 2, y8 * 2, sz8 * 2);
        self.quant = *quant;
        self.cquant = *cquant;
        self.cdef_point_marked = *cdef_point_marked;
        self.aq = aq.clone();
        self.lr_ref_h = *lr_ref_h;
        self.lr_ref_v = *lr_ref_v;
        for p in 0..3 {
            let (bx, by, bw, _) = r.plane_rect(p);
            if !self.recon[p].is_empty() {
                let pw = if p == 0 { r.w } else { r.cw };
                restore_2d(&mut self.recon[p], pw, bx, by, bw.min(pw - bx), &recon[p]);
            }
            restore_1d(&mut self.a_coef[p], bx / 4, &a_coef[p]);
            restore_1d(&mut self.l_coef[p], by / 4, &l_coef[p]);
        }
        restore_1d(&mut self.a_tx, x4, a_tx);
        restore_1d(&mut self.l_tx, y4, l_tx);
        restore_1d(&mut self.a_skip, x4, a_skip);
        restore_1d(&mut self.l_skip, y4, l_skip);
        restore_1d(&mut self.a_mode, x4, a_mode);
        restore_1d(&mut self.l_mode, y4, l_mode);
        restore_1d(&mut self.a_uv_mode, x4, a_uv_mode);
        restore_1d(&mut self.l_uv_mode, y4, l_uv_mode);
        restore_1d(&mut self.a_palette, x4, a_palette);
        restore_1d(&mut self.a_palette_uv, x4, a_palette_uv);
        restore_1d(&mut self.l_palette, y4, l_palette);
        restore_1d(&mut self.l_palette_uv, y4, l_palette_uv);
        restore_1d(&mut self.a_part, x8, a_part);
        restore_1d(&mut self.l_part, y8, l_part);
        let s4 = r.w / 4;
        let s8 = r.w.div_ceil(8);
        if !self.ibc_mv.is_empty() {
            restore_2d(&mut self.ibc_mv, s4, x4, y4, n4, ibc_mv);
        }
        let m = &mut self.maps;
        restore_2d(&mut m.blk4, s4, x4, y4, n4, blk4);
        restore_2d(&mut m.blk4h, s4, x4, y4, n4, blk4h);
        restore_2d(&mut m.blk4v, s4, x4, y4, n4, blk4v);
        restore_2d(&mut m.blk4t, s4, x4, y4, n4, blk4t);
        restore_2d(&mut m.pblk4, s4, x4, y4, n4, pblk4);
        restore_2d(&mut m.pblk4h, s4, x4, y4, n4, pblk4h);
        restore_2d(&mut m.pblk4v, s4, x4, y4, n4, pblk4v);
        restore_2d(&mut m.pblk4t, s4, x4, y4, n4, pblk4t);
        restore_2d(&mut m.skip8, s8, x8, y8, sz8, skip8);
    }
}

impl<'a> LossyTile<'a> {
    /// Bitmask of square-node sizes (in 8px units: 8 = 64x64, 4 = 32x32,
    /// 2 = 16x16) whose NONE-vs-SPLIT choice is made by trial encoding.
    #[inline]
    fn trial_levels(&self) -> u32 {
        let t = crate::tuning::get();
        let mut m = t.trial_part;
        if self.cfg.speed == Speed::ExtraSlow {
            m |= t.trial_part_xslow;
        }
        m
    }

    /// Rect legs in the 16x16 trial: the study knob, or the ExtraSlow package.
    #[inline]
    fn trial_rect_on(&self) -> bool {
        let t = crate::tuning::get();
        t.trial_rect || (self.cfg.speed == Speed::ExtraSlow && t.trial_rect_xslow)
    }

    /// Effective `trial_rect_sse_gate` (study knob first, else the ExtraSlow
    /// package value at that tier).
    #[inline]
    fn trial_sse_gate(&self) -> u32 {
        let t = crate::tuning::get();
        if t.trial_rect_sse_gate != 0 {
            t.trial_rect_sse_gate
        } else if self.cfg.speed == Speed::ExtraSlow && t.trial_rect_xslow {
            t.trial_rect_sse_gate_xslow
        } else {
            0
        }
    }

    /// Effective `trial_struct_k` (same precedence).
    #[inline]
    fn trial_struct_k(&self) -> f64 {
        let t = crate::tuning::get();
        if t.trial_struct_k > 0.0 {
            t.trial_struct_k as f64
        } else if self.cfg.speed == Speed::ExtraSlow && t.trial_rect_xslow {
            t.trial_struct_k_xslow as f64
        } else {
            0.0
        }
    }

    /// Whether the node of size `sz8` at this call takes the trial path:
    /// enabled for the level, chroma present (mono has no whole-32/16
    /// emitters), and no fixed-size short circuit. The caller additionally skips a node whose partition is
    /// already forced.
    #[inline]
    fn trial_level_on(&self, sz8: usize) -> bool {
        self.trial_levels() & sz8 as u32 != 0
            && !self.cfg.is_mono()
            && crate::tuning::fixed_size(self.cfg.speed) == 0
    }

    /// Whether a proxy verdict is one the trial re-decides under `trial_gate`
    /// (non-square and IntraBC verdicts always stand).
    #[inline]
    fn trial_may_overturn(proxy: Part16, gate: u32) -> bool {
        match proxy {
            Part16::None => gate != 2,
            Part16::Split => gate != 1 && gate != 3,
            _ => false,
        }
    }

    /// Gate 3 at the 16x16 node: a proxy SPLIT verdict is re-decided against
    /// the rect legs only (NONE was already rejected by the proxy).
    #[inline]
    fn trial_rect_vs_split(&self, proxy: Part16, gate: u32, sz8: usize) -> bool {
        gate == 3 && sz8 == 2 && proxy == Part16::Split && self.trial_rect_on()
    }

    /// Relative margin between two proxy leg costs (`trial_margin` axis);
    /// INFINITY when either leg is pruned, so the proxy verdict stands.
    #[inline]
    fn proxy_margin(a: f32, b: f32) -> f32 {
        if a.is_finite() && b.is_finite() {
            (a - b).abs() / a.min(b).max(f32::EPSILON)
        } else {
            f32::INFINITY
        }
    }

    /// Proxy-margin gate (`trial_margin`, 0 = off): trial only where the
    /// proxy's relative NONE-vs-SPLIT margin is below the knob.
    #[inline]
    fn trial_margin_ok(&self, margin: f32, px: usize, py: usize, dim: usize) -> bool {
        let t = crate::tuning::get();
        let m = t.trial_margin;
        if m <= 0.0 || margin < m {
            return true;
        }
        if margin.is_finite() {
            return false;
        }
        match t.trial_margin_pruned {
            0 => true,
            1 => false,
            _ => self.trial_src_gate_std(px, py, dim, t.trial_margin_edge, 0.0),
        }
    }

    /// Source gate (`trial_edge_hi` / `trial_edge_lo`): see the knobs.
    fn trial_src_gate(&self, px: usize, py: usize, dim: usize) -> bool {
        let t = crate::tuning::get();
        self.trial_src_gate_std(px, py, dim, t.trial_edge_hi, t.trial_edge_lo)
    }

    /// [`Self::trial_src_gate`] with explicit std thresholds.
    fn trial_src_gate_std(&self, px: usize, py: usize, dim: usize, edge_hi: f32, edge_lo: f32) -> bool {
        if edge_hi <= 0.0 {
            return true;
        }
        let (hi, lo) = (edge_hi * edge_hi, edge_lo * edge_lo);
        let (bw, bh) = (dim.min(self.cfg.w.saturating_sub(px)), dim.min(self.cfg.h.saturating_sub(py)));
        let (mut any_hi, mut any_lo) = (false, edge_lo <= 0.0);
        for y in (0..bh).step_by(8) {
            for x in (0..bw).step_by(8) {
                let (w, h) = ((bw - x).min(8), (bh - y).min(8));
                let (sum, sum_sq) = block_moments_i32(&self.cfg.src[0], self.cfg.w, px + x, py + y, w, h);
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
            (self.cfg.chroma.sub_x(), self.cfg.chroma.sub_y())
        }
    }

    #[inline]
    fn snap_region(&self, x8: usize, y8: usize, sz8: usize) -> SnapRegion {
        SnapRegion {
            x8,
            y8,
            sz8,
            w: self.cfg.w,
            cw: self.cfg.cw,
            sub: (self.cfg.chroma.sub_x(), self.cfg.chroma.sub_y()),
        }
    }

    fn trial_snapshot(&self, x8: usize, y8: usize, sz8: usize) -> CodingCtx {
        self.ctx.snapshot_region(&self.snap_region(x8, y8, sz8))
    }

    fn trial_restore(&mut self, x8: usize, y8: usize, sz8: usize, s: &CodingCtx) {
        let r = self.snap_region(x8, y8, sz8);
        self.ctx.restore_region(&r, s);
    }

    /// Drop every epoch-stamped R-D memo entry computed after `epoch`: after a
    /// restore to the state of `epoch`, those were computed against a
    /// reconstruction that no longer exists, while entries at or before it
    /// are exact for the restored state.
    fn trial_purge_caches(&self, epoch: u64) {
        self.memo.chroma_rd_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.memo.rd16_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.memo.rect_leaf_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
        self.memo.split4_rd_cache.borrow_mut().retain(|_, v| v.0 <= epoch);
    }

    /// Per-quadrant (luma, chroma) SSE of the current reconstruction against
    /// the source over the `dim`-square at (px, py), visible pixels only.
    fn trial_distortion(&self, px: usize, py: usize, dim: usize) -> [(i64, i64); 4] {
        let half = dim / 2;
        let mut out = [(0i64, 0i64); 4];
        for (qi, slot) in out.iter_mut().enumerate() {
            let (qx, qy) = (px + (qi & 1) * half, py + (qi >> 1) * half);
            let y = self.k.rd.sse_u16(
                self.src_blk(0, qx, qy, half, half),
                &self.ctx.recon[0],
                self.cfg.w,
                qx,
                qy,
            );
            let mut c = 0i64;
            if !self.cfg.is_mono() {
                for p in 1..3 {
                    let (sx, sy) = self.plane_shift(p);
                    let (cx, cy, cw, ch) = (qx >> sx, qy >> sy, half >> sx, half >> sy);
                    c += self.k.rd.sse_u16(
                        self.src_blk(p, cx, cy, cw, ch),
                        &self.ctx.recon[p],
                        self.cfg.cw,
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
            let mut slot = self.model.trial_cost.borrow_mut();
            if slot.is_none() {
                let mut c = (*self.model.dec_cdfs).clone();
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
        std::cell::Ref::map(self.model.trial_cost.borrow(), |o| {
            o.as_ref().expect("initialized above")
        })
    }

    /// Combined luma+chroma cost of one leg on the luma-SSE axis of this node:
    /// `sum_q lam * (Dy_q / lam_y(q) + Dc_q / lam_c(q)) + lam * bits`, where
    /// the per-quadrant lambdas are the node's own (default) or each 16x16
    /// quadrant's (`trial_child_prdo`, mirroring the bottom-up proxy sum).
    /// Structural penalty of the current reconstruction over the node
    /// (`trial_struct_k`): per visible 8x8 luma cell, the squared difference
    /// of source and reconstruction standard deviation times the cell area,
    /// i.e. lost (blurred) or invented (ringing) local contrast in SSE units.
    fn trial_struct_penalty(&self, px: usize, py: usize, dim: usize) -> f64 {
        let (bw, bh) = (dim.min(self.cfg.w.saturating_sub(px)), dim.min(self.cfg.h.saturating_sub(py)));
        let mut pen = 0f64;
        for y in (0..bh).step_by(8) {
            for x in (0..bw).step_by(8) {
                let (w, h) = ((bw - x).min(8), (bh - y).min(8));
                let n = (w * h) as f64;
                let sd = |plane: &[u16]| {
                    let (sum, sq) = block_moments_i32(plane, self.cfg.w, px + x, py + y, w, h);
                    let mean = sum as f64 / n;
                    (sq as f64 / n - mean * mean).max(0.0).sqrt()
                };
                let d = sd(&self.cfg.src[0]) - sd(&self.ctx.recon[0]);
                pen += d * d * n;
            }
        }
        pen
    }

    /// Luma SSE of a leg summed over the node's quadrants.
    fn trial_luma_sse(d: &[(i64, i64); 4]) -> i64 {
        d.iter().map(|&(dy, _)| dy).sum()
    }

    #[allow(clippy::too_many_arguments)]
    fn trial_cost(
        &self,
        d: &[(i64, i64); 4],
        bits: f32,
        px: usize,
        py: usize,
        dim: usize,
        dy_w: f64,
        extra_dy: f64,
    ) -> f64 {
        let child = crate::tuning::get().trial_child_prdo;
        // The axis scale `lam` is the node's `lambda_part`; the per-plane
        // divisors are the wSSE lambdas (`lambda_d`, `emit_mlam_c`) so the
        // distortion terms are in bits.
        let lam = (self.mlam() * self.perceptual_rd_scale(px, py, dim)) as f64;
        let ly_node = (self.lambda_d() * self.perceptual_rd_scale(px, py, dim)) as f64;
        let lam_c = self.emit_mlam_c(px, py, dim) as f64;
        let half = dim / 2;
        let mut j = 0f64;
        for (qi, &(dy, dc)) in d.iter().enumerate() {
            let (ly, lc) = if child {
                let (qx, qy) = (px + (qi & 1) * half, py + (qi >> 1) * half);
                (
                    (self.lambda_d() * self.perceptual_rd_scale(qx, qy, half)) as f64,
                    self.emit_mlam_c(qx, qy, half) as f64,
                )
            } else {
                (ly_node, lam_c)
            };
            j += lam * (dy_w * dy as f64 / ly + if self.cfg.is_mono() { 0.0 } else { dc as f64 / lc });
        }
        // Extra luma-SSE-unit distortion (structural penalty), on the node's
        // own luma axis.
        j += extra_dy;
        j + lam * bits as f64
    }

    /// Trial-encode NONE-vs-SPLIT decision for the square node of `sz8` 8px
    /// units at (x8, y8): codes both legs for real into scratch state, keeps
    /// the cheaper leg's reconstruction/contexts and commits its symbols to
    /// the live entropy coder. On return the node is fully emitted.
    #[allow(clippy::too_many_arguments)]
    fn trial_partition(
        &mut self,
        bl: usize,
        x8: usize,
        y8: usize,
        sz8: usize,
        thr: bool,
        lhb: bool,
        skip_none: bool,
    ) {
        let (px, py, dim) = (x8 * 8, y8 * 8, sz8 * 8);
        let e0 = self.memo.emit_epoch.get();
        let snap0 = self.trial_snapshot(x8, y8, sz8);
        #[cfg(debug_assertions)]
        let full0 = self.ctx.clone();

        // Swap in the scratch entropy state: a token sink plus the slot-tagged
        // CDF copy whose tags key the tokens (a nested trial finds these taken
        // and builds its own; the outer one wins the slot on the way back).
        let mut enc = self.run.trial_enc.take().unwrap_or_default();
        let mut cdfs = match self.run.trial_cdfs.take() {
            Some(c) => c,
            None => {
                let mut c = Box::new(self.ent.cdfs.clone());
                let slots = c.semantic_slots();
                enc.set_semantic_cdfs(&slots);
                c
            }
        };
        enc.begin_semantic_sink();
        std::mem::swap(&mut self.ent.enc, &mut enc);
        std::mem::swap(&mut self.ent.cdfs, &mut *cdfs);
        let stats0 = self.ent.cdfs.skip_stats;

        // Every leg codes the node for real from the same pre-trial state.
        let legs = self.trial_legs(sz8, skip_none);
        let mut tokens: Vec<crate::odec::EntropyTokens> = Vec::with_capacity(legs.len());
        let mut stats: Vec<SkipStats> = Vec::with_capacity(legs.len());
        let mut dists: Vec<[(i64, i64); 4]> = Vec::with_capacity(legs.len());
        let mut pens: Vec<f64> = Vec::with_capacity(legs.len());
        let struct_k = self.trial_struct_k();
        let mut snaps: Vec<Option<CodingCtx>> = Vec::with_capacity(legs.len());
        let mut e_max = e0;
        for (i, &leg) in legs.iter().enumerate() {
            if i > 0 {
                // Back to the pre-trial state, memos included.
                self.trial_restore(x8, y8, sz8, &snap0);
                #[cfg(debug_assertions)]
                debug_assert!(
                    full0 == self.ctx,
                    "trial snapshot does not cover everything the {:?} leg mutated",
                    legs[i - 1]
                );
                self.trial_purge_caches(e0);
                self.memo.emit_epoch.set(e0);
                self.ent.cdfs.skip_stats = stats0;
                self.ent.enc.begin_semantic_sink();
            }
            self.decode_sb(bl, x8, y8, sz8, thr, lhb, Some(leg));
            tokens.push(self.ent.enc.take_semantic());
            stats.push(skip_stats_sub(&self.ent.cdfs.skip_stats, &stats0));
            dists.push(self.trial_distortion(px, py, dim));
            // Unification V5: the penalty is a plain-SSE quantity added to the
            // trial axis, whose unit is `K_PART_LAMBDA` x wSSE
            // (`trial_struct_wsse` lifts it there).
            let unit = if crate::tuning::get().trial_struct_wsse {
                f64::from(K_PART_LAMBDA)
            } else {
                1.0
            };
            pens.push(if struct_k > 0.0 { struct_k * unit * self.trial_struct_penalty(px, py, dim) } else { 0.0 });
            // The last leg's state is live; earlier legs need a snapshot to
            // be restorable if they win.
            snaps.push((i + 1 < legs.len()).then(|| self.trial_snapshot(x8, y8, sz8)));
            e_max = e_max.max(self.memo.emit_epoch.get());
        }

        let bits: Vec<f32> = {
            let tables = self.trial_cost_tables();
            tokens.iter().map(|t| t.estimate_bits(&tables)).collect()
        };
        let tune = crate::tuning::get();
        let split_bias = tune.trial_split_bias as f64;
        let rect_bias = tune.trial_rect_bias as f64;
        let rect_dy_w = tune.trial_rect_dy_w as f64;
        let is_rect = |leg: Part16| matches!(leg, Part16::Horz | Part16::Vert);
        let mut costs: Vec<f64> = legs
            .iter()
            .enumerate()
            .map(|(i, &leg)| {
                let dy_w = if is_rect(leg) { rect_dy_w } else { 1.0 };
                let j = self.trial_cost(&dists[i], bits[i], px, py, dim, dy_w, pens[i]);
                match leg {
                    Part16::Split => j * split_bias,
                    Part16::Horz | Part16::Vert => j * rect_bias,
                    _ => j,
                }
            })
            .collect();
        let sse_gate = self.trial_sse_gate();
        if sse_gate != 0 && legs.iter().any(|&l| is_rect(l)) {
            // A square leg's luma SSE bounds what a rect may spend: the
            // best-cost square's (1) or the lower of the squares' (2).
            let squares = (0..legs.len()).filter(|&i| !is_rect(legs[i]));
            let sq_dy = if sse_gate == 1 {
                let best_sq = squares
                    .min_by(|&a, &b| costs[a].total_cmp(&costs[b]))
                    .expect("a square leg is always present");
                Self::trial_luma_sse(&dists[best_sq])
            } else {
                squares
                    .map(|i| Self::trial_luma_sse(&dists[i]))
                    .min()
                    .expect("a square leg is always present")
            };
            for i in 0..legs.len() {
                if is_rect(legs[i]) && Self::trial_luma_sse(&dists[i]) > sq_dy {
                    costs[i] = f64::INFINITY;
                }
            }
        }
        let mut win = 0usize;
        for i in 1..legs.len() {
            if costs[i] < costs[win] {
                win = i;
            }
        }
        if win + 1 < legs.len() {
            let snap = snaps[win].take().expect("non-final leg has a snapshot");
            self.trial_restore(x8, y8, sz8, &snap);
        }
        let tokens = tokens.swap_remove(win);
        let stats = stats[win];
        // Every memo entry now refers to a superseded reconstruction, exactly
        // as after any emitted block.
        self.memo.emit_epoch.set(e_max + 1);

        // Live entropy state back; commit the winner's symbols through it.
        std::mem::swap(&mut self.ent.enc, &mut enc);
        std::mem::swap(&mut self.ent.cdfs, &mut *cdfs);
        self.run.trial_enc = Some(enc);
        self.run.trial_cdfs = Some(cdfs);
        let mut slots = self.ent.cdfs.semantic_slots();
        self.ent.enc.replay_semantic(&tokens, &mut slots);
        skip_stats_add(&mut self.ent.cdfs.skip_stats, &stats);
    }

    /// The partitions the trial codes for real at a node of `sz8` 8px units:
    /// NONE and SPLIT always; with `trial_rect`, HORZ and VERT at the 16x16
    /// node where this chroma format has the emitter (VERT is spec-forbidden
    /// at 4:2:2).
    fn trial_legs(&self, sz8: usize, skip_none: bool) -> Vec<Part16> {
        let mut legs = if skip_none { vec![Part16::Split] } else { vec![Part16::None, Part16::Split] };
        if sz8 == 2 && self.trial_rect_on() {
            legs.push(Part16::Horz);
            if !self.cfg.is_422() {
                legs.push(Part16::Vert);
            }
        }
        legs
    }
}

