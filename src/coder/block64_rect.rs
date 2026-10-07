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

// BLOCK_64X32 / BLOCK_32X64: PARTITION_HORZ / PARTITION_VERT of a 64x64
// superblock. Each half is one prediction block coded like a half of the
// whole-64 leaf (`code_block64`): one shared luma mode reconstructed as two
// running TX_32X32 (`tx_depth = 1`, `t_dim.sub` of RTX_64X32 / RTX_32X64),
// DC chroma (CfL is illegal above 32x32), no filter intra. Chroma transforms
// follow `get_tx_size()`: 4:2:0 one RTX_32X16 / RTX_16X32, 4:2:2 (HORZ only —
// VERT has no legal 4:2:2 chroma transform) one TX_32X32, 4:4:4 two TX_32X32.

/// Chroma layout of a rect64 half: plane origin, the transform grid
/// (half-local origins in chroma samples), transform size and dav1d's
/// `not_one_blk` flag for the chroma `txb_skip` context.
struct Rect64Chroma {
    cx: usize,
    cy: usize,
    grid: &'static [(usize, usize)],
    tw: usize,
    th: usize,
    split: bool,
}

/// Chroma modes searched at 64-level leaves (CfL is illegal there). At
/// angle delta 0 every one reads only the above / left / above-left edge
/// (AV1 7.11.2.4: the top-right extension is used for pAngle < 90, the
/// bottom-left for pAngle > 180), so its prediction does not depend on the
/// edge-availability flags and is exact for any chroma transform tiling.
const UV64_MODES: [usize; 10] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    D113_PRED,
    D135_PRED,
    D157_PRED,
    SMOOTH_PRED,
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

impl<'a> LossyTile<'a> {
    #[inline]
    fn rect64_dims(vert: bool) -> (usize, usize) {
        if vert { (32, 64) } else { (64, 32) }
    }

    /// Half-local origins of the two TX_32X32 sub-transforms, coding order.
    #[inline]
    fn rect64_subs(vert: bool) -> [(usize, usize); 2] {
        if vert { [(0, 0), (0, 32)] } else { [(0, 0), (32, 0)] }
    }

    /// Whether PARTITION_HORZ (`vert = false`) / PARTITION_VERT is legal and
    /// enabled here. 4:2:2 VERT has no chroma transform (dav1d
    /// `max_txfm_size_for_bs[BS_32x64][I422] == 0`); monochrome has no whole-64
    /// path at all.
    #[inline]
    fn rect64_allowed(&self, vert: bool) -> bool {
        crate::tuning::get().rect64
            && self.speed.at_least_slow()
            && !self.mono
            && !(vert && self.ss422)
    }

    /// Origin and block-level `(have_tr, have_bl)` of half `half` (0/1) of
    /// the 64x64 at (x8, y8). Edge flags follow dav1d's intra-edge tree for a
    /// non-leaf node (`h[0] = f | LEFT_HAS_BOTTOM`, `h[1] = f & LEFT_HAS_BOTTOM`,
    /// `v[0] = f | TOP_HAS_RIGHT`, `v[1] = f & TOP_HAS_RIGHT`), combined with
    /// the frame bounds exactly like `decode_sb` does for square children.
    fn rect64_half(
        &self,
        x8: usize,
        y8: usize,
        vert: bool,
        half: usize,
        thr: bool,
        lhb: bool,
    ) -> (usize, usize, bool, bool) {
        let (lw, lh) = Self::rect64_dims(vert);
        let (px, py) = if vert {
            (x8 * 8 + half * 32, y8 * 8)
        } else {
            (x8 * 8, y8 * 8 + half * 32)
        };
        let (t, l) = match (vert, half) {
            (false, 0) => (thr, true),
            (false, _) => (false, lhb),
            (true, 0) => (true, lhb),
            (true, _) => (thr, false),
        };
        (
            px,
            py,
            t && py > 0 && px + lw < self.w,
            l && px > 0 && py + lh < self.h,
        )
    }

    /// Per-TX_32X32 intra-edge availability inside a half (dav1d
    /// `recon_b_intra`: an interior sub-TX sees the block's own running
    /// reconstruction, the outer one inherits the block's flag).
    #[inline]
    fn rect64_sub_edges(
        vert: bool,
        si: usize,
        px: usize,
        py: usize,
        have_tr: bool,
        have_bl: bool,
    ) -> (bool, bool) {
        match (vert, si) {
            (false, 0) => (py > 0, have_bl),
            (false, _) => (have_tr, false),
            (true, 0) => (have_tr, px > 0),
            (true, _) => (false, have_bl),
        }
    }

    fn rect64_chroma(&self, px: usize, py: usize, vert: bool) -> Rect64Chroma {
        static G1: [(usize, usize); 1] = [(0, 0)];
        static GH: [(usize, usize); 2] = [(0, 0), (32, 0)];
        static GV: [(usize, usize); 2] = [(0, 0), (0, 32)];
        if self.ss420 {
            let (tw, th) = if vert { (16, 32) } else { (32, 16) };
            Rect64Chroma { cx: px / 2, cy: py / 2, grid: &G1, tw, th, split: false }
        } else if self.ss422 {
            debug_assert!(!vert, "4:2:2 VERT at 64 is illegal");
            Rect64Chroma { cx: px / 2, cy: py, grid: &G1, tw: 32, th: 32, split: false }
        } else {
            Rect64Chroma {
                cx: px,
                cy: py,
                grid: if vert { &GV } else { &GH },
                tw: 32,
                th: 32,
                split: true,
            }
        }
    }

    #[inline]
    fn rect64_chroma_dc(&self, plane: usize, x: usize, y: usize, tw: usize, th: usize) -> i32 {
        let (r, s, b) = (&self.recon[plane], self.cw, self.bd as i32);
        match (tw, th) {
            (32, 32) => self.intrapred.dc_pred_32x32(r, s, x, y, b),
            (32, 16) => self.intrapred.dc_pred_32x16(r, s, x, y, b),
            _ => self.intrapred.dc_pred_16x32(r, s, x, y, b),
        }
    }

    #[inline]
    fn rect64_chroma_scan(tw: usize, th: usize) -> &'static [u32] {
        match (tw, th) {
            (32, 32) => &SCAN_32X32,
            (32, 16) => &SCAN_32X16,
            _ => &SCAN_16X32,
        }
    }

    /// Exact NOCFL `uv_mode` (+ zero `angle_delta_uv`) rate at a 64-level leaf.
    fn uv64_mode_bits(&self, y_mode: usize, uv: usize) -> f32 {
        let c = self.dcdf();
        let mut bits = cdf_cost(&c.uv_mode[y_mode], uv);
        if (V_PRED..=VERT_LEFT_PRED).contains(&uv) {
            bits += cdf_cost(&c.angle_delta[uv - V_PRED], 3);
        }
        bits
    }

    /// Chroma prediction of one 64-level-leaf transform in `uv` (one of
    /// [`UV64_MODES`]) from the running chroma reconstruction.
    #[allow(clippy::too_many_arguments)]
    fn uv64_pred(
        &self,
        plane: usize,
        uv: usize,
        x: usize,
        y: usize,
        tw: usize,
        th: usize,
        cftype: bool,
        pred: &mut [i32; 1024],
    ) {
        if uv == DC_PRED {
            pred[..tw * th].fill(self.rect64_chroma_dc(plane, x, y, tw, th));
        } else {
            let plane_h = if self.ss420 { self.h / 2 } else { self.h };
            self.intrapred.predict_nd(
                uv,
                &self.recon[plane],
                self.cw,
                x,
                y,
                tw,
                th,
                false,
                false,
                self.cw,
                plane_h,
                cftype,
                &mut pred[..tw * th],
                self.bd,
            );
        }
    }

    /// Transform, trellis and reconstruct (in place) one chroma transform
    /// against `pred`; returns its SSE and leaves the levels in `cf`.
    #[allow(clippy::too_many_arguments)]
    fn uv64_tx(
        &mut self,
        plane: usize,
        x: usize,
        y: usize,
        tw: usize,
        th: usize,
        pred: &[i32],
        lam: f32,
        cf: &mut [i32; 1024],
    ) -> i64 {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let maxv = (1i32 << self.bd) - 1;
        let n = tw * th;
        let scan = Self::rect64_chroma_scan(tw, th);
        let mut resid = self.sbuf_i1024();
        self.rd
            .residual_pred(&mut resid[..n], &pred[..n], &self.src[plane], self.cw, x, y, tw, th);
        let mut rr = self.sbuf_i1024();
        if n == 1024 {
            let (mut q, tf) = self.dct.dct32x32_t(&resid, &self.cquant);
            self.chroma_rect_trellis(&mut q, &tf, dcq, acq, scan, lam, tw, th, plane, x, y);
            self.rd.preserve_dc(&mut q[0], &resid[..]);
            *rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
            *cf = q;
        } else {
            let r512: &[i32; 512] = resid[..512].try_into().unwrap();
            let (mut q, tf) = if tw == 32 {
                self.dct.dct32x16_t(r512, &self.cquant)
            } else {
                self.dct.dct16x32_t(r512, &self.cquant)
            };
            self.chroma_rect_trellis(&mut q, &tf, dcq, acq, scan, lam, tw, th, plane, x, y);
            self.rd.preserve_dc(&mut q[0], &resid[..512]);
            let r = if tw == 32 {
                self.idct.idct_dequant_32x16(&q, &self.cquant)
            } else {
                self.idct.idct_dequant_16x32(&q, &self.cquant)
            };
            rr[..512].copy_from_slice(&r);
            cf[..512].copy_from_slice(&q);
        }
        let sse = self
            .rd
            .sse_recon(&pred[..n], &rr[..n], self.src_blk(plane, x, y, tw, th), self.bd);
        for ry in 0..th {
            let drow = &mut self.recon[plane][(y + ry) * self.cw + x..];
            recon_add_pred(&mut drow[..tw], &pred[ry * tw..], &rr[ry * tw..], maxv);
        }
        sse
    }

    /// Code both chroma planes of a 64-level leaf in `uv`: each transform of
    /// the grid predicted from the RUNNING chroma recon (written in place).
    /// Returns the R-D cost (SSE + exact coefficient rate under `mlam_c`);
    /// levels land in `out[plane - 1][gi * n..]`.
    #[allow(clippy::too_many_arguments)]
    fn uv64_pass(
        &mut self,
        cx: usize,
        cy: usize,
        grid: &[(usize, usize)],
        tw: usize,
        th: usize,
        uv: usize,
        cftype: bool,
        lam: f32,
        mlam_c: f32,
        out: &mut [SBuf<[i32; 4096]>; 2],
    ) -> f32 {
        let n = tw * th;
        let scan = Self::rect64_chroma_scan(tw, th);
        let mut total = 0.0f32;
        for (ci, dst) in out.iter_mut().enumerate() {
            let plane = ci + 1;
            for (gi, &(gx, gy)) in grid.iter().enumerate() {
                let (x, y) = (cx + gx, cy + gy);
                let mut pred = self.sbuf_i1024();
                self.uv64_pred(plane, uv, x, y, tw, th, cftype, &mut pred);
                let mut q = self.sbuf_i1024();
                let sse = self.uv64_tx(plane, x, y, tw, th, &pred[..], lam, &mut q);
                total += rd_cost_i64(
                    sse,
                    mlam_c,
                    self.chroma_rect_bits(&q[..n], scan, tw, th, plane, x, y),
                );
                dst[gi * n..gi * n + n].copy_from_slice(&q[..n]);
            }
        }
        total
    }

    /// Chroma mode of a 64-level leaf (block at luma (px, py), chroma block
    /// `cbw x cbh` at (cx, cy) tiled by `grid` of `tw x th` transforms):
    /// SATD-rank [`UV64_MODES`] over the grid, fully code DC plus the three
    /// best others, return the cheapest. Restores the chroma recon.
    #[allow(clippy::too_many_arguments)]
    fn pick_uv64(
        &mut self,
        px: usize,
        py: usize,
        cx: usize,
        cy: usize,
        cbw: usize,
        cbh: usize,
        grid: &[(usize, usize)],
        tw: usize,
        th: usize,
        y_mode: usize,
        prdo: f32,
    ) -> usize {
        if !self.speed.full_chroma_rdo() {
            return DC_PRED;
        }
        let cftype = self.chroma_filter_type(px, py);
        let lam = trellis_lambda() * prdo;
        let mlam_c = self.mlam_c() * prdo;
        let mut ranked = FixedList::<(u64, usize), 10>::new((0, DC_PRED));
        for &uv in UV64_MODES.iter().skip(1) {
            let mut score = 0u64;
            for plane in 1..=2 {
                for &(gx, gy) in grid {
                    let (x, y) = (cx + gx, cy + gy);
                    let mut pred = self.sbuf_i1024();
                    self.uv64_pred(plane, uv, x, y, tw, th, cftype, &mut pred);
                    score += self
                        .rd
                        .satd_sad_proxy(self.src_blk(plane, x, y, tw, th), &pred[..tw * th], tw);
                }
            }
            ranked.push((score, uv));
        }
        ranked
            .as_mut_slice()
            .sort_unstable_by_key(|&(score, uv)| (score, uv));
        // Two statements: a `RefMut` from `self.sc()` lives to the end of its
        // full expression, so taking both buffers in one array literal panics
        // with "RefCell already borrowed".
        let saved_u = self.sc().take_u4096();
        let saved_v = self.sc().take_u4096();
        let mut saved = [saved_u, saved_v];
        for (ci, sv) in saved.iter_mut().enumerate() {
            for ry in 0..cbh {
                sv[ry * cbw..ry * cbw + cbw]
                    .copy_from_slice(&self.recon[ci + 1][(cy + ry) * self.cw + cx..][..cbw]);
            }
        }
        let restore = |t: &mut Self, saved: &[Box<[u16; 4096]>; 2]| {
            for (ci, sv) in saved.iter().enumerate() {
                for ry in 0..cbh {
                    t.recon[ci + 1][(cy + ry) * t.cw + cx..][..cbw]
                        .copy_from_slice(&sv[ry * cbw..ry * cbw + cbw]);
                }
            }
        };
        let mut scratch = [self.sbuf_i4096(), self.sbuf_i4096()];
        let mut best = (DC_PRED, f32::INFINITY);
        let cands = std::iter::once(DC_PRED).chain(ranked.iter().take(3).map(|&(_, uv)| uv));
        for uv in cands {
            restore(self, &saved);
            let cost = rate_cost(mlam_c, self.uv64_mode_bits(y_mode, uv))
                + self.uv64_pass(cx, cy, grid, tw, th, uv, cftype, lam, mlam_c, &mut scratch);
            if cost < best.1 {
                best = (uv, cost);
            }
        }
        restore(self, &saved);
        let [s0, s1] = saved;
        self.sc().put_u4096(s0);
        self.sc().put_u4096(s1);
        best.0
    }

    /// DC-predicted, trellis'd chroma transform at (x, y): returns the
    /// quantized levels (first `tw * th` live) and the reconstruction residual.
    #[allow(clippy::too_many_arguments)]
    fn rect64_chroma_tx(
        &self,
        plane: usize,
        x: usize,
        y: usize,
        tw: usize,
        th: usize,
        dc: i32,
        lam: f32,
        cf: &mut [i32; 1024],
        rr: &mut [i32; 1024],
    ) {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let n = tw * th;
        let scan = Self::rect64_chroma_scan(tw, th);
        let mut resid = self.sbuf_i1024();
        self.rd
            .residual_dc(&mut resid[..n], &self.src[plane], self.cw, x, y, tw, th, dc);
        if n == 1024 {
            let (mut q, tf) = self.dct.dct32x32_t(&resid, &self.cquant);
            self.chroma_rect_trellis(&mut q, &tf, dcq, acq, scan, lam, tw, th, plane, x, y);
            self.rd.preserve_dc(&mut q[0], &resid[..]);
            *rr = self.idct.idct_dequant_32x32(&q, &self.cquant);
            *cf = q;
        } else {
            let r512: &[i32; 512] = resid[..512].try_into().unwrap();
            let (mut q, tf) = if tw == 32 {
                self.dct.dct32x16_t(r512, &self.cquant)
            } else {
                self.dct.dct16x32_t(r512, &self.cquant)
            };
            self.chroma_rect_trellis(&mut q, &tf, dcq, acq, scan, lam, tw, th, plane, x, y);
            self.rd.preserve_dc(&mut q[0], &resid[..512]);
            let r = if tw == 32 {
                self.idct.idct_dequant_32x16(&q, &self.cquant)
            } else {
                self.idct.idct_dequant_16x32(&q, &self.cquant)
            };
            rr[..512].copy_from_slice(&r);
            cf[..512].copy_from_slice(&q);
        }
    }

    /// Rank shared luma modes for a rect64 half by two-sub SATD+SAD (the
    /// rect analogue of `rank_luma64_modes`): Slow keeps DC/SMOOTH/PAETH plus
    /// the two best others; Medium/Fast keep the fast set.
    fn rank_rect64_modes(
        &self,
        px: usize,
        py: usize,
        vert: bool,
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
        let ftype = self.luma_filter_type(px, py);
        let mut ranked = FixedList::<(u64, usize), 13>::new((0, DC_PRED));
        for &mode in nd_modes() {
            let mut score = 0u64;
            for (si, (sx, sy)) in Self::rect64_subs(vert).into_iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (tr, bl) = Self::rect64_sub_edges(vert, si, px, py, have_tr, have_bl);
                let mut pred = self.sbuf_i1024();
                self.rect64_luma_pred(mode, 0, bx, by, tr, bl, ftype, &mut pred);
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

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn rect64_luma_pred(
        &self,
        mode: usize,
        delta: i32,
        bx: usize,
        by: usize,
        tr: bool,
        bl: bool,
        ftype: bool,
        pred: &mut [i32; 1024],
    ) {
        if mode == DC_PRED {
            pred.fill(self.intrapred.dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32));
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
                ftype,
                &mut pred[..],
                self.bd,
            );
        }
    }

    /// Code the luma of a rect64 half in `mode`: two TX_32X32 predicted from
    /// the RUNNING reconstruction (written into `self.recon[0]`), each
    /// trellis'd with exact contexts. Returns the R-D cost under `mlam` and
    /// leaves the levels in `cf`.
    #[allow(clippy::too_many_arguments)]
    fn rect64_luma_pass(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        ftype: bool,
        lam: f32,
        mlam: f32,
        cf: &mut [[i32; 1024]; 2],
    ) -> f32 {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let maxv = (1i32 << self.bd) - 1;
        let mut total = rate_cost(mlam, self.mode_bits(px, py, mode));
        if (V_PRED..=VERT_LEFT_PRED).contains(&mode) {
            total += rate_cost(
                mlam,
                cdf_cost(&self.dcdf().angle_delta[mode - V_PRED], (delta + 3) as usize),
            );
        }
        for (si, (sx, sy)) in Self::rect64_subs(vert).into_iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = Self::rect64_sub_edges(vert, si, px, py, have_tr, have_bl);
            let mut pred = self.sbuf_i1024();
            self.rect64_luma_pred(mode, delta, bx, by, tr, bl, ftype, &mut pred);
            let mut resid = self.sbuf_i1024();
            self.rd
                .residual_pred(&mut resid[..], &pred[..], &self.src[0], self.w, bx, by, 32, 32);
            let (mut q, tf) = self.dct.dct32x32_t(&resid, &self.quant);
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
                3,
                0,
                &self.dcdf().eob_bin_1024_l,
                self.dc_sign_ctx_32(0, bx / 4, by / 4),
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.idct_dequant_32x32(&q, &self.quant);
            let distortion =
                self.luma_partition_distortion(bx, by, 32, 32, acq, &pred[..], 0, &rr[..]);
            total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&q, &SCAN_32X32, 32, bx, by, mode, 0),
            );
            for ry in 0..32 {
                let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                recon_add_pred(&mut drow[..32], &pred[ry * 32..], &rr[ry * 32..], maxv);
            }
            cf[si] = q;
        }
        total
    }

    /// Pick the shared luma mode of a rect64 half by real two-sub coding
    /// (the rect analogue of `rd_pick_luma64`); restores the half's recon.
    fn rd_pick_rect64(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) -> (usize, i32) {
        let (lw, lh) = Self::rect64_dims(vert);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let ftype = self.luma_filter_type(px, py);
        let mut saved = self.sc().take_u4096();
        for ry in 0..lh {
            saved[ry * lw..ry * lw + lw]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..lw]);
        }
        let mut cf = [[0i32; 1024]; 2];
        let mut best = (DC_PRED, 0i32, f32::INFINITY);
        // Stage 1: ranked modes at delta 0; stage 2: the directional winner's
        // other six deltas (per-sub edge flags are dav1d-exact).
        let mut cands: Vec<(usize, i32)> = self
            .rank_rect64_modes(px, py, vert, have_tr, have_bl)
            .iter()
            .map(|&m| (m, 0))
            .collect();
        for stage in 0..2 {
            for &(mode, delta) in &cands {
                for ry in 0..lh {
                    self.recon[0][(py + ry) * self.w + px..][..lw]
                        .copy_from_slice(&saved[ry * lw..ry * lw + lw]);
                }
                let cost = self.rect64_luma_pass(
                    px, py, vert, mode, delta, have_tr, have_bl, ftype, lam, mlam, &mut cf,
                );
                if cost < best.2 {
                    best = (mode, delta, cost);
                }
            }
            cands.clear();
            if stage == 0
                && (V_PRED..=VERT_LEFT_PRED).contains(&best.0)
                && self.speed.try_angle_deltas_av1(64, self.base_q_idx)
            {
                cands.extend([-3, -2, -1, 1, 2, 3].map(|d| (best.0, d)));
            }
        }
        for ry in 0..lh {
            self.recon[0][(py + ry) * self.w + px..][..lw]
                .copy_from_slice(&saved[ry * lw..ry * lw + lw]);
        }
        self.sc().put_u4096(saved);
        (best.0, best.1)
    }

    /// Mode-aware luma estimate of one prediction region made of TX_32X32
    /// sub-transforms (the whole 64x64, or a rect64 half): SATD-rank the full
    /// mode set over the subs, then price the winner with the real transform,
    /// plain trellis and entropy rate. Unlike the DC-only whole-64 proxy this
    /// sees what a rect partition buys — a different mode per half — so the
    /// three leaf shapes are compared on one footing.
    #[allow(clippy::too_many_arguments)]
    fn rd_cost_luma_region64(
        &self,
        px: usize,
        py: usize,
        subs: &[(usize, usize)],
        edges: impl Fn(usize) -> (bool, bool),
        prdo: f32,
    ) -> f32 {
        let (acq, dcq) = (self.quant.ac_q() as f32, self.quant.dc_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let ftype = self.luma_filter_type(px, py);
        let mut best = (u64::MAX, DC_PRED);
        for &mode in nd_modes() {
            let mut score = 0u64;
            for (si, &(sx, sy)) in subs.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (tr, bl) = edges(si);
                let mut pred = self.sbuf_i1024();
                self.rect64_luma_pred(mode, 0, bx, by, tr, bl, ftype, &mut pred);
                score += self.rd.satd_sad_proxy(self.src_blk(0, bx, by, 32, 32), &pred[..], 32);
            }
            if (score, mode) < best {
                best = (score, mode);
            }
        }
        let mode = best.1;
        let mut total = rate_cost(mlam, self.mode_bits(px, py, mode));
        if (V_PRED..=VERT_LEFT_PRED).contains(&mode) {
            total += rate_cost(mlam, cdf_cost(&self.dcdf().angle_delta[mode - V_PRED], 3));
        }
        for (si, &(sx, sy)) in subs.iter().enumerate() {
            let (bx, by) = (px + sx, py + sy);
            let (tr, bl) = edges(si);
            let mut pred = self.sbuf_i1024();
            self.rect64_luma_pred(mode, 0, bx, by, tr, bl, ftype, &mut pred);
            let mut resid = self.sbuf_i1024();
            self.rd
                .residual_pred(&mut resid[..], &pred[..], &self.src[0], self.w, bx, by, 32, 32);
            let (mut cf, tf) = self.dct.dct32x32_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
            let rr = self.idct.idct_dequant_32x32(&cf, &self.quant);
            let distortion =
                self.luma_partition_distortion(bx, by, 32, 32, acq, &pred[..], 0, &rr[..]);
            total += crate::partition_rd::rd_cost(
                distortion,
                mlam,
                self.luma_bits(&cf, &SCAN_32X32, 32, bx, by, mode, 0),
            );
        }
        total
    }

    /// Mode-aware estimate of the whole-64 leaf (luma via
    /// [`Self::rd_cost_luma_region64`] + the DC chroma proxy), the reference the
    /// rect64 legs are measured against.
    fn rd_cost_whole64_modes(&self, x8: usize, y8: usize, have_tr: bool, have_bl: bool, prdo: f32) -> f32 {
        let (px, py) = (x8 * 8, y8 * 8);
        self.rd_cost_luma_region64(
            px,
            py,
            &Self::Q64,
            |si| {
                let (sx, sy) = Self::Q64[si];
                Self::quad_edges(sx, sy, px, py, have_tr, have_bl)
            },
            prdo,
        ) + self.rd_cost_chroma64(px, py, prdo)
    }

    /// Mode-aware R-D estimate of PARTITION_HORZ / PARTITION_VERT at the
    /// 64x64 at (x8, y8) (each half's luma via
    /// [`Self::rd_cost_luma_region64`], DC chroma), excluding the partition
    /// symbol. Compared against [`Self::rd_cost_whole64_modes`].
    fn rd_cost_rect64(&self, x8: usize, y8: usize, vert: bool, thr: bool, lhb: bool, prdo: f32) -> f32 {
        let lam = trellis_lambda() * prdo;
        let mlam_c = self.mlam_c() * prdo;
        let (lw, lh) = Self::rect64_dims(vert);
        let subs = Self::rect64_subs(vert);
        let mut total = 0.0f32;
        for half in 0..2 {
            let (px, py, have_tr, have_bl) = self.rect64_half(x8, y8, vert, half, thr, lhb);
            total += self.rd_cost_luma_region64(
                px,
                py,
                &subs,
                |si| Self::rect64_sub_edges(vert, si, px, py, have_tr, have_bl),
                prdo,
            );
            let c = self.rect64_chroma(px, py, vert);
            let n = c.tw * c.th;
            let scan = Self::rect64_chroma_scan(c.tw, c.th);
            let mut chroma = 0.0f32;
            for plane in 1..=2 {
                for &(gx, gy) in c.grid {
                    let (x, y) = (c.cx + gx, c.cy + gy);
                    let dc = self.rect64_chroma_dc(plane, x, y, c.tw, c.th);
                    let mut q = self.sbuf_i1024();
                    let mut rr = self.sbuf_i1024();
                    self.rect64_chroma_tx(plane, x, y, c.tw, c.th, dc, lam, &mut q, &mut rr);
                    let sse = self.rd.chroma_sse(
                        self.src_blk(plane, x, y, c.tw, c.th),
                        self.bd,
                        &[],
                        dc,
                        &rr[..n],
                    );
                    chroma += crate::partition_rd::rd_cost(
                        sse,
                        mlam_c,
                        self.chroma_rect_bits(&q[..n], scan, c.tw, c.th, plane, x, y),
                    );
                }
            }
            total += self.chroma_partition_weight_at(px, py, lw, lh) * chroma;
        }
        total
    }

    /// Code the two halves of PARTITION_HORZ (`vert = false`) / PARTITION_VERT
    /// of the in-frame 64x64 at (x8, y8). The partition symbol and the
    /// 64-level partition context are written by the caller.
    fn code_block64_rect(&mut self, x8: usize, y8: usize, vert: bool, thr: bool, lhb: bool) {
        // Lambdas anchored at the 64x64 parent, like the decision that chose
        // this partition.
        let prdo = self.perceptual_rd_scale(x8 * 8, y8 * 8, 64);
        for half in 0..2 {
            let (px, py, have_tr, have_bl) = self.rect64_half(x8, y8, vert, half, thr, lhb);
            self.code_rect64_half(px, py, vert, have_tr, have_bl, prdo);
        }
    }

    fn code_rect64_half(
        &mut self,
        px: usize,
        py: usize,
        vert: bool,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) {
        let (lw, lh) = Self::rect64_dims(vert);
        let (x8, y8) = (px / 8, py / 8);
        let (bx4, by4) = (px / 4, py / 4);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let subs = Self::rect64_subs(vert);

        // Deblock footprint: one prediction block, two TX_32X32 transforms.
        self.emit_epoch.set(self.emit_epoch.get() + 1);
        self.record_blk_rect(x8, y8, (lw / 4) as u8, (lh / 4) as u8);
        for (sx, sy) in subs {
            self.record_tx_blk((px + sx) / 8, (py + sy) / 8, 8);
        }
        // dav1d derives the intra-edge filter type ONCE at the block origin.
        let ftype = self.luma_filter_type(px, py);

        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();

        // --- Luma.
        let mut lcf = [[0i32; 1024]; 2];
        let (y_mode, y_delta) = if let Some(r) = rl {
            if let Some(cf) = rl_cf {
                for (si, dst) in lcf.iter_mut().enumerate() {
                    dst.copy_from_slice(&cf[si * 1024..si * 1024 + 1024]);
                }
            }
            (r.mode as usize, r.delta as i32)
        } else {
            let (mode, delta) = self.rd_pick_rect64(px, py, vert, have_tr, have_bl, prdo);
            self.rect64_luma_pass(
                px, py, vert, mode, delta, have_tr, have_bl, ftype, lam, mlam, &mut lcf,
            );
            (mode, delta)
        };
        let luma_zero = lcf.iter().all(|q| self.rd.all_zero_i32(&q[..]));

        // --- Chroma: searched non-CfL mode, each transform predicted from the
        // running recon.
        let c = self.rect64_chroma(px, py, vert);
        let n = c.tw * c.th;
        let ng = c.grid.len();
        let mut uv = [self.sbuf_i4096(), self.sbuf_i4096()];
        if let Some((cf, _)) = ru_cf.as_ref() {
            for (ci, dst) in uv.iter_mut().enumerate() {
                dst[..ng * n].copy_from_slice(&cf[ci][..ng * n]);
            }
        } else if ru.is_some() {
            uv[0][..ng * n].fill(0);
            uv[1][..ng * n].fill(0);
        }
        let (cbw, cbh) = if self.ss420 {
            (lw / 2, lh / 2)
        } else if self.ss422 {
            (lw / 2, lh)
        } else {
            (lw, lh)
        };
        let uv_mode = match ru {
            Some(r) => r.uv as usize,
            None => self.pick_uv64(px, py, c.cx, c.cy, cbw, cbh, c.grid, c.tw, c.th, y_mode, prdo),
        };
        if ru.is_none() {
            let cftype = self.chroma_filter_type(px, py);
            let mlam_c = self.mlam_c() * prdo;
            self.uv64_pass(c.cx, c.cy, c.grid, c.tw, c.th, uv_mode, cftype, lam, mlam_c, &mut uv);
        }
        let chroma_zero = uv.iter().all(|q| self.rd.all_zero_i32(&q[..ng * n]));
        // Not the superblock size, so a skipped half still carries delta_q
        // (5.11.6 `read_delta_qindex` only returns early for MiSize == sbSize).
        let block_skip = luma_zero && chroma_zero;

        // Record the winner for the wavefront (Capture only; no-ops otherwise).
        self.push_luma_sel(LumaSel {
            mode: y_mode as u8,
            delta: y_delta as i8,
            palette: 0,
            filter: NO_FILTER,
            tx: TxSel::SplitDct([1; 4]),
        });
        let mut flat = self.sbuf_i4096();
        flat[..1024].copy_from_slice(&lcf[0]);
        flat[1024..2048].copy_from_slice(&lcf[1]);
        self.push_luma_cf(&flat[..2048]);
        self.push_uv_sel(UvSel {
            uv: uv_mode as u8,
            palette: 0,
        });
        self.push_uv_cf(&uv[0][..ng * n], &uv[1][..ng * n], [0, 0]);

        // --- Header (decoder order): skip, y_mode, angle, uv_mode, palette,
        // tx_depth. No filter intra above 32x32.
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8_rect(x8, y8, lw / 8, lh / 8, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&y_mode) {
            self.enc.encode_symbol(
                (y_delta + 3) as usize,
                &mut self.cdfs.angle_delta[y_mode - V_PRED],
            );
        }
        // CfL is disallowed above 32x32: NOCFL uv_mode CDF (index y_mode).
        self.enc
            .encode_symbol(uv_mode, &mut self.cdfs.uv_mode[y_mode]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&uv_mode) {
            self.enc
                .encode_symbol(3, &mut self.cdfs.angle_delta[uv_mode - V_PRED]);
        }
        self.commit_uv_mode(px, py, lw, lh, uv_mode);
        self.emit_palette_mode_info(px, py, lw, lh, y_mode, !self.mono, None, None);
        self.code_tx_depth(px, py, lw, lh, 1);
        let (aw, ah) = (lw / 4, lh / 4);
        let sv = block_skip as u8;
        self.a_skip[bx4..bx4 + aw].fill(sv);
        self.l_skip[by4..by4 + ah].fill(sv);
        self.a_mode[bx4..bx4 + aw].fill(y_mode as u8);
        self.l_mode[by4..by4 + ah].fill(y_mode as u8);

        // --- Luma coefficients: the two TX_32X32 in coding order.
        if block_skip {
            self.a_coef[0][bx4..bx4 + aw].fill(0x40);
            self.l_coef[0][by4..by4 + ah].fill(0x40);
        } else {
            for (si, (sx, sy)) in subs.into_iter().enumerate() {
                let (qbx4, qby4) = ((px + sx) / 4, (py + sy) / 4);
                let sk = self.skip_ctx_split(qbx4, qby4, 8, 8);
                let ds = self.dc_sign_ctx_32(0, qbx4, qby4);
                let res_ctx =
                    encode_tx32_coeffs_adapt(&mut self.enc, &mut self.cdfs, &lcf[si], false, sk, ds);
                self.a_coef[0][qbx4..qbx4 + 8].fill(res_ctx);
                self.l_coef[0][qby4..qby4 + 8].fill(res_ctx);
            }
        }
        // --- Chroma coefficients, raster order per plane.
        let (tw4, th4) = (c.tw / 4, c.th / 4);
        for (ci, coefs) in uv.iter().enumerate() {
            let plane = ci + 1;
            for (gi, &(gx, gy)) in c.grid.iter().enumerate() {
                let (gbx4, gby4) = ((c.cx + gx) / 4, (c.cy + gy) / 4);
                let cres = if block_skip {
                    0x40
                } else {
                    let a = &self.a_coef[plane];
                    let l = &self.l_coef[plane];
                    let ca = a[gbx4..gbx4 + tw4].iter().any(|&x| x != 0x40) as usize;
                    let cl = l[gby4..gby4 + th4].iter().any(|&x| x != 0x40) as usize;
                    let sk = 7 + if c.split { 3 } else { 0 } + ca + cl;
                    let ds = self.dc_sign_ctx_span(plane, gbx4, gby4, tw4, th4);
                    let q = &coefs[gi * n..gi * n + n];
                    match (c.tw, c.th) {
                        (32, 32) => encode_tx32_coeffs_adapt(
                            &mut self.enc,
                            &mut self.cdfs,
                            q.first_chunk::<1024>().unwrap(),
                            true,
                            sk,
                            ds,
                        ),
                        (32, 16) => encode_32x16_chroma_coeffs(
                            &mut self.enc,
                            &mut self.cdfs,
                            q.first_chunk::<512>().unwrap(),
                            sk,
                            ds,
                        ),
                        _ => encode_16x32_chroma_coeffs(
                            &mut self.enc,
                            &mut self.cdfs,
                            q.first_chunk::<512>().unwrap(),
                            sk,
                            ds,
                        ),
                    }
                };
                self.a_coef[plane][gbx4..gbx4 + tw4].fill(cres);
                self.l_coef[plane][gby4..gby4 + th4].fill(cres);
            }
        }
    }
}
