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

impl<'a> LossyTile<'a> {
    /// Luma raster quadrants of a 64x64 block, as (dx, dy) pixel offsets.
    const Q64: [(usize, usize); 4] = [(0, 0), (32, 0), (0, 32), (32, 32)];

    /// Luma-only R-D proxy for coding a 64x64 as one BLOCK_64X64 (four DC-pred
    /// TX_32X32 quadrants). Chroma is added by the caller. Mirrors
    /// `rd_cost_split32`'s style: SATD distortion + `block_rate_bits` rate.
    fn rd_cost_none64_luma(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;
        let mut total = 0.0f32;
        for (sx, sy) in Self::Q64 {
            let (bx, by) = (px + sx, py + sy);
            let dc = dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32);
            let mut resid = [0i32; 1024];
            crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, bx, by, 32, 32, dc);
            let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
            trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
            let rr = idct_dequant_32x32(&cf, &self.quant);
            let distortion = self.luma_partition_distortion(bx, by, 32, 32, acq, |i| dc + rr[i]);
            total +=
                crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, &SCAN_32X32));
        }
        total
    }

    /// DC-pred chroma R-D proxy for a 64x64 luma block's chroma, over whatever
    /// TX_32X32 grid the subsampling implies (see `chroma64_geom`). Both planes,
    /// SSE distortion, scaled by `CHROMA_PART_RD_WEIGHT` so it sits on the SAME
    /// axis as the SPLIT leg's `rd_cost_chroma_partition` — without this the
    /// NONE leg carried 8x the chroma weight and 4:4:4 / 4:2:2 (whose chroma is
    /// 4x / 2x the 4:2:0 area) could never win.
    fn rd_cost_chroma64(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let (cx, cy, cgrid, _) = self.chroma64_geom(px, py);
        let mut total = 0.0f32;
        for plane in 1..=2 {
            for &(gx, gy) in cgrid {
                let (tx0, ty0) = (cx + gx, cy + gy);
                let dc = dc_pred_32x32(&self.recon[plane], self.cw, tx0, ty0, self.bd as i32);
                let mut resid = [0i32; 1024];
                crate::rd_sse::residual_dc(
                    &mut resid,
                    &self.src[plane],
                    self.cw,
                    tx0,
                    ty0,
                    32,
                    32,
                    dc,
                );
                let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
                let rr = idct_dequant_32x32(&cf, &self.cquant);
                let sse = sse_recon::<1024, 32>(
                    &[dc; 1024],
                    &rr,
                    &self.src[plane],
                    self.cw,
                    tx0,
                    ty0,
                    self.bd,
                );
                total += rd_cost_i64(sse, mlam, block_rate_bits(&cf, &SCAN_32X32));
            }
        }
        CHROMA_PART_RD_WEIGHT * total
    }

    /// SB-level NONE-vs-SPLIT decision for a fully-in-frame 64x64.
    /// Returns `Part16::None` to code one BLOCK_64X64, else `Part16::Split`.
    fn choose_64(&self, x8: usize, y8: usize) -> Part16 {
        let (px, py) = (x8 * 8, y8 * 8);
        let prdo = self.perceptual_rd_scale(px, py, 64);
        let part_lam = self.mlam() * prdo;
        // Whole-64: four TX_32X32 luma + one 32x32 chroma (per plane).
        // The SPLIT leg's 32x32 children each run a full chroma mode search
        // INCLUDING CfL. BLOCK_64X64 can never use CfL (disallowed above 32x32)
        // and codes DC-only chroma, but the DC proxy on both legs does not model
        // that gap — so NONE looks better than it codes, most where chroma
        // carries the most information (4:4:4 full-res, 4:2:2 half). Charge the
        // NONE chroma for the headroom it gives up.
        let chroma_handicap = if self.ss420 {
            1.0
        } else if self.ss422 {
            CHROMA64_HANDICAP_422
        } else {
            CHROMA64_HANDICAP_444
        };
        let rd_none = (self.rd_cost_none64_luma(px, py, prdo)
            + chroma_handicap * self.rd_cost_chroma64(px, py, prdo))
            * NONE64_SPLIT_BIAS;
        // Split into four BLOCK_32X32 (each priced as a whole-32 NONE + its 16x16
        // chroma) plus the split-symbol overhead. An upper bound on the real
        // split cost (children may split further), so it is conservative toward
        // keeping the split — it never over-merges detail.
        let mut rd_split = rate_cost(part_lam, SPLIT_SIGNAL_BITS);
        for (sx, sy) in [(0usize, 0usize), (32, 0), (0, 32), (32, 32)] {
            let (qx, qy) = (px + sx, py + sy);
            rd_split += self.rd_cost_none32(qx, qy, prdo)
                + self.rd_cost_chroma_partition(qx, qy, 32, Part16::None, prdo);
        }
        if rd_none <= rd_split {
            Part16::None
        } else {
            Part16::Split
        }
    }

    /// Code a fully-in-frame 64x64 region as one BLOCK_64X64. Luma is a
    /// non-directional intra mode (DC / SMOOTH / PAETH) reconstructed as four
    /// TX_32X32 quadrants; chroma is a DC-predicted TX_32X32 grid whose shape
    /// follows the subsampling (4:4:4 2x2, 4:2:2 1x2, 4:2:0 single).
    fn code_block64(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (cx, cy, cgrid, csplit) = self.chroma64_geom(px, py);
        let maxv = (1i32 << self.bd) - 1;
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let prdo = self.perceptual_rd_scale(px, py, 64);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam() * prdo;

        // Deblock footprint: four TX_32X32 tiles so the filter sees the interior
        // 32-sample transform edges (mirrors block16's tx-split re-record).
        for (sx, sy) in Self::Q64 {
            self.record_blk((px + sx) / 8, (py + sy) / 8, 8);
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
        let mut lcf = [[0i32; 1024]; 4];
        let mut y_mode = DC_PRED;
        if let Some(r) = rl {
            y_mode = r.mode as usize;
            if let Some(cf) = rl_cf {
                for qi in 0..4 {
                    lcf[qi].copy_from_slice(&cf[qi * 1024..qi * 1024 + 1024]);
                }
            }
        } else {
            // Cheap mode pick: score DC/SMOOTH/PAETH by summed four-quadrant SSE
            // + rate, each quadrant predicted from the current (pre-block) recon.
            let mut best_eff = f32::INFINITY;
            for &m in fast_nd_modes() {
                let mut eff = 0.0f32;
                for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
                    let (bx, by) = (px + sx, py + sy);
                    let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                    let mut pred = [0i32; 1024];
                    if m == DC_PRED {
                        pred =
                            [dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                    } else {
                        intra_predict_nd(
                            m,
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
                            &mut pred,
                            self.bd,
                        );
                    }
                    let mut resid = [0i32; 1024];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &pred,
                        &self.src[0],
                        self.w,
                        bx,
                        by,
                        32,
                        32,
                    );
                    let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, &SCAN_32X32, lam);
                    let rr = idct_dequant_32x32(&cf, &self.quant);
                    let sse =
                        sse_recon::<1024, 32>(&pred, &rr, &self.src[0], self.w, bx, by, self.bd);
                    eff += rd_cost_i64(sse, mlam, block_rate_bits(&cf, &SCAN_32X32));
                    let _ = qi;
                }
                eff += rate_cost(mlam, mode_signal_bits(m));
                if eff < best_eff {
                    best_eff = eff;
                    y_mode = m;
                }
            }
        }
        // Real coding of the winner: four TX_32X32, each predicted from the
        // running reconstruction, coefficients captured into `lcf`. Skipped in
        // Replay (recon preinstalled, coeffs loaded from the record above).
        if rl.is_none() {
            for (qi, &(sx, sy)) in Self::Q64.iter().enumerate() {
                let (bx, by) = (px + sx, py + sy);
                let (qbx4, qby4) = (bx / 4, by / 4);
                let (tr, bl) = Self::quad_edges(sx, sy, px, py, have_tr, have_bl);
                let mut pred = [0i32; 1024];
                if y_mode == DC_PRED {
                    pred = [dc_pred_32x32(&self.recon[0], self.w, bx, by, self.bd as i32); 1024];
                } else {
                    intra_predict_nd(
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
                        &mut pred,
                        self.bd,
                    );
                }
                let mut resid = [0i32; 1024];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    bx,
                    by,
                    32,
                    32,
                );
                let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_32X32,
                    lam,
                    32,
                    self.dcdf(),
                    3,
                    0,
                    &self.dcdf().eob_bin_1024_l,
                    self.dc_sign_ctx_32(0, qbx4, qby4),
                );
                let rr = idct_dequant_32x32(&cf, &self.quant);
                for ry in 0..32 {
                    let drow = &mut self.recon[0][(by + ry) * self.w + bx..];
                    recon_add_pred(&mut drow[..32], &pred[ry * 32..], &rr[ry * 32..], maxv);
                }
                lcf[qi] = cf;
            }
        }
        let luma_zero = lcf.iter().all(|q| q.iter().all(|&c| c == 0));

        // --- Chroma: DC prediction, a TX_32X32 grid per plane (see
        // `chroma64_geom`). Each transform predicts from the RUNNING chroma
        // reconstruction, exactly as the decoder does per transform block.
        let ncg = cgrid.len();
        let mut ccf = [[[0i32; 1024]; 4]; 2];
        if let Some((cf, _)) = ru_cf.as_ref() {
            for ci in 0..2 {
                for (gi, dst) in ccf[ci].iter_mut().enumerate().take(ncg) {
                    dst.copy_from_slice(&cf[ci][gi * 1024..gi * 1024 + 1024]);
                }
            }
        }
        if ru.is_none() {
            #[allow(clippy::needless_range_loop)]
            for ci in 0..2 {
                let plane = ci + 1;
                for (gi, &(gx, gy)) in cgrid.iter().enumerate() {
                    let (tx0, ty0) = (cx + gx, cy + gy);
                    let dc = dc_pred_32x32(&self.recon[plane], self.cw, tx0, ty0, self.bd as i32);
                    let mut resid = [0i32; 1024];
                    crate::rd_sse::residual_dc(
                        &mut resid,
                        &self.src[plane],
                        self.cw,
                        tx0,
                        ty0,
                        32,
                        32,
                        dc,
                    );
                    let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.cquant);
                    trellis_optimize(&mut cf, &tf, cdcq, cacq, &SCAN_32X32, lam);
                    let mean = resid.iter().sum::<i32>() / 1024;
                    if cf[0] == 0 && mean.abs() >= 8 {
                        cf[0] = if mean > 0 { 1 } else { -1 };
                    }
                    // Reconstruct now so the next transform predicts off it.
                    let rr = idct_dequant_32x32(&cf, &self.cquant);
                    for ry in 0..32 {
                        let drow = &mut self.recon[plane][(ty0 + ry) * self.cw + tx0..];
                        recon_add_dc(&mut drow[..32], dc, &rr[ry * 32..], maxv);
                    }
                    ccf[ci][gi] = cf;
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
            tx: TxSel::SplitDct,
        });
        let mut flat = [0i32; 4096];
        for qi in 0..4 {
            flat[qi * 1024..qi * 1024 + 1024].copy_from_slice(&lcf[qi]);
        }
        self.push_luma_cf(&flat);
        self.push_uv_sel(UvSel { uv: DC_PRED as u8 });
        let mut uflat = [0i32; 4096];
        let mut vflat = [0i32; 4096];
        for gi in 0..ncg {
            uflat[gi * 1024..gi * 1024 + 1024].copy_from_slice(&ccf[0][gi]);
            vflat[gi * 1024..gi * 1024 + 1024].copy_from_slice(&ccf[1][gi]);
        }
        self.push_uv_cf(&uflat[..ncg * 1024], &vflat[..ncg * 1024], [0, 0]);

        // --- Header syntax (decoder order): skip, y_mode, uv_mode, tx_depth.
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens_64(block_skip, sctx);
        self.mark_skip8(x8, y8, 8, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(y_mode, &mut self.cdfs.kf_y[yctx]);
        // CfL is not allowed at 64x64, so uv_mode uses the NOCFL CDF (index m,
        // not 13+m) — emit it directly rather than via `emit_uv_mode`.
        self.enc
            .encode_symbol(DC_PRED, &mut self.cdfs.uv_mode[y_mode]);
        self.commit_uv_mode(px, py, 64, 64, DC_PRED);
        self.emit_palette_mode_info(px, py, 64, 64, y_mode, !self.mono, None);
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
                        &ccf[ci][gi],
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
