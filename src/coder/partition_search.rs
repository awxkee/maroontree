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
    fn rd_cost_square(
        &self,
        px: usize,
        py: usize,
        dim: usize,
        have_tr: bool,
        have_bl: bool,
        prdo: f32,
    ) -> f32 {
        let acq = self.quant.ac_q() as f32;
        let dcq = self.quant.dc_q() as f32;
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let modes: &[usize] = &[DC_PRED, SMOOTH_PRED, PAETH_PRED];
        let mut best = f32::INFINITY;
        match dim {
            8 => {
                let scan = &SCAN_8X8;
                for &m in modes {
                    let mut pred = [0i32; 64];
                    if m == DC_PRED {
                        let d = dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 64];
                    } else {
                        intra_predict_nd(
                            m,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            8,
                            8,
                            have_tr,
                            have_bl,
                            self.w,
                            self.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.bd,
                        );
                    }
                    let mut resid = [0i32; 64];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &pred,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        8,
                        8,
                    );
                    let (mut cf, tf) = forward_dct_quant_8x8_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_8x8(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        8,
                        8,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                    }
                }
            }
            16 => {
                let scan = &SCAN_16X16;
                for &m in modes {
                    let mut pred = [0i32; 256];
                    if m == DC_PRED {
                        let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 256];
                    } else {
                        intra_predict_nd(
                            m,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            16,
                            16,
                            have_tr,
                            have_bl,
                            self.w,
                            self.h,
                            self.luma_filter_type(px, py),
                            &mut pred,
                            self.bd,
                        );
                    }
                    let mut resid = [0i32; 256];
                    crate::rd_sse::residual_pred(
                        &mut resid,
                        &pred,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        16,
                        16,
                    );
                    let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_16x16(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        16,
                        16,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                    }
                }
            }
            32 => {
                let scan = &SCAN_32X32;
                for &m in modes {
                    let mut pred = [0i32; 1024];
                    if m == DC_PRED {
                        let d = dc_pred_32x32(&self.recon[0], self.w, px, py, self.bd as i32);
                        pred = [d; 1024];
                    } else {
                        intra_predict_nd(
                            m,
                            &self.recon[0],
                            self.w,
                            px,
                            py,
                            32,
                            32,
                            have_tr,
                            have_bl,
                            self.w,
                            self.h,
                            self.luma_filter_type(px, py),
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
                        px,
                        py,
                        32,
                        32,
                    );
                    let (mut cf, tf) = forward_dct_quant_32x32_t(&resid, &self.quant);
                    trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
                    let rr = idct_dequant_32x32(&cf, &self.quant);
                    let distortion = self.luma_partition_distortion(
                        px,
                        py,
                        32,
                        32,
                        self.quant.ac_q() as f32,
                        |i| pred[i] + rr[i],
                    );
                    let bits = block_rate_bits(&cf, scan) + mode_signal_bits(m);
                    let cost = crate::partition_rd::rd_cost(distortion, mlam, bits);
                    if cost < best {
                        best = cost;
                    }
                }
            }
            _ => unreachable!("rd_cost_square dim {}", dim),
        }
        best
    }

    fn rd_cost_rect16_leaf_with_dc(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        prdo: f32,
        dc: i32,
    ) -> f32 {
        let acq = self.quant.ac_q() as f32;
        let dcq = self.quant.dc_q() as f32;
        let lam = trellis_lambda();
        let (lam, mlam) = (lam * prdo, self.mlam() * prdo);
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let mut resid = [0i32; 128];
        crate::rd_sse::residual_dc(&mut resid, &self.src[0], self.w, px, py, w, h, dc);
        let (mut cf, tf) = if vert {
            dct8x16_t(&resid, &self.quant)
        } else {
            let (cf, tf) = dct16x8_t(&resid, &self.quant);
            (cf, tf)
        };
        let scan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        trellis_optimize(&mut cf, &tf, dcq, acq, scan, lam);
        let rr = if vert {
            idct_dequant_8x16(&cf, &self.quant)
        } else {
            idct_dequant_16x8(&cf, &self.quant)
        };
        let distortion =
            self.luma_partition_distortion(px, py, w, h, self.quant.ac_q() as f32, |i| dc + rr[i]);
        crate::partition_rd::rd_cost(distortion, mlam, block_rate_bits(&cf, scan))
    }

    fn rd_cost_rect16_leaf(&self, px: usize, py: usize, vert: bool, prdo: f32) -> f32 {
        let dc = if vert {
            dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc)
    }

    /// Reprice a rectangular leaf whose top or left edge belongs to an earlier
    /// sibling in the same asymmetric partition. The normal RDO view still has
    /// stale pixels there; source samples are a close proxy for the sibling's
    /// eventual reconstruction and, by taking the worse cost, prevent a stale
    /// zero edge from making the partition look artificially cheap.
    fn rd_cost_rect16_dependent(
        &self,
        px: usize,
        py: usize,
        vert: bool,
        source_above: bool,
        source_left: bool,
        prdo: f32,
    ) -> f32 {
        let base = self.rd_cost_rect16_leaf(px, py, vert, prdo);
        let (w, h) = if vert { (8usize, 16usize) } else { (16, 8) };
        let above = py > 0;
        let left = px > 0;
        let mut sum = 0i32;
        let mut count = 0i32;
        if above {
            let plane = if source_above {
                &self.src[0]
            } else {
                &self.recon[0]
            };
            sum += plane[(py - 1) * self.w + px..][..w].iter().sum::<i32>();
            count += w as i32;
        }
        if left {
            let plane = if source_left {
                &self.src[0]
            } else {
                &self.recon[0]
            };
            sum += plane[py * self.w + px - 1..]
                .iter()
                .step_by(self.w)
                .take(h)
                .sum::<i32>();
            count += h as i32;
        }
        let dc = if count == 0 {
            1 << (self.bd - 1)
        } else {
            (sum + count / 2) / count
        };
        base.max(self.rd_cost_rect16_leaf_with_dc(px, py, vert, prdo, dc))
    }

    fn rd_cost_horz(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, SPLIT_SIGNAL_BITS)
            + self.rd_cost_rect16_leaf(px, py, false, prdo)
            + self.rd_cost_rect16_leaf(px, py + 8, false, prdo)
    }

    fn rd_cost_vert(&self, px: usize, py: usize, prdo: f32) -> f32 {
        let mlam = self.mlam() * prdo;
        rate_cost(mlam, SPLIT_SIGNAL_BITS)
            + self.rd_cost_rect16_leaf(px, py, true, prdo)
            + self.rd_cost_rect16_leaf(px + 8, py, true, prdo)
    }

    /// Code a 16x16 luma region as PARTITION_H: two stacked 16x8 sub-blocks.
    /// Minimal first implementation: 4:4:4 only, DC prediction, DCT_DCT, no CfL /
    /// SMOOTH_V / mode search. Each 16x8 sub-block is a full intra block (own skip,
    /// y_mode, uv_mode, luma RTX_16X8 coeffs + chroma RTX_16X8 coeffs). Mirrors the
    /// decoder's two `decode_b(PARTITION_H)` calls.
    /// 4:4:4 VERT: two side-by-side 8x16 blocks (luma + chroma), DC intra.
    fn code_block16_vert_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8 + half * 8, y8 * 8);
            let (bx4, by4) = (px / 4, py / 4);
            let lpred = dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            for ry in 0..16 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for cx in 0..8 {
                    lresid[ry * 8 + cx] = srow[cx] - lpred;
                }
            }
            let (mut lcf, ltf) = dct8x16_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_8X16, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_8x16(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                for ry in 0..16 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for cx in 0..8 {
                        resid[ry * 8 + cx] = srow[cx] - dc;
                    }
                }
                let (mut q, qt) = dct8x16_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_8X16, lam);
                let mean_c = resid.iter().sum::<i32>() / 128;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(x8 + half, y8, 2, 4);
            self.mark_skip8_rect(x8 + half, y8, 1, 2, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, 8, 16);
            self.emit_filter_intra(DC_PRED, 8, 16, None);
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 2].fill(sv);
            self.l_skip[by4..by4 + 4].fill(sv);
            self.a_mode[bx4..bx4 + 2].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 4].fill(DC_PRED as u8);
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_8x16_luma();
                let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
                encode_8x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 2].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 4].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_8x16(&lcf, &self.quant)
            };
            for ry in 0..16 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..8], lpred, &lrr[ry * 8..], maxval);
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_8x16_chroma(plane, bx4, by4);
                    let ds = self.dc_sign_ctx_8x16_chroma(plane, bx4, by4);
                    encode_8x16_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][bx4..bx4 + 2].fill(cres_ctx);
                self.l_coef[plane][by4..by4 + 4].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 128]
                } else {
                    idct_dequant_8x16(&ccf[ci], &self.cquant)
                };
                for ry in 0..16 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    recon_add_dc(&mut drow[..8], cpred[ci], &rr[ry * 8..], maxval);
                }
            }
        }
    }

    /// 4:2:0 rect: HORZ = two 16x8 luma + 8x4 chroma; VERT = two 8x16 + 4x8.
    fn code_block16_rect_420(&mut self, x8: usize, y8: usize, vert: bool) {
        for half in 0..2 {
            let (sx8, sy8) = if vert {
                (x8 + half, y8)
            } else {
                (x8, y8 + half)
            };
            self.code_block16_rect_leaf_420(sx8, sy8, vert);
        }
    }

    /// Emit one 16x8/8x16 leaf. Used by binary and asymmetric partitions.
    fn code_block16_rect_leaf_420(&mut self, x8: usize, y8: usize, vert: bool) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let (lw, lh) = if vert { (8usize, 16usize) } else { (16, 8) };
        let lpred = if vert {
            dc_pred_8x16(&self.recon[0], self.w, px, py, self.bd as i32)
        } else {
            dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32)
        };
        let mut lresid = [0i32; 128];
        for ry in 0..lh {
            let srow = &self.src[0][(py + ry) * self.w + px..];
            for cx in 0..lw {
                lresid[ry * lw + cx] = srow[cx] - lpred;
            }
        }
        let (mut lcf, ltf) = if vert {
            dct8x16_t(&lresid, &self.quant)
        } else {
            dct16x8_t(&lresid, &self.quant)
        };
        let lscan: &[u32] = if vert { &SCAN_8X16 } else { &SCAN_16X8 };
        trellis_optimize(&mut lcf, &ltf, dcq, acq, lscan, lam);
        let mean_l = lresid[..lw * lh].iter().sum::<i32>() / (lw * lh) as i32;
        if lcf[0] == 0 && mean_l.abs() >= 8 {
            lcf[0] = if mean_l > 0 { 1 } else { -1 };
        }
        // chroma 8x4 (horz) or 4x8 (vert) at chroma coords.
        let (cx, cy) = (px / 2, py / 2);
        let (cbx4, cby4) = (cx / 4, cy / 4);
        let (cw, ch) = if vert { (4usize, 8usize) } else { (8, 4) };
        let mut ccf = [[0i32; 32]; 2];
        let mut cpred = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = if vert {
                dc_pred_4x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            } else {
                dc_pred_8x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32)
            };
            cpred[ci] = dc;
            let mut resid = [0i32; 32];
            for ry in 0..ch {
                let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                for c in 0..cw {
                    resid[ry * cw + c] = srow[c] - dc;
                }
            }
            let (mut q, qt) = if vert {
                dct4x8_t(&resid, &self.cquant)
            } else {
                dct8x4_t(&resid, &self.cquant)
            };
            let cscan: &[u32] = if vert { &SCAN_4X8 } else { &SCAN_8X4 };
            trellis_optimize(&mut q, &qt, cdcq, cacq, cscan, lam);
            let mean_c = resid[..cw * ch].iter().sum::<i32>() / (cw * ch) as i32;
            if q[0] == 0 && mean_c.abs() >= 8 {
                q[0] = if mean_c > 0 { 1 } else { -1 };
            }
            ccf[ci] = q;
        }
        let luma_zero = lcf.iter().all(|&v| v == 0);
        let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
        let block_skip = luma_zero && chroma_zero;
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        if vert {
            self.record_blk_rect(x8, y8, 2, 4);
            self.mark_skip8_rect(x8, y8, 1, 2, block_skip);
        } else {
            self.record_blk_rect(x8, y8, 4, 2);
            self.mark_skip8_rect(x8, y8, 2, 1, block_skip);
        }
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
        self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, lw, lh);
        self.emit_filter_intra(DC_PRED, lw, lh, None);
        let sv = block_skip as u8;
        let (aw, ah) = (lw / 4, lh / 4);
        self.a_skip[bx4..bx4 + aw].fill(sv);
        self.l_skip[by4..by4 + ah].fill(sv);
        self.a_mode[bx4..bx4 + aw].fill(DC_PRED as u8);
        self.l_mode[by4..by4 + ah].fill(DC_PRED as u8);
        let lres_ctx = if block_skip {
            0x40
        } else if vert {
            let sk = self.skip_ctx_8x16_luma();
            let ds = self.dc_sign_ctx_8x16_luma(bx4, by4);
            encode_8x16_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
        } else {
            let sk = self.skip_ctx_16x8_luma();
            let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
            encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
        };
        self.a_coef[0][bx4..bx4 + aw].fill(lres_ctx);
        self.l_coef[0][by4..by4 + ah].fill(lres_ctx);
        let lrr = if block_skip {
            [0i32; 128]
        } else if vert {
            idct_dequant_8x16(&lcf, &self.quant)
        } else {
            idct_dequant_16x8(&lcf, &self.quant)
        };
        for ry in 0..lh {
            let drow = &mut self.recon[0][(py + ry) * self.w + px..];
            recon_add_dc(&mut drow[..lw], lpred, &lrr[ry * lw..], maxval);
        }
        let (caw, cah) = (cw / 4, (ch / 4).max(1));
        for ci in 0..2 {
            let plane = ci + 1;
            let cres_ctx = if block_skip {
                0x40
            } else if vert {
                let sk = self.skip_ctx_4x8_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_4x8_chroma(plane, cbx4, cby4);
                encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            } else {
                let sk = self.skip_ctx_8x4_chroma(plane, cbx4, cby4);
                let ds = self.dc_sign_ctx_8x4_chroma(plane, cbx4, cby4);
                encode_8x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
            };
            let fillw = caw.max(1);
            self.a_coef[plane][cbx4..cbx4 + fillw].fill(cres_ctx);
            self.l_coef[plane][cby4..cby4 + cah].fill(cres_ctx);
            let rr = if block_skip {
                [0i32; 32]
            } else if vert {
                idct_dequant_4x8(&ccf[ci], &self.cquant)
            } else {
                idct_dequant_8x4(&ccf[ci], &self.cquant)
            };
            for ry in 0..ch {
                let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                recon_add_dc(&mut drow[..cw], cpred[ci], &rr[ry * cw..], maxval);
            }
        }
    }

    /// 4:2:2 HORZ: two 16x8 luma + 8x8 chroma (h-subsampled, v-full). V forbidden in 422.
    fn code_block16_horz_422(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4);
            let lpred = dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            for ry in 0..8 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for cx in 0..16 {
                    lresid[ry * 16 + cx] = srow[cx] - lpred;
                }
            }
            let (mut lcf, ltf) = dct16x8_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_16X8, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            let (cx, cy) = (px / 2, py);
            let (cbx4, cby4) = (cx / 4, cy / 4);
            let mut ccf = [[0i32; 64]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_8x8(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 64];
                for ry in 0..8 {
                    let srow = &self.src[plane][(cy + ry) * self.cw + cx..];
                    for c in 0..8 {
                        resid[ry * 8 + c] = srow[c] - dc;
                    }
                }
                let (mut q, qt) = dct8x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_8X8, lam);
                let mean_c = resid.iter().sum::<i32>() / 64;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, 16, 8);
            self.emit_filter_intra(DC_PRED, 16, 8, None);
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 2].fill(DC_PRED as u8);
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..16], lpred, &lrr[ry * 16..], maxval);
            }
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_8x8_chroma(plane, cbx4, cby4);
                    let ds = self.dc_sign_ctx_8x8_chroma(plane, cbx4, cby4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf[ci],
                        true,
                        sk,
                        ds,
                        DC_PRED,
                        0,
                    )
                };
                self.a_coef[plane][cbx4..cbx4 + 2].fill(cres_ctx);
                self.l_coef[plane][cby4..cby4 + 2].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 64]
                } else {
                    idct_dequant_8x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    recon_add_dc(&mut drow[..8], cpred[ci], &rr[ry * 8..], maxval);
                }
            }
        }
    }

    fn code_block16_horz_444(&mut self, x8: usize, y8: usize) {
        let maxval = (1 << self.bd) - 1;
        let lam = trellis_lambda();
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let (cdcq, cacq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        // Two sub-blocks: half = 0 (top, py), half = 1 (bottom, py+8).
        for half in 0..2 {
            let (px, py) = (x8 * 8, y8 * 8 + half * 8);
            let (bx4, by4) = (px / 4, py / 4); // luma 4-unit coords
            // --- luma 16x8: DC predict, residual, forward, trellis, dc-snap ---
            let lpred = dc_pred_16x8(&self.recon[0], self.w, px, py, self.bd as i32);
            let mut lresid = [0i32; 128];
            for ry in 0..8 {
                let srow = &self.src[0][(py + ry) * self.w + px..];
                for cx in 0..16 {
                    lresid[ry * 16 + cx] = srow[cx] - lpred;
                }
            }
            let (mut lcf, ltf) = dct16x8_t(&lresid, &self.quant);
            trellis_optimize(&mut lcf, &ltf, dcq, acq, &SCAN_16X8, lam);
            let mean_l = lresid.iter().sum::<i32>() / 128;
            if lcf[0] == 0 && mean_l.abs() >= 8 {
                lcf[0] = if mean_l > 0 { 1 } else { -1 };
            }
            // --- chroma 16x8 (4:4:4): DC predict each plane ---
            let mut ccf = [[0i32; 128]; 2];
            let mut cpred = [0i32; 2];
            for ci in 0..2 {
                let plane = ci + 1;
                let dc = dc_pred_16x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = dc;
                let mut resid = [0i32; 128];
                for ry in 0..8 {
                    let srow = &self.src[plane][(py + ry) * self.w + px..];
                    for cx in 0..16 {
                        resid[ry * 16 + cx] = srow[cx] - dc;
                    }
                }
                let (mut q, qt) = dct16x8_t(&resid, &self.cquant);
                trellis_optimize(&mut q, &qt, cdcq, cacq, &SCAN_16X8, lam);
                let mean_c = resid.iter().sum::<i32>() / 128;
                if q[0] == 0 && mean_c.abs() >= 8 {
                    q[0] = if mean_c > 0 { 1 } else { -1 };
                }
                ccf[ci] = q;
            }
            // block_skip iff all planes have no coefficients.
            let luma_zero = lcf.iter().all(|&v| v == 0);
            let chroma_zero = ccf[0].iter().all(|&v| v == 0) && ccf[1].iter().all(|&v| v == 0);
            let block_skip = luma_zero && chroma_zero;
            // --- header: skip, delta-q (once), y_mode (DC), uv_mode (DC) ---
            let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
            self.code_skip_and_sb_tokens(block_skip, sctx);
            // record the 16x8 footprint for the deblock filter: width 4 units,
            // height 2 units (vertical edges every 16, horizontal every 8).
            self.record_blk_rect(x8, y8 + half, 4, 2);
            self.mark_skip8_rect(x8, y8 + half, 2, 1, block_skip);
            let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
                + INTRA_MODE_CTX[self.l_mode[by4] as usize];
            self.enc.encode_symbol(DC_PRED, &mut self.cdfs.kf_y[yctx]);
            self.emit_uv_mode(DC_PRED, DC_PRED, None, px, py, 16, 8);
            self.emit_filter_intra(DC_PRED, 16, 8, None);
            // footprint update: skip/mode over 4 wide x 2 tall units.
            let sv = block_skip as u8;
            self.a_skip[bx4..bx4 + 4].fill(sv);
            self.l_skip[by4..by4 + 2].fill(sv);
            self.a_mode[bx4..bx4 + 4].fill(DC_PRED as u8);
            self.l_mode[by4..by4 + 2].fill(DC_PRED as u8);
            // --- luma coeffs (RTX_16X8) ---
            let lres_ctx = if block_skip {
                0x40
            } else {
                let sk = self.skip_ctx_16x8_luma();
                let ds = self.dc_sign_ctx_16x8_luma(bx4, by4);
                encode_16x8_luma_coeffs(&mut self.enc, &mut self.cdfs, &lcf, sk, ds, DC_PRED, 1)
            };
            self.a_coef[0][bx4..bx4 + 4].fill(lres_ctx);
            self.l_coef[0][by4..by4 + 2].fill(lres_ctx);
            // reconstruct luma
            let lrr = if block_skip {
                [0i32; 128]
            } else {
                idct_dequant_16x8(&lcf, &self.quant)
            };
            for ry in 0..8 {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_dc(&mut drow[..16], lpred, &lrr[ry * 16..], maxval);
            }
            // --- chroma coeffs + reconstruct (4:4:4, both planes RTX_16X8) ---
            for ci in 0..2 {
                let plane = ci + 1;
                let cres_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_16x8_chroma(plane, bx4, by4);
                    let ds = self.dc_sign_ctx_16x8_chroma(plane, bx4, by4);
                    encode_16x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf[ci], sk, ds)
                };
                self.a_coef[plane][bx4..bx4 + 4].fill(cres_ctx);
                self.l_coef[plane][by4..by4 + 2].fill(cres_ctx);
                let rr = if block_skip {
                    [0i32; 128]
                } else {
                    idct_dequant_16x8(&ccf[ci], &self.cquant)
                };
                for ry in 0..8 {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    recon_add_dc(&mut drow[..16], cpred[ci], &rr[ry * 16..], maxval);
                }
            }
        }
    }

    fn code_block16(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 4);
        let (px, py) = (x8 * 8, y8 * 8);
        // luma 16x16 (identical for all subsampling modes)
        // Luma 16x16: same non-directional intra mode search as the 8x8 path.
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda();
        let mlam = self.mlam();
        let dcs16 = self.dc_sign_ctx_16(0, px / 4, py / 4);
        let prdo = self.perceptual_rd_scale(px, py, 16);
        let (lam, mlam) = (lam * prdo, mlam * prdo);
        let mut best_mode = DC_PRED;
        let mut txtp16: u8 = 0; // 0=DCT_DCT 1=ADST_ADST 2=ADST_DCT 3=DCT_ADST
        let mut lpred_arr = [0i32; 256];
        let mut lcf = [0i32; 256];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        let mut ltf = [0f32; 256]; // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Pure-emit replay: the recorded winner + its captured coefficients
        // replace every sub-search below — no candidate is evaluated at all
        // (see code_block in block8.rs for the pattern).
        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let directional_top = if rl.is_none() {
            self.rank_luma_directionals::<256>(modes, px, py, 16, 16, have_tr, have_bl)
        } else {
            DirectionalTopK::new()
        };
        for &m in modes {
            if rl.is_some() {
                break;
            }
            if is_directional_mode(m) && !directional_top.contains(m) {
                continue;
            }
            let mut pred = [0i32; 256];
            if m == DC_PRED {
                let d = dc_pred_16x16(&self.recon[0], self.w, px, py, self.bd as i32);
                pred = [d; 256];
            } else {
                intra_predict_nd(
                    m,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 16, 16);
            let blk_sse16 = |rr: &[i32; 256]| -> i64 {
                sse_recon::<256, 16>(&pred, rr, &self.src[0], self.w, px, py, self.bd)
            };
            let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
            if self.speed.per_candidate_rdoq() {
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
            }
            let sse = blk_sse16(&idct_dequant_16x16(&cf, &self.quant));
            let bits = block_rate_bits(&cf, &SCAN_16X16);
            let filter_bits = if m == DC_PRED {
                cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(16, 16)], 0)
            } else {
                0.0
            };
            let cost = rd_cost_i64(sse, mlam, bits + mode_signal_bits(m) + filter_bits);
            if cost < best_eff {
                best_eff = cost;
                best_mode = m;
                lpred_arr = pred;
                lcf = cf;
                ltf = tf;
                best_dct_sse = sse;
                best_dct_bits = bits;
                best_filter_intra = None;
            }
        }
        if rl.is_none() && self.speed == Speed::Slow {
            let bsize = av1_block_size_index(16, 16);
            for filter_mode in FILTER_INTRA_MODES {
                let mut pred = [0i32; 256];
                filter_intra_predict(
                    filter_mode,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                trellis_optimize_ctx(
                    &mut cf,
                    &tf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
                let rr = idct_dequant_16x16(&cf, &self.quant);
                let sse =
                    sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = block_rate_bits(&cf, &SCAN_16X16);
                let syntax_bits = mode_signal_bits(DC_PRED)
                    + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                    + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                if rl.is_some() || (filter_intra_sse_allowed(sse, best_dct_sse) && cost < best_eff)
                {
                    best_eff = cost;
                    best_mode = DC_PRED;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                    best_filter_intra = Some(filter_mode);
                }
            }
        }
        // Angle-delta winner refinement (see code_block: diagonals only, -3..=3).
        let mut best_delta: i32 = 0;
        if rl.is_none()
            && angle_delta_enabled()
            && self.speed.try_angle_deltas()
            && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
            && best_mode != V_PRED
            && best_mode != H_PRED
        {
            let mut ad_cdf = [0u16; 7];
            ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
            let mut best_ad_cost =
                rd_cost_i64(best_dct_sse, mlam, best_dct_bits + cdf_cost(&ad_cdf, 3));
            for d in [-3i32, -2, -1, 1, 2, 3] {
                let mut pred = [0i32; 256];
                intra_predict_nd_ad(
                    best_mode,
                    d,
                    &self.recon[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                    have_tr,
                    have_bl,
                    self.w,
                    self.h,
                    self.luma_filter_type(px, py),
                    &mut pred,
                    self.bd,
                );
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &pred,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut cf, tf) = forward_dct_quant_16x16_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_16X16,
                        lam,
                        16,
                        self.dcdf(),
                        2,
                        0,
                        &self.dcdf().eob_bin_256_l,
                        dcs16,
                    );
                }
                let rr = idct_dequant_16x16(&cf, &self.quant);
                let sse = sse_recon::<256, 16>(&pred, &rr, &self.src[0], self.w, px, py, self.bd);
                let bits = block_rate_bits(&cf, &SCAN_16X16);
                let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                if rl.is_some() || cost < best_ad_cost {
                    best_ad_cost = cost;
                    best_delta = d;
                    lpred_arr = pred;
                    lcf = cf;
                    ltf = tf;
                    best_dct_sse = sse;
                    best_dct_bits = bits;
                }
            }
        }
        // Fast path: run RDOQ once, on the winning mode only (libaom
        // winner-mode coeff opt). The decision above used un-trellised costs.
        if rl.is_none() && !self.speed.per_candidate_rdoq() {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
            );
        }
        // Winner-only ADST_ADST refinement. Full and Medium try it; only Fast
        // prunes the transform-type search to DCT_DCT (libaom-style).
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 256];
            crate::rd_sse::residual_pred(
                &mut resid,
                &lpred_arr,
                &self.src[0],
                self.w,
                px,
                py,
                16,
                16,
            );
            let (mut acf, atf) = adst16x16_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_16X16,
                lam,
                16,
                self.dcdf(),
                2,
                0,
                &self.dcdf().eob_bin_256_l,
                dcs16,
            );
            let rr = iadst_dequant_16x16(&acf, &self.quant);
            let asse = sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
            let abits = block_rate_bits(&acf, &SCAN_16X16);
            // Quality guard: only accept ADST if it does not meaningfully worsen
            // SSE. At low quality lambda (~quantizer^2) is enormous, so a pure
            // RD test would pick ADST whenever it shaves a few bits even while
            // inflating distortion ~2x; that tanks perceptual quality (SSIMULACRA2)
            // for a trivial rate gain. Requiring SSE-non-worsening keeps the
            // genuine high-quality ADST wins (where it lowers SSE) and blocks the
            // low-quality "trade quality for bits" pathology.
            if rl.is_some()
                || (asse <= best_dct_sse + (best_dct_sse >> 5)
                    && rd_cost_i64(asse, mlam, abits)
                        < rd_cost_i64(best_dct_sse, mlam, best_dct_bits))
            {
                lcf = acf;
                txtp16 = 1;
            }
        }
        // Asymmetric-ADST refinement (ADST_DCT / DCT_ADST) for TX_16X16, same
        // rationale as the 8x8 path. Competes with the running tx winner.
        if rl.is_none() && self.speed.try_adst() && asym_adst_enabled() {
            let mut best_txtp16_sse = if txtp16 == 1 { i64::MAX } else { best_dct_sse };
            let mut best_txtp16_bits = best_dct_bits;
            if txtp16 == 1 {
                // recompute the ADST_ADST winner cost as the bar to beat
                let rr = iadst_dequant_16x16(&lcf, &self.quant);
                best_txtp16_sse =
                    sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                best_txtp16_bits = block_rate_bits(&lcf, &SCAN_16X16);
            }
            for (fwd_dctadst, inv_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 256];
                crate::rd_sse::residual_pred(
                    &mut resid,
                    &lpred_arr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    16,
                    16,
                );
                let (mut acf, atf) = if fwd_dctadst {
                    dctadst16x16_t(&resid, &self.quant)
                } else {
                    adstdct16x16_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_16X16,
                    lam,
                    16,
                    self.dcdf(),
                    2,
                    0,
                    &self.dcdf().eob_bin_256_l,
                    dcs16,
                );
                let rr = if inv_dctadst {
                    idctadst_dequant_16x16(&acf, &self.quant)
                } else {
                    iadstdct_dequant_16x16(&acf, &self.quant)
                };
                let asse =
                    sse_recon::<256, 16>(&lpred_arr, &rr, &self.src[0], self.w, px, py, self.bd);
                let abits = block_rate_bits(&acf, &SCAN_16X16);
                if rl.is_some()
                    || (asse <= best_dct_sse + (best_dct_sse >> 5)
                        && rd_cost_i64(asse, mlam, abits)
                            < rd_cost_i64(best_txtp16_sse, mlam, best_txtp16_bits))
                {
                    lcf = acf;
                    txtp16 = if inv_dctadst { 3 } else { 2 };
                    best_txtp16_sse = asse;
                    best_txtp16_bits = abits;
                }
            }
        }
        // Pure-emit replay: install the recorded winner and its captured
        // post-trellis coefficients (every luma sub-search above was skipped).
        if let Some(r) = rl {
            best_mode = r.mode as usize;
            best_delta = r.delta as i32;
            best_filter_intra = FILTER_INTRA_MODES
                .iter()
                .copied()
                .find(|&f| f as u8 == r.filter);
            txtp16 = match r.tx {
                TxSel::Adst => 1,
                TxSel::AdstDct => 2,
                TxSel::DctAdst => 3,
                _ => 0,
            };
        }
        if let Some(cf) = rl_cf {
            lcf.copy_from_slice(&cf);
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            // No IDTX sub-search at 16x16; txtp16 covers DCT/ADST/asym only.
            tx: TxSel::from_flags(txtp16 == 1, false, txtp16 == 2, txtp16 == 3),
        });
        self.push_luma_cf(&lcf);
        let luma_zero = lcf.iter().all(|&c| c == 0);
        if self.ss420 {
            self.code_block16_420(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra,
            );
        } else if self.ss422 {
            self.code_block16_422(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra,
            );
        } else {
            self.code_block16_444(
                x8, y8, &lcf, &lpred_arr, best_mode, luma_zero, txtp16, best_delta,
                best_filter_intra,
            );
        }
    }
}
