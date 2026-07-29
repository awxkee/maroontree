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

/// Keep large, independent mode-search phases in separate LLVM optimization
/// units. Each call is once per block, so the call boundary is negligible next
/// to the transform search it outlines.
#[inline(never)]
fn outline_block8<R>(f: impl FnOnce() -> R) -> R {
    f()
}

const JOINT_LUMA_BEAM: usize = 2;
const JOINT_LARGE_BEAM: usize = 2;

#[derive(Clone)]
struct Luma8BeamCandidate {
    luma_cost: f32,
    mode: usize,
    pred: [i32; 64],
    cf: [i32; 64],
    tf: [f32; 64],
    sse: i64,
    bits: f32,
    palette: Option<LossyLumaPalette>,
}

/// Exact DC-vs-CfL chroma score for one reconstructed luma candidate.  The
/// shape-specific wrapper supplies fixed-size transforms so this stays
/// allocation-free even when it is used by the partition proxy.
macro_rules! joint_uv_shape_cost {
    ($this:expr, $n:expr, $cw:expr, $ch:expr, $cx:expr, $cy:expr, $ac:expr,
     $scan:expr, $fwd:expr, $inv:expr, $bits:expr, $dc_pred:expr,
     $y_mode:expr, $lam:expr, $mlam:expr, $sbi:ident, $sbu:ident) => {{
        let ac_ref = $ac;
        let mut dc_sse = 0i64;
        let mut dc_bits = 0.0f32;
        let mut cfl_sse = 0i64;
        let mut cfl_bits = 0.0f32;
        let mut alpha = [0i32; 2];
        for ci in 0..2 {
            let plane = ci + 1;
            let dc = $dc_pred(plane);
            let mut src = $this.$sbu();
            $this.rd.copy_block_u16(
                &mut src[..$n],
                &$this.src[plane],
                $this.cw,
                $cx,
                $cy,
                $cw,
                $ch,
            );
            let mut dr = $this.$sbi();
            $this
                .rd
                .residual_dc(&mut dr[..], &src[..], $cw, 0, 0, $cw, $ch, dc);
            let (mut dcf, dtf) = $fwd(&dr, &$this.cquant);
            trellis_optimize(
                &mut dcf,
                &dtf,
                $this.cquant.dc_q() as f32,
                $this.cquant.ac_q() as f32,
                $scan,
                $lam,
            );
            let drr = $inv(&dcf, &$this.cquant);
            dc_sse += $this
                .rd
                .sse_recon(&[dc; $n], &drr, &src[..], $cw, 0, 0, $cw, $ch, $this.bd);
            dc_bits += $bits(&dcf, plane);

            let a = $this
                .intrapred
                .cfl_best_alpha(&ac_ref[..], &src[..], dc, $n, $this.bd);
            alpha[ci] = a;
            let mut pred = $this.$sbi();
            $this
                .intrapred
                .cfl_pred(&mut pred[..$n], &ac_ref[..$n], dc, a, $this.bd);
            let mut cr = $this.$sbi();
            $this
                .rd
                .residual_pred(&mut cr[..], &pred[..], &src[..], $cw, 0, 0, $cw, $ch);
            let (mut ccf, ctf) = $fwd(&cr, &$this.cquant);
            trellis_optimize(
                &mut ccf,
                &ctf,
                $this.cquant.dc_q() as f32,
                $this.cquant.ac_q() as f32,
                $scan,
                $lam,
            );
            let crr = $inv(&ccf, &$this.cquant);
            cfl_sse += $this
                .rd
                .sse_recon(&pred[..], &crr, &src[..], $cw, 0, 0, $cw, $ch, $this.bd);
            cfl_bits += $bits(&ccf, plane);
        }
        let dc = rd_cost_i64(
            dc_sse,
            $mlam,
            dc_bits + $this.uv_mode_bits($y_mode, DC_PRED, None),
        );
        let cfl = if alpha != [0, 0] {
            rd_cost_i64(
                cfl_sse,
                $mlam,
                cfl_bits + $this.uv_mode_bits($y_mode, CFL_PRED, Some(alpha)),
            )
        } else {
            f32::INFINITY
        };
        dc.min(cfl)
    }};
}

impl<'a> LossyTile<'a> {
    /// Best DC/CfL chroma cost for one reconstructed 8x8 luma candidate.  These
    /// are the UV alternatives whose prediction actually depends on luma; the
    /// directional UV residuals are invariant across luma candidates and differ
    /// only by `uv_mode[13 + y_mode]`, which is included by `uv_mode_bits`.
    /// Trial-code an 8x8 luma block as FOUR TX_4X4 (`tx_depth = 1`), raster
    /// order, per-TX sequential intra prediction (spec) with the full 4x4
    /// tx-type set per sub (DCT/ADST trellis'd, IDTX/V/H plain — the same
    /// candidate set and gates as the SPLIT4 4x4 trial). This is the measured
    /// aom pattern on grainy photos: small transforms localize noise energy
    /// (sparse eob) where one TX_8X8 smears it across a dense block.
    /// Temporarily writes candidate recon into `self.recon[0]` and restores.
    /// Returns (packed cf quadrant-major 4x16, recon, sse, bits, per-sub txtp).
    #[allow(clippy::too_many_arguments)]
    fn split4tx_try(
        &mut self,
        px: usize,
        py: usize,
        mode: usize,
        delta: i32,
        have_tr: bool,
        have_bl: bool,
        lam: f32,
        rd_lam: f32,
    ) -> ([i32; 64], [u16; 64], i64, f32, [u8; 4]) {
        let mut saved = [0u16; 64];
        for ry in 0..8 {
            saved[ry * 8..ry * 8 + 8]
                .copy_from_slice(&self.recon[0][(py + ry) * self.w + px..][..8]);
        }
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let block_ftype = self.luma_filter_type(px, py);
        let mut cf4 = [0i32; 64];
        let mut rec = [0u16; 64];
        let mut sse_sum = 0i64;
        let mut bits_sum = 0f32;
        let mut txtps = [1u8; 4];
        for (qi, &(sx, sy)) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)]
            .iter()
            .enumerate()
        {
            let (bx, by) = (px + sx, py + sy);
            // Per-TX edge availability (same z-order rules as the SplitDct-16
            // quadrants, scaled to 4px).
            let (tr, bl) = match (sx, sy) {
                (0, 0) => (py > 0, px > 0),
                (4, 0) => (have_tr, false),
                (0, 4) => (true, have_bl),
                _ => (false, false),
            };
            let mut pred = [0i32; 16];
            if mode == DC_PRED {
                let d = self
                    .intrapred
                    .dc_pred_4x4(&self.recon[0], self.w, bx, by, self.bd as i32);
                pred = [d; 16];
            } else {
                self.intrapred.predict_nd_ad(
                    mode,
                    delta,
                    &self.recon[0],
                    self.w,
                    bx,
                    by,
                    4,
                    4,
                    tr,
                    bl,
                    self.w,
                    self.h,
                    block_ftype,
                    &mut pred,
                    self.bd,
                );
            }
            let mut resid = [0i32; 16];
            self.rd
                .residual_pred(&mut resid, &pred, &self.src[0], self.w, bx, by, 4, 4);
            let (mut dcf, dtf) = self.dct.dct4x4_t(&resid, &self.quant);
            trellis_optimize(&mut dcf, &dtf, dcq, acq, &SCAN_4X4, lam);
            let drr = self.idct.idct_dequant_4x4(&dcf, &self.quant);
            let dct_sse =
                sse_recon::<16, 4>(&self.rd, &pred, &drr, &self.src[0], self.w, bx, by, self.bd);
            let dct_bits = self.luma_bits(&dcf, &SCAN_4X4, 4, bx, by, mode, 1);
            let mut best = (dcf, drr, dct_sse, dct_bits, 1u8);
            // V_DCT/H_DCT are NOT trialled here. Ablation 2026-07-25: the two
            // 1-D classes cost 3-5.5% of encode time at 4x4 and buy nothing
            // measurable (420 tuning +0.03 / holdout +0.03, 444 -0.00 /
            // -0.02 -- noise in both directions). ADST and IDTX carry the
            // whole 4x4 program; the kernels stay for the sizes that earn.
            for txtp in [4u8, 0] {
                let (mut acf, atf) = match txtp {
                    4 => self.dct.adst4x4_t(&resid, &self.quant),
                    _ => self.dct.idtx4x4_t(&resid, &self.quant),
                };
                if txtp == 4 {
                    trellis_optimize(&mut acf, &atf, dcq, acq, &SCAN_4X4, lam);
                }
                let arr = match txtp {
                    4 => self.idct.iadst_dequant_4x4(&acf, &self.quant),
                    _ => self.idct.iidentity_dequant_4x4(&acf, &self.quant),
                };
                let asse = sse_recon::<16, 4>(
                    &self.rd,
                    &pred,
                    &arr,
                    &self.src[0],
                    self.w,
                    bx,
                    by,
                    self.bd,
                );
                if asse > dct_sse + (dct_sse >> 5) {
                    continue;
                }
                let bits_bound = (rd_cost_i64(best.2, rd_lam, best.3) - asse as f32) / rd_lam;
                let abits = self.luma_bits_bounded(
                    &acf,
                    &SCAN_4X4,
                    4,
                    bx,
                    by,
                    mode,
                    txtp as usize,
                    bits_bound,
                );
                if rd_cost_i64(asse, rd_lam, abits) < rd_cost_i64(best.2, rd_lam, best.3) {
                    best = (acf, arr, asse, abits, txtp);
                }
            }
            let (bcf, brr, bsse, bbits, btxtp) = best;
            self.rd.reconstruct(
                &mut self.recon[0][by * self.w + bx..],
                self.w,
                Some((&mut rec[sy * 8 + sx..], 8)),
                &pred,
                &brr,
                4,
                4,
                self.bd,
            );
            cf4[qi * 16..qi * 16 + 16].copy_from_slice(&bcf);
            txtps[qi] = btxtp;
            sse_sum += bsse;
            bits_sum += bbits;
        }
        for ry in 0..8 {
            self.recon[0][(py + ry) * self.w + px..][..8]
                .copy_from_slice(&saved[ry * 8..ry * 8 + 8]);
        }
        (cf4, rec, sse_sum, bits_sum, txtps)
    }

    fn joint_uv_cost8(
        &self,
        lpred: &[i32; 64],
        lcf: &[i32; 64],
        y_mode: usize,
        px: usize,
        py: usize,
        prdo: f32,
    ) -> f32 {
        if self.mono {
            return 0.0;
        }
        let lrr = self.idct.idct_dequant_8x8(lcf, &self.quant);
        let mut luma_rec = [0u16; 64];
        recon_add_pred(&mut luma_rec, lpred, &lrr, (1 << self.bd) - 1);
        let (dcq, acq) = (self.cquant.dc_q() as f32, self.cquant.ac_q() as f32);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let cx = px >> 1;

        macro_rules! score_shape {
            ($n:expr, $cw:expr, $ch:expr, $cx:expr, $cy:expr, $ac:expr,
             $scan:expr, $fwd:expr, $inv:expr, $bits:expr, $dc_pred:expr) => {{
                let ac_ref = $ac;
                let mut dc_sse = 0i64;
                let mut dc_bits = 0.0f32;
                let mut cfl_sse = 0i64;
                let mut cfl_bits = 0.0f32;
                let mut alpha = [0i32; 2];
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = $dc_pred(plane);
                    let mut src = [0u16; $n];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, $cx, $cy, $cw, $ch);
                    let mut dr = [0i32; $n];
                    self.rd.residual_dc(&mut dr, &src, $cw, 0, 0, $cw, $ch, dc);
                    let (mut dcf, dtf) = $fwd(&dr, &self.cquant);
                    trellis_optimize(&mut dcf, &dtf, dcq, acq, $scan, lam);
                    let drr = $inv(&dcf, &self.cquant);
                    dc_sse += self
                        .rd
                        .sse_recon(&[dc; $n], &drr, &src, $cw, 0, 0, $cw, $ch, self.bd);
                    dc_bits += $bits(&dcf, plane);

                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac_ref, &src, dc, $n, self.bd);
                    alpha[ci] = a;
                    let mut pred = [0i32; $n];
                    self.intrapred
                        .cfl_pred(&mut pred, &ac_ref[..$n], dc, a, self.bd);
                    let mut cr = [0i32; $n];
                    self.rd
                        .residual_pred(&mut cr, &pred, &src, $cw, 0, 0, $cw, $ch);
                    let (mut ccf, ctf) = $fwd(&cr, &self.cquant);
                    trellis_optimize(&mut ccf, &ctf, dcq, acq, $scan, lam);
                    let crr = $inv(&ccf, &self.cquant);
                    cfl_sse += self
                        .rd
                        .sse_recon(&pred, &crr, &src, $cw, 0, 0, $cw, $ch, self.bd);
                    cfl_bits += $bits(&ccf, plane);
                }
                let dc = rd_cost_i64(
                    dc_sse,
                    mlam,
                    dc_bits + self.uv_mode_bits(y_mode, DC_PRED, None),
                );
                let cfl = if alpha != [0, 0] {
                    rd_cost_i64(
                        cfl_sse,
                        mlam,
                        cfl_bits + self.uv_mode_bits(y_mode, CFL_PRED, Some(alpha)),
                    )
                } else {
                    f32::INFINITY
                };
                dc.min(cfl)
            }};
        }

        // Normalize the sum of the two chroma planes to a common full-luma
        // footprint. The 1/32 calibration matches the shared-tree SSE/SATD
        // scale; inverse sample density prevents subsampled colored edges from
        // disappearing merely because they contain fewer stored samples.
        if self.ss420 {
            let cy = py >> 1;
            let mut ac = [0i32; 16];
            self.intrapred
                .cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
            0.125
                * score_shape!(
                    16,
                    4,
                    4,
                    cx,
                    cy,
                    ac,
                    &SCAN_4X4,
                    |r, q| self.dct.dct4x4_t(r, q),
                    |levels, q| self.idct.idct_dequant_4x4(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_4X4, 4, plane, cx, cy),
                    |plane: usize| self.intrapred.dc_pred_4x4(
                        self.recon[plane].as_slice(),
                        self.cw,
                        cx,
                        cy,
                        self.bd as i32
                    )
                )
        } else if self.ss422 {
            let mut ac = [0i32; 32];
            self.intrapred
                .cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
            0.0625
                * score_shape!(
                    32,
                    4,
                    8,
                    cx,
                    py,
                    ac,
                    &SCAN_4X8,
                    |r, q| self.dct.dct4x8_t(r, q),
                    |levels, q| self.idct.idct_dequant_4x8(levels, q),
                    |cf: &[i32], plane| self.chroma_rect_bits(cf, &SCAN_4X8, 4, 8, plane, cx, py),
                    |plane: usize| self.intrapred.dc_pred_4x8(
                        self.recon[plane].as_slice(),
                        self.cw,
                        cx,
                        py,
                        self.bd as i32
                    )
                )
        } else {
            let mut ac = [0i32; 64];
            self.intrapred.cfl_ac_444(&luma_rec, 8, 8, &mut ac);
            0.03125
                * score_shape!(
                    64,
                    8,
                    8,
                    px,
                    py,
                    ac,
                    &SCAN_8X8,
                    |r, q| self.dct.dct8x8_t(r, q),
                    |levels, q| self.idct.idct_dequant_8x8(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_8X8, 8, plane, px, py),
                    |plane: usize| self.intrapred.dc_pred_8x8(
                        self.recon[plane].as_slice(),
                        self.w,
                        px,
                        py,
                        self.bd as i32
                    )
                )
        }
    }

    /// 16x16 counterpart of `joint_uv_cost8`, used by the final luma beam and
    /// by square-leaf partition RDO.  Only DC and CfL are relevant here because
    /// other UV predictors do not depend on the luma candidate.
    fn joint_uv_cost16(
        &self,
        lpred: &[i32; 256],
        lcf: &[i32; 256],
        y_mode: usize,
        px: usize,
        py: usize,
        prdo: f32,
    ) -> f32 {
        if self.mono {
            return 0.0;
        }
        let lrr = self.idct.idct_dequant_16x16(lcf, &self.quant);
        let mut luma_rec = self.sbuf_u256();
        recon_add_pred(&mut luma_rec[..], lpred, &lrr, (1 << self.bd) - 1);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let cx = px >> 1;
        if self.ss420 {
            let cy = py >> 1;
            let mut ac = self.sbuf_i64();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 16, 8, 8, true, true, &mut ac[..]);
            0.125
                * joint_uv_shape_cost!(
                    self,
                    64,
                    8,
                    8,
                    cx,
                    cy,
                    ac,
                    &SCAN_8X8,
                    |r, q| self.dct.dct8x8_t(r, q),
                    |levels, q| self.idct.idct_dequant_8x8(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_8X8, 8, plane, cx, cy),
                    |plane: usize| self.intrapred.dc_pred_8x8(
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i64,
                    sbuf_u64
                )
        } else if self.ss422 {
            let mut ac = self.sbuf_i128();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 16, 8, 16, true, false, &mut ac[..]);
            0.0625
                * joint_uv_shape_cost!(
                    self,
                    128,
                    8,
                    16,
                    cx,
                    py,
                    ac,
                    &SCAN_8X16,
                    |r, q| self.dct.dct8x16_t(r, q),
                    |levels, q| self.idct.idct_dequant_8x16(levels, q),
                    |cf: &[i32], plane| self.chroma_rect_bits(cf, &SCAN_8X16, 8, 16, plane, cx, py),
                    |plane: usize| self.intrapred.dc_pred_8x16(
                        &self.recon[plane],
                        self.cw,
                        cx,
                        py,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i128,
                    sbuf_u128
                )
        } else {
            let mut ac = self.sbuf_i256();
            self.intrapred
                .cfl_ac_444(&luma_rec[..], 16, 16, &mut ac[..]);
            0.03125
                * joint_uv_shape_cost!(
                    self,
                    256,
                    16,
                    16,
                    px,
                    py,
                    ac,
                    &SCAN_16X16,
                    |r, q| self.dct.dct16x16_t(r, q),
                    |levels, q| self.idct.idct_dequant_16x16(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_16X16, 16, plane, px, py),
                    |plane: usize| self.intrapred.dc_pred_16x16(
                        &self.recon[plane],
                        self.w,
                        px,
                        py,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i256,
                    sbuf_u256
                )
        }
    }

    /// 32x32 counterpart of `joint_uv_cost8`.  TX_32X32 is DCT-only, making
    /// the reconstructed luma candidate identical to the one eventually coded.
    fn joint_uv_cost32(
        &self,
        lpred: &[i32; 1024],
        lcf: &[i32; 1024],
        y_mode: usize,
        px: usize,
        py: usize,
        prdo: f32,
    ) -> f32 {
        if self.mono {
            return 0.0;
        }
        let lrr = self.idct.idct_dequant_32x32(lcf, &self.quant);
        let mut luma_rec = self.sbuf_u1024();
        recon_add_pred(&mut luma_rec[..], lpred, &lrr, (1 << self.bd) - 1);
        let lam = trellis_lambda() * prdo;
        let mlam = self.mlam_c() * prdo;
        let cx = px >> 1;
        if self.ss420 {
            let cy = py >> 1;
            let mut ac = self.sbuf_i256();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 32, 16, 16, true, true, &mut ac[..]);
            0.125
                * joint_uv_shape_cost!(
                    self,
                    256,
                    16,
                    16,
                    cx,
                    cy,
                    ac,
                    &SCAN_16X16,
                    |r, q| self.dct.dct16x16_t(r, q),
                    |levels, q| self.idct.idct_dequant_16x16(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_16X16, 16, plane, cx, cy),
                    |plane: usize| self.intrapred.dc_pred_16x16(
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i256,
                    sbuf_u256
                )
        } else if self.ss422 {
            let mut ac = self.sbuf_i512();
            self.intrapred
                .cfl_ac_sub(&luma_rec[..], 32, 16, 32, true, false, &mut ac[..]);
            0.0625
                * joint_uv_shape_cost!(
                    self,
                    512,
                    16,
                    32,
                    cx,
                    py,
                    ac,
                    &SCAN_16X32,
                    |r, q| self.dct.dct16x32_t(r, q),
                    |levels, q| self.idct.idct_dequant_16x32(levels, q),
                    |cf: &[i32], plane| self.chroma_rect_bits(
                        cf,
                        &SCAN_16X32,
                        16,
                        32,
                        plane,
                        cx,
                        py
                    ),
                    |plane: usize| self.intrapred.dc_pred_16x32(
                        &self.recon[plane],
                        self.cw,
                        cx,
                        py,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i512,
                    sbuf_u512
                )
        } else {
            let mut ac = self.sbuf_i1024();
            self.intrapred
                .cfl_ac_444(&luma_rec[..], 32, 32, &mut ac[..]);
            0.03125
                * joint_uv_shape_cost!(
                    self,
                    1024,
                    32,
                    32,
                    px,
                    py,
                    ac,
                    &SCAN_32X32,
                    |r, q| self.dct.dct32x32_t(r, q),
                    |levels, q| self.idct.idct_dequant_32x32(levels, q),
                    |cf: &[i32], plane| self.chroma_bits(cf, &SCAN_32X32, 32, plane, px, py),
                    |plane: usize| self.intrapred.dc_pred_32x32(
                        &self.recon[plane],
                        self.w,
                        px,
                        py,
                        self.bd as i32
                    ),
                    y_mode,
                    lam,
                    mlam,
                    sbuf_i1024,
                    sbuf_u1024
                )
        }
    }

    fn code_block(&mut self, x8: usize, y8: usize, have_tr: bool, have_bl: bool) {
        self.record_blk(x8, y8, 2);
        let (px, py) = (x8 * 8, y8 * 8);
        let (bx4, by4) = (px / 4, py / 4);
        let cx = px / 2; // chroma column for 4:2:2

        // Forward-transform/quantize all planes up front to decide block skip.
        // Luma is always 8x8; chroma is 8x8 (4:4:4) or 4x8 (4:2:2).
        // Luma 8x8: search the non-directional intra modes (DC + SMOOTH*/PAETH)
        // and keep the one minimizing pixel SSE + lambda * estimated bits. The
        // chosen prediction is per-pixel; reconstruction uses the same array so
        // the decoder (which re-derives the identical prediction) stays bit-exact.
        let (dcq, acq) = (self.quant.dc_q() as f32, self.quant.ac_q() as f32);
        let lam = trellis_lambda() * self.emit_prdo(x8 * 8, y8 * 8, 8);
        let mlam = self.emit_mlam(x8 * 8, y8 * 8, 8);
        // emit_prdo/emit_mlam already carry the perceptual scale
        let prdo = self.perceptual_rd_scale(px, py, 8);
        let mut best_mode = DC_PRED;
        let mut best_is_adst = false;
        let mut best_is_idtx = false;
        let mut best_is_vdct = false;
        let mut best_is_hdct = false;
        let mut best_is_adstdct = false;
        let mut best_is_dctadst = false;
        let mut lpred_arr = [0i32; 64];
        let mut lcf = [0i32; 64];
        let mut best_eff = f32::INFINITY;
        let mut best_dct_sse = 0i64;
        let mut best_dct_bits = 0f32;
        let mut best_filter_intra = None;
        let mut best_palette: Option<LossyLumaPalette> = None;
        let mut luma_beam: [Option<Luma8BeamCandidate>; JOINT_LUMA_BEAM] =
            std::array::from_fn(|_| None);
        let dc_sgn = self.dc_sign_ctx(0, px / 4, py / 4);
        let mut ltf = [0f32; 64]; // winner transform coeffs (f32, for winner-only RDOQ)
        let modes = if self.speed.reduced_modes() {
            fast_nd_modes()
        } else {
            nd_modes()
        };
        // Pure-emit replay: the recorded winner + its captured coefficients
        // replace every sub-search below — no candidate is evaluated at all;
        // the winner state installs just before `push_luma_sel`.
        let rl = self.luma_sel_replay();
        let rl_cf = self.luma_cf_replay();
        let mode_shortlist = if rl.is_none() {
            self.rank_luma_modes::<64>(
                modes,
                px,
                py,
                8,
                8,
                have_tr,
                have_bl,
                self.luma_mode_budget_eff(),
            )
        } else {
            FixedList::new(DC_PRED)
        };
        outline_block8(|| {
            for &m in modes {
                if rl.is_some() {
                    break;
                }
                if !mode_shortlist.contains(&m) {
                    continue;
                }
                let mut pred = [0i32; 64];
                if m == DC_PRED {
                    let d =
                        self.intrapred
                            .dc_pred_8x8(&self.recon[0], self.w, px, py, self.bd as i32);
                    pred = [d; 64];
                } else {
                    self.intrapred.predict_nd(
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
                self.rd
                    .residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
                // Mode decision uses DCT_DCT only (cheap); the ADST_ADST transform
                // choice is refined once for the winning mode after the loop.
                // KEY NEG (2026-07-21, do not retry): ranking candidates by
                // min(DCT, guarded-ADST) instead — full joint mode+tx selection
                // at 8x8 AND 16x16 — was ceiling-probed at -0.01%/-0.06%
                // HOLDOUT (420/444); winner-only refinement already captures
                // everything that generalizes.
                let blk_sse = |rr: &[i32; 64]| -> i64 {
                    sse_recon::<64, 8>(&self.rd, &pred, rr, &self.src[0], self.w, px, py, self.bd)
                };
                let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                if self.speed.per_candidate_rdoq_av1() {
                    trellis_optimize_ctx(
                        &mut cf,
                        &tf,
                        dcq,
                        acq,
                        &SCAN_8X8,
                        lam,
                        8,
                        8,
                        self.dcdf(),
                        1,
                        0,
                        &self.dcdf().eob_bin_64_l,
                        dc_sgn,
                        self.quant.qm_level(),
                        self.quant.qidx() as i32,
                    );
                }
                let sse = blk_sse(&self.idct.idct_dequant_8x8(&cf, &self.quant));
                let bits = self.luma_bits(&cf, &SCAN_8X8, 8, px, py, m, 1);
                let filter_bits = if m == DC_PRED {
                    cdf_cost(&self.dcdf().filter_intra[av1_block_size_index(8, 8)], 0)
                } else {
                    0.0
                };
                let cost = rd_cost_i64(sse, mlam, bits + self.mode_bits(px, py, m) + filter_bits);
                let candidate = Luma8BeamCandidate {
                    luma_cost: cost,
                    mode: m,
                    pred,
                    cf,
                    tf,
                    sse,
                    bits,
                    palette: None,
                };
                let mut pos = JOINT_LUMA_BEAM;
                for (i, slot) in luma_beam.iter().enumerate() {
                    if slot.as_ref().is_none_or(|old| cost < old.luma_cost) {
                        pos = i;
                        break;
                    }
                }
                if pos < JOINT_LUMA_BEAM {
                    for i in (pos + 1..JOINT_LUMA_BEAM).rev() {
                        luma_beam[i] = luma_beam[i - 1].clone();
                    }
                    luma_beam[pos] = Some(candidate);
                }
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
        });
        outline_block8(|| {
            if rl.is_none()
                && self.try_palette()
                && let Some(hist) = block_color_histogram(&self.src[0], self.w, px, py, 8, 8)
            {
                for (palette, pred) in
                    self.rank_luma_palette_candidates::<64>(&hist, px, py, 8, 8, mlam)
                {
                    let mut resid = [0i32; 64];
                    self.rd
                        .residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
                    let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                    if self.speed.per_candidate_rdoq_av1() {
                        trellis_optimize_ctx(
                            &mut cf,
                            &tf,
                            dcq,
                            acq,
                            &SCAN_8X8,
                            lam,
                            8,
                            8,
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                            self.quant.qm_level(),
                            self.quant.qidx() as i32,
                        );
                    }
                    let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(
                        &self.rd,
                        &pred,
                        &rr,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    let coeff_bits = self.luma_bits(&cf, &SCAN_8X8, 8, px, py, DC_PRED, 1);
                    let bits = coeff_bits
                        + self.mode_bits(px, py, DC_PRED)
                        + self.palette_rate_bits(px, py, &palette);
                    let cost = rd_cost_i64(sse, mlam, bits);
                    let candidate = Luma8BeamCandidate {
                        luma_cost: cost,
                        mode: DC_PRED,
                        pred,
                        cf,
                        tf,
                        sse,
                        bits: coeff_bits,
                        palette: Some(palette.clone()),
                    };
                    let mut pos = JOINT_LUMA_BEAM;
                    for (i, slot) in luma_beam.iter().enumerate() {
                        if slot.as_ref().is_none_or(|old| cost < old.luma_cost) {
                            pos = i;
                            break;
                        }
                    }
                    if pos < JOINT_LUMA_BEAM {
                        for i in (pos + 1..JOINT_LUMA_BEAM).rev() {
                            luma_beam[i] = luma_beam[i - 1].clone();
                        }
                        luma_beam[pos] = Some(candidate);
                    }
                    if cost < best_eff {
                        best_eff = cost;
                        best_mode = DC_PRED;
                        best_filter_intra = None;
                        best_palette = Some(palette);
                        lpred_arr = pred;
                        lcf = cf;
                        ltf = tf;
                        best_dct_sse = sse;
                        best_dct_bits = coeff_bits;
                    }
                }
            }
        });
        // Join the two strongest ordinary/palette luma candidates with their
        // exact DC/CfL UV outcome before committing the luma predictor. This is
        // intentionally a two-entry beam: it captures luma/CfL coupling without
        // multiplying the full directional and transform searches.
        if rl.is_none() && !self.mono && self.speed == Speed::Slow && joint_luma_uv_enabled() {
            let mut joint_best = f32::INFINITY;
            let mut selected = None;
            for candidate in luma_beam.into_iter().flatten() {
                let cost = candidate.luma_cost
                    + self.joint_uv_cost8(
                        &candidate.pred,
                        &candidate.cf,
                        candidate.mode,
                        px,
                        py,
                        prdo,
                    );
                if cost < joint_best {
                    joint_best = cost;
                    selected = Some(candidate);
                }
            }
            if let Some(candidate) = selected {
                best_eff = candidate.luma_cost;
                best_mode = candidate.mode;
                lpred_arr = candidate.pred;
                lcf = candidate.cf;
                ltf = candidate.tf;
                best_dct_sse = candidate.sse;
                best_dct_bits = candidate.bits;
                best_filter_intra = None;
                best_palette = candidate.palette;
            }
        }
        outline_block8(|| {
            if rl.is_none() && self.speed == Speed::Slow {
                let bsize = av1_block_size_index(8, 8);
                for &filter_mode in self
                    .rank_filter_intra_modes::<64>(
                        px,
                        py,
                        8,
                        8,
                        self.speed.filter_intra_refine_budget(),
                    )
                    .iter()
                {
                    let mut pred = [0i32; 64];
                    self.intrapred.filter_predict(
                        filter_mode,
                        &self.recon[0],
                        self.w,
                        px,
                        py,
                        8,
                        8,
                        &mut pred,
                        self.bd,
                    );
                    let mut resid = [0i32; 64];
                    self.rd
                        .residual_pred(&mut resid, &pred, &self.src[0], self.w, px, py, 8, 8);
                    let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                    if self.speed.per_candidate_rdoq_av1() {
                        trellis_optimize_ctx(
                            &mut cf,
                            &tf,
                            dcq,
                            acq,
                            &SCAN_8X8,
                            lam,
                            8,
                            8,
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                            self.quant.qm_level(),
                            self.quant.qidx() as i32,
                        );
                    }
                    let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(
                        &self.rd,
                        &pred,
                        &rr,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    let bits = self.luma_bits(&cf, &SCAN_8X8, 8, px, py, DC_PRED, 1);
                    let syntax_bits = self.mode_bits(px, py, DC_PRED)
                        + cdf_cost(&self.dcdf().filter_intra[bsize], 1)
                        + cdf_cost(&self.dcdf().filter_intra_mode, filter_mode as usize);
                    let cost = rd_cost_i64(sse, mlam, bits + syntax_bits);
                    if rl.is_some()
                        || raw_sse_guard_choice(
                            "filter8",
                            RawSseGuard::FilterIntra,
                            best_dct_sse,
                            sse,
                            best_eff,
                            cost,
                            sse <= best_dct_sse && cost < best_eff,
                        )
                    {
                        best_eff = cost;
                        best_mode = DC_PRED;
                        lpred_arr = pred;
                        lcf = cf;
                        ltf = tf;
                        best_dct_sse = sse;
                        best_dct_bits = bits;
                        best_filter_intra = Some(filter_mode);
                        best_palette = None;
                    }
                }
            }
        });
        // Angle-delta winner refinement: if the winning luma mode is one of the
        // six pure diagonals, try angle_delta in -3..=3 (3 deg steps) and keep the
        // best by SSE + lambda*(coeff bits + angle_delta symbol bits). V/H and the
        // non-directional modes stay at delta 0. ~6 extra predictions per block.
        let mut best_delta: i32 = 0;
        outline_block8(|| {
            if rl.is_none()
                && angle_delta_enabled()
                && self.speed.try_angle_deltas_av1(8, self.base_q_idx)
                && (D45_PRED..=VERT_LEFT_PRED).contains(&best_mode)
                && best_mode != V_PRED
                && best_mode != H_PRED
            {
                let mut ad_cdf = [0u16; 7];
                ad_cdf.copy_from_slice(&self.dcdf().angle_delta[best_mode - V_PRED]);
                let mut best_ad_cost =
                    rd_cost_i64(best_dct_sse, mlam, best_dct_bits + cdf_cost(&ad_cdf, 3));
                let mut ad_pred0 = self.sbuf_i64();
                let mut ad_pred1 = self.sbuf_i64();
                let mut ad_scratch = self.sbuf_i64();
                let mut ad_preds = [&mut *ad_pred0, &mut *ad_pred1, &mut *ad_scratch];
                for (di, &d) in self
                    .rank_angle_deltas::<64>(
                        best_mode,
                        px,
                        py,
                        8,
                        8,
                        have_tr,
                        have_bl,
                        2,
                        &mut ad_preds,
                    )
                    .iter()
                    .enumerate()
                {
                    let pred: &[i32; 64] = &*ad_preds[di];
                    let mut resid = [0i32; 64];
                    self.rd
                        .residual_pred(&mut resid, pred, &self.src[0], self.w, px, py, 8, 8);
                    let (mut cf, tf) = self.dct.dct8x8_t(&resid, &self.quant);
                    if self.speed.per_candidate_rdoq_av1() {
                        trellis_optimize_ctx(
                            &mut cf,
                            &tf,
                            dcq,
                            acq,
                            &SCAN_8X8,
                            lam,
                            8,
                            8,
                            self.dcdf(),
                            1,
                            0,
                            &self.dcdf().eob_bin_64_l,
                            dc_sgn,
                            self.quant.qm_level(),
                            self.quant.qidx() as i32,
                        );
                    }
                    let rr = self.idct.idct_dequant_8x8(&cf, &self.quant);
                    let sse = sse_recon::<64, 8>(
                        &self.rd,
                        pred,
                        &rr,
                        &self.src[0],
                        self.w,
                        px,
                        py,
                        self.bd,
                    );
                    let bits = self.luma_bits(&cf, &SCAN_8X8, 8, px, py, best_mode, 1);
                    let cost = rd_cost_i64(sse, mlam, bits + cdf_cost(&ad_cdf, (d + 3) as usize));
                    if rl.is_some() || cost < best_ad_cost {
                        best_ad_cost = cost;
                        best_delta = d;
                        lpred_arr = *pred;
                        lcf = cf;
                        ltf = tf;
                        best_dct_sse = sse;
                        best_dct_bits = bits;
                    }
                }
            }
        });
        // Fast path: winner-only RDOQ (libaom winner-mode coeff opt).
        // NB never on 1-D winners: the trellis contexts are 2-D-class only.
        if rl.is_none() && !self.speed.per_candidate_rdoq_av1() && !best_is_vdct && !best_is_hdct {
            trellis_optimize_ctx(
                &mut lcf,
                &ltf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                8,
                self.dcdf(),
                1,
                0,
                &self.dcdf().eob_bin_64_l,
                dc_sgn,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
        }
        let mut best_txtp_sse = best_dct_sse;
        let mut best_txtp_bits = best_dct_bits;
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 64];
            self.rd
                .residual_pred(&mut resid, &lpred_arr, &self.src[0], self.w, px, py, 8, 8);
            let (mut acf, atf) = self.dct.adst8x8_t(&resid, &self.quant);
            trellis_optimize_ctx(
                &mut acf,
                &atf,
                dcq,
                acq,
                &SCAN_8X8,
                lam,
                8,
                8,
                self.dcdf(),
                1,
                0,
                &self.dcdf().eob_bin_64_l,
                dc_sgn,
                self.quant.qm_level(),
                self.quant.qidx() as i32,
            );
            let rr = self.idct.iadst_dequant_8x8(&acf, &self.quant);
            let asse = sse_recon::<64, 8>(
                &self.rd,
                &lpred_arr,
                &rr,
                &self.src[0],
                self.w,
                px,
                py,
                self.bd,
            );
            // Quality guard (see 16x16 ADST note): block low-q distortion-for-rate trades.
            let base_rd = rd_cost_i64(best_dct_sse, mlam, best_dct_bits);
            let bits_bound = if rl.is_some() {
                f32::INFINITY
            } else {
                (base_rd - asse as f32) / mlam
            };
            let abits = self.luma_bits_bounded(
                &acf,
                &SCAN_8X8,
                8,
                px,
                py,
                best_mode,
                ADST_ADST_TX8_IDX,
                bits_bound,
            );
            let candidate_rd = rd_cost_i64(asse, mlam, abits);
            if rl.is_some()
                || raw_sse_guard_choice(
                    "adst8",
                    RawSseGuard::TxType,
                    best_dct_sse,
                    asse,
                    base_rd,
                    candidate_rd,
                    asse <= best_dct_sse + (best_dct_sse >> 5) && candidate_rd < base_rd,
                )
            {
                lcf = acf;
                best_is_adst = true;
                best_txtp_sse = asse;
                best_txtp_bits = abits;
            }
        }
        // Per-block asymmetric-ADST refinement. Intra residual is anisotropic:
        // it grows away from the reference edge in one direction (wants ADST
        // there) and is flat across it (wants DCT). ADST_DCT = vertical ADST,
        // DCT_ADST = horizontal ADST. Each competes with the running tx winner.
        if rl.is_none() && self.speed.try_adst() && asym_adst_enabled() {
            for (fwd_t, inv_is_dctadst) in [(false, false), (true, true)] {
                let mut resid = [0i32; 64];
                self.rd
                    .residual_pred(&mut resid, &lpred_arr, &self.src[0], self.w, px, py, 8, 8);
                let (mut acf, atf) = if fwd_t {
                    self.dct.dctadst8x8_t(&resid, &self.quant)
                } else {
                    self.dct.adstdct8x8_t(&resid, &self.quant)
                };
                trellis_optimize_ctx(
                    &mut acf,
                    &atf,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    8,
                    self.dcdf(),
                    1,
                    0,
                    &self.dcdf().eob_bin_64_l,
                    dc_sgn,
                    self.quant.qm_level(),
                    self.quant.qidx() as i32,
                );
                let rr = if inv_is_dctadst {
                    self.idct.idctadst_dequant_8x8(&acf, &self.quant)
                } else {
                    self.idct.iadstdct_dequant_8x8(&acf, &self.quant)
                };
                let asse = sse_recon::<64, 8>(
                    &self.rd,
                    &lpred_arr,
                    &rr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    self.bd,
                );
                let base_rd = rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits);
                let bits_bound = if rl.is_some() {
                    f32::INFINITY
                } else {
                    (base_rd - asse as f32) / mlam
                };
                let abits = self.luma_bits_bounded(
                    &acf,
                    &SCAN_8X8,
                    8,
                    px,
                    py,
                    best_mode,
                    if inv_is_dctadst {
                        DCT_ADST_TX8_IDX
                    } else {
                        ADST_DCT_TX8_IDX
                    },
                    bits_bound,
                );
                let candidate_rd = rd_cost_i64(asse, mlam, abits);
                if rl.is_some()
                    || raw_sse_guard_choice(
                        "asym-adst8",
                        RawSseGuard::TxType,
                        best_txtp_sse,
                        asse,
                        base_rd,
                        candidate_rd,
                        asse <= best_dct_sse + (best_dct_sse >> 5) && candidate_rd < base_rd,
                    )
                {
                    lcf = acf;
                    best_is_adst = false;
                    best_is_idtx = false;
                    best_is_vdct = false;
                    best_is_hdct = false;
                    best_is_adstdct = !inv_is_dctadst;
                    best_is_dctadst = inv_is_dctadst;
                    best_txtp_sse = asse;
                    best_txtp_bits = abits;
                }
            }
        }
        if rl.is_none() && self.speed.try_adst() {
            let mut resid = [0i32; 64];
            self.rd
                .residual_pred(&mut resid, &lpred_arr, &self.src[0], self.w, px, py, 8, 8);
            let (icf, _itf) = self.dct.idtx8x8_t(&resid, &self.quant);
            let rr = self.idct.iidentity_dequant_8x8(&icf, &self.quant);
            let isse = sse_recon::<64, 8>(
                &self.rd,
                &lpred_arr,
                &rr,
                &self.src[0],
                self.w,
                px,
                py,
                self.bd,
            );
            let ibits = self.luma_bits(&icf, &SCAN_8X8, 8, px, py, best_mode, 0); // IDTX
            // Quality guard (see ADST note): identity spreads residual energy and
            // is cheap to code, so at low-q lambda a pure RD test over-selects it
            // and flattens detail. Require SSE-non-worsening vs the best real tx.
            let base_rd = rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits);
            let candidate_rd = rd_cost_i64(isse, mlam, ibits);
            if rl.is_some()
                || raw_sse_guard_choice(
                    "idtx8",
                    RawSseGuard::TxType,
                    best_txtp_sse,
                    isse,
                    base_rd,
                    candidate_rd,
                    isse <= best_txtp_sse + (best_txtp_sse >> 5) && candidate_rd < base_rd,
                )
            {
                lcf = icf;
                best_is_adst = false;
                best_is_idtx = true;
                best_is_vdct = false;
                best_is_hdct = false;
                best_is_adstdct = false;
                best_is_dctadst = false;
                best_txtp_sse = isse;
                best_txtp_bits = ibits;
            }
        }
        if rl.is_none() && self.speed.try_adst() && !self.ss420 {
            for vertical in [true, false] {
                let mut resid = [0i32; 64];
                self.rd
                    .residual_pred(&mut resid, &lpred_arr, &self.src[0], self.w, px, py, 8, 8);
                let (vcf, _vtf) = if vertical {
                    self.dct.fvdct8x8_t(&resid, &self.quant)
                } else {
                    self.dct.fhdct8x8_t(&resid, &self.quant)
                };
                let rr = if vertical {
                    self.idct.ivdct_dequant_8x8(&vcf, &self.quant)
                } else {
                    self.idct.ihdct_dequant_8x8(&vcf, &self.quant)
                };
                let vsse = sse_recon::<64, 8>(
                    &self.rd,
                    &lpred_arr,
                    &rr,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    self.bd,
                );
                let vbits = self.luma_bits_1d_8x8(&vcf, vertical, px, py, best_mode);
                let base_rd = rd_cost_i64(best_txtp_sse, mlam, best_txtp_bits);
                let candidate_rd = rd_cost_i64(vsse, mlam, vbits);
                if raw_sse_guard_choice(
                    if vertical { "vdct8" } else { "hdct8" },
                    RawSseGuard::TxType,
                    best_txtp_sse,
                    vsse,
                    base_rd,
                    candidate_rd,
                    // Strict SSE-non-worsening vs the running winner: the +3%
                    // tolerance used by the 2-D refinements measurably trades
                    // SSIMU2 for rate on screen content when a 1-D class wins.
                    vsse <= best_txtp_sse && candidate_rd < base_rd,
                ) {
                    lcf = vcf;
                    best_is_adst = false;
                    best_is_idtx = false;
                    best_is_adstdct = false;
                    best_is_dctadst = false;
                    best_is_vdct = vertical;
                    best_is_hdct = !vertical;
                    best_txtp_sse = vsse;
                    best_txtp_bits = vbits;
                }
            }
        }
        // tx_depth = 1 refinement: four TX_4X4 with per-TX prediction and
        // tx types (see `split4tx_try`) — the measured aom pattern on grainy
        // photos. Competes with the whole-TX winner on plain RD (+1 bit
        // tx_depth allowance).
        let mut best_is_txsplit4 = false;
        let mut s4_txtps = [1u8; 4];
        let mut s4_rec = [0u16; 64];
        if rl.is_none()
            && self.speed.try_adst()
            && best_palette.is_none()
            && best_filter_intra.is_none()
        {
            let (cf4, rec4, ssse, sbits, txtps) =
                self.split4tx_try(px, py, best_mode, best_delta, have_tr, have_bl, lam, mlam);
            let cand = rd_cost_i64(ssse, mlam, sbits + self.tx_depth_bits(px, py, 8, 8, 1));
            let cur = rd_cost_i64(
                best_txtp_sse,
                mlam,
                best_txtp_bits + self.tx_depth_bits(px, py, 8, 8, 0),
            );
            if cand < cur {
                lcf = cf4;
                best_is_txsplit4 = true;
                s4_txtps = txtps;
                s4_rec = rec4;
                best_is_adst = false;
                best_is_idtx = false;
                best_is_adstdct = false;
                best_is_dctadst = false;
                best_is_vdct = false;
                best_is_hdct = false;
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
            best_is_adst = r.tx == TxSel::Adst;
            best_is_idtx = r.tx == TxSel::Idtx;
            best_is_adstdct = r.tx == TxSel::AdstDct;
            best_is_dctadst = r.tx == TxSel::DctAdst;
            best_is_vdct = r.tx == TxSel::VDct;
            best_is_hdct = r.tx == TxSel::HDct;
            if let TxSel::Split4Tx(t) = r.tx {
                best_is_txsplit4 = true;
                s4_txtps = t;
            }
            best_palette = if r.palette == 0 {
                None
            } else {
                lossy_luma_palette(
                    &self.kmeans,
                    &self.src[0],
                    self.w,
                    px,
                    py,
                    8,
                    8,
                    r.palette as usize,
                )
            };
        }
        if let Some(cf) = rl_cf {
            lcf.copy_from_slice(&cf);
        }
        self.push_luma_sel(LumaSel {
            mode: best_mode as u8,
            delta: best_delta as i8,
            palette: best_palette
                .as_ref()
                .map_or(0, |p| (p.colors.len() + if p.top { 8 } else { 0 }) as u8),
            filter: best_filter_intra.map_or(NO_FILTER, |f| f as u8),
            tx: if best_is_txsplit4 {
                TxSel::Split4Tx(s4_txtps)
            } else if best_is_vdct {
                TxSel::VDct
            } else if best_is_hdct {
                TxSel::HDct
            } else {
                TxSel::from_flags(best_is_adst, best_is_idtx, best_is_adstdct, best_is_dctadst)
            },
        });
        self.push_luma_cf(&lcf);
        // Chroma winner (popped here, pushed at the end of the chroma searches;
        // exactly one per call in every format, mono included).
        let ru = self.uv_sel_replay();
        let ru_cf = self.uv_cf_replay();
        let mut ccf8 = [[0i32; 64]; 2];
        let mut ccf48 = [[0i32; 32]; 2];
        let mut ccf44 = [[0i32; 16]; 2];
        let mut cpred = [0i32; 2];
        let cy = py / 2; // chroma row for 4:2:0
        // Pure-emit replay skips this DC baseline for 4:4:4 (its `block_skip`
        // below reads the FINAL coeffs, installed from the record). 4:2:0 and
        // 4:2:2 must still run it: their `block_skip` is derived from these
        // baseline coeffs BEFORE the CfL/directional searches overwrite them,
        // so its zero-ness can differ from the captured winner's.
        let run_dc_baseline = ru.is_none() || self.ss420 || self.ss422;
        for ci in 0..(if self.mono || !run_dc_baseline { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let pred =
                    self.intrapred
                        .dc_pred_4x4(&self.recon[plane], self.cw, cx, cy, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 16];
                self.rd
                    .residual_dc(&mut resid, &self.src[plane], self.cw, cx, cy, 4, 4, pred);
                let (q, qt) = self.dct.dct4x4_t(&resid, &self.cquant);
                ccf44[ci] = q;
                trellis_optimize(&mut ccf44[ci], &qt, dcq, acq, &SCAN_4X4, lam);
            } else if self.ss422 {
                let pred =
                    self.intrapred
                        .dc_pred_4x8(&self.recon[plane], self.cw, cx, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 32];
                self.rd
                    .residual_dc(&mut resid, &self.src[plane], self.cw, cx, py, 4, 8, pred);
                let (q, qt) = self.dct.dct4x8_t(&resid, &self.cquant);
                ccf48[ci] = q;
                self.chroma_rect_trellis(
                    &mut ccf48[ci],
                    &qt,
                    dcq,
                    acq,
                    &SCAN_4X8,
                    lam,
                    4,
                    8,
                    plane,
                    cx,
                    py,
                );
            } else {
                let pred =
                    self.intrapred
                        .dc_pred_8x8(&self.recon[plane], self.w, px, py, self.bd as i32);
                cpred[ci] = pred;
                let mut resid = [0i32; 64];
                self.rd
                    .residual_dc(&mut resid, &self.src[plane], self.w, px, py, 8, 8, pred);
                let (q, qt) = self.dct.dct8x8_t(&resid, &self.cquant);
                ccf8[ci] = q;
                self.chroma_rect_trellis(
                    &mut ccf8[ci],
                    &qt,
                    dcq,
                    acq,
                    &SCAN_8X8,
                    lam,
                    8,
                    8,
                    plane,
                    px,
                    py,
                );
            }
        }

        // 4:4:4 Cfl: try predicting U/V from the reconstructed luma
        // block (scaled, mean-removed) and pick CfL over plain DC per block.
        let mut cpred444 = [[0i32; 64]; 2];
        let mut cpred420 = [[0i32; 16]; 2];
        let mut cpred422 = [[0i32; 32]; 2];
        let mut use_cfl = false;
        let mut cfl_alpha_uv = [0i32; 2];
        // Pure-emit replay never evaluates CfL; the captured winner installs
        // below (recon comes preinstalled from the record).
        outline_block8(|| {
            if self.speed.full_chroma_rdo()
                && !self.mono
                && !self.ss420
                && !self.ss422
                && ru.is_none()
            {
                // CfL luma reference must use the SAME inverse transform the decoder
                // will apply (the signaled luma tx-type), or the chroma CfL prediction
                // desyncs. Previously this was unconditionally idct, which diverged
                // whenever the luma block won with ADST or IDTX.
                let lrr_cfl = if best_is_vdct {
                    self.idct.ivdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_hdct {
                    self.idct.ihdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_idtx {
                    self.idct.iidentity_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adst {
                    self.idct.iadst_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adstdct {
                    self.idct.iadstdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_dctadst {
                    self.idct.idctadst_dequant_8x8(&lcf, &self.quant)
                } else {
                    self.idct.idct_dequant_8x8(&lcf, &self.quant)
                };
                let mut luma_rec = [0u16; 64];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr_cfl, (1 << self.bd) - 1);
                if best_is_txsplit4 {
                    luma_rec = s4_rec;
                }
                let mut ac = [0i32; 64];
                self.intrapred.cfl_ac_444(&luma_rec, 8, 8, &mut ac);
                let mut cfl_ccf = [[0i32; 64]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut dc_sse, mut dc_bits) = ([0i64; 2], [0f32; 2]);
                let (mut cfl_sse, mut cfl_bits) = ([0i64; 2], [0f32; 2]);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0u16; 64];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.w, px, py, 8, 8);
                    // DC option distortion/rate (from the coeffs already computed)
                    let dcrr = self.idct.idct_dequant_8x8(&ccf8[ci], &self.cquant);
                    dc_sse[ci] =
                        sse_recon::<64, 8>(&self.rd, &[dc; 64], &dcrr, &src, 8, 0, 0, self.bd);
                    dc_bits[ci] = self.chroma_bits(&ccf8[ci], &SCAN_8X8, 8, plane, px, py);
                    // CfL option
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &src, dc, 64, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 64];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..64], dc, a, self.bd);
                    let mut resid = [0i32; 64];
                    self.rd.residual_pred(&mut resid, &cpr, &src, 8, 0, 0, 8, 8);
                    let (mut q, qt) = self.dct.dct8x8_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, px, py,
                    );
                    let rr = self.idct.idct_dequant_8x8(&q, &self.cquant);
                    cfl_ccf[ci] = q;
                    cfl_sse[ci] = sse_recon::<64, 8>(&self.rd, &cpr, &rr, &src, 8, 0, 0, self.bd);
                    cfl_bits[ci] = self.chroma_bits(&q, &SCAN_8X8, 8, plane, px, py);
                    cpred444[ci] = cpr;
                }
                // joint signaling cost estimate (sign symbol + 1 magnitude per non-zero plane)
                let sig = self.uv_mode_bits(best_mode, CFL_PRED, Some(cfl_a));
                let dc_total = rd_cost_i64(
                    dc_sse[0] + dc_sse[1],
                    mlam,
                    dc_bits[0] + dc_bits[1] + self.uv_mode_bits(best_mode, DC_PRED, None),
                );
                let cfl_total = rd_cost_i64(
                    cfl_sse[0] + cfl_sse[1],
                    mlam,
                    cfl_bits[0] + cfl_bits[1] + sig,
                );
                // Let the RD comparison decide DC-vs-CfL across the whole quality
                // range; the old `ac_q() > 300` quality gate suppressed CfL exactly
                // where it helps most (high quality).
                if ru.is_some() || (cfl_total < dc_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf8[..2].copy_from_slice(&cfl_ccf[..2]);
                } else {
                    for ci in 0..2 {
                        cpred444[ci] = [cpred[ci]; 64];
                    }
                }
            }
        });
        // Pure-emit replay (4:4:4): install the captured chroma winner before
        // `chosen_uv_444` / `chroma_zero` read it. Prediction buffers stay
        // empty — recon is preinstalled from the record, never rewritten here.
        if !self.mono
            && !self.ss420
            && !self.ss422
            && let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            for (dst, src) in ccf8.iter_mut().zip(cf.iter()) {
                dst.copy_from_slice(src);
            }
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
        }

        // 4:4:4 directional chroma: PAETH_PRED and SMOOTH_PRED, both mapped to
        // ADST_ADST (the decoder derives the chroma tx-type from uv_mode, so
        // signaling either selects ADST_ADST automatically). These track
        // edges/gradients that plain DC smooths away — exactly the over-smoothing
        // the chroma path suffers from. Only considered when CfL did not win, and
        // chosen on a real RD margin over DC.
        let mut chosen_uv_444 = if use_cfl { CFL_PRED } else { DC_PRED };
        let mut paeth_pred444 = [[0i32; 64]; 2];
        let mut uv_pal8: Option<LossyUvPalette> = None;
        // NOTE: the 4:4:4 8x8 chroma path has a pre-existing reconstruction
        // divergence from the decoder at >8-bit (present in plain DC chroma,
        // independent of directional modes — luma and the 4:2:0/4:2:2 4x4/4x8
        // chroma paths are byte-exact at 10/12-bit). Directional modes propagate
        // that corrupted reconstruction, so restrict them to 8-bit here until the
        // baseline 4:4:4 high-bit-depth chroma issue is fixed. 4:2:0/4:2:2 below
        // are byte-exact at all bit depths and stay enabled.
        outline_block8(|| {
            if self.speed.full_chroma_rdo()
                && !self.mono
                && !self.ss420
                && !self.ss422
                && !use_cfl
                && ru.is_none()
            {
                // DC reference cost (current `ccf8`).
                let mut dc_total = 0f32;
                let mut src_planes = [[0u16; 64]; 2];
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut src = [0u16; 64];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.w, px, py, 8, 8);
                    src_planes[ci] = src;
                    let dcrr = self.idct.idct_dequant_8x8(&ccf8[ci], &self.cquant);
                    let sse = sse_recon::<64, 8>(
                        &self.rd,
                        &[cpred[ci]; 64],
                        &dcrr,
                        &src,
                        8,
                        0,
                        0,
                        self.bd,
                    );
                    dc_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_bits(&ccf8[ci], &SCAN_8X8, 8, plane, px, py),
                    );
                }
                // Try each directional candidate with its mode-derived transform;
                // keep the best that also beats DC by the mode-signaling margin.
                // V/H additionally emit a chroma angle_delta symbol (only valid here
                // at 8x8 4:4:4 chroma), costed below.
                let mut best_total = dc_total;
                let mut best_mode_uv = DC_PRED;
                let mut best_ccf = ccf8;
                let mut best_pred = [[0i32; 64]; 2];
                static CANDIDATES: [usize; 9] = [
                    PAETH_PRED,
                    SMOOTH_PRED,
                    SMOOTH_V_PRED,
                    SMOOTH_H_PRED,
                    V_PRED,
                    H_PRED,
                    D135_PRED,
                    D113_PRED,
                    D157_PRED,
                ];
                let directional_top = if ru.is_none() {
                    self.rank_chroma_modes::<64>(&CANDIDATES, px, py, px, py, 8, 8)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in CANDIDATES.iter() {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && !directional_top.contains(cand) {
                        continue;
                    }
                    let tx = chroma_tx_for_mode(cand);
                    // mode symbol (~4 bits) + angle_delta symbol (~3 bits) for the
                    // directional modes (V/H and the Z2 angulars D135/D113/D157).
                    let sig_bits = self.uv_mode_bits(best_mode, cand, None);
                    let mut cand_ccf = [[0i32; 64]; 2];
                    let mut cand_pred = [[0i32; 64]; 2];
                    let mut cand_total = rate_cost(mlam, sig_bits);
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let mut pp = [0i32; 64];
                        self.intrapred.predict_nd(
                            cand,
                            &self.recon[plane],
                            self.w,
                            px,
                            py,
                            8,
                            8,
                            false,
                            false,
                            self.w,
                            self.h,
                            self.chroma_filter_type(px, py),
                            &mut pp,
                            self.bd,
                        );
                        let mut resid = [0i32; 64];
                        self.rd
                            .residual_pred(&mut resid, &pp, &src_planes[ci], 8, 0, 0, 8, 8);
                        let (mut q, qt) = fwd_chroma_8x8(&self.dct, tx, &resid, &self.cquant);
                        self.chroma_rect_trellis(
                            &mut q, &qt, dcq, acq, &SCAN_8X8, lam, 8, 8, plane, px, py,
                        );
                        let rr = inv_chroma_8x8(&self.idct, tx, &q, &self.cquant);
                        let sse = sse_recon::<64, 8>(
                            &self.rd,
                            &pp,
                            &rr,
                            &src_planes[ci],
                            8,
                            0,
                            0,
                            self.bd,
                        );
                        cand_total += rd_cost_i64(
                            sse,
                            mlam,
                            self.chroma_bits(&q, &SCAN_8X8, 8, plane, px, py),
                        );
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
                        best_total = cand_total;
                        best_mode_uv = cand;
                        best_ccf = cand_ccf;
                        best_pred = cand_pred;
                    }
                }
                if best_mode_uv != DC_PRED {
                    chosen_uv_444 = best_mode_uv;
                    ccf8[..2].copy_from_slice(&best_ccf[..2]);
                    paeth_pred444[..2].copy_from_slice(&best_pred[..2]);
                }
                // UV palette candidate at 8x8 (2026-07-23: the size aom's UV
                // palettes actually ride — 64-px maps are cheap; exact first,
                // else lossy, both with RESIDUAL coefficients over the
                // palette prediction). 8-bit only, same as the directional
                // gate above (the 444-8 high-bd chroma baseline issue).
                if self.bd == 8 && self.try_palette() {
                    let exact = exact_uv_palette(&self.src[1], &self.src[2], self.w, px, py, 8, 8);
                    let pcands: Vec<LossyUvPalette> = if let Some(u) = exact {
                        vec![u]
                    } else {
                        [(8usize, false), (4, false), (8, true), (4, true)]
                            .iter()
                            .filter_map(|&(k, top)| {
                                lossy_uv_palette(
                                    &self.kmeans,
                                    &self.src[1],
                                    &self.src[2],
                                    self.w,
                                    px,
                                    py,
                                    8,
                                    8,
                                    k,
                                    top,
                                )
                            })
                            .collect()
                    };
                    for up in pcands {
                        let mut pal_pred = [[0i32; 64]; 2];
                        let [pred_u, pred_v] = &mut pal_pred;
                        palette_uv_pred(pred_u, pred_v, &up.map, &up.u, &up.v);
                        let mut bits = self.uv_mode_bits(best_mode, DC_PRED, None)
                            + self.palette_uv_rate_bits(false, &up);
                        let mut sse = 0i64;
                        let mut pal_ccf = [[0i32; 64]; 2];
                        for ci in 0..2 {
                            let plane = ci + 1;
                            let mut resid = [0i32; 64];
                            self.rd.residual_pred(
                                &mut resid,
                                &pal_pred[ci],
                                &self.src[plane],
                                self.w,
                                px,
                                py,
                                8,
                                8,
                            );
                            let (mut q, qt) = self.dct.dct8x8_t(&resid, &self.cquant);
                            trellis_optimize(
                                &mut q,
                                &qt,
                                self.cquant.dc_q() as f32,
                                self.cquant.ac_q() as f32,
                                &SCAN_8X8,
                                trellis_lambda(),
                            );
                            let rr = self.idct.idct_dequant_8x8(&q, &self.cquant);
                            sse += sse_recon::<64, 8>(
                                &self.rd,
                                &pal_pred[ci],
                                &rr,
                                &self.src[plane],
                                self.w,
                                px,
                                py,
                                self.bd,
                            );
                            pal_ccf[ci] = q;
                            bits += self.chroma_bits(&q, &SCAN_8X8, 8, plane, px, py);
                        }
                        let cand_total = rd_cost_i64(sse, mlam, bits);
                        if cand_total < best_total {
                            best_total = cand_total;
                            chosen_uv_444 = DC_PRED;
                            uv_pal8 = Some(up.clone());
                            ccf8[..2].copy_from_slice(&pal_ccf[..2]);
                            paeth_pred444[..2].copy_from_slice(&pal_pred[..2]);
                        }
                    }
                }
            }
        });
        // Pure-emit replay (4:4:4): a recorded directional winner sets the
        // signaled uv mode directly (coeffs were installed above).
        if !self.mono
            && !self.ss420
            && !self.ss422
            && !use_cfl
            && let Some(r) = ru
            && r.uv != DC_PRED as u8
        {
            chosen_uv_444 = r.uv as usize;
        }
        if !self.mono
            && !self.ss420
            && !self.ss422
            && let Some(r) = ru
            && r.palette > 0
        {
            uv_pal8 = Some(uv_palette_rederive(
                &self.kmeans,
                &self.src[1],
                &self.src[2],
                self.w,
                px,
                py,
                8,
                8,
                r.palette as usize,
            ));
        }

        let chroma_zero = |ci: usize| {
            if self.ss420 {
                self.rd.all_zero_i32(&ccf44[ci])
            } else if self.ss422 {
                self.rd.all_zero_i32(&ccf48[ci])
            } else {
                self.rd.all_zero_i32(&ccf8[ci])
            }
        };
        // Palette color indices are carried with the transform-token payload;
        // an intra skip block has no such payload, so palette blocks must code
        // `skip_txfm = 0` even when every quantized residual is zero.
        let block_skip = best_palette.is_none()
            && uv_pal8.is_none()
            && self.rd.all_zero_i32(&lcf)
            && (self.mono || (chroma_zero(0) && chroma_zero(1)));
        #[cfg(test)]
        if best_palette.is_some() && lcf.iter().any(|&c| c != 0) && !self.enc.sink {
            LOSSY_PALETTE_RESIDUAL_EMITTED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }

        // block-level mode info: skip (ctx = above_skip + left_skip), y/uv = DC
        let sctx = (self.a_skip[bx4] + self.l_skip[by4]) as usize;
        self.code_skip_and_sb_tokens(block_skip, sctx);
        self.mark_skip8(x8, y8, 1, block_skip);
        let yctx = INTRA_MODE_CTX[self.a_mode[bx4] as usize] * 5
            + INTRA_MODE_CTX[self.l_mode[by4] as usize];
        self.enc.encode_symbol(best_mode, &mut self.cdfs.kf_y[yctx]);
        if (V_PRED..=VERT_LEFT_PRED).contains(&best_mode) {
            // angle_delta refined for diagonals (above); V/H stay at delta 0.
            self.enc.encode_symbol(
                (best_delta + 3) as usize,
                &mut self.cdfs.angle_delta[best_mode - V_PRED],
            );
        }
        let smooth_v_active_ss420 = false;
        let mut sv_preds_420 = [[0i32; 16]; 2];
        let mut chosen_uv_block = DC_PRED;
        // 4:2:0 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x4). Populated by
        // the search block below (after CfL); recon uses `iadst_dequant_4x4`.
        let mut chosen_uv_420 = DC_PRED;
        let mut paeth_pred420 = [[0i32; 16]; 2];
        // 4:2:2 directional chroma (PAETH/SMOOTH -> ADST_ADST 4x8).
        let mut chosen_uv_422 = DC_PRED;
        let mut paeth_pred422 = [[0i32; 32]; 2];
        outline_block8(|| {
            if !self.mono && self.ss420 && smooth_v_active_ss420 {
                let (dcq2, acq2, lam2) = (
                    self.cquant.dc_q() as f32,
                    self.cquant.ac_q() as f32,
                    trellis_lambda(),
                );
                let mlam_c = self.mlam_c() * (self.emit_mlam(x8 * 8, y8 * 8, 8) / self.mlam());
                let sv_tx = chroma_tx_for_mode(SMOOTH_V_PRED);
                let mut sv_ccf44_2 = [[0i32; 16]; 2];
                // Real R-D on both legs. The old test compared raw PREDICTION error
                // (`src - pred`) and ignored rate entirely, so it took SMOOTH_V
                // whenever the predictor looked closer — even when the coded block
                // ended up bigger and worse. Score reconstructed distortion + rate.
                let mut dc_rd = 0f32;
                let mut sv_rd = 0f32;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let dcrr = self.idct.idct_dequant_4x4(&ccf44[ci], &self.cquant);
                    let sse_dc = self.rd.sse_recon(
                        &[dc; 16],
                        &dcrr,
                        &self.src[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                        self.bd,
                    );
                    dc_rd += rd_cost_i64(
                        sse_dc,
                        mlam_c,
                        self.chroma_bits(&ccf44[ci], &SCAN_4X4, 4, plane, cx, cy),
                    );

                    self.intrapred.predict_nd(
                        SMOOTH_V_PRED,
                        &self.recon[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                        false,
                        false,
                        self.cw,
                        self.h,
                        self.chroma_filter_type(px, py),
                        &mut sv_preds_420[ci],
                        self.bd,
                    );
                    let mut resid = [0i32; 16];
                    self.rd.residual_pred(
                        &mut resid,
                        &sv_preds_420[ci],
                        &self.src[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                    );
                    let (q, qt) = fwd_chroma_4x4(&self.dct, sv_tx, &resid, &self.cquant);
                    sv_ccf44_2[ci] = q;
                    trellis_optimize(&mut sv_ccf44_2[ci], &qt, dcq2, acq2, &SCAN_4X4, lam2);
                    let svrr = inv_chroma_4x4(&self.idct, sv_tx, &sv_ccf44_2[ci], &self.cquant);
                    let sse_sv = self.rd.sse_recon(
                        &sv_preds_420[ci],
                        &svrr,
                        &self.src[plane],
                        self.cw,
                        cx,
                        cy,
                        4,
                        4,
                        self.bd,
                    );
                    sv_rd += rd_cost_i64(
                        sse_sv,
                        mlam_c,
                        self.chroma_bits(&sv_ccf44_2[ci], &SCAN_4X4, 4, plane, cx, cy),
                    );
                }
                // SMOOTH_V also costs a non-DC uv_mode symbol.
                if sv_rd + rate_cost(mlam_c, smooth_v_uv_signal_bits()) < dc_rd {
                    ccf44[..2].copy_from_slice(&sv_ccf44_2[..2]);
                    chosen_uv_block = SMOOTH_V_PRED;
                }
            }
        });
        // Note: SMOOTH_V for 4:4:4 8x8 (code_block small-block path) is intentionally
        // not added here — it introduces too many DC↔SV mode transitions at 8-row
        // boundaries that are visible as faint lines at quality 50-75.
        // 4:2:0 chroma-from-luma: predict the 4x4 U/V from the 2x2-subsampled
        // reconstructed luma of this 8x8 block (dav1d cfl_ac, ss_hor=ss_ver=1).
        // Competes with the current DC/SMOOTH_V choice on rate-distortion.
        outline_block8(|| {
            if self.speed.full_chroma_rdo() && !self.mono && self.ss420 && ru.is_none() {
                let (dcq2, acq2, lam2) = (
                    self.cquant.dc_q() as f32,
                    self.cquant.ac_q() as f32,
                    trellis_lambda(),
                );
                let lrr = if best_is_vdct {
                    self.idct.ivdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_hdct {
                    self.idct.ihdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_idtx {
                    self.idct.iidentity_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adst {
                    self.idct.iadst_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adstdct {
                    self.idct.iadstdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_dctadst {
                    self.idct.idctadst_dequant_8x8(&lcf, &self.quant)
                } else {
                    self.idct.idct_dequant_8x8(&lcf, &self.quant)
                };
                let mut luma_rec = [0u16; 64];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr, (1 << self.bd) - 1);
                if best_is_txsplit4 {
                    luma_rec = s4_rec;
                }
                let mut ac = [0i32; 16];
                self.intrapred
                    .cfl_ac_sub(&luma_rec, 8, 4, 4, true, true, &mut ac);
                let mut cfl_ccf = [[0i32; 16]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
                let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0u16; 16];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, cx, cy, 4, 4);
                    let curr = if chosen_uv_block == SMOOTH_V_PRED {
                        inv_chroma_4x4(
                            &self.idct,
                            chroma_tx_for_mode(SMOOTH_V_PRED),
                            &ccf44[ci],
                            &self.cquant,
                        )
                    } else {
                        self.idct.idct_dequant_4x4(&ccf44[ci], &self.cquant)
                    };
                    let cur_pred = if chosen_uv_block == SMOOTH_V_PRED {
                        sv_preds_420[ci]
                    } else {
                        [dc; 16]
                    };
                    cur_sse +=
                        sse_recon::<16, 4>(&self.rd, &cur_pred, &curr, &src, 4, 0, 0, self.bd);
                    cur_bits += self.chroma_bits(&ccf44[ci], &SCAN_4X4, 4, plane, cx, cy);
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &src, dc, 16, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 16];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..16], dc, a, self.bd);
                    let mut resid = [0i32; 16];
                    self.rd.residual_pred(&mut resid, &cpr, &src, 4, 0, 0, 4, 4);
                    let (mut q, qt) = self.dct.dct4x4_t(&resid, &self.cquant);
                    trellis_optimize(&mut q, &qt, dcq2, acq2, &SCAN_4X4, lam2);
                    let rr = self.idct.idct_dequant_4x4(&q, &self.cquant);
                    cfl_sse += sse_recon::<16, 4>(&self.rd, &cpr, &rr, &src, 4, 0, 0, self.bd);
                    cfl_bits += self.chroma_bits(&q, &SCAN_4X4, 4, plane, cx, cy);
                    cfl_ccf[ci] = q;
                    cpred420[ci] = cpr;
                }
                let sig = self.uv_mode_bits(best_mode, CFL_PRED, Some(cfl_a));
                let cur_total = rd_cost_i64(
                    cur_sse,
                    mlam,
                    cur_bits + self.uv_mode_bits(best_mode, chosen_uv_block, None),
                );
                let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
                if ru.is_some() || (cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf44[..2].copy_from_slice(&cfl_ccf[..2]);
                }
            }
        });
        // 4:2:0 directional chroma: PAETH_PRED / SMOOTH_PRED (both -> ADST_ADST,
        // now available at 4x4). Same rationale and structure as the 4:4:4 path:
        // tracks chroma edges/gradients that plain DC over-smooths. Considered
        // only when CfL did not win; chosen on a real RD margin over DC.
        outline_block8(|| {
            if self.speed.full_chroma_rdo()
                && !self.mono
                && self.ss420
                && !use_cfl
                && self.cquant.ac_q() < 120
                && ru.is_none()
            {
                let mut src_planes = [[0u16; 16]; 2];
                let mut dc_total = 0f32;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut src = [0u16; 16];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, cx, cy, 4, 4);
                    src_planes[ci] = src;
                    let dcrr = self.idct.idct_dequant_4x4(&ccf44[ci], &self.cquant);
                    let sse = sse_recon::<16, 4>(
                        &self.rd,
                        &[cpred[ci]; 16],
                        &dcrr,
                        &src,
                        4,
                        0,
                        0,
                        self.bd,
                    );
                    dc_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_bits(&ccf44[ci], &SCAN_4X4, 4, plane, cx, cy),
                    );
                }
                let mut best_total = dc_total;
                let mut best_mode_uv = DC_PRED;
                let mut best_ccf = ccf44;
                let mut best_pred = [[0i32; 16]; 2];
                let candidates = &[
                    PAETH_PRED,
                    SMOOTH_PRED,
                    SMOOTH_V_PRED,
                    SMOOTH_H_PRED,
                    V_PRED,
                    H_PRED,
                    D135_PRED,
                    D113_PRED,
                    D157_PRED,
                ];
                let directional_top = if ru.is_none() {
                    self.rank_chroma_modes::<16>(candidates, px, py, cx, cy, 4, 4)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in candidates {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && !directional_top.contains(cand) {
                        continue;
                    }
                    let tx = chroma_tx_for_mode(cand);
                    let mut cand_ccf = [[0i32; 16]; 2];
                    let mut cand_pred = [[0i32; 16]; 2];
                    let sig_bits = self.uv_mode_bits(best_mode, cand, None);
                    let mut cand_total = rate_cost(mlam, sig_bits); // non-DC uv_mode (+angle_delta for V/H)
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let mut pp = [0i32; 16];
                        self.intrapred.predict_nd(
                            cand,
                            &self.recon[plane],
                            self.cw,
                            cx,
                            cy,
                            4,
                            4,
                            false,
                            false,
                            self.cw,
                            self.h,
                            self.chroma_filter_type(px, py),
                            &mut pp,
                            self.bd,
                        );
                        let mut resid = [0i32; 16];
                        self.rd
                            .residual_pred(&mut resid, &pp, &src_planes[ci], 4, 0, 0, 4, 4);
                        let (mut q, qt) = fwd_chroma_4x4(&self.dct, tx, &resid, &self.cquant);
                        trellis_optimize(&mut q, &qt, dcq, acq, &SCAN_4X4, lam);
                        let rr = inv_chroma_4x4(&self.idct, tx, &q, &self.cquant);
                        let sse = sse_recon::<16, 4>(
                            &self.rd,
                            &pp,
                            &rr,
                            &src_planes[ci],
                            4,
                            0,
                            0,
                            self.bd,
                        );
                        cand_total += rd_cost_i64(
                            sse,
                            mlam,
                            self.chroma_bits(&q, &SCAN_4X4, 4, plane, cx, cy),
                        );
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
                        best_total = cand_total;
                        best_mode_uv = cand;
                        best_ccf = cand_ccf;
                        best_pred = cand_pred;
                    }
                }
                if best_mode_uv != DC_PRED {
                    chosen_uv_420 = best_mode_uv;
                    ccf44[..2].copy_from_slice(&best_ccf[..2]);
                    paeth_pred420[..2].copy_from_slice(&best_pred[..2]);
                }
            }
        });
        // reconstructed luma (dav1d cfl_ac, ss_hor=1, ss_ver=0).
        outline_block8(|| {
            if self.speed.full_chroma_rdo() && !self.mono && self.ss422 && ru.is_none() {
                let (dcq2, acq2, lam2) = (
                    self.cquant.dc_q() as f32,
                    self.cquant.ac_q() as f32,
                    trellis_lambda(),
                );
                let lrr = if best_is_vdct {
                    self.idct.ivdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_hdct {
                    self.idct.ihdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_idtx {
                    self.idct.iidentity_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adst {
                    self.idct.iadst_dequant_8x8(&lcf, &self.quant)
                } else if best_is_adstdct {
                    self.idct.iadstdct_dequant_8x8(&lcf, &self.quant)
                } else if best_is_dctadst {
                    self.idct.idctadst_dequant_8x8(&lcf, &self.quant)
                } else {
                    self.idct.idct_dequant_8x8(&lcf, &self.quant)
                };
                let mut luma_rec = [0u16; 64];
                recon_add_pred(&mut luma_rec, &lpred_arr, &lrr, (1 << self.bd) - 1);
                if best_is_txsplit4 {
                    luma_rec = s4_rec;
                }
                let mut ac = [0i32; 32];
                self.intrapred
                    .cfl_ac_sub(&luma_rec, 8, 4, 8, true, false, &mut ac);
                let mut cfl_ccf = [[0i32; 32]; 2];
                let mut cfl_a = [0i32; 2];
                let (mut cur_sse, mut cfl_sse) = (0i64, 0i64);
                let (mut cur_bits, mut cfl_bits) = (0f32, 0f32);
                for ci in 0..2 {
                    let plane = ci + 1;
                    let dc = cpred[ci];
                    let mut src = [0u16; 32];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, cx, py, 4, 8);
                    let curr = self.idct.idct_dequant_4x8(&ccf48[ci], &self.cquant);
                    cur_sse += self
                        .rd
                        .sse_recon(&[dc; 32], &curr, &src, 4, 0, 0, 4, 8, self.bd);
                    cur_bits += self.chroma_rect_bits(&ccf48[ci], &SCAN_4X8, 4, 8, plane, cx, py);
                    let a = self
                        .intrapred
                        .cfl_best_alpha(&ac, &src, dc, 32, self.bd);
                    cfl_a[ci] = a;
                    let mut cpr = [0i32; 32];
                    self.intrapred.cfl_pred(&mut cpr, &ac[..32], dc, a, self.bd);
                    let mut resid = [0i32; 32];
                    self.rd.residual_pred(&mut resid, &cpr, &src, 4, 0, 0, 4, 8);
                    let (mut q, qt) = self.dct.dct4x8_t(&resid, &self.cquant);
                    self.chroma_rect_trellis(
                        &mut q, &qt, dcq2, acq2, &SCAN_4X8, lam2, 4, 8, plane, cx, py,
                    );
                    let rr = self.idct.idct_dequant_4x8(&q, &self.cquant);
                    cfl_sse += self.rd.sse_recon(&cpr, &rr, &src, 4, 0, 0, 4, 8, self.bd);
                    cfl_bits += self.chroma_rect_bits(&q, &SCAN_4X8, 4, 8, plane, cx, py);
                    cfl_ccf[ci] = q;
                    cpred422[ci] = cpr;
                }
                let sig = self.uv_mode_bits(best_mode, CFL_PRED, Some(cfl_a));
                let cur_total = rd_cost_i64(
                    cur_sse,
                    mlam,
                    cur_bits + self.uv_mode_bits(best_mode, chosen_uv_block, None),
                );
                let cfl_total = rd_cost_i64(cfl_sse, mlam, cfl_bits + sig);
                if ru.is_some() || (cfl_total < cur_total && (cfl_a[0] != 0 || cfl_a[1] != 0)) {
                    use_cfl = true;
                    cfl_alpha_uv = cfl_a;
                    ccf48[..2].copy_from_slice(&cfl_ccf[..2]);
                }
            }
        });
        // 4:2:2 directional chroma: PAETH_PRED / SMOOTH_PRED (-> ADST_ADST 4x8).
        // Same rationale/structure as the 4:2:0 path; block is 4 wide x 8 tall at
        // chroma coords (cx, py). Gated to higher quality (chroma ac_q < 120) and
        // only when CfL did not win; chosen on a real RD margin over DC.
        outline_block8(|| {
            if self.speed.full_chroma_rdo()
                && !self.mono
                && self.ss422
                && !use_cfl
                && self.cquant.ac_q() < 120
                && ru.is_none()
            {
                let mut src_planes = [[0u16; 32]; 2];
                let mut dc_total = 0f32;
                for ci in 0..2 {
                    let plane = ci + 1;
                    let mut src = [0u16; 32];
                    self.rd
                        .copy_block_u16(&mut src, &self.src[plane], self.cw, cx, py, 4, 8);
                    src_planes[ci] = src;
                    let dcrr = self.idct.idct_dequant_4x8(&ccf48[ci], &self.cquant);
                    let sse =
                        self.rd
                            .sse_recon(&[cpred[ci]; 32], &dcrr, &src, 4, 0, 0, 4, 8, self.bd);
                    dc_total += rd_cost_i64(
                        sse,
                        mlam,
                        self.chroma_rect_bits(&ccf48[ci], &SCAN_4X8, 4, 8, plane, cx, py),
                    );
                }
                let mut best_total = dc_total;
                let mut best_mode_uv = DC_PRED;
                let mut best_ccf = ccf48;
                let mut best_pred = [[0i32; 32]; 2];
                let candidates = &[
                    PAETH_PRED,
                    SMOOTH_PRED,
                    SMOOTH_V_PRED,
                    SMOOTH_H_PRED,
                    V_PRED,
                    H_PRED,
                    D135_PRED,
                    D113_PRED,
                    D157_PRED,
                ];
                let directional_top = if ru.is_none() {
                    self.rank_chroma_modes::<32>(candidates, px, py, cx, py, 4, 8)
                } else {
                    DirectionalTopK::new()
                };
                for &cand in candidates {
                    // V/H are cheap enough for every tier; Fast skips diagonal angles.
                    if ru.is_some_and(|r| cand as u8 != r.uv) {
                        continue;
                    }
                    if ru.is_none()
                        && cand != V_PRED
                        && cand != H_PRED
                        && (V_PRED..=VERT_LEFT_PRED).contains(&cand)
                        && !self.speed.chroma_angle_directional()
                    {
                        continue;
                    }
                    if ru.is_none() && !directional_top.contains(cand) {
                        continue;
                    }
                    let tx = chroma_tx_for_mode(cand);
                    let mut cand_ccf = [[0i32; 32]; 2];
                    let mut cand_pred = [[0i32; 32]; 2];
                    let sig_bits = self.uv_mode_bits(best_mode, cand, None);
                    let mut cand_total = rate_cost(mlam, sig_bits);
                    for ci in 0..2 {
                        let plane = ci + 1;
                        let mut pp = [0i32; 32];
                        self.intrapred.predict_nd(
                            cand,
                            &self.recon[plane],
                            self.cw,
                            cx,
                            py,
                            4,
                            8,
                            false,
                            false,
                            self.cw,
                            self.h,
                            self.chroma_filter_type(px, py),
                            &mut pp,
                            self.bd,
                        );
                        let mut resid = [0i32; 32];
                        self.rd
                            .residual_pred(&mut resid, &pp, &src_planes[ci], 4, 0, 0, 4, 8);
                        let (mut q, qt) = fwd_chroma_4x8(&self.dct, tx, &resid, &self.cquant);
                        self.chroma_rect_trellis(
                            &mut q, &qt, dcq, acq, &SCAN_4X8, lam, 4, 8, plane, cx, py,
                        );
                        let rr = inv_chroma_4x8(&self.idct, tx, &q, &self.cquant);
                        let sse =
                            self.rd
                                .sse_recon(&pp, &rr, &src_planes[ci], 4, 0, 0, 4, 8, self.bd);
                        cand_total += rd_cost_i64(
                            sse,
                            mlam,
                            self.chroma_rect_bits(&q, &SCAN_4X8, 4, 8, plane, cx, py),
                        );
                        cand_ccf[ci] = q;
                        cand_pred[ci] = pp;
                    }
                    if ru.is_some() || cand_total < best_total {
                        best_total = cand_total;
                        best_mode_uv = cand;
                        best_ccf = cand_ccf;
                        best_pred = cand_pred;
                    }
                }
                if best_mode_uv != DC_PRED {
                    chosen_uv_422 = best_mode_uv;
                    ccf48[..2].copy_from_slice(&best_ccf[..2]);
                    paeth_pred422[..2].copy_from_slice(&best_pred[..2]);
                }
            }
        });
        // Pure-emit replay (4:2:0/4:2:2): install the captured chroma winner.
        // This lands AFTER `block_skip` was derived from the DC baseline above,
        // matching the Off ordering where the searches overwrite ccf44/ccf48
        // only after the skip flag is already coded.
        if !self.mono
            && (self.ss420 || self.ss422)
            && let Some(r) = ru
            && let Some((cf, al)) = ru_cf.as_ref()
        {
            if self.ss420 {
                for (dst, src) in ccf44.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            } else {
                for (dst, src) in ccf48.iter_mut().zip(cf.iter()) {
                    dst.copy_from_slice(src);
                }
            }
            use_cfl = r.uv == CFL_PRED as u8;
            cfl_alpha_uv = *al;
            if !use_cfl && r.uv != DC_PRED as u8 {
                if self.ss420 {
                    chosen_uv_420 = r.uv as usize;
                } else {
                    chosen_uv_422 = r.uv as usize;
                }
            }
        }
        // Capture the final chroma winner (CfL folded in as CFL_PRED; mono
        // pushes a DC dummy so the record cursor stays aligned per call).
        // NB the dead SMOOTH_V path (`chosen_uv_block`, gated off) would need
        // its own record entry if ever re-enabled — its coeffs come from a
        // different evaluation than the directional loop's.
        {
            let uv_final = if self.mono {
                DC_PRED
            } else if use_cfl {
                CFL_PRED
            } else if !self.ss420 && !self.ss422 {
                chosen_uv_444
            } else if self.ss420 {
                if chosen_uv_420 != DC_PRED {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                }
            } else if chosen_uv_422 != DC_PRED {
                chosen_uv_422
            } else {
                chosen_uv_block
            };
            self.push_uv_sel(UvSel {
                uv: uv_final as u8,
                palette: uv_pal8
                    .as_ref()
                    .map_or(0, |p| (p.u.len() + if p.top { 8 } else { 0 }) as u8),
            });
            let cfl_rec = if use_cfl { cfl_alpha_uv } else { [0, 0] };
            if self.mono {
                self.push_uv_cf(&[], &[], [0, 0]);
            } else if self.ss420 {
                self.push_uv_cf(&ccf44[0], &ccf44[1], cfl_rec);
            } else if self.ss422 {
                self.push_uv_cf(&ccf48[0], &ccf48[1], cfl_rec);
            } else {
                self.push_uv_cf(&ccf8[0], &ccf8[1], cfl_rec);
            }
        }
        if !self.mono {
            // 4:4:4 uses the directional (PAETH) decision; 4:2:0/4:2:2 use their
            // own block-mode choice. CfL overrides via the alpha argument.
            let uv_mode_sym = if !self.ss420 && !self.ss422 {
                chosen_uv_444
            } else if self.ss420 {
                // 4:2:0: directional (PAETH/SMOOTH via ADST4) overrides DC; the
                // legacy SMOOTH_V path stays gated off (chosen_uv_block == DC).
                if chosen_uv_420 != DC_PRED {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                }
            } else if self.ss422 {
                if chosen_uv_422 != DC_PRED {
                    chosen_uv_422
                } else {
                    chosen_uv_block
                }
            } else {
                chosen_uv_block
            };
            self.emit_uv_mode(
                best_mode,
                uv_mode_sym,
                if use_cfl { Some(cfl_alpha_uv) } else { None },
                px,
                py,
                8,
                8,
            );
        }
        self.emit_palette_mode_info(
            px,
            py,
            8,
            8,
            best_mode,
            !self.mono,
            best_palette.as_ref(),
            uv_pal8.as_ref(),
        );
        if best_palette.is_none() {
            self.emit_filter_intra(best_mode, 8, 8, best_filter_intra);
        }
        if let Some(palette) = best_palette.as_ref() {
            self.emit_palette_map(palette);
        }
        if let Some(up) = uv_pal8.as_ref() {
            self.emit_palette_uv_map(up);
        }
        self.code_tx_depth(px, py, 8, 8, best_is_txsplit4 as usize);
        if best_is_txsplit4 {
            // Deblock runs on TRANSFORM edges: mark the four 4x4 TX cells
            // (same pattern as the SPLIT4 partition and 16/32 TX splits).
            let nc4 = self.w / 4;
            for uy in 0..2 {
                for ux in 0..2 {
                    let cell = (by4 + uy) * nc4 + (bx4 + ux);
                    self.blk4[cell] = 1;
                    self.blk4h[cell] = 1;
                    self.blk4v[cell] = true;
                    self.blk4t[cell] = true;
                }
            }
        }
        let sv = block_skip as u8;
        self.a_skip[bx4] = sv;
        self.a_skip[bx4 + 1] = sv;
        self.l_skip[by4] = sv;
        self.l_skip[by4 + 1] = sv;
        let mv = best_mode as u8;
        self.a_mode[bx4] = mv;
        self.a_mode[bx4 + 1] = mv;
        self.l_mode[by4] = mv;
        self.l_mode[by4 + 1] = mv;

        // luma (TX_8X8, or four TX_4X4 when the depth-1 refinement won)
        let lres_ctx = if block_skip {
            0x40
        } else if best_is_txsplit4 {
            // Raster-order sub-TXs with progressive coefficient contexts
            // (each cell's a/l ctx byte written as it codes, like dav1d).
            let mut last = 0u8;
            for (qi, &(sx, sy)) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)]
                .iter()
                .enumerate()
            {
                let (qbx4, qby4) = ((px + sx) / 4, (py + sy) / 4);
                let mut cfq = [0i32; 16];
                cfq.copy_from_slice(&lcf[qi * 16..qi * 16 + 16]);
                let sk = self.skip_ctx_split(qbx4, qby4, 1, 1);
                let ds = self.dc_sign_ctx_420(0, qbx4, qby4);
                let res = if s4_txtps[qi] == 2 || s4_txtps[qi] == 3 {
                    encode_tx4_coeffs_1d(
                        &mut self.enc,
                        &mut self.cdfs,
                        &cfq,
                        s4_txtps[qi] == 2,
                        sk,
                        ds,
                        best_mode,
                    )
                } else {
                    encode_tx4_luma_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &cfq,
                        sk,
                        ds,
                        best_mode,
                        s4_txtps[qi] as usize,
                    )
                };
                self.a_coef[0][qbx4] = res;
                self.l_coef[0][qby4] = res;
                last = res;
            }
            last
        } else {
            let sk = self.skip_ctx(0, bx4, by4, false);
            let ds = self.dc_sign_ctx(0, bx4, by4);
            if best_is_vdct || best_is_hdct {
                encode_tx8_coeffs_1d(
                    &mut self.enc,
                    &mut self.cdfs,
                    &lcf,
                    best_is_vdct,
                    sk,
                    ds,
                    filter_intra_tx_mode(best_filter_intra, best_mode),
                )
            } else {
                encode_tx8_coeffs_adapt(
                    &mut self.enc,
                    &mut self.cdfs,
                    &lcf,
                    false,
                    sk,
                    ds,
                    filter_intra_tx_mode(best_filter_intra, best_mode),
                    if best_is_idtx {
                        0
                    } else if best_is_adst {
                        ADST_ADST_TX8_IDX
                    } else if best_is_adstdct {
                        ADST_DCT_TX8_IDX
                    } else if best_is_dctadst {
                        DCT_ADST_TX8_IDX
                    } else {
                        1
                    },
                )
            }
        };
        if !best_is_txsplit4 || block_skip {
            self.a_coef[0][bx4] = lres_ctx;
            self.a_coef[0][bx4 + 1] = lres_ctx;
            self.l_coef[0][by4] = lres_ctx;
            self.l_coef[0][by4 + 1] = lres_ctx;
        }
        // Pure-emit replay: recon is preinstalled from the record; the writes
        // below would need the prediction we no longer compute.
        if self.sb_mode != SbMode::Replay && best_is_txsplit4 {
            if !block_skip {
                for ry in 0..8 {
                    self.recon[0][(py + ry) * self.w + px..][..8]
                        .copy_from_slice(&s4_rec[ry * 8..ry * 8 + 8]);
                }
            } else {
                // Skipped split block: the decoder predicts per TX with ZERO
                // residual — recompute the sequential prediction (the trial
                // recon carries residual feedback the decoder never sees).
                let block_ftype = self.luma_filter_type(px, py);
                for &(sx, sy) in [(0usize, 0usize), (4, 0), (0, 4), (4, 4)].iter() {
                    let (bx, by) = (px + sx, py + sy);
                    let (tr, bl) = match (sx, sy) {
                        (0, 0) => (py > 0, px > 0),
                        (4, 0) => (have_tr, false),
                        (0, 4) => (true, have_bl),
                        _ => (false, false),
                    };
                    let mut pred = [0i32; 16];
                    if best_mode == DC_PRED {
                        let d = self.intrapred.dc_pred_4x4(
                            &self.recon[0],
                            self.w,
                            bx,
                            by,
                            self.bd as i32,
                        );
                        pred = [d; 16];
                    } else {
                        self.intrapred.predict_nd_ad(
                            best_mode,
                            best_delta,
                            &self.recon[0],
                            self.w,
                            bx,
                            by,
                            4,
                            4,
                            tr,
                            bl,
                            self.w,
                            self.h,
                            block_ftype,
                            &mut pred,
                            self.bd,
                        );
                    }
                    self.rd.reconstruct(
                        &mut self.recon[0][by * self.w + bx..],
                        self.w,
                        None,
                        &pred,
                        &[],
                        4,
                        4,
                        self.bd,
                    );
                }
            }
        } else if self.sb_mode != SbMode::Replay {
            let lrr = if block_skip {
                [0i32; 64]
            } else if best_is_vdct {
                self.idct.ivdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_hdct {
                self.idct.ihdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_idtx {
                self.idct.iidentity_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adst {
                self.idct.iadst_dequant_8x8(&lcf, &self.quant)
            } else if best_is_adstdct {
                self.idct.iadstdct_dequant_8x8(&lcf, &self.quant)
            } else if best_is_dctadst {
                self.idct.idctadst_dequant_8x8(&lcf, &self.quant)
            } else {
                self.idct.idct_dequant_8x8(&lcf, &self.quant)
            };
            for (ry, (prow, rrow)) in lpred_arr
                .as_chunks::<8>()
                .0
                .iter()
                .zip(lrr.as_chunks::<8>().0.iter())
                .enumerate()
            {
                let drow = &mut self.recon[0][(py + ry) * self.w + px..];
                recon_add_pred(drow, prow, rrow, (1 << self.bd) - 1);
            }
        }

        for ci in 0..(if self.mono { 0 } else { 2 }) {
            let plane = ci + 1;
            if self.ss420 {
                let (bx4c, by4c) = (cx / 4, cy / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_420(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_420(plane, bx4c, by4c);
                    encode_4x4_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf44[ci], sk, ds)
                };
                // TX_4X4: 1 coef-context unit wide and tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth420 = chosen_uv_420 != DC_PRED;
                // Derive the inverse from the uv_mode that is ACTUALLY SIGNALLED,
                // in the same precedence the emitter uses. Selecting on
                // `chosen_uv_block` alone desyncs whenever CfL subsequently wins:
                // the stream then says CFL_PRED (decoder derives DCT_DCT) while
                // the encoder reconstructed with ADST_DCT. That decodes without
                // error but collapses quality, and CfL wins more often at high
                // quality — which is why this only showed above q60.
                let uv_eff = if use_cfl {
                    CFL_PRED
                } else if paeth420 {
                    chosen_uv_420
                } else {
                    chosen_uv_block
                };
                let rr = if block_skip {
                    [0i32; 16]
                } else {
                    inv_chroma_4x4(
                        &self.idct,
                        chroma_tx_for_mode(uv_eff),
                        &ccf44[ci],
                        &self.cquant,
                    )
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(cy + ry) * self.cw + cx..];
                    // Precedence MUST match `uv_final` / `uv_eff` above:
                    // CfL > directional(chosen_uv_420) > SMOOTH_V > DC. Ranking
                    // SMOOTH_V ahead of the directional mode reconstructs with a
                    // different predictor than the one signalled whenever both
                    // fire — silent at low quality, but the directional chroma
                    // search fires far more above q60, which is where this
                    // showed up as a quality collapse at unchanged size.
                    if use_cfl {
                        recon_add_pred(&mut drow[..4], &cpred420[ci][ry * 4..], rrow, max);
                    } else if paeth420 {
                        recon_add_pred(&mut drow[..4], &paeth_pred420[ci][ry * 4..], rrow, max);
                    } else if chosen_uv_block == SMOOTH_V_PRED {
                        recon_add_pred(&mut drow[..4], &sv_preds_420[ci][ry * 4..], rrow, max);
                    } else {
                        recon_add_dc(drow, cpred[ci], rrow, max);
                    }
                }
            } else if self.ss422 {
                let (bx4c, by4c) = (cx / 4, py / 4);
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx_422(plane, bx4c, by4c);
                    let ds = self.dc_sign_ctx_422(plane, bx4c, by4c);
                    encode_4x8_chroma_coeffs(&mut self.enc, &mut self.cdfs, &ccf48[ci], sk, ds)
                };
                // RTX_4X8: 1 coef-context unit wide, 2 units tall
                self.a_coef[plane][bx4c] = res_ctx;
                self.l_coef[plane][by4c] = res_ctx;
                self.l_coef[plane][by4c + 1] = res_ctx;
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth422 = chosen_uv_422 != DC_PRED;
                let rr = if block_skip {
                    [0i32; 32]
                } else if paeth422 {
                    inv_chroma_4x8(
                        &self.idct,
                        chroma_tx_for_mode(chosen_uv_422),
                        &ccf48[ci],
                        &self.cquant,
                    )
                } else {
                    self.idct.idct_dequant_4x8(&ccf48[ci], &self.cquant)
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<4>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.cw + cx..];
                    if use_cfl {
                        recon_add_pred(drow, &cpred422[ci][ry * 4..], rrow, max);
                    } else if paeth422 {
                        recon_add_pred(drow, &paeth_pred422[ci][ry * 4..], rrow, max);
                    } else {
                        recon_add_dc(drow, cpred[ci], rrow, max);
                    }
                }
            } else {
                let res_ctx = if block_skip {
                    0x40
                } else {
                    let sk = self.skip_ctx(plane, bx4, by4, true);
                    let ds = self.dc_sign_ctx(plane, bx4, by4);
                    encode_tx8_coeffs_adapt(
                        &mut self.enc,
                        &mut self.cdfs,
                        &ccf8[ci],
                        true,
                        sk,
                        ds,
                        0,
                        1,
                    )
                };
                self.a_coef[plane][bx4] = res_ctx;
                self.a_coef[plane][bx4 + 1] = res_ctx;
                self.l_coef[plane][by4] = res_ctx;
                self.l_coef[plane][by4 + 1] = res_ctx;
                if self.sb_mode == SbMode::Replay {
                    continue; // recon preinstalled
                }
                let paeth =
                    (chosen_uv_444 != DC_PRED && chosen_uv_444 != CFL_PRED) || uv_pal8.is_some();
                let rr = if block_skip {
                    [0i32; 64]
                } else if paeth {
                    // Directional chroma: tx derived from uv_mode (Mode_To_Txfm).
                    inv_chroma_8x8(
                        &self.idct,
                        chroma_tx_for_mode(chosen_uv_444),
                        &ccf8[ci],
                        &self.cquant,
                    )
                } else {
                    self.idct.idct_dequant_8x8(&ccf8[ci], &self.cquant)
                };
                let max = (1 << self.bd) - 1;
                for (ry, rrow) in rr.as_chunks::<8>().0.iter().enumerate() {
                    let drow = &mut self.recon[plane][(py + ry) * self.w + px..];
                    if use_cfl {
                        recon_add_pred(drow, &cpred444[ci][ry * 8..], rrow, max);
                    } else if paeth {
                        recon_add_pred(drow, &paeth_pred444[ci][ry * 8..], rrow, max);
                    } else {
                        // Plain DC chroma: use the scalar predictor directly so the
                        // reconstruction never depends on the CfL evaluation block
                        // having populated `cpred444`.
                        recon_add_dc(drow, cpred[ci], rrow, max);
                    }
                }
            }
        }
    }
}
