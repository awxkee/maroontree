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

use super::*;

impl Av2Encoder {
    /// Native 4:0:0 (monochrome) edge-partition walk. Luma-only — the 4:4:4 luma walk
    /// with all chroma coding stripped and `has_chroma = false`. Because there is no
    /// chroma, every luma leaf geometry the encoders support is available with no
    /// corner restrictions, so this covers the full `lossy_native_mi` residue set.
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv400_partition(
        &self,
        enc: &mut RangeEncoder,
        luma: LumaPlanes,
        ctx: &PartitionPass,
        nb: PartitionNeighbors,
    ) {
        let LumaPlanes { rec: recy, src: yp } = luma;
        let &PartitionPass {
            luma_stride: pw,
            width,
            height,
            sb_rows,
            sb_cols,
            tmc,
            tmr,
            quant:
                QuantCtx {
                    qc,
                    neutral,
                    qstep: qstep_i,
                },
            ..
        } = ctx;
        let PartitionNeighbors {
            above,
            left,
            above_pctx,
            left_pctx,
        } = nb;
        let bases = &self.bases;
        for row in 0..sb_rows {
            left_pctx.iter_mut().for_each(|p| *p = 0);
            for col in 0..sb_cols {
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    tmr as usize,
                    tmc as usize,
                    above_pctx,
                    left_pctx,
                );
                for op in &ops {
                    let (bw_mi, bh_mi, pc, _lmr, _lmc) = match op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(*cdf, *val);
                            continue;
                        }
                        partition::Op::Leaf {
                            bw_mi,
                            bh_mi,
                            part_cdf,
                            mi_row,
                            mi_col,
                        } => (*bw_mi, *bh_mi, part_cdf.unwrap_or(12276), *mi_row, *mi_col),
                    };
                    let sb_y = _lmr * 4;
                    let sb_x = _lmc * 4;
                    match (bw_mi, bh_mi) {
                        (16, 16) => {
                            let (tus, mode_idx) = encode_luma_sb(
                                recy,
                                yp,
                                pw,
                                width,
                                height,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip_cdfs, dc_sign_ctxs) =
                                sb_tu_contexts(&tus, sb_y, sb_x, above, left, qc, tmc, tmr);
                            encode_luma_block_split(
                                enc,
                                &tus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                mode_idx,
                                false,
                                pc,
                            );
                        }
                        (16, 8) => {
                            let (tus2, mode_idx) = encode_luma_leaf32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) =
                                sb_tu_contexts_64x32(&tus2, sb_y, sb_x, above, left, qc, tmc, tmr);
                            encode_luma_leaf_64x32(enc, &tus2, &skip2, &dcs2, mode_idx, false, pc);
                        }
                        (8, 16) => {
                            let (tus2, mode_idx) = encode_luma_leaf_v32x64(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0), (32, 0)],
                                &tus2,
                                sb_y,
                                sb_x,
                                above,
                                left,
                                qc,
                                tmc,
                                tmr,
                                false,
                            );
                            let s2 = [skip2[0], skip2[1]];
                            let d2 = [dcs2[0], dcs2[1]];
                            encode_luma_leaf_32x64(enc, &tus2, &s2, &d2, mode_idx, false, pc);
                        }
                        (8, 8) => {
                            let (tu, mode_idx) = encode_luma_leaf_s32x32(
                                recy,
                                yp,
                                pw,
                                tmc,
                                tmr,
                                sb_y,
                                sb_x,
                                &bases.luma,
                                qstep_i,
                                &tables::SCAN,
                                neutral,
                                qc,
                                self.tune.rdoq_lambda,
                                self.speed,
                                self.bit_depth as i32,
                            );
                            let (skip2, dcs2) = sb_tu_contexts_pos(
                                &[(0, 0)],
                                std::slice::from_ref(&tu),
                                sb_y,
                                sb_x,
                                above,
                                left,
                                qc,
                                tmc,
                                tmr,
                                true,
                            );
                            encode_luma_leaf_32x32(
                                enc, &tu, skip2[0], dcs2[0], mode_idx, false, pc,
                            );
                        }
                        (4, 16) => {
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma16x64.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 64, pred),
                                0.0,
                                &SCAN16X32,
                            );
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                64,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN16X32,
                                    16,
                                    64,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 16, true,
                            );
                            encode_luma_leaf_16x64(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (16, 4) => {
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let lev = bases.luma64x16.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 64, 16, pred),
                                0.0,
                                &SCAN32X16,
                            );
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                64,
                                16,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN32X16,
                                    64,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let tu = levels_to_coeffs(&lev);
                            let (skip, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 16, 4, true,
                            );
                            encode_luma_leaf_64x16(enc, &tu, skip, dcs, 0, false, pc);
                        }
                        (2, 8) => {
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let mut lev = bases.luma8x32.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 8, 32, pred),
                                0.0,
                                &SCAN8X32,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
                            }
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                8,
                                32,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN8X32,
                                    8,
                                    32,
                                    self.bit_depth as i32,
                                ),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 2, 8, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                enc, dc_level, skip, dcs, 0, false, pc, 18958,
                            );
                        }
                        (8, 2) => {
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let mut lev = bases.luma32x8.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 32, 8, pred),
                                0.0,
                                &SCAN32X8,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
                            }
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                32,
                                8,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN32X8,
                                    32,
                                    8,
                                    self.bit_depth as i32,
                                ),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 8, 2, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                enc, dc_level, skip, dcs, 0, false, pc, 18958,
                            );
                        }
                        (4, 4) => {
                            let pred = dc_pred_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                neutral,
                                self.bit_depth as i32,
                            );
                            let mut lev = bases.luma16x16.project_scan(
                                &get_residual_rect(yp, pw, sb_y, sb_x, 16, 16, pred),
                                0.0,
                                &SCAN16,
                            );
                            for v in lev[1..].iter_mut() {
                                *v = 0.0;
                            }
                            put_block_rect(
                                recy,
                                pw,
                                sb_y,
                                sb_x,
                                16,
                                16,
                                &crate::av2::itx422::reconstruct_chroma(
                                    pred,
                                    &lev,
                                    qstep_i,
                                    &SCAN16,
                                    16,
                                    16,
                                    self.bit_depth as i32,
                                ),
                            );
                            let dc_level = lev[0] as i32;
                            let tu: Vec<Coeff> = if dc_level != 0 {
                                vec![(0, dc_level)]
                            } else {
                                vec![]
                            };
                            let (_s, dcs) = sb_tu_contexts_rect(
                                &tu, sb_y, sb_x, above, left, qc, tmc, tmr, 4, 4, true,
                            );
                            let skip = SKIP_TX16_QC[qc][0] as u32;
                            encode_luma_leaf_dc_class2(
                                enc, dc_level, skip, dcs, 0, false, pc, 11074,
                            );
                        }
                        other => unreachable!("unsupported native 4:0:0 leaf {:?}", other),
                    }
                }
            }
        }
    }

    /// Encode a 4:0:0 (monochrome / luma-only) still. `y` is `width × height`.
    /// Four 32x32 luma TUs per superblock; no chroma is coded or signalled
    /// (`has_chroma = false` ⇒ no chroma intra mode, profile 0, layout uvlc 1).
    pub fn encode_yuv400<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_400()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);

        let layout = Layout::Monochrome;
        let config = self.config(layout);

        if config.lossless {
            return Ok(
                self.encode_yuv400_lossless(&yp, pw, ph, width, height, &config, color, threads)
            );
        }

        let mut recy = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let qstep_i = quant::qstep(self.base_q_idx as u32) as i32;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;

        let native_mi = lossy_native_mi(width, height);
        let (tmc, tmr) = native_mi.unwrap_or(((pw / 4) as i64, (ph / 4) as i64));
        // Same edge fix as 4:2:0: residues {10,12,14} return native (un-padded) mi
        // extents from `lossy_native_mi` but `lossy_needs_partition` is false, so the old
        // code took the fast whole-SB path with a native/padded extent mismatch and
        // desynced the decoder on the partial edge SB (any side ≡ 40/48/56 mod 64). Route
        // every non-64-aligned dimension through the edge-aware partition walk instead.
        let mc_edge = (((width + 7) & !7) / 4) as i64 % 16;
        let mr_edge = (((height + 7) & !7) / 4) as i64 % 16;
        let needs_partition = native_mi.is_some()
            && (lossy_needs_partition(width, height) || mc_edge != 0 || mr_edge != 0);
        if needs_partition {
            let mut above_pctx = vec![0u8; tmc as usize + 16];
            let mut left_pctx = vec![0u8; 16];
            self.encode_yuv400_partition(
                &mut enc,
                LumaPlanes {
                    rec: &mut recy,
                    src: &yp,
                },
                &PartitionPass {
                    luma_stride: pw,
                    chroma_stride: 0,
                    width,
                    height,
                    sb_rows,
                    sb_cols,
                    tmc,
                    tmr,
                    quant: QuantCtx {
                        qc,
                        neutral,
                        qstep: qstep_i,
                    },
                },
                PartitionNeighbors {
                    above: &mut above,
                    left: &mut left,
                    above_pctx: &mut above_pctx,
                    left_pctx: &mut left_pctx,
                },
            );
            return Ok(self.finish(enc, &config, pw, ph, width, height, color));
        }

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let (tus, mode_idx) = encode_luma_sb(
                    &mut recy,
                    &yp,
                    pw,
                    width,
                    height,
                    sb_y,
                    sb_x,
                    &bases.luma,
                    qstep_i,
                    &crate::av2::tables::SCAN,
                    neutral,
                    qc,
                    self.tune.rdoq_lambda,
                    self.speed,
                    self.bit_depth as i32,
                );
                let (skip_cdfs, dc_sign_ctxs) = sb_tu_contexts(
                    &tus,
                    sb_y,
                    sb_x,
                    &mut above,
                    &mut left,
                    qc,
                    (pw / 4) as i64,
                    (ph / 4) as i64,
                );
                encode_luma_block_split(
                    &mut enc,
                    &tus,
                    &skip_cdfs,
                    &dc_sign_ctxs,
                    mode_idx,
                    false,
                    12276,
                );
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Lossless (base_q=0) monochrome encode: each 64x64 superblock is coded as 256
    /// 4x4 transform units (forced TX_4X4), DC-predicted per TU and carried by the 4x4
    /// WHT. `yp` is the SB-padded source plane. The pixel reconstruction is bit-exact;
    /// the 4x4 coefficient CDFs/contexts are still being validated against the decoder.
    #[allow(clippy::too_many_arguments)]
    fn encode_yuv400_lossless(
        &self,
        yp: &[f32],
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        config: &Config,
        color: &Cicp,
        threads: usize,
    ) -> Av2Frame {
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx); // base_q=0 -> q-context 0
        let neutral = self.dc_neutral();
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        // mi grid is 8px-aligned (avm dec_set_mb_mi); the recursive forced-split coder
        // handles every boundary geometry, so we always code the real (8-aligned) grid.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;

        // Phase A: per-SB TU generation is independent in lossless (recon == source),
        // so generate the clipped SB TU grids in parallel across `threads`.
        let nsb = sb_rows * sb_cols;
        let mut sbtus: Vec<Vec<Vec<Coeff>>> = (0..nsb).map(|_| Vec::new()).collect();
        let nthreads = Self::resolve_threads(threads);
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                let (row, col) = (idx / sb_cols, idx % sb_cols);
                let (rr, rc) = ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16));
                *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (code_mc, code_mr) = (code_mc, code_mr);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (row, col) = ((base + k) / sb_cols, (base + k) % sb_cols);
                            let rr = (code_mr - row * 16).min(16);
                            let rc = (code_mc - col * 16).min(16);
                            *slot = lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc);
                        }
                    });
                }
            });
        }

        let mut above_pctx = vec![0u8; code_mc + 16];

        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let rr = (code_mr - row * 16).min(16);
                let rc = (code_mc - col * 16).min(16);
                // SB grid of in-frame 4x4 TUs (precomputed in Phase A).
                let tus = &sbtus[row * sb_cols + col];
                let ops = partition::sb_partition_ops(
                    row,
                    col,
                    code_mr,
                    code_mc,
                    &mut above_pctx,
                    &mut left_pctx,
                );
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val } => {
                            enc.encode_bool(cdf, val);
                        }
                        partition::Op::Leaf {
                            mi_row,
                            mi_col,
                            bw_mi,
                            bh_mi,
                            part_cdf,
                        } => {
                            let lr = mi_row - row * 16;
                            let lc = mi_col - col * 16;
                            let lrows = bh_mi.min(rr - lr);
                            let lcols = bw_mi.min(rc - lc);
                            let mut ltus = Vec::with_capacity(lrows * lcols);
                            for i in 0..lrows {
                                for j in 0..lcols {
                                    ltus.push(tus[(lr + i) * rc + (lc + j)].clone());
                                }
                            }
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
                            let (skip_ctx, dc_sign_ctxs) =
                                sb_tu4_contexts(&ltus, ly, lx, &mut above, &mut left, lrows, lcols);
                            let skip_cdfs: Vec<u32> = skip_ctx
                                .iter()
                                .map(|&c| TXB_SKIP_TX4_Q0[c] as u32)
                                .collect();
                            encode_lossless_luma_sb(
                                &mut enc,
                                &ltus,
                                &skip_cdfs,
                                &dc_sign_ctxs,
                                0,
                                false,
                                part_cdf,
                            );
                        }
                    }
                }
            }
        }
        self.finish(enc, config, pw, ph, width, height, color)
    }

    /// Encode a luma-only (4:0:0 / monochrome) image to AV2.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_400<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_400()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let plane = img.planes[0].to_vec();
        self.encode_yuv400(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [plane, Vec::new(), Vec::new(), Vec::new()],
            },
            color,
            threads,
        )
    }
}
