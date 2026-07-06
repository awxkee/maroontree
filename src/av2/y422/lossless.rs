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

use super::super::*;

impl Av2Encoder {
    /// Lossless 4:2:2: identical recursion to the 4:4:4 lossless path, but the chroma
    /// planes are half-width, so each luma leaf of `lcols × lrows` 4×4 WHT TUs maps to a
    /// `(lcols/2) × lrows` chroma block at chroma column `lc/2`. The mode-info grid is
    /// 8px-aligned so every leaf starts on an even mi column, keeping the halving exact.
    pub(super) fn encode_yuv422_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_422()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph);
        let (cw, chh) = (width.div_ceil(2), height);
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), cw, chh, pcw, pch);
        let vp = pad_plane(&to_plane(cr), cw, chh, pcw, pch);
        let inter_tile = self.inter_tile.load(std::sync::atomic::Ordering::Relaxed);
        let ibc_planes = [
            lossless_rd::IbcPlane {
                data: &yp,
                stride: pw,
                subsampling_x: 0,
                subsampling_y: 0,
            },
            lossless_rd::IbcPlane {
                data: &up,
                stride: pcw,
                subsampling_x: 1,
                subsampling_y: 0,
            },
            lossless_rd::IbcPlane {
                data: &vp,
                stride: pcw,
                subsampling_x: 1,
                subsampling_y: 0,
            },
        ];
        let mut config = self.config(Layout::I422);
        config.allow_intrabc = !inter_tile
            && self.tune.tile_cols == 1
            && self.tune.tile_rows == 1
            && lossless_rd::frame_has_default_ibc(&ibc_planes, width, height);
        let mut enc = RangeEncoder::new();
        enc.inter_tile = inter_tile;
        enc.qc = get_q_ctx(self.base_q_idx);
        if self.tune.updating_cdf && self.base_q_idx != 0 {
            enc.enable_adaptive_cdf(enc.qc);
        }
        enc.cfl = self.tune.cfl && self.base_q_idx != 0;
        enc.mhccp = self.tune.mhccp && self.base_q_idx != 0;
        enc.mhccp_ssx = true;
        enc.mhccp_ssy = false;
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        // Chroma neighbor grids live in chroma-pixel space (half-width above grid).
        let mut ua = vec![0u8; pcw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pcw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        type PackedCoeff = Vec<Coeff>;
        // Phase A: per-SB 4×4 WHT TUs. Luma over the full SB (rr×rc), chroma over the
        // half-width SB (rr × rc/2) at chroma origin sb_x/2.
        // Work-stealing across SBs; tiny frames stay serial.
        let nthreads = if nsb < 8 {
            1
        } else {
            Self::resolve_threads(threads)
        };
        let sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> =
            par_map_indexed(nthreads, nsb, |idx| {
                let (row, col) = (idx / sb_cols, idx % sb_cols);
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                    lossless_sb_tus(&vp, pcw, sb_y, sb_x / 2, neutral, rr, rc / 2),
                )
            });
        // Phase B: serial partition + entropy walk (shared luma tree drives both planes).
        let mut above_pctx = vec![0u8; code_mc + 16];
        let mut palette_grid = lossless::PaletteGrid::new(code_mr, code_mc);
        let mut ibc_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);
        let mut skip_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);
        let mut directional_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);
        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (rr, rc) = rem(row, col);
                let (_ytus, utus, vtus) = &sbtus[row * sb_cols + col];
                let ccols = rc / 2; // chroma TU columns in this SB's grid
                let split_for_ibc = config.allow_intrabc
                    && rr == 16
                    && rc == 16
                    && lossless_rd::sb_needs_ibc_split(&ibc_planes, width, height, sb_x, sb_y);
                let ops = if split_for_ibc {
                    partition::sb_square_split_ops(row, col, &mut above_pctx, &mut left_pctx)
                } else {
                    partition::sb_partition_ops(
                        row,
                        col,
                        code_mr,
                        code_mc,
                        &mut above_pctx,
                        &mut left_pctx,
                    )
                };
                for op in &ops {
                    match *op {
                        partition::Op::RectType { cdf, val, ctx } => {
                            enc.bool_rect_type(cdf, val, ctx);
                        }
                        partition::Op::Split {
                            do_split_cdf,
                            square_cdf,
                        } => {
                            enc.bool_do_split(do_split_cdf, 1);
                            if square_cdf != 0 {
                                enc.bool_do_square_split(square_cdf, 1);
                            }
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
                            let (ly0, lx0) = (sb_y + lr * 4, sb_x + lc * 4);
                            let block_width = bw_mi * 4;
                            let block_height = bh_mi * 4;
                            let ibc_allowed = config.allow_intrabc
                                && block_width <= 64
                                && block_height <= 64
                                && (block_width != 64 || block_height != 64);
                            let use_intrabc = ibc_allowed
                                && lossless_rd::exact_default_ibc(
                                    &ibc_planes,
                                    width,
                                    height,
                                    lossless_rd::IbcBlock {
                                        x: lx0,
                                        y: ly0,
                                        width: block_width,
                                        height: block_height,
                                    },
                                );
                            let use_ctx =
                                ibc_grid.npos_context(mi_row, mi_col, bh_mi, bw_mi, false);
                            let block_skip_ctx =
                                skip_grid.npos_context(mi_row, mi_col, bh_mi, bw_mi, true);
                            let y_ctx =
                                directional_grid.joint_mode_context(mi_row, mi_col, bh_mi, bw_mi);

                            // Chroma is half-width: clc = lc/2, ccols_leaf = lcols/2.
                            let clc = lc / 2;
                            let ccols_leaf = lcols / 2;
                            let cslice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * ccols_leaf);
                                for i in 0..lrows {
                                    for j in 0..ccols_leaf {
                                        v.push(g[(lr + i) * ccols + (clc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (yslice, lutus, lvtus, ymode, palette, dpcm) = if use_intrabc {
                                (
                                    vec![Vec::new(); lrows * lcols],
                                    vec![Vec::new(); lrows * ccols_leaf],
                                    vec![Vec::new(); lrows * ccols_leaf],
                                    0,
                                    None,
                                    LosslessDpcm::default(),
                                )
                            } else {
                                // Per-leaf luma intra mode RD (lossless: pure rate);
                                // DC wins reproduce the previous bitstream exactly.
                                let ycand = lossless_rd::best_luma_block(
                                    &yp,
                                    pw,
                                    lossless_rd::LosslessBlockRd {
                                        y: ly0,
                                        x: lx0,
                                        rows: lrows,
                                        cols: lcols,
                                        neutral,
                                        qc: enc.qc,
                                        y_ctx,
                                    },
                                );
                                let palette_ctx = palette_grid.context(mi_row, mi_col);
                                let palette = (palette_ctx == 0
                                    && lx0 + block_width <= width
                                    && ly0 + block_height <= height)
                                    .then(|| {
                                        lossless_rd::exact_luma_palette(
                                            &yp,
                                            pw,
                                            ly0,
                                            lx0,
                                            block_width,
                                            block_height,
                                            self.bit_depth,
                                        )
                                    })
                                    .flatten();
                                let ymode = if palette.is_some() { 0 } else { ycand.mode_idx };
                                let dpcm_y = if palette.is_none() { ycand.dpcm } else { None };
                                let yslice = if palette.is_some() {
                                    vec![Vec::new(); lrows * lcols]
                                } else {
                                    ycand.tus
                                };
                                let base_u = cslice(utus);
                                let base_v = cslice(vtus);
                                let ccand = lossless_rd::best_chroma_block(
                                    &up,
                                    &vp,
                                    pcw,
                                    &base_u,
                                    &base_v,
                                    lossless_rd::ChromaBlockRd {
                                        y: ly0,
                                        x: lx0 / 2,
                                        rows: lrows,
                                        cols: ccols_leaf,
                                        neutral,
                                        qc: enc.qc,
                                        luma_directional: dpcm_y.is_some(),
                                    },
                                );
                                (
                                    yslice,
                                    ccand.u_tus,
                                    ccand.v_tus,
                                    ymode,
                                    palette,
                                    LosslessDpcm {
                                        y: dpcm_y,
                                        uv: ccand.dpcm,
                                    },
                                )
                            };
                            let cx = lx0 / 2;
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&yslice, ly0, lx0, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly0, cx, &mut ua, &mut ul, false, false, lrows, ccols_leaf,
                            );
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly0, cx, &mut va, &mut vl, true, u_last_nz, lrows,
                                ccols_leaf,
                            );
                            enc.cur_bw4 = bw_mi;
                            enc.cur_bh4 = bh_mi;
                            enc.y_ctx = y_ctx;
                            encode_lossless_luma_sb(
                                &mut enc,
                                &LosslessLumaBlock {
                                    tus: &yslice,
                                    skip_cdfs: &yskip_cdfs,
                                    dc_sign_ctxs: &ydcs,
                                    mode_idx: ymode,
                                    has_chroma: true,
                                    partition_cdf: part_cdf,
                                    palette: palette.as_ref(),
                                    width: block_width,
                                    height: block_height,
                                    bit_depth: self.bit_depth,
                                    intrabc: LosslessIntrabc {
                                        allowed: ibc_allowed,
                                        selected: use_intrabc,
                                        use_ctx,
                                        skip_ctx: block_skip_ctx,
                                    },
                                    dpcm,
                                },
                            );
                            palette_grid.set_block(mi_row, mi_col, lrows, lcols, palette.is_some());
                            ibc_grid.set_block(mi_row, mi_col, lrows, lcols, use_intrabc);
                            skip_grid.set_block(mi_row, mi_col, lrows, lcols, use_intrabc);
                            directional_grid.set_block(
                                mi_row,
                                mi_col,
                                lrows,
                                lcols,
                                dpcm.y.is_some(),
                            );
                            if !use_intrabc {
                                for (i, tu) in lutus.iter().enumerate() {
                                    encode_chroma_tu4(
                                        &mut enc,
                                        tu,
                                        TXB_SKIP_TX4_Q0[uskip[i]] as u32,
                                        false,
                                    );
                                }
                                for (i, tu) in lvtus.iter().enumerate() {
                                    encode_chroma_tu4(
                                        &mut enc,
                                        tu,
                                        V_TXB_SKIP_TX4_Q0[vskip[i]] as u32,
                                        true,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }
}
