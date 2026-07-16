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
    /// 4:4:4 lossless (q=0): luma + full-resolution U/V, all TX_4X4 WHT. Per superblock
    /// the block codes regular or lossless-DPCM intra modes, then luma, U and V
    /// TUs in AVM shared-tree plane order.
    pub(super) fn encode_yuv444_lossless<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
        threads: usize,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);
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
                stride: pw,
                subsampling_x: 0,
                subsampling_y: 0,
            },
            lossless_rd::IbcPlane {
                data: &vp,
                stride: pw,
                subsampling_x: 0,
                subsampling_y: 0,
            },
        ];
        let mut config = self.config(Layout::I444);
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
        enc.mhccp_ssx = false;
        enc.mhccp_ssy = false;
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
        // mi grid is 8px-aligned; recursion handles every boundary -> always exact.
        let code_mc = ((width + 7) & !7) / 4;
        let code_mr = ((height + 7) & !7) / 4;
        let rem = |row: usize, col: usize| -> (usize, usize) {
            ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16))
        };
        // luma ctx grids (0x40 = neutral DC-sign packing); chroma grids store cul (init 0).
        let mut ya = vec![0x40u8; pw / 4 + 16];
        let mut yl = vec![0x40u8; ph / 4 + 16];
        let mut ua = vec![0u8; pw / 4 + 16];
        let mut ul = vec![0u8; ph / 4 + 16];
        let mut va = vec![0u8; pw / 4 + 16];
        let mut vl = vec![0u8; ph / 4 + 16];

        let nsb = sb_rows * sb_cols;
        // Phase A: per-SB TU generation (DC-pred + WHT + levels). Independent across SBs
        // (lossless reconstruction == source), so this is data-parallel.
        type PackedCoeff = Vec<Coeff>;
        // Work-stealing across SBs; tiny frames stay serial.
        let nthreads = if nsb < 8 {
            1
        } else {
            Self::resolve_threads(threads)
        };
        let sbtus: Vec<(Vec<PackedCoeff>, Vec<PackedCoeff>, Vec<PackedCoeff>)> =
            par_map_indexed(nthreads, nsb, |idx| {
                let (sb_y, sb_x) = ((idx / sb_cols) * 64, (idx % sb_cols) * 64);
                let (rr, rc) = rem(idx / sb_cols, idx % sb_cols);
                (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&up, pw, sb_y, sb_x, neutral, rr, rc),
                    lossless_sb_tus(&vp, pw, sb_y, sb_x, neutral, rr, rc),
                )
            });
        // Phase B: serial context derivation (cross-SB grids) + entropy coding.
        // Partition context arrays (av2 update_partition_context): `above` persists
        // down columns frame-wide; `left` is len-16 and zeroed per SB row.
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
                            let (ly, lx) = (sb_y + lr * 4, sb_x + lc * 4);
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
                                        x: lx,
                                        y: ly,
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
                            let slice = |g: &[Vec<Coeff>]| -> Vec<Vec<Coeff>> {
                                let mut v = Vec::with_capacity(lrows * lcols);
                                for i in 0..lrows {
                                    for j in 0..lcols {
                                        v.push(g[(lr + i) * rc + (lc + j)].clone());
                                    }
                                }
                                v
                            };
                            let (lytus, lutus, lvtus, ymode, palette, dpcm) = if use_intrabc {
                                let empty = || vec![Vec::new(); lrows * lcols];
                                (empty(), empty(), empty(), 0, None, LosslessDpcm::default())
                            } else {
                                let ycand = lossless_rd::best_luma_block(
                                    &yp,
                                    pw,
                                    lossless_rd::LosslessBlockRd {
                                        y: ly,
                                        x: lx,
                                        rows: lrows,
                                        cols: lcols,
                                        frame_width: width,
                                        frame_height: height,
                                        neutral,
                                        qc: enc.qc,
                                        y_ctx,
                                    },
                                );
                                let palette_ctx = palette_grid.context(mi_row, mi_col);
                                let palette = (palette_ctx == 0
                                    && lx + block_width <= width
                                    && ly + block_height <= height)
                                    .then(|| {
                                        lossless_rd::exact_luma_palette(
                                            &yp,
                                            pw,
                                            ly,
                                            lx,
                                            block_width,
                                            block_height,
                                            self.bit_depth,
                                        )
                                    })
                                    .flatten();
                                let ymode = if palette.is_some() { 0 } else { ycand.mode_idx };
                                let dpcm_y = if palette.is_none() { ycand.dpcm } else { None };
                                let lytus = if palette.is_some() {
                                    vec![Vec::new(); lrows * lcols]
                                } else {
                                    ycand.tus
                                };
                                let base_u = slice(utus);
                                let base_v = slice(vtus);
                                let ccand = lossless_rd::best_chroma_block(&base_u, &base_v);
                                (
                                    lytus,
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
                            let (yskip, ydcs) =
                                sb_tu4_contexts(&lytus, ly, lx, &mut ya, &mut yl, lrows, lcols);
                            let yskip_cdfs: Vec<u32> =
                                yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                            let uskip = sb_tu4_chroma_skip(
                                &lutus, ly, lx, &mut ua, &mut ul, false, false, lrows, lcols,
                            );
                            // avm's eob_u_flag is the LAST U TU of the block, used by every V TU.
                            let u_last_nz =
                                lutus.last().is_some_and(|t| t.iter().any(|&(_, l)| l != 0));
                            let vskip = sb_tu4_chroma_skip(
                                &lvtus, ly, lx, &mut va, &mut vl, true, u_last_nz, lrows, lcols,
                            );
                            // modes (incl. uv) + luma coeffs, then U, then V (shared-tree order)
                            enc.cur_bw4 = bw_mi;
                            enc.cur_bh4 = bh_mi;
                            enc.y_ctx = y_ctx;
                            encode_lossless_luma_sb(
                                &mut enc,
                                &LosslessLumaBlock {
                                    tus: &lytus,
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
