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
use crate::av2::tiling::{
    MultitileAssembly, assemble_multitile, extract_subplane, lossless_tile_payload, tile_specs,
};

impl Av2Encoder {
    /// Lossless (base_q=0) monochrome encode: each 64x64 superblock is coded as 256
    /// 4x4 transform units (forced TX_4X4), DC-predicted per TU and carried by the 4x4
    /// WHT. `yp` is the SB-padded source plane. The pixel reconstruction is bit-exact;
    /// the 4x4 coefficient CDFs/contexts are still being validated against the decoder.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_yuv400_lossless(
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
        let inter_tile = self.inter_tile.load(std::sync::atomic::Ordering::Relaxed);
        let ibc_planes = [lossless_rd::IbcPlane {
            data: yp,
            stride: pw,
            subsampling_x: 0,
            subsampling_y: 0,
        }];
        let mut config = config.clone();
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
        enc.cfl = self.tune.cfl && self.base_q_idx != 0; // base_q=0 -> q-context 0
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
        // Work-stealing across SBs; tiny frames stay serial.
        let nthreads = if nsb < 8 {
            1
        } else {
            Self::resolve_threads(threads)
        };
        let sbtus: Vec<Vec<Vec<Coeff>>> = par_map_indexed(nthreads, nsb, |idx| {
            let (row, col) = (idx / sb_cols, idx % sb_cols);
            let (rr, rc) = ((code_mr - row * 16).min(16), (code_mc - col * 16).min(16));
            lossless_sb_tus(yp, pw, row * 64, col * 64, neutral, rr, rc)
        });

        let mut above_pctx = vec![0u8; code_mc + 16];
        let mut palette_grid = lossless::PaletteGrid::new(code_mr, code_mc);
        let mut ibc_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);
        let mut skip_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);
        let mut directional_grid = lossless::BlockFlagGrid::new(code_mr, code_mc);

        for row in 0..sb_rows {
            let mut left_pctx = [0u8; 16];
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let rr = (code_mr - row * 16).min(16);
                let rc = (code_mc - col * 16).min(16);
                // Phase-A DC TU grid retained for chroma-less skip-context warm
                // paths; luma TUs are now regenerated per leaf by the mode RD.
                let _tus = &sbtus[row * sb_cols + col];
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
                            let (ltus, mode_idx, palette, dpcm_y) = if use_intrabc {
                                (vec![Vec::new(); lrows * lcols], 0, None, None)
                            } else {
                                let cand = lossless_rd::best_luma_block(
                                    yp,
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
                                            yp,
                                            pw,
                                            ly,
                                            lx,
                                            block_width,
                                            block_height,
                                            self.bit_depth,
                                        )
                                    })
                                    .flatten();
                                let mode_idx = if palette.is_some() { 0 } else { cand.mode_idx };
                                let dpcm_y = if palette.is_none() { cand.dpcm } else { None };
                                let ltus = if palette.is_some() {
                                    vec![Vec::new(); lrows * lcols]
                                } else {
                                    cand.tus
                                };
                                (ltus, mode_idx, palette, dpcm_y)
                            };
                            let (skip_ctx, dc_sign_ctxs) =
                                sb_tu4_contexts(&ltus, ly, lx, &mut above, &mut left, lrows, lcols);
                            let skip_cdfs: Vec<u32> = skip_ctx
                                .iter()
                                .map(|&c| TXB_SKIP_TX4_Q0[c] as u32)
                                .collect();
                            enc.cur_bw4 = bw_mi;
                            enc.cur_bh4 = bh_mi;
                            enc.y_ctx = y_ctx;
                            encode_lossless_luma_sb(
                                &mut enc,
                                &LosslessLumaBlock {
                                    tus: &ltus,
                                    skip_cdfs: &skip_cdfs,
                                    dc_sign_ctxs: &dc_sign_ctxs,
                                    mode_idx,
                                    has_chroma: false,
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
                                    dpcm: LosslessDpcm {
                                        y: dpcm_y,
                                        uv: None,
                                    },
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
                                dpcm_y.is_some(),
                            );
                        }
                    }
                }
            }
        }
        self.finish(enc, &config, pw, ph, width, height, color)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_400_lossless_tiled(
        &self,
        yp: &[f32],
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        config: &Config,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
    ) -> Av2Frame {
        let specs = tile_specs(pw, ph, log2c, log2r);
        // Tiles are independent sub-frame encodes; parallelize across them (each tile
        // single-threaded). `par_map_indexed` preserves index order, so the assembled
        // stream is byte-identical to the previous serial loop. Mirrors the 4:4:4 path.
        let n = specs.len();
        let nthreads = Self::resolve_threads(self.threads).min(n.max(1));
        let tiles: Vec<Vec<u8>> = par_map_indexed(nthreads, n, |i| {
            let (x0, y0, tw, th) = specs[i];
            let tile_y = extract_subplane(yp, pw, x0, y0, tw, th);
            let frame = self.encode_yuv400_lossless(&tile_y, tw, th, tw, th, config, color, 1);
            lossless_tile_payload(&frame, config, tw, th)
        });
        assemble_multitile(
            &MultitileAssembly {
                config,
                coded_width: pw,
                coded_height: ph,
                display_width: width,
                display_height: height,
                color,
                log2_cols: log2c,
                log2_rows: log2r,
                bit_depth: self.bit_depth,
                chroma_format: ChromaFormat::Monochrome,
            },
            &tiles,
        )
    }
}
