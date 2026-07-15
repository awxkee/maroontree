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
use crate::av2::tiling::{MultitileAssembly, assemble_multitile, extract_subplane, tile_specs};

pub(super) struct Tiled400Request<'a> {
    pub(super) yp: &'a [f32],
    pub(super) pw: usize,
    pub(super) ph: usize,
    pub(super) width: usize,
    pub(super) height: usize,
    pub(super) config: &'a Config,
    pub(super) color: &'a Cicp,
    pub(super) log2_cols: usize,
    pub(super) log2_rows: usize,
    pub(super) threads: usize,
}

impl Av2Encoder {
    /// Parallel lossy monochrome tiles. Each tile runs a single-threaded capture;
    /// optional CDEF is decided once from the stitched frame and replayed per tile.
    pub(super) fn encode_400_tiled(&self, request: &Tiled400Request<'_>) -> Av2Frame {
        let Tiled400Request {
            yp,
            pw,
            ph,
            width,
            height,
            config,
            color,
            log2_cols,
            log2_rows,
            threads,
        } = *request;
        let specs = tile_specs(pw, ph, log2_cols, log2_rows);
        let n = specs.len();
        let nthreads = Self::resolve_threads(threads).min(n.max(1));

        use std::sync::atomic::Ordering::Relaxed;
        let cdef_search = self.tune.cdef;
        let capture_recon = self.capture_recon.load(Relaxed);
        if cdef_search {
            self.capture_recon.store(true, Relaxed);
        }
        type TileOut = (Vec<u8>, Vec<f32>, replay::DecisionRecord);
        let first: Vec<TileOut> = par_map_indexed(nthreads, n, |i| {
            let _guard = replay::TileSubencodeGuard::enter();
            let (x0, y0, tw, th) = specs[i];
            let ty = extract_subplane(yp, pw, x0, y0, tw, th);
            let mut record = replay::DecisionRecord::new();
            let mut enc = self.encode_400_core(
                super::lossy::Core400Frame {
                    yp: &ty,
                    pw: tw,
                    ph: th,
                    width: tw,
                    height: th,
                },
                None,
                replay::DecideMode::Capture(&mut record),
            );
            let recon = enc.recon.pop().unwrap_or_default();
            (enc.finish(), recon, record)
        });
        self.capture_recon.store(capture_recon, Relaxed);

        let mut recon = if cdef_search {
            vec![0f32; pw * ph]
        } else {
            Vec::new()
        };
        if cdef_search {
            for ((_, tile_recon, _), &(x0, y0, tw, th)) in first.iter().zip(&specs) {
                for row in 0..th {
                    recon[(y0 + row) * pw + x0..(y0 + row) * pw + x0 + tw]
                        .copy_from_slice(&tile_recon[row * tw..row * tw + tw]);
                }
            }
        }

        let decision = if cdef_search {
            cdef_est::search_per_block(
                std::slice::from_ref(&recon),
                &[yp.to_vec()],
                pw,
                ph,
                pw,
                ph,
                0,
                0,
                false,
                self.base_q_idx,
                self.bit_depth,
            )
        } else {
            None
        };

        let owned_config;
        let (tiles, config): (Vec<Vec<u8>>, &Config) = if let Some(decision) = &decision {
            let bytes = par_map_indexed(nthreads, n, |i| {
                let _guard = replay::TileSubencodeGuard::enter();
                let (x0, y0, tw, th) = specs[i];
                let ty = extract_subplane(yp, pw, x0, y0, tw, th);
                let (sbx0, sby0) = (x0 / 64, y0 / 64);
                let (tile_cols, tile_rows) = (tw / 64, th / 64);
                let mut grid = vec![0u8; tile_cols * tile_rows];
                for row in 0..tile_rows {
                    for col in 0..tile_cols {
                        grid[row * tile_cols + col] =
                            decision.grid[(sby0 + row) * decision.sb_cols + sbx0 + col];
                    }
                }
                let tile_decision = cdef_est::CdefDecision {
                    damping: decision.damping,
                    y_str: decision.y_str,
                    uv_str: decision.uv_str,
                    grid,
                    sb_cols: tile_cols,
                };
                let cursor = replay::DecisionCursor::new(&first[i].2);
                self.encode_400_core(
                    super::lossy::Core400Frame {
                        yp: &ty,
                        pw: tw,
                        ph: th,
                        width: tw,
                        height: th,
                    },
                    Some(&tile_decision),
                    replay::DecideMode::Replay(cursor),
                )
                .finish()
            });
            let mut c = config.clone();
            c.cdef = Some((decision.y_str, decision.uv_str, decision.damping));
            c.cdef_per_block = true;
            owned_config = c;
            (bytes, &owned_config)
        } else {
            (
                first.iter().map(|(bytes, _, _)| bytes.clone()).collect(),
                config,
            )
        };

        assemble_multitile(
            &MultitileAssembly {
                config,
                coded_width: pw,
                coded_height: ph,
                display_width: width,
                display_height: height,
                color,
                log2_cols,
                log2_rows,
                bit_depth: self.bit_depth,
                chroma_format: ChromaFormat::Monochrome,
            },
            &tiles,
        )
    }
}
