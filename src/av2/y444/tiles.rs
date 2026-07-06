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
    MultitileAssembly, TiledEncodeRequest, assemble_multitile, extract_subplane,
    extract_subplane_pixels, lossless_tile_payload, pad_plane_pixels, tile_specs,
};

impl Av2Encoder {
    pub(super) fn encode_444_lossless_tiled<T: Pixel>(
        &self,
        image: &PlanarImage<T>,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
    ) -> Av2Frame {
        let (width, height) = (image.width, image.height);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let planes = [
            pad_plane_pixels(&image.planes[0], width, height, pw, ph),
            pad_plane_pixels(&image.planes[1], width, height, pw, ph),
            pad_plane_pixels(&image.planes[2], width, height, pw, ph),
        ];
        let specs = tile_specs(pw, ph, log2c, log2r);
        let config = self.config(Layout::I444);
        // Tiles are independent sub-frame encodes (CDFs/contexts reset, tile boundary
        // == frame boundary for prediction), so they run on a work-stealing map — each
        // tile encoded single-threaded (threads=1) while workers parallelise across
        // tiles. `par_map_indexed` preserves order, so the assembled stream is
        // byte-identical to the previous serial loop. Mirrors the lossy `encode_*_tiled`.
        let n = specs.len();
        let nthreads = Self::resolve_threads(self.threads).min(n.max(1));
        let tiles: Vec<Vec<u8>> = par_map_indexed(nthreads, n, |i| {
            let (x0, y0, tw, th) = specs[i];
            let tile = PlanarImage {
                width: tw,
                height: th,
                bit_depth: image.bit_depth,
                planes: [
                    extract_subplane_pixels(&planes[0], pw, x0, y0, tw, th),
                    extract_subplane_pixels(&planes[1], pw, x0, y0, tw, th),
                    extract_subplane_pixels(&planes[2], pw, x0, y0, tw, th),
                    Vec::new(),
                ],
            };
            let frame = self.encode_yuv444_lossless(&tile, color, 1).unwrap();
            lossless_tile_payload(&frame, &config, tw, th)
        });
        assemble_multitile(
            &MultitileAssembly {
                config: &config,
                coded_width: pw,
                coded_height: ph,
                display_width: width,
                display_height: height,
                color,
                log2_cols: log2c,
                log2_rows: log2r,
                bit_depth: self.bit_depth,
                chroma_format: ChromaFormat::Yuv444,
            },
            &tiles,
        )
    }

    /// Multi-tile 4:4:4 assembly. Each tile is encoded as an independent sub-frame
    /// (CDFs/contexts reset, tile boundary == frame boundary for prediction), then
    /// concatenated under one multi-tile frame header with size prefixes. Tiles are
    /// independent, so the per-tile encodes run in parallel across `threads` workers.
    pub(super) fn encode_444_tiled(&self, request: &TiledEncodeRequest<'_>) -> Av2Frame {
        let TiledEncodeRequest {
            y: yf,
            u: cbf,
            v: crf,
            width,
            height,
            config,
            color,
            log2_cols: log2c,
            log2_rows: log2r,
            threads,
        } = *request;
        // Tile column/row boundaries fall on 64-px superblock edges, so every interior
        // tile is SB-aligned; only the right-column / bottom-row tiles inherit the
        // frame's partial edge. A tile decodes correctly in-frame when its dimensions
        // are boundary-exact (lossy_native_mi is Some — SB-aligned or a supported
        // residue edge). When *every* tile is exact we signal the real frame size and
        // each tile carries its own native partial-edge entropy (byte-identical to a
        // standalone encode of that region). When some edge tile is NOT exact (e.g. a
        // residue-2 corner that would otherwise pad+clap per tile — which is invalid in
        // a shared multi-tile frame), we instead pad the WHOLE frame to SB-aligned, carve
        // tiles on the padded grid so all of them are SB-aligned, signal the padded size,
        // and let the AVIF muxer crop back to width×height with one frame-level clap.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| lossy_native_mi(tw, th).is_some());
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (sig_w, sig_h, stride, planes, specs) = if exact {
            (
                width,
                height,
                width,
                (yf.to_vec(), cbf.to_vec(), crf.to_vec()),
                native_specs,
            )
        } else {
            (
                pw,
                ph,
                pw,
                (
                    pad_plane(yf, width, height, pw, ph),
                    pad_plane(cbf, width, height, pw, ph),
                    pad_plane(crf, width, height, pw, ph),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let n = specs.len();
        // Inter frames predict from the previous recon; carve it into per-tile
        // sub-windows so each independent tile encodes local block coordinates
        // address the correct reference region (see the 4:2:0 path). 4:4:4 chroma is
        // full-resolution, so chroma uses the same rectangle and stride as luma.
        let inter = self.inter_tile.load(Relaxed);
        let full_ref: std::sync::Arc<Vec<Vec<f32>>> = if inter {
            std::sync::Arc::clone(&self.last_ref.lock().unwrap())
        } else {
            std::sync::Arc::new(Vec::new())
        };
        let has_ref = full_ref.len() >= 3 && !full_ref[0].is_empty();
        // Frame-level CDEF strength is chosen from the assembled recon, so the
        // per-tile encodes must capture their recon planes for stitching. Remember
        // the prior state and restore it after (video DPB relies on it).
        use std::sync::atomic::Ordering::Relaxed;
        let cdef_search = self.tune.cdef;
        let prev_capture = self.capture_recon.load(Relaxed);
        if cdef_search {
            self.capture_recon.store(true, Relaxed);
        }
        // Tiles are independent sub-frame encodes with unequal cost, so they run on a
        // work-stealing map: each worker claims the next tile instead of a fixed chunk.
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        // Staged pipeline (per tile): pass 1 decides + captures each tile's whole-64
        // winners; pass 2 (CDEF) replays them, skipping their search.
        type TileOut = (Vec<u8>, Vec<Vec<f32>>, replay::DecisionRecord);
        let tiles3: Vec<TileOut> = par_map_indexed(nthreads, n, |i| {
            // Mark this worker as running a per-tile sub-encode so the core's SB
            // wavefront stays off — tiles already parallelise across this map.
            let _tsg = replay::TileSubencodeGuard::enter();
            let (x0, y0, tw, th) = specs[i];
            let ty = extract_subplane(yf, stride, x0, y0, tw, th);
            let tu = extract_subplane(cbf, stride, x0, y0, tw, th);
            let tv = extract_subplane(crf, stride, x0, y0, tw, th);
            let tile_ref = if has_ref {
                Some(tiling::TileRefCtx {
                    planes: std::sync::Arc::clone(&full_ref),
                    luma_stride: stride,
                    chroma_stride: stride, // 4:4:4 chroma is full-resolution
                    x0,
                    y0,
                })
            } else {
                None
            };
            let core_planes = super::lossy::Core444Planes {
                y: &ty,
                u: &tu,
                v: &tv,
            };
            let mut rec = replay::DecisionRecord::new();
            let mut enc = self.encode_444_core(
                core_planes,
                tw,
                th,
                tile_ref.as_ref(),
                None,
                replay::DecideMode::Capture(&mut rec),
            );
            let recon = std::mem::take(&mut enc.recon);
            (enc.finish(), recon, rec)
        });
        let mut records: Vec<replay::DecisionRecord> = Vec::with_capacity(n);
        let mut tiles: Vec<(Vec<u8>, Vec<Vec<f32>>)> = Vec::with_capacity(n);
        for (b, r, rec) in tiles3 {
            records.push(rec);
            tiles.push((b, r));
        }
        let recon = stitch_tile_recon_444(&tiles, &specs, sig_w, sig_h);
        // Per-block CDEF two-pass. Pass 1 (above) captured each tile's recon; stitch
        // it and decide the frame-level per-SB grid. When any SB benefits, pass 2
        // re-encodes every tile with its slice of the grid (each tile independent,
        // so the grid is re-indexed to the tile's local superblock coordinates).
        let frame_dec = if cdef_search && recon.len() >= 3 && !recon[0].is_empty() {
            cdef_est::search_per_block(
                &recon,
                &[yf.clone(), cbf.clone(), crf.clone()],
                stride,
                sig_h,
                stride,
                sig_h,
                0,
                0,
                true,
                self.base_q_idx,
                self.bit_depth,
            )
        } else {
            None
        };
        let cfg_owned;
        let (tiles_bytes, config): (Vec<Vec<u8>>, &Config) = if let Some(dec) = &frame_dec {
            let fcols = dec.sb_cols;
            let bytes: Vec<Vec<u8>> = par_map_indexed(nthreads, n, |i| {
                let (x0, y0, tw, th) = specs[i];
                // Slice the frame grid into this tile's local superblock grid.
                let (sbx0, sby0) = (x0 / 64, y0 / 64);
                let (lcols, lrows) = (tw.div_ceil(64), th.div_ceil(64));
                let mut lg = vec![0u8; lcols * lrows];
                for lr in 0..lrows {
                    for lc in 0..lcols {
                        let fidx = (sby0 + lr) * fcols + (sbx0 + lc);
                        lg[lr * lcols + lc] = dec.grid.get(fidx).copied().unwrap_or(0);
                    }
                }
                let tdec = cdef_est::CdefDecision {
                    damping: dec.damping,
                    y_str: dec.y_str,
                    uv_str: dec.uv_str,
                    grid: lg,
                    sb_cols: lcols,
                };
                let ty = extract_subplane(yf, stride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, stride, x0, y0, tw, th);
                let tv = extract_subplane(crf, stride, x0, y0, tw, th);
                let tile_ref = if has_ref {
                    Some(tiling::TileRefCtx {
                        planes: std::sync::Arc::clone(&full_ref),
                        luma_stride: stride,
                        chroma_stride: stride,
                        x0,
                        y0,
                    })
                } else {
                    None
                };
                let core_planes = super::lossy::Core444Planes {
                    y: &ty,
                    u: &tu,
                    v: &tv,
                };
                let cur = replay::DecisionCursor::new(&records[i]);
                self.encode_444_core(
                    core_planes,
                    tw,
                    th,
                    tile_ref.as_ref(),
                    Some(&tdec),
                    replay::DecideMode::Replay(cur),
                )
                .finish()
            });
            let mut c = config.clone();
            c.cdef = Some((dec.y_str, dec.uv_str, dec.damping));
            c.cdef_per_block = true;
            cfg_owned = c;
            (bytes, &cfg_owned)
        } else {
            (tiles.iter().map(|(b, _)| b.clone()).collect(), config)
        };
        self.capture_recon.store(prev_capture, Relaxed);
        let mut frame = assemble_multitile(
            &MultitileAssembly {
                config,
                coded_width: sig_w,
                coded_height: sig_h,
                display_width: width,
                display_height: height,
                color,
                log2_cols: log2c,
                log2_rows: log2r,
                bit_depth: self.bit_depth,
                chroma_format: ChromaFormat::Yuv444,
            },
            &tiles_bytes,
        );
        frame.recon = recon;
        frame
    }
}

/// Reassemble a full-frame `(Y,U,V)` recon at `sig_w` stride from per-tile recon
/// planes (4:4:4: chroma is full-resolution, same geometry as luma). Returns
/// empty when no tile captured recon.
fn stitch_tile_recon_444(
    tiles: &[(Vec<u8>, Vec<Vec<f32>>)],
    specs: &[(usize, usize, usize, usize)],
    sig_w: usize,
    sig_h: usize,
) -> Vec<Vec<f32>> {
    if tiles.iter().all(|(_, r)| r.len() < 3) {
        return Vec::new();
    }
    let mut planes = vec![vec![0f32; sig_w * sig_h]; 3];
    for ((_, recon), &(x0, y0, tw, th)) in tiles.iter().zip(specs.iter()) {
        if recon.len() < 3 {
            continue;
        }
        let tpw = sb_align(tw);
        for (pi, out) in planes.iter_mut().enumerate() {
            for r in 0..th {
                let src = &recon[pi][r * tpw..r * tpw + tw];
                out[(y0 + r) * sig_w + x0..(y0 + r) * sig_w + x0 + tw].copy_from_slice(src);
            }
        }
    }
    planes
}
