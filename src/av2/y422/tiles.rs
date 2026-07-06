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
    pub(super) fn encode_422_lossless_tiled<T: Pixel>(
        &self,
        image: &PlanarImage<T>,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
    ) -> Av2Frame {
        let (width, height) = (image.width, image.height);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let pcw = pw / 2;
        let planes = [
            pad_plane_pixels(&image.planes[0], width, height, pw, ph),
            pad_plane_pixels(&image.planes[1], width / 2, height, pcw, ph),
            pad_plane_pixels(&image.planes[2], width / 2, height, pcw, ph),
        ];
        let specs = tile_specs(pw, ph, log2c, log2r);
        let config = self.config(Layout::I422);
        // Tiles are independent sub-frame encodes; parallelise across them (each tile
        // single-threaded). `par_map_indexed` preserves index order, so the assembled
        // stream is byte-identical to the previous serial loop. Mirrors the 4:4:4 path.
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
                    extract_subplane_pixels(&planes[1], pcw, x0 / 2, y0, tw / 2, th),
                    extract_subplane_pixels(&planes[2], pcw, x0 / 2, y0, tw / 2, th),
                    Vec::new(),
                ],
            };
            let frame = self.encode_yuv422_lossless(&tile, color, 1).unwrap();
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
                chroma_format: ChromaFormat::Yuv422,
            },
            &tiles,
        )
    }

    /// Multi-tile 4:2:2 assembly. Each tile is an independent sub-frame encode; tiles
    /// run in parallel across `threads` workers (raster order preserved). 4:2:2 chroma
    /// is half-width/full-height, so a luma tile at `(x0, tw)` maps to chroma
    /// `(x0/2, tw/2)` — both even because SB boundaries and 4:2:2 width are even.
    pub(super) fn encode_422_tiled(&self, request: &TiledEncodeRequest<'_>) -> Av2Frame {
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
        // See encode_444_tiled: pad the whole frame SB-aligned (one frame-level clap)
        // when any edge tile isn't boundary-exact; otherwise signal the real size.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| native_422_mi(tw, th).is_some());
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (sig_w, sig_h, lstride, cstride, planes, specs) = if exact {
            (
                width,
                height,
                width,
                width.div_ceil(2),
                (yf.to_vec(), cbf.to_vec(), crf.to_vec()),
                native_specs,
            )
        } else {
            (
                pw,
                ph,
                pw,
                pw / 2,
                (
                    pad_plane(yf, width, height, pw, ph),
                    pad_plane(cbf, width.div_ceil(2), height, pw / 2, ph),
                    pad_plane(crf, width.div_ceil(2), height, pw / 2, ph),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let cw = cstride; // chroma plane stride (4:2:2)
        let n = specs.len();
        // Independent per-tile encodes on a work-stealing map (no idle static chunks).
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        // Per-block CDEF: capture per-tile recon so pass 1 can stitch + decide the grid.
        use std::sync::atomic::Ordering::Relaxed;
        let cdef_search = self.tune.cdef;
        let prev_capture = self.capture_recon.load(Relaxed);
        if cdef_search {
            self.capture_recon.store(true, Relaxed);
        }
        // Staged replay (per tile): pass 1 captures the whole-64 luma winners.
        type TileOut422 = (Vec<u8>, Vec<Vec<f32>>, crate::av2::replay::DecisionRecord);
        let tiles3: Vec<TileOut422> = par_map_indexed(nthreads, n, |i| {
            // Mark this worker as running a per-tile sub-encode so the core's SB
            // wavefront stays off — tiles already parallelise across this map.
            let _tsg = crate::av2::replay::TileSubencodeGuard::enter();
            let (x0, y0, tw, th) = specs[i];
            let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
            let tu = extract_subplane(cbf, cw, x0 / 2, y0, tw.div_ceil(2), th);
            let tv = extract_subplane(crf, cw, x0 / 2, y0, tw.div_ceil(2), th);
            let mut rec = crate::av2::replay::DecisionRecord::new();
            let mut enc = self.encode_422_core(
                super::lossy::Core422Planes {
                    y: &ty,
                    u: &tu,
                    v: &tv,
                },
                tw,
                th,
                None,
                crate::av2::replay::DecideMode::Capture(&mut rec),
            );
            let recon = std::mem::take(&mut enc.recon);
            (enc.finish(), recon, rec)
        });
        let mut records: Vec<crate::av2::replay::DecisionRecord> = Vec::with_capacity(n);
        let mut tiles: Vec<(Vec<u8>, Vec<Vec<f32>>)> = Vec::with_capacity(n);
        for (b, r, rec) in tiles3 {
            records.push(rec);
            tiles.push((b, r));
        }
        self.capture_recon.store(prev_capture, Relaxed);
        let recon = stitch_tile_recon_422(&tiles, &specs, sig_w, sig_h, cw);
        let frame_dec = if cdef_search && recon.len() >= 3 && !recon[0].is_empty() {
            crate::av2::cdef_est::search_per_block(
                &recon,
                &[yf.clone(), cbf.clone(), crf.clone()],
                lstride,
                sig_h,
                cw,
                sig_h,
                1,
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
                let (sbx0, sby0) = (x0 / 64, y0 / 64);
                let (lcols, lrows) = (tw.div_ceil(64), th.div_ceil(64));
                let mut lg = vec![0u8; lcols * lrows];
                for lr in 0..lrows {
                    for lc in 0..lcols {
                        let fidx = (sby0 + lr) * fcols + (sbx0 + lc);
                        lg[lr * lcols + lc] = dec.grid.get(fidx).copied().unwrap_or(0);
                    }
                }
                let tdec = crate::av2::cdef_est::CdefDecision {
                    damping: dec.damping,
                    y_str: dec.y_str,
                    uv_str: dec.uv_str,
                    grid: lg,
                    sb_cols: lcols,
                };
                let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                let tv = extract_subplane(crf, cw, x0 / 2, y0, tw.div_ceil(2), th);
                let cur = crate::av2::replay::DecisionCursor::new(&records[i]);
                self.encode_422_core(
                    super::lossy::Core422Planes {
                        y: &ty,
                        u: &tu,
                        v: &tv,
                    },
                    tw,
                    th,
                    Some(&tdec),
                    crate::av2::replay::DecideMode::Replay(cur),
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
        assemble_multitile(
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
                chroma_format: ChromaFormat::Yuv422,
            },
            &tiles_bytes,
        )
    }
}

/// Reassemble a full-frame `(Y,U,V)` recon from per-tile recon planes for 4:2:2
/// (chroma is half-width, full-height). Luma frame stride `sig_w`, chroma frame
/// stride `cw`. Returns empty when no tile captured recon.
fn stitch_tile_recon_422(
    tiles: &[(Vec<u8>, Vec<Vec<f32>>)],
    specs: &[(usize, usize, usize, usize)],
    sig_w: usize,
    sig_h: usize,
    cw: usize,
) -> Vec<Vec<f32>> {
    if tiles.iter().all(|(_, r)| r.len() < 3) {
        return Vec::new();
    }
    let mut planes = vec![
        vec![0f32; sig_w * sig_h],
        vec![0f32; cw * sig_h],
        vec![0f32; cw * sig_h],
    ];
    for ((_, recon), &(x0, y0, tw, th)) in tiles.iter().zip(specs.iter()) {
        if recon.len() < 3 {
            continue;
        }
        let tpw = sb_align(tw); // luma tile stride
        let tcw = tpw / 2; // chroma tile stride (tile pcw)
        let ctw = tw.div_ceil(2); // chroma cols to copy
        let cx0 = x0 / 2;
        for r in 0..th {
            let src = &recon[0][r * tpw..r * tpw + tw];
            let base = (y0 + r) * sig_w + x0;
            planes[0][base..base + tw].copy_from_slice(src);
            for pi in 1..3 {
                let cs = &recon[pi][r * tcw..r * tcw + ctw];
                let cbase = (y0 + r) * cw + cx0;
                planes[pi][cbase..cbase + ctw].copy_from_slice(cs);
            }
        }
    }
    planes
}
