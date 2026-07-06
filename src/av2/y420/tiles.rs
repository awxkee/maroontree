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
use std::sync::atomic::Ordering;

impl Av2Encoder {
    pub(super) fn encode_420_lossless_tiled<T: Pixel>(
        &self,
        image: &PlanarImage<T>,
        color: &Cicp,
        log2c: usize,
        log2r: usize,
    ) -> Av2Frame {
        let (width, height) = (image.width, image.height);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph / 2);
        let planes = [
            pad_plane_pixels(&image.planes[0], width, height, pw, ph),
            pad_plane_pixels(&image.planes[1], width / 2, height / 2, pcw, pch),
            pad_plane_pixels(&image.planes[2], width / 2, height / 2, pcw, pch),
        ];
        let specs = tile_specs(pw, ph, log2c, log2r);
        let config = self.config(Layout::I420);
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
                    extract_subplane_pixels(&planes[1], pcw, x0 / 2, y0 / 2, tw / 2, th / 2),
                    extract_subplane_pixels(&planes[2], pcw, x0 / 2, y0 / 2, tw / 2, th / 2),
                    Vec::new(),
                ],
            };
            let frame = self.encode_yuv420_lossless(&tile, color, 1).unwrap();
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
                chroma_format: ChromaFormat::Yuv420,
            },
            &tiles,
        )
    }

    /// Multi-tile 4:2:0 assembly. Each tile is an independent sub-frame encode; tiles
    /// run in parallel across `threads` workers (raster order preserved). 4:2:0 chroma
    /// is half-width/half-height, so a luma tile at `(x0, y0, tw, th)` maps to chroma
    /// `(x0/2, y0/2, tw/2, th/2)` — all even because SB boundaries are multiples of 64.
    pub(super) fn encode_420_tiled(&self, request: &TiledEncodeRequest<'_>) -> Av2Frame {
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
        // See encode_444_tiled: when any edge tile isn't boundary-exact, pad the whole
        // frame SB-aligned, carve all-SB-aligned tiles, signal the padded size, and let
        // the AVIF muxer clap back to width×height. Otherwise, signal the real size.
        let native_specs = tile_specs(width, height, log2c, log2r);
        let exact = native_specs
            .iter()
            .all(|&(_, _, tw, th)| native_420_mi(tw, th).is_some());
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
                    pad_plane(cbf, width.div_ceil(2), height.div_ceil(2), pw / 2, ph / 2),
                    pad_plane(crf, width.div_ceil(2), height.div_ceil(2), pw / 2, ph / 2),
                ),
                tile_specs(pw, ph, log2c, log2r),
            )
        };
        let (yf, cbf, crf) = (&planes.0, &planes.1, &planes.2);
        let cw = cstride; // chroma plane stride (4:2:0)
        let n = specs.len();
        // Inter frames predict from the previous recon. The full-frame reference
        // is carved into per-tile sub-windows here so each independent tile encode
        // sees a reference whose geometry matches its local block coordinates. The
        // reference is stored at the same (`sig_w`) stride as this frame's source
        // planes, so the tile rectangles used for the source apply unchanged.
        let inter = self.inter_tile.load(Ordering::Relaxed);
        let full_ref: std::sync::Arc<Vec<Vec<f32>>> = if inter {
            std::sync::Arc::clone(&self.last_ref.lock().unwrap())
        } else {
            std::sync::Arc::new(Vec::new())
        };
        let has_ref = full_ref.len() >= 3 && !full_ref[0].is_empty();
        // Independent per-tile encodes on a work-stealing map (no idle static chunks).
        let nthreads = Self::resolve_threads(threads).min(n.max(1));
        // Per-block CDEF: capture per-tile recon so pass 1 can stitch + decide the grid.
        use std::sync::atomic::Ordering::Relaxed;
        let cdef_search = self.tune.cdef;
        let prev_capture = self.capture_recon.load(Relaxed);
        if cdef_search {
            self.capture_recon.store(true, Relaxed);
        }
        let mk_tile_ref = |x0: usize, y0: usize| {
            if has_ref {
                Some(tiling::TileRefCtx {
                    planes: std::sync::Arc::clone(&full_ref),
                    luma_stride: lstride,
                    chroma_stride: cw,
                    x0,
                    y0,
                })
            } else {
                None
            }
        };
        // Staged replay (per tile): pass 1 captures the whole-64 luma winners.
        type TileOut420 = (Vec<u8>, Vec<Vec<f32>>, replay::DecisionRecord);
        let tiles3: Vec<TileOut420> = par_map_indexed(nthreads, n, |i| {
            // Mark this worker as running a per-tile sub-encode so the core's SB
            // wavefront stays off — tiles already parallelise across this map.
            let _tsg = replay::TileSubencodeGuard::enter();
            let (x0, y0, tw, th) = specs[i];
            let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
            let tu = extract_subplane(cbf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
            let tv = extract_subplane(crf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
            let tile_ref = mk_tile_ref(x0, y0);
            let mut rec = replay::DecisionRecord::new();
            let mut enc = self.encode_420_core(
                &ty,
                &tu,
                &tv,
                tw,
                th,
                None,
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
        self.capture_recon.store(prev_capture, Relaxed);
        // Stitch the per-tile recon back into a full-frame plane set (also used by the
        // video DPB refresh). Each tile's recon is at its own padded stride.
        let mut recon = stitch_tile_recon_420(&tiles, &specs, sig_w, sig_h);
        let frame_dec = if cdef_search && recon.len() >= 3 && !recon[0].is_empty() {
            cdef_est::search_per_block(
                &recon,
                &[yf.clone(), cbf.clone(), crf.clone()],
                lstride,
                sig_h,
                cw,
                sig_h.div_ceil(2),
                1,
                1,
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
                let tdec = cdef_est::CdefDecision {
                    damping: dec.damping,
                    y_str: dec.y_str,
                    uv_str: dec.uv_str,
                    grid: lg,
                    sb_cols: lcols,
                };
                let ty = extract_subplane(yf, lstride, x0, y0, tw, th);
                let tu = extract_subplane(cbf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
                let tv = extract_subplane(crf, cw, x0 / 2, y0 / 2, tw.div_ceil(2), th.div_ceil(2));
                let tile_ref = mk_tile_ref(x0, y0);
                let cur = crate::av2::replay::DecisionCursor::new(&records[i]);
                self.encode_420_core(
                    &ty,
                    &tu,
                    &tv,
                    tw,
                    th,
                    None,
                    tile_ref.as_ref(),
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
                chroma_format: ChromaFormat::Yuv420,
            },
            &tiles_bytes,
        );
        if let Some(decision) = &frame_dec {
            cdef_est::apply_per_block(
                &mut recon,
                sig_w,
                sig_h,
                cw,
                sig_h.div_ceil(2),
                1,
                1,
                true,
                decision,
                self.bit_depth,
            );
        }
        frame.recon = recon;
        frame
    }
}

/// Reassemble a full-frame `(Y,U,V)` recon (at `sig_w` luma stride) from the
/// per-tile recon planes. Each tile's recon is stored at its own padded stride
/// `sb_align(tw)`; only the `tw x th` (chroma `tw/2 x th/2`) top-left region is
/// valid and is copied to the tile's frame position. Returns empty when no tile
/// captured recon (still image / intra frame).
fn stitch_tile_recon_420(
    tiles: &[(Vec<u8>, Vec<Vec<f32>>)],
    specs: &[(usize, usize, usize, usize)],
    sig_w: usize,
    sig_h: usize,
) -> Vec<Vec<f32>> {
    if tiles.iter().all(|(_, r)| r.len() < 3) {
        return Vec::new();
    }
    let cwf = sig_w.div_ceil(2);
    let chf = sig_h.div_ceil(2);
    let mut y = vec![0f32; sig_w * sig_h];
    let mut u = vec![0f32; cwf * chf];
    let mut v = vec![0f32; cwf * chf];
    for ((_, recon), &(x0, y0, tw, th)) in tiles.iter().zip(specs.iter()) {
        if recon.len() < 3 {
            continue;
        }
        let tpw = sb_align(tw);
        let tcw = tpw / 2;
        let (cx, cy) = (x0 / 2, y0 / 2);
        let (ctw, cth) = (tw.div_ceil(2), th.div_ceil(2));
        for r in 0..th {
            let src = &recon[0][r * tpw..r * tpw + tw];
            y[(y0 + r) * sig_w + x0..(y0 + r) * sig_w + x0 + tw].copy_from_slice(src);
        }
        for r in 0..cth {
            let su = &recon[1][r * tcw..r * tcw + ctw];
            let sv = &recon[2][r * tcw..r * tcw + ctw];
            u[(cy + r) * cwf + cx..(cy + r) * cwf + cx + ctw].copy_from_slice(su);
            v[(cy + r) * cwf + cx..(cy + r) * cwf + cx + ctw].copy_from_slice(sv);
        }
    }
    vec![y, u, v]
}

#[cfg(test)]
mod tile_ref_tests {
    use super::*;
    use crate::av2::tiling::{extract_subplane, tile_specs};

    // Source sub-planes and stitched recon use `extract_subplane`; this pins that
    // a tile's local coordinates map to the correct frame coordinates (the source
    // carve and the recon stitch must be exact inverses). Inter *reference* reads
    // no longer crop — they index the full frame at frame-absolute coords (see the
    // mc.rs `tile_mc_reads_frame_absolute_reference` test).
    #[test]
    fn tile_subplane_extract_is_frame_aligned() {
        let (fw, fh) = (128usize, 128usize);
        let frame: Vec<f32> = (0..fw * fh).map(|i| i as f32).collect();
        let specs = tile_specs(fw, fh, 1, 1);
        assert_eq!(specs.len(), 4);
        for &(x0, y0, tw, th) in &specs {
            let tref = extract_subplane(&frame, fw, x0, y0, tw, th);
            for ly in 0..th {
                for lx in 0..tw {
                    assert_eq!(
                        tref[ly * tw + lx],
                        frame[(y0 + ly) * fw + (x0 + lx)],
                        "tile ({x0},{y0}) local ({lx},{ly}) misaligned"
                    );
                }
            }
        }
    }

    // Stitching per-tile recon must reproduce the whole frame at frame stride, so
    // the DPB reference for the next inter frame is hole-free and correctly placed.
    #[test]
    fn stitched_recon_reproduces_frame() {
        let (fw, fh) = (128usize, 128usize);
        let specs = tile_specs(fw, fh, 1, 1);
        let mk = |x0: usize, y0: usize, tw: usize, th: usize, plane: usize| {
            let tpw = sb_align(tw);
            let tcw = tpw / 2;
            let (w, h, stride) = if plane == 0 {
                (tw, th, tpw)
            } else {
                (tw.div_ceil(2), th.div_ceil(2), tcw)
            };
            let rows = if plane == 0 { th } else { th.div_ceil(2) };
            let mut v = vec![0f32; stride * rows];
            for ly in 0..h {
                for lx in 0..w {
                    let (fx, fy) = if plane == 0 {
                        (x0 + lx, y0 + ly)
                    } else {
                        (x0 / 2 + lx, y0 / 2 + ly)
                    };
                    v[ly * stride + lx] = (plane * 1_000_000 + fy * 1000 + fx) as f32;
                }
            }
            v
        };
        let tiles: Vec<(Vec<u8>, Vec<Vec<f32>>)> = specs
            .iter()
            .map(|&(x0, y0, tw, th)| {
                (
                    Vec::new(),
                    vec![
                        mk(x0, y0, tw, th, 0),
                        mk(x0, y0, tw, th, 1),
                        mk(x0, y0, tw, th, 2),
                    ],
                )
            })
            .collect();
        let recon = stitch_tile_recon_420(&tiles, &specs, fw, fh);
        assert_eq!(recon.len(), 3);
        for y in 0..fh {
            for x in 0..fw {
                assert_eq!(recon[0][y * fw + x], (y * 1000 + x) as f32, "Y ({x},{y})");
            }
        }
        let (cwf, chf) = (fw / 2, fh / 2);
        for y in 0..chf {
            for x in 0..cwf {
                assert_eq!(recon[1][y * cwf + x], (1_000_000 + y * 1000 + x) as f32);
                assert_eq!(recon[2][y * cwf + x], (2_000_000 + y * 1000 + x) as f32);
            }
        }
    }
}
