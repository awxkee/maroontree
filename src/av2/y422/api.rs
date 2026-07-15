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
use crate::av2::tiling::TiledEncodeRequest;
use crate::av2::tiling::tile_grid_for;

impl Av2Encoder {
    /// Encode a 4:2:2 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height` (half width, full height). Luma is four 32×32 TUs per
    /// superblock; each chroma plane is one 32-wide × 64-tall (TX_32X64) transform per
    /// superblock. `width` must be even. Chroma coefficient coding is identical to 4:4:4
    /// (avm codes TX_32X64 with the 32×32 scan and TX_64X64 entropy context); only the
    /// basis is the rectangular `chroma422` set.
    pub fn encode_yuv422<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        let width = planar_image.width;
        let height = planar_image.height;
        planar_image.validate_422()?;
        if self.base_q_idx == 0 {
            if let Some((log2c, log2r)) =
                tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
            {
                return Ok(self.encode_422_lossless_tiled(planar_image, color, log2c, log2r));
            }
            return self.encode_yuv422_lossless(planar_image, color, self.threads);
        }
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let yf = to_plane(&planar_image.planes[0]);
        let cbf = to_plane(&planar_image.planes[1]);
        let crf = to_plane(&planar_image.planes[2]);
        let core_planes = super::lossy::Core422Planes {
            y: &yf,
            u: &cbf,
            v: &crf,
        };
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I422);
        let multithreaded = Self::resolve_threads(self.threads) > 1;
        let video = self.video_mode.load(std::sync::atomic::Ordering::Relaxed);
        // 4:2:2 default stays on the tile-parallel bitstream: at q40 the single-tile
        // wavefront's Capture pass can't fire (the AQ-grid model doesn't reproduce
        // padded base_q<=38 mixed accumulation), so the per-block-CDEF 2-pass runs
        // *serially* twice — ~2.6x slower than multi-tile, which parallelises every
        // tile in one pass.
        let _ = (multithreaded, video);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_422_tiled(&TiledEncodeRequest {
                y: &yf,
                u: &cbf,
                v: &crf,
                width,
                height,
                config: &config,
                color,
                log2_cols: log2c,
                log2_rows: log2r,
                threads: self.threads,
            }));
        }
        // Per-block CDEF two-pass: pass 1 captures the whole-64 luma winners, pass 2
        // replays them (skips `encode_luma_sb`) while emitting the grid.
        let mut rec = crate::av2::replay::DecisionRecord::new();
        let mut enc = self.encode_422_core(
            core_planes,
            width,
            height,
            None,
            replay::DecideMode::Capture(&mut rec),
        );
        if self.tune.cdef
            && self.base_q_idx != 0
            && let Some(dec) = enc.cdef_decided.clone()
        {
            let cur = replay::DecisionCursor::new(&rec);
            enc = self.encode_422_core(
                core_planes,
                width,
                height,
                Some(&dec),
                replay::DecideMode::Replay(cur),
            );
        } else if multithreaded && !video {
            // The parallel-decide Capture pass emits nothing (per-cell throwaway
            // encoders); the serial Replay produces the real bitstream from the record.
            let cur = replay::DecisionCursor::new(&rec);
            enc = self.encode_422_core(
                core_planes,
                width,
                height,
                None,
                replay::DecideMode::Replay(cur),
            );
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:2:2 AV2. Converts RGB→YCbCr and downsamples
    /// chroma horizontally with a 2-tap box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_422<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        // q0 (lossless) is honored: the RGB→YCbCr422 conversion below is
        // independent of the quantizer, and `encode_yuv422` routes base_q_idx==0
        // to its lossless coder. The 422 subsampling is inherently lossy, so this
        // is "lossless coding of the 422 representation", not RGB-lossless (use
        // 4:4:4 for that). Matches `encode_image_444`, which also handles q0.
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = (1i32 << (bd - 1)) << Q;
        let mx_i = maxv;
        let cw = w.div_ceil(2);
        let mut y = vec![0i32; w * h];
        let mut fcb_q = vec![0i32; w * h];
        let mut fcr_q = vec![0i32; w * h];
        for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
            .iter_mut()
            .zip(fcb_q.iter_mut())
            .zip(fcr_q.iter_mut())
            .zip(img.planes[2].iter())
            .zip(img.planes[0].iter())
            .zip(img.planes[1].iter())
        {
            let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
            *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
            *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
            *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
        }
        const HALF_AVG: i32 = 1 << Q;
        let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);
        let pair_sum = |pair: &[i32]| {
            let first = *pair.first().expect("chroma pair is non-empty");
            first + pair.get(1).copied().unwrap_or(first)
        };
        let dst_rows = cb.chunks_exact_mut(cw).zip(cr.chunks_exact_mut(cw));
        let src_rows = fcb_q.chunks_exact(w).zip(fcr_q.chunks_exact(w));
        for ((cb_row, cr_row), (fcb_row, fcr_row)) in dst_rows.zip(src_rows) {
            let dst = cb_row.iter_mut().zip(cr_row);
            let src = fcb_row.chunks(2).zip(fcr_row.chunks(2));
            for ((cb, cr), (fcb, fcr)) in dst.zip(src) {
                *cb = ((pair_sum(fcb) + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
                *cr = ((pair_sum(fcr) + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            }
        }
        self.encode_yuv422(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr, Vec::new()],
            },
            color,
        )
    }
}
