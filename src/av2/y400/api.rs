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
use crate::av2::tiling::tile_grid_for;

impl Av2Encoder {
    /// Encode a 4:0:0 (monochrome / luma-only) still. `y` is `width × height`.
    /// Four 32x32 luma TUs per superblock; no chroma is coded or signaled
    /// (`has_chroma = false` ⇒ no chroma intra mode, profile 0, layout uvlc 1).
    pub fn encode_yuv400<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_400()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(&planar_image.planes[0]), width, height, pw, ph);
        let config = self.config(Layout::Monochrome);
        let core_frame = super::lossy::Core400Frame {
            yp: &yp,
            pw,
            ph,
            width,
            height,
        };

        if config.lossless {
            if let Some((log2c, log2r)) =
                tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
            {
                return Ok(self.encode_400_lossless_tiled(
                    &yp, pw, ph, width, height, &config, color, log2c, log2r,
                ));
            }
            return Ok(self.encode_yuv400_lossless(
                &yp,
                pw,
                ph,
                width,
                height,
                &config,
                color,
                self.threads,
            ));
        }

        // Respect an explicitly configured tile grid. This is the low-latency path
        // used by the simple/CLI API; a single tile retains the compression-oriented
        // whole-frame wavefront below.
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_400_tiled(&super::tiles::Tiled400Request {
                yp: &yp,
                pw,
                ph,
                width,
                height,
                config: &config,
                color,
                log2_cols: log2c,
                log2_rows: log2r,
                threads: self.threads,
            }));
        }

        // A single-tile 4:0:0 lossy encode uses the SB wavefront for multi-threaded
        // stills; video stays serial.
        let multithreaded = Self::resolve_threads(self.threads) > 1;
        let video = self.video_mode.load(std::sync::atomic::Ordering::Relaxed);

        // Per-block CDEF two-pass: pass 1 builds recon + decides the grid; pass 2
        // re-emits with it. Single pass when no superblock benefits.
        let mut rec = crate::av2::replay::DecisionRecord::new();
        let mut enc = self.encode_400_core(
            core_frame,
            None,
            crate::av2::replay::DecideMode::Capture(&mut rec),
        );
        if self.tune.cdef
            && self.base_q_idx != 0
            && let Some(dec) = enc.cdef_decided.clone()
        {
            let cur = crate::av2::replay::DecisionCursor::new(&rec);
            enc = self.encode_400_core(
                core_frame,
                Some(&dec),
                crate::av2::replay::DecideMode::Replay(cur),
            );
        } else if multithreaded && !video {
            // The parallel-decide Capture pass emits nothing meaningful (per-cell
            // throwaway encoders); the serial Replay produces the real bitstream from
            // the merged record.
            let cur = crate::av2::replay::DecisionCursor::new(&rec);
            enc = self.encode_400_core(
                core_frame,
                None,
                crate::av2::replay::DecideMode::Replay(cur),
            );
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode a luma-only (4:0:0 / monochrome) image to AV2.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_400<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
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
        )
    }
}
