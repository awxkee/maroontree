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
    /// Encode a 4:4:4 YCbCr still. `y`, `cb`, `cr` are full-resolution
    /// (`width × height`). Luma is four 32x32 transform units per 64x64 superblock;
    /// each chroma plane is one 64x64 transform per superblock.
    pub fn encode_yuv444<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        planar_image.validate_444()?;
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        if self.base_q_idx == 0 {
            if let Some((log2c, log2r)) =
                tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
            {
                return Ok(self.encode_444_lossless_tiled(planar_image, color, log2c, log2r));
            }
            return self.encode_yuv444_lossless(planar_image, color, self.threads);
        }
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (yf, cbf, crf) = (to_plane(y), to_plane(cb), to_plane(cr));
        let core_planes = super::lossy::Core444Planes {
            y: &yf,
            u: &cbf,
            v: &crf,
        };
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I444);
        let multithreaded = Self::resolve_threads(self.threads) > 1;
        let video = self.video_mode.load(std::sync::atomic::Ordering::Relaxed);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_444_tiled(&TiledEncodeRequest {
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
        // Per-block CDEF two-pass: pass 1 builds recon and, as a side computation,
        // decides the per-SB strength grid; pass 2 re-emits with the grid signalled.
        // When no superblock benefits, pass 1's bitstream is final (single pass).
        // Staged pipeline: pass 1 decides + captures each whole-64 winner into `rec`;
        // pass 2 (CDEF) replays those winners — skipping their search — instead of
        // re-running the full RD. Non-converted paths (split/native) still re-search in
        // pass 2 (they don't touch the record, so the cursor stays aligned).
        let mut rec = replay::DecisionRecord::new();
        let mut enc = self.encode_444_core(
            core_planes,
            width,
            height,
            None,
            None,
            crate::av2::replay::DecideMode::Capture(&mut rec),
        );
        if self.tune.cdef
            && self.base_q_idx != 0
            && let Some(dec) = enc.cdef_decided.clone()
        {
            let cur = crate::av2::replay::DecisionCursor::new(&rec);
            enc = self.encode_444_core(
                core_planes,
                width,
                height,
                None,
                Some(&dec),
                crate::av2::replay::DecideMode::Replay(cur),
            );
        } else if multithreaded && !video {
            // The parallel-decide Capture pass emits nothing (per-cell throwaway
            // encoders); the serial Replay produces the real bitstream from the record.
            // Skipped for video (`video_mode`): the core disables the wavefront there, so
            // the Capture pass already emitted the real serial bitstream.
            let cur = crate::av2::replay::DecisionCursor::new(&rec);
            enc = self.encode_444_core(
                core_planes,
                width,
                height,
                None,
                None,
                crate::av2::replay::DecideMode::Replay(cur),
            );
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:4:4 AV2. Converts RGB→YCbCr internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_444<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let bd = img.bit_depth;
        let maxv = (1i32 << bd.bits()) - 1;
        let off_q = (1i32 << (bd.bits() - 1)) << Q;
        let mx_i = maxv;
        let n = img.planes[0].len();
        // Lossless (q0) codes the R/G/B planes verbatim through the identity
        // (GBR) mapping — no RGB→YCbCr conversion, so the result is bit-exact.
        // The bitstream must therefore signal matrix_coefficients = Identity(0);
        // signaling the caller's YCbCr matrix would make the decoder apply an
        // inverse color transform to raw RGB and reconstruct garbage. Primaries,
        // transfer and range still describe the source, so only the matrix is
        // overridden. (The lossy branch below converts to real YCbCr and keeps
        // the caller's matrix.)
        let mut effective_color = *color;
        let (y, cb, cr) = if self.base_q_idx == 0 {
            effective_color.matrix = crate::color::MatrixCoefficients::Identity;
            let y = img.planes[0].iter().map(|x| x.to_i32()).collect::<Vec<_>>();
            let cb = img.planes[1].iter().map(|x| x.to_i32()).collect::<Vec<_>>();
            let cr = img.planes[2].iter().map(|x| x.to_i32()).collect::<Vec<_>>();
            (y, cb, cr)
        } else {
            let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
            for (((((yv, cbv), crv), &rr), &gg), &bb) in y
                .iter_mut()
                .zip(cb.iter_mut())
                .zip(cr.iter_mut())
                .zip(img.planes[2].iter())
                .zip(img.planes[0].iter())
                .zip(img.planes[1].iter())
            {
                let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
                *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
                *cbv = ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
                *crv = ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
            }
            (y, cb, cr)
        };
        self.encode_yuv444(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr, Vec::new()],
            },
            &effective_color,
        )
    }
}
