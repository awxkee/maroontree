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
use super::core::CcsoPrecomp;
use crate::av2::tiling::TiledEncodeRequest;
use crate::av2::tiling::tile_grid_for;

impl Av2Encoder {
    /// Encode a 4:2:0 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height/2`. Luma is four 32x32 TUs per superblock; each chroma plane
    /// is one 32x32 transform per superblock. `width`/`height` must be even.
    pub fn encode_yuv420<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        let width = planar_image.width;
        let height = planar_image.height;
        validate_dims(width as u32, height as u32)?;
        planar_image.validate_420()?;
        if self.base_q_idx == 0 {
            if let Some((log2c, log2r)) =
                tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
            {
                return Ok(self.encode_420_lossless_tiled(planar_image, color, log2c, log2r));
            }
            return self.encode_yuv420_lossless(planar_image, color, self.threads);
        }
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let yf = to_plane(&planar_image.planes[0]);
        let cbf = to_plane(&planar_image.planes[1]);
        let crf = to_plane(&planar_image.planes[2]);
        let (pw, ph) = (sb_align(width), sb_align(height));
        let config = self.config(Layout::I420);
        // Respect the caller's configured tile grid. In particular, a latency-tuned
        // caller may deliberately request many small tiles; silently replacing that
        // grid with the whole-frame wavefront adds its serial record-replay tail and
        // regresses small/medium still latency even when it wins on very large frames.
        let multithreaded = Self::resolve_threads(self.threads) > 1;
        let video = self.video_mode.load(std::sync::atomic::Ordering::Relaxed);
        if let Some((log2c, log2r)) =
            tile_grid_for(self.tune.tile_cols, self.tune.tile_rows, width, height)
        {
            return Ok(self.encode_420_tiled(&TiledEncodeRequest {
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
        // CCSO with per-SB RD (Phase 3): a first pass builds the recon, searches
        // the filter, and decides per-SB on/off; a second pass re-emits with those
        // decisions (gated filter + correct flags). When CCSO is off, or the
        // decision pass turned both planes off, the single pass is used directly.
        // Staged replay: pass 1 captures each whole-64 luma mode-search winner;
        // pass 2 (CDEF/CCSO) replays them, skipping `encode_luma_sb`.
        let mut rec = replay::DecisionRecord::new();
        let mut enc = self.encode_420_core(
            &yf,
            &cbf,
            &crf,
            width,
            height,
            None,
            None,
            None,
            replay::DecideMode::Capture(&mut rec),
        );
        // Pass 1 emits a complete CCSO-off / CDEF-off bitstream and, as a side
        // computation, derives both the CCSO per-SB decision and the per-block CDEF
        // grid from its reconstruction. Pass 2 re-emits with whichever were chosen;
        // when neither fires (common at high quality / flat content) pass 1 is final.
        let ccso_on = self.tune.ccso
            && self.base_q_idx != 0
            && (enc.ccso_decided_u.is_some() || enc.ccso_decided_v.is_some());
        let cdef_dec = enc.cdef_decided.clone();
        if ccso_on || cdef_dec.is_some() {
            let pre = if ccso_on {
                Some(CcsoPrecomp {
                    u: enc.ccso_decided_u.take(),
                    v: enc.ccso_decided_v.take(),
                    sb_cols: enc.ccso_sb_cols_out,
                })
            } else {
                None
            };
            let cur = replay::DecisionCursor::new(&rec);
            enc = self.encode_420_core(
                &yf,
                &cbf,
                &crf,
                width,
                height,
                pre.as_ref(),
                None,
                cdef_dec.as_ref(),
                crate::av2::replay::DecideMode::Replay(cur),
            );
        } else if multithreaded && !video {
            // The parallel-decide Capture pass emits nothing meaningful (per-cell
            // throwaway encoders); the serial Replay produces the real bitstream from
            // the merged record. Skipped for video (`video_mode`): the core disables the
            // wavefront there, so the Capture pass already emitted the real serial stream.
            let cur = replay::DecisionCursor::new(&rec);
            enc = self.encode_420_core(
                &yf,
                &cbf,
                &crf,
                width,
                height,
                None,
                None,
                None,
                replay::DecideMode::Replay(cur),
            );
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode an RGB image to 4:2:0 AV2. Converts RGB→YCbCr and downsamples
    /// chroma with a 2×2 box filter internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383), if
    /// `img.bit_depth` is not 8, 10, or 12, or if `base_q_idx` is 0 (use the
    /// lossless path for that).
    pub fn encode_image_420<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &Cicp,
    ) -> Result<Av2Frame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        // q0 (lossless) is honored: the RGB→YCbCr420 conversion below is
        // independent of the quantizer, and `encode_yuv420` routes base_q_idx==0
        // to its lossless coder. The 420 subsampling is inherently lossy, so this
        // is "lossless coding of the 420 representation", not RGB-lossless (use
        // 4:4:4 for that). Matches `encode_image_444`, which also handles q0.
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = (1i32 << (bd - 1)) << Q;
        let mx_i = maxv;
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
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
        const HALF_AVG: i32 = 1 << (Q + 1); // rounding bias for >> (Q+2)
        let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);
        let pair_sum = |pair: &[i32]| {
            let first = *pair.first().expect("chroma pair is non-empty");
            first + pair.get(1).copied().unwrap_or(first)
        };
        let dst_rows = cb.chunks_exact_mut(cw).zip(cr.chunks_exact_mut(cw));
        let src_rows = fcb_q.chunks(2 * w).zip(fcr_q.chunks(2 * w));
        for ((cb_row, cr_row), (fcb_rows, fcr_rows)) in dst_rows.zip(src_rows) {
            let (fcb_top, fcb_bottom) = if fcb_rows.len() > w {
                fcb_rows.split_at(w)
            } else {
                (fcb_rows, fcb_rows)
            };
            let (fcr_top, fcr_bottom) = if fcr_rows.len() > w {
                fcr_rows.split_at(w)
            } else {
                (fcr_rows, fcr_rows)
            };
            let dst = cb_row.iter_mut().zip(cr_row);
            let src = fcb_top
                .chunks(2)
                .zip(fcb_bottom.chunks(2))
                .zip(fcr_top.chunks(2).zip(fcr_bottom.chunks(2)));
            for ((cb, cr), ((fcb_top, fcb_bottom), (fcr_top, fcr_bottom))) in dst.zip(src) {
                let cb_sum = pair_sum(fcb_top) + pair_sum(fcb_bottom);
                let cr_sum = pair_sum(fcr_top) + pair_sum(fcr_bottom);
                *cb = ((cb_sum + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
                *cr = ((cr_sum + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            }
        }
        self.encode_yuv420(
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
