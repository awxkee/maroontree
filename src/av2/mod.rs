/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
mod avif;
mod cdfs_qctx;
mod cdfx_4tx;
mod coder;
mod entropy;
mod headers;
mod helpers;
mod layout;
mod lossless;
mod proj;
mod quant;
mod tables;
mod tables_tx32;
mod wht;

use crate::av2::avif::Av2Format;
use crate::av2::cdfs_qctx::CHROMA_SKIP_V_QC;
use crate::av2::cdfx_4tx::{TXB_SKIP_TX4_Q0, V_TXB_SKIP_TX4_Q0};
use crate::av2::coder::{
    Coeff, encode_chroma_block, encode_chroma_tu4, encode_lossless_luma_sb, encode_luma_block_split,
};
use crate::av2::entropy::RangeEncoder;
use crate::av2::headers::{Config, frame_header, obu, sequence_header};
use crate::av2::helpers::{
    dc_pred, dc_pred_rect, get_residual, get_residual_rect, levels_to_coeffs, lossless_sb_tus,
    pad_plane, put_block, put_block_rect, sb_align, sb_tu_contexts, sb_tu4_chroma_skip,
    sb_tu4_contexts,
};
use crate::av2::layout::Layout;
use crate::avif::validate_dims;
use crate::err::EncodeError;
use crate::{ChromaFormat, ColorEncoding, Pixel, PlanarImage};

// Q0.13 coefficients  (value = round(f * 8192))
const Q: i32 = 13;
const HALF: i32 = 1 << (Q - 1); // 0.5 rounding bias

const Y_R: i32 = 2449; // round( 0.299    * 8192)
const Y_G: i32 = 4809; // round( 0.587    * 8192)
const Y_B: i32 = 934; // round( 0.114    * 8192)

const CB_R: i32 = -1382; // round(-0.168736 * 8192)
const CB_G: i32 = -2714; // round(-0.331264 * 8192)
const CB_B: i32 = 4096; // round( 0.5 * 8192)

const CR_R: i32 = 4096; // round( 0.5 * 8192)
const CR_G: i32 = -3430; // round(-0.418688 * 8192)
const CR_B: i32 = -666; // round(-0.081312 * 8192)

/// Maximum dimension. AV1 level 6.3 handles frames up to 35 651 584 luma
/// samples; with both axes capped here the largest possible frame is ~268 MP.
const MAX_DIM: u32 = 16_383;

pub fn get_q_ctx(q: u8) -> usize {
    if q <= 90 {
        0
    } else if q <= 140 {
        1
    } else if q <= 190 {
        2
    } else {
        3
    }
}

/// Result of an encode: the AV2 bitstream plus the metadata needed to interpret it.
pub struct AvFrame {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub bit_depth: u8,
    pub base_q_idx: u8,
    pub color: ColorEncoding,
    pub chroma_format: ChromaFormat,
}

/// A reusable still-image encoder configured for one quality.
///
/// `Av2Encoder::new(q)` loads the bundled q120 bases and rescales them to the target
/// `base_q_idx` once (see [`proj::Bases::rescaled_to_q`]); the per-superblock encode
/// then reuses that precomputed set. Lower `base_q_idx` → finer quantiser → larger,
/// higher-quality output; higher → coarser/smaller.
pub struct Av2Encoder {
    bases: proj::Bases,
    base_q_idx: u8,
    bit_depth: u8,
}

impl Av2Encoder {
    /// Build an 8-bit encoder for `base_q_idx`. Honors the `BASES` env override for
    /// the source basis file, otherwise uses the embedded q120 set, then rescales.
    pub fn new(base_q_idx: u8) -> Self {
        Self::with_bit_depth(base_q_idx, 8)
    }

    /// Build an encoder for `base_q_idx` at a given coded bit depth (8, 10 or 12).
    /// The avm quantiser step is bit-depth-independent, so only the sample range,
    /// reconstruction clamp, DC-prediction neutral and the sequence-header signalling
    /// differ; the bases are unchanged.
    pub fn with_bit_depth(base_q_idx: u8, bit_depth: u8) -> Self {
        assert!(
            matches!(bit_depth, 8 | 10 | 12),
            "bit_depth must be 8, 10 or 12, got {bit_depth}"
        );
        let mut bases = match std::env::var("BASES") {
            Ok(p) => proj::load_bases(&p),
            Err(_) => proj::default_bases(),
        }
        .rescaled_to_q(base_q_idx as u32);
        bases.set_bit_depth(bit_depth);
        Av2Encoder {
            bases,
            base_q_idx,
            bit_depth,
        }
    }

    /// The quality this encoder is configured for.
    pub fn base_q_idx(&self) -> u8 {
        self.base_q_idx
    }

    fn config(&self, layout: Layout) -> Config {
        Config {
            layout,
            base_q: self.base_q_idx as u32,
            deblock: false,
            delta_q: 0,
            tx_switchable: true,
            guided_deblock: None,
            bit_depth: self.bit_depth,
            lossless: self.base_q_idx == 0,
        }
    }

    /// DC-prediction neutral value for the first block (1 << (bit_depth-1)).
    fn dc_neutral(&self) -> f32 {
        (1u32 << (self.bit_depth - 1)) as f32
    }

    /// Encode a 4:4:4 YCbCr still. `y`, `cb`, `cr` are full-resolution
    /// (`width × height`). Luma is four 32x32 transform units per 64x64 superblock;
    /// each chroma plane is one 64x64 transform per superblock.
    pub fn encode_yuv444<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
    ) -> Result<AvFrame, EncodeError> {
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        assert_eq!(y.len(), width * height, "Y plane size mismatch");
        assert_eq!(
            cb.len(),
            width * height,
            "Cb plane must be full-resolution (4:4:4)"
        );
        assert_eq!(
            cr.len(),
            width * height,
            "Cr plane must be full-resolution (4:4:4)"
        );
        if self.base_q_idx == 0 {
            return Ok(self.encode_yuv444_lossless(y, cb, cr, width, height, color));
        }
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);

        let layout = Layout::I444;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pw * ph];
        let mut recv = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (i, &(ty, tx)) in [(0, 0), (0, 32), (32, 0), (32, 32)].iter().enumerate() {
                    let (y0, x0) = (sb_y + ty, sb_x + tx);
                    let pred = dc_pred(&recy, pw, y0, x0, 32, neutral);
                    let lev = bases
                        .luma
                        .project(&get_residual(&yp, pw, y0, x0, 32, pred), 0.0);
                    put_block(
                        &mut recy,
                        pw,
                        y0,
                        x0,
                        32,
                        &bases.luma.reconstruct(pred, &lev),
                    );
                    tus[i] = levels_to_coeffs(&lev);
                }
                let (skip_cdfs, dc_sign_ctxs) =
                    sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc);
                encode_luma_block_split(&mut enc, &tus, &skip_cdfs, &dc_sign_ctxs, 0, true);

                let predu = dc_pred(&recu, pw, sb_y, sb_x, 64, neutral);
                let levu = bases
                    .chroma444
                    .project(&get_residual(&up, pw, sb_y, sb_x, 64, predu), 0.0);
                put_block(
                    &mut recu,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &bases.chroma444.reconstruct(predu, &levu),
                );
                let predv = dc_pred(&recv, pw, sb_y, sb_x, 64, neutral);
                let levv = bases
                    .chroma444
                    .project(&get_residual(&vp, pw, sb_y, sb_x, 64, predv), 0.0);
                put_block(
                    &mut recv,
                    pw,
                    sb_y,
                    sb_x,
                    64,
                    &bases.chroma444.reconstruct(predv, &levv),
                );
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
            }
        }
        Ok(self.finish(enc, &config, pw, ph, width, height, color))
    }

    /// Encode a 4:2:0 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height/2`. Luma is four 32x32 TUs per superblock; each chroma plane
    /// is one 32x32 transform per superblock. `width`/`height` must be even.
    pub fn encode_yuv420<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
    ) -> AvFrame {
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        assert!(
            width % 2 == 0 && height % 2 == 0,
            "4:2:0 requires even width and height"
        );
        assert_eq!(y.len(), width * height, "Y plane size mismatch");
        assert_eq!(
            cb.len(),
            (width / 2) * (height / 2),
            "Cb plane must be width/2 x height/2 (4:2:0)"
        );
        assert_eq!(
            cr.len(),
            (width / 2) * (height / 2),
            "Cr plane must be width/2 x height/2 (4:2:0)"
        );
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph / 2);
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width / 2, height / 2, pcw, pch);
        let vp = pad_plane(&to_plane(cr), width / 2, height / 2, pcw, pch);

        let layout = Layout::I420;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (i, &(ty, tx)) in [(0, 0), (0, 32), (32, 0), (32, 32)].iter().enumerate() {
                    let (y0, x0) = (sb_y + ty, sb_x + tx);
                    let pred = dc_pred(&recy, pw, y0, x0, 32, neutral);
                    let lev = bases
                        .luma
                        .project(&get_residual(&yp, pw, y0, x0, 32, pred), 0.0);
                    put_block(
                        &mut recy,
                        pw,
                        y0,
                        x0,
                        32,
                        &bases.luma.reconstruct(pred, &lev),
                    );
                    tus[i] = levels_to_coeffs(&lev);
                }
                let (skip_cdfs, dc_sign_ctxs) =
                    sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc);
                encode_luma_block_split(&mut enc, &tus, &skip_cdfs, &dc_sign_ctxs, 0, true);

                let (cy, cx) = (sb_y / 2, sb_x / 2);
                let predu = dc_pred(&recu, pcw, cy, cx, 32, neutral);
                let levu = bases
                    .chroma420
                    .project(&get_residual(&up, pcw, cy, cx, 32, predu), 0.0);
                put_block(
                    &mut recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    &bases.chroma420.reconstruct(predu, &levu),
                );
                let predv = dc_pred(&recv, pcw, cy, cx, 32, neutral);
                let levv = bases
                    .chroma420
                    .project(&get_residual(&vp, pcw, cy, cx, 32, predv), 0.0);
                put_block(
                    &mut recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    &bases.chroma420.reconstruct(predv, &levv),
                );
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
            }
        }
        self.finish(enc, &config, pw, ph, width, height, color)
    }

    /// Encode a 4:2:2 YCbCr still. `y` is `width × height`; `cb`/`cr` are
    /// `width/2 × height` (half width, full height). Luma is four 32×32 TUs per
    /// superblock; each chroma plane is one 32-wide × 64-tall (TX_32X64) transform per
    /// superblock. `width` must be even. Chroma coefficient coding is identical to 4:4:4
    /// (avm codes TX_32X64 with the 32×32 scan and TX_64X64 entropy context); only the
    /// basis is the rectangular `chroma422` set.
    pub fn encode_yuv422<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
    ) -> AvFrame {
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        let cb = &planar_image.planes[1];
        let cr = &planar_image.planes[2];
        assert!(width % 2 == 0, "4:2:2 requires even width");
        assert_eq!(y.len(), width * height, "Y plane size mismatch");
        assert_eq!(
            cb.len(),
            (width / 2) * height,
            "Cb plane must be width/2 x height (4:2:2)"
        );
        assert_eq!(
            cr.len(),
            (width / 2) * height,
            "Cr plane must be width/2 x height (4:2:2)"
        );
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let (pcw, pch) = (pw / 2, ph); // chroma: half width, full height
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width / 2, height, pcw, pch);
        let vp = pad_plane(&to_plane(cr), width / 2, height, pcw, pch);

        let layout = Layout::I422;
        let config = self.config(layout);
        let mut recy = vec![0f32; pw * ph];
        let mut recu = vec![0f32; pcw * pch + 1];
        let mut recv = vec![0f32; pcw * pch + 1];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut u_has = vec![0i32; sb_cols * sb_rows];
        let mut v_has = vec![0i32; sb_cols * sb_rows];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (i, &(ty, tx)) in [(0, 0), (0, 32), (32, 0), (32, 32)].iter().enumerate() {
                    let (y0, x0) = (sb_y + ty, sb_x + tx);
                    let pred = dc_pred(&recy, pw, y0, x0, 32, neutral);
                    let lev = bases
                        .luma
                        .project(&get_residual(&yp, pw, y0, x0, 32, pred), 0.0);
                    put_block(
                        &mut recy,
                        pw,
                        y0,
                        x0,
                        32,
                        &bases.luma.reconstruct(pred, &lev),
                    );
                    tus[i] = levels_to_coeffs(&lev);
                }
                let (skip_cdfs, dc_sign_ctxs) =
                    sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc);
                encode_luma_block_split(&mut enc, &tus, &skip_cdfs, &dc_sign_ctxs, 0, true);

                // Chroma block: 32 wide (sb_x/2) × 64 tall (sb_y), one TX_32X64 per plane.
                let (cy, cx) = (sb_y, sb_x / 2);
                let predu = dc_pred_rect(&recu, pcw, cy, cx, 32, 64, neutral);
                let levu = bases
                    .chroma422
                    .project(&get_residual_rect(&up, pcw, cy, cx, 32, 64, predu), 0.0);
                put_block_rect(
                    &mut recu,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &bases.chroma422.reconstruct(predu, &levu),
                );
                let predv = dc_pred_rect(&recv, pcw, cy, cx, 32, 64, neutral);
                let levv = bases
                    .chroma422
                    .project(&get_residual_rect(&vp, pcw, cy, cx, 32, 64, predv), 0.0);
                put_block_rect(
                    &mut recv,
                    pcw,
                    cy,
                    cx,
                    32,
                    64,
                    &bases.chroma422.reconstruct(predv, &levv),
                );
                let ucoeffs = levels_to_coeffs(&levu);
                let vcoeffs = levels_to_coeffs(&levv);

                let at = |g: &[i32], dr: usize, dc: usize| g[(row - dr) * sb_cols + (col - dc)];
                let ua = if row > 0 { at(&u_has, 1, 0) } else { 0 };
                let ul = if col > 0 { at(&u_has, 0, 1) } else { 0 };
                let va = if row > 0 { at(&v_has, 1, 0) } else { 0 };
                let vl = if col > 0 { at(&v_has, 0, 1) } else { 0 };
                let u_skip = layout.chroma_u_skip(qc)[(6 + ua + ul) as usize] as u32;
                encode_chroma_block(&mut enc, &ucoeffs, u_skip, true);
                let u_present = ucoeffs.iter().any(|&(_, l)| l != 0);
                let v_skip =
                    CHROMA_SKIP_V_QC[qc][(6 * (u_present as i32) + va + vl) as usize] as u32;
                encode_chroma_block(&mut enc, &vcoeffs, v_skip, false);
                u_has[row * sb_cols + col] = u_present as i32;
                v_has[row * sb_cols + col] = vcoeffs.iter().any(|&(_, l)| l != 0) as i32;
            }
        }
        self.finish(enc, &config, pw, ph, width, height, color)
    }

    /// Encode a 4:0:0 (monochrome / luma-only) still. `y` is `width × height`.
    /// Four 32x32 luma TUs per superblock; no chroma is coded or signalled
    /// (`has_chroma = false` ⇒ no chroma intra mode, profile 0, layout uvlc 1).
    pub fn encode_yuv400<T: Pixel>(
        &self,
        planar_image: &PlanarImage<T>,
        color: &ColorEncoding,
    ) -> AvFrame {
        let width = planar_image.width;
        let height = planar_image.height;
        let y = &planar_image.planes[0];
        assert_eq!(y.len(), width * height, "Y plane size mismatch");
        let bases = &self.bases;
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);

        let layout = Layout::Monochrome;
        let config = self.config(layout);

        if config.lossless {
            return self.encode_yuv400_lossless(&yp, pw, ph, width, height, &config, color);
        }

        let mut recy = vec![0f32; pw * ph];
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let qc = enc.qc;
        let neutral = self.dc_neutral();
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let sb_y = row * 64;
                let sb_x = col * 64;
                let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
                for (i, &(ty, tx)) in [(0, 0), (0, 32), (32, 0), (32, 32)].iter().enumerate() {
                    let (y0, x0) = (sb_y + ty, sb_x + tx);
                    let pred = dc_pred(&recy, pw, y0, x0, 32, neutral);
                    let lev = bases
                        .luma
                        .project(&get_residual(&yp, pw, y0, x0, 32, pred), 0.0);
                    put_block(
                        &mut recy,
                        pw,
                        y0,
                        x0,
                        32,
                        &bases.luma.reconstruct(pred, &lev),
                    );
                    tus[i] = levels_to_coeffs(&lev);
                }
                let (skip_cdfs, dc_sign_ctxs) =
                    sb_tu_contexts(&tus, sb_y, sb_x, &mut above, &mut left, qc);
                encode_luma_block_split(&mut enc, &tus, &skip_cdfs, &dc_sign_ctxs, 0, false);
            }
        }
        self.finish(enc, &config, pw, ph, width, height, color)
    }

    /// Lossless (base_q=0) monochrome encode: each 64x64 superblock is coded as 256
    /// 4x4 transform units (forced TX_4X4), DC-predicted per TU and carried by the 4x4
    /// WHT. `yp` is the SB-padded source plane. The pixel reconstruction is bit-exact;
    /// the 4x4 coefficient CDFs/contexts are still being validated against the decoder.
    fn encode_yuv400_lossless(
        &self,
        yp: &[f32],
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        config: &Config,
        color: &ColorEncoding,
    ) -> AvFrame {
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx); // base_q=0 -> q-context 0
        let neutral = self.dc_neutral();
        let sb_cols = pw / 64;
        let sb_rows = ph / 64;
        let mut above = vec![0x40u8; pw / 4 + 16];
        let mut left = vec![0x40u8; ph / 4 + 16];

        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                // 16x16 grid of 4x4 transform units, raster order within the superblock.
                // Lossless reconstruction == source, so DC prediction reads `yp` directly.
                let tus = lossless_sb_tus(yp, pw, sb_y, sb_x, neutral);
                let (skip_ctx, dc_sign_ctxs) =
                    sb_tu4_contexts(&tus, sb_y, sb_x, &mut above, &mut left);
                let skip_cdfs: Vec<u32> = skip_ctx
                    .iter()
                    .map(|&c| TXB_SKIP_TX4_Q0[c] as u32)
                    .collect();
                encode_lossless_luma_sb(&mut enc, &tus, &skip_cdfs, &dc_sign_ctxs, 0, false);
            }
        }
        self.finish(enc, config, pw, ph, width, height, color)
    }

    /// 4:4:4 lossless (q=0): luma + full-resolution U/V, all TX_4X4 WHT. Per superblock
    /// the block codes intra modes (incl. use_dpcm_y/uv = 0 and DC uv mode), then 256
    /// luma TUs, 256 U TUs, 256 V TUs — matching avm's shared-tree plane order.
    fn encode_yuv444_lossless<T: Pixel>(
        &self,
        y: &[T],
        cb: &[T],
        cr: &[T],
        width: usize,
        height: usize,
        color: &ColorEncoding,
    ) -> AvFrame {
        let to_plane = |s: &[T]| s.iter().map(|p| p.to_f32()).collect::<Vec<f32>>();
        let (pw, ph) = (sb_align(width), sb_align(height));
        let yp = pad_plane(&to_plane(y), width, height, pw, ph);
        let up = pad_plane(&to_plane(cb), width, height, pw, ph);
        let vp = pad_plane(&to_plane(cr), width, height, pw, ph);
        let config = self.config(Layout::I444);
        let mut enc = RangeEncoder::new();
        enc.qc = get_q_ctx(self.base_q_idx);
        let neutral = self.dc_neutral();
        let (sb_cols, sb_rows) = (pw / 64, ph / 64);
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
        let mut sbtus: Vec<(Vec<Vec<Coeff>>, Vec<Vec<Coeff>>, Vec<Vec<Coeff>>)> = (0..nsb)
            .map(|_| (Vec::new(), Vec::new(), Vec::new()))
            .collect();
        let gen_val =
            |idx: usize, slot: &mut (Vec<Vec<Coeff>>, Vec<Vec<Coeff>>, Vec<Vec<Coeff>>)| {
                let (sb_y, sb_x) = ((idx / sb_cols) * 64, (idx % sb_cols) * 64);
                *slot = (
                    lossless_sb_tus(&yp, pw, sb_y, sb_x, neutral),
                    lossless_sb_tus(&up, pw, sb_y, sb_x, neutral),
                    lossless_sb_tus(&vp, pw, sb_y, sb_x, neutral),
                );
            };
        let nthreads = std::env::var("SLIMAV_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        if nthreads <= 1 || nsb < 8 {
            for (idx, slot) in sbtus.iter_mut().enumerate() {
                gen_val(idx, slot);
            }
        } else {
            let chunk = nsb.div_ceil(nthreads);
            let (yp, up, vp) = (&yp, &up, &vp);
            std::thread::scope(|sc| {
                for (ci, slice) in sbtus.chunks_mut(chunk).enumerate() {
                    let base = ci * chunk;
                    sc.spawn(move || {
                        for (k, slot) in slice.iter_mut().enumerate() {
                            let (sb_y, sb_x) =
                                (((base + k) / sb_cols) * 64, ((base + k) % sb_cols) * 64);
                            *slot = (
                                lossless_sb_tus(yp, pw, sb_y, sb_x, neutral),
                                lossless_sb_tus(up, pw, sb_y, sb_x, neutral),
                                lossless_sb_tus(vp, pw, sb_y, sb_x, neutral),
                            );
                        }
                    });
                }
            });
        }
        // Phase B: serial context derivation (cross-SB grids) + entropy coding.
        for row in 0..sb_rows {
            for col in 0..sb_cols {
                let (sb_y, sb_x) = (row * 64, col * 64);
                let (ytus, utus, vtus) = &sbtus[row * sb_cols + col];
                let (yskip, ydcs) = sb_tu4_contexts(ytus, sb_y, sb_x, &mut ya, &mut yl);
                let yskip_cdfs: Vec<u32> =
                    yskip.iter().map(|&c| TXB_SKIP_TX4_Q0[c] as u32).collect();
                let uskip = sb_tu4_chroma_skip(utus, sb_y, sb_x, &mut ua, &mut ul, false, false);
                // avm's eob_u_flag is the LAST U TU of the block, applied to every V TU.
                let u_last_nz = utus
                    .last()
                    .map_or(false, |t| t.iter().any(|&(_, l)| l != 0));
                let vskip = sb_tu4_chroma_skip(vtus, sb_y, sb_x, &mut va, &mut vl, true, u_last_nz);
                // modes (incl. uv) + luma coeffs, then U, then V (avm shared-tree order)
                encode_lossless_luma_sb(&mut enc, ytus, &yskip_cdfs, &ydcs, 0, true);
                for (i, tu) in utus.iter().enumerate() {
                    encode_chroma_tu4(&mut enc, tu, TXB_SKIP_TX4_Q0[uskip[i]] as u32, false);
                }
                for (i, tu) in vtus.iter().enumerate() {
                    encode_chroma_tu4(&mut enc, tu, V_TXB_SKIP_TX4_Q0[vskip[i]] as u32, true);
                }
            }
        }
        self.finish(enc, &config, pw, ph, width, height, color)
    }

    /// Encode an RGB image to 4:4:4 AV2. Converts RGB→YCbCr internally.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_444<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<AvFrame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let bd = img.bit_depth;
        let maxv = (1i32 << bd.bits()) - 1;
        let off_q = (1i32 << (bd.bits() - 1)) << Q;
        let mx_i = maxv;
        let n = img.planes[0].len();
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
        self.encode_yuv444(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
        )
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
        base_q_idx: u8,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<AvFrame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        if base_q_idx == 0 {
            return Err(EncodeError::InvalidQuality);
        }
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
        for row in 0..ch {
            for c in 0..cw {
                let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
                let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));
                let avg_q =
                    |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];
                cb[row * cw + c] = ((avg_q(&fcb_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
                cr[row * cw + c] = ((avg_q(&fcr_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            }
        }
        Ok(self.encode_yuv420(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
        ))
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
        base_q_idx: u8,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<AvFrame, EncodeError> {
        img.validate_444()?;
        validate_dims(img.width as u32, img.height as u32)?;
        if base_q_idx == 0 {
            return Err(EncodeError::InvalidQuality);
        }
        let (w, h) = (img.width, img.height);
        let bd = img.bit_depth.bits();
        let maxv = (1i32 << bd) - 1;
        let off_q = ((1i32 << (bd - 1)) as i32) << Q;
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
        for row in 0..h {
            for c in 0..cw {
                let x0 = 2 * c;
                let x1 = (2 * c + 1).min(w - 1);
                let cb0 = fcb_q[row * w + x0];
                let cb1 = fcb_q[row * w + x1];
                let cr0 = fcr_q[row * w + x0];
                let cr1 = fcr_q[row * w + x1];
                cb[row * cw + c] = ((cb0 + cb1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
                cr[row * cw + c] = ((cr0 + cr1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            }
        }
        Ok(self.encode_yuv422(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [y, cb, cr],
            },
            color,
        ))
    }

    /// Encode a luma-only (4:0:0 / monochrome) image to AV2.
    ///
    /// Returns `Err` if dimensions are out of range (0 or > 16 383) or if
    /// `img.bit_depth` is not 8, 10, or 12.
    pub fn encode_image_400<T: Pixel>(
        &self,
        img: &PlanarImage<T>,
        color: &ColorEncoding,
        threads: usize,
    ) -> Result<AvFrame, EncodeError> {
        img.validate_400()?;
        validate_dims(img.width as u32, img.height as u32)?;
        let plane = img.planes[0].to_vec();
        Ok(self.encode_yuv400(
            &PlanarImage {
                width: img.width,
                height: img.height,
                bit_depth: img.bit_depth,
                planes: [plane, vec![], vec![]],
            },
            color,
        ))
    }

    fn finish(
        &self,
        enc: RangeEncoder,
        config: &Config,
        pw: usize,
        ph: usize,
        width: usize,
        height: usize,
        color: &ColorEncoding,
    ) -> AvFrame {
        let tile = enc.finish();
        // AV2 derives its mode-info grid by rounding the frame to 4px
        // (ALIGN_POWER_OF_TWO(dim, MI_SIZE_LOG2)); superblocks are 64px (16 mi).
        // A square superblock at the right/bottom edge is force-split (no bits read)
        // only when *less than half* of it (<=32px, i.e. <=8 mi) is in-frame — see
        // is_partition_implied_at_boundary. When >32px is in-frame, every SB stays
        // PARTITION_NONE exactly as in the padded encode, so we can signal the real
        // size and let the decoder crop: the coded tile is byte-identical.
        let mi_cols = (width + 3) / 4;
        let mi_rows = (height + 3) / 4;
        const MIB: usize = 16; // 64px superblock in 4px mode-info units
        let sb_cols = (mi_cols + MIB - 1) / MIB;
        let sb_rows = (mi_rows + MIB - 1) / MIB;
        let safe_w = (sb_cols - 1) * MIB + 8 < mi_cols;
        let safe_h = (sb_rows - 1) * MIB + 8 < mi_rows;
        let exact = safe_w && safe_h && !config.lossless; // lossless can't code boundary SBs -> pad
        // Signaled dimensions: real size when boundary-safe, else the padded size.
        let (sw, sh) = if exact { (width, height) } else { (pw, ph) };
        let mut frame = frame_header(config, sw as u32, sh as u32);
        frame.extend(&tile);
        let mut data = vec![];
        data.extend(obu(2, &[]));
        data.extend(obu(1, &sequence_header(config, sw as u32, sh as u32)));
        data.extend(obu(4, &frame));
        AvFrame {
            data,
            width,
            height,
            // Coded bit depth signaled in the sequence header (8/10/12). av2C/pixi in
            // the AVIF muxer must use this.
            bit_depth: self.bit_depth,
            base_q_idx: self.base_q_idx,
            color: *color,
            chroma_format: match config.layout {
                Layout::Monochrome => ChromaFormat::Monochrome,
                Layout::I420 => ChromaFormat::Yuv420,
                Layout::I422 => ChromaFormat::Yuv422,
                Layout::I444 => ChromaFormat::Yuv444,
            },
        }
    }

    /// Finish wrapping a color AV1 OBU stream in an AVIF container.
    pub fn wrap_avif(frame: &AvFrame) -> Result<Vec<u8>, EncodeError> {
        Ok(avif::to_avif(
            frame,
            &Av2Format {
                bit_depth: frame.bit_depth,
                monochrome: frame.chroma_format == ChromaFormat::Monochrome,
                chroma_sub_x: frame.chroma_format == ChromaFormat::Yuv422
                    || frame.chroma_format == ChromaFormat::Yuv420,
                chroma_sub_y: frame.chroma_format == ChromaFormat::Yuv420,
            },
        ))
    }
}
