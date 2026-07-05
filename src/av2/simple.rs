/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its contributors may
 *     be used to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use super::{Av2Encoder, Av2Frame, av2_map_quality};
use crate::avif::{validate_buf, validate_dims};
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Orientation};
use crate::{BitDepth, ChromaFormat, Cicp, EncodeConfig, Pixel, PlanarImage, TxPart};

fn resolve_color(cfg: &EncodeConfig) -> Cicp {
    cfg.color_encoding.unwrap_or_else(Cicp::srgb_ycbcr)
}

/// Encode an interleaved-RGB (GBR-plane) image to a colour [`Av2Frame`],
/// dispatching on the requested chroma format.
fn encode_rgb_color<T: Pixel>(
    enc: &Av2Encoder,
    img: &PlanarImage<T>,
    chroma: ChromaFormat,
    color: &Cicp,
) -> Result<Av2Frame, EncodeError> {
    match chroma {
        ChromaFormat::Yuv444 => enc.encode_image_444(img, color),
        ChromaFormat::Yuv422 => enc.encode_image_422(img, color),
        ChromaFormat::Yuv420 => enc.encode_image_420(img, color),
        ChromaFormat::Monochrome => enc.encode_image_400(img, color),
    }
}

/// Encode a pre-converted YCbCr image to a colour [`Av2Frame`], dispatching on
/// the requested chroma format.
fn encode_yuv_color<T: Pixel>(
    enc: &Av2Encoder,
    img: &PlanarImage<T>,
    chroma: ChromaFormat,
    color: &Cicp,
) -> Result<Av2Frame, EncodeError> {
    match chroma {
        ChromaFormat::Yuv444 => enc.encode_yuv444(img, color),
        ChromaFormat::Yuv422 => enc.encode_yuv422(img, color),
        ChromaFormat::Yuv420 => enc.encode_yuv420(img, color),
        ChromaFormat::Monochrome => enc.encode_yuv400(img, color),
    }
}

/// Encode the alpha plane (already a standalone monochrome image) as a
/// **lossless** AV2 mono frame, matching the AV1 behavior of coding alpha
/// without loss.
fn encode_alpha<T: Pixel>(
    alpha_mono: &PlanarImage<T>,
    bit_depth: BitDepth,
    color: &Cicp,
    threads: usize,
) -> Result<Av2Frame, EncodeError> {
    // base_q_idx = 0 ⇒ the mono path takes its lossless branch.
    Av2Encoder::with_bit_depth(0, bit_depth.bits())
        .with_threads(threads)
        .encode_image_400(alpha_mono, color)
}

/// Common preamble: enforce the expected bit depth, validate dims + config, and
/// build a color encoder at the mapped quality.
fn prepare(
    bit_depth: BitDepth,
    want: BitDepth,
    width: usize,
    height: usize,
    cfg: &EncodeConfig,
) -> Result<Av2Encoder, EncodeError> {
    if bit_depth != want {
        return Err(EncodeError::UnsupportedChromaBitDepth(bit_depth));
    }
    validate_dims(width as u32, height as u32)?;
    cfg.validate()?;
    Ok(Av2Encoder::with_bit_depth(
        if cfg.quality == 0 {
            0
        } else {
            av2_map_quality(cfg.quality)
        },
        bit_depth.bits(),
    )
    .with_tiles(8, 8)
    .with_txpart(TxPart::ThreeWay)
    .with_speed(cfg.speed)
    .with_threads(cfg.threads)
    .with_cfl(true)
    .with_aq(cfg.adaptive_quant))
}

fn prepare_lossless_capable(
    bit_depth: BitDepth,
    want: BitDepth,
    width: usize,
    height: usize,
    cfg: &EncodeConfig,
) -> Result<Av2Encoder, EncodeError> {
    if bit_depth != want {
        return Err(EncodeError::UnsupportedChromaBitDepth(bit_depth));
    }
    validate_dims(width as u32, height as u32)?;
    cfg.validate()?;
    let q = if cfg.quality == 0 {
        0
    } else {
        av2_map_quality(cfg.quality)
    };
    Ok(Av2Encoder::with_bit_depth(q, bit_depth.bits())
        .with_tiles(8, 8)
        .with_txpart(TxPart::ThreeWay)
        .with_speed(cfg.speed)
        .with_threads(cfg.threads)
        .with_aq(cfg.adaptive_quant))
}

#[inline]
fn icc(cfg: &EncodeConfig) -> Option<&[u8]> {
    cfg.icc.as_deref()
}

#[inline]
fn exif(cfg: &EncodeConfig) -> Option<&[u8]> {
    cfg.metadata.exif.as_deref()
}

#[inline]
fn orientation(cfg: &EncodeConfig) -> Orientation {
    cfg.metadata.orientation
}

#[inline]
fn clli(cfg: &EncodeConfig) -> Option<ContentLightLevel> {
    cfg.metadata.content_light_level
}

fn rgb_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = prepare(img.bit_depth, want, w, h, cfg)?;
    validate_buf(&img.planes[0], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[1], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[2], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let frame = encode_rgb_color(&enc, img, cfg.chroma, &color)?;
    Av2Encoder::wrap_avif(&frame, icc(cfg), exif(cfg), orientation(cfg), clli(cfg))
}

/// Same as [`rgb_core`] but the input carries an alpha plane that is **dropped**.
fn rgba_drop_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = prepare(img.bit_depth, want, w, h, cfg)?;
    validate_buf(&img.planes[0], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[1], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[2], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[3], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let frame = encode_rgb_color(&enc, img, cfg.chroma, &color)?;
    Av2Encoder::wrap_avif(&frame, icc(cfg), exif(cfg), orientation(cfg), clli(cfg))
}

fn rgba_alpha_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = prepare(img.bit_depth, want, w, h, cfg)?;
    validate_buf(&img.planes[0], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[1], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[2], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[3], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let color_frame = encode_rgb_color(&enc, img, cfg.chroma, &color)?;
    let alpha_frame = encode_alpha(&img.packed_alpha_4(), img.bit_depth, &color, cfg.threads)?;
    Av2Encoder::wrap_avif_alpha(
        &color_frame,
        &alpha_frame,
        icc(cfg),
        exif(cfg),
        orientation(cfg),
        clli(cfg),
    )
}

fn gray_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = prepare_lossless_capable(img.bit_depth, want, w, h, cfg)?;
    validate_buf(&img.planes[0], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let frame = enc.encode_image_400(img, &color)?;
    Av2Encoder::wrap_avif(&frame, icc(cfg), exif(cfg), orientation(cfg), clli(cfg))
}

fn yuv_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = match cfg.chroma {
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            prepare_lossless_capable(img.bit_depth, want, w, h, cfg)?
        }
        _ => prepare(img.bit_depth, want, w, h, cfg)?,
    };
    img.validate_with(cfg.chroma)?;
    let color = resolve_color(cfg);
    let frame = encode_yuv_color(&enc, img, cfg.chroma, &color)?;
    Av2Encoder::wrap_avif(&frame, icc(cfg), exif(cfg), orientation(cfg), clli(cfg))
}

fn yuva_alpha_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = match cfg.chroma {
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            prepare_lossless_capable(img.bit_depth, want, w, h, cfg)?
        }
        _ => prepare(img.bit_depth, want, w, h, cfg)?,
    };
    img.validate_with(cfg.chroma)?;
    validate_buf(&img.planes[3], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let color_frame = encode_yuv_color(&enc, img, cfg.chroma, &color)?;
    let alpha_frame = encode_alpha(&img.packed_alpha_4(), img.bit_depth, &color, cfg.threads)?;
    Av2Encoder::wrap_avif_alpha(
        &color_frame,
        &alpha_frame,
        icc(cfg),
        exif(cfg),
        orientation(cfg),
        clli(cfg),
    )
}

fn gray_alpha_core<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
    want: BitDepth,
) -> Result<Vec<u8>, EncodeError> {
    let (w, h) = (img.width, img.height);
    let enc = prepare_lossless_capable(img.bit_depth, want, w, h, cfg)?;
    validate_buf(&img.planes[0], w as u32, h as u32, 1)?;
    validate_buf(&img.planes[1], w as u32, h as u32, 1)?;
    let color = resolve_color(cfg);
    let color_frame = enc.encode_image_400(img, &color)?;
    let alpha_frame = encode_alpha(&img.packed_alpha_2(), img.bit_depth, &color, cfg.threads)?;
    Av2Encoder::wrap_avif_alpha(
        &color_frame,
        &alpha_frame,
        icc(cfg),
        exif(cfg),
        orientation(cfg),
        clli(cfg),
    )
}

/// Encode an 8-bit RGB image (identity-RGB GBR planes) to AV2 AVIF.
pub fn encode_rgb8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgb_core(img, cfg, BitDepth::Eight)
}
/// Encode a 10-bit RGB image to AV2 AVIF.
pub fn encode_rgb10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgb_core(img, cfg, BitDepth::Ten)
}
/// Encode a 12-bit RGB image to AV2 AVIF.
pub fn encode_rgb12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgb_core(img, cfg, BitDepth::Twelve)
}

/// Encode an 8-bit RGBA image to AV2 AVIF. The alpha channel is **discarded**.
pub fn encode_rgba8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgba_drop_core(img, cfg, BitDepth::Eight)
}
/// Encode a 10-bit RGBA image to AV2 AVIF. Alpha is discarded.
pub fn encode_rgba10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgba_drop_core(img, cfg, BitDepth::Ten)
}
/// Encode a 12-bit RGBA image to AV2 AVIF. Alpha is discarded.
pub fn encode_rgba12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    rgba_drop_core(img, cfg, BitDepth::Twelve)
}

/// Encode an 8-bit RGBA image to AV2 AVIF with a lossless alpha auxiliary item.
pub fn encode_rgba8_with_alpha(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    rgba_alpha_core(img, cfg, BitDepth::Eight)
}
/// Encode a 10-bit RGBA image to AV2 AVIF with a lossless alpha auxiliary item.
pub fn encode_rgba10_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    rgba_alpha_core(img, cfg, BitDepth::Ten)
}
/// Encode a 12-bit RGBA image to AV2 AVIF with a lossless alpha auxiliary item.
pub fn encode_rgba12_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    rgba_alpha_core(img, cfg, BitDepth::Twelve)
}

/// Encode an 8-bit grayscale image to AV2 AVIF (monochrome).
pub fn encode_gray8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    gray_core(img, cfg, BitDepth::Eight)
}
/// Encode a 10-bit grayscale image to AV2 AVIF.
pub fn encode_gray10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    gray_core(img, cfg, BitDepth::Ten)
}
/// Encode a 12-bit grayscale image to AV2 AVIF.
pub fn encode_gray12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    gray_core(img, cfg, BitDepth::Twelve)
}

/// Encode a pre-converted 8-bit planar YCbCr image to AV2 AVIF.
pub fn encode_yuv8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    yuv_core(img, cfg, BitDepth::Eight)
}
/// Encode a pre-converted 10-bit planar YCbCr image to AV2 AVIF.
pub fn encode_yuv10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    yuv_core(img, cfg, BitDepth::Ten)
}
/// Encode a pre-converted 12-bit planar YCbCr image to AV2 AVIF.
pub fn encode_yuv12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    yuv_core(img, cfg, BitDepth::Twelve)
}

/// Encode pre-converted 8-bit YCbCr + alpha to AV2 AVIF (lossless alpha item).
pub fn encode_yuva8_with_alpha(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    yuva_alpha_core(img, cfg, BitDepth::Eight)
}
/// Encode pre-converted 10-bit YCbCr + alpha to AV2 AVIF (lossless alpha item).
pub fn encode_yuva10_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    yuva_alpha_core(img, cfg, BitDepth::Ten)
}
/// Encode pre-converted 12-bit YCbCr + alpha to AV2 AVIF (lossless alpha item).
pub fn encode_yuva12_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    yuva_alpha_core(img, cfg, BitDepth::Twelve)
}

pub fn encode_gray_alpha8(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    gray_alpha_core(img, cfg, BitDepth::Eight)
}
/// Encode a 10-bit grayscale image + alpha to AV2 AVIF (lossless alpha item).
pub fn encode_gray_alpha10(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    gray_alpha_core(img, cfg, BitDepth::Ten)
}
/// Encode a 12-bit grayscale image + alpha to AV2 AVIF (lossless alpha item).
pub fn encode_gray_alpha12(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    gray_alpha_core(img, cfg, BitDepth::Twelve)
}
