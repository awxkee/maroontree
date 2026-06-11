/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! AVIF still-image encoder.
//!
//! Wraps [`crate`] for AV1 bitstream encoding inside the AVIF ISOBMFF container.
//! All entry points validate their inputs and return `Result<Vec<u8>, EncodeError>`;
//! the `Vec<u8>` is a self-contained `.avif` file that any conformant AVIF decoder
//! should accept.
//!
//! # Quick start
//!
//! ```ignore
//! use crate_avif::{encode_rgb8, EncodeConfig, ChromaFormat, ColorMetadata, ColorEncoding};
//!
//! let cfg = EncodeConfig::new()
//!     .with_quality(85)
//!     .with_chroma(ChromaFormat::Yuv420);
//! let avif: Vec<u8> = encode_rgb8(&rgb_pixels, 1920, 1080, &cfg)?;
//! ```
//!
//! # YUV direct path
//!
//! When the caller already has pre-converted YCbCr planes (e.g. a video pipeline),
//! use the `encode_yuv*` functions to bypass the internal RGB→YCbCr step:
//!
//! ```ignore
//! let avif = encode_yuv8(&y, &cb, &cr, width, height, &cfg)?;
//! // cb/cr must be ceil(w/2)×ceil(h/2) samples when cfg.chroma == Yuv420
//! ```

use crate::color::ColorEncoding;
use crate::encoder::{
    encode_lossy_gray_obu, encode_still_lossy, encode_still_lossy_420, encode_still_lossy_422,
    encode_yuv420_obu, encode_yuv422_obu, encode_yuv444_obu,
};
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Metadata, Orientation};
use crate::{BitDepth, PlanarImage, isobmff};

const MIN_DIM: u32 = 1;
/// Maximum dimension. AV1 level 6.3 handles frames up to 35 651 584 luma
/// samples; with both axes capped here the largest possible frame is ~268 MP.
const MAX_DIM: u32 = 16_383;

/// Chroma subsampling format for the AV1 encoder.
///
/// Determines the AV1 profile in the bitstream and the chroma plane dimensions
/// expected by the `encode_yuv*` entry points.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// 4:2:0 — chroma halved both horizontally and vertically (AV1 profile 0).
    /// Standard for photo and internet-video delivery; best compression per bit.
    #[default]
    Yuv420,
    /// 4:2:2 — chroma halved horizontally only (AV1 profile 2).
    Yuv422,
    /// 4:4:4 — full-resolution chroma (AV1 profile 1 for ≤10-bit; 2 for 12-bit).
    /// Best color fidelity; preferred for graphics, HDR, and lossless content.
    Yuv444,
    /// 4:0:0 — luma only; monochrome (AV1 profile 0). Used by the `gray*` entry
    /// points and automatically selected for the alpha auxiliary image.
    Monochrome,
}

/// Encoder configuration shared by all entry points.
///
/// Build with [`EncodeConfig::new`] and the `with_*` builder methods.
///
/// ```ignore
/// let cfg = EncodeConfig::new()
///     .with_quality(90)
///     .with_chroma(ChromaFormat::Yuv444)
///     .with_color(ColorMetadata::Cicp(ColorEncoding::bt2020_pq()));
/// ```
#[derive(Clone, Debug)]
pub struct EncodeConfig {
    /// Visual quality 1..=100 (higher = better quality, larger file).
    /// Maps to AV1 `base_q_idx`: quality 100 → q ≈ 1 (near-lossless),
    /// quality 1 → q = 255. For pixel-perfect lossless use the `encode_*_lossless`
    /// entry points instead.
    pub quality: u8,
    /// Chroma subsampling format. Ignored by the `gray*` entry points, which
    /// always use [`ChromaFormat::Monochrome`].
    pub chroma: ChromaFormat,
    /// Color metadata written to the `colr` box in the container.
    pub color_encoding: Option<ColorEncoding>,
    pub icc: Option<Vec<u8>>,
    /// Optional image metadata (orientation, HDR content light level, EXIF).
    pub metadata: Metadata,
    /// Worker threads for tile-level parallelism.
    /// `0` = all available cores; `1` = serial (default); `N` = up to N.
    pub threads: usize,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 80,
            chroma: ChromaFormat::Yuv420,
            color_encoding: Some(ColorEncoding::srgb_ycbcr()),
            icc: None,
            metadata: Metadata::default(),
            threads: 1,
        }
    }
}

impl EncodeConfig {
    /// Default settings: quality = 80, 4:2:0 chroma, sRGB CICP, serial.
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = quality;
        self
    }

    pub fn with_chroma(mut self, chroma: ChromaFormat) -> Self {
        self.chroma = chroma;
        self
    }

    pub fn with_cicp(mut self, color: ColorEncoding) -> Self {
        self.color_encoding = Some(color);
        self
    }

    pub fn with_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.icc = Some(icc);
        self
    }

    pub fn with_metadata(mut self, metadata: Metadata) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_orientation(mut self, o: Orientation) -> Self {
        self.metadata.orientation = o;
        self
    }

    pub fn with_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.metadata.content_light_level = Some(cll);
        self
    }

    pub fn with_exif(mut self, exif: Vec<u8>) -> Self {
        self.metadata.exif = Some(exif);
        self
    }

    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads;
        self
    }

    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        validate_quality(self.quality)
    }
}

pub(crate) fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn validate_quality(quality: u8) -> Result<(), EncodeError> {
    if quality == 0 || quality > 100 {
        return Err(EncodeError::InvalidQuality);
    }
    Ok(())
}

pub(crate) fn validate_buf_u8<T>(buf: &[T], w: u32, h: u32, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<T>(w as usize, h as usize, ch)?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

fn validate_buf_u16(buf: &[u16], w: u32, h: u32, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<u16>(w as usize, h as usize, ch)?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

pub(crate) fn checked_buffer_size<T>(w: usize, h: usize, ch: usize) -> Result<usize, EncodeError> {
    _ = w
        .checked_mul(h)
        .and_then(|n| n.checked_mul(ch))
        .and_then(|n| n.checked_mul(size_of::<T>()))
        .and_then(|n| isize::try_from(n).ok())
        .ok_or(EncodeError::DimensionTooLarge {
            width: w,
            height: h,
        })?;

    w.checked_mul(h)
        .and_then(|n| n.checked_mul(ch))
        .ok_or(EncodeError::DimensionTooLarge {
            width: w,
            height: h,
        })
}

/// Map quality 1..=100 → AV1 base_q_idx 1..=255.
/// quality 100 → q = 1 (near-lossless), quality 1 → q = 255.
fn quality_to_q(quality: u8) -> u8 {
    (1u32 + (100 - quality as u32) * 254 / 99).clamp(1, 255) as u8
}

/// AV1 sequence profile from bit depth + chroma format.
fn av1_profile(bit_depth: u8, chroma: ChromaFormat) -> u8 {
    match chroma {
        _ if bit_depth == 12 => 2,
        ChromaFormat::Yuv422 => 2,
        ChromaFormat::Yuv444 => 1,
        ChromaFormat::Yuv420 | ChromaFormat::Monochrome => 0,
    }
}

/// Build the [`isobmff::Av1cParams`] for a given encode. Extracts the sequence
/// header OBU from the encoder output and embeds it as `configOBUs`.
pub(crate) fn make_av1c(
    _obu: &[u8],
    bit_depth: u8,
    width: u32,
    height: u32,
    chroma: ChromaFormat,
) -> isobmff::Av1cParams {
    let (sub_x, sub_y) = match chroma {
        ChromaFormat::Yuv420 => (true, true),
        ChromaFormat::Yuv422 => (true, false),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => (false, false),
    };
    isobmff::Av1cParams {
        seq_profile: av1_profile(bit_depth, chroma),
        seq_level_idx: isobmff::level_for(width, height),
        high_bitdepth: bit_depth > 8,
        twelve_bit: bit_depth == 12,
        monochrome: matches!(chroma, ChromaFormat::Monochrome),
        chroma_sub_x: sub_x,
        chroma_sub_y: sub_y,
        // avifenc / libavif leave configOBUs empty; the decoder reads the
        // sequence header from the sample data.  Embedding it here is valid
        // per spec but Apple's profile-2 path is stricter and rejects when
        // configOBUs is present, so we omit it to match the reference encoder.
        seq_header_obu: vec![],
    }
}

/// Finish wrapping a color AV1 OBU stream in an AVIF container.
fn finalize_color(
    av1_obu: Vec<u8>,
    width: u32,
    height: u32,
    bit_depth: u8,
    chroma: ChromaFormat,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let av1c = make_av1c(&av1_obu, bit_depth, width, height, chroma);
    let channels: u8 = if matches!(chroma, ChromaFormat::Monochrome) {
        1
    } else {
        3
    };
    isobmff::wrap_av1_image(
        &av1_obu,
        width,
        height,
        bit_depth,
        channels,
        &av1c,
        cfg.color_encoding.as_ref(),
        cfg.icc.as_deref(),
        &cfg.metadata,
    )
}

/// Finish wrapping a color + alpha OBU pair in an AVIF container.
pub(crate) fn finalize_with_alpha(
    color_obu: Vec<u8>,
    alpha_obu: Vec<u8>,
    width: u32,
    height: u32,
    bit_depth: u8,
    chroma: ChromaFormat,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    let av1c_color = make_av1c(&color_obu, bit_depth, width, height, chroma);
    let av1c_alpha = make_av1c(
        &alpha_obu,
        bit_depth,
        width,
        height,
        ChromaFormat::Monochrome,
    );
    isobmff::wrap_av1_image_with_alpha(
        &color_obu,
        &alpha_obu,
        width,
        height,
        bit_depth,
        &av1c_color,
        &av1c_alpha,
        cfg.color_encoding.as_ref(),
        cfg.icc.as_deref(),
        &cfg.metadata,
    )
}

/// Generic lossy dispatch over chroma format (crate `Pixel`-generic path).
/// The `img` planes must already be in crate's expected GBR order
/// (set by `PlanarImage::from_interleaved_rgb`).
fn dispatch_lossy<T: crate::Pixel>(
    img: &PlanarImage<T>,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&ColorEncoding>,
    threads: usize,
) -> Vec<u8> {
    match chroma {
        ChromaFormat::Yuv420 | ChromaFormat::Monochrome => {
            encode_still_lossy_420(img, q, color, threads)
        }
        ChromaFormat::Yuv422 => encode_still_lossy_422(img, q, color, threads),
        ChromaFormat::Yuv444 => encode_still_lossy(img, q, color, threads),
    }
}

/// Encode an 8-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order.
pub fn encode_rgb8(
    rgb: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgb, width, height, 3)?;
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Eight, rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 8, cfg.chroma, cfg)
}

/// Encode an 8-bit RGBA image to AVIF. The alpha channel is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
pub fn encode_rgba8(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    let rgb: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Eight, &rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 8, cfg.chroma, cfg)
}

/// Encode an 8-bit RGBA image to AVIF with a separate alpha auxiliary image.
///
/// Produces two `av01` items in the container: a color image and a monochrome
/// alpha image linked by an `auxl` reference. `rgba` must hold exactly
/// `width * height * 4` bytes in R, G, B, A order.
pub fn encode_rgba8_with_alpha(
    rgba: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(rgba, width, height, 4)?;
    let (w, h) = (width as usize, height as usize);
    let q = quality_to_q(cfg.quality);
    let mut rgb = vec![0u8; w * h * 3];
    let mut alpha = vec![0u8; w * h];
    for ((px, dst_rgb), alpha) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
        .zip(alpha.iter_mut())
    {
        dst_rgb[0] = px[0];
        dst_rgb[1] = px[1];
        dst_rgb[2] = px[2];
        *alpha = px[3];
    }
    let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb);
    let color_obu = dispatch_lossy(
        &img,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: img.width,
            height: img.height,
            bit_depth: BitDepth::Eight,
            planes: [alpha, vec![], vec![]],
        },
        BitDepth::Eight,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(color_obu, alpha_obu, width, height, 8, cfg.chroma, cfg)
}

/// Encode a 10-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` `u16` samples, each in `0..=1023`.
pub fn encode_rgb10(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgb, width, height, 3)?;
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Ten, rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 10, cfg.chroma, cfg)
}

/// Encode a 10-bit RGBA image to AVIF. Alpha is discarded.
///
/// `rgba` must hold exactly `width * height * 4` `u16` samples in `0..=1023`.
pub fn encode_rgba10(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let rgb: Vec<u16> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Ten, &rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 10, cfg.chroma, cfg)
}

/// Encode a 10-bit RGBA image to AVIF with a separate alpha auxiliary image.
pub fn encode_rgba10_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let (w, h) = (width as usize, height as usize);
    let q = quality_to_q(cfg.quality);
    let mut rgb = vec![0u16; w * h * 3];
    let mut alpha = vec![0u16; w * h];
    for ((px, dst_rgb), alpha) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
        .zip(alpha.iter_mut())
    {
        dst_rgb[0] = px[0];
        dst_rgb[1] = px[1];
        dst_rgb[2] = px[2];
        *alpha = px[3];
    }
    let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Ten, &rgb);
    let color_obu = dispatch_lossy(
        &img,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: img.width,
            height: img.height,
            bit_depth: BitDepth::Ten,
            planes: [alpha, vec![], vec![]],
        },
        BitDepth::Ten,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(color_obu, alpha_obu, width, height, 10, cfg.chroma, cfg)
}

/// Encode a 12-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` samples, each in `0..=4095`,
/// packed as `u16`.
pub fn encode_rgb12(
    rgb: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgb, width, height, 3)?;
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Twelve, rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 12, cfg.chroma, cfg)
}

/// Encode a 12-bit RGBA image to AVIF. Alpha is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let rgb: Vec<u16> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|px| [px[0], px[1], px[2]])
        .collect();
    let img =
        PlanarImage::from_interleaved_rgb(width as usize, height as usize, BitDepth::Twelve, &rgb);
    let obu = dispatch_lossy(
        &img,
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    finalize_color(obu, width, height, 12, cfg.chroma, cfg)
}

/// Encode a 12-bit RGBA image to AVIF with a separate alpha auxiliary image.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12_with_alpha(
    rgba: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(rgba, width, height, 4)?;
    let (w, h) = (width as usize, height as usize);
    let q = quality_to_q(cfg.quality);
    let mut rgb = vec![0u16; w * h * 3];
    let mut alpha = vec![0u16; w * h];
    for ((px, dst_rgb), alpha) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
        .zip(alpha.iter_mut())
    {
        dst_rgb[0] = px[0];
        dst_rgb[1] = px[1];
        dst_rgb[2] = px[2];
        *alpha = px[3];
    }
    let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Twelve, &rgb);
    let color_obu = dispatch_lossy(
        &img,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    );
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: img.width,
            height: img.height,
            bit_depth: BitDepth::Twelve,
            planes: [alpha, vec![], vec![]],
        },
        BitDepth::Twelve,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(color_obu, alpha_obu, width, height, 12, cfg.chroma, cfg)
}

/// Encode an 8-bit grayscale image to AVIF using AV1 monochrome coding.
///
/// `gray` must hold exactly `width * height` bytes. The output uses
/// `mono_chrome = 1` (NumPlanes = 1) and AV1 profile 0.
pub fn encode_gray8(
    gray: &[u8],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u8(gray, width, height, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: width as usize,
            height: height as usize,
            bit_depth: BitDepth::Eight,
            planes: [gray.to_vec(), vec![], vec![]],
        },
        BitDepth::Eight,
        q,
        true,
        cfg.threads,
    )?;
    finalize_color(obu, width, height, 8, ChromaFormat::Monochrome, cfg)
}

/// Encode a 10-bit grayscale image to AVIF.
///
/// `gray` must hold exactly `width * height` `u16` samples in `0..=1023`.
pub fn encode_gray10(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(gray, width, height, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: width as usize,
            height: height as usize,
            bit_depth: BitDepth::Ten,
            planes: [gray.to_vec(), vec![], vec![]],
        },
        BitDepth::Ten,
        q,
        true,
        cfg.threads,
    )?;
    finalize_color(obu, width, height, 10, ChromaFormat::Monochrome, cfg)
}

/// Encode a 12-bit grayscale image to AVIF.
///
/// `gray` must hold exactly `width * height` `u16` samples in `0..=4095`.
pub fn encode_gray12(
    gray: &[u16],
    width: u32,
    height: u32,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(width, height)?;
    cfg.validate()?;
    validate_buf_u16(gray, width, height, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: width as usize,
            height: height as usize,
            bit_depth: BitDepth::Eight,
            planes: [gray.to_vec(), vec![], vec![]],
        },
        BitDepth::Twelve,
        q,
        true,
        cfg.threads,
    )?;
    finalize_color(obu, width, height, 12, ChromaFormat::Monochrome, cfg)
}

/// Encode a pre-converted 8-bit planar YCbCr image to AVIF.
///
/// `y` must be `width × height` bytes; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv8(
    planar_image: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u8(
        planar_image,
        BitDepth::Eight,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    finalize_color(
        obu,
        planar_image.width as u32,
        planar_image.height as u32,
        8,
        cfg.chroma,
        cfg,
    )
}

/// Encode a pre-converted 10-bit planar YCbCr image to AVIF.
///
/// Each sample is a `u16` in `0..=1023`; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv10(
    planar_image: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u16(
        planar_image,
        BitDepth::Ten,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    finalize_color(
        obu,
        planar_image.width as u32,
        planar_image.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode a pre-converted 12-bit planar YCbCr image to AVIF.
///
/// Each sample is a `u16` in `0..=4095`; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv12(
    planar_image: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u16(
        planar_image,
        BitDepth::Twelve,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    finalize_color(
        obu,
        planar_image.width as u32,
        planar_image.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 8-bit YCbCr + a separate 8-bit alpha plane to AVIF.
///
/// `a` must be `width * height` bytes; YCbCr subsampling must match `cfg.chroma`.
pub fn encode_yuva8_with_alpha(
    planar_image: &PlanarImage<u8>,
    a: &[u8],
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    validate_buf_u8(a, planar_image.width as u32, planar_image.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u8(
        planar_image,
        BitDepth::Eight,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: planar_image.width,
            height: planar_image.height,
            bit_depth: BitDepth::Eight,
            planes: [a.to_vec(), vec![], vec![]],
        },
        BitDepth::Eight,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        planar_image.width as u32,
        planar_image.height as u32,
        8,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 10-bit YCbCr + a separate 10-bit alpha plane to AVIF.
pub fn encode_yuva10_with_alpha(
    planar_image: &PlanarImage<u16>,
    a: &[u16],
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    validate_buf_u16(a, planar_image.width as u32, planar_image.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u16(
        planar_image,
        BitDepth::Ten,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: planar_image.width,
            height: planar_image.height,
            bit_depth: BitDepth::Ten,
            planes: [a.to_vec(), vec![], vec![]],
        },
        BitDepth::Ten,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        planar_image.width as u32,
        planar_image.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 12-bit YCbCr + a separate 12-bit alpha plane to AVIF.
pub fn encode_yuva12_with_alpha(
    planar_image: &PlanarImage<u16>,
    a: &[u16],
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(planar_image.width as u32, planar_image.height as u32)?;
    cfg.validate()?;
    planar_image.validate_with(cfg.chroma)?;
    validate_buf_u16(a, planar_image.width as u32, planar_image.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u16(
        &PlanarImage {
            width: planar_image.width,
            height: planar_image.height,
            bit_depth: BitDepth::Twelve,
            planes: [a.to_vec(), vec![], vec![]],
        },
        BitDepth::Twelve,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
    )?;
    let alpha_obu = encode_lossy_gray_obu(
        &PlanarImage {
            width: planar_image.width,
            height: planar_image.height,
            bit_depth: BitDepth::Twelve,
            planes: [a.to_vec(), vec![], vec![]],
        },
        BitDepth::Twelve,
        q,
        true,
        cfg.threads,
    )?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        planar_image.width as u32,
        planar_image.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

fn dispatch_yuv_u8(
    planar_image: &PlanarImage<u8>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&ColorEncoding>,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(planar_image, bd, q, color, threads),
        ChromaFormat::Yuv422 => encode_yuv422_obu(planar_image, bd, q, color, threads),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            encode_yuv444_obu(planar_image, bd, q, color, threads)
        }
    }
}

fn dispatch_yuv_u16(
    planar_image: &PlanarImage<u16>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&ColorEncoding>,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(planar_image, bd, q, color, threads),
        ChromaFormat::Yuv422 => encode_yuv422_obu(planar_image, bd, q, color, threads),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            encode_yuv444_obu(planar_image, bd, q, color, threads)
        }
    }
}
