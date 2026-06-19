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

use crate::color::Cicp;
use crate::encoder::{
    encode_lossy_gray_obu, encode_still_lossy, encode_still_lossy_420, encode_still_lossy_422,
    encode_yuv420_obu, encode_yuv422_obu, encode_yuv444_obu,
};
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Metadata, Orientation};
use crate::{BitDepth, PlanarImage, encode_lossless_gray_obu, isobmff};

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

/// Rate-distortion effort for the encoder's mode search.
///
/// Higher effort spends more time per block searching for the smallest stream at
/// a given quality; faster effort trades a little compression for speed using
/// libaom-style shortcuts. Currently honoured only by the AV1 lossy path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Speed {
    /// Exhaustive rate-distortion
    #[default]
    Slow,
    /// Balanced: RDOQ is run once on the chosen mode only
    Medium,
    /// Fast path
    Fast,
}

impl Speed {
    /// Whether RDOQ (trellis) is run for *every* candidate during the mode
    /// search. Only [`Speed::Slow`] does; the faster tiers apply RDOQ once, to
    /// the winning mode.
    pub(crate) fn per_candidate_rdoq(self) -> bool {
        matches!(self, Speed::Slow)
    }

    /// Whether the winning mode is refined with an ADST_ADST transform-type
    /// search. Off only for [`Speed::Fast`] (DCT-only).
    pub(crate) fn try_adst(self) -> bool {
        !matches!(self, Speed::Fast)
    }

    /// Whether the directional angle-delta search (Δ = ±1..±3) is run. Only
    /// [`Speed::Slow`] does; the faster tiers trial nominal angles (Δ=0) only,
    /// since each directional mode otherwise costs 7× the RD candidates.
    pub(crate) fn try_angle_deltas(self) -> bool {
        matches!(self, Speed::Slow)
    }

    /// Whether the intra candidate set is reduced. Only [`Speed::Fast`] does.
    pub(crate) fn reduced_modes(self) -> bool {
        matches!(self, Speed::Fast)
    }
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
    pub color_encoding: Option<Cicp>,
    pub icc: Option<Vec<u8>>,
    /// Optional image metadata (orientation, HDR content light level, EXIF).
    pub metadata: Metadata,
    /// Worker threads for tile-level parallelism.
    /// `0` = all available cores; `1` = serial (default); `N` = up to N.
    pub threads: usize,
    /// RDO effort (AV1 lossy path). See [`Speed`]; defaults to [`Speed::Slow`].
    pub speed: Speed,
    pub adaptive_quant: bool,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 80,
            chroma: ChromaFormat::Yuv420,
            color_encoding: Some(Cicp::srgb_ycbcr()),
            icc: None,
            metadata: Metadata::default(),
            threads: 1,
            speed: Speed::Slow,
            adaptive_quant: true,
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

    pub fn with_cicp(mut self, color: Cicp) -> Self {
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

    /// Set the RDO effort level (AV1 lossy path). See [`Speed`].
    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    pub fn with_adaptive_quant(mut self, v: bool) -> Self {
        self.adaptive_quant = v;
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

pub(crate) fn validate_buf<T>(buf: &[T], w: u32, h: u32, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<T>(w as usize, h as usize, ch)?;
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
pub(crate) fn finalize_color(
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
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
) -> Vec<u8> {
    match chroma {
        ChromaFormat::Yuv420 | ChromaFormat::Monochrome => {
            encode_still_lossy_420(img, q, color, threads, speed)
        }
        ChromaFormat::Yuv422 => encode_still_lossy_422(img, q, color, threads, speed),
        ChromaFormat::Yuv444 => encode_still_lossy(img, q, color, threads, speed),
    }
}

/// Encode an 8-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` bytes in R, G, B order.
pub fn encode_rgb8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(obu, img.width as u32, img.height as u32, 8, cfg.chroma, cfg)
}

/// Encode an 8-bit RGBA image to AVIF. The alpha channel is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` bytes in R, G, B, A order.
pub fn encode_rgba8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(obu, img.width as u32, img.height as u32, 8, cfg.chroma, cfg)
}

/// Encode an 8-bit RGBA image to AVIF with a separate alpha auxiliary image.
///
/// Produces two `av01` items in the container: a color image and a monochrome
/// alpha image linked by an `auxl` reference. `rgba` must hold exactly
/// `width * height * 4` bytes in R, G, B, A order.
pub fn encode_rgba8_with_alpha(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_lossy(
        &img.packed_3(),
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        8,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 10-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` `u16` samples, each in `0..=1023`.
pub fn encode_rgb10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 10-bit RGBA image to AVIF. Alpha is discarded.
///
/// `rgba` must hold exactly `width * height * 4` `u16` samples in `0..=1023`.
pub fn encode_rgba10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 10-bit RGBA image to AVIF with a separate alpha auxiliary image.
pub fn encode_rgba10_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_lossy(
        &img.packed_3(),
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 12-bit RGB image to AVIF.
///
/// `rgb` must hold exactly `width * height * 3` samples, each in `0..=4095`,
/// packed as `u16`.
pub fn encode_rgb12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 12-bit RGBA image to AVIF. Alpha is **discarded**.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    let obu = dispatch_lossy(
        &img.packed_3(),
        quality_to_q(cfg.quality),
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode a 12-bit RGBA image to AVIF with a separate alpha auxiliary image.
///
/// `rgba` must hold exactly `width * height * 4` samples in R, G, B, A order,
/// each in `0..=4095`, packed as `u16`.
pub fn encode_rgba12_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_lossy(
        &img.packed_3(),
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    );
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode an 8-bit grayscale image to AVIF using AV1 monochrome coding.
///
/// `gray` must hold exactly `width * height` bytes. The output uses
/// `mono_chrome = 1` (NumPlanes = 1) and AV1 profile 0.
pub fn encode_gray8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Eight,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        8,
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a 10-bit grayscale image to AVIF.
///
/// `gray` must hold exactly `width * height` `u16` samples in `0..=1023`.
pub fn encode_gray10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Ten,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        10,
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a 12-bit grayscale image to AVIF.
///
/// `gray` must hold exactly `width * height` `u16` samples in `0..=4095`.
pub fn encode_gray12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Twelve,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        12,
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a pre-converted 8-bit planar YCbCr image to AVIF.
///
/// `y` must be `width × height` bytes; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv8(img: &PlanarImage<u8>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u8(
        img,
        BitDepth::Eight,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(obu, img.width as u32, img.height as u32, 8, cfg.chroma, cfg)
}

/// Encode a pre-converted 10-bit planar YCbCr image to AVIF.
///
/// Each sample is a `u16` in `0..=1023`; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv10(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u16(
        img,
        BitDepth::Ten,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode a pre-converted 12-bit planar YCbCr image to AVIF.
///
/// Each sample is a `u16` in `0..=4095`; `cb`/`cr` must match `cfg.chroma`.
pub fn encode_yuv12(img: &PlanarImage<u16>, cfg: &EncodeConfig) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    let q = quality_to_q(cfg.quality);
    let obu = dispatch_yuv_u16(
        img,
        BitDepth::Twelve,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 8-bit YCbCr + a separate 8-bit alpha plane to AVIF.
///
/// `a` must be `width * height` bytes; YCbCr subsampling must match `cfg.chroma`.
pub fn encode_yuva8_with_alpha(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u8(
        &img.packed_3(),
        BitDepth::Eight,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        8,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 10-bit YCbCr + a separate 10-bit alpha plane to AVIF.
pub fn encode_yuva10_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u16(
        &img.packed_3(),
        BitDepth::Ten,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        10,
        cfg.chroma,
        cfg,
    )
}

/// Encode pre-converted 12-bit YCbCr + a separate 12-bit alpha plane to AVIF.
pub fn encode_yuva12_with_alpha(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    img.validate_with(cfg.chroma)?;
    validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;

    let q = quality_to_q(cfg.quality);
    let color_obu = dispatch_yuv_u16(
        &img.packed_3(),
        BitDepth::Twelve,
        q,
        cfg.chroma,
        cfg.color_encoding.as_ref(),
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_4(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        12,
        cfg.chroma,
        cfg,
    )
}

/// Encode an 8-bit grayscale image plus a separate 8-bit alpha plane to AVIF.
pub fn encode_gray_alpha8(
    img: &PlanarImage<u8>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Eight {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Eight,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_2(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        8,
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a 10-bit grayscale image plus a separate 10-bit alpha plane to AVIF.
pub fn encode_gray_alpha10(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Ten {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Ten,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_2(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        10,
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a 12-bit grayscale image plus a separate 12-bit alpha plane to AVIF.
pub fn encode_gray_alpha12(
    img: &PlanarImage<u16>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    if img.bit_depth != BitDepth::Twelve {
        return Err(EncodeError::UnsupportedChromaBitDepth(img.bit_depth));
    }
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    let q = quality_to_q(cfg.quality);
    let color_obu = encode_lossy_gray_obu(
        &img.packed_1(),
        BitDepth::Twelve,
        q,
        true,
        cfg.threads,
        cfg.speed,
    )?;
    let alpha_obu = encode_lossless_gray_obu(&img.packed_alpha_2(), true, cfg.threads)?;
    finalize_with_alpha(
        color_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        12,
        ChromaFormat::Monochrome,
        cfg,
    )
}

fn dispatch_yuv_u8(
    planar_image: &PlanarImage<u8>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(planar_image, bd, q, color, threads, speed),
        ChromaFormat::Yuv422 => encode_yuv422_obu(planar_image, bd, q, color, threads, speed),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            encode_yuv444_obu(planar_image, bd, q, color, threads, speed)
        }
    }
}

fn dispatch_yuv_u16(
    planar_image: &PlanarImage<u16>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(planar_image, bd, q, color, threads, speed),
        ChromaFormat::Yuv422 => encode_yuv422_obu(planar_image, bd, q, color, threads, speed),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => {
            encode_yuv444_obu(planar_image, bd, q, color, threads, speed)
        }
    }
}
