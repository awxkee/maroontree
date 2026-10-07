/*
 * // Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! HDR gain maps (ISO 21496-1) for AVIF, laid out the way libavif writes them:
//!
//! ```text
//!   item 1  av01  "Color"  primary, the base rendition (pitm)
//!  [item 2  av01  "Alpha"  auxl -> 1]
//!   item N  tmap  "GMap"   derived tone-mapped image: version 0 + ISO 21496-1
//!                          metadata; dimg -> [1, N+1]; carries the alternate
//!                          rendition's ispe/colr/[pixi]/[clli]
//!   item N+1 av01 "GMap"   hidden gain map image
//!   grpl/altr  [N, 1]      tmap preferred, base is the fallback
//!   ftyp compatible brand `tmap`
//! ```
//!
//! A decoder that does not understand `tmap` shows the primary item (the base
//! rendition); a gain-map aware one (libavif, Chrome, Apple) reconstructs the
//! alternate rendition from the base, the gain map and the metadata.
//!
//! The metadata blob uses the *final* ISO 21496-1 syntax that libavif reads in
//! a `tmap` item: after the two flag bits it expects six reserved bits and an
//! explicit numerator/denominator for every value. The Ultra HDR / libultrahdr
//! extensions (common-denominator and backward-direction flag bits) are not
//! representable: an HDR base is expressed by
//! `base_hdr_headroom > alternate_hdr_headroom` instead.

use crate::avif::{ChromaFormat, EncodeConfig};
use crate::color::{Cicp, MatrixCoefficients, Primaries, TransferFunction};
use crate::encoder::PlanarImage;
use crate::err::EncodeError;
use crate::metadata::ContentLightLevel;
use crate::pixel::{BitDepth, Pixel};
use std::fmt;

/// Binary ISO 21496-1 gain map metadata.
///
/// Every value is a rational: `*_n` is the numerator, `*_d` the denominator.
/// Per-channel arrays are `[R, G, B]`; three identical channels are written
/// in the single-channel form.
///
/// Build one with [`IsoGainMap::from_floats`] or fill the fields directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IsoGainMap {
    /// log2 of the minimum gain (per channel).
    pub gain_map_min_n: [i32; 3],
    pub gain_map_min_d: [u32; 3],
    /// log2 of the maximum gain (per channel).
    pub gain_map_max_n: [i32; 3],
    pub gain_map_max_d: [u32; 3],
    /// Gamma applied to the stored gain map samples (per channel).
    pub gain_map_gamma_n: [u32; 3],
    pub gain_map_gamma_d: [u32; 3],
    /// Offset added to the base image before applying the gain (per channel).
    pub base_offset_n: [i32; 3],
    pub base_offset_d: [u32; 3],
    /// Offset added to the alternate image before applying the gain (per channel).
    pub alternate_offset_n: [i32; 3],
    pub alternate_offset_d: [u32; 3],
    /// log2 of the base rendition's HDR headroom (`0` for an SDR base).
    pub base_hdr_headroom_n: u32,
    pub base_hdr_headroom_d: u32,
    /// log2 of the alternate rendition's HDR headroom.
    pub alternate_hdr_headroom_n: u32,
    pub alternate_hdr_headroom_d: u32,
    /// The gain is applied in the base rendition's color space (`false`: in
    /// the alternate rendition's color space).
    pub use_base_color_space: bool,
}

impl Default for IsoGainMap {
    /// Identity gain map: gain 1 everywhere, gamma 1, no offsets, no headroom.
    /// Same defaults as libavif's `avifGainMapSetDefaults`.
    fn default() -> Self {
        Self {
            gain_map_min_n: [0; 3],
            gain_map_min_d: [1; 3],
            gain_map_max_n: [0; 3],
            gain_map_max_d: [1; 3],
            gain_map_gamma_n: [1; 3],
            gain_map_gamma_d: [1; 3],
            base_offset_n: [0; 3],
            base_offset_d: [1; 3],
            alternate_offset_n: [0; 3],
            alternate_offset_d: [1; 3],
            base_hdr_headroom_n: 0,
            base_hdr_headroom_d: 1,
            alternate_hdr_headroom_n: 0,
            alternate_hdr_headroom_d: 1,
            use_base_color_space: true,
        }
    }
}

/// Floating-point gain map parameters, the convenient way to build an
/// [`IsoGainMap`]. Per-channel arrays are `[R, G, B]`; pass three identical
/// values for a single-channel (luminance) gain map.
///
/// `min`/`max` are `log2` of the smallest/largest gain in the map,
/// `base_hdr_headroom`/`alternate_hdr_headroom` are `log2` of each rendition's
/// headroom (`0.0` for an SDR rendition), offsets are in normalized linear
/// units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GainMapFloats {
    pub min: [f32; 3],
    pub max: [f32; 3],
    pub gamma: [f32; 3],
    pub base_offset: [f32; 3],
    pub alternate_offset: [f32; 3],
    pub base_hdr_headroom: f32,
    pub alternate_hdr_headroom: f32,
    pub use_base_color_space: bool,
}

impl Default for GainMapFloats {
    /// Identity gain map, gamma 1, Ultra HDR default offsets (1/64).
    fn default() -> Self {
        Self {
            min: [0.0; 3],
            max: [0.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 0.0,
            use_base_color_space: true,
        }
    }
}

fn bad(msg: &'static str) -> EncodeError {
    EncodeError::InvalidGainMap(msg)
}

/// Continued-fraction approximation of `v` as `n/d` with `n <= max_numerator`
/// (libavif `avifDoubleToUnsignedFractionImpl`, via gainforge/jixel).
fn float_to_unsigned_fraction_impl(v: f32, max_numerator: u32) -> Option<(u32, u32)> {
    if v.is_nan() || v < 0.0 || v > max_numerator as f32 {
        return None;
    }
    let max_d = if v <= 1.0 {
        u32::MAX as u64
    } else {
        (max_numerator as f64 / v.floor() as f64) as u64
    };
    let mut denominator: u32 = 1;
    let mut previous_d: u32 = 0;
    let mut current_v = v.fract() as f64;
    for _ in 0..39 {
        let numerator_double = (denominator as f64) * (v as f64);
        if numerator_double > max_numerator as f64 {
            return None;
        }
        let numerator = numerator_double.round() as u32;
        if (numerator_double - numerator as f64).abs() == 0.0 {
            return Some((numerator, denominator));
        }
        current_v = 1.0 / current_v;
        let new_d = previous_d as u64 + (current_v.floor() as u64) * (denominator as u64);
        if new_d > max_d {
            return Some((numerator, denominator));
        }
        previous_d = denominator;
        if new_d > u32::MAX as u64 {
            return None;
        }
        denominator = new_d as u32;
        current_v -= current_v.floor();
    }
    let numerator = ((denominator as f64) * (v as f64)).round() as u32;
    Some((numerator, denominator))
}

fn float_to_signed_fraction(v: f32) -> Option<(i32, u32)> {
    let (n, d) = float_to_unsigned_fraction_impl(v.abs(), i32::MAX as u32)?;
    Some((if v < 0.0 { -(n as i32) } else { n as i32 }, d))
}

fn float_to_unsigned_fraction(v: f32) -> Option<(u32, u32)> {
    float_to_unsigned_fraction_impl(v, i32::MAX as u32)
}

const IS_MULTICHANNEL_MASK: u8 = 1 << 7;
const USE_BASE_COLORSPACE_MASK: u8 = 1 << 6;

impl IsoGainMap {
    /// Build the metadata from floating-point parameters. Fails when a value
    /// is NaN, out of range, or negative where the standard requires a
    /// non-negative number (gamma, headroom).
    pub fn from_floats(v: &GainMapFloats) -> Result<Self, EncodeError> {
        let mut m = IsoGainMap {
            use_base_color_space: v.use_base_color_space,
            ..IsoGainMap::default()
        };
        for c in 0..3 {
            (m.gain_map_min_n[c], m.gain_map_min_d[c]) =
                float_to_signed_fraction(v.min[c]).ok_or_else(|| bad("gain map min"))?;
            (m.gain_map_max_n[c], m.gain_map_max_d[c]) =
                float_to_signed_fraction(v.max[c]).ok_or_else(|| bad("gain map max"))?;
            if v.gamma[c].is_nan() || v.gamma[c] <= 0.0 {
                return Err(bad("gain map gamma must be positive"));
            }
            (m.gain_map_gamma_n[c], m.gain_map_gamma_d[c]) =
                float_to_unsigned_fraction(v.gamma[c]).ok_or_else(|| bad("gain map gamma"))?;
            (m.base_offset_n[c], m.base_offset_d[c]) =
                float_to_signed_fraction(v.base_offset[c]).ok_or_else(|| bad("base offset"))?;
            (m.alternate_offset_n[c], m.alternate_offset_d[c]) =
                float_to_signed_fraction(v.alternate_offset[c])
                    .ok_or_else(|| bad("alternate offset"))?;
        }
        (m.base_hdr_headroom_n, m.base_hdr_headroom_d) =
            float_to_unsigned_fraction(v.base_hdr_headroom)
                .ok_or_else(|| bad("base HDR headroom must be finite and non-negative"))?;
        (m.alternate_hdr_headroom_n, m.alternate_hdr_headroom_d) =
            float_to_unsigned_fraction(v.alternate_hdr_headroom)
                .ok_or_else(|| bad("alternate HDR headroom must be finite and non-negative"))?;
        m.validate()?;
        Ok(m)
    }

    /// Floating-point view of the parameters.
    pub fn to_floats(&self) -> GainMapFloats {
        let s = |n: &[i32; 3], d: &[u32; 3]| std::array::from_fn(|c| n[c] as f32 / d[c] as f32);
        let u = |n: &[u32; 3], d: &[u32; 3]| std::array::from_fn(|c| n[c] as f32 / d[c] as f32);
        GainMapFloats {
            min: s(&self.gain_map_min_n, &self.gain_map_min_d),
            max: s(&self.gain_map_max_n, &self.gain_map_max_d),
            gamma: u(&self.gain_map_gamma_n, &self.gain_map_gamma_d),
            base_offset: s(&self.base_offset_n, &self.base_offset_d),
            alternate_offset: s(&self.alternate_offset_n, &self.alternate_offset_d),
            base_hdr_headroom: self.base_hdr_headroom_n as f32 / self.base_hdr_headroom_d as f32,
            alternate_hdr_headroom: self.alternate_hdr_headroom_n as f32
                / self.alternate_hdr_headroom_d as f32,
            use_base_color_space: self.use_base_color_space,
        }
    }

    /// True when a per-channel parameter differs between channels, i.e. the
    /// metadata must be written in the three-channel form.
    pub fn is_multichannel(&self) -> bool {
        fn same<A: PartialEq + Copy>(n: &[A; 3], d: &[u32; 3]) -> bool {
            n[0] == n[1] && n[1] == n[2] && d[0] == d[1] && d[1] == d[2]
        }
        !(same(&self.gain_map_min_n, &self.gain_map_min_d)
            && same(&self.gain_map_max_n, &self.gain_map_max_d)
            && same(&self.gain_map_gamma_n, &self.gain_map_gamma_d)
            && same(&self.base_offset_n, &self.base_offset_d)
            && same(&self.alternate_offset_n, &self.alternate_offset_d))
    }

    /// The checks libavif's `avifGainMapValidateMetadata` applies on both
    /// encode and decode: no zero denominator, `max >= min`, non-zero gamma.
    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        for c in 0..3 {
            if self.gain_map_min_d[c] == 0
                || self.gain_map_max_d[c] == 0
                || self.gain_map_gamma_d[c] == 0
                || self.base_offset_d[c] == 0
                || self.alternate_offset_d[c] == 0
            {
                return Err(bad("zero denominator in a per-channel parameter"));
            }
            if (self.gain_map_max_n[c] as i64) * (self.gain_map_min_d[c] as i64)
                < (self.gain_map_min_n[c] as i64) * (self.gain_map_max_d[c] as i64)
            {
                return Err(bad("gain map max is less than gain map min"));
            }
            if self.gain_map_gamma_n[c] == 0 {
                return Err(bad("gain map gamma must be non-zero"));
            }
        }
        if self.base_hdr_headroom_d == 0 || self.alternate_hdr_headroom_d == 0 {
            return Err(bad("zero denominator in HDR headroom"));
        }
        Ok(())
    }

    /// ISO 21496-1 `GainMapMetadata` (clause C.2.2), big-endian:
    ///
    /// ```text
    /// u16 minimum_version = 0
    /// u16 writer_version  = 0
    /// u8  is_multichannel(1) use_base_colour_space(1) reserved(6)
    /// u32 base_hdr_headroom n, d      u32 alternate_hdr_headroom n, d
    /// per channel (1 or 3): min n(s32),d  max n(s32),d  gamma n,d
    ///                       base_offset n(s32),d  alternate_offset n(s32),d
    /// ```
    pub fn to_metadata(&self) -> Result<Vec<u8>, EncodeError> {
        self.validate()?;
        let channels = if self.is_multichannel() { 3 } else { 1 };
        let mut out = Vec::with_capacity(5 + 16 + 40 * channels);
        out.extend_from_slice(&0u16.to_be_bytes()); // minimum_version
        out.extend_from_slice(&0u16.to_be_bytes()); // writer_version
        let mut flags = 0u8;
        if channels == 3 {
            flags |= IS_MULTICHANNEL_MASK;
        }
        if self.use_base_color_space {
            flags |= USE_BASE_COLORSPACE_MASK;
        }
        out.push(flags);
        for v in [
            self.base_hdr_headroom_n,
            self.base_hdr_headroom_d,
            self.alternate_hdr_headroom_n,
            self.alternate_hdr_headroom_d,
        ] {
            out.extend_from_slice(&v.to_be_bytes());
        }
        for c in 0..channels {
            for v in [
                self.gain_map_min_n[c] as u32,
                self.gain_map_min_d[c],
                self.gain_map_max_n[c] as u32,
                self.gain_map_max_d[c],
                self.gain_map_gamma_n[c],
                self.gain_map_gamma_d[c],
                self.base_offset_n[c] as u32,
                self.base_offset_d[c],
                self.alternate_offset_n[c] as u32,
                self.alternate_offset_d[c],
            ] {
                out.extend_from_slice(&v.to_be_bytes());
            }
        }
        Ok(out)
    }

    /// The `tmap` item payload: `ToneMapImage` (ISO/IEC 23008-12:2024 AMD 1
    /// 6.6.2.4.2) = `u8 version = 0` followed by [`Self::to_metadata`].
    pub(crate) fn tmap_payload(&self) -> Result<Vec<u8>, EncodeError> {
        let mut out = vec![0u8]; // version
        out.extend_from_slice(&self.to_metadata()?);
        Ok(out)
    }

    /// Parse [`Self::to_metadata`]'s layout (the final ISO 21496-1 syntax; the
    /// reserved flag bits are ignored, as libavif does).
    pub fn from_metadata(data: &[u8]) -> Result<Self, EncodeError> {
        let mut pos = 0usize;
        let u32_at = |pos: &mut usize| -> Result<u32, EncodeError> {
            let s = data
                .get(*pos..*pos + 4)
                .ok_or_else(|| bad("gain map metadata truncated"))?;
            *pos += 4;
            Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
        };
        if data.len() < 5 {
            return Err(bad("gain map metadata truncated"));
        }
        let minimum_version = u16::from_be_bytes([data[0], data[1]]);
        let writer_version = u16::from_be_bytes([data[2], data[3]]);
        if minimum_version != 0 || writer_version < minimum_version {
            return Err(bad("unsupported gain map metadata version"));
        }
        let flags = data[4];
        pos += 5;
        let channels = if flags & IS_MULTICHANNEL_MASK != 0 {
            3
        } else {
            1
        };
        let mut m = IsoGainMap {
            use_base_color_space: flags & USE_BASE_COLORSPACE_MASK != 0,
            base_hdr_headroom_n: u32_at(&mut pos)?,
            base_hdr_headroom_d: u32_at(&mut pos)?,
            alternate_hdr_headroom_n: u32_at(&mut pos)?,
            alternate_hdr_headroom_d: u32_at(&mut pos)?,
            ..IsoGainMap::default()
        };
        for c in 0..channels {
            m.gain_map_min_n[c] = u32_at(&mut pos)? as i32;
            m.gain_map_min_d[c] = u32_at(&mut pos)?;
            m.gain_map_max_n[c] = u32_at(&mut pos)? as i32;
            m.gain_map_max_d[c] = u32_at(&mut pos)?;
            m.gain_map_gamma_n[c] = u32_at(&mut pos)?;
            m.gain_map_gamma_d[c] = u32_at(&mut pos)?;
            m.base_offset_n[c] = u32_at(&mut pos)? as i32;
            m.base_offset_d[c] = u32_at(&mut pos)?;
            m.alternate_offset_n[c] = u32_at(&mut pos)? as i32;
            m.alternate_offset_d[c] = u32_at(&mut pos)?;
        }
        for c in channels..3 {
            m.gain_map_min_n[c] = m.gain_map_min_n[0];
            m.gain_map_min_d[c] = m.gain_map_min_d[0];
            m.gain_map_max_n[c] = m.gain_map_max_n[0];
            m.gain_map_max_d[c] = m.gain_map_max_d[0];
            m.gain_map_gamma_n[c] = m.gain_map_gamma_n[0];
            m.gain_map_gamma_d[c] = m.gain_map_gamma_d[0];
            m.base_offset_n[c] = m.base_offset_n[0];
            m.base_offset_d[c] = m.base_offset_d[0];
            m.alternate_offset_n[c] = m.alternate_offset_n[0];
            m.alternate_offset_d[c] = m.alternate_offset_d[0];
        }
        if writer_version == 0 && pos != data.len() {
            return Err(bad("trailing bytes after gain map metadata"));
        }
        m.validate()?;
        Ok(m)
    }
}

/// Samples of a gain map image. Gray maps carry one luminance gain; RGB maps
/// carry per-channel gain and use the same GBR plane layout as the color
/// entry points ([`PlanarImage::from_interleaved_rgb`]). Depth comes from the
/// image's [`BitDepth`] (8, 10 or 12 bits).
#[derive(Clone)]
pub enum GainMapImage {
    Gray8(PlanarImage<u8>),
    Gray16(PlanarImage<u16>),
    Rgb8(PlanarImage<u8>),
    Rgb16(PlanarImage<u16>),
}

impl GainMapImage {
    pub fn width(&self) -> usize {
        match self {
            Self::Gray8(i) | Self::Rgb8(i) => i.width,
            Self::Gray16(i) | Self::Rgb16(i) => i.width,
        }
    }

    pub fn height(&self) -> usize {
        match self {
            Self::Gray8(i) | Self::Rgb8(i) => i.height,
            Self::Gray16(i) | Self::Rgb16(i) => i.height,
        }
    }

    pub fn bit_depth(&self) -> BitDepth {
        match self {
            Self::Gray8(i) | Self::Rgb8(i) => i.bit_depth,
            Self::Gray16(i) | Self::Rgb16(i) => i.bit_depth,
        }
    }

    /// Color channels per pixel (1 or 3).
    pub fn channels(&self) -> u8 {
        match self {
            Self::Gray8(_) | Self::Gray16(_) => 1,
            Self::Rgb8(_) | Self::Rgb16(_) => 3,
        }
    }
}

impl fmt::Debug for GainMapImage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match self {
            Self::Gray8(_) => "Gray8",
            Self::Gray16(_) => "Gray16",
            Self::Rgb8(_) => "Rgb8",
            Self::Rgb16(_) => "Rgb16",
        };
        write!(
            f,
            "GainMapImage::{kind} {{ {}x{}, {:?} }}",
            self.width(),
            self.height(),
            self.bit_depth()
        )
    }
}

/// An HDR gain map attached to an encode via [`EncodeConfig::with_gain_map`].
///
/// The encoded image stays the base rendition (the primary item every AVIF
/// decoder shows); the gain map is a second, hidden AV1 item, and a `tmap`
/// derived item carries the [`IsoGainMap`] metadata and the color description
/// of the alternate (usually HDR) rendition it reconstructs. Its dimensions
/// are independent of the base image (quarter-resolution maps are common).
///
/// ```no_run
/// use maroontree::{BitDepth, EncodeConfig, GainMap, GainMapFloats, IsoGainMap, PlanarImage};
/// # let (base, gain, w, h): (PlanarImage<u8>, Vec<u8>, usize, usize) = todo!();
/// let metadata = IsoGainMap::from_floats(&GainMapFloats {
///     max: [2.0; 3],               // log2 of the largest gain
///     alternate_hdr_headroom: 2.0, // log2 of the HDR rendition's headroom
///     ..GainMapFloats::default()
/// })?;
/// let map = PlanarImage::from_luma(w / 4, h / 4, BitDepth::Eight, &gain)?;
/// let cfg = EncodeConfig::new().with_gain_map(GainMap::gray8(map, metadata).with_quality(70));
/// let avif = maroontree::encode_rgb8(&base, &cfg)?;
/// # Ok::<(), maroontree::EncodeError>(())
/// ```
#[derive(Debug, Clone)]
pub struct GainMap {
    /// Gain map samples.
    pub image: GainMapImage,
    /// ISO 21496-1 metadata describing how to apply the map.
    pub metadata: IsoGainMap,
    /// Quality of the gain map item (same scale as [`EncodeConfig::quality`]).
    /// `None` inherits the base image's quality, like libavif's
    /// `qualityGainMap` default.
    pub quality: Option<u8>,
    /// Chroma subsampling of an RGB gain map (default 4:4:4). Ignored for gray
    /// maps, which are always coded monochrome.
    pub chroma: ChromaFormat,
    /// CICP of the alternate rendition, written as the `tmap` item's `nclx`
    /// `colr`. `None` writes libavif's default: unspecified 2/2/2, full range.
    pub alternate_cicp: Option<Cicp>,
    /// ICC profile of the alternate rendition (`tmap` `prof` `colr`).
    pub alternate_icc_profile: Option<Vec<u8>>,
    /// Content light level of the alternate rendition (`tmap` `clli`).
    pub alternate_content_light_level: Option<ContentLightLevel>,
    /// `(channels, bit depth)` of the alternate rendition, written as the
    /// `tmap` item's optional `pixi`. `None` omits it.
    pub alternate_pixel_info: Option<(u8, u8)>,
}

impl GainMap {
    fn new(image: GainMapImage, metadata: IsoGainMap) -> Self {
        Self {
            image,
            metadata,
            quality: None,
            chroma: ChromaFormat::Yuv444,
            alternate_cicp: None,
            alternate_icc_profile: None,
            alternate_content_light_level: None,
            alternate_pixel_info: None,
        }
    }

    /// 8-bit single-channel gain map (`image.planes[0]` holds the samples).
    pub fn gray8(image: PlanarImage<u8>, metadata: IsoGainMap) -> Self {
        Self::new(GainMapImage::Gray8(image), metadata)
    }

    /// 10/12-bit single-channel gain map; depth from `image.bit_depth`.
    pub fn gray16(image: PlanarImage<u16>, metadata: IsoGainMap) -> Self {
        Self::new(GainMapImage::Gray16(image), metadata)
    }

    /// 8-bit RGB gain map in GBR plane layout.
    pub fn rgb8(image: PlanarImage<u8>, metadata: IsoGainMap) -> Self {
        Self::new(GainMapImage::Rgb8(image), metadata)
    }

    /// 10/12-bit RGB gain map in GBR plane layout; depth from `image.bit_depth`.
    pub fn rgb16(image: PlanarImage<u16>, metadata: IsoGainMap) -> Self {
        Self::new(GainMapImage::Rgb16(image), metadata)
    }

    /// Quality of the gain map item (0..=100).
    pub fn with_quality(mut self, quality: u8) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Chroma subsampling of an RGB gain map.
    pub fn with_chroma(mut self, chroma: ChromaFormat) -> Self {
        self.chroma = chroma;
        self
    }

    /// CICP of the alternate rendition (e.g. [`Cicp::bt2020_pq`]).
    pub fn with_alternate_cicp(mut self, cicp: Cicp) -> Self {
        self.alternate_cicp = Some(cicp);
        self
    }

    /// ICC profile of the alternate rendition.
    pub fn with_alternate_icc_profile(mut self, icc: Vec<u8>) -> Self {
        self.alternate_icc_profile = Some(icc);
        self
    }

    /// Content light level of the alternate rendition.
    pub fn with_alternate_content_light_level(mut self, cll: ContentLightLevel) -> Self {
        self.alternate_content_light_level = Some(cll);
        self
    }

    /// Channel count (1 or 3) and bit depth of the alternate rendition.
    pub fn with_alternate_pixel_info(mut self, channels: u8, bit_depth: u8) -> Self {
        self.alternate_pixel_info = Some((channels, bit_depth));
        self
    }

    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        self.metadata.validate()?;
        let (w, h) = (self.image.width() as u32, self.image.height() as u32);
        crate::avif::validate_dims(w, h)?;
        match &self.image {
            GainMapImage::Gray8(i) => i.validate_400()?,
            GainMapImage::Gray16(i) => i.validate_400()?,
            GainMapImage::Rgb8(i) => i.validate_444()?,
            GainMapImage::Rgb16(i) => i.validate_444()?,
        }
        match &self.image {
            GainMapImage::Gray8(i) | GainMapImage::Rgb8(i) if i.bit_depth != BitDepth::Eight => {
                return Err(EncodeError::UnsupportedChromaBitDepth(i.bit_depth));
            }
            GainMapImage::Gray16(i) | GainMapImage::Rgb16(i) if i.bit_depth == BitDepth::Eight => {
                return Err(EncodeError::UnsupportedChromaBitDepth(i.bit_depth));
            }
            _ => {}
        }
        if self.image.channels() == 3 && self.chroma == ChromaFormat::Monochrome {
            return Err(bad(
                "an RGB gain map cannot be coded monochrome; use a gray map",
            ));
        }
        if self.quality.is_some_and(|q| q > 100) {
            return Err(EncodeError::InvalidQuality);
        }
        if matches!(self.alternate_icc_profile.as_deref(), Some(&[])) {
            return Err(bad("alternate ICC profile is empty"));
        }
        if let Some((channels, depth)) = self.alternate_pixel_info
            && (!matches!(channels, 1 | 3) || depth == 0)
        {
            return Err(bad(
                "alternate pixel info must be 1 or 3 channels of non-zero depth",
            ));
        }
        Ok(())
    }
}

/// CICP the gain map item is signalled with. libavif requires unspecified
/// primaries and transfer (2/2); RGB maps go through the encoder's fixed
/// BT.601 full-range RGB->YCbCr, so that matrix is what the item declares.
fn gain_map_cicp(channels: u8) -> Cicp {
    Cicp {
        primaries: Primaries::Unspecified,
        transfer: TransferFunction::Unspecified,
        matrix: if channels == 1 {
            MatrixCoefficients::Unspecified
        } else {
            MatrixCoefficients::Smpte170m
        },
        full_range: true,
        chroma_sample_position: crate::color::ChromaSamplePosition::Unknown,
    }
}

/// A gain map coded into its AV1 item, ready for the container.
pub(crate) struct EncodedGainMap {
    pub obu: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub chroma: ChromaFormat,
    pub cicp: Cicp,
    pub tmap_payload: Vec<u8>,
}

fn encode_gray_map<T: Pixel>(
    img: &PlanarImage<T>,
    q: u8,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    crate::encoder::encode_lossy_gray_obu(
        &img.packed_1(),
        img.bit_depth,
        q,
        true,
        cfg.threads,
        cfg.speed,
        cfg.adaptive_quant,
        cfg.vb(q),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )
}

fn encode_rgb_map<T: Pixel>(
    img: &PlanarImage<T>,
    q: u8,
    chroma: ChromaFormat,
    cicp: &Cicp,
    cfg: &EncodeConfig,
) -> Vec<u8> {
    crate::avif::dispatch_lossy(
        &img.packed_3(),
        q,
        chroma,
        Some(cicp),
        cfg.threads,
        cfg.speed,
        cfg.adaptive_quant,
        cfg.vb(q),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )
}

/// Code the gain map attached to `cfg` with the base encode's tools (speed,
/// threads, AQ, filters) at the gain map's own quality.
pub(crate) fn encode_gain_map(
    gm: &GainMap,
    cfg: &EncodeConfig,
) -> Result<EncodedGainMap, EncodeError> {
    gm.validate()?;
    let quality = gm.quality.unwrap_or(cfg.quality);
    // Quality 100 maps to the lossless qindex 0 on the gray path only; keep
    // the gain map lossy (q 1 is visually lossless) so both paths agree.
    let q = crate::avif::quality_to_q(quality).max(1);
    let cicp = gain_map_cicp(gm.image.channels());
    let (obu, chroma) = match &gm.image {
        GainMapImage::Gray8(i) => (encode_gray_map(i, q, cfg)?, ChromaFormat::Monochrome),
        GainMapImage::Gray16(i) => (encode_gray_map(i, q, cfg)?, ChromaFormat::Monochrome),
        GainMapImage::Rgb8(i) => (encode_rgb_map(i, q, gm.chroma, &cicp, cfg), gm.chroma),
        GainMapImage::Rgb16(i) => (encode_rgb_map(i, q, gm.chroma, &cicp, cfg), gm.chroma),
    };
    Ok(EncodedGainMap {
        obu,
        width: gm.image.width() as u32,
        height: gm.image.height() as u32,
        bit_depth: gm.image.bit_depth().bits(),
        chroma,
        cicp,
        tmap_payload: gm.metadata.tmap_payload()?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> GainMapFloats {
        GainMapFloats {
            min: [-0.5, -0.25, 0.0],
            max: [2.0, 2.5, 3.0],
            gamma: [1.0, 1.2, 0.8],
            alternate_hdr_headroom: 2.32,
            ..GainMapFloats::default()
        }
    }

    #[test]
    fn multichannel_round_trip_is_exact() {
        let m = IsoGainMap::from_floats(&sample()).unwrap();
        assert!(m.is_multichannel());
        let bytes = m.to_metadata().unwrap();
        assert_eq!(bytes.len(), 5 + 16 + 3 * 40);
        assert_eq!(bytes[4], IS_MULTICHANNEL_MASK | USE_BASE_COLORSPACE_MASK);
        assert_eq!(IsoGainMap::from_metadata(&bytes).unwrap(), m);
    }

    #[test]
    fn single_channel_matches_libavif_layout() {
        // Hand-assembled libavif avifWriteGainmapMetadata output: explicit
        // denominators, reserved flag bits zero, one channel.
        let mut b = vec![0, 0, 0, 0, USE_BASE_COLORSPACE_MASK];
        for v in [0u32, 1, 3, 1, (-1i32) as u32, 2, 3, 1, 1, 1, 1, 64, 1, 64] {
            b.extend_from_slice(&v.to_be_bytes());
        }
        let m = IsoGainMap::from_metadata(&b).unwrap();
        assert!(!m.is_multichannel());
        assert_eq!(m.gain_map_min_n, [-1; 3]);
        assert_eq!(m.gain_map_min_d, [2; 3]);
        assert_eq!(m.alternate_hdr_headroom_n, 3);
        assert_eq!(m.to_metadata().unwrap(), b);
        let tmap = m.tmap_payload().unwrap();
        assert_eq!(tmap[0], 0);
        assert_eq!(&tmap[1..], b.as_slice());
    }

    #[test]
    fn floats_survive_the_rational_round_trip() {
        let f = sample();
        let back = IsoGainMap::from_floats(&f).unwrap().to_floats();
        for c in 0..3 {
            assert!((back.min[c] - f.min[c]).abs() < 1e-6);
            assert!((back.max[c] - f.max[c]).abs() < 1e-6);
            assert!((back.gamma[c] - f.gamma[c]).abs() < 1e-6);
            assert!((back.base_offset[c] - f.base_offset[c]).abs() < 1e-6);
        }
        assert!((back.alternate_hdr_headroom - f.alternate_hdr_headroom).abs() < 1e-6);
    }

    #[test]
    fn rejects_what_libavif_rejects() {
        let gamma0 = GainMapFloats {
            gamma: [1.0, 0.0, 1.0],
            ..GainMapFloats::default()
        };
        assert!(IsoGainMap::from_floats(&gamma0).is_err());
        let inverted = GainMapFloats {
            min: [1.0; 3],
            max: [0.5; 3],
            ..GainMapFloats::default()
        };
        assert!(IsoGainMap::from_floats(&inverted).is_err());
        let mut zero = IsoGainMap::default();
        zero.base_offset_d[1] = 0;
        assert!(zero.to_metadata().is_err());
        let bytes = IsoGainMap::from_floats(&sample())
            .unwrap()
            .to_metadata()
            .unwrap();
        assert!(IsoGainMap::from_metadata(&bytes[..bytes.len() - 1]).is_err());
    }

    #[test]
    fn validation_catches_bad_gain_maps() {
        let m = IsoGainMap::default();
        let gray = PlanarImage::from_luma(4, 4, BitDepth::Eight, &[0u8; 16]).unwrap();
        assert!(GainMap::gray8(gray.clone(), m).validate().is_ok());
        assert!(
            GainMap::gray8(gray.clone(), m)
                .with_quality(101)
                .validate()
                .is_err()
        );
        let mut short = gray.clone();
        short.planes[0].pop();
        assert!(GainMap::gray8(short, m).validate().is_err());
        let wide = PlanarImage::from_luma(2, 2, BitDepth::Eight, &[0u16; 4]).unwrap();
        assert!(GainMap::gray16(wide, m).validate().is_err());
        let rgb = PlanarImage::from_interleaved_rgb(2, 2, BitDepth::Eight, &[0u8; 12]).unwrap();
        assert!(
            GainMap::rgb8(rgb, m)
                .with_chroma(ChromaFormat::Monochrome)
                .validate()
                .is_err()
        );
        assert!(
            GainMap::gray8(gray, m)
                .with_alternate_icc_profile(vec![])
                .validate()
                .is_err()
        );
    }
}
