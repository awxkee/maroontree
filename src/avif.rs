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
    encode_lossless_gray_obu_with_cdf, encode_lossy_gray_obu, encode_still_lossy_420_with_cdf,
    encode_still_lossy_422_with_cdf, encode_still_lossy_with_cdf, encode_yuv420_obu,
    encode_yuv422_obu, encode_yuv444_obu,
};
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Metadata, Orientation};
use crate::{BitDepth, PlanarImage, isobmff};
use std::num::NonZeroUsize;

const MIN_DIM: u32 = 1;
/// Maximum dimension. AV1 level 6.3 handles frames up to 35 651 584 luma
/// samples; with both axes capped here the largest possible frame is ~268 MP.
const MAX_DIM: u32 = 16_383;

/// Chroma subsampling format for the AV1 encoder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// 4:2:0 — chroma halved both horizontally and vertically
    #[default]
    Yuv420,
    /// 4:2:2 — chroma halved horizontally only
    Yuv422,
    /// 4:4:4 — full-resolution chroma
    Yuv444,
    /// 4:0:0 — luma only; monochrome
    Monochrome,
}

/// Rate-distortion effort for the encoder's mode search
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Speed {
    /// Highest-effort rate-distortion search.
    #[default]
    Slow,
    /// Balanced: RDOQ is run once on the chosen mode only
    Medium,
    /// Fast path
    Fast,
}

impl Speed {
    /// Slow trellis-refines the compact model-ranked beam. Medium/Fast refine
    /// only the selected winner. (AV2 paths; the AV1 coder uses
    /// [`Self::per_candidate_rdoq_av1`].)
    pub(crate) fn per_candidate_rdoq(self) -> bool {
        matches!(self, Speed::Slow)
    }

    /// AV1-coder RDOQ staging. Split from [`Self::per_candidate_rdoq`] so the
    /// AV1 winner-only experiment cannot silently change AV2 behavior.
    /// MEASURED 2026-07-25 — winner-only at Slow is a BAD trade in every
    /// form: pure winner-only = 420 +1.25/+0.85/+1.76 (tuning/holdout/mid),
    /// 444 tuning +0.61 with UNIFORM kodak damage (+0.4..+1.5; the 5-image
    /// holdout's +0.01 was luck), for only ~4% time. Staged (plain-trellis
    /// candidates + ctx winner) is DOMINATED: worse than winner-only at 444
    /// (+1.71 holdout) and worse than per-candidate at 420. Per-candidate
    /// exact-ctx RDOQ at Slow EARNS its time; winner-only IS the Medium
    /// tier, and the delta is real quality, not fat.
    pub(crate) fn per_candidate_rdoq_av1(self) -> bool {
        matches!(self, Speed::Slow)
    }

    /// Whether the winning mode is refined with an ADST_ADST transform-type
    /// search. Off only for [`Speed::Fast`] (DCT-only).
    pub(crate) fn try_adst(self) -> bool {
        !matches!(self, Speed::Fast)
    }

    /// AV2 directional angle-delta control. Slow and Medium refine the nominal
    /// direction; Fast keeps Δ=0.
    pub(crate) fn try_angle_deltas(self) -> bool {
        !matches!(self, Speed::Fast)
    }

    /// AV1 Medium keeps angle refinement where small diagonal edges benefit,
    /// but omits the low-yield TX32 expansion. Split from the AV2 control so
    /// this staging cannot silently alter AV2 preset behavior.
    pub(crate) fn try_angle_deltas_av1(self, dim: usize, qidx: u8) -> bool {
        if matches!(self, Speed::Medium) && !crate::tuning::get().angle_deltas_medium {
            return false;
        }
        match self {
            Speed::Slow => true,
            Speed::Medium => dim <= 16 || qidx >= 112,
            Speed::Fast => false,
        }
    }

    /// Whether the intra candidate set is reduced. Only [`Speed::Fast`] does.
    pub(crate) fn reduced_modes(self) -> bool {
        matches!(self, Speed::Fast)
    }

    pub(crate) fn luma_mode_budget(self, full_res: bool) -> usize {
        match self {
            Speed::Fast => 1,
            Speed::Medium => crate::tuning::get().mode_budget_medium as usize,
            Speed::Slow => {
                if full_res {
                    5
                } else {
                    3
                }
            }
        }
    }

    /// Palette clustering is deliberately outside the Fast tier. It performs
    pub(crate) fn try_palette(self) -> bool {
        match self {
            Speed::Fast => false,
            Speed::Medium => crate::tuning::get().palette_medium,
            Speed::Slow => true,
        }
    }

    pub(crate) fn palette_refine_budget(self) -> usize {
        match self {
            Speed::Fast => 0,
            Speed::Medium => crate::tuning::get().palette_budget_medium as usize,
            Speed::Slow => 2,
        }
    }

    /// Fast uses a square, luma-led partition model. Rectangular candidates and
    /// full chroma partition RDO are reserved for the quality tiers.
    pub(crate) fn full_partition_rdo(self) -> bool {
        match self {
            Speed::Fast => false,
            Speed::Medium => crate::tuning::get().full_part_rdo_medium,
            Speed::Slow => true,
        }
    }

    pub(crate) fn partition_refine_budget(self) -> usize {
        let t = crate::tuning::get();
        (match self {
            Speed::Fast => t.part_budget_fast,
            Speed::Medium => t.part_budget_medium,
            Speed::Slow => t.part_budget_slow,
        }) as usize
    }

    pub(crate) fn filter_intra_refine_budget(self) -> usize {
        match self {
            Speed::Slow => 0,
            Speed::Medium | Speed::Fast => 0,
        }
    }

    /// Fast codes chroma with the baseline DC predictor. CfL and directional
    /// chroma each multiply transform/trellis work on both chroma planes and
    /// dominate finely partitioned screen content.
    pub(crate) fn full_chroma_rdo(self) -> bool {
        match self {
            Speed::Fast => false,
            Speed::Medium => crate::tuning::get().full_chroma_rdo_medium,
            Speed::Slow => true,
        }
    }

    /// Whether diagonal chroma modes (D45..D203) are searched. Medium retains
    /// the nominal V/H directionals; Slow adds the diagonal refinement beam.
    pub(crate) fn chroma_angle_directional(self) -> bool {
        matches!(self, Speed::Slow)
    }

    pub(crate) fn try_directional(&self) -> bool {
        true
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
    /// `0` = all available cores; `1` = serial; `N` = up to N.
    pub threads: usize,
    /// RDO effort for AV1 lossy and lossless paths. See [`Speed`]; defaults to
    /// [`Speed::Slow`]. In lossless mode, only Slow refines directional modes
    /// with nonzero angle deltas.
    pub speed: Speed,
    /// Update AV1 entropy CDFs after each coded symbol. Enabled by default.
    pub updating_cdf: bool,
    pub adaptive_quant: bool,
    pub variance_boost: bool,
    pub dark_aq: bool,
    /// Enable CDEF
    pub cdef: bool,
    /// Enable luma Wiener loop restoration (off by default)
    pub wiener: bool,
    /// Enable AV1 quantization matrices. These are standard AV1 syntax and
    /// redistribute quantization error toward higher spatial frequencies.
    pub quantization_matrices: bool,
    /// Search screen-content coding tools (palette). Palette clustering runs a
    /// histogram, weighted k-means and a color-map trial per block; it pays for
    /// itself on synthetic/screen material and is close to inert on camera
    /// photographs. On by default so the encoder needs no content hint.
    pub screen_content: bool,
    /// Frame-level IntraBC (exact-copy blocks from the reconstructed frame).
    /// AV1 ties `allow_intrabc` to disabling ALL in-loop filters for the
    /// frame, so the encoder only enables it when a coverage pre-scan finds
    /// enough exact 16x16 duplicates to pay that trade; this switch vetoes it
    /// outright. Lossless exact-copy blocks are unaffected (lossless frames
    /// carry no loop filters, so IntraBC costs nothing there).
    pub intrabc: bool,
    /// Explicit matrix level (0..=15), or `None` for the conservative level-10
    /// tuning validated on the Jixel still-image corpus. Lower levels weight
    /// high frequencies more strongly; level 15 is flat.
    pub qmatrix_level: Option<u8>,
}

impl Default for EncodeConfig {
    fn default() -> Self {
        EncodeConfig {
            quality: 80,
            chroma: ChromaFormat::Yuv420,
            color_encoding: Some(Cicp::srgb_ycbcr()),
            icc: None,
            metadata: Metadata::default(),
            threads: std::thread::available_parallelism()
                .unwrap_or(NonZeroUsize::new(1).unwrap())
                .get(),
            speed: Speed::Slow,
            updating_cdf: true,
            adaptive_quant: true,
            variance_boost: true,
            dark_aq: true,
            cdef: false,
            wiener: false,
            quantization_matrices: true,
            screen_content: true,
            intrabc: true,
            qmatrix_level: None,
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

    pub fn without_cicp(mut self) -> Self {
        self.color_encoding = None;
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

    /// Set the RDO effort level for AV1 lossy and lossless paths. See [`Speed`].
    pub fn with_speed(mut self, speed: Speed) -> Self {
        self.speed = speed;
        self
    }

    /// Enable or disable AV1 CDF adaptation for both lossless and lossy paths.
    pub fn with_updating_cdf(mut self, updating_cdf: bool) -> Self {
        self.updating_cdf = updating_cdf;
        self
    }

    pub fn with_adaptive_quant(mut self, v: bool) -> Self {
        self.adaptive_quant = v;
        self
    }

    /// Enable the AV2-style Variance Boost AQ scheme on the AV1 path. Has no effect
    /// unless [`Self::with_adaptive_quant`] is also enabled.
    pub fn with_variance_boost(mut self, v: bool) -> Self {
        self.variance_boost = v;
        self
    }

    /// Search the screen-content tools (palette). Leaving this on costs
    /// photographic content a little encode time; turning it off drops palette
    /// coding entirely, which is a large regression on synthetic content.
    pub fn with_screen_content(mut self, v: bool) -> Self {
        self.screen_content = v;
        self
    }

    /// Allow or veto lossy frame-level IntraBC (see [`EncodeConfig::intrabc`]).
    pub fn with_intrabc(mut self, v: bool) -> Self {
        self.intrabc = v;
        self
    }

    /// Enable the AV2-style dark-structured-detail AQ protection on the AV1 path.
    /// Has no effect unless [`Self::with_adaptive_quant`] is also enabled.
    pub fn with_dark_aq(mut self, v: bool) -> Self {
        self.dark_aq = v;
        self
    }

    /// Enable the in-loop CDEF filter
    pub fn with_cdef(mut self, v: bool) -> Self {
        self.cdef = v;
        self
    }

    /// Enable luma Wiener loop restoration (off by default).
    pub fn with_wiener(mut self, v: bool) -> Self {
        self.wiener = v;
        self
    }

    /// Enable or disable standard AV1 quantization matrices.
    pub fn with_quantization_matrices(mut self, v: bool) -> Self {
        self.quantization_matrices = v;
        self
    }

    /// Use one explicit matrix level for luma and chroma. This also enables
    /// quantization matrices. Level 15 is the flat/no-reshaping matrix.
    pub fn with_qmatrix_level(mut self, level: u8) -> Self {
        self.quantization_matrices = true;
        self.qmatrix_level = Some(level);
        self
    }

    pub(crate) fn vb(&self, base_q_idx: u8) -> crate::coder::VarianceBoost {
        let mut vb = if self.variance_boost {
            crate::coder::VarianceBoost::on()
        } else {
            crate::coder::VarianceBoost::off()
        };
        // Dark protection is independent of the Variance Boost scheme, so honor its
        // own flag rather than inheriting `variance_boost`.
        vb.dark = if self.dark_aq {
            crate::aq_common::DarkAq::on()
        } else {
            crate::aq_common::DarkAq::off()
        };
        vb.qm = if self.quantization_matrices {
            self.qmatrix_level
                .map(crate::quant::QmLevels::uniform)
                .unwrap_or_else(|| {
                    let sub = match self.chroma {
                        ChromaFormat::Yuv420 => 2,
                        ChromaFormat::Yuv422 => 1,
                        _ => 0,
                    };
                    let c = crate::quant::qm_chroma_level_law(base_q_idx, sub);
                    crate::quant::QmLevels {
                        y: crate::quant::qm_level_law(base_q_idx, sub),
                        u: c,
                        v: c,
                    }
                })
        } else {
            crate::quant::QmLevels::FLAT
        };
        vb
    }

    pub(crate) fn validate(&self) -> Result<(), EncodeError> {
        validate_quality(self.quality)?;
        if self.qmatrix_level.is_some_and(|level| level > 15) {
            return Err(EncodeError::InvalidQuality);
        }
        Ok(())
    }
}

pub(crate) fn validate_dims(width: u32, height: u32) -> Result<(), EncodeError> {
    if width < MIN_DIM || height < MIN_DIM || width > MAX_DIM || height > MAX_DIM {
        return Err(EncodeError::InvalidDimensions { width, height });
    }
    Ok(())
}

fn validate_quality(quality: u8) -> Result<(), EncodeError> {
    if quality > 100 {
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
/// Quality -> AV1 `base_q_idx`, recalibrated 2026-07-25 to LIBAVIF PARITY:
/// each label targets the same output size as `avifenc -q <label>` (fit on a
/// 4-image photo corpus at 4:2:0, per-image spread ~+-5 qindex; the old
/// linear map ran 25-40 qindex COARSER than aom at every label, which is why
/// same-label ladders showed us "20 SS2 points behind" while matched-rate
/// comparisons were at parity). Labels are now comparable across encoders;
/// BD curves are unaffected (pure re-labeling of the same RD curve).
/// q100 stays near-lossless (qindex 1) by design; parity resumes at q98.
fn quality_to_q(quality: u8) -> u8 {
    const ANCHORS: [(u8, u8); 14] = [
        (1, 215),
        (30, 140),
        (40, 116),
        (50, 91),
        (60, 70),
        (70, 53),
        (80, 35),
        (85, 28),
        (90, 17),
        (93, 12),
        (96, 7),
        (98, 5),
        (99, 3),
        (100, 1),
    ];
    let q = quality.clamp(1, 100);
    let mut prev = ANCHORS[0];
    for &(l, qi) in &ANCHORS[1..] {
        if q <= l {
            let (l0, q0) = (prev.0 as i32, prev.1 as i32);
            let (l1, q1) = (l as i32, qi as i32);
            let t = q as i32 - l0;
            return (q0 + (q1 - q0) * t / (l1 - l0).max(1)).clamp(1, 255) as u8;
        }
        prev = (l, qi);
    }
    1
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

#[allow(clippy::too_many_arguments)]
fn dispatch_lossy<T: crate::Pixel>(
    img: &PlanarImage<T>,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    sc: bool,
    intrabc: bool,
) -> Vec<u8> {
    match chroma {
        ChromaFormat::Yuv420 | ChromaFormat::Monochrome => encode_still_lossy_420_with_cdf(
            img,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            sc,
            intrabc,
        ),
        ChromaFormat::Yuv422 => encode_still_lossy_422_with_cdf(
            img,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            sc,
            intrabc,
        ),
        ChromaFormat::Yuv444 => encode_still_lossy_with_cdf(
            img,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            sc,
            intrabc,
        ),
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    );
    // An opaque auxiliary image carries no information. PNG decoders commonly
    // expose RGBA for screenshots even when every alpha sample is maximum;
    // running the complete lossless monochrome encoder in that case dominated
    // total encode time by several seconds.
    if img.planes[3].iter().all(|&a| a == u8::MAX) {
        return finalize_color(
            color_obu,
            img.width as u32,
            img.height as u32,
            8,
            cfg.chroma,
            cfg,
        );
    }
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    );
    if img.planes[3].iter().all(|&a| a == 1023) {
        return finalize_color(
            color_obu,
            img.width as u32,
            img.height as u32,
            10,
            cfg.chroma,
            cfg,
        );
    }
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    );
    if img.planes[3].iter().all(|&a| a == 4095) {
        return finalize_color(
            color_obu,
            img.width as u32,
            img.height as u32,
            12,
            cfg.chroma,
            cfg,
        );
    }
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_4(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_2(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_2(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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
        cfg.adaptive_quant,
        cfg.vb(quality_to_q(cfg.quality)),
        cfg.cdef,
        cfg.wiener,
        cfg.updating_cdf,
        cfg.screen_content,
        cfg.intrabc,
    )?;
    let alpha_obu = encode_lossless_gray_obu_with_cdf(
        &img.packed_alpha_2(),
        true,
        cfg.threads,
        cfg.updating_cdf,
    )?;
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

#[allow(clippy::too_many_arguments)]
fn dispatch_yuv_u8(
    planar_image: &PlanarImage<u8>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
        ChromaFormat::Yuv422 => encode_yuv422_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => encode_yuv444_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_yuv_u16(
    planar_image: &PlanarImage<u16>,
    bd: BitDepth,
    q: u8,
    chroma: ChromaFormat,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_with(chroma)?;
    match chroma {
        ChromaFormat::Yuv420 => encode_yuv420_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
        ChromaFormat::Yuv422 => encode_yuv422_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => encode_yuv444_obu(
            planar_image,
            bd,
            q,
            color,
            threads,
            speed,
            aq,
            vb,
            cdef,
            wiener,
            updating_cdf,
            screen_content,
            intrabc,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::process::Command;

    fn patterned_rgb(width: usize, height: usize) -> PlanarImage<u8> {
        let mut rgb = vec![0u8; width * height * 3];
        for y in 0..height {
            for x in 0..width {
                let i = (y * width + x) * 3;
                rgb[i] = ((x * 37 + y * 11) & 255) as u8;
                rgb[i + 1] = (((x / 4) * 83 + (y / 7) * 29) & 255) as u8;
                rgb[i + 2] = ((x * 13 + y * 53 + ((x ^ y) & 15) * 7) & 255) as u8;
            }
        }
        PlanarImage::from_interleaved_rgb(width, height, BitDepth::Eight, &rgb).unwrap()
    }

    fn png_dimensions(bytes: &[u8]) -> Option<(usize, usize)> {
        if bytes.len() < 24 || &bytes[..8] != b"\x89PNG\r\n\x1a\n" || &bytes[12..16] != b"IHDR" {
            return None;
        }
        let width = u32::from_be_bytes(bytes[16..20].try_into().ok()?) as usize;
        let height = u32::from_be_bytes(bytes[20..24].try_into().ok()?) as usize;
        Some((width, height))
    }

    fn temp_path(stem: &str, ext: &str) -> PathBuf {
        std::env::temp_dir().join(format!("maroontree-{stem}-{}.{ext}", std::process::id()))
    }

    #[test]
    fn chroma_partition_boundary_sizes_encode_and_decode() {
        let sizes = [
            (8, 8),
            (9, 9),
            (15, 17),
            (16, 16),
            (17, 19),
            (31, 33),
            (32, 32),
            (33, 35),
            (63, 65),
            (64, 64),
            (65, 67),
            (127, 95),
        ];
        let formats = [
            ChromaFormat::Yuv420,
            ChromaFormat::Yuv422,
            ChromaFormat::Yuv444,
        ];
        let avifdec = std::env::var_os("AVIFDEC");

        for &(width, height) in &sizes {
            let image = patterned_rgb(width, height);
            for &chroma in &formats {
                let cfg = EncodeConfig::new()
                    .with_quality(35)
                    .with_chroma(chroma)
                    .with_threads(1)
                    .with_speed(Speed::Fast)
                    .with_adaptive_quant(false);
                let avif = encode_rgb8(&image, &cfg)
                    .unwrap_or_else(|e| panic!("{width}x{height} {chroma:?} encode failed: {e}"));
                assert!(
                    !avif.is_empty(),
                    "{width}x{height} {chroma:?} produced an empty AVIF"
                );

                if let Some(decoder) = &avifdec {
                    let tag = format!("{width}x{height}-{chroma:?}");
                    let input = temp_path(&tag, "avif");
                    let output = temp_path(&tag, "png");
                    std::fs::write(&input, &avif).unwrap();
                    let result = Command::new(decoder)
                        .arg(&input)
                        .arg(&output)
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "avifdec rejected {width}x{height} {chroma:?}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    let decoded = std::fs::read(&output).unwrap();
                    assert_eq!(
                        png_dimensions(&decoded),
                        Some((width, height)),
                        "decoded dimensions changed for {width}x{height} {chroma:?}"
                    );
                    let _ = std::fs::remove_file(input);
                    let _ = std::fs::remove_file(output);
                }
            }
        }
    }

    #[test]
    fn lossy_wavefront_matches_serial_bytes() {
        // More cells than workers makes every worker reuse its tile-local
        // reconstruction for unrelated SBs, covering halo reload/rezeroing as
        // well as semantic entropy packing.
        let image = patterned_rgb(320, 192);
        let config = |threads| {
            EncodeConfig::new()
                .with_quality(60)
                .with_chroma(ChromaFormat::Yuv444)
                .with_threads(threads)
                .with_speed(Speed::Medium)
                .with_variance_boost(true)
                .with_updating_cdf(true)
        };
        let serial = encode_rgb8(&image, &config(1)).unwrap();
        let wavefront = encode_rgb8(&image, &config(4)).unwrap();
        assert_eq!(wavefront, serial);
    }
}
