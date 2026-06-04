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

//! Colour and metadata signalling for the bitstream.
//!
//! This module carries the "describe the pixels" signals an encoder must emit
//! alongside the coded samples: CICP code points (ISO/IEC 23091-2 — the
//! `color_primaries` / `transfer_characteristics` / `matrix_coefficients`
//! triplet, plus full/limited range and chroma sample position), the HDR
//! mastering-display and content-light metadata, raw ITU-T T.35 user data, and
//! an ICC profile.
//!
//! Where each signal lives:
//! - **CICP + range + chroma position** → the AV1 sequence header
//!   `color_config()` (see [`crate::obu`]). These travel *inside* the coded
//!   bitstream and a bare AV1 decoder reads them.
//! - **HDR CLL / MDCV, ITU-T T.35** → AV1 *metadata OBUs* (also in-bitstream).
//! - **ICC profile** → NOT an AV1 OBU. In AVIF it is a container property
//!   (`colr` box, colour type `prof`/`rICC`); we carry the bytes here so the
//!   future AVIF muxer can emit them. It is intentionally ignored by the raw
//!   AV1 OBU writer.
//!
//! The point of separating the *data* (here) from the *coding* (obu.rs) is that
//! the same `ImageMetadata` can later be consumed by an AVIF muxer, which will
//! route CICP to an `nclx` box (or leave it in the OBU), the ICC to a `prof`
//! box, and the HDR metadata to either OBUs or boxes as appropriate.

/// CICP `color_primaries` code points (ISO/IEC 23091-2 / H.273 Table 2).
pub mod primaries {
    pub const BT709: u8 = 1;
    pub const UNSPECIFIED: u8 = 2;
    pub const BT470M: u8 = 4;
    pub const BT470BG: u8 = 5;
    pub const BT601: u8 = 6;
    pub const SMPTE240: u8 = 7;
    pub const GENERIC_FILM: u8 = 8;
    pub const BT2020: u8 = 9;
    pub const XYZ: u8 = 10;
    pub const SMPTE431: u8 = 11; // DCI P3
    pub const SMPTE432: u8 = 12; // Display P3
    pub const EBU3213: u8 = 22;
}

/// CICP `transfer_characteristics` code points.
pub mod transfer {
    pub const BT709: u8 = 1;
    pub const UNSPECIFIED: u8 = 2;
    pub const BT470M: u8 = 4;
    pub const BT470BG: u8 = 5;
    pub const BT601: u8 = 6;
    pub const SMPTE240: u8 = 7;
    pub const LINEAR: u8 = 8;
    pub const LOG100: u8 = 9;
    pub const IEC61966_2_4: u8 = 11;
    pub const BT1361: u8 = 12;
    pub const SRGB: u8 = 13; // IEC 61966-2-1
    pub const BT2020_10: u8 = 14;
    pub const BT2020_12: u8 = 15;
    pub const PQ: u8 = 16; // SMPTE ST 2084
    pub const SMPTE428: u8 = 17;
    pub const HLG: u8 = 18; // ARIB STD-B67
}

/// CICP `matrix_coefficients` code points.
pub mod matrix {
    pub const IDENTITY: u8 = 0; // GBR / RGB — requires 4:4:4 + full range
    pub const BT709: u8 = 1;
    pub const UNSPECIFIED: u8 = 2;
    pub const FCC: u8 = 4;
    pub const BT470BG: u8 = 5;
    pub const BT601: u8 = 6;
    pub const SMPTE240: u8 = 7;
    pub const YCGCO: u8 = 8;
    pub const BT2020_NCL: u8 = 9;
    pub const BT2020_CL: u8 = 10;
    pub const SMPTE2085: u8 = 11;
    pub const CHROMAT_NCL: u8 = 12;
    pub const CHROMAT_CL: u8 = 13;
    pub const ICTCP: u8 = 14;
}

/// AV1 `chroma_sample_position` (only meaningful for 4:2:0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ChromaSamplePosition {
    Unknown = 0,
    Vertical = 1,  // co-sited horizontally, interstitial vertically (MPEG-2)
    Colocated = 2, // co-sited both (MPEG-4 / H.264 type-0... "co-located")
}

/// CICP colour signalling: the primaries / transfer / matrix triplet plus
/// full-vs-limited range and (for 4:2:0) chroma sample position. This is the
/// `nclx`/`colr` information and the AV1 sequence-header `color_config`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub color_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
    pub chroma_sample_position: ChromaSamplePosition,
}

impl Cicp {
    /// Identity (GBR) matrix — the lossless RGB path. AV1 requires this to be
    /// 4:4:4 and full range.
    pub fn identity_rgb() -> Self {
        Cicp {
            color_primaries: primaries::BT709,
            transfer_characteristics: transfer::SRGB,
            matrix_coefficients: matrix::IDENTITY,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }
    /// sRGB still image carried as full-range BT.601 YCbCr (the lossy default:
    /// the decoder decorrelates the planes back to RGB on output).
    pub fn srgb_ycbcr() -> Self {
        Cicp {
            color_primaries: primaries::BT709,
            transfer_characteristics: transfer::SRGB,
            matrix_coefficients: matrix::BT601,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }
    /// BT.709 limited-range HD video colour.
    pub fn bt709() -> Self {
        Cicp {
            color_primaries: primaries::BT709,
            transfer_characteristics: transfer::BT709,
            matrix_coefficients: matrix::BT709,
            full_range: false,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }
    /// BT.2020 NCL with the PQ (ST 2084) transfer — HDR10.
    pub fn bt2020_pq() -> Self {
        Cicp {
            color_primaries: primaries::BT2020,
            transfer_characteristics: transfer::PQ,
            matrix_coefficients: matrix::BT2020_NCL,
            full_range: false,
            chroma_sample_position: ChromaSamplePosition::Colocated,
        }
    }
    /// BT.2020 NCL with the HLG transfer.
    pub fn bt2020_hlg() -> Self {
        Cicp {
            color_primaries: primaries::BT2020,
            transfer_characteristics: transfer::HLG,
            matrix_coefficients: matrix::BT2020_NCL,
            full_range: false,
            chroma_sample_position: ChromaSamplePosition::Colocated,
        }
    }
    /// True when the matrix is identity, which AV1 constrains to 4:4:4 +
    /// full range (the encoder must honour this).
    pub fn is_identity(&self) -> bool {
        self.matrix_coefficients == matrix::IDENTITY
    }
}

/// HDR content light level (CTA-861.3), emitted as an AV1 `HDR_CLL` metadata OBU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ContentLightLevel {
    pub max_cll: u16,  // maximum content light level (cd/m^2)
    pub max_fall: u16, // maximum frame-average light level (cd/m^2)
}

/// HDR mastering display colour volume (SMPTE ST 2086), emitted as an AV1
/// `HDR_MDCV` metadata OBU. Chromaticities are in 0.00002 increments
/// (`x * 50000` rounded); luminance in 0.0001 cd/m^2 increments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasteringDisplay {
    pub primaries: [(u16, u16); 3], // R, G, B chromaticity (x, y), each *50000
    pub white_point: (u16, u16),    // white point (x, y), *50000
    pub max_luminance: u32,         // *10000 cd/m^2
    pub min_luminance: u32,         // *10000 cd/m^2
}

impl MasteringDisplay {
    /// Helper to build from floating-point chromaticities (0..1) and nits.
    pub fn from_floats(
        primaries: [(f64, f64); 3],
        white: (f64, f64),
        max_nits: f64,
        min_nits: f64,
    ) -> Self {
        let q = |v: f64| (v * 50000.0).round() as u16;
        MasteringDisplay {
            primaries: [
                (q(primaries[0].0), q(primaries[0].1)),
                (q(primaries[1].0), q(primaries[1].1)),
                (q(primaries[2].0), q(primaries[2].1)),
            ],
            white_point: (q(white.0), q(white.1)),
            max_luminance: (max_nits * 10000.0).round() as u32,
            min_luminance: (min_nits * 10000.0).round() as u32,
        }
    }
}

/// Raw ITU-T T.35 user metadata (e.g. the carrier for HDR10+ or Dolby Vision
/// RPU), emitted as an AV1 `ITUT_T35` metadata OBU. `country_code` is the
/// Recommendation T.35 terminal-provider country code (0xFF selects the
/// extension byte). `payload` is the provider-defined bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItutT35 {
    pub country_code: u8,
    pub country_code_extension: Option<u8>, // present iff country_code == 0xFF
    pub payload: Vec<u8>,
}

/// All "describe the pixels" signals for one image. CICP is mandatory; the rest
/// are optional. The ICC profile is container-level (AVIF `colr`/`prof`) and is
/// carried here for the muxer — it is NOT written into the AV1 OBU stream.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImageMetadata {
    pub cicp: Cicp,
    pub cll: Option<ContentLightLevel>,
    pub mdcv: Option<MasteringDisplay>,
    pub t35: Vec<ItutT35>,
    pub icc_profile: Option<Vec<u8>>,
}

impl ImageMetadata {
    /// Minimal metadata: just a CICP triplet, no HDR/ICC.
    pub fn new(cicp: Cicp) -> Self {
        ImageMetadata {
            cicp,
            cll: None,
            mdcv: None,
            t35: Vec::new(),
            icc_profile: None,
        }
    }
    pub fn with_cll(mut self, cll: ContentLightLevel) -> Self {
        self.cll = Some(cll);
        self
    }
    pub fn with_mdcv(mut self, mdcv: MasteringDisplay) -> Self {
        self.mdcv = Some(mdcv);
        self
    }
    pub fn with_t35(mut self, t35: ItutT35) -> Self {
        self.t35.push(t35);
        self
    }
    pub fn with_icc(mut self, icc: Vec<u8>) -> Self {
        self.icc_profile = Some(icc);
        self
    }
}
