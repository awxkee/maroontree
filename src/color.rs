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

/// CICP color primaries (ISO/IEC 23091-2 Table 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Primaries {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional color gamut system and extended color gamut system (historical)<br />
    /// IEC 61966-2-1 sRGB or sYCC IEC 61966-2-4<br />
    /// Society of Motion Picture and Television Engineers (MPTE) RP 177 (1993) Annex B<br />
    Bt709 = 1,
    /// Unspecified<br />
    /// Image characteristics are unknown or are determined by the application.
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical) Rec. ITU-R BT.601-7 625<br />
    /// Rec. ITU-R BT.1358-0 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical) Rec. ITU-R BT.1700-0 NTSC<br />
    /// SMPTE 170M (2004)<br />
    /// (functionally the same as the value 7)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical) (functionally the same as the value 6)<br />
    Smpte240 = 7,
    /// Generic film (color filters using Illuminant C)<br />
    GenericFilm = 8,
    /// Rec. ITU-R BT.2020-2<br />
    /// Rec. ITU-R BT.2100-0<br />
    Bt2020 = 9,
    /// SMPTE ST 428-1<br />
    /// (CIE 1931 XYZ as in ISO 11664-1)<br />
    Xyz = 10,
    /// SMPTE RP 431-2 (2011)<br />
    Smpte431 = 11,
    /// SMPTE EG 432-1 (2010)<br />
    Smpte432 = 12,
    /// EBU Tech. 3213-E (1975)<br />
    Ebu3213 = 22,
}

/// CICP transfer characteristics (ISO/IEC 23091-2 Table 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TransferFunction {
    /// For future use by ITU-T | ISO/IEC
    Reserved,
    /// Rec. ITU-R BT.709-6<br />
    /// Rec. ITU-R BT.1361-0 conventional color gamut system (historical)<br />
    /// (functionally the same as the values 6, 14 and 15)    <br />
    Bt709 = 1,
    /// Image characteristics are unknown or are determined by the application.<br />
    Unspecified = 2,
    /// Rec. ITU-R BT.470-6 System M (historical)<br />
    /// United States National Television System Committee 1953 Recommendation for transmission standards for color television<br />
    /// United States Federal Communications Commission (2003) Title 47 Code of Federal Regulations 73.682 (a) (20)<br />
    /// Rec. ITU-R BT.1700-0 625 PAL and 625 SECAM<br />
    Bt470M = 4,
    /// Rec. ITU-R BT.470-6 System B, G (historical)<br />
    Bt470Bg = 5,
    /// Rec. ITU-R BT.601-7 525 or 625<br />
    /// Rec. ITU-R BT.1358-1 525 or 625 (historical)<br />
    /// Rec. ITU-R BT.1700-0 NTSC SMPTE 170M (2004)<br />
    /// (functionally the same as the values 1, 14 and 15)<br />
    Bt601 = 6,
    /// SMPTE 240M (1999) (historical)<br />
    Smpte240 = 7,
    /// Linear transfer characteristics<br />
    Linear = 8,
    /// Logarithmic transfer characteristic (100:1 range)<br />
    Log100 = 9,
    /// Logarithmic transfer characteristic (100 * Sqrt( 10 ) : 1 range)<br />
    Log100sqrt10 = 10,
    /// IEC 61966-2-4<br />
    Iec61966 = 11,
    /// Rec. ITU-R BT.1361-0 extended color gamut system (historical)<br />
    Bt1361 = 12,
    /// IEC 61966-2-1 sRGB or sYCC<br />
    Srgb = 13,
    /// Rec. ITU-R BT.2020-2 (10-bit system)<br />
    /// (functionally the same as the values 1, 6 and 15)<br />
    Bt202010bit = 14,
    /// Rec. ITU-R BT.2020-2 (12-bit system)<br />
    /// (functionally the same as the values 1, 6 and 14)<br />
    Bt202012bit = 15,
    /// SMPTE ST 2084 for 10-, 12-, 14- and 16-bitsystems<br />
    /// Rec. ITU-R BT.2100-0 perceptual quantization (PQ) system<br />
    Smpte2084 = 16,
    /// SMPTE ST 428-1<br />
    Smpte428 = 17,
    /// ARIB STD-B67<br />
    /// Rec. ITU-R BT.2100-0 hybrid log- gamma (HLG) system<br />
    Hlg = 18,
}

/// CICP matrix coefficients for RGB→YCbCr (ISO/IEC 23091-2 Table 4).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MatrixCoefficients {
    Identity = 0,                // RGB (Identity matrix)
    Bt709 = 1,                   // Rec. 709
    Unspecified = 2,             // Unspecified
    Reserved = 3,                // Reserved
    Fcc = 4,                     // FCC
    Bt470Bg = 5,                 // BT.470BG / BT.601-625
    Smpte170m = 6,               // SMPTE 170M / BT.601-525
    Smpte240m = 7,               // SMPTE 240M
    YCgCo = 8,                   // YCgCo
    Bt2020Ncl = 9,               // BT.2020 (non-constant luminance)
    Bt2020Cl = 10,               // BT.2020 (constant luminance)
    Smpte2085 = 11,              // SMPTE ST 2085
    ChromaticityDerivedNCL = 12, // Chromaticity-derived non-constant luminance
    ChromaticityDerivedCL = 13,  // Chromaticity-derived constant luminance
    ICtCp = 14,                  // ICtCp
    IPtC2 = 15,                  // Color representation developed in SMPTE as IPT-PQ-C2
    YCgCoRe = 16,                // YCgCo-Re (YCgCo-R type even),
    YCgCoRo = 17,                // YCgCo-Ro (YCgCo-R type odd),
}

impl MatrixCoefficients {
    pub fn from_u8(v: u8) -> Self {
        match v {
            0 => MatrixCoefficients::Identity,
            1 => MatrixCoefficients::Bt709,
            2 => MatrixCoefficients::Unspecified,
            3 => MatrixCoefficients::Reserved,
            4 => MatrixCoefficients::Fcc,
            5 => MatrixCoefficients::Bt470Bg,
            6 => MatrixCoefficients::Smpte170m,
            7 => MatrixCoefficients::Smpte240m,
            8 => MatrixCoefficients::YCgCo,
            9 => MatrixCoefficients::Bt2020Ncl,
            10 => MatrixCoefficients::Bt2020Cl,
            11 => MatrixCoefficients::Smpte2085,
            12 => MatrixCoefficients::ChromaticityDerivedNCL,
            13 => MatrixCoefficients::ChromaticityDerivedCL,
            14 => MatrixCoefficients::ICtCp,
            15 => MatrixCoefficients::IPtC2,
            16 => MatrixCoefficients::YCgCoRe,
            17 => MatrixCoefficients::YCgCoRo,
            _ => MatrixCoefficients::Unspecified,
        }
    }
}

/// How the output color space is described in the AVIF container.
///
/// * `Cicp` → `colr` box of type `nclx` (11-byte enumerated CICP payload).
/// * `Icc`  → `colr` box of type `prof` (embedded ICC profile bytes).
#[derive(Clone, Debug)]
pub enum ColorMetadata {
    /// Enumerated CICP code points. Use one of the [`ColorEncoding`] constructors
    /// or build a custom value with the enum fields.
    Cicp(ColorEncoding),
    /// Embedded ICC profile bytes (the raw profile, without any container header).
    Icc(Vec<u8>),
}

impl ColorMetadata {
    /// The color encoding implied by these metadata, used to drive the AV1
    /// sequence-header `color_config()`. An ICC profile is treated as sRGB.
    pub fn color_encoding(&self) -> ColorEncoding {
        match self {
            ColorMetadata::Cicp(c) => *c,
            ColorMetadata::Icc(_) => ColorEncoding::srgb(),
        }
    }
}

impl Default for ColorMetadata {
    fn default() -> Self {
        ColorMetadata::Cicp(ColorEncoding::srgb())
    }
}

/// CICP-style color encoding: the primaries + transfer + matrix the image is
/// authored in, plus the sample range. Drives both the HEIF `colr` (nclx) box and
/// the HEVC VUI signaling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColorEncoding {
    pub primaries: Primaries,
    pub transfer: TransferFunction,
    pub matrix: MatrixCoefficients,
    pub full_range: bool,
    pub chroma_sample_position: ChromaSamplePosition,
}

impl ColorEncoding {
    pub fn identity_rgb() -> ColorEncoding {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Identity,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }
}

impl ColorEncoding {
    /// sRGB: BT.709 primaries, sRGB transfer, BT.709 matrix, full range. This is the
    /// encoder's working color space (full-range BT.709 RGB→YCbCr).
    pub const fn srgb() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }

    /// BT.709 primaries, sRGB transfer, **BT.601 (Smpte170m) matrix**, full range.
    /// This is the encoding the lossy RGB→YCbCr path actually uses internally and
    /// what the AV1 bitstream signals by default when no custom color is passed.
    /// Use this when you want the container `colr` and OBU `color_config` to both
    /// agree with the encoder's actual YCbCr coefficients.
    pub const fn srgb_ycbcr() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Srgb,
            matrix: MatrixCoefficients::Smpte170m,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }

    /// BT.709 video: BT.709 primaries/transfer/matrix, full range.
    pub const fn bt709() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt709,
            transfer: TransferFunction::Bt709,
            matrix: MatrixCoefficients::Bt709,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }

    /// BT.2020 PQ (HDR10): BT.2020 primaries, PQ transfer, BT.2020 NCL matrix.
    pub const fn bt2020_pq() -> Self {
        ColorEncoding {
            primaries: Primaries::Bt2020,
            transfer: TransferFunction::Smpte2084,
            matrix: MatrixCoefficients::Bt2020Ncl,
            full_range: true,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }

    /// The `nclx` payload for a HEIF `colr` box (without the box header):
    /// `color_type` ('nclx') + the four CICP fields. The `full_range_flag` occupies
    /// the top bit of the final byte; the low 7 bits are reserved zero.
    pub fn nclx_payload(&self) -> Vec<u8> {
        let mut p = Vec::with_capacity(11);
        p.extend_from_slice(b"nclx");
        p.extend_from_slice(&(self.primaries as u16).to_be_bytes());
        p.extend_from_slice(&(self.transfer as u16).to_be_bytes());
        p.extend_from_slice(&(self.matrix as u16).to_be_bytes());
        p.push(if self.full_range { 0x80 } else { 0x00 });
        p
    }
}

impl Default for ColorEncoding {
    fn default() -> Self {
        ColorEncoding::srgb()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ChromaSamplePosition {
    Unknown = 0,
    Vertical = 1,
    Colocated = 2,
}

/// HDR mastering display color volume (SMPTE ST 2086) — emitted as AV1 `HDR_MDCV` metadata OBU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MasteringDisplay {
    pub primaries: [(u16, u16); 3],
    pub white_point: (u16, u16),
    pub max_luminance: u32,
    pub min_luminance: u32,
}

impl MasteringDisplay {
    pub fn from_floats(p: [(f64, f64); 3], w: (f64, f64), max: f64, min: f64) -> Self {
        let q = |v: f64| (v * 50000.0).round() as u16;
        MasteringDisplay {
            primaries: [
                (q(p[0].0), q(p[0].1)),
                (q(p[1].0), q(p[1].1)),
                (q(p[2].0), q(p[2].1)),
            ],
            white_point: (q(w.0), q(w.1)),
            max_luminance: (max * 10000.0).round() as u32,
            min_luminance: (min * 10000.0).round() as u32,
        }
    }
}

/// Raw ITU-T T.35 user metadata — emitted as AV1 `ITUT_T35` metadata OBU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ItutT35 {
    pub country_code: u8,
    pub country_code_extension: Option<u8>,
    pub payload: Vec<u8>,
}
