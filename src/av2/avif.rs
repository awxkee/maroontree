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

//! AV2-in-ISOBMFF still-image container ("AVIF-style").
//!
//! AVIF (ISO/IEC 23000-22) standardizes AV1 in a MIAF/ISOBMFF file: item type
//! `av01`, codec-config box `av1C`. AV2 has no ratified ISOBMFF binding yet, so
//! this module mirrors that layout with AV2-specific four-character codes that
//! are this project's convention:
//!
//!   * item type      `av02`   (vs `av01`)
//!   * codec config   `av2C`   (vs `av1C`, identical 4-byte record layout)
//!   * compat brand   `av2f`   (added alongside `avif`/`mif1`/`miaf`)
//!
//! Everything else (`ftyp`/`meta`/`mdat`, `ispe`/`pixi`/`colr`, `iloc`/`iinf`/
//! `ipma`) is byte-for-byte the structure libavif/avifenc emit, so the file is a
//! valid MIAF image container. A standard AVIF decoder won't decode the AV2
//! sample, but the OBU stream in `mdat` round-trips losslessly and feeds straight
//! into the AV2 decoder (avmdec).
//!
//! Box helpers (`w16`/`w32`/`write_box`/`write_fullbox`/`patch`) follow the same
//! shape as the project's `isobmff.rs` so this drops into that file as a sibling
//! of `wrap_av1_image`.
use crate::ColorEncoding;
use crate::av2::Av2Frame;

/// Color information for the `colr` box: either enumerated CICP (`nclx`) or an
/// embedded ICC profile (`prof`). An ICC profile can only live in the container
/// — the AV2 bitstream carries CICP indices only, no place for a profile blob.
pub enum Av2Color {
    /// Enumerated CICP primaries/transfer/matrix + range → `colr` type `nclx`.
    Cicp(ColorEncoding),
    /// Raw ICC profile bytes (no extra header) → `colr` type `prof`.
    Icc(Vec<u8>),
    /// Both: an `nclx` box (CICP, carries the matrix/range) and a `prof` box
    /// (ICC, for display). Allowed by MIAF (one colr per colour_type) and what
    /// libavif/avifenc emit, since ICC alone cannot signal matrix_coefficients.
    Both { cicp: ColorEncoding, icc: Vec<u8> },
}

fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}
fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}
/// Open a plain box with a 4-byte size placeholder; returns the start offset.
fn write_box(buf: &mut Vec<u8>, cc: &[u8; 4]) -> usize {
    let s = buf.len();
    w32(buf, 0);
    buf.extend_from_slice(cc);
    s
}
/// Open a FullBox (adds version + 24-bit flags); returns the start offset.
fn write_fullbox(buf: &mut Vec<u8>, cc: &[u8; 4], ver: u8, flags: u32) -> usize {
    let s = buf.len();
    w32(buf, 0);
    buf.extend_from_slice(cc);
    buf.push(ver);
    buf.push((flags >> 16) as u8);
    buf.push((flags >> 8) as u8);
    buf.push(flags as u8);
    s
}
/// Backfill a box's size field now that its contents are complete.
fn patch(buf: &mut [u8], start: usize) {
    let size = (buf.len() - start) as u32;
    buf[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

/// Chroma/bit-depth descriptor for the codec-config record.
#[derive(Clone, Copy)]
pub struct Av2Format {
    pub bit_depth: u8,
    pub monochrome: bool,
    pub chroma_sub_x: bool,
    pub chroma_sub_y: bool,
}

impl Av2Format {
    fn channels(&self) -> u8 {
        if self.monochrome { 1 } else { 3 }
    }
    /// seq_profile, mirroring AV1's mapping (444→1, 422 or 12-bit→2, else 0).
    fn seq_profile(&self) -> u8 {
        if self.bit_depth == 12 || self.chroma_sub(true, false) {
            2
        } else if !self.chroma_sub_x && !self.chroma_sub_y && !self.monochrome {
            1
        } else {
            0
        }
    }
    fn chroma_sub(&self, x: bool, y: bool) -> bool {
        self.chroma_sub_x == x && self.chroma_sub_y == y
    }
}

/// Pick an `seq_level_idx_0` large enough for the picture (same thresholds as the
/// AV1 path; level numbers are shared between AV1 and AV2 sequence headers).
fn level_for(width: u32, height: u32) -> u8 {
    let pels = width as u64 * height as u64;
    if pels <= 147_456 {
        0 // 2.0  (e.g. 256x192-ish)
    } else if pels <= 278_784 {
        1 // 2.1
    } else if pels <= 665_856 {
        4 // 3.0
    } else if pels <= 1_065_024 {
        5 // 3.1
    } else if pels <= 2_359_296 {
        8 // 4.0  (1920x1080 ≈ 2.07M)
    } else if pels <= 8_912_896 {
        12 // 5.0
    } else {
        16 // 6.0
    }
}

/// The 4-byte AV2 codec-configuration record (same field layout as `av1C`).
fn build_av2c(fmt: &Av2Format, width: u32, height: u32) -> [u8; 4] {
    let high_bitdepth = fmt.bit_depth > 8;
    let twelve_bit = fmt.bit_depth == 12;
    // Byte 0: marker(1)=1, version(7)=1
    let b0 = 0x81u8;
    // Byte 1: seq_profile(3) | seq_level_idx_0(5)
    let b1 = ((fmt.seq_profile() & 0x7) << 5) | (level_for(width, height) & 0x1f);
    // Byte 2: tier(1)=0 | high_bitdepth | twelve_bit | monochrome | sub_x | sub_y | sample_pos(2)=0
    let b2 = (if high_bitdepth { 0x40 } else { 0 })
        | (if twelve_bit { 0x20 } else { 0 })
        | (if fmt.monochrome { 0x10 } else { 0 })
        | (if fmt.chroma_sub_x { 0x08 } else { 0 })
        | (if fmt.chroma_sub_y { 0x04 } else { 0 });
    // Byte 3: reserved(3)=0 | initial_presentation_delay_present(1)=0 | reserved(4)=0
    [b0, b1, b2, 0x00]
}

/// Wrap an AV2 OBU stream (`Encoded::data` = TD + sequence + frame OBUs) into an
/// AVIF-style ISOBMFF file. `width`/`height` are the *display* dimensions
/// (`ispe`); the bitstream may decode to a padded size and be cropped on output.
pub fn wrap_av2_image(
    obu: &[u8],
    width: u32,
    height: u32,
    disp_width: u32,
    disp_height: u32,
    fmt: &Av2Format,
    color: &Av2Color,
    exif: Option<&[u8]>,
) -> Vec<u8> {
    let channels = fmt.channels();
    let av2c = build_av2c(fmt, width, height);
    // EXIF item data is `ExifDataBlock`: a 4-byte exif_tiff_header_offset (0 when
    // the payload starts at the TIFF header) followed by the raw TIFF/EXIF bytes.
    let has_exif = exif.is_some();
    let exif_block: Vec<u8> = exif
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes()); // exif_tiff_header_offset = 0
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();
    let mut f = Vec::with_capacity(obu.len() + exif_block.len() + 512);

    // ── ftyp ────────────────────────────────────────────────────────────────
    {
        let s = write_box(&mut f, b"ftyp");
        f.extend_from_slice(b"avif"); // major_brand
        w32(&mut f, 0); // minor_version
        for b in [b"avif", b"mif1", b"miaf", b"av2f"] {
            f.extend_from_slice(b);
        }
        patch(&mut f, s);
    }

    // ── meta ──────────────────────────────────────────────────────────────────
    let meta_start = write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr — 'pict'
    {
        let s = write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0); // pre_defined
        f.extend_from_slice(b"pict"); // handler_type
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0); // reserved
        f.push(0); // name (empty, null-terminated)
        patch(&mut f, s);
    }
    // pitm — primary item is ID 1
    {
        let s = write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }
    // iloc — image item (1), plus an Exif item (2) when present. Offsets patched
    // once the mdat position is known.
    let iloc_offset_pos;
    let mut iloc_exif_pos = 0usize;
    {
        let s = write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, if has_exif { 2 } else { 1 }); // item_count
        // item 1: the AV2 image
        w16(&mut f, 1); // item_ID
        w16(&mut f, 0); // data_reference_index (0 = this file)
        w16(&mut f, 1); // extent_count
        iloc_offset_pos = f.len();
        w32(&mut f, 0); // extent_offset — patched after mdat is placed
        w32(&mut f, obu.len() as u32); // extent_length
        if has_exif {
            // item 2: the Exif metadata
            w16(&mut f, 2);
            w16(&mut f, 0);
            w16(&mut f, 1);
            iloc_exif_pos = f.len();
            w32(&mut f, 0); // extent_offset — patched later
            w32(&mut f, exif_block.len() as u32);
        }
        patch(&mut f, s);
    }
    // iinf → infe ('av02', and 'Exif' when present)
    {
        let s = write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, if has_exif { 2 } else { 1 }); // entry_count
        {
            let si = write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 1); // item_ID
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"av02"); // item_type — AV2 image (this project's 4CC)
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        if has_exif {
            let si = write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 2); // item_ID
            w16(&mut f, 0);
            f.extend_from_slice(b"Exif"); // item_type — Exif metadata
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }
    // iref — the Exif item (2) describes the primary image (1) via 'cdsc'.
    if has_exif {
        let s = write_fullbox(&mut f, b"iref", 0, 0);
        let si = write_box(&mut f, b"cdsc");
        w16(&mut f, 2); // from_item_ID = Exif
        w16(&mut f, 1); // reference_count
        w16(&mut f, 1); // to_item_ID = image
        patch(&mut f, si);
        patch(&mut f, s);
    }
    // iprp → ipco { ispe, pixi, av2C, colr } + ipma
    {
        let s = write_box(&mut f, b"iprp");
        let ipco = write_box(&mut f, b"ipco");
        // prop 1: ispe (image spatial extents)
        {
            let p = write_fullbox(&mut f, b"ispe", 0, 0);
            w32(&mut f, width);
            w32(&mut f, height);
            patch(&mut f, p);
        }
        // prop 2: pixi (bits per channel)
        {
            let p = write_fullbox(&mut f, b"pixi", 0, 0);
            f.push(channels);
            for _ in 0..channels {
                f.push(fmt.bit_depth);
            }
            patch(&mut f, p);
        }
        // prop 3: av2C (codec configuration) — essential
        {
            let p = write_box(&mut f, b"av2C");
            f.extend_from_slice(&av2c);
            patch(&mut f, p);
        }
        // colr properties. MIAF allows at most one per color_type, so an `nclx`
        // (CICP: primaries/transfer/matrix/range) and a `prof` (ICC) may coexist.
        // Keep `nclx` whenever CICP is known: an ICC profile cannot carry
        // matrix_coefficients, so the YUV→RGB matrix must live in nclx (or the
        // bitstream). Track each colr's 1-based property index for `ipma`.
        let mut colr_props: Vec<u8> = Vec::new();
        let mut next_prop: u8 = 4; // ispe=1, pixi=2, av2C=3 precede these
        let write_nclx = |f: &mut Vec<u8>, c: &ColorEncoding| {
            let p = write_box(f, b"colr");
            f.extend_from_slice(b"nclx");
            w16(f, c.primaries as u16);
            w16(f, c.transfer as u16);
            w16(f, c.matrix as u16);
            f.push(if c.full_range { 0x80 } else { 0x00 });
            patch(f, p);
        };
        let write_prof = |f: &mut Vec<u8>, icc: &[u8]| {
            let p = write_box(f, b"colr");
            f.extend_from_slice(b"prof");
            f.extend_from_slice(icc);
            patch(f, p);
        };
        match color {
            Av2Color::Cicp(c) => {
                write_nclx(&mut f, c);
                colr_props.push(next_prop);
                next_prop += 1;
            }
            Av2Color::Icc(icc) => {
                write_prof(&mut f, icc);
                colr_props.push(next_prop);
                next_prop += 1;
            }
            Av2Color::Both { cicp, icc } => {
                write_nclx(&mut f, cicp);
                colr_props.push(next_prop);
                next_prop += 1;
                write_prof(&mut f, icc);
                colr_props.push(next_prop);
                next_prop += 1;
            }
        }
        let _ = next_prop;
        // Optional clap (clean aperture): crop the coded (ispe) image down to the
        // display size. Required for padded lossy frames so every reader — not just
        // ispe-aware ones — shows the real dimensions. Center-relative per ISO 14496-12:
        // horizOff = (disp - coded)/2 with denominator 2; numerators are signed.
        let mut clap_prop: Option<u8> = None;
        if disp_width != width || disp_height != height {
            let p = write_box(&mut f, b"clap");
            w32(&mut f, disp_width); // cleanApertureWidthN
            w32(&mut f, 1); // cleanApertureWidthD
            w32(&mut f, disp_height); // cleanApertureHeightN
            w32(&mut f, 1); // cleanApertureHeightD
            w32(&mut f, (disp_width as i32 - width as i32) as u32); // horizOffN (signed)
            w32(&mut f, 2); // horizOffD
            w32(&mut f, (disp_height as i32 - height as i32) as u32); // vertOffN (signed)
            w32(&mut f, 2); // vertOffD
            patch(&mut f, p);
            clap_prop = Some(next_prop);
            next_prop += 1;
        }
        let _ = next_prop;
        patch(&mut f, ipco);
        // ipma — associate item 1 with ispe(1), pixi(2), av2C(3, essential), and
        // each colr property. (None of the colr boxes are marked essential.)
        {
            let p = write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1); // entry_count
            w16(&mut f, 1); // item_ID
            let assoc = 3 + colr_props.len() as u8 + clap_prop.is_some() as u8;
            f.push(assoc); // association_count
            f.push(1); // ispe
            f.push(2); // pixi
            f.push(0x80 | 3); // av2C (essential)
            for &idx in &colr_props {
                f.push(idx); // colr (non-essential)
            }
            // clap is transformative -> essential (applied after ispe to crop).
            if let Some(idx) = clap_prop {
                f.push(0x80 | idx);
            }
            patch(&mut f, p);
        }
        patch(&mut f, s);
    }
    patch(&mut f, meta_start);

    // ── mdat ──────────────────────────────────────────────────────────────────
    let mdat_start = write_box(&mut f, b"mdat");
    let payload_off = f.len();
    f.extend_from_slice(obu);
    let exif_off = f.len();
    if has_exif {
        f.extend_from_slice(&exif_block);
    }
    patch(&mut f, mdat_start);

    // Backfill the iloc extent offsets (absolute file positions in the mdat).
    f[iloc_offset_pos..iloc_offset_pos + 4].copy_from_slice(&(payload_off as u32).to_be_bytes());
    if has_exif {
        f[iloc_exif_pos..iloc_exif_pos + 4].copy_from_slice(&(exif_off as u32).to_be_bytes());
    }

    f
}

/// Wrap an `Encoded` result into an AVIF-style file with explicit colour info.
pub fn to_avif_color(
    enc: &Av2Frame,
    fmt: &Av2Format,
    color: &Av2Color,
    exif: Option<&[u8]>,
) -> Vec<u8> {
    wrap_av2_image(
        &enc.data,
        enc.coded_width as u32,
        enc.coded_height as u32,
        enc.width as u32,
        enc.height as u32,
        fmt,
        color,
        exif,
    )
}

/// Convenience: wrap an `Encoded` result using its CICP colour metadata (`nclx`).
pub fn to_avif(enc: &Av2Frame, fmt: &Av2Format) -> Vec<u8> {
    to_avif_color(enc, fmt, &Av2Color::Cicp(enc.color.clone()), None)
}

/// Convenience: wrap an `Encoded` result embedding an ICC profile (`prof`).
pub fn to_avif_icc(enc: &Av2Frame, fmt: &Av2Format, icc: Vec<u8>) -> Vec<u8> {
    to_avif_color(enc, fmt, &Av2Color::Icc(icc), None)
}

/// Convenience: wrap an `Encoded` result with both CICP (`nclx`) and ICC (`prof`).
pub fn to_avif_cicp_icc(enc: &Av2Frame, fmt: &Av2Format, icc: Vec<u8>) -> Vec<u8> {
    to_avif_color(
        enc,
        fmt,
        &Av2Color::Both {
            cicp: enc.color,
            icc,
        },
        None,
    )
}

/// Convenience: wrap an `Encoded` result with CICP plus an optional ICC profile
/// and/or an EXIF metadata item.
pub fn to_avif_full(
    enc: &Av2Frame,
    fmt: &Av2Format,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Vec<u8> {
    let color = match icc {
        Some(icc) => Av2Color::Both {
            cicp: enc.color,
            icc: icc.to_vec(),
        },
        None => Av2Color::Cicp(enc.color),
    };
    to_avif_color(enc, fmt, &color, exif)
}
