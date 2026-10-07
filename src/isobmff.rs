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

//! ISO Base Media File Format writer for AVIF still images.
//!
//! Adapted from the HEIC isobmff writer. The box hierarchy is identical; the
//! only structural differences from HEIC are:
//!
//! * `ftyp` major brand `avif` instead of `heic`.
//! * Item type `av01` instead of `hvc1`.
//! * Codec configuration box `av1C` instead of `hvcC` (much simpler: a 4-byte
//!   header plus the raw AV1 sequence header OBU as `configOBUs`).
//! * Alpha auxiliary URN `urn:mpeg:mpegB:cicp:systems:auxiliary:alpha` instead
//!   of the HEVC variant.
//! * The `mdat` payload is raw AV1 OBU bytes straight from the encoder (no
//!   HEVC-style length-prefixed NALU framing needed).
//!
//! Box hierarchy (single-item path):
//! ```text
//!   ftyp
//!   meta  (fullbox version=0)
//!     hdlr
//!     pitm
//!     iloc  (version=1, offset_size=4, length_size=4, base_offset_size=0)
//!     iinf → infe
//!     [iref → cdsc]   (only when EXIF item present)
//!     iprp → ipco → { av1C, ispe, pixi, [colr], [irot], [imir], [clli] }
//!            ipma
//!   mdat  ← iloc extent_offset patched after mdat is laid out
//! ```
//!
//! `color_meta` is optional: when `None`, no enumerated-CICP `nclx` `colr` box
//! is written. An ICC `prof` `colr` box may still be present independently. If
//! both are absent the image carries no `colr` property at all (decoders fall
//! back to the CICP values carried in the AV1 sequence header).

use crate::Cicp;
use crate::err::EncodeError;
use crate::metadata::{ContentLightLevel, Metadata};

#[inline]
fn w32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_be_bytes());
}
#[inline]
fn w16(buf: &mut Vec<u8>, v: u16) {
    buf.extend_from_slice(&v.to_be_bytes());
}

/// Write a FullBox header: size placeholder (0), 4-char code, version, flags.
fn write_fullbox(buf: &mut Vec<u8>, cc: &[u8; 4], ver: u8, flags: u32) {
    w32(buf, 0);
    buf.extend_from_slice(cc);
    buf.push(ver);
    buf.push((flags >> 16) as u8);
    buf.push((flags >> 8) as u8);
    buf.push(flags as u8);
}

/// Write a plain Box header: size placeholder (0) + 4-char code.
fn write_box(buf: &mut Vec<u8>, cc: &[u8; 4]) {
    w32(buf, 0);
    buf.extend_from_slice(cc);
}

/// Back-patch the size field at `start` with the current `buf.len() - start`.
fn patch(buf: &mut [u8], start: usize) {
    let size = (buf.len() - start) as u32;
    buf[start..start + 4].copy_from_slice(&size.to_be_bytes());
}

/// Write an enumerated-CICP `nclx` `colr` box from a `ColorEncoding`.
fn write_colr(f: &mut Vec<u8>, color: &Cicp) {
    let sh = f.len();
    write_box(f, b"colr");
    f.extend_from_slice(&color.nclx_payload());
    patch(f, sh);
}

/// Write an ICC-profile `prof` `colr` box.
fn write_colr_icc(f: &mut Vec<u8>, icc: &[u8]) {
    let sh = f.len();
    write_box(f, b"colr");
    f.extend_from_slice(b"prof");
    f.extend_from_slice(icc);
    patch(f, sh);
}

/// Write the color-information `colr` boxes (optional `nclx`, optional `prof`),
/// pushing each box's 1-based ipco index into `colr_props` and advancing
/// `next_idx`. Either, both, or neither may be written.
fn write_colr_boxes(
    f: &mut Vec<u8>,
    color_meta: Option<&Cicp>,
    icc_profile: Option<&[u8]>,
    next_idx: &mut u8,
    colr_props: &mut Vec<u8>,
) {
    if let Some(cm) = color_meta {
        write_colr(f, cm);
        colr_props.push(*next_idx);
        *next_idx += 1;
    }
    if let Some(icc) = icc_profile {
        write_colr_icc(f, icc);
        colr_props.push(*next_idx);
        *next_idx += 1;
    }
}

// ─── AV1 codec config (av1C) ─────────────────────────────────────────────────

/// Parameters for building an `av1C` (AV1 Codec Configuration Record) box.
///
/// Derived from the known encode parameters; the raw AV1 sequence header OBU is
/// extracted from the encoder's output and embedded as `configOBUs` so that
/// decoders can use it without parsing the item data.
pub(crate) struct Av1cParams {
    /// AV1 sequence profile: 0 (main 4:2:0/4:0:0), 1 (high 4:4:4 ≤10-bit),
    /// 2 (professional 4:2:2 or 12-bit).
    pub seq_profile: u8,
    /// `seq_level_idx_0`: computed from image size (see [`level_for`]).
    pub seq_level_idx: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_sub_x: bool,
    pub chroma_sub_y: bool,
    /// The raw sequence header OBU bytes (type 1) from the encoder's output.
    pub seq_header_obu: Vec<u8>,
}

/// Select an AV1 `seq_level_idx_0` large enough for the given picture size.
/// Uses the max-luma-picture-size thresholds from AV1 spec Table A.2.
#[allow(clippy::match_overlapping_arm)]
pub(crate) fn level_for(width: u32, height: u32) -> u8 {
    let pixels = (width as u64) * (height as u64);
    match pixels {
        0..=147_456 => 0,    // 2.0
        ..=278_784 => 1,     // 2.1
        ..=737_280 => 4,     // 3.0
        ..=2_228_224 => 8,   // 4.0
        ..=8_912_896 => 12,  // 5.0
        ..=35_651_584 => 16, // 6.0
        _ => 19,             // 6.3  (max defined for still images)
    }
}

/// Build the raw `av1C` box payload (without the box header).
///
/// The 4-byte header encodes profile, level, tier, bit-depth flags, and chroma
/// subsampling per ISO/IEC 23000-22 §2.3.3. The sequence header OBU (if
/// non-empty) follows as `configOBUs`.
fn build_av1c(p: &Av1cParams) -> Vec<u8> {
    let mut r = Vec::new();
    // Byte 0: marker=1 (bit 7), version=1 (bits 6-0)
    r.push(0x81);
    // Byte 1: seq_profile(3) | seq_level_idx_0(5)
    r.push(((p.seq_profile & 0x7) << 5) | (p.seq_level_idx & 0x1f));
    // Byte 2: seq_tier_0(1) | high_bitdepth(1) | twelve_bit(1) | monochrome(1) |
    //         chroma_sub_x(1) | chroma_sub_y(1) | chroma_sample_position(2)=0
    #[allow(clippy::identity_op)]
    let b2: u8 = 0 // seq_tier_0 = 0 (main tier)
        | (if p.high_bitdepth { 0x40 } else { 0 })
        | (if p.twelve_bit    { 0x20 } else { 0 })
        | (if p.monochrome    { 0x10 } else { 0 })
        | (if p.chroma_sub_x  { 0x08 } else { 0 })
        | (if p.chroma_sub_y  { 0x04 } else { 0 });
    r.push(b2);
    // Byte 3: reserved(3) | initial_presentation_delay_present(1) = 0 | reserved(4) = 0
    r.push(0x00);
    // configOBUs: the sequence header OBU so decoders don't have to scan the item data.
    r.extend_from_slice(&p.seq_header_obu);
    r
}

/// Wrap a single AV1 image into an AVIF file.
///
/// `av1_obu` is the raw AV1 bitstream (temporal delimiter + sequence header +
/// frame OBU(s)) produced by the encoder. `channels` is 3 for color, 1 for
/// grayscale/monochrome. `color_meta` is optional: when `None`, no `nclx`
/// `colr` box is written.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wrap_av1_image(
    av1_obu: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    channels: u8,
    av1c: &Av1cParams,
    color_meta: Option<&Cicp>,
    icc_profile: Option<&[u8]>,
    metadata: &Metadata,
) -> Result<Vec<u8>, EncodeError> {
    let av1c_data = build_av1c(av1c);

    // EXIF item payload: 4-byte offset prefix (always 0 for us) + raw TIFF bytes.
    let has_exif = metadata.exif.is_some();
    let exif_payload: Vec<u8> = metadata
        .exif
        .as_ref()
        .map(|e| {
            let mut p = Vec::with_capacity(e.len() + 4);
            p.extend_from_slice(&0u32.to_be_bytes()); // exif_tiff_header_offset = 0
            p.extend_from_slice(e);
            p
        })
        .unwrap_or_default();

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ──────────────────────────────────────────────────────────────────
    {
        let s = f.len();
        write_box(&mut f, b"ftyp");
        f.extend_from_slice(b"avif"); // major brand
        w32(&mut f, 0); // minor version
        f.extend_from_slice(b"avif"); // AVIF still image
        f.extend_from_slice(b"mif1"); // HEIF base
        f.extend_from_slice(b"miaf"); // Multi-Image Application Format

        patch(&mut f, s);
    }

    // ── meta ──────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr
    {
        let s = f.len();
        write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0); // pre_defined
        f.extend_from_slice(b"pict"); // handler_type
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0); // reserved
        f.push(0); // name (empty)
        patch(&mut f, s);
    }

    // pitm — primary item is ID 1
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    // iloc — version=0; no construction_method field (matches libavif).
    let iloc_offset_patch_pos;
    let mut iloc_exif_patch_pos = 0usize;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, if has_exif { 2 } else { 1 }); // item_count
        // item 1: AV1 image
        w16(&mut f, 1); // item_ID
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1); // extent_count
        iloc_offset_patch_pos = f.len();
        w32(&mut f, 0); // extent_offset — patched later
        w32(&mut f, av1_obu.len() as u32); // extent_length
        if has_exif {
            w16(&mut f, 2); // item_ID
            w16(&mut f, 0); // data_reference_index
            w16(&mut f, 1);
            iloc_exif_patch_pos = f.len();
            w32(&mut f, 0);
            w32(&mut f, exif_payload.len() as u32);
        }
        patch(&mut f, s);
    }

    // iinf
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, if has_exif { 2 } else { 1 }); // entry_count
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 1); // item_ID
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(b"av01"); // item_type — AV1 image
            f.push(0); // item_name (empty)
            patch(&mut f, si);
        }
        if has_exif {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, 2);
            w16(&mut f, 0);
            f.extend_from_slice(b"Exif");
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref — EXIF item (2) describes primary image (1) via 'cdsc'.
    if has_exif {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            let si = f.len();
            write_box(&mut f, b"cdsc");
            w16(&mut f, 2); // from_item_ID = EXIF
            w16(&mut f, 1); // reference_count
            w16(&mut f, 1); // to_item_ID = image
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iprp
    {
        let extra_props;
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco — property container. 1-based indices (matching libavif order):
        //   1 ispe  2 pixi  3 av1C (essential)  4+ colr(s)
        //   then optional: irot, imir, clli
        {
            let si = f.len();
            write_box(&mut f, b"ipco");

            // prop 1: ispe (image spatial extents)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // prop 2: pixi (pixel information)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(channels);
                for _ in 0..channels {
                    f.push(bit_depth);
                }
                patch(&mut f, sh);
            }
            // prop 3: av1C (essential — AV1 decoder configuration)
            {
                let sh = f.len();
                write_box(&mut f, b"av1C");
                f.extend_from_slice(&av1c_data);
                patch(&mut f, sh);
            }
            // prop 4 (+5): colr. MIAF allows at most one colr per color_type,
            // so a CICP `nclx` and an ICC `prof` may coexist. Either may be
            // absent: when `color_meta` is None we skip `nclx`; when no ICC is
            // supplied we skip `prof`. Each written box's 1-based ipco index is
            // tracked for `ipma`.
            let mut colr_props: Vec<u8> = Vec::new();
            let mut next_prop: u8 = 4;
            write_colr_boxes(
                &mut f,
                color_meta,
                icc_profile,
                &mut next_prop,
                &mut colr_props,
            );

            // Optional transform + HDR properties (after the colr boxes)
            let mut irot_idx = 0u8;
            let mut imir_idx = 0u8;
            let mut clli_idx = 0u8;

            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 0x03);
                patch(&mut f, sh);
                irot_idx = next_prop;
                next_prop += 1;
            }
            if let Some(horizontal_axis) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if horizontal_axis { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next_prop;
                next_prop += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next_prop;
                next_prop += 1;
            }
            let _ = next_prop;
            extra_props = (colr_props, irot_idx, imir_idx, clli_idx);
            patch(&mut f, si);
        }

        // ipma — associations for item 1
        {
            let (colr_props, irot_idx, imir_idx, clli_idx) = extra_props;
            // ispe(1), pixi(2), av1C(3, essential), colr(es), then optionals.
            let mut assoc: Vec<u8> = vec![1, 2, 0x80 | 3];
            assoc.extend(colr_props.iter().copied()); // colr boxes (non-essential)
            if irot_idx != 0 {
                assoc.push(0x80 | irot_idx);
            } // essential
            if imir_idx != 0 {
                assoc.push(0x80 | imir_idx);
            } // essential
            if clli_idx != 0 {
                assoc.push(clli_idx);
            } // descriptive only

            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1); // entry_count
            w16(&mut f, 1); // item_ID
            f.push(assoc.len() as u8); // association_count
            f.extend_from_slice(&assoc);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat ──────────────────────────────────────────────────────────────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let av1_abs_offset = f.len() as u32;
    f.extend_from_slice(av1_obu);
    let exif_abs_offset = f.len() as u32;
    if has_exif {
        f.extend_from_slice(&exif_payload);
    }
    patch(&mut f, mdat_start);

    // Patch iloc extent_offsets with real absolute file offsets.
    f[iloc_offset_patch_pos..iloc_offset_patch_pos + 4]
        .copy_from_slice(&av1_abs_offset.to_be_bytes());
    if has_exif {
        f[iloc_exif_patch_pos..iloc_exif_patch_pos + 4]
            .copy_from_slice(&exif_abs_offset.to_be_bytes());
    }

    Ok(f)
}

/// Wrap a color AV1 image plus a monochrome alpha auxiliary image into an AVIF file.
///
/// * Item 1 = color (primary, type `av01`)
/// * Item 2 = alpha (auxiliary, type `av01`)
/// * `iref auxl`: alpha (2) → color (1)
/// * Alpha item carries an `auxC` property with the AVIF alpha URN.
/// * Item 3 = `Exif` (`cdsc` → 1) when `metadata.exif` is set.
/// * `ipma` associates {av1C,ispe,pixi,[colr]} to color and {av1C,ispe,pixi,auxC} to alpha;
///   `irot`/`imir` go to both items (libavif: transforms must also be applied to alpha
///   auxiliary items), `clli` to the color item only.
///
/// `color_meta` is optional: when `None`, no `nclx` `colr` box is written and
/// the alpha item's property indices shift down accordingly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wrap_av1_image_with_alpha(
    color_obu: &[u8],
    alpha_obu: &[u8],
    width: u32,
    height: u32,
    bit_depth: u8,
    av1c_color: &Av1cParams,
    av1c_alpha: &Av1cParams,
    color_meta: Option<&Cicp>,
    icc_profile: Option<&[u8]>,
    metadata: &Metadata,
) -> Result<Vec<u8>, EncodeError> {
    let color_av1c = build_av1c(av1c_color);
    let alpha_av1c = build_av1c(av1c_alpha);

    // AVIF alpha auxiliary URN (ISO/IEC 23000-22:2019 Annex D)
    const ALPHA_URN: &[u8] = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";

    // EXIF item payload: 4-byte offset prefix (always 0 for us) + raw TIFF bytes.
    let exif_payload: Option<Vec<u8>> = metadata.exif.as_ref().map(|e| {
        let mut p = Vec::with_capacity(e.len() + 4);
        p.extend_from_slice(&0u32.to_be_bytes()); // exif_tiff_header_offset = 0
        p.extend_from_slice(e);
        p
    });
    let item_count: u16 = if exif_payload.is_some() { 3 } else { 2 };

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ──────────────────────────────────────────────────────────────────
    {
        let s = f.len();
        write_box(&mut f, b"ftyp");
        f.extend_from_slice(b"avif");
        w32(&mut f, 0);
        f.extend_from_slice(b"avif");
        f.extend_from_slice(b"mif1");
        f.extend_from_slice(b"miaf");

        patch(&mut f, s);
    }

    // ── meta ──────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);

    // hdlr
    {
        let s = f.len();
        write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0);
        f.extend_from_slice(b"pict");
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0);
        f.push(0);
        patch(&mut f, s);
    }

    // pitm → primary (color) item is ID 1
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, 1);
        patch(&mut f, s);
    }

    // iloc — two (three with EXIF) items; offsets patched after mdat. Version=0
    // (no construction_method).
    let color_offset_patch_pos;
    let alpha_offset_patch_pos;
    let mut exif_offset_patch_pos = 0usize;
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, item_count);
        // item 1: color
        w16(&mut f, 1);
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1);
        color_offset_patch_pos = f.len();
        w32(&mut f, 0);
        w32(&mut f, color_obu.len() as u32);
        // item 2: alpha
        w16(&mut f, 2);
        w16(&mut f, 0); // data_reference_index
        w16(&mut f, 1);
        alpha_offset_patch_pos = f.len();
        w32(&mut f, 0);
        w32(&mut f, alpha_obu.len() as u32);
        // item 3: EXIF
        if let Some(p) = &exif_payload {
            w16(&mut f, 3);
            w16(&mut f, 0); // data_reference_index
            w16(&mut f, 1);
            exif_offset_patch_pos = f.len();
            w32(&mut f, 0);
            w32(&mut f, p.len() as u32);
        }
        patch(&mut f, s);
    }

    // iinf — color, alpha, [EXIF]
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, item_count); // entry_count
        for (id, kind) in [(1u16, b"av01"), (2, b"av01"), (3, b"Exif")]
            .into_iter()
            .take(item_count as usize)
        {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, 0);
            w16(&mut f, id);
            w16(&mut f, 0);
            f.extend_from_slice(kind);
            f.push(0);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref — alpha (2) is auxiliary-for color (1): 'auxl' 2 → 1
    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        {
            let sr = f.len();
            write_box(&mut f, b"auxl");
            w16(&mut f, 2); // from_item_ID = alpha
            w16(&mut f, 1); // reference_count
            w16(&mut f, 1); // to_item_ID = color
            patch(&mut f, sr);
        }
        // EXIF (3) describes the color item (1) via 'cdsc' (libavif order: after auxl).
        if exif_payload.is_some() {
            let sr = f.len();
            write_box(&mut f, b"cdsc");
            w16(&mut f, 3); // from_item_ID = EXIF
            w16(&mut f, 1); // reference_count
            w16(&mut f, 1); // to_item_ID = color
            patch(&mut f, sr);
        }
        patch(&mut f, s);
    }

    // iprp
    {
        let s = f.len();
        write_box(&mut f, b"iprp");

        // ipco — property container (1-based, libavif order):
        //   1 ispe  2 pixi(3ch)  3 av1C(color,essential)  4+ colr(s)
        //   then av1C(alpha)  pixi(1ch)  auxC   then optional (irot/imir/clli)
        let mut irot_idx = 0u8;
        let mut imir_idx = 0u8;
        let mut clli_idx = 0u8;
        // Dynamic property indices. The number of colr boxes written (0, 1, or 2)
        // shifts every alpha property index that follows.
        let mut colr_props: Vec<u8> = Vec::new();
        let alpha_av1c_idx;
        let alpha_pixi_idx;
        let auxc_idx;
        {
            let si = f.len();
            write_box(&mut f, b"ipco");

            // 1: ispe (shared dimensions)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"ispe", 0, 0);
                w32(&mut f, width);
                w32(&mut f, height);
                patch(&mut f, sh);
            }
            // 2: pixi (color, 3 channels)
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(3);
                f.push(bit_depth);
                f.push(bit_depth);
                f.push(bit_depth);
                patch(&mut f, sh);
            }
            // 3: av1C (color, essential)
            {
                let sh = f.len();
                write_box(&mut f, b"av1C");
                f.extend_from_slice(&color_av1c);
                patch(&mut f, sh);
            }
            // 4 (+5): colr for the color item — optional nclx and/or prof (see
            // single-image path). The alpha item's properties follow, so their
            // indices depend on how many colr boxes were written.
            let mut next_colr: u8 = 4;
            write_colr_boxes(
                &mut f,
                color_meta,
                icc_profile,
                &mut next_colr,
                &mut colr_props,
            );

            // av1C (alpha)
            alpha_av1c_idx = next_colr;
            {
                let sh = f.len();
                write_box(&mut f, b"av1C");
                f.extend_from_slice(&alpha_av1c);
                patch(&mut f, sh);
            }
            // pixi (alpha, 1 channel)
            alpha_pixi_idx = next_colr + 1;
            {
                let sh = f.len();
                write_fullbox(&mut f, b"pixi", 0, 0);
                f.push(1);
                f.push(bit_depth);
                patch(&mut f, sh);
            }
            // auxC (alpha auxiliary type URN)
            auxc_idx = next_colr + 2;
            {
                let sh = f.len();
                write_fullbox(&mut f, b"auxC", 0, 0);
                f.extend_from_slice(ALPHA_URN);
                patch(&mut f, sh);
            }

            // Optional transform (both items) + HDR (color item only) properties,
            // after auxC.
            let mut next_prop: u8 = next_colr + 3;
            if metadata.orientation.irot_steps() != 0 {
                let sh = f.len();
                write_box(&mut f, b"irot");
                f.push(metadata.orientation.irot_steps() & 0x03);
                patch(&mut f, sh);
                irot_idx = next_prop;
                next_prop += 1;
            }
            if let Some(horizontal_axis) = metadata.orientation.imir_axis() {
                let sh = f.len();
                write_box(&mut f, b"imir");
                f.push(if horizontal_axis { 1 } else { 0 });
                patch(&mut f, sh);
                imir_idx = next_prop;
                next_prop += 1;
            }
            if let Some(cll) = metadata.content_light_level {
                let sh = f.len();
                write_box(&mut f, b"clli");
                f.extend_from_slice(&cll.clli_payload());
                patch(&mut f, sh);
                clli_idx = next_prop;
                next_prop += 1;
            }
            let _ = next_prop;
            patch(&mut f, si);
        }

        // ipma — associations for both items
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 2); // entry_count
            // color item 1: ispe(1), pixi(2), av1C(3,essential), colr(es), clli,
            // then irot/imir — HEIF requires descriptive before transformative.
            let mut c_assoc: Vec<u8> = vec![1, 2, 0x80 | 3];
            c_assoc.extend(colr_props.iter().copied());
            if clli_idx != 0 {
                c_assoc.push(clli_idx);
            }
            if irot_idx != 0 {
                c_assoc.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                c_assoc.push(0x80 | imir_idx);
            }
            w16(&mut f, 1);
            f.push(c_assoc.len() as u8);
            f.extend_from_slice(&c_assoc);
            // alpha item 2: ispe(1), av1C(essential), pixi, auxC (indices shift
            // with the number of colr boxes carried by the color item), then the
            // color item's irot/imir: libavif applies cropping, rotation and
            // mirroring to alpha auxiliary items too.
            let mut a_assoc: Vec<u8> = vec![1, 0x80 | alpha_av1c_idx, alpha_pixi_idx, auxc_idx];
            if irot_idx != 0 {
                a_assoc.push(0x80 | irot_idx);
            }
            if imir_idx != 0 {
                a_assoc.push(0x80 | imir_idx);
            }
            w16(&mut f, 2);
            f.push(a_assoc.len() as u8);
            f.extend_from_slice(&a_assoc);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat ──────────────────────────────────────────────────────────────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let color_abs = f.len() as u32;
    f.extend_from_slice(color_obu);
    let alpha_abs = f.len() as u32;
    f.extend_from_slice(alpha_obu);
    let exif_abs = f.len() as u32;
    if let Some(p) = &exif_payload {
        f.extend_from_slice(p);
    }
    patch(&mut f, mdat_start);

    f[color_offset_patch_pos..color_offset_patch_pos + 4].copy_from_slice(&color_abs.to_be_bytes());
    f[alpha_offset_patch_pos..alpha_offset_patch_pos + 4].copy_from_slice(&alpha_abs.to_be_bytes());
    if exif_payload.is_some() {
        f[exif_offset_patch_pos..exif_offset_patch_pos + 4]
            .copy_from_slice(&exif_abs.to_be_bytes());
    }

    Ok(f)
}

// ─── Gain map container (item model) ─────────────────────────────────────────

/// One coded AV1 image item of a gain-map file.
pub(crate) struct CodedItem<'a> {
    pub obu: &'a [u8],
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    /// 1 for monochrome / alpha, 3 for color.
    pub channels: u8,
    pub av1c: Av1cParams,
}

/// The gain map half of a gain-map file: the coded map, its `nclx`, and the
/// `tmap` derived item's payload and alternate-rendition properties.
pub(crate) struct GainMapItems<'a> {
    pub gain_map: CodedItem<'a>,
    pub gain_map_cicp: Cicp,
    pub tmap_payload: &'a [u8],
    pub alternate_cicp: Cicp,
    pub alternate_icc: Option<&'a [u8]>,
    pub alternate_clli: Option<ContentLightLevel>,
    pub alternate_pixi: Option<(u8, u8)>,
}

/// ipco builder that, like libavif's `avifItemPropertyDedup`, writes a
/// byte-identical property once and shares its index between items.
struct PropertyContainer {
    ipco: Vec<u8>,
    props: Vec<Vec<u8>>,
}

impl PropertyContainer {
    /// Add a complete property box; returns its ipma association byte.
    fn add(&mut self, bx: Vec<u8>, essential: bool) -> Result<u8, EncodeError> {
        let idx = match self.props.iter().position(|p| *p == bx) {
            Some(i) => i + 1,
            None => {
                self.ipco.extend_from_slice(&bx);
                self.props.push(bx);
                self.props.len()
            }
        };
        if idx > 0x7f {
            return Err(EncodeError::IsobmffError(
                "more than 127 item properties".into(),
            ));
        }
        Ok(if essential { 0x80 } else { 0 } | idx as u8)
    }
}

fn property(cc: &[u8; 4], full: Option<(u8, u32)>, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(12 + body.len());
    match full {
        Some((ver, flags)) => write_fullbox(&mut b, cc, ver, flags),
        None => write_box(&mut b, cc),
    }
    b.extend_from_slice(body);
    patch(&mut b, 0);
    b
}

fn ispe(width: u32, height: u32) -> Vec<u8> {
    let mut body = width.to_be_bytes().to_vec();
    body.extend_from_slice(&height.to_be_bytes());
    property(b"ispe", Some((0, 0)), &body)
}

fn pixi(channels: u8, depth: u8) -> Vec<u8> {
    let mut body = vec![channels];
    body.extend(std::iter::repeat_n(depth, channels as usize));
    property(b"pixi", Some((0, 0)), &body)
}

fn colr_prof(icc: &[u8]) -> Vec<u8> {
    let mut body = b"prof".to_vec();
    body.extend_from_slice(icc);
    property(b"colr", None, &body)
}

fn coded_item_props(pc: &mut PropertyContainer, item: &CodedItem) -> Result<Vec<u8>, EncodeError> {
    Ok(vec![
        pc.add(ispe(item.width, item.height), false)?,
        pc.add(pixi(item.channels, item.bit_depth), false)?,
        pc.add(property(b"av1C", None, &build_av1c(&item.av1c)), true)?,
    ])
}

/// `irot` / `imir` (transformative, essential) for `metadata.orientation`.
fn transform_props(
    pc: &mut PropertyContainer,
    metadata: &Metadata,
) -> Result<Vec<u8>, EncodeError> {
    let mut assoc = Vec::new();
    let steps = metadata.orientation.irot_steps();
    if steps != 0 {
        assoc.push(pc.add(property(b"irot", None, &[steps & 0x03]), true)?);
    }
    if let Some(horizontal_axis) = metadata.orientation.imir_axis() {
        assoc.push(pc.add(property(b"imir", None, &[horizontal_axis as u8]), true)?);
    }
    Ok(assoc)
}

/// Wrap a base image (with optional alpha) and an ISO 21496-1 gain map into an
/// AVIF file, with the item layout libavif's `avifEncoderAddImageInternal`
/// produces (see `crate::gain_map`):
///
/// * 1 `av01` base, primary; 2 `av01` alpha (`auxl` → 1) when present;
///   then `tmap` (`dimg` → [base, gain map]); then the hidden `av01` gain map;
///   then `Exif` (`cdsc` → 1).
/// * `grpl`/`altr` = [tmap, base] so gain-map aware readers prefer `tmap`.
/// * `ftyp` gains the `tmap` brand, without which libavif ignores the item.
///
/// Property placement follows libavif: descriptive properties before
/// transformative ones; base/alpha/gain map items carry the `irot`/`imir`,
/// the `tmap` item carries none; the base color `colr`/`clli` describe the
/// base rendition, the `tmap` ones the alternate rendition.
#[allow(clippy::too_many_arguments)]
pub(crate) fn wrap_av1_image_with_gain_map(
    base: &CodedItem,
    alpha: Option<&CodedItem>,
    gain: &GainMapItems,
    color_meta: Option<&Cicp>,
    icc_profile: Option<&[u8]>,
    metadata: &Metadata,
) -> Result<Vec<u8>, EncodeError> {
    const ALPHA_URN: &[u8] = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";

    let exif_payload: Option<Vec<u8>> = metadata.exif.as_ref().map(|e| {
        let mut p = Vec::with_capacity(e.len() + 4);
        p.extend_from_slice(&0u32.to_be_bytes()); // exif_tiff_header_offset = 0
        p.extend_from_slice(e);
        p
    });

    struct Item<'a> {
        id: u16,
        kind: &'a [u8; 4],
        name: &'a [u8],
        hidden: bool,
        data: &'a [u8],
        assoc: Vec<u8>,
    }

    let mut pc = PropertyContainer {
        ipco: Vec::new(),
        props: Vec::new(),
    };
    let mut items: Vec<Item> = Vec::new();
    let mut next_id = 1u16;
    let mut take_id = || {
        let id = next_id;
        next_id += 1;
        id
    };

    // Base color item: ispe, pixi, av1C, colr prof, colr nclx, clli, irot/imir.
    let base_id = take_id();
    let mut assoc = coded_item_props(&mut pc, base)?;
    if let Some(icc) = icc_profile {
        assoc.push(pc.add(colr_prof(icc), false)?);
    }
    if let Some(cm) = color_meta {
        assoc.push(pc.add(property(b"colr", None, &cm.nclx_payload()), false)?);
    }
    if let Some(cll) = metadata.content_light_level {
        assoc.push(pc.add(property(b"clli", None, &cll.clli_payload()), false)?);
    }
    assoc.extend(transform_props(&mut pc, metadata)?);
    items.push(Item {
        id: base_id,
        kind: b"av01",
        name: b"Color\0",
        hidden: false,
        data: base.obu,
        assoc,
    });

    // Alpha: ispe, pixi, av1C, auxC, irot/imir (transforms apply to alpha too).
    let alpha_id = match alpha {
        Some(a) => {
            let id = take_id();
            let mut assoc = coded_item_props(&mut pc, a)?;
            assoc.push(pc.add(property(b"auxC", Some((0, 0)), ALPHA_URN), false)?);
            assoc.extend(transform_props(&mut pc, metadata)?);
            items.push(Item {
                id,
                kind: b"av01",
                name: b"Alpha\0",
                hidden: false,
                data: a.obu,
                assoc,
            });
            Some(id)
        }
        None => None,
    };

    // tmap: ispe (base size), [pixi], colr prof, colr nclx, clli — all of the
    // alternate rendition. No transformative properties (libavif).
    let tmap_id = take_id();
    {
        let mut assoc = vec![pc.add(ispe(base.width, base.height), false)?];
        if let Some((channels, depth)) = gain.alternate_pixi {
            assoc.push(pc.add(pixi(channels, depth), false)?);
        }
        if let Some(icc) = gain.alternate_icc {
            assoc.push(pc.add(colr_prof(icc), false)?);
        }
        assoc.push(pc.add(
            property(b"colr", None, &gain.alternate_cicp.nclx_payload()),
            false,
        )?);
        if let Some(cll) = gain.alternate_clli {
            assoc.push(pc.add(property(b"clli", None, &cll.clli_payload()), false)?);
        }
        items.push(Item {
            id: tmap_id,
            kind: b"tmap",
            name: b"GMap\0",
            hidden: false,
            data: gain.tmap_payload,
            assoc,
        });
    }

    // Gain map: ispe, pixi, av1C, colr nclx, then the base's irot/imir.
    let gain_id = take_id();
    {
        let mut assoc = coded_item_props(&mut pc, &gain.gain_map)?;
        assoc.push(pc.add(
            property(b"colr", None, &gain.gain_map_cicp.nclx_payload()),
            false,
        )?);
        assoc.extend(transform_props(&mut pc, metadata)?);
        items.push(Item {
            id: gain_id,
            kind: b"av01",
            name: b"GMap\0",
            hidden: true,
            data: gain.gain_map.obu,
            assoc,
        });
    }

    let exif_id = match exif_payload.as_deref() {
        Some(p) => {
            let id = take_id();
            items.push(Item {
                id,
                kind: b"Exif",
                name: b"Exif\0",
                hidden: false,
                data: p,
                assoc: Vec::new(),
            });
            Some(id)
        }
        None => None,
    };
    let group_id = take_id() as u32; // must differ from every item_ID

    let mut f: Vec<u8> = Vec::new();

    // ── ftyp ──────────────────────────────────────────────────────────────────
    {
        let s = f.len();
        write_box(&mut f, b"ftyp");
        f.extend_from_slice(b"avif");
        w32(&mut f, 0);
        for brand in [b"avif", b"mif1", b"miaf", b"tmap"] {
            f.extend_from_slice(brand);
        }
        patch(&mut f, s);
    }

    // ── meta ──────────────────────────────────────────────────────────────────
    let meta_start = f.len();
    write_fullbox(&mut f, b"meta", 0, 0);
    {
        let s = f.len();
        write_fullbox(&mut f, b"hdlr", 0, 0);
        w32(&mut f, 0);
        f.extend_from_slice(b"pict");
        w32(&mut f, 0);
        w32(&mut f, 0);
        w32(&mut f, 0);
        f.push(0);
        patch(&mut f, s);
    }
    {
        let s = f.len();
        write_fullbox(&mut f, b"pitm", 0, 0);
        w16(&mut f, base_id);
        patch(&mut f, s);
    }

    // iloc — version 0, 4-byte offsets/lengths, one extent per item.
    let mut offset_patches = Vec::with_capacity(items.len());
    {
        let s = f.len();
        write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44);
        f.push(0x00);
        w16(&mut f, items.len() as u16);
        for item in &items {
            let len = u32::try_from(item.data.len())
                .map_err(|_| EncodeError::IsobmffError("item exceeds 4 GiB".into()))?;
            w16(&mut f, item.id);
            w16(&mut f, 0); // data_reference_index
            w16(&mut f, 1); // extent_count
            offset_patches.push(f.len());
            w32(&mut f, 0); // extent_offset — patched after mdat
            w32(&mut f, len);
        }
        patch(&mut f, s);
    }

    // iinf
    {
        let s = f.len();
        write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, items.len() as u16);
        for item in &items {
            let si = f.len();
            write_fullbox(&mut f, b"infe", 2, item.hidden as u32);
            w16(&mut f, item.id);
            w16(&mut f, 0); // item_protection_index
            f.extend_from_slice(item.kind);
            f.extend_from_slice(item.name);
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // iref — libavif order: auxl (alpha), dimg (tmap → base, gain map), cdsc.
    {
        let s = f.len();
        write_fullbox(&mut f, b"iref", 0, 0);
        let reference = |f: &mut Vec<u8>, kind: &[u8; 4], from: u16, to: &[u16]| {
            let si = f.len();
            write_box(f, kind);
            w16(f, from);
            w16(f, to.len() as u16);
            for &t in to {
                w16(f, t);
            }
            patch(f, si);
        };
        if let Some(a) = alpha_id {
            reference(&mut f, b"auxl", a, &[base_id]);
        }
        // Order matters: the first dimg input is the base, the second the map.
        reference(&mut f, b"dimg", tmap_id, &[base_id, gain_id]);
        if let Some(e) = exif_id {
            reference(&mut f, b"cdsc", e, &[base_id]);
        }
        patch(&mut f, s);
    }

    // iprp → ipco, ipma
    {
        let s = f.len();
        write_box(&mut f, b"iprp");
        {
            let si = f.len();
            write_box(&mut f, b"ipco");
            f.extend_from_slice(&pc.ipco);
            patch(&mut f, si);
        }
        {
            let si = f.len();
            write_fullbox(&mut f, b"ipma", 0, 0);
            let with_props: Vec<&Item> = items.iter().filter(|i| !i.assoc.is_empty()).collect();
            w32(&mut f, with_props.len() as u32);
            for item in with_props {
                w16(&mut f, item.id);
                f.push(item.assoc.len() as u8);
                f.extend_from_slice(&item.assoc);
            }
            patch(&mut f, si);
        }
        patch(&mut f, s);
    }

    // grpl → altr: the tone-mapped image is the preferred alternative.
    {
        let s = f.len();
        write_box(&mut f, b"grpl");
        let sa = f.len();
        write_fullbox(&mut f, b"altr", 0, 0);
        w32(&mut f, group_id);
        w32(&mut f, 2); // num_entities_in_group
        w32(&mut f, tmap_id as u32);
        w32(&mut f, base_id as u32);
        patch(&mut f, sa);
        patch(&mut f, s);
    }

    patch(&mut f, meta_start);

    // ── mdat ──────────────────────────────────────────────────────────────────
    let mdat_start = f.len();
    write_box(&mut f, b"mdat");
    let mut offsets = Vec::with_capacity(items.len());
    for item in &items {
        offsets.push(
            u32::try_from(f.len())
                .map_err(|_| EncodeError::IsobmffError("file exceeds 4 GiB".into()))?,
        );
        f.extend_from_slice(item.data);
    }
    patch(&mut f, mdat_start);
    for (pos, off) in offset_patches.into_iter().zip(offsets) {
        f[pos..pos + 4].copy_from_slice(&off.to_be_bytes());
    }

    Ok(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_av1c() -> Av1cParams {
        Av1cParams {
            seq_profile: 0,
            seq_level_idx: 12,
            high_bitdepth: false,
            twelve_bit: false,
            monochrome: false,
            chroma_sub_x: true,
            chroma_sub_y: true,
            seq_header_obu: vec![],
        }
    }

    // Count colr boxes and return (count, has_nclx, has_prof, ipma_assoc_count).
    fn colr_summary(b: &[u8]) -> (usize, bool, bool, u8) {
        let mut count = 0;
        let mut nclx = false;
        let mut prof = false;
        let mut i = 0;
        while i + 8 <= b.len() {
            if &b[i + 4..i + 8] == b"colr" {
                count += 1;
                match &b[i + 8..i + 12] {
                    b"nclx" => nclx = true,
                    b"prof" => prof = true,
                    _ => {}
                }
            }
            i += 1;
        }
        let ipma = b.array_windows::<4>().position(|w| w == b"ipma").unwrap();
        // fullbox(12) + entry_count(4) + item_ID(2) => association_count byte
        let ac = b[ipma + 8 + 4 + 2]; // fullbox(8..12) + entry_count(4) + item_ID(2)
        (count, nclx, prof, ac)
    }

    #[test]
    fn cicp_plus_icc_writes_both_colr() {
        let icc = vec![0xAAu8; 64];
        let b = wrap_av1_image(
            &[0x00],
            16,
            16,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::srgb()),
            Some(&icc),
            &Metadata::default(),
        )
        .unwrap();
        let (count, nclx, prof, ac) = colr_summary(&b);
        assert_eq!(count, 2, "expect nclx + prof");
        assert!(nclx && prof, "both colr types present");
        assert_eq!(ac, 5, "ipma: ispe,pixi,av1C,nclx,prof");
        // prof payload must contain the ICC bytes verbatim
        let p = b.array_windows::<4>().rposition(|w| w == b"prof").unwrap();
        assert_eq!(&b[p + 4..p + 4 + icc.len()], &icc[..], "ICC bytes embedded");
    }

    #[test]
    fn cicp_only_single_colr() {
        let b = wrap_av1_image(
            &[0x00],
            16,
            16,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::srgb()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        let (count, nclx, prof, ac) = colr_summary(&b);
        assert_eq!((count, nclx, prof, ac), (1, true, false, 4));
    }

    #[test]
    fn icc_only_single_colr() {
        let icc = vec![0x3Cu8; 48];
        let b = wrap_av1_image(
            &[0x00],
            16,
            16,
            8,
            3,
            &dummy_av1c(),
            None,
            Some(&icc),
            &Metadata::default(),
        )
        .unwrap();
        let (count, nclx, prof, ac) = colr_summary(&b);
        // Only the prof colr is written; ipma assoc = ispe,pixi,av1C,prof = 4.
        assert_eq!((count, nclx, prof, ac), (1, false, true, 4));
        let p = b.array_windows::<4>().rposition(|w| w == b"prof").unwrap();
        assert_eq!(&b[p + 4..p + 4 + icc.len()], &icc[..], "ICC bytes embedded");
    }

    #[test]
    fn no_color_meta_no_colr_box() {
        let b = wrap_av1_image(
            &[0x00],
            16,
            16,
            8,
            3,
            &dummy_av1c(),
            None,
            None,
            &Metadata::default(),
        )
        .unwrap();
        let (count, nclx, prof, ac) = colr_summary(&b);
        // No colr boxes at all; ipma assoc = ispe,pixi,av1C = 3.
        assert_eq!((count, nclx, prof, ac), (0, false, false, 3));
    }

    #[test]
    fn alpha_with_cicp_plus_icc_shifts_indices() {
        let icc = vec![0x5Au8; 40];
        let b = wrap_av1_image_with_alpha(
            &[0x10],
            &[0x20],
            16,
            16,
            8,
            &dummy_av1c(),
            &dummy_av1c(),
            Some(&crate::color::Cicp::srgb()),
            Some(&icc),
            &Metadata::default(),
        )
        .unwrap();
        // two colr boxes on the color item
        let ncolr = (0..b.len().saturating_sub(8))
            .filter(|&i| &b[i + 4..i + 8] == b"colr")
            .count();
        assert_eq!(ncolr, 2, "color item should have nclx + prof");
        // ipma second entry (alpha) must reference av1C at index 6 (shifted from 5)
        let ipma = b.array_windows::<4>().position(|w| w == b"ipma").unwrap();
        // entry1: item_ID(2) assoc_count(1) + assoc bytes; color assoc =
        // [1,2,0x83,4,5] => count 5. Then entry2 item_ID(2)=2, count(1)=4, assoc...
        let p = ipma + 8 + 4; // -> first entry item_ID
        let c_count = b[p + 2] as usize; // color assoc_count
        assert_eq!(c_count, 5, "color: ispe,pixi,av1C,nclx,prof");
        let a = p + 3 + c_count; // alpha entry item_ID
        let a_count = b[a + 2] as usize;
        let a_assoc = &b[a + 3..a + 3 + a_count];
        assert_eq!(
            a_assoc,
            &[1u8, 0x80 | 6, 7, 8],
            "alpha: ispe, av1C(6,ess), pixi(7), auxC(8)"
        );
    }

    #[test]
    fn alpha_without_color_meta_shifts_indices_down() {
        // No nclx and no ICC → zero colr boxes, so alpha props start at index 4.
        let b = wrap_av1_image_with_alpha(
            &[0x10],
            &[0x20],
            16,
            16,
            8,
            &dummy_av1c(),
            &dummy_av1c(),
            None,
            None,
            &Metadata::default(),
        )
        .unwrap();
        let ncolr = (0..b.len().saturating_sub(8))
            .filter(|&i| &b[i + 4..i + 8] == b"colr")
            .count();
        assert_eq!(
            ncolr, 0,
            "no colr box when color_meta and icc are both None"
        );
        let ipma = b.array_windows::<4>().position(|w| w == b"ipma").unwrap();
        let p = ipma + 8 + 4;
        let c_count = b[p + 2] as usize; // color assoc_count = ispe,pixi,av1C = 3
        assert_eq!(c_count, 3, "color: ispe,pixi,av1C (no colr)");
        let a = p + 3 + c_count;
        let a_count = b[a + 2] as usize;
        let a_assoc = &b[a + 3..a + 3 + a_count];
        assert_eq!(
            a_assoc,
            &[1u8, 0x80 | 4, 5, 6],
            "alpha: ispe, av1C(4,ess), pixi(5), auxC(6)"
        );
    }

    #[test]
    fn ftyp_brand_avif() {
        let b = wrap_av1_image(
            &[0xAB, 0xCD],
            16,
            16,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        let ftyp_start = u32::from_be_bytes(*b.first_chunk().unwrap()) as usize;
        assert_eq!(&b[4..8], b"ftyp", "first box must be ftyp");
        assert_eq!(&b[8..12], b"avif", "major brand must be avif");
        // meta follows ftyp
        assert_eq!(
            &b[ftyp_start + 4..ftyp_start + 8],
            b"meta",
            "meta must follow ftyp"
        );
    }

    #[test]
    fn av01_item_type() {
        let b = wrap_av1_image(
            &[0x00],
            8,
            8,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        assert!(
            b.array_windows::<4>().any(|w| w == b"av01"),
            "av01 item type must be present"
        );
        assert!(
            !b.array_windows::<4>().any(|w| w == b"hvc1"),
            "hvc1 must not appear"
        );
    }

    #[test]
    fn av1c_box_present() {
        let b = wrap_av1_image(
            &[0x00],
            8,
            8,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        assert!(
            b.array_windows::<4>().any(|w| w == b"av1C"),
            "av1C box must be present"
        );
        assert!(
            !b.array_windows::<4>().any(|w| w == b"hvcC"),
            "hvcC must not appear"
        );
    }

    #[test]
    fn alpha_container_structure() {
        let b = wrap_av1_image_with_alpha(
            &[0x10],
            &[0x20],
            16,
            16,
            8,
            &dummy_av1c(),
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        let s = b.as_slice();
        assert!(
            s.array_windows::<4>().any(|w| w == b"iref"),
            "iref box required"
        );
        assert!(
            s.array_windows::<4>().any(|w| w == b"auxl"),
            "auxl reference required"
        );
        assert!(
            s.array_windows::<4>().any(|w| w == b"auxC"),
            "auxC property required"
        );
        assert!(
            s.windows(ALPHA_URN.len() - 1)
                .any(|w| w == &ALPHA_URN[..ALPHA_URN.len() - 1]),
            "AVIF alpha URN must be present"
        );
        let ipma_pos = s.array_windows::<4>().position(|w| w == b"ipma").unwrap();
        let entry_count = u32::from_be_bytes(*s[ipma_pos + 8..].first_chunk().unwrap());
        assert_eq!(entry_count, 2, "ipma must have 2 entries (color + alpha)");
    }

    #[test]
    fn iloc_offset_points_into_mdat() {
        let payload = b"hello av1";
        let b = wrap_av1_image(
            payload,
            8,
            8,
            8,
            3,
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            &Metadata::default(),
        )
        .unwrap();
        // Find mdat start
        let mut pos = 0;
        let mut mdat_payload_start = 0u32;
        while pos + 8 <= b.len() {
            let sz = u32::from_be_bytes(*b[pos..].first_chunk().unwrap()) as usize;
            if &b[pos + 4..pos + 8] == b"mdat" {
                mdat_payload_start = (pos + 8) as u32;
                break;
            }
            pos += sz;
        }
        assert!(mdat_payload_start > 0);
        // Locate the av1 extent_offset in iloc
        let iloc_pos = b.array_windows::<4>().position(|w| w == b"iloc").unwrap() - 4;
        // iloc: 8(box) + 4(fullbox) + 2(fields) + 2(item_count) = 16 bytes before first item
        // first item v0: 2(id)+2(ref)+2(cnt) = 6, then extent_offset (4)
        let off_pos = iloc_pos + 16 + 6;
        let extent_off = u32::from_be_bytes(*b[off_pos..].first_chunk().unwrap());
        assert_eq!(
            extent_off, mdat_payload_start,
            "iloc extent_offset must point into mdat"
        );
        // Payload at that offset must match
        let data_at_offset = &b[extent_off as usize..extent_off as usize + payload.len()];
        assert_eq!(data_at_offset, payload);
    }

    #[test]
    fn av1c_header_bytes() {
        let p = Av1cParams {
            seq_profile: 1,
            seq_level_idx: 12,
            high_bitdepth: true,
            twelve_bit: false,
            monochrome: false,
            chroma_sub_x: false,
            chroma_sub_y: false,
            seq_header_obu: vec![],
        };
        let av1c = build_av1c(&p);
        assert_eq!(av1c[0], 0x81, "marker=1, version=1");
        assert_eq!((av1c[1] >> 5) & 0x7, 1, "seq_profile=1");
        assert_eq!(av1c[1] & 0x1f, 12, "seq_level_idx=12");
        assert_ne!(av1c[2] & 0x40, 0, "high_bitdepth must be set");
        assert_eq!(av1c[2] & 0x20, 0, "twelve_bit must be clear");
    }

    const ALPHA_URN: &[u8] = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";

    /// Child boxes of `b[start..end]` as (fourcc, payload start, box end).
    fn boxes(b: &[u8], start: usize, end: usize) -> Vec<([u8; 4], usize, usize)> {
        let mut out = Vec::new();
        let mut pos = start;
        while pos + 8 <= end {
            let sz = u32::from_be_bytes(*b[pos..].first_chunk().unwrap()) as usize;
            out.push((*b[pos + 4..].first_chunk().unwrap(), pos + 8, pos + sz));
            pos += sz;
        }
        out
    }

    /// The (payload start, end) of the first child `cc` of `b[start..end]`.
    fn child(b: &[u8], start: usize, end: usize, cc: &[u8; 4]) -> (usize, usize) {
        let (_, s, e) = *boxes(b, start, end)
            .iter()
            .find(|(c, ..)| c == cc)
            .unwrap_or_else(|| panic!("no {} box", String::from_utf8_lossy(cc)));
        (s, e)
    }

    fn rd16(b: &[u8], at: usize) -> u16 {
        u16::from_be_bytes(*b[at..].first_chunk().unwrap())
    }

    fn rd32(b: &[u8], at: usize) -> u32 {
        u32::from_be_bytes(*b[at..].first_chunk().unwrap())
    }

    /// The `meta` payload range (after the full-box header).
    fn meta(b: &[u8]) -> (usize, usize) {
        let (s, e) = child(b, 0, b.len(), b"meta");
        (s + 4, e)
    }

    /// ipma associations as (item_ID, association bytes); version 0, flags 0.
    fn ipma_entries(b: &[u8]) -> Vec<(u16, Vec<u8>)> {
        let (ms, me) = meta(b);
        let (ps, pe) = child(b, ms, me, b"iprp");
        let (s, _) = child(b, ps, pe, b"ipma");
        let mut pos = s + 4;
        let count = rd32(b, pos);
        pos += 4;
        (0..count)
            .map(|_| {
                let id = rd16(b, pos);
                let n = b[pos + 2] as usize;
                let assoc = b[pos + 3..pos + 3 + n].to_vec();
                pos += 3 + n;
                (id, assoc)
            })
            .collect()
    }

    /// 1-based ipco index of the first property `cc`.
    fn ipco_index(b: &[u8], cc: &[u8; 4]) -> u8 {
        let (ms, me) = meta(b);
        let (ps, pe) = child(b, ms, me, b"iprp");
        let (s, e) = child(b, ps, pe, b"ipco");
        boxes(b, s, e).iter().position(|(c, ..)| c == cc).unwrap() as u8 + 1
    }

    fn wrap_alpha(metadata: &Metadata) -> Vec<u8> {
        wrap_av1_image_with_alpha(
            &[0x10, 0x11],
            &[0x20, 0x21, 0x22],
            16,
            16,
            8,
            &dummy_av1c(),
            &dummy_av1c(),
            Some(&Cicp::default()),
            None,
            metadata,
        )
        .unwrap()
    }

    #[test]
    fn alpha_with_exif_writes_exif_item_and_cdsc() {
        let exif = b"MM\0\x2a\0\0\0\x08exif-body".to_vec();
        let b = wrap_alpha(&Metadata::new().with_exif(exif.clone()));
        let (ms, me) = meta(&b);

        // iinf: color, alpha, Exif.
        let (is, ie) = child(&b, ms, me, b"iinf");
        assert_eq!(rd16(&b, is + 4), 3, "iinf entry_count");
        let infes: Vec<(u16, [u8; 4])> = boxes(&b, is + 6, ie)
            .iter()
            .map(|&(_, s, _)| (rd16(&b, s + 4), *b[s + 8..].first_chunk().unwrap()))
            .collect();
        assert_eq!(infes, vec![(1, *b"av01"), (2, *b"av01"), (3, *b"Exif")]);

        // iref: auxl 2 → 1, then cdsc 3 → 1.
        let (rs, re) = child(&b, ms, me, b"iref");
        let refs: Vec<([u8; 4], u16, u16, u16)> = boxes(&b, rs + 4, re)
            .iter()
            .map(|&(c, s, _)| (c, rd16(&b, s), rd16(&b, s + 2), rd16(&b, s + 4)))
            .collect();
        assert_eq!(refs, vec![(*b"auxl", 2, 1, 1), (*b"cdsc", 3, 1, 1)]);

        // iloc item 3 points at the 4-byte tiff offset + EXIF bytes in mdat;
        // items 1 and 2 still point at their OBUs.
        let (ls, _) = child(&b, ms, me, b"iloc");
        assert_eq!(rd16(&b, ls + 6), 3, "iloc item_count");
        let mut expected = vec![0u8; 4];
        expected.extend_from_slice(&exif);
        let payloads: [&[u8]; 3] = [&[0x10, 0x11], &[0x20, 0x21, 0x22], &expected];
        for (i, want) in payloads.iter().enumerate() {
            let at = ls + 8 + i * 14;
            assert_eq!(rd16(&b, at) as usize, i + 1, "iloc item_ID");
            let off = rd32(&b, at + 6) as usize;
            let len = rd32(&b, at + 10) as usize;
            assert_eq!(&b[off..off + len], *want, "item {} extent", i + 1);
        }
        let (_, md_s, md_e) = *boxes(&b, 0, b.len())
            .iter()
            .find(|(c, ..)| c == b"mdat")
            .unwrap();
        assert_eq!(md_e - md_s, 2 + 3 + expected.len(), "mdat payload size");
    }

    #[test]
    fn alpha_transforms_associated_with_both_items() {
        // Transpose = irot + imir; clli must stay on the color item only.
        let md = Metadata::new()
            .with_orientation(crate::metadata::Orientation::Transpose)
            .with_content_light_level(crate::metadata::ContentLightLevel::new(1000, 400));
        let b = wrap_alpha(&md);
        let irot = 0x80 | ipco_index(&b, b"irot");
        let imir = 0x80 | ipco_index(&b, b"imir");
        let clli = ipco_index(&b, b"clli");
        let auxc = ipco_index(&b, b"auxC");
        let entries = ipma_entries(&b);
        assert_eq!(entries.len(), 2);
        let (color, alpha) = (&entries[0], &entries[1]);
        assert_eq!((color.0, alpha.0), (1, 2));
        for (id, assoc) in [color, alpha] {
            assert!(assoc.contains(&irot), "irot not associated with item {id}");
            assert!(assoc.contains(&imir), "imir not associated with item {id}");
            // Transformative properties come after every descriptive one.
            let first_tx = assoc.iter().position(|&a| a == irot).unwrap();
            assert!(
                assoc[first_tx..].iter().all(|&a| a == irot || a == imir),
                "item {id}: descriptive property after a transform"
            );
        }
        assert!(color.1.contains(&clli), "clli on color");
        assert!(!alpha.1.contains(&clli), "clli must not be on alpha");
        assert_eq!(&alpha.1[3..], &[auxc, irot, imir]);
    }

    #[test]
    fn alpha_without_metadata_has_no_exif_or_transforms() {
        let b = wrap_alpha(&Metadata::default());
        let (ms, me) = meta(&b);
        let (is, _) = child(&b, ms, me, b"iinf");
        assert_eq!(rd16(&b, is + 4), 2, "iinf entry_count");
        let (rs, re) = child(&b, ms, me, b"iref");
        assert_eq!(boxes(&b, rs + 4, re).len(), 1, "auxl only");
        let entries = ipma_entries(&b);
        assert_eq!(entries[1].1.len(), 4, "alpha: ispe, av1C, pixi, auxC");
        assert!(!b.array_windows::<4>().any(|w| w == b"irot" || w == b"imir"));
    }
}
