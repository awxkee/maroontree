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
//! AVIF (ISO/IEC 23000-22) standardises AV1 in a MIAF/ISOBMFF file: item type
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
use crate::av2::AvFrame;

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
    pub fn yuv444(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            monochrome: false,
            chroma_sub_x: false,
            chroma_sub_y: false,
        }
    }
    pub fn yuv422(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            monochrome: false,
            chroma_sub_x: true,
            chroma_sub_y: false,
        }
    }
    pub fn yuv420(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            monochrome: false,
            chroma_sub_x: true,
            chroma_sub_y: true,
        }
    }
    pub fn mono(bit_depth: u8) -> Self {
        Self {
            bit_depth,
            monochrome: true,
            chroma_sub_x: true,
            chroma_sub_y: true,
        }
    }
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
    fmt: &Av2Format,
    color: &ColorEncoding,
) -> Vec<u8> {
    let channels = fmt.channels();
    let av2c = build_av2c(fmt, width, height);
    let mut f = Vec::with_capacity(obu.len() + 512);

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
    // iloc — one item, one extent; offset patched once mdat position is known.
    let iloc_offset_pos;
    {
        let s = write_fullbox(&mut f, b"iloc", 0, 0);
        f.push(0x44); // offset_size=4, length_size=4
        f.push(0x00); // base_offset_size=0, index_size=0
        w16(&mut f, 1); // item_count
        w16(&mut f, 1); // item_ID
        w16(&mut f, 0); // data_reference_index (0 = this file)
        w16(&mut f, 1); // extent_count
        iloc_offset_pos = f.len();
        w32(&mut f, 0); // extent_offset — patched after mdat is placed
        w32(&mut f, obu.len() as u32); // extent_length
        patch(&mut f, s);
    }
    // iinf → infe ('av02')
    {
        let s = write_fullbox(&mut f, b"iinf", 0, 0);
        w16(&mut f, 1); // entry_count
        let si = write_fullbox(&mut f, b"infe", 2, 0);
        w16(&mut f, 1); // item_ID
        w16(&mut f, 0); // item_protection_index
        f.extend_from_slice(b"av02"); // item_type — AV2 image (this project's 4CC)
        f.push(0); // item_name (empty)
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
        // prop 4: colr (nclx CICP)
        {
            let p = write_box(&mut f, b"colr");
            f.extend_from_slice(b"nclx");
            w16(&mut f, color.primaries as u16);
            w16(&mut f, color.transfer as u16);
            w16(&mut f, color.matrix as u16);
            f.push(if color.full_range { 0x80 } else { 0x00 });
            patch(&mut f, p);
        }
        patch(&mut f, ipco);
        // ipma — associate item 1 with the four properties (av2C is essential).
        {
            let p = write_fullbox(&mut f, b"ipma", 0, 0);
            w32(&mut f, 1); // entry_count
            w16(&mut f, 1); // item_ID
            f.push(4); // association_count
            f.push(1); // ispe (non-essential)
            f.push(2); // pixi
            f.push(0x80 | 3); // av2C (essential bit set)
            f.push(4); // colr
            patch(&mut f, p);
        }
        patch(&mut f, s);
    }
    patch(&mut f, meta_start);

    // ── mdat ──────────────────────────────────────────────────────────────────
    let mdat_start = write_box(&mut f, b"mdat");
    let payload_off = f.len();
    f.extend_from_slice(obu);
    patch(&mut f, mdat_start);

    // Backfill the iloc extent offset (absolute file position of the OBU bytes).
    f[iloc_offset_pos..iloc_offset_pos + 4].copy_from_slice(&(payload_off as u32).to_be_bytes());

    f
}

/// Convenience: wrap an `Encoded` result straight into an AVIF-style file.
pub fn to_avif(enc: &AvFrame, fmt: &Av2Format) -> Vec<u8> {
    wrap_av2_image(
        &enc.data,
        enc.width as u32,
        enc.height as u32,
        fmt,
        &enc.color,
    )
}
