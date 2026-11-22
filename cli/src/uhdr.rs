/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! Ultra HDR JPEGs: a primary SDR JPEG followed by a gain-map JPEG, with
//! ISO 21496-1 binary metadata (APP2) or Adobe `hdrgm` XMP (APP1) in the
//! gain-map image. The map is returned as a [`ParsedGainMap`], so conversion
//! to PQ/HLG and re-encoding as an AVIF `tmap`, JXL `jhgm` or HEIC gain map
//! go through the same paths as the other containers.

use crate::gainmap::ParsedGainMap;
use anyhow::{Context, Result, bail, ensure};
use gainforge::{IsoGainMap, MpfInfo};
use image::{DynamicImage, ImageFormat};

const ISO_TAG: &[u8] = b"urn:iso:std:iso:ts:21496:-1\0";
const XMP_TAG: &[u8] = b"http://ns.adobe.com/xap/1.0/\0";
const ICC_TAG: &[u8] = b"ICC_PROFILE\0";
const MPF_TAG: &[u8] = b"MPF\0";
const HDRGM_NS: &str = "http://ns.adobe.com/hdr-gain-map/1.0/";

struct Segment<'a> {
    marker: u8,
    /// Offset of `payload` in the JPEG it was read from.
    offset: usize,
    payload: &'a [u8],
}

/// Marker segments before the first scan, and the offset of the first SOS
/// marker (`None` when the stream ends or is malformed before a scan).
fn header_segments(jpeg: &[u8]) -> (Vec<Segment<'_>>, Option<usize>) {
    let mut segments = Vec::new();
    if !jpeg.starts_with(&[0xFF, 0xD8]) {
        return (segments, None);
    }
    let mut pos = 2;
    loop {
        if jpeg.get(pos) != Some(&0xFF) {
            return (segments, None);
        }
        let marker_start = pos;
        while jpeg.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let Some(&marker) = jpeg.get(pos) else {
            return (segments, None);
        };
        pos += 1;
        match marker {
            0xDA => return (segments, Some(marker_start)),
            0x00 | 0xD8 | 0xD9 => return (segments, None),
            0x01 | 0xD0..=0xD7 => continue,
            _ => {}
        }
        let Some(length) = jpeg
            .get(pos..pos + 2)
            .map(|b| usize::from(u16::from_be_bytes([b[0], b[1]])))
        else {
            return (segments, None);
        };
        let Some(payload) = (length >= 2)
            .then(|| jpeg.get(pos + 2..pos + length))
            .flatten()
        else {
            return (segments, None);
        };
        segments.push(Segment {
            marker,
            offset: pos + 2,
            payload,
        });
        pos += length;
    }
}

/// End of the JPEG that starts at `jpeg[0]` (just past its EOI), walking
/// marker segments and entropy-coded data so progressive and multi-scan
/// streams are handled.
fn jpeg_end(jpeg: &[u8]) -> Option<usize> {
    let (_, sos) = header_segments(jpeg);
    let mut pos = sos?;
    loop {
        // At a marker.
        while jpeg.get(pos) == Some(&0xFF) {
            pos += 1;
        }
        let marker = *jpeg.get(pos)?;
        pos += 1;
        match marker {
            0xD9 => return Some(pos),
            0x01 | 0xD0..=0xD7 => continue,
            0x00 | 0xD8 => return None,
            _ => {}
        }
        let length = usize::from(u16::from_be_bytes(jpeg.get(pos..pos + 2)?.try_into().ok()?));
        pos = pos.checked_add(length)?;
        if marker == 0xDA {
            // Entropy-coded data: stuffed 0xFF00 and RSTn stay inside it.
            loop {
                let ff = pos + jpeg.get(pos..)?.iter().position(|&b| b == 0xFF)?;
                let mut next = ff + 1;
                while jpeg.get(next) == Some(&0xFF) {
                    next += 1;
                }
                match *jpeg.get(next)? {
                    0x00 | 0xD0..=0xD7 => pos = next + 1,
                    _ => {
                        pos = ff;
                        break;
                    }
                }
            }
        }
    }
}

/// Secondary JPEGs listed by the primary's MPF index, then the JPEG that
/// follows the primary's EOI (libultrahdr and gainforge use the latter).
fn secondary_images<'a>(file: &'a [u8], primary: &[Segment<'_>]) -> Vec<&'a [u8]> {
    let mut images: Vec<&[u8]> = Vec::new();
    for segment in primary.iter().filter(|s| s.marker == 0xE2) {
        let Some(tiff) = segment.payload.strip_prefix(MPF_TAG) else {
            continue;
        };
        // MPF offsets are relative to the TIFF header that follows "MPF\0".
        let base = segment.offset + MPF_TAG.len();
        let Ok(info) = MpfInfo::from_bytes(tiff) else {
            continue;
        };
        for entry in info.entries.iter().filter(|e| e.offset != 0) {
            let start = base.saturating_add(entry.offset as usize);
            let end = start.saturating_add(entry.size as usize).min(file.len());
            if let Some(image) = file.get(start..end)
                && image.starts_with(&[0xFF, 0xD8])
            {
                images.push(image);
            }
        }
    }
    if let Some(end) = jpeg_end(file)
        && let Some(start) = file[end..].windows(2).position(|b| b == [0xFF, 0xD8])
    {
        let image = &file[end + start..];
        if !images.iter().any(|i| i.as_ptr() == image.as_ptr()) {
            images.push(image);
        }
    }
    images
}

fn icc_profile(segments: &[Segment<'_>]) -> Option<Vec<u8>> {
    let mut chunks: Vec<(u8, u8, &[u8])> = segments
        .iter()
        .filter(|s| s.marker == 0xE2)
        .filter_map(|s| {
            let rest = s.payload.strip_prefix(ICC_TAG)?;
            Some((*rest.first()?, *rest.get(1)?, rest.get(2..)?))
        })
        .collect();
    let count = chunks.first()?.1;
    if count == 0 || chunks.len() != usize::from(count) || chunks.iter().any(|c| c.1 != count) {
        return None;
    }
    chunks.sort_by_key(|c| c.0);
    if chunks
        .iter()
        .enumerate()
        .any(|(i, c)| usize::from(c.0) != i + 1)
    {
        return None;
    }
    Some(
        chunks
            .into_iter()
            .flat_map(|c| c.2.iter().copied())
            .collect(),
    )
}

/// Whether the primary declares itself an Ultra HDR / ISO gain-map base, so a
/// missing or broken gain map is an error rather than an ordinary JPEG.
fn primary_is_gain_map_base(segments: &[Segment<'_>]) -> bool {
    segments.iter().any(|s| {
        (s.marker == 0xE2 && s.payload.starts_with(ISO_TAG))
            || (s.marker == 0xE1
                && s.payload.starts_with(XMP_TAG)
                && contains(s.payload, HDRGM_NS.as_bytes()))
    })
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Gain-map metadata from a secondary JPEG header: binary ISO 21496-1 wins
/// over XMP, as in libultrahdr.
fn gain_map_metadata(segments: &[Segment<'_>]) -> Result<Option<IsoGainMap>> {
    if let Some(iso) = segments
        .iter()
        .filter(|s| s.marker == 0xE2)
        .find_map(|s| s.payload.strip_prefix(ISO_TAG))
        // A version-only APP2 marks a primary image, not a gain map.
        .filter(|iso| iso.len() > 4)
    {
        let metadata = IsoGainMap::from_metadata(iso)
            .map_err(|e| anyhow::anyhow!("invalid ISO 21496-1 gain-map metadata: {e:?}"))?;
        return Ok(Some(metadata));
    }
    for xmp in segments
        .iter()
        .filter(|s| s.marker == 0xE1)
        .filter_map(|s| s.payload.strip_prefix(XMP_TAG))
    {
        if let Some(metadata) = hdrgm_metadata(xmp)? {
            return Ok(Some(metadata));
        }
    }
    Ok(None)
}

/// Adobe gain map (`hdrgm`) XMP, as written by Ultra HDR v1 and Lightroom.
/// `None` when the packet has no `hdrgm` namespace; gainforge (>= 0.5.2)
/// resolves namespaces, attribute/element/`rdf:Seq` forms and spec defaults.
fn hdrgm_metadata(xmp: &[u8]) -> Result<Option<IsoGainMap>> {
    if !contains(xmp, HDRGM_NS.as_bytes()) {
        return Ok(None);
    }
    IsoGainMap::from_xml_data(xmp)
        .map(Some)
        .map_err(|e| anyhow::anyhow!("invalid gain-map XMP: {e:?}"))
}

fn decode_gain_map_image(jpeg: &[u8]) -> Result<DynamicImage> {
    let image = image::load_from_memory_with_format(jpeg, ImageFormat::Jpeg)
        .context("cannot decode the gain-map JPEG")?;
    ensure!(
        image.width() > 0 && image.height() > 0,
        "empty Ultra HDR gain map"
    );
    // Store the encoded (gamma-applied) samples at 16 bits like the other
    // containers; the gain math normalizes them by u16::MAX.
    Ok(match image {
        DynamicImage::ImageLuma8(_) | DynamicImage::ImageLuma16(_) => {
            DynamicImage::ImageLuma16(image.into_luma16())
        }
        _ => DynamicImage::ImageRgb16(image.into_rgb16()),
    })
}

/// The gain map of an Ultra HDR JPEG, `None` for an ordinary JPEG.
pub(crate) fn parse_uhdr_gain_map(file: &[u8]) -> Result<Option<ParsedGainMap>> {
    let (primary, _) = header_segments(file);
    let required = primary_is_gain_map_base(&primary);
    for image in secondary_images(file, &primary) {
        let (segments, _) = header_segments(image);
        let Some(metadata) = gain_map_metadata(&segments)? else {
            continue;
        };
        let decoded = decode_gain_map_image(image)?;
        // The gain map's ICC profile describes the alternate rendition; none
        // means it inherits the base's (libultrahdr applyGainMap).
        let alternate_icc = icc_profile(&segments);
        return Ok(Some(ParsedGainMap {
            image: decoded,
            metadata,
            alternate_cicp: None,
            alternate_icc,
        }));
    }
    if required {
        bail!("Ultra HDR gain map or its metadata is missing");
    }
    Ok(None)
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

const MPF_PAYLOAD_LEN: usize = 4 + 82;
/// Offset of the MP entries from the MPF TIFF header (8-byte header, 2-byte
/// count, three 12-byte tags, 4-byte next-IFD offset).
const MPF_ENTRIES_OFFSET: u32 = 50;

/// Big-endian MPF APP2 payload for a primary image and one gain map
/// (CIPA DC-007; the same layout libultrahdr writes). Offsets are relative to
/// the TIFF header following `MPF\0`.
fn mpf_payload(primary_size: u32, secondary_size: u32, secondary_offset: u32) -> Vec<u8> {
    const UNDEFINED: u16 = 7;
    const LONG: u16 = 4;
    let mut p = Vec::with_capacity(MPF_PAYLOAD_LEN);
    p.extend_from_slice(MPF_TAG);
    p.extend_from_slice(b"MM\0\x2A");
    p.extend_from_slice(&8u32.to_be_bytes());
    p.extend_from_slice(&3u16.to_be_bytes());
    // MPFVersion "0100".
    p.extend_from_slice(&0xB000u16.to_be_bytes());
    p.extend_from_slice(&UNDEFINED.to_be_bytes());
    p.extend_from_slice(&4u32.to_be_bytes());
    p.extend_from_slice(b"0100");
    // NumberOfImages.
    p.extend_from_slice(&0xB001u16.to_be_bytes());
    p.extend_from_slice(&LONG.to_be_bytes());
    p.extend_from_slice(&1u32.to_be_bytes());
    p.extend_from_slice(&2u32.to_be_bytes());
    // MPEntry: two 16-byte entries.
    p.extend_from_slice(&0xB002u16.to_be_bytes());
    p.extend_from_slice(&UNDEFINED.to_be_bytes());
    p.extend_from_slice(&32u32.to_be_bytes());
    p.extend_from_slice(&MPF_ENTRIES_OFFSET.to_be_bytes());
    // No next IFD.
    p.extend_from_slice(&0u32.to_be_bytes());
    // Primary: JPEG, baseline MP primary image, offset 0.
    p.extend_from_slice(&0x0003_0000u32.to_be_bytes());
    p.extend_from_slice(&primary_size.to_be_bytes());
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&[0; 4]);
    // Gain map: JPEG, undefined type.
    p.extend_from_slice(&0u32.to_be_bytes());
    p.extend_from_slice(&secondary_size.to_be_bytes());
    p.extend_from_slice(&secondary_offset.to_be_bytes());
    p.extend_from_slice(&[0; 4]);
    debug_assert_eq!(p.len(), MPF_PAYLOAD_LEN);
    p
}

fn xmp_segment(xml: &str) -> Vec<u8> {
    [XMP_TAG, xml.as_bytes()].concat()
}

fn primary_xmp(gain_map_len: usize) -> String {
    format!(
        concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="maroontree">"#,
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmlns:Container="http://ns.google.com/photos/1.0/container/" "#,
            r#"xmlns:Item="http://ns.google.com/photos/1.0/container/item/" "#,
            r#"xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0">"#,
            r#"<Container:Directory><rdf:Seq>"#,
            r#"<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="Primary" Item:Mime="image/jpeg"/></rdf:li>"#,
            r#"<rdf:li rdf:parseType="Resource"><Container:Item Item:Semantic="GainMap" Item:Mime="image/jpeg" Item:Length="{}"/></rdf:li>"#,
            r#"</rdf:Seq></Container:Directory></rdf:Description></rdf:RDF></x:xmpmeta>"#
        ),
        gain_map_len
    )
}

/// Adobe `hdrgm` description of `m`, for readers without ISO 21496-1
/// support. Per-channel values become `rdf:Seq` elements.
fn gain_map_xmp(m: &IsoGainMap) -> String {
    let mut attributes = String::new();
    let mut elements = String::new();
    let mut put = |name: &str, v: [f64; 3]| {
        if v[0] == v[1] && v[1] == v[2] {
            attributes.push_str(&format!(r#" hdrgm:{name}="{}""#, v[0]));
        } else {
            elements.push_str(&format!(
                "<hdrgm:{name}><rdf:Seq><rdf:li>{}</rdf:li><rdf:li>{}</rdf:li><rdf:li>{}</rdf:li></rdf:Seq></hdrgm:{name}>",
                v[0], v[1], v[2]
            ));
        }
    };
    put("GainMapMin", m.map_min());
    put("GainMapMax", m.map_max());
    put("Gamma", m.gain_map_gamma());
    put("OffsetSDR", m.map_base_offset());
    put("OffsetHDR", m.map_alternate_offset());
    put("HDRCapacityMin", [m.base_hdr_headroom(); 3]);
    put("HDRCapacityMax", [m.alternate_hdr_headroom(); 3]);
    format!(
        concat!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="maroontree">"#,
            r#"<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">"#,
            r#"<rdf:Description rdf:about="" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" "#,
            r#"hdrgm:Version="1.0"{} hdrgm:BaseRenditionIsHDR="False">{}</rdf:Description>"#,
            r#"</rdf:RDF></x:xmpmeta>"#
        ),
        attributes, elements
    )
}

/// ISO 21496-1 binary metadata in the final syntax, rationals copied exactly.
/// Not jixel's writer: it emits a common-denominator flag (bit 3) that the
/// final standard does not define, and Apple ImageIO then ignores the map.
fn iso_metadata(m: &IsoGainMap) -> Result<Vec<u8>> {
    ensure!(
        !m.backward_direction,
        "backward gain maps cannot be written"
    );
    maroontree::IsoGainMap {
        gain_map_min_n: m.gain_map_min_n,
        gain_map_min_d: m.gain_map_min_d,
        gain_map_max_n: m.gain_map_max_n,
        gain_map_max_d: m.gain_map_max_d,
        gain_map_gamma_n: m.gain_map_gamma_n,
        gain_map_gamma_d: m.gain_map_gamma_d,
        base_offset_n: m.base_offset_n,
        base_offset_d: m.base_offset_d,
        alternate_offset_n: m.alternate_offset_n,
        alternate_offset_d: m.alternate_offset_d,
        base_hdr_headroom_n: m.base_hdr_headroom_n,
        base_hdr_headroom_d: m.base_hdr_headroom_d,
        alternate_hdr_headroom_n: m.alternate_hdr_headroom_n,
        alternate_hdr_headroom_d: m.alternate_hdr_headroom_d,
        use_base_color_space: m.use_base_color_space,
    }
    .to_metadata()
    .map_err(|e| anyhow::anyhow!("cannot serialize ISO 21496-1 metadata: {e}"))
}

/// Whether `m` can be written as Ultra HDR: an SDR base with a brighter
/// alternate, as both the ISO block and the `hdrgm` XMP require.
pub(crate) fn can_write(m: &IsoGainMap) -> bool {
    !m.backward_direction && m.alternate_hdr_headroom() > m.base_hdr_headroom()
}

/// Ultra HDR JPEG: the SDR base `img` with `gain_map` appended as a second
/// JPEG, linked by MPF and described by ISO 21496-1 APP2 and `hdrgm` XMP
/// (libultrahdr's layout).
pub(crate) fn encode_uhdr_jpeg(
    img: &DynamicImage,
    args: &crate::Args,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    gain_map: &ParsedGainMap,
) -> Result<Vec<u8>> {
    use crate::jpeg::{JpegOptions, encode_jpeg_with};
    let metadata = &gain_map.metadata;
    ensure!(
        can_write(metadata),
        "Ultra HDR needs an SDR base with a brighter alternate rendition"
    );
    let options = JpegOptions::from_args(args).map_err(anyhow::Error::msg)?;
    if options.arithmetic {
        // Valid JPEG (libjpeg-turbo decodes both images), but Apple ImageIO
        // then ignores the gain map and zune-jpeg cannot read it at all.
        eprintln!(
            "warning: arithmetic-coded Ultra HDR: most viewers ignore its gain map; use Huffman coding for HDR display"
        );
    }

    let mut map_options = options.clone();
    map_options.quality = options.gain_map_quality.unwrap_or(options.quality);
    map_options.sampling = jhm17::SamplingFactor::R_4_4_4;
    let map = match &gain_map.image {
        DynamicImage::ImageLuma16(_) | DynamicImage::ImageLuma8(_) => {
            DynamicImage::ImageLuma8(gain_map.image.to_luma8())
        }
        _ => DynamicImage::ImageRgb8(gain_map.image.to_rgb8()),
    };
    let iso = iso_metadata(metadata)?;
    let map_xmp = gain_map_xmp(metadata);
    let map_jpeg = encode_jpeg_with(&map, &map_options, false, |encoder| {
        encoder.add_app_segment(1, xmp_segment(&map_xmp))?;
        encoder.add_app_segment(2, [ISO_TAG, &iso].concat())?;
        if let Some(icc) = &gain_map.alternate_icc {
            encoder.add_icc_profile(icc)?;
        }
        Ok(())
    })?;

    let base_xmp = primary_xmp(map_jpeg.len());
    let mut primary = encode_jpeg_with(img, &options, args.verbose, |encoder| {
        if let Some(exif) = exif {
            encoder.add_exif_metadata(exif.strip_prefix(b"Exif\0\0").unwrap_or(exif))?;
        }
        encoder.add_app_segment(1, xmp_segment(&base_xmp))?;
        // minimum_version 0, writer_version 0: the full metadata lives in
        // the gain map's APP2.
        encoder.add_app_segment(2, [ISO_TAG, &[0u8; 4][..]].concat())?;
        if let Some(icc) = icc {
            encoder.add_icc_profile(icc)?;
        }
        encoder.add_app_segment(2, mpf_payload(0, 0, 0))?;
        Ok(())
    })?;

    // Patch the MPF sizes and offset now that the primary's length is known.
    let (segments, _) = header_segments(&primary);
    let mpf = segments
        .iter()
        .find(|s| s.marker == 0xE2 && s.payload.starts_with(MPF_TAG))
        .map(|s| s.offset)
        .context("MPF segment missing from the encoded primary")?;
    let tiff = mpf + MPF_TAG.len();
    let primary_len = u32::try_from(primary.len()).context("primary JPEG exceeds 4 GiB")?;
    let map_len = u32::try_from(map_jpeg.len()).context("gain-map JPEG exceeds 4 GiB")?;
    let patched = mpf_payload(primary_len, map_len, primary_len - tiff as u32);
    primary[mpf..mpf + MPF_PAYLOAD_LEN].copy_from_slice(&patched);
    primary.extend_from_slice(&map_jpeg);
    Ok(primary)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdrgm_attributes_and_sequences() {
        let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description xmlns:g="http://ns.adobe.com/hdr-gain-map/1.0/" g:Version="1.0" g:GainMapMin="0" g:HDRCapacityMax="2.3" g:OffsetSDR="0" g:OffsetHDR="0">
<g:GainMapMax><rdf:Seq><rdf:li>2.3</rdf:li><rdf:li>2.0</rdf:li><rdf:li>1.5</rdf:li></rdf:Seq></g:GainMapMax>
</rdf:Description></rdf:RDF></x:xmpmeta>"#;
        let m = hdrgm_metadata(xmp).unwrap().unwrap();
        let max = m.map_max();
        assert!((max[0] - 2.3).abs() < 1e-4 && (max[2] - 1.5).abs() < 1e-4);
        assert!((m.alternate_hdr_headroom() - 2.3).abs() < 1e-4);
        assert_eq!(m.base_hdr_headroom(), 0.0);
        assert_eq!(m.map_base_offset(), [0.0; 3]);
        assert_eq!(m.gain_map_gamma(), [1.0; 3]);
        assert!(!m.backward_direction);
    }

    #[test]
    fn unrelated_xmp_is_not_a_gain_map() {
        let xmp = br#"<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><rdf:Description xmlns:dc="http://purl.org/dc/elements/1.1/" dc:format="image/jpeg"/></rdf:RDF></x:xmpmeta>"#;
        assert!(hdrgm_metadata(xmp).unwrap().is_none());
    }

    #[test]
    fn hdr_base_rendition_is_rejected() {
        let xmp = br#"<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/" hdrgm:Version="1.0" hdrgm:GainMapMax="2" hdrgm:HDRCapacityMax="2" hdrgm:BaseRenditionIsHDR="True"/>"#;
        assert!(hdrgm_metadata(xmp).is_err());
    }

    #[test]
    fn uhdr_jpeg_round_trips_metadata_and_mpf() {
        use image::GenericImageView;
        let metadata = IsoGainMap {
            gain_map_min_n: [-1, 0, 0],
            gain_map_min_d: [4, 1, 1],
            gain_map_max_n: [3, 5, 2],
            gain_map_max_d: [1, 2, 1],
            gain_map_gamma_n: [1; 3],
            gain_map_gamma_d: [1; 3],
            base_offset_n: [1; 3],
            base_offset_d: [64; 3],
            alternate_offset_n: [1; 3],
            alternate_offset_d: [64; 3],
            base_hdr_headroom_n: 0,
            base_hdr_headroom_d: 1,
            alternate_hdr_headroom_n: 3,
            alternate_hdr_headroom_d: 1,
            backward_direction: false,
            use_base_color_space: true,
        };
        let gain_map = ParsedGainMap {
            image: DynamicImage::ImageRgb16(image::ImageBuffer::from_fn(16, 8, |x, y| {
                image::Rgb([(x * 4000) as u16, (y * 8000) as u16, 30000])
            })),
            metadata,
            alternate_cicp: None,
            alternate_icc: None,
        };
        let base = DynamicImage::ImageRgb8(image::ImageBuffer::from_fn(32, 16, |x, y| {
            image::Rgb([(x * 8) as u8, (y * 16) as u8, 128])
        }));
        let args = crate::args::try_parse(&["cli", "-e", "jpeg", "in.png"]).unwrap();
        let file = encode_uhdr_jpeg(&base, &args, None, None, &gain_map).unwrap();

        let (primary, _) = header_segments(&file);
        assert!(primary_is_gain_map_base(&primary));
        // MPF and the EOI fallback agree on where the gain map starts.
        let images = secondary_images(&file, &primary);
        assert_eq!(images.len(), 1);
        let decoded = parse_uhdr_gain_map(&file).unwrap().unwrap();
        assert_eq!(format!("{:?}", decoded.metadata), format!("{metadata:?}"));
        assert_eq!(decoded.image.dimensions(), (16, 8));
        let (map_segments, _) = header_segments(images[0]);
        let xmp = map_segments
            .iter()
            .find_map(|s| s.payload.strip_prefix(XMP_TAG))
            .unwrap();
        let from_xmp = hdrgm_metadata(xmp).unwrap().unwrap();
        for (a, b) in from_xmp.map_max().iter().zip(metadata.map_max()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert!((from_xmp.map_min()[0] + 0.25).abs() < 1e-6);
        assert_eq!(
            image::load_from_memory(&file).unwrap().dimensions(),
            (32, 16)
        );
    }

    #[test]
    fn iso_metadata_uses_final_syntax_flags() {
        let mut m = IsoGainMap {
            gain_map_min_d: [1_000_000; 3],
            gain_map_max_n: [2_000_000; 3],
            gain_map_max_d: [1_000_000; 3],
            gain_map_gamma_n: [1_000_000; 3],
            gain_map_gamma_d: [1_000_000; 3],
            base_offset_d: [1_000_000; 3],
            alternate_offset_d: [1_000_000; 3],
            base_hdr_headroom_d: 1_000_000,
            alternate_hdr_headroom_n: 2_000_000,
            alternate_hdr_headroom_d: 1_000_000,
            use_base_color_space: true,
            ..IsoGainMap::default()
        };
        let blob = iso_metadata(&m).unwrap();
        // Versions, then only multichannel/use_base_colour_space flag bits.
        assert_eq!(&blob[..5], &[0, 0, 0, 0, 0x40]);
        assert_eq!(blob.len(), 5 + 16 + 40);
        m.backward_direction = true;
        assert!(iso_metadata(&m).is_err());
    }
}
