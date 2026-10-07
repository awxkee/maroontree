/*
 * Copyright (c) Radzivon Bartoshyk 9/2026. All rights reserved.
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

//! Import legacy Apple HEIF gain maps as ISO 21496-1 data. See
//! `cli/APPLE_GAIN_MAP.md` for the format references and conversion contract.

use crate::orientation::apply_orientation;
use anyhow::{Context, Result, bail, ensure};
use gainforge::{IsoGainMap, TransferFunction};
use hpvcd::GainMapFrame;
use image::{DynamicImage, ImageBuffer, Luma};
use quick_xml::events::Event;
use quick_xml::name::{Namespace, ResolveResult};
use quick_xml::reader::NsReader;

const APPLE_NS: &str = "http://ns.apple.com/HDRGainMap/1.0/";
const APPLE_MAKER_HEADER: &[u8] = b"Apple iOS\0\0\x01MM";

/// Oriented, normalized logarithmic gain samples, without a color transfer
/// function. Keep the converted samples at 16-bit precision for JXL encoding.
#[derive(Debug)]
pub(crate) struct ParsedGainMap {
    pub image: ImageBuffer<Luma<u16>, Vec<u16>>,
    pub metadata: IsoGainMap,
}

pub(crate) fn parse_apple_gain_map(
    gain: &GainMapFrame,
    base_exif: Option<&[u8]>,
) -> Result<ParsedGainMap> {
    let metadata = apple_metadata(
        gain.metadata
            .as_deref()
            .context("missing Apple gain-map XMP")?,
        base_exif,
    )?;
    ensure!(
        gain.channels() == 1,
        "unsupported multichannel Apple gain map"
    );
    let width = u32::try_from(gain.width())?;
    let height = u32::try_from(gain.height())?;
    ensure!(width > 0 && height > 0, "empty Apple gain map");
    let depth = gain.bit_depth.bits();
    ensure!(
        matches!(depth, 8 | 10 | 12),
        "unsupported Apple gain-map bit depth"
    );
    let full_range = gain.color.cicp.is_none_or(|cicp| cicp.full_range);
    let max_sample = (1u32 << depth) - 1;
    let (black, scale) = if full_range {
        (0.0, max_sample as f32)
    } else {
        (
            (16u32 << (depth - 8)) as f32,
            (219u32 << (depth - 8)) as f32,
        )
    };
    let stops = metadata.alternate_hdr_headroom();
    let headroom = stops.exp2();
    let convert = |sample: u32| {
        let encoded = ((sample as f32 - black) / scale).clamp(0.0, 1.0);
        let linear = TransferFunction::Rec709.linearize(encoded, 1.0) as f64;
        // Apple: multiplier = 1 + (H - 1) * linear_gain.
        // ISO (min=0, max=log2(H), gamma=1): multiplier = 2^(sample * max).
        let normalized = ((headroom - 1.0) * linear).ln_1p() / std::f64::consts::LN_2 / stops;
        (normalized.clamp(0.0, 1.0) * u16::MAX as f64).round() as u16
    };
    let lookup: Vec<u16> = (0..=max_sample).map(convert).collect();
    let samples: Vec<u16> = if let Some(planes) = gain.planes.as_u8() {
        ensure!(depth == 8, "Apple gain-map sample depth mismatch");
        planes
            .y
            .rows()
            .flatten()
            .map(|&v| lookup[v as usize])
            .collect()
    } else if let Some(planes) = gain.planes.as_u16() {
        ensure!(depth > 8, "Apple gain-map sample depth mismatch");
        ensure!(
            planes.y.rows().flatten().all(|&v| v as u32 <= max_sample),
            "Apple gain-map sample exceeds its bit depth"
        );
        planes
            .y
            .rows()
            .flatten()
            .map(|&v| lookup[v as usize])
            .collect()
    } else {
        bail!("unsupported Apple gain-map sample storage");
    };
    let image = ImageBuffer::from_raw(width, height, samples)
        .context("Apple gain-map dimensions do not match its samples")?;
    let image = apply_orientation(DynamicImage::ImageLuma16(image), gain.orientation).into_luma16();
    Ok(ParsedGainMap { image, metadata })
}

/// AVIF gain map item for `gain_map`. libavif caps gain maps at 12 bits, so
/// the 16-bit normalized gains are rounded to a 10-bit monochrome map (the
/// Apple source maps are 8-bit, so nothing is lost). The rationals are copied
/// exactly; an AVIF `tmap` cannot signal the Ultra HDR backward-direction flag.
pub(crate) fn avif_gain_map(gain_map: &ParsedGainMap) -> Result<maroontree::GainMap> {
    let m = gain_map.metadata;
    ensure!(
        !m.backward_direction,
        "backward-direction gain maps are not representable in AVIF"
    );
    let metadata = maroontree::IsoGainMap {
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
    };
    let samples: Vec<u16> = gain_map
        .image
        .as_raw()
        .iter()
        .map(|&v| ((v as u32 * 1023 + 32767) / 65535) as u16)
        .collect();
    let image = maroontree::PlanarImage::from_luma(
        gain_map.image.width() as usize,
        gain_map.image.height() as usize,
        maroontree::BitDepth::Ten,
        &samples,
    )?;
    Ok(maroontree::GainMap::gray16(image, metadata))
}

/// HEIC gain map for `gain_map`; hpvca writes it both as an Apple HDR gain
/// map and as an ISO 21496-1 `tmap`. hpvca takes Apple-encoded samples, so
/// each ISO gain `2^L` is mapped back onto Apple's curve with headroom
/// `2^max`: Rec.709-coded `(2^L - 1) / (headroom - 1)`. For an 8-bit Apple
/// source this exactly inverts [`parse_apple_gain_map`]. Apple's form has no
/// offsets (they are dropped) and cannot darken (gains below 1 clamp to 1).
pub(crate) fn hevc_gain_map(gain_map: &ParsedGainMap) -> Result<hpvca::GainMap> {
    let m = gain_map.metadata;
    ensure!(
        !m.backward_direction,
        "backward-direction gain maps are not representable in HEIC"
    );
    let (min, max, gamma) = (m.map_min()[0], m.map_max()[0], m.gain_map_gamma()[0]);
    ensure!(
        min.is_finite() && max.is_finite() && max > 0.0 && gamma > 0.0,
        "gain map has no HDR headroom"
    );
    let headroom = max.exp2();
    let lookup: Vec<u8> = (0..=u16::MAX)
        .map(|v| {
            let g = (v as f64 / u16::MAX as f64).powf(1.0 / gamma);
            let multiplier = (min + (max - min) * g).exp2();
            let linear = ((multiplier - 1.0) / (headroom - 1.0)).clamp(0.0, 1.0);
            ((TransferFunction::Rec709.gamma(linear as f32) * 255.0) + 0.5)
                .clamp(0.0, 255.0) as u8
        })
        .collect();
    let samples = gain_map
        .image
        .as_raw()
        .iter()
        .map(|&v| lookup[v as usize])
        .collect();
    Ok(hpvca::GainMap::gray8(
        samples,
        gain_map.image.width(),
        gain_map.image.height(),
        headroom as f32,
    ))
}

fn apple_metadata(xmp: &[u8], base_exif: Option<&[u8]>) -> Result<IsoGainMap> {
    let headroom = apple_xmp_headroom(xmp)?;
    let stops = match headroom {
        Some(h) if h > 1.0 => h.log2(),
        // libavif also falls back to MakerNotes for XMP headroom equal to 1.
        _ => apple_exif_stops(base_exif.context("missing Apple headroom in XMP and EXIF")?)?,
    };
    ensure!(
        stops.is_finite() && stops > 0.0,
        "Apple headroom must exceed SDR white"
    );
    // Microstop precision fits every finite f64 headroom in an ISO signed
    // numerator. Reject a rounded zero instead of creating degenerate metadata.
    const DENOMINATOR: u32 = 1_000_000;
    let numerator = (stops * DENOMINATOR as f64).round();
    ensure!(
        (1.0..=i32::MAX as f64).contains(&numerator)
            && (numerator / DENOMINATOR as f64).exp2().is_finite(),
        "Apple headroom cannot be represented as ISO metadata"
    );
    Ok(IsoGainMap {
        gain_map_min_n: [0; 3],
        gain_map_min_d: [1; 3],
        gain_map_max_n: [numerator as i32; 3],
        gain_map_max_d: [DENOMINATOR; 3],
        gain_map_gamma_n: [1; 3],
        gain_map_gamma_d: [1; 3],
        base_offset_n: [0; 3],
        base_offset_d: [1; 3],
        alternate_offset_n: [0; 3],
        alternate_offset_d: [1; 3],
        base_hdr_headroom_n: 0,
        base_hdr_headroom_d: 1,
        alternate_hdr_headroom_n: numerator as u32,
        alternate_hdr_headroom_d: DENOMINATOR,
        backward_direction: false,
        use_base_color_space: true,
    })
}

/// Resolve namespace URIs, not a particular producer's choice of XML prefix.
/// XMP permits scalar properties as either attributes or child elements.
fn apple_xmp_headroom(xmp: &[u8]) -> Result<Option<f64>> {
    let mut reader = NsReader::from_reader(xmp);
    let mut version = None;
    let mut headroom = None;
    let mut depth = 0usize;
    let mut roots = 0;
    loop {
        let event = reader.read_event().context("invalid Apple gain-map XMP")?;
        match event {
            Event::Start(ref element) | Event::Empty(ref element) => {
                if depth == 0 {
                    roots += 1;
                }
                for attr in element.attributes() {
                    let attr = attr?;
                    let (ns, name) = reader.resolver().resolve_attribute(attr.key);
                    if ns == ResolveResult::Bound(Namespace(APPLE_NS)) {
                        set_xmp_property(
                            name.as_ref(),
                            attr.normalized_value(quick_xml::XmlVersion::Implicit1_0)?
                                .into_owned(),
                            &mut version,
                            &mut headroom,
                        )?;
                    }
                }
                let (ns, name) = reader.resolver().resolve_element(element.name());
                let property = ns == ResolveResult::Bound(Namespace(APPLE_NS))
                    && matches!(name.as_ref(), "HDRGainMapVersion" | "HDRGainMapHeadroom");
                if property {
                    let name = name.as_ref().to_owned();
                    let value = if matches!(event, Event::Start(_)) {
                        quick_xml::escape::unescape(&reader.read_text(element.name())?)?
                            .into_owned()
                    } else {
                        String::new()
                    };
                    set_xmp_property(&name, value, &mut version, &mut headroom)?;
                } else if matches!(event, Event::Start(_)) {
                    depth += 1;
                }
            }
            Event::End(_) => depth = depth.checked_sub(1).context("invalid XMP nesting")?,
            Event::DocType(_) => bail!("unsupported XMP document type"),
            Event::Eof => break,
            _ => {}
        }
    }
    ensure!(depth == 0 && roots == 1, "incomplete Apple gain-map XMP");
    ensure!(version.is_some(), "missing Apple HDRGainMapVersion");
    headroom
        .map(|h| {
            let value = h
                .trim()
                .parse::<f64>()
                .context("invalid HDRGainMapHeadroom")?;
            ensure!(
                value.is_finite() && value >= 1.0,
                "invalid HDRGainMapHeadroom"
            );
            Ok(value)
        })
        .transpose()
}

fn set_xmp_property(
    name: &str,
    value: String,
    version: &mut Option<String>,
    headroom: &mut Option<String>,
) -> Result<()> {
    let slot = match name {
        "HDRGainMapVersion" => version,
        "HDRGainMapHeadroom" => headroom,
        _ => return Ok(()),
    };
    ensure!(slot.is_none(), "duplicate Apple gain-map XMP property");
    *slot = Some(value);
    Ok(())
}

fn apple_exif_stops(exif: &[u8]) -> Result<f64> {
    // hpvcd strips the HEIF Exif item's four-byte offset. Apple payloads can
    // still include the JPEG-style Exif identifier before their TIFF header.
    let tiff = exif.strip_prefix(b"Exif\0\0").unwrap_or(exif);
    let parsed = exif::Reader::new()
        .read_raw(tiff.to_vec())
        .context("invalid base EXIF")?;
    let field = parsed
        .get_field(exif::Tag::MakerNote, exif::In::PRIMARY)
        .context("missing Apple MakerNote")?;
    let exif::Value::Undefined(maker, _) = &field.value else {
        bail!("invalid Apple MakerNote type");
    };
    apple_maker_stops(maker)
}

fn apple_maker_stops(maker: &[u8]) -> Result<f64> {
    ensure!(
        maker.starts_with(APPLE_MAKER_HEADER),
        "unrecognized Apple MakerNote header"
    );
    let u16_at = |offset: usize| -> Result<u16> {
        Ok(u16::from_be_bytes(
            *maker
                .get(offset..)
                .and_then(<[u8]>::first_chunk)
                .context("truncated Apple MakerNote")?,
        ))
    };
    let u32_at = |offset: usize| -> Result<u32> {
        Ok(u32::from_be_bytes(
            *maker
                .get(offset..)
                .and_then(<[u8]>::first_chunk)
                .context("truncated Apple MakerNote")?,
        ))
    };
    let count = u16_at(APPLE_MAKER_HEADER.len())? as usize;
    let start = APPLE_MAKER_HEADER.len() + 2;
    ensure!(
        maker.len() >= start + count * 12,
        "truncated Apple MakerNote IFD"
    );
    let mut values = [None, None];
    for entry in (start..start + count * 12).step_by(12) {
        let index = match u16_at(entry)? {
            33 => 0,
            48 => 1,
            _ => continue,
        };
        ensure!(values[index].is_none(), "duplicate Apple headroom tag");
        ensure!(
            u32_at(entry + 4)? == 1,
            "invalid Apple headroom component count"
        );
        let value = match u16_at(entry + 2)? {
            // Stored MakerNotes use SRATIONAL, despite ImageIO exposing floats.
            10 => {
                let offset = usize::try_from(u32_at(entry + 8)?)?;
                ensure!(
                    offset <= maker.len().saturating_sub(8),
                    "invalid Apple headroom offset"
                );
                let n = u32_at(offset)? as i32;
                let d = u32_at(offset + 4)? as i32;
                ensure!(d != 0, "zero Apple headroom denominator");
                n as f64 / d as f64
            }
            11 => f32::from_bits(u32_at(entry + 8)?) as f64,
            _ => bail!("unsupported Apple headroom tag type"),
        };
        ensure!(value.is_finite(), "non-finite Apple headroom tag");
        values[index] = Some(value);
    }
    ensure!(
        values.iter().any(Option::is_some),
        "missing Apple headroom tags 33 and 48"
    );
    // Like libavif/Skia, tolerate older photos containing only one of the tags.
    let maker33 = values[0].unwrap_or(0.0);
    let maker48 = values[1].unwrap_or(0.0);
    let stops = match (maker33 < 1.0, maker48 <= 0.01) {
        (true, true) => 1.8 - 20.0 * maker48,
        (true, false) => 1.601 - 0.101 * maker48,
        (false, true) => 3.0 - 70.0 * maker48,
        (false, false) => 2.303 - 0.303 * maker48,
    };
    Ok(stops.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xmp(headroom: Option<&str>) -> Vec<u8> {
        let headroom = headroom
            .map(|h| format!("<a:HDRGainMapHeadroom>{h}</a:HDRGainMapHeadroom>"))
            .unwrap_or_default();
        format!(
            r#"<x:xmpmeta xmlns:x="adobe:ns:meta/">
            <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description xmlns:a="http://ns.apple.com/HDRGainMap/1.0/">
            <a:HDRGainMapVersion>65536</a:HDRGainMapVersion>{headroom}
            </rdf:Description></rdf:RDF></x:xmpmeta>"#
        )
        .into_bytes()
    }

    fn maker(tags: &[(u16, i32, i32)]) -> Vec<u8> {
        let mut data = APPLE_MAKER_HEADER.to_vec();
        data.extend_from_slice(&(tags.len() as u16).to_be_bytes());
        let offset = 16 + tags.len() * 12 + 4;
        for (index, &(tag, _, _)) in tags.iter().enumerate() {
            data.extend_from_slice(&tag.to_be_bytes());
            data.extend_from_slice(&10u16.to_be_bytes());
            data.extend_from_slice(&1u32.to_be_bytes());
            data.extend_from_slice(&((offset + index * 8) as u32).to_be_bytes());
        }
        data.extend_from_slice(&0u32.to_be_bytes());
        for &(_, n, d) in tags {
            data.extend_from_slice(&n.to_be_bytes());
            data.extend_from_slice(&d.to_be_bytes());
        }
        data
    }

    fn exif(maker: &[u8], little: bool) -> Vec<u8> {
        let u16_bytes = |n: u16| {
            if little {
                n.to_le_bytes()
            } else {
                n.to_be_bytes()
            }
        };
        let u32_bytes = |n: u32| {
            if little {
                n.to_le_bytes()
            } else {
                n.to_be_bytes()
            }
        };
        let mut data = if little { b"II" } else { b"MM" }.to_vec();
        data.extend_from_slice(&u16_bytes(42));
        data.extend_from_slice(&u32_bytes(8));
        // IFD0 -> Exif IFD -> opaque MakerNote. MakerNote offsets are local,
        // and remain big endian even when the enclosing TIFF is little endian.
        for (tag, kind, count, offset) in [(0x8769, 4, 1, 26), (0x927c, 7, maker.len() as u32, 44)]
        {
            data.extend_from_slice(&u16_bytes(1));
            data.extend_from_slice(&u16_bytes(tag));
            data.extend_from_slice(&u16_bytes(kind));
            data.extend_from_slice(&u32_bytes(count));
            data.extend_from_slice(&u32_bytes(offset));
            data.extend_from_slice(&u32_bytes(0));
        }
        data.extend_from_slice(maker);
        data
    }

    #[test]
    fn xmp_elements_and_attributes_use_namespace_uris() {
        assert_eq!(apple_xmp_headroom(&xmp(Some(" 4.0 "))).unwrap(), Some(4.0));
        let attrs = br#"<rdf:Description xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
            xmlns:renamed="http://ns.apple.com/HDRGainMap/1.0/"
            renamed:HDRGainMapVersion="65536" renamed:HDRGainMapHeadroom="&#52;"/>"#;
        assert_eq!(apple_xmp_headroom(attrs).unwrap(), Some(4.0));
        let wrong_ns = String::from_utf8(xmp(Some("4"))).unwrap().replace(
            "http://ns.apple.com/HDRGainMap/1.0/",
            "http://example.com/unrelated/",
        );
        assert!(apple_xmp_headroom(wrong_ns.as_bytes()).is_err());
    }

    #[test]
    fn iso_metadata_has_valid_fractions_and_xmp_takes_precedence() {
        let meta = apple_metadata(&xmp(Some("4")), Some(b"invalid EXIF is not needed")).unwrap();
        assert_eq!(meta.gain_map_min_n, [0; 3]);
        assert_eq!(meta.gain_map_min_d, [1; 3]);
        assert_eq!(meta.gain_map_max_n, [2_000_000; 3]);
        assert_eq!(meta.gain_map_max_d, [1_000_000; 3]);
        assert_eq!(meta.gain_map_gamma(), [1.0; 3]);
        assert_eq!(meta.base_offset_n, [0; 3]);
        assert_eq!(meta.alternate_offset_n, [0; 3]);
        assert_eq!(meta.base_offset_d, [1; 3]);
        assert_eq!(meta.alternate_offset_d, [1; 3]);
        assert_eq!(meta.base_hdr_headroom_n, 0);
        assert_eq!(meta.base_hdr_headroom_d, 1);
        assert!(meta.use_base_color_space);
        assert!(!meta.backward_direction);
    }

    #[test]
    fn old_exif_headroom_supports_both_tiff_byte_orders_and_exif_prefix() {
        for little in [false, true] {
            let raw = exif(&maker(&[(33, 1, 1), (48, 1, 100)]), little);
            for prefix in [&b""[..], &b"Exif\0\0"[..]] {
                let data = [prefix, &raw].concat();
                assert!((apple_exif_stops(&data).unwrap() - 2.3).abs() < 1e-9);
                for h in [None, Some("1")] {
                    let meta = apple_metadata(&xmp(h), Some(&data)).unwrap();
                    assert_eq!(meta.alternate_hdr_headroom_n, 2_300_000);
                    assert_eq!(meta.alternate_hdr_headroom_d, 1_000_000);
                }
            }
        }
    }

    #[test]
    fn maker_note_formula_branches_and_missing_tags() {
        for (a, b, expected) in [(0, 0, 1.8), (1, 0, 3.0), (0, 1, 1.5), (1, 1, 2.0)] {
            assert!(
                (apple_maker_stops(&maker(&[(33, a, 1), (48, b, 1)])).unwrap() - expected).abs()
                    < 1e-9
            );
        }
        assert_eq!(apple_maker_stops(&maker(&[(33, 1, 1)])).unwrap(), 3.0);
        assert_eq!(apple_maker_stops(&maker(&[(48, 1, 1)])).unwrap(), 1.5);
        assert_eq!(
            apple_maker_stops(&maker(&[(33, 1, 1), (48, 100, 1)])).unwrap(),
            0.0
        );
        let mut float = maker(&[(33, 0, 1)]);
        float[18..20].copy_from_slice(&11u16.to_be_bytes());
        float[24..28].copy_from_slice(&1f32.to_bits().to_be_bytes());
        assert_eq!(apple_maker_stops(&float).unwrap(), 3.0);
    }

    #[test]
    fn malformed_optional_metadata_is_rejected_without_panicking() {
        for value in ["NaN", "inf", "-1", "0", "0.5", "bad", "4 extra", ""] {
            assert!(apple_metadata(&xmp(Some(value)), None).is_err(), "{value}");
        }
        assert!(apple_metadata(&xmp(None), None).is_err());
        assert!(apple_metadata(&xmp(Some("1")), None).is_err());
        let xml = xmp(Some("4"));
        for end in 0..xml.len() {
            assert!(apple_xmp_headroom(&xml[..end]).is_err());
        }
        let duplicate = String::from_utf8(xml).unwrap().replace(
            "</rdf:Description>",
            "<a:HDRGainMapHeadroom>8</a:HDRGainMapHeadroom></rdf:Description>",
        );
        assert!(apple_xmp_headroom(duplicate.as_bytes()).is_err());
        assert!(apple_maker_stops(&maker(&[])).is_err());
        assert!(apple_maker_stops(&maker(&[(33, 1, 0)])).is_err());
        let mut note = maker(&[(33, 1, 1), (48, 1, 1)]);
        for end in 0..note.len() {
            assert!(apple_maker_stops(&note[..end]).is_err());
        }
        note[24..28].copy_from_slice(&u32::MAX.to_be_bytes());
        assert!(apple_maker_stops(&note).is_err());
    }

    fn decoded_gain(depth: u32) -> GainMapFrame {
        // Odd visible width exercises the decoder's coded padding/row strides.
        let samples: Vec<u16> = (0..17 * 16)
            .map(|v| ((v % 256) << (depth - 8)) as u16)
            .collect();
        let config = hpvca::EncodeConfig::new()
            .with_lossless(true)
            .with_threads(1);
        let heic = match depth {
            8 => hpvca::encode_gray(
                &samples.iter().map(|&v| v as u8).collect::<Vec<_>>(),
                17,
                16,
                &config,
            ),
            10 => hpvca::encode_gray10(&samples, 17, 16, &config),
            12 => hpvca::encode_gray12(&samples, 17, 16, &config),
            _ => unreachable!(),
        }
        .unwrap();
        let decoded = hpvcd::decode_heic_yuv(&heic).unwrap();
        GainMapFrame {
            planes: decoded.planes,
            bit_depth: decoded.bit_depth,
            chroma: decoded.chroma,
            color: decoded.color,
            orientation: hpvcd::Orientation::Normal,
            metadata: Some(xmp(Some("4"))),
        }
    }

    #[test]
    fn iso_samples_reconstruct_apple_multipliers_at_all_supported_depths() {
        for depth in [8, 10, 12] {
            let mut gain = decoded_gain(depth);
            gain.color.cicp.as_mut().unwrap().full_range = true;
            let parsed = parse_apple_gain_map(&gain, None).unwrap();
            assert_eq!(parsed.image.dimensions(), (17, 16));
            for (index, &sample) in parsed.image.as_raw().iter().enumerate() {
                let encoded = ((index % 256) << (depth - 8)) as f64 / ((1u32 << depth) - 1) as f64;
                let linear = if encoded < 4.5 * 0.018053968510807 {
                    encoded / 4.5
                } else {
                    ((encoded + 0.09929682680944) / 1.09929682680944).powf(1.0 / 0.45)
                };
                let apple_multiplier = 1.0 + 3.0 * linear;
                let iso_multiplier = (sample as f64 / 65535.0 * 2.0).exp2();
                assert!((iso_multiplier - apple_multiplier).abs() < 0.00005);
            }
            assert_eq!(parsed.image.as_raw()[0], 0);
        }
    }

    #[test]
    fn hevc_gain_map_inverts_the_apple_import() {
        let mut gain = decoded_gain(8);
        gain.color.cicp.as_mut().unwrap().full_range = true;
        let parsed = parse_apple_gain_map(&gain, None).unwrap();
        let heic = hevc_gain_map(&parsed).unwrap();
        assert_eq!((heic.width, heic.height), (17, 16));
        assert!((heic.headroom - 4.0).abs() < 1e-5);
        let hpvca::GainMapPixels::Gray8(samples) = &heic.pixels else {
            panic!("expected an 8-bit gain map");
        };
        for (index, &sample) in samples.iter().enumerate() {
            assert_eq!(sample as usize, index % 256, "sample {index}");
        }
    }

    #[test]
    fn range_and_orientation_are_applied_to_gain_samples() {
        let mut gain = decoded_gain(8);
        gain.color.cicp.as_mut().unwrap().full_range = false;
        let parsed = parse_apple_gain_map(&gain, None).unwrap();
        assert_eq!(parsed.image.as_raw()[16], 0);
        assert_eq!(parsed.image.as_raw()[235], 65535);
        gain.orientation = hpvcd::Orientation::Rotate90;
        let rotated = parse_apple_gain_map(&gain, None).unwrap();
        assert_eq!(rotated.image, image::imageops::rotate90(&parsed.image));
        gain.chroma = hpvcd::ChromaFormat::Yuv420;
        assert!(parse_apple_gain_map(&gain, None).is_err());
    }
}
