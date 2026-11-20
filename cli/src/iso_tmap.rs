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

//! Read still-image items and ISO 21496-1 tone maps from ISOBMFF metadata.

use crate::PngCicp;
use anyhow::{Context, Result, bail, ensure};
use gainforge::IsoGainMap;
use std::{borrow::Cow, collections::HashMap};

const ALPHA_AUX_TYPES: [&[u8]; 2] = [
    b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0",
    b"urn:mpeg:hevc:2015:auxid:1\0",
];

pub(crate) struct ToneMap {
    pub gain_item: u32,
    pub metadata: IsoGainMap,
    pub alternate_cicp: Option<PngCicp>,
    pub alternate_icc: Option<Vec<u8>>,
}

pub(crate) struct IsoBox<'a> {
    pub kind: [u8; 4],
    pub data: &'a [u8],
}

fn next_box<'a>(bytes: &mut &'a [u8]) -> Result<IsoBox<'a>> {
    let mut header = *bytes;
    let size = read_u32(&mut header)?;
    let kind = take(&mut header, 4)?.try_into()?;
    let (size, header_size) = match size {
        0 => (bytes.len(), 8),
        1 => (usize::try_from(read_uint(&mut header, 8)?)?, 16),
        size => (size as usize, 8),
    };
    ensure!(size >= header_size, "invalid ISOBMFF box size");
    let data = &take(bytes, size)?[header_size..];
    Ok(IsoBox { kind, data })
}

fn find_box<'a>(mut bytes: &'a [u8], kind: &[u8; 4]) -> Result<Option<&'a [u8]>> {
    while !bytes.is_empty() {
        let child = next_box(&mut bytes)?;
        if &child.kind == kind {
            return Ok(Some(child.data));
        }
    }
    Ok(None)
}

fn take<'a>(bytes: &mut &'a [u8], len: usize) -> Result<&'a [u8]> {
    let (data, rest) = bytes
        .split_at_checked(len)
        .context("truncated ISOBMFF data")?;
    *bytes = rest;
    Ok(data)
}

fn read_uint(bytes: &mut &[u8], len: usize) -> Result<u64> {
    ensure!(len <= 8, "invalid ISOBMFF integer width");
    Ok(take(bytes, len)?
        .iter()
        .fold(0u64, |n, &b| (n << 8) | u64::from(b)))
}

fn read_u16(bytes: &mut &[u8]) -> Result<u16> {
    Ok(u16::from_be_bytes(take(bytes, 2)?.try_into()?))
}

fn read_u32(bytes: &mut &[u8]) -> Result<u32> {
    Ok(u32::from_be_bytes(take(bytes, 4)?.try_into()?))
}

fn read_id(bytes: &mut &[u8], wide: bool) -> Result<u32> {
    if wide {
        read_u32(bytes)
    } else {
        Ok(u32::from(read_u16(bytes)?))
    }
}

fn full_box<'a>(data: &mut &'a [u8]) -> Result<&'a [u8; 4]> {
    Ok(take(data, 4)?.try_into()?)
}

fn meta_box(bytes: &[u8]) -> Result<Option<&[u8]>> {
    let Some(mut meta) = find_box(bytes, b"meta")? else {
        return Ok(None);
    };
    let [version, ..] = *full_box(&mut meta)?;
    ensure!(version == 0, "unsupported ISOBMFF meta version {version}");
    Ok(Some(meta))
}

fn primary_field(mut pitm: &[u8]) -> Result<(&[u8], u32)> {
    let [version, ..] = *full_box(&mut pitm)?;
    ensure!(version <= 1, "unsupported ISOBMFF pitm version {version}");
    let field = pitm
        .get(..if version == 1 { 4 } else { 2 })
        .context("truncated primary item ID")?;
    Ok((field, read_id(&mut pitm, version == 1)?))
}

/// Select the gain map item for a decoder that exposes only the primary image.
/// The field is replaced in place so item extents and associated properties
/// retain their original offsets. Return the previous item ID for restoration.
#[cfg(any(feature = "heic", test))]
pub(crate) fn set_primary_item(file: &mut [u8], item_id: u32) -> Result<u32> {
    let meta = meta_box(file)?.context("missing ISOBMFF image metadata")?;
    let (field, previous) =
        primary_field(find_box(meta, b"pitm")?.context("missing primary item")?)?;
    let offset = field.as_ptr() as usize - file.as_ptr() as usize;
    match field.len() {
        2 => file[offset..offset + 2].copy_from_slice(&u16::try_from(item_id)?.to_be_bytes()),
        _ => file[offset..offset + 4].copy_from_slice(&item_id.to_be_bytes()),
    }
    Ok(previous)
}

fn item_kinds(meta: &[u8]) -> Result<HashMap<u32, [u8; 4]>> {
    let Some(mut iinf) = find_box(meta, b"iinf")? else {
        return Ok(HashMap::new());
    };
    let [version, ..] = *full_box(&mut iinf)?;
    ensure!(version <= 1, "unsupported ISOBMFF iinf version {version}");
    let count = read_id(&mut iinf, version == 1)?;
    let mut items = HashMap::new();
    for _ in 0..count {
        let mut item = next_box(&mut iinf)?;
        if &item.kind != b"infe" {
            continue;
        }
        let [version, ..] = *full_box(&mut item.data)?;
        if !matches!(version, 2 | 3) {
            continue;
        }
        let id = read_id(&mut item.data, version == 3)?;
        let protection = read_u16(&mut item.data)?;
        let kind = take(&mut item.data, 4)?.try_into()?;
        if protection == 0 {
            ensure!(
                items.insert(id, kind).is_none(),
                "duplicate ISOBMFF item ID {id}"
            );
        }
    }
    Ok(items)
}

struct ItemReference<'a> {
    from: u32,
    targets: &'a [u8],
    id_width: usize,
}

impl ItemReference<'_> {
    fn targets(&self) -> impl ExactSizeIterator<Item = u32> + '_ {
        self.targets.chunks_exact(self.id_width).map(|id| {
            id.iter()
                .fold(0u32, |value, &byte| (value << 8) | u32::from(byte))
        })
    }
}

fn item_references<'a>(meta: &'a [u8], kind: &[u8; 4]) -> Result<Vec<ItemReference<'a>>> {
    let Some(mut iref) = find_box(meta, b"iref")? else {
        return Ok(Vec::new());
    };
    let [version, ..] = *full_box(&mut iref)?;
    ensure!(version <= 1, "unsupported ISOBMFF iref version {version}");
    let id_width = if version == 1 { 4 } else { 2 };
    let mut references = Vec::new();
    while !iref.is_empty() {
        let mut reference = next_box(&mut iref)?;
        if &reference.kind != kind {
            continue;
        }
        let from = read_id(&mut reference.data, version == 1)?;
        let count = usize::from(read_u16(&mut reference.data)?);
        let targets = take(&mut reference.data, count * id_width)?;
        references.push(ItemReference {
            from,
            targets,
            id_width,
        });
    }
    Ok(references)
}

fn gain_reference(
    meta: &[u8],
    base_item: u32,
    kinds: &HashMap<u32, [u8; 4]>,
) -> Result<Option<(u32, u32)>> {
    let mut result = None;
    for reference in item_references(meta, b"dimg")? {
        if kinds.get(&reference.from) != Some(b"tmap") {
            continue;
        }
        let mut targets = reference.targets();
        ensure!(
            targets.len() == 2,
            "tone map must have two image references"
        );
        let base = targets.next().unwrap();
        let gain = targets.next().unwrap();
        if base == base_item {
            ensure!(
                gain != base && gain != reference.from && gain != 0,
                "invalid gain map item reference"
            );
            ensure!(
                result.is_none(),
                "multiple tone maps reference the primary image"
            );
            result = Some((reference.from, gain));
        }
    }
    Ok(result)
}

struct ItemLocation<'a> {
    construction: u16,
    data_reference: u16,
    base_offset: u64,
    extent_count: usize,
    extents: &'a [u8],
}

struct ItemLocations<'a> {
    items: HashMap<u32, ItemLocation<'a>>,
    index_size: usize,
    offset_size: usize,
    length_size: usize,
}

impl<'a> ItemLocations<'a> {
    fn parse(meta: &'a [u8]) -> Result<Self> {
        let mut iloc = find_box(meta, b"iloc")?.context("missing ISOBMFF item locations")?;
        let [version, ..] = *full_box(&mut iloc)?;
        ensure!(version <= 2, "unsupported ISOBMFF iloc version {version}");
        let &[sizes, base_sizes] = take(&mut iloc, 2)?.try_into()?;
        let (offset_size, length_size) = (usize::from(sizes >> 4), usize::from(sizes & 15));
        let base_size = usize::from(base_sizes >> 4);
        let index_size = if version == 0 {
            0
        } else {
            usize::from(base_sizes & 15)
        };
        ensure!(
            [base_size, offset_size, length_size, index_size]
                .into_iter()
                .all(|width| width <= 8),
            "invalid ISOBMFF item location integer width"
        );
        let extent_size = index_size + offset_size + length_size;
        let count = read_id(&mut iloc, version == 2)?;
        let mut items = HashMap::new();
        for _ in 0..count {
            let item_id = read_id(&mut iloc, version == 2)?;
            let construction = if version == 0 {
                0
            } else {
                read_u16(&mut iloc)? & 15
            };
            let data_reference = read_u16(&mut iloc)?;
            let base_offset = read_uint(&mut iloc, base_size)?;
            let extent_count = usize::from(read_u16(&mut iloc)?);
            let extents = take(&mut iloc, extent_count * extent_size)?;
            let location = ItemLocation {
                construction,
                data_reference,
                base_offset,
                extent_count,
                extents,
            };
            ensure!(
                items.insert(item_id, location).is_none(),
                "duplicate ISOBMFF item location for item {item_id}"
            );
        }
        Ok(Self {
            items,
            index_size,
            offset_size,
            length_size,
        })
    }

    fn data(
        &self,
        file: &'a [u8],
        meta: &'a [u8],
        item: u32,
        limit: usize,
    ) -> Result<Cow<'a, [u8]>> {
        let location = self
            .items
            .get(&item)
            .with_context(|| format!("missing ISOBMFF location for item {item}"))?;
        ensure!(
            location.data_reference == 0,
            "external ISOBMFF data for item {item} is unsupported"
        );
        let storage = match location.construction {
            0 => file,
            1 => find_box(meta, b"idat")?.context("missing ISOBMFF item data box")?,
            method => bail!("unsupported ISOBMFF construction method {method} for item {item}"),
        };
        let mut extents = location.extents;
        let mut segments = Vec::new();
        let mut total = 0usize;
        for _ in 0..location.extent_count {
            take(&mut extents, self.index_size)?;
            let offset = read_uint(&mut extents, self.offset_size)?;
            let length = usize::try_from(read_uint(&mut extents, self.length_size)?)?;
            let start = usize::try_from(
                location
                    .base_offset
                    .checked_add(offset)
                    .context("ISOBMFF item offset overflow")?,
            )?;
            let segment = storage
                .get(start..)
                .and_then(|bytes| bytes.get(..length))
                .with_context(|| format!("ISOBMFF extent for item {item} is outside its data"))?;
            ensure!(length <= limit - total, "oversized ISOBMFF item {item}");
            if location.extent_count == 1 {
                return Ok(Cow::Borrowed(segment));
            }
            if !segment.is_empty() {
                total += length;
                segments.push(segment);
            }
        }
        match segments.as_slice() {
            [] => Ok(Cow::Borrowed(&[])),
            &[segment] => Ok(Cow::Borrowed(segment)),
            _ => {
                let mut data = Vec::new();
                data.try_reserve_exact(total)?;
                for segment in segments {
                    data.extend_from_slice(segment);
                }
                Ok(Cow::Owned(data))
            }
        }
    }
}

pub(crate) struct AvifGrid {
    pub width: u32,
    pub height: u32,
    pub columns: usize,
    pub tiles: Vec<u32>,
}

/// AVIF still-image items. Index locations once so grids can load tiles
/// without repeatedly scanning the location table.
pub(crate) struct AvifItems<'a> {
    file: &'a [u8],
    meta: &'a [u8],
    primary: u32,
    kinds: HashMap<u32, [u8; 4]>,
    locations: ItemLocations<'a>,
}

impl<'a> AvifItems<'a> {
    pub(crate) fn parse(file: &'a [u8]) -> Result<Self> {
        let meta = meta_box(file)?.context("missing AVIF item metadata")?;
        let (_, primary) =
            primary_field(find_box(meta, b"pitm")?.context("missing AVIF primary item")?)?;
        let kinds = item_kinds(meta)?;
        ensure!(
            kinds.contains_key(&primary),
            "missing AVIF primary item {primary}"
        );
        let locations = ItemLocations::parse(meta)?;
        Ok(Self {
            file,
            meta,
            primary,
            kinds,
            locations,
        })
    }

    pub(crate) fn primary(&self) -> u32 {
        self.primary
    }

    pub(crate) fn item_data(&self, item: u32) -> Result<Cow<'a, [u8]>> {
        ensure!(
            self.kinds.get(&item) == Some(b"av01"),
            "AVIF item {item} is not an AV1 image"
        );
        // Bound concatenated extents even if they refer to the same bytes.
        self.locations
            .data(self.file, self.meta, item, self.file.len())
    }

    pub(crate) fn grid(&self, item: u32) -> Result<Option<AvifGrid>> {
        let Some(kind) = self.kinds.get(&item) else {
            bail!("missing AVIF item {item}");
        };
        if kind != b"grid" {
            return Ok(None);
        }
        let data = self.locations.data(self.file, self.meta, item, 12)?;
        let mut data = data.as_ref();
        let &[version, flags, last_row, last_column] = take(&mut data, 4)?.try_into()?;
        ensure!(version == 0, "unsupported AVIF grid version {version}");
        let width = read_id(&mut data, flags & 1 != 0)?;
        let height = read_id(&mut data, flags & 1 != 0)?;
        let rows = usize::from(last_row) + 1;
        let columns = usize::from(last_column) + 1;
        let mut tiles = Vec::new();
        for reference in item_references(self.meta, b"dimg")? {
            if reference.from == item {
                tiles.extend(reference.targets());
            }
        }
        ensure!(
            tiles.len() == rows * columns
                && tiles.iter().all(|id| self.kinds.get(id) == Some(b"av01")),
            "invalid tile references for AVIF grid {item}"
        );
        Ok(Some(AvifGrid {
            width,
            height,
            columns,
            tiles,
        }))
    }

    pub(crate) fn alpha(&self) -> Result<Option<u32>> {
        for reference in item_references(self.meta, b"auxl")? {
            if reference.targets().any(|id| id == self.primary)
                && self.properties(reference.from)?.iter().any(|property| {
                    &property.kind == b"auxC"
                        && property.data.get(4..).is_some_and(|data| {
                            ALPHA_AUX_TYPES
                                .iter()
                                .any(|aux_type| data.starts_with(aux_type))
                        })
                })
            {
                return Ok(Some(reference.from));
            }
        }
        Ok(None)
    }

    pub(crate) fn premultiplied_alpha(&self, alpha: u32) -> Result<bool> {
        Ok(item_references(self.meta, b"prem")?
            .into_iter()
            .any(|reference| {
                reference.from == self.primary && reference.targets().any(|id| id == alpha)
            }))
    }

    pub(crate) fn properties(&self, item: u32) -> Result<Vec<IsoBox<'a>>> {
        item_properties(self.meta, item)
    }

    pub(crate) fn color(&self, item: u32) -> Result<(Option<PngCicp>, Option<Vec<u8>>)> {
        read_item_color(self.meta, item)
    }

    pub(crate) fn tone_map(&self) -> Result<Option<ToneMap>> {
        let Some((tone_item, gain_item)) = gain_reference(self.meta, self.primary, &self.kinds)?
        else {
            return Ok(None);
        };
        tone_map(self.file, self.meta, tone_item, gain_item, &self.locations).map(Some)
    }
}

fn associated_properties<'a>(mut iprp: &'a [u8], item: u32) -> Result<Vec<IsoBox<'a>>> {
    let Some(mut ipco) = find_box(iprp, b"ipco")? else {
        return Ok(Vec::new());
    };
    let mut indexes = Vec::new();
    while !iprp.is_empty() {
        let mut association = next_box(&mut iprp)?;
        if &association.kind != b"ipma" {
            continue;
        }
        let [version, _, _, flags] = *full_box(&mut association.data)?;
        ensure!(version <= 1, "unsupported ISOBMFF ipma version {version}");
        let wide = flags & 1 != 0;
        let count = read_u32(&mut association.data)?;
        for _ in 0..count {
            let id = read_id(&mut association.data, version == 1)?;
            let entries = take(&mut association.data, 1)?[0];
            if id != item {
                take(
                    &mut association.data,
                    usize::from(entries) * if wide { 2 } else { 1 },
                )?;
                continue;
            }
            for _ in 0..entries {
                let index = if wide {
                    read_u16(&mut association.data)? & 0x7fff
                } else {
                    u16::from(take(&mut association.data, 1)?[0] & 0x7f)
                };
                if index != 0 {
                    indexes.push(usize::from(index));
                }
            }
        }
    }

    indexes.sort_unstable();
    indexes.dedup();
    let mut indexes = indexes.into_iter().peekable();
    let mut properties = Vec::new();
    let mut index = 1;
    while indexes.peek().is_some() && !ipco.is_empty() {
        let property = next_box(&mut ipco)?;
        if indexes.peek() == Some(&index) {
            indexes.next();
            properties.push(property);
        }
        index += 1;
    }
    ensure!(indexes.peek().is_none(), "invalid ISOBMFF property index");
    Ok(properties)
}

fn normalized_metadata(data: &[u8]) -> Result<IsoGainMap> {
    let (&version, metadata) = data.split_first().context("empty tone-map item")?;
    ensure!(version == 0, "unsupported ISO tone-map version {version}");
    // The final BMFF metadata syntax reserves flags used by some JPEG/JXL
    // gain maps. Use the BMFF parser before copying into the CLI representation.
    let info = maroontree::IsoGainMap::from_metadata(metadata)?;
    Ok(IsoGainMap {
        gain_map_min_n: info.gain_map_min_n,
        gain_map_min_d: info.gain_map_min_d,
        gain_map_max_n: info.gain_map_max_n,
        gain_map_max_d: info.gain_map_max_d,
        gain_map_gamma_n: info.gain_map_gamma_n,
        gain_map_gamma_d: info.gain_map_gamma_d,
        base_offset_n: info.base_offset_n,
        base_offset_d: info.base_offset_d,
        alternate_offset_n: info.alternate_offset_n,
        alternate_offset_d: info.alternate_offset_d,
        base_hdr_headroom_n: info.base_hdr_headroom_n,
        base_hdr_headroom_d: info.base_hdr_headroom_d,
        alternate_hdr_headroom_n: info.alternate_hdr_headroom_n,
        alternate_hdr_headroom_d: info.alternate_hdr_headroom_d,
        backward_direction: false,
        use_base_color_space: info.use_base_color_space,
    })
}

fn item_properties(meta: &[u8], item: u32) -> Result<Vec<IsoBox<'_>>> {
    match find_box(meta, b"iprp")? {
        Some(iprp) => associated_properties(iprp, item),
        None => Ok(Vec::new()),
    }
}

fn read_item_color(meta: &[u8], item: u32) -> Result<(Option<PngCicp>, Option<Vec<u8>>)> {
    let mut cicp = None;
    let mut icc = None;
    for mut property in item_properties(meta, item)? {
        if &property.kind != b"colr" {
            continue;
        }
        match take(&mut property.data, 4)? {
            b"nclx" => {
                cicp = Some(PngCicp {
                    color_primaries: u8::try_from(read_u16(&mut property.data)?).unwrap_or(2),
                    transfer_function: u8::try_from(read_u16(&mut property.data)?).unwrap_or(2),
                    matrix_coefficients: u8::try_from(read_u16(&mut property.data)?).unwrap_or(2),
                    full_range: take(&mut property.data, 1)?[0] & 0x80 != 0,
                });
            }
            b"prof" | b"rICC" => icc = Some(property.data.to_vec()),
            _ => (),
        }
    }
    Ok((cicp, icc))
}

fn tone_map(
    file: &[u8],
    meta: &[u8],
    tone_item: u32,
    gain_item: u32,
    locations: &ItemLocations<'_>,
) -> Result<ToneMap> {
    // Standard payloads are at most 142 bytes. Allow larger future versions
    // without allocating arbitrarily large or repeated extents.
    let metadata = normalized_metadata(&locations.data(file, meta, tone_item, 1024 * 1024)?)?;
    let (alternate_cicp, alternate_icc) = read_item_color(meta, tone_item)?;
    Ok(ToneMap {
        gain_item,
        metadata,
        alternate_cicp,
        alternate_icc,
    })
}

#[cfg(any(feature = "heic", test))]
pub(crate) fn read_tone_map(file: &[u8]) -> Result<Option<ToneMap>> {
    let Some(meta) = meta_box(file)? else {
        return Ok(None);
    };
    let Some(pitm) = find_box(meta, b"pitm")? else {
        return Ok(None);
    };
    let (_, primary) = primary_field(pitm)?;
    let Some((tone_item, gain_item)) = gain_reference(meta, primary, &item_kinds(meta)?)? else {
        return Ok(None);
    };
    tone_map(
        file,
        meta,
        tone_item,
        gain_item,
        &ItemLocations::parse(meta)?,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn boxed(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
        let size = (data.len() as u32 + 8).to_be_bytes();
        [&size[..], &kind[..], data].concat()
    }

    #[test]
    fn identifies_alpha_grid_and_preserves_primary_color_profiles() {
        let pitm = boxed(b"pitm", &[0, 0, 0, 0, 0, 1]);
        let primary = boxed(b"infe", b"\x02\0\0\0\0\x01\0\0av01\0");
        let grid = boxed(b"infe", b"\x02\0\0\0\0\x02\0\0grid\0");
        let tile = boxed(b"infe", b"\x02\0\0\0\0\x03\0\0av01\0");
        let iinf = boxed(
            b"iinf",
            &[vec![0, 0, 0, 0, 0, 3], primary, grid, tile].concat(),
        );
        let iref = boxed(
            b"iref",
            &[
                vec![0; 4],
                boxed(b"auxl", &[0, 2, 0, 1, 0, 1]),
                boxed(b"prem", &[0, 1, 0, 1, 0, 2]),
                boxed(b"dimg", &[0, 2, 0, 1, 0, 3]),
            ]
            .concat(),
        );
        let idat = boxed(b"idat", b"\0\0\0\0\0\x03\0\x02tile");
        let mut iloc = vec![1, 0, 0, 0, 0x44, 0, 0, 2];
        for (item, offset, length) in [(2u16, 0u32, 8u32), (3, 8, 4)] {
            iloc.extend_from_slice(&item.to_be_bytes());
            iloc.extend_from_slice(&[0, 1, 0, 0, 0, 1]); // idat, local reference, one extent
            iloc.extend_from_slice(&offset.to_be_bytes());
            iloc.extend_from_slice(&length.to_be_bytes());
        }
        let iloc = boxed(b"iloc", &iloc);
        let nclx = boxed(b"colr", b"nclx\0\x09\0\x10\0\x09\x80");
        let icc = boxed(b"colr", b"profcolor");
        let ipma = boxed(
            b"ipma",
            &[0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 2, 1, 2, 0, 2, 1, 3],
        );
        for alpha_type in [
            b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0".as_slice(),
            b"urn:mpeg:hevc:2015:auxid:1\0".as_slice(),
        ] {
            let auxc = boxed(b"auxC", &[&[0; 4], alpha_type].concat());
            let ipco = boxed(b"ipco", &[&nclx[..], &icc, &auxc].concat());
            let iprp = boxed(b"iprp", &[&ipco[..], &ipma].concat());
            let mut file = boxed(
                b"meta",
                &[&[0; 4], pitm.as_slice(), &iinf, &iref, &iprp, &iloc, &idat].concat(),
            );

            let items = AvifItems::parse(&file).unwrap();
            assert_eq!(items.primary(), 1);
            assert_eq!(items.alpha().unwrap(), Some(2));
            assert!(items.premultiplied_alpha(2).unwrap());
            let grid = items.grid(2).unwrap().unwrap();
            assert_eq!((grid.width, grid.height, grid.columns), (3, 2, 1));
            assert_eq!(grid.tiles, [3]);
            assert!(matches!(
                items.item_data(3).unwrap(),
                Cow::Borrowed(b"tile")
            ));
            let (cicp, icc) = items.color(1).unwrap();
            assert_eq!(cicp.unwrap().transfer_function, 16);
            assert_eq!(icc.as_deref(), Some(&b"color"[..]));
            assert_eq!(set_primary_item(&mut file, 2).unwrap(), 1);
            assert_eq!(AvifItems::parse(&file).unwrap().alpha().unwrap(), None);
        }
    }

    #[test]
    fn reads_split_file_extents_for_a_wide_avif_item_id() {
        let id = 65_537u32;
        let id_bytes = id.to_be_bytes();
        let mdat = boxed(b"mdat", b"a-bc");
        let pitm = boxed(b"pitm", &[&[1, 0, 0, 0], id_bytes.as_slice()].concat());
        let item = boxed(
            b"infe",
            &[&[3, 0, 0, 0], id_bytes.as_slice(), b"\0\0av01\0"].concat(),
        );
        let iinf = boxed(b"iinf", &[vec![1, 0, 0, 0, 0, 0, 0, 1], item].concat());
        let make_file = |last_length: u32| {
            let mut iloc = vec![2, 0, 0, 0, 0x44, 0, 0, 0, 0, 1];
            iloc.extend_from_slice(&id_bytes);
            iloc.extend_from_slice(&[0, 0, 0, 0, 0, 2]); // file-relative, local, two extents
            for (offset, length) in [(8u32, 1u32), (10, last_length)] {
                iloc.extend_from_slice(&offset.to_be_bytes());
                iloc.extend_from_slice(&length.to_be_bytes());
            }
            let metadata = [
                vec![0; 4],
                pitm.clone(),
                iinf.clone(),
                boxed(b"iloc", &iloc),
            ]
            .concat();
            [mdat.clone(), boxed(b"meta", &metadata)].concat()
        };
        let file = make_file(2);
        let items = AvifItems::parse(&file).unwrap();
        assert_eq!(items.primary(), id);
        assert!(matches!(items.item_data(id).unwrap(), Cow::Owned(data) if data == b"abc"));
        let truncated = make_file(u32::MAX);
        assert!(AvifItems::parse(&truncated).unwrap().item_data(id).is_err());
    }

    #[test]
    fn reads_split_tone_map_extents_and_alternate_color() {
        let metadata = maroontree::IsoGainMap::from_floats(&maroontree::GainMapFloats {
            alternate_hdr_headroom: 3.0,
            ..Default::default()
        })
        .unwrap()
        .to_metadata()
        .unwrap();
        let pitm = boxed(b"pitm", &[0, 0, 0, 0, 0, 1]);
        let item = boxed(b"infe", b"\x02\0\0\0\0\x03\0\0tmap\0");
        let iinf = boxed(b"iinf", &[&[0, 0, 0, 0, 0, 1], item.as_slice()].concat());
        let dimg = boxed(b"dimg", &[0, 3, 0, 2, 0, 1, 0, 2]);
        let iref = boxed(b"iref", &[&[0, 0, 0, 0], dimg.as_slice()].concat());

        // Store the payload version separately; item extents are relative
        // to idat and may skip over unrelated bytes.
        let idat = boxed(b"idat", &[&[0, 255], metadata.as_slice()].concat());
        let mut location = vec![1, 0, 0, 0, 0x44, 0, 0, 1, 0, 3, 0, 1, 0, 0, 0, 2];
        for (offset, length) in [(0u32, 1u32), (2, metadata.len() as u32)] {
            location.extend_from_slice(&offset.to_be_bytes());
            location.extend_from_slice(&length.to_be_bytes());
        }
        let iloc = boxed(b"iloc", &location);

        let nclx = boxed(b"colr", b"nclx\0\x09\0\x12\0\0\x80");
        let icc = boxed(b"colr", b"profcolor");
        let ipco = boxed(b"ipco", &[nclx, icc].concat());
        let ipma = boxed(b"ipma", &[0, 0, 0, 1, 0, 0, 0, 1, 0, 3, 2, 0, 1, 0, 2]);
        let iprp = boxed(b"iprp", &[ipco, ipma].concat());
        let file = boxed(
            b"meta",
            &[vec![0; 4], pitm, iinf, iref, iloc, iprp, idat].concat(),
        );

        let map = read_tone_map(&file).unwrap().unwrap();
        assert_eq!(map.gain_item, 2);
        assert_eq!(map.metadata.alternate_hdr_headroom(), 3.0);
        assert_eq!(map.alternate_cicp.unwrap().transfer_function, 18);
        assert_eq!(map.alternate_icc.as_deref(), Some(&b"color"[..]));
    }
}
