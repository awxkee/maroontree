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

//! Decode AV1 AVIF still images without tone mapping the encoded RGB samples.

use crate::{
    LoadedImage, PngCicp,
    gainmap::ParsedGainMap,
    iso_tmap::{AvifGrid, AvifItems, IsoBox},
    resize::resize_bilinear_u16,
};
use anyhow::{Context, Result, bail, ensure};
use image::{DynamicImage, GenericImage, ImageBuffer, metadata::Orientation};
use rav1d::include::dav1d::{
    data::Dav1dData,
    dav1d::{Dav1dContext, Dav1dSettings},
    headers::{
        DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422,
        DAV1D_PIXEL_LAYOUT_I444,
    },
    picture::Dav1dPicture,
};
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_picture_unref, dav1d_send_data,
};
use std::{mem::MaybeUninit, path::Path, ptr::NonNull};
use yuv::{
    YuvChromaSubsampling::{self, Yuv420, Yuv422, Yuv444},
    YuvPlanarImage, YuvRange, YuvStandardMatrix,
};

const MAX_FRAME_PIXELS: u32 = 120_000_000;

pub(crate) fn decode_avif_file(path: &Path, load_gain_map: bool) -> Result<LoadedImage> {
    let bytes = std::fs::read(path).context("reading AVIF")?;
    let items = AvifItems::parse(&bytes)?;
    let primary = items.primary();
    let (nclx, icc) = items.color(primary)?;
    let (image, depth, cicp) = decode_item(&items, primary, nclx, false)?;
    let image = if let Some(alpha) = items.alpha()? {
        let (plane, _, _) = decode_item(&items, alpha, None, true)?;
        attach_alpha(
            image,
            plane.into_luma16(),
            items.premultiplied_alpha(alpha)?,
        )?
    } else {
        image
    };
    let image = if depth == 8 {
        match image {
            DynamicImage::ImageLuma16(_) => DynamicImage::ImageLuma8(image.into_luma8()),
            DynamicImage::ImageLumaA16(_) => DynamicImage::ImageLumaA8(image.into_luma_alpha8()),
            DynamicImage::ImageRgb16(_) => DynamicImage::ImageRgb8(image.into_rgb8()),
            _ => DynamicImage::ImageRgba8(image.into_rgba8()),
        }
    } else {
        image
    };
    let image = apply_geometry(image, &items.properties(primary)?)?;

    let gain_map = if load_gain_map {
        items
            .tone_map()?
            .map(|tone_map| {
                let item = tone_map.gain_item;
                let (nclx, _) = items.color(item)?;
                let (image, _, _) = decode_item(&items, item, nclx, false)
                    .with_context(|| format!("decoding AVIF gain map item {item}"))?;
                Ok::<_, anyhow::Error>(ParsedGainMap {
                    image: apply_geometry(image, &items.properties(item)?)?,
                    metadata: tone_map.metadata,
                    alternate_cicp: tone_map.alternate_cicp,
                    alternate_icc: tone_map.alternate_icc,
                })
            })
            .transpose()?
    } else {
        None
    };
    Ok(LoadedImage {
        image,
        icc,
        cicp: Some(PngCicp {
            matrix_coefficients: 0,
            full_range: true,
            ..cicp
        }),
        gain_map,
    })
}

fn decode_item(
    items: &AvifItems<'_>,
    id: u32,
    nclx: Option<PngCicp>,
    alpha: bool,
) -> Result<(DynamicImage, u8, PngCicp)> {
    if let Some(grid) = items.grid(id)? {
        decode_grid(items, grid, nclx, alpha)
    } else {
        decode_pixels(&items.item_data(id)?, nclx, alpha)
    }
}

struct Decoder(Option<Dav1dContext>);
struct Packet(Dav1dData);
struct Picture(Dav1dPicture);

impl Drop for Decoder {
    fn drop(&mut self) {
        // SAFETY: the pointer refers to the stored context created by dav1d_open.
        unsafe { dav1d_close(Some(NonNull::from(&mut self.0))) };
    }
}

impl Drop for Packet {
    fn drop(&mut self) {
        // SAFETY: the packet is initialized and may be empty after sending.
        unsafe { dav1d_data_unref(Some(NonNull::from(&mut self.0))) };
    }
}

impl Drop for Picture {
    fn drop(&mut self) {
        // SAFETY: the picture is initialized and owns any returned decoder reference.
        unsafe { dav1d_picture_unref(Some(NonNull::from(&mut self.0))) };
    }
}

fn decode_picture(data: &[u8], alpha: bool) -> Result<Picture> {
    ensure!(!data.is_empty(), "empty AV1 item");
    let mut decoder = Decoder(None);
    let mut packet = Packet(Dav1dData::default());
    let mut picture = Picture(Dav1dPicture::default());
    let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
    // SAFETY: all pointers refer to live, appropriately initialized stack or
    // decoder allocations. The three guards release their own references.
    unsafe {
        dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap());
        let mut settings = settings.assume_init();
        settings.max_frame_delay = 1;
        settings.apply_grain = i32::from(!alpha);
        settings.frame_size_limit = MAX_FRAME_PIXELS;
        let result = dav1d_open(
            Some(NonNull::from(&mut decoder.0)),
            Some(NonNull::from(&mut settings)),
        );
        ensure!(
            result.0 == 0,
            "could not initialize AV1 decoder ({})",
            result.0
        );
        let buffer = dav1d_data_create(Some(NonNull::from(&mut packet.0)), data.len());
        ensure!(!buffer.is_null(), "could not allocate AV1 packet");
        std::ptr::copy_nonoverlapping(data.as_ptr(), buffer, data.len());
        let result = dav1d_send_data(decoder.0, Some(NonNull::from(&mut packet.0)));
        ensure!(result.0 == 0, "could not decode AV1 item ({})", result.0);
        let result = dav1d_get_picture(decoder.0, Some(NonNull::from(&mut picture.0)));
        ensure!(
            result.0 == 0,
            "AV1 item produced no decodable picture ({})",
            result.0
        );
    }
    Ok(picture)
}

impl Picture {
    fn plane<T: bytemuck::Pod>(
        &self,
        index: usize,
        width: u32,
        height: u32,
    ) -> Result<(&[T], u32)> {
        let pointer = self.0.data[index]
            .context("missing AV1 pixel plane")?
            .cast::<T>();
        let byte_stride = u32::try_from(self.0.stride[usize::from(index != 0)])?;
        let sample_size = size_of::<T>() as u32;
        ensure!(
            sample_size == if self.0.p.bpc > 8 { 2 } else { 1 },
            "invalid AV1 plane sample size"
        );
        let stride = byte_stride / sample_size;
        ensure!(
            byte_stride % sample_size == 0
                && stride >= width
                && height > 0
                && pointer.as_ptr().is_aligned(),
            "invalid AV1 plane stride"
        );
        let sample_count = (height as usize - 1)
            .checked_mul(stride as usize)
            .and_then(|len| len.checked_add(width as usize))
            .context("AV1 plane size overflow")?;
        ensure!(
            sample_count <= isize::MAX as usize / sample_size as usize,
            "AV1 plane is too large"
        );
        // SAFETY: callers choose u8 or native-endian u16 from the coded depth.
        // The default allocator initializes the entire plane, including padding.
        // The slice ends in the last active row and borrows this Picture.
        let samples = unsafe { std::slice::from_raw_parts(pointer.as_ptr(), sample_count) };
        Ok((samples, stride))
    }

    fn planar<T: bytemuck::Pod + std::fmt::Debug>(
        &self,
        width: u32,
        height: u32,
        sampling: YuvChromaSubsampling,
    ) -> Result<YuvPlanarImage<'_, T>> {
        let (horizontal, vertical) = chroma_factors(sampling);
        let chroma_width = width.div_ceil(horizontal);
        let chroma_height = height.div_ceil(vertical);
        let (y_plane, y_stride) = self.plane(0, width, height)?;
        let (u_plane, u_stride) = self.plane(1, chroma_width, chroma_height)?;
        let (v_plane, v_stride) = self.plane(2, chroma_width, chroma_height)?;
        Ok(YuvPlanarImage {
            y_plane,
            y_stride,
            u_plane,
            u_stride,
            v_plane,
            v_stride,
            width,
            height,
        })
    }
}

fn chroma_factors(sampling: YuvChromaSubsampling) -> (u32, u32) {
    match sampling {
        Yuv420 => (2, 2),
        Yuv422 => (2, 1),
        Yuv444 => (1, 1),
    }
}

fn sample_scale(depth: u8, range: YuvRange) -> Result<impl Fn(u32) -> u16> {
    let (black, span) = match range {
        YuvRange::Full => (0, (1u32 << depth) - 1),
        YuvRange::Limited => {
            ensure!(depth >= 8, "limited range requires at least 8-bit RGB");
            (16u32 << (depth - 8), 219u32 << (depth - 8))
        }
    };
    Ok(move |sample: u32| {
        let sample = sample.saturating_sub(black).min(span);
        ((sample * u32::from(u16::MAX) + span / 2) / span) as u16
    })
}

fn decode_luma<T: bytemuck::Pod + Into<u32>>(
    picture: &Picture,
    width: u32,
    height: u32,
    depth: u8,
    range: YuvRange,
) -> Result<DynamicImage> {
    let (samples, stride) = picture.plane::<T>(0, width, height)?;
    let scale = sample_scale(depth, range)?;
    let samples = samples
        .chunks(stride as usize)
        .flat_map(|row| {
            row[..width as usize]
                .iter()
                .map(|&sample| scale(sample.into()))
        })
        .collect();
    let image = ImageBuffer::from_raw(width, height, samples).context("invalid AV1 luma image")?;
    Ok(DynamicImage::ImageLuma16(image))
}

fn standard_matrix(cicp: PngCicp) -> Result<YuvStandardMatrix> {
    use YuvStandardMatrix::{Bt601, Bt709, Bt2020, Fcc, Smpte240};
    Ok(match cicp.matrix_coefficients {
        1 => Bt709,
        2 if cicp.color_primaries == 9 => Bt2020,
        2 | 5 | 6 => Bt601,
        4 => Fcc,
        7 => Smpte240,
        9 => Bt2020,
        matrix => bail!("unsupported AVIF matrix coefficients {matrix}"),
    })
}

fn decode_rgb8(
    planes: &YuvPlanarImage<'_, u8>,
    sampling: YuvChromaSubsampling,
    cicp: PngCicp,
    range: YuvRange,
) -> Result<Vec<u16>> {
    let stride = planes.width * 3;
    let mut rgb = vec![0u8; stride as usize * planes.height as usize];
    match cicp.matrix_coefficients {
        0 => yuv::gbr_to_rgb(planes, &mut rgb, stride, range)?,
        8 => {
            let convert = match sampling {
                Yuv420 => yuv::ycgco420_to_rgb,
                Yuv422 => yuv::ycgco422_to_rgb,
                Yuv444 => yuv::ycgco444_to_rgb,
            };
            convert(planes, &mut rgb, stride, range)?;
        }
        _ => {
            let convert = match sampling {
                Yuv420 => yuv::yuv420_to_rgb,
                Yuv422 => yuv::yuv422_to_rgb,
                Yuv444 => yuv::yuv444_to_rgb,
            };
            convert(planes, &mut rgb, stride, range, standard_matrix(cicp)?)?;
        }
    }
    Ok(rgb
        .into_iter()
        .map(|sample| u16::from(sample) * 257)
        .collect())
}

fn decode_re10(
    planes: &YuvPlanarImage<'_, u16>,
    sampling: YuvChromaSubsampling,
    range: YuvRange,
) -> Result<Vec<u16>> {
    let stride = planes.width * 3;
    let mut rgb = vec![0u8; stride as usize * planes.height as usize];
    let convert = match sampling {
        Yuv420 => yuv::icgc_re010_to_rgb,
        Yuv422 => yuv::icgc_re210_to_rgb,
        Yuv444 => yuv::icgc_re410_to_rgb,
    };
    // Re/Ro range expansion applies to the reconstructed RGB samples.
    convert(planes, &mut rgb, stride, YuvRange::Full)?;
    let scale = sample_scale(8, range)?;
    Ok(rgb
        .into_iter()
        .map(|sample| scale(u32::from(sample)))
        .collect())
}

fn decode_rgb_wide(
    planes: &YuvPlanarImage<'_, u16>,
    sampling: YuvChromaSubsampling,
    cicp: PngCicp,
    depth: u8,
    range: YuvRange,
) -> Result<Vec<u16>> {
    let stride = planes.width * 3;
    let mut rgb = vec![0u16; stride as usize * planes.height as usize];
    let (rgb_depth, rgb_range) = match cicp.matrix_coefficients {
        0 => {
            let convert = if depth == 10 {
                yuv::gb10_to_rgb10
            } else {
                yuv::gb12_to_rgb12
            };
            convert(planes, &mut rgb, stride, range)?;
            (depth, YuvRange::Full)
        }
        8 => {
            let convert = match (depth, sampling) {
                (10, Yuv420) => yuv::icgc010_to_rgb10,
                (10, Yuv422) => yuv::icgc210_to_rgb10,
                (10, Yuv444) => yuv::icgc410_to_rgb10,
                (_, Yuv420) => yuv::icgc012_to_rgb12,
                (_, Yuv422) => yuv::icgc212_to_rgb12,
                (_, Yuv444) => yuv::icgc412_to_rgb12,
            };
            convert(planes, &mut rgb, stride, range)?;
            (depth, YuvRange::Full)
        }
        16 => {
            let convert = match sampling {
                Yuv420 => yuv::icgc_re012_to_rgb10,
                Yuv422 => yuv::icgc_re212_to_rgb10,
                Yuv444 => yuv::icgc_re412_to_rgb10,
            };
            convert(planes, &mut rgb, stride, YuvRange::Full)?;
            (10, range)
        }
        _ => {
            let convert = match (depth, sampling) {
                (10, Yuv420) => yuv::i010_to_rgb10,
                (10, Yuv422) => yuv::i210_to_rgb10,
                (10, Yuv444) => yuv::i410_to_rgb10,
                (_, Yuv420) => yuv::i012_to_rgb12,
                (_, Yuv422) => yuv::i212_to_rgb12,
                (_, Yuv444) => yuv::i412_to_rgb12,
            };
            convert(planes, &mut rgb, stride, range, standard_matrix(cicp)?)?;
            (depth, YuvRange::Full)
        }
    };
    let scale = sample_scale(rgb_depth, rgb_range)?;
    for sample in &mut rgb {
        *sample = scale(u32::from(*sample));
    }
    Ok(rgb)
}

/// Exact inverse lifting for depths without a yuv crate kernel (H.273, equations 62–65).
/// The crate handles 10/12-bit coded Re, and its Ro kernels expect 9/11-bit coded samples.
fn decode_reversible<T: Copy + std::fmt::Debug + Into<i32>>(
    planes: &YuvPlanarImage<'_, T>,
    sampling: YuvChromaSubsampling,
    coded_depth: u8,
    rgb_depth: u8,
    range: YuvRange,
) -> Result<Vec<u16>> {
    let scale = sample_scale(rgb_depth, range)?;
    let bias = 1i32 << (coded_depth - 1);
    let max_rgb = (1i32 << rgb_depth) - 1;
    let clip = |sample: i32| sample.clamp(0, max_rgb);
    let (horizontal, vertical) = chroma_factors(sampling);
    let mut rgb = Vec::with_capacity(planes.width as usize * planes.height as usize * 3);
    for y in 0..planes.height as usize {
        let chroma_y = y / vertical as usize;
        for x in 0..planes.width as usize {
            let chroma_x = x / horizontal as usize;
            let luma = planes.y_plane[y * planes.y_stride as usize + x].into();
            let cg = planes.u_plane[chroma_y * planes.u_stride as usize + chroma_x].into() - bias;
            let co = planes.v_plane[chroma_y * planes.v_stride as usize + chroma_x].into() - bias;
            let t = luma - (cg >> 1);
            let blue = clip(t - (co >> 1));
            rgb.extend([clip(blue + co), clip(t + cg), blue].map(|sample| scale(sample as u32)));
        }
    }
    Ok(rgb)
}

fn decode_pixels(
    data: &[u8],
    nclx: Option<PngCicp>,
    alpha: bool,
) -> Result<(DynamicImage, u8, PngCicp)> {
    let picture = decode_picture(data, alpha)?;
    let (width, height) = (u32::try_from(picture.0.p.w)?, u32::try_from(picture.0.p.h)?);
    check_dimensions(width, height)?;
    let depth = u8::try_from(picture.0.p.bpc)?;
    ensure!(matches!(depth, 8 | 10 | 12), "unsupported AV1 sample depth");
    let seq = picture.0.seq_hdr.context("missing AV1 sequence header")?;
    // SAFETY: the picture owns the sequence header reference.
    let seq = unsafe { seq.as_ref() };
    let cicp = nclx.unwrap_or(PngCicp {
        color_primaries: u8::try_from(seq.pri).unwrap_or(2),
        transfer_function: u8::try_from(seq.trc).unwrap_or(2),
        matrix_coefficients: u8::try_from(seq.mtrx).unwrap_or(2),
        full_range: seq.color_range != 0,
    });
    let range = if cicp.full_range || alpha {
        YuvRange::Full
    } else {
        YuvRange::Limited
    };
    if picture.0.p.layout == DAV1D_PIXEL_LAYOUT_I400 {
        let image = if depth == 8 {
            decode_luma::<u8>(&picture, width, height, depth, range)?
        } else {
            decode_luma::<u16>(&picture, width, height, depth, range)?
        };
        return Ok((image, depth, cicp));
    }
    ensure!(!alpha, "AVIF alpha item must be monochrome");
    let sampling = match picture.0.p.layout {
        DAV1D_PIXEL_LAYOUT_I420 => Yuv420,
        DAV1D_PIXEL_LAYOUT_I422 => Yuv422,
        DAV1D_PIXEL_LAYOUT_I444 => Yuv444,
        _ => bail!("unsupported AV1 pixel layout"),
    };
    let matrix = cicp.matrix_coefficients;
    ensure!(
        matrix != 0 || sampling == Yuv444,
        "AVIF identity matrix requires 4:4:4"
    );
    let rgb = if depth == 8 {
        let planes = picture.planar::<u8>(width, height, sampling)?;
        match matrix {
            16 => decode_reversible(&planes, sampling, depth, 6, range)?,
            17 => decode_reversible(&planes, sampling, depth, 7, range)?,
            _ => decode_rgb8(&planes, sampling, cicp, range)?,
        }
    } else {
        let planes = picture.planar::<u16>(width, height, sampling)?;
        match (matrix, depth) {
            (16, 10) => decode_re10(&planes, sampling, range)?,
            (17, _) => decode_reversible(&planes, sampling, depth, depth - 1, range)?,
            _ => decode_rgb_wide(&planes, sampling, cicp, depth, range)?,
        }
    };
    let image = ImageBuffer::from_raw(width, height, rgb).context("invalid AV1 RGB image")?;
    Ok((DynamicImage::ImageRgb16(image), depth, cicp))
}

fn check_dimensions(width: u32, height: u32) -> Result<()> {
    ensure!(
        width > 0
            && height > 0
            && u64::from(width) * u64::from(height) <= u64::from(MAX_FRAME_PIXELS),
        "invalid or too large AVIF dimensions"
    );
    Ok(())
}

fn decode_grid(
    items: &AvifItems<'_>,
    grid: AvifGrid,
    nclx: Option<PngCicp>,
    alpha: bool,
) -> Result<(DynamicImage, u8, PngCicp)> {
    let AvifGrid {
        width,
        height,
        columns,
        tiles,
    } = grid;
    check_dimensions(width, height)?;
    let first = *tiles.first().context("empty AVIF grid")?;
    let (first, depth, cicp) = decode_pixels(&items.item_data(first)?, nclx, alpha)?;
    let (tile_width, tile_height) = (first.width(), first.height());
    let rows = height.div_ceil(tile_height) as usize;
    ensure!(
        columns == width.div_ceil(tile_width) as usize && rows * columns == tiles.len(),
        "AVIF grid dimensions do not match the tiles"
    );
    let mut image = DynamicImage::new(width, height, first.color());
    copy_tile(&mut image, first, 0, 0)?;
    for (i, &id) in tiles.iter().enumerate().skip(1) {
        let (tile, tile_depth, tile_cicp) = decode_pixels(&items.item_data(id)?, nclx, alpha)?;
        ensure!(
            tile.width() == tile_width
                && tile.height() == tile_height
                && tile.color() == image.color()
                && tile_depth == depth
                && tile_cicp == cicp,
            "inconsistent AVIF grid tile"
        );
        let x = (i % columns) as u32 * tile_width;
        let y = (i / columns) as u32 * tile_height;
        copy_tile(&mut image, tile, x, y)?;
    }
    Ok((image, depth, cicp))
}

fn copy_tile(dst: &mut DynamicImage, tile: DynamicImage, x: u32, y: u32) -> Result<()> {
    let w = tile.width().min(dst.width() - x);
    let h = tile.height().min(dst.height() - y);
    let tile = if w != tile.width() || h != tile.height() {
        tile.crop_imm(0, 0, w, h)
    } else {
        tile
    };
    // DynamicImage's generic copy works in RGBA8, so copy typed 16-bit views.
    match (dst, tile) {
        (DynamicImage::ImageRgb16(dst), DynamicImage::ImageRgb16(tile)) => {
            dst.copy_from(&tile, x, y)?
        }
        (DynamicImage::ImageLuma16(dst), DynamicImage::ImageLuma16(tile)) => {
            dst.copy_from(&tile, x, y)?
        }
        _ => bail!("inconsistent AVIF grid tile layout"),
    }
    Ok(())
}

fn attach_alpha(
    base: DynamicImage,
    alpha: ImageBuffer<image::Luma<u16>, Vec<u16>>,
    premultiplied: bool,
) -> Result<DynamicImage> {
    let (w, h) = (base.width(), base.height());
    let alpha = if alpha.dimensions() != (w, h) {
        resize_bilinear_u16::<1, _>(&alpha, (w, h)).context("resizing AVIF alpha")?
    } else {
        alpha
    };
    let unpremultiply = |sample: u16, alpha: u16| {
        if !premultiplied || alpha == 0 {
            return sample;
        }
        ((u32::from(sample) * u32::from(u16::MAX) + u32::from(alpha) / 2) / u32::from(alpha))
            .min(u32::from(u16::MAX)) as u16
    };
    Ok(match base {
        DynamicImage::ImageLuma16(base) => {
            DynamicImage::ImageLumaA16(ImageBuffer::from_fn(w, h, |x, y| {
                let a = alpha.get_pixel(x, y).0[0];
                image::LumaA([unpremultiply(base.get_pixel(x, y).0[0], a), a])
            }))
        }
        DynamicImage::ImageRgb16(base) => {
            DynamicImage::ImageRgba16(ImageBuffer::from_fn(w, h, |x, y| {
                let a = alpha.get_pixel(x, y).0[0];
                let [r, g, b] = base
                    .get_pixel(x, y)
                    .0
                    .map(|sample| unpremultiply(sample, a));
                image::Rgba([r, g, b, a])
            }))
        }
        _ => bail!("unexpected AVIF base pixel layout"),
    })
}

fn apply_geometry(mut image: DynamicImage, properties: &[IsoBox<'_>]) -> Result<DynamicImage> {
    if let Some(aperture) = properties.iter().find(|property| &property.kind == b"clap") {
        let (fields, tail) = aperture.data.as_chunks::<4>();
        ensure!(
            fields.len() == 8 && tail.is_empty(),
            "invalid AVIF clean aperture"
        );
        let fields: [[u8; 4]; 8] = fields.try_into()?;
        let [width_n, width_d, height_n, height_d, dx_n, dx_d, dy_n, dy_d] =
            fields.map(u32::from_be_bytes);
        let width = f64::from(width_n) / f64::from(width_d);
        let height = f64::from(height_n) / f64::from(height_d);
        let dx = f64::from(dx_n as i32) / f64::from(dx_d);
        let dy = f64::from(dy_n as i32) / f64::from(dy_d);
        let x = ((f64::from(image.width()) - width) / 2.0 + dx).round();
        let y = ((f64::from(image.height()) - height) / 2.0 + dy).round();
        let width = width.round();
        let height = height.round();
        ensure!(
            x >= 0.0
                && y >= 0.0
                && width > 0.0
                && height > 0.0
                && x + width <= f64::from(image.width())
                && y + height <= f64::from(image.height()),
            "AVIF clean aperture is outside the decoded image"
        );
        image = image.crop_imm(x as u32, y as u32, width as u32, height as u32);
    }
    let orientation_property = |kind: &[u8; 4]| {
        properties
            .iter()
            .find(|property| &property.kind == kind)
            .map(|property| {
                let [value] = property.data else {
                    bail!("invalid AVIF orientation property");
                };
                Ok(*value)
            })
            .transpose()
    };
    let rotation = orientation_property(b"irot")?.unwrap_or(0);
    let mirror = orientation_property(b"imir")?;
    // MIAF applies the counterclockwise rotation before top-bottom (0) or
    // left-right (1) mirroring. Combine both operations in one orientation.
    let orientation = match (rotation, mirror) {
        (0, None) => Orientation::NoTransforms,
        (1, None) => Orientation::Rotate270,
        (2, None) => Orientation::Rotate180,
        (3, None) => Orientation::Rotate90,
        (0, Some(0)) | (2, Some(1)) => Orientation::FlipVertical,
        (0, Some(1)) | (2, Some(0)) => Orientation::FlipHorizontal,
        (1, Some(0)) | (3, Some(1)) => Orientation::Rotate90FlipH,
        (1, Some(1)) | (3, Some(0)) => Orientation::Rotate270FlipH,
        _ => anyhow::bail!("invalid AVIF orientation"),
    };
    image.apply_orientation(orientation);
    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Luma;

    #[test]
    fn resizes_avif_alpha_at_16_bit_precision() {
        let base = DynamicImage::ImageLuma16(ImageBuffer::from_pixel(4, 1, Luma([30000])));
        let alpha = ImageBuffer::from_raw(2, 1, vec![0, u16::MAX]).unwrap();
        let output = attach_alpha(base, alpha, false).unwrap();
        let output = output.as_luma_alpha16().unwrap();
        assert_eq!(output.dimensions(), (4, 1));
        for (pixel, expected_alpha) in output.pixels().zip([0, 16384, 49151, u16::MAX]) {
            assert_eq!(pixel.0[0], 30000);
            assert!(pixel.0[1].abs_diff(expected_alpha) <= 1);
        }
    }
}
