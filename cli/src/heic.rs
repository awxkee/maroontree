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
use crate::orientation::apply_orientation;
use crate::{Args, Chroma, Depth, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12};
use hpvca::ChromaFormat;
use image::{DynamicImage, Luma};
use std::fs;
use std::path::PathBuf;
use thiserror::Error;

fn expand_to_16bit(buffer: &mut [u16], is12_bit: bool) {
    if is12_bit {
        for px in buffer.iter_mut() {
            *px = (*px << 4) | (*px >> 8);
        }
    } else {
        for px in buffer.iter_mut() {
            *px = (*px << 6) | (*px >> 4);
        }
    }
}

#[derive(Error, Debug)]
pub enum HeicError {
    #[error("An error happened while decoding: heic`{0}`")]
    Format(String),
    #[error("Cannot read file due to an error:`{0}`")]
    Io(String),
    #[error("Lossless allowed only on 4:4:4 chroma but it was: `{0:?}`")]
    Lossless444(ChromaFormat),
}

fn plane_stride<T>(plane: &hpvcd::PlaneBuffer<T>) -> Result<u32, HeicError> {
    u32::try_from(plane.stride())
        .map_err(|_| HeicError::Format("HEIC plane stride exceeds u32".into()))
}

fn tight_plane<T: Copy>(plane: &hpvcd::PlaneBuffer<T>) -> Vec<T> {
    plane.rows().flatten().copied().collect()
}

pub(crate) fn decode_heic_file_url(file: &PathBuf) -> Result<DynamicImage, HeicError> {
    use hpvcd::{ChromaFormat, MatrixCoefficients};
    use yuv::{
        YuvPlanarImage, YuvPlanarImageWithAlpha, YuvRange, YuvStandardMatrix, i010_alpha_to_rgba10,
        i010_to_rgb10, i012_alpha_to_rgba12, i012_to_rgb12, i210_alpha_to_rgba10, i210_to_rgb10,
        i212_alpha_to_rgba12, i212_to_rgb12, i410_alpha_to_rgba10, i410_to_rgb10,
        i412_alpha_to_rgba12, i412_to_rgb12, icgc010_alpha_to_rgba10, icgc010_to_rgb10,
        icgc012_alpha_to_rgba12, icgc012_to_rgb12, icgc210_alpha_to_rgba10, icgc210_to_rgb10,
        icgc212_alpha_to_rgba12, icgc212_to_rgb12, icgc410_alpha_to_rgba10, icgc410_to_rgb10,
        icgc412_alpha_to_rgba12, icgc412_to_rgb12, ycgco420_alpha_to_rgba, ycgco420_to_rgb,
        ycgco422_alpha_to_rgba, ycgco422_to_rgb, ycgco444_alpha_to_rgba, ycgco444_to_rgb,
        yuv420_alpha_to_rgba, yuv420_to_rgb, yuv422_alpha_to_rgba, yuv422_to_rgb,
        yuv444_alpha_to_rgba, yuv444_to_rgb,
    };

    let dec = hpvcd::decode_heic_yuv(&fs::read(file).map_err(|x| HeicError::Io(x.to_string()))?)
        .map_err(|e| HeicError::Format(format!("hpvcd: {e}")))?;

    let w = u32::try_from(dec.width())
        .map_err(|_| HeicError::Format("HEIC width exceeds u32".into()))?;
    let h = u32::try_from(dec.height())
        .map_err(|_| HeicError::Format("HEIC height exceeds u32".into()))?;
    let bit_depth = dec.bit_depth.bits() as u32;
    let high_bit = bit_depth > 8;
    let is_12 = bit_depth >= 12;

    // Chroma subsampling factors select the conversion routine. Plane strides
    // come from hpvcd because decoded images may retain coded padding/crops.
    let (sub_w, sub_h) = match dec.chroma {
        ChromaFormat::Yuv420 => (2u32, 2u32),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => (1, 1),
    };

    let mut matrix = YuvStandardMatrix::Bt601;
    let mut range = YuvRange::Limited;
    let mut is_ycgco = false;
    if let Some(enc) = &dec.color.cicp {
        matrix = match enc.matrix {
            MatrixCoefficients::Smpte170m => YuvStandardMatrix::Bt601,
            MatrixCoefficients::Bt709 => YuvStandardMatrix::Bt709,
            MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => {
                YuvStandardMatrix::Bt2020
            }
            MatrixCoefficients::YCgCo
            | MatrixCoefficients::YCgCoRe
            | MatrixCoefficients::YCgCoRo => {
                is_ycgco = true;
                YuvStandardMatrix::Bt601
            }
            _ => YuvStandardMatrix::Bt601,
        };
        range = if enc.full_range {
            YuvRange::Full
        } else {
            YuvRange::Limited
        };
    }

    if dec.chroma == ChromaFormat::Monochrome {
        return finish_monochrome(&dec, w, h, high_bit, is_12);
    }

    let rgb_stride = w * 3;
    let rgba_stride = w * 4;
    let has_alpha = dec.alpha.is_some();

    let img = if high_bit {
        let planes = dec.planes.as_u16().ok_or_else(|| {
            HeicError::Format("HEIC RGBA10/RGBA12 planes are not actually 10/12-bit".into())
        })?;
        let cb = planes.cb.as_ref().ok_or_else(|| {
            HeicError::Format("HEIC RGBA10/RGBA12 is missing the Cb plane".into())
        })?;
        let cr = planes.cr.as_ref().ok_or_else(|| {
            HeicError::Format("HEIC RGBA10/RGBA12 is missing the Cr plane".into())
        })?;
        let y16 = planes.y.data();
        let cb16 = cb.data();
        let cr16 = cr.data();
        let y_stride = plane_stride(&planes.y)?;
        let cb_stride = plane_stride(cb)?;
        let cr_stride = plane_stride(cr)?;

        if is_ycgco {
            if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u16()) {
                let yuva = YuvPlanarImageWithAlpha {
                    y_plane: y16,
                    y_stride,
                    u_plane: cb16,
                    u_stride: cb_stride,
                    v_plane: cr16,
                    v_stride: cr_stride,
                    a_plane: a.data(),
                    a_stride: plane_stride(a)?,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 4 * h) as usize];
                match (sub_w, sub_h, is_12) {
                    (2, 2, false) => icgc010_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (2, 1, false) => icgc210_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (_, _, false) => icgc410_alpha_to_rgba10(&yuva, &mut out, w * 4, range),
                    (2, 2, true) => icgc012_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                    (2, 1, true) => icgc212_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                    (_, _, true) => icgc412_alpha_to_rgba12(&yuva, &mut out, w * 4, range),
                }
                .map_err(|e| HeicError::Format(format!("icgc→rgba16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgba16_image(w, h, out)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y16,
                    y_stride,
                    u_plane: cb16,
                    u_stride: cb_stride,
                    v_plane: cr16,
                    v_stride: cr_stride,
                    width: w,
                    height: h,
                };
                let mut out = vec![0u16; (w * 3 * h) as usize];
                match (sub_w, sub_h, is_12) {
                    (2, 2, false) => icgc010_to_rgb10(&yuv, &mut out, w * 3, range),
                    (2, 1, false) => icgc210_to_rgb10(&yuv, &mut out, w * 3, range),
                    (_, _, false) => icgc410_to_rgb10(&yuv, &mut out, w * 3, range),
                    (2, 2, true) => icgc012_to_rgb12(&yuv, &mut out, w * 3, range),
                    (2, 1, true) => icgc212_to_rgb12(&yuv, &mut out, w * 3, range),
                    (_, _, true) => icgc412_to_rgb12(&yuv, &mut out, w * 3, range),
                }
                .map_err(|e| HeicError::Format(format!("icgc→rgb16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgb16_image(w, h, out)?
            }
        } else if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u16()) {
            let yuva = YuvPlanarImageWithAlpha {
                y_plane: y16,
                y_stride,
                u_plane: cb16,
                u_stride: cb_stride,
                v_plane: cr16,
                v_stride: cr_stride,
                a_plane: a.data(),
                a_stride: plane_stride(a)?,
                width: w,
                height: h,
            };
            let mut out = vec![0u16; (w * 4 * h) as usize];
            match (sub_w, sub_h, is_12) {
                (2, 2, false) => i010_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (2, 1, false) => i210_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (_, _, false) => i410_alpha_to_rgba10(&yuva, &mut out, w * 4, range, matrix),
                (2, 2, true) => i012_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
                (2, 1, true) => i212_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
                (_, _, true) => i412_alpha_to_rgba12(&yuva, &mut out, w * 4, range, matrix),
            }
            .map_err(|e| HeicError::Format(format!("yuv→rgba16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgba16_image(w, h, out)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y16,
                y_stride,
                u_plane: cb16,
                u_stride: cb_stride,
                v_plane: cr16,
                v_stride: cr_stride,
                width: w,
                height: h,
            };
            let mut out = vec![0u16; (w * 3 * h) as usize];
            match (sub_w, sub_h, is_12) {
                (2, 2, false) => i010_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (2, 1, false) => i210_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (_, _, false) => i410_to_rgb10(&yuv, &mut out, w * 3, range, matrix),
                (2, 2, true) => i012_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (2, 1, true) => i212_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
                (_, _, true) => i412_to_rgb12(&yuv, &mut out, w * 3, range, matrix),
            }
            .map_err(|e| HeicError::Format(format!("yuv→rgb16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgb16_image(w, h, out)?
        }
    } else {
        let planes = dec
            .planes
            .as_u8()
            .ok_or_else(|| HeicError::Format("HEIC RGBA8 planes are not actually 8-bit".into()))?;
        let cb = planes
            .cb
            .as_ref()
            .ok_or_else(|| HeicError::Format("HEIC RGBA8 is missing the Cb plane".into()))?;
        let cr = planes
            .cr
            .as_ref()
            .ok_or_else(|| HeicError::Format("HEIC RGBA8 is missing the Cr plane".into()))?;
        let y8 = planes.y.data();
        let cb8 = cb.data();
        let cr8 = cr.data();
        let y_stride = plane_stride(&planes.y)?;
        let cb_stride = plane_stride(cb)?;
        let cr_stride = plane_stride(cr)?;

        if is_ycgco {
            if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u8()) {
                let yuv = YuvPlanarImageWithAlpha {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: cb_stride,
                    v_plane: cr8,
                    v_stride: cr_stride,
                    a_plane: a.data(),
                    a_stride: plane_stride(a)?,
                    width: w,
                    height: h,
                };
                let mut rgba = vec![0u8; (rgba_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    (2, 1) => ycgco422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    _ => ycgco444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                }
                .map_err(|e| HeicError::Format(format!("ycgco→rgba: {e}")))?;
                rgba8_image(w, h, rgba)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: cb_stride,
                    v_plane: cr8,
                    v_stride: cr_stride,
                    width: w,
                    height: h,
                };
                let mut rgb = vec![0u8; (rgb_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    (2, 1) => ycgco422_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    _ => ycgco444_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                }
                .map_err(|e| HeicError::Format(format!("ycgco→rgb: {e}")))?;
                rgb8_image(w, h, rgb)?
            }
        } else if let Some(a) = dec.alpha.as_ref().and_then(|a| a.as_u8()) {
            let yuv = YuvPlanarImageWithAlpha {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: cb_stride,
                v_plane: cr8,
                v_stride: cr_stride,
                a_plane: a.data(),
                a_stride: plane_stride(a)?,
                width: w,
                height: h,
            };
            let mut rgba = vec![0u8; (rgba_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                (2, 1) => yuv422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                _ => yuv444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
            }
            .map_err(|e| HeicError::Format(format!("yuv→rgba: {e}")))?;
            rgba8_image(w, h, rgba)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: cb_stride,
                v_plane: cr8,
                v_stride: cr_stride,
                width: w,
                height: h,
            };
            let mut rgb = vec![0u8; (rgb_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                (2, 1) => yuv422_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                _ => yuv444_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
            }
            .map_err(|e| HeicError::Format(format!("yuv→rgb: {e}")))?;
            rgb8_image(w, h, rgb)?
        }
    };

    let _ = has_alpha; // (kept for parity/readability)
    Ok(apply_orientation(img, dec.orientation))
}

fn rgb8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, HeicError> {
    Ok(DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| HeicError::Format("HEIC RGB mismatch".into()))?,
    ))
}
fn rgba8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, HeicError> {
    Ok(DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(w, h, buf)
            .ok_or_else(|| HeicError::Format("HEIC RGBA mismatch".into()))?,
    ))
}
fn rgb16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, HeicError> {
    Ok(DynamicImage::ImageRgb16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| HeicError::Format("HEIC RGB16 mismatch".into()))?,
    ))
}
fn rgba16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, HeicError> {
    Ok(DynamicImage::ImageRgba16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| HeicError::Format("HEIC RGBA16 mismatch".into()))?,
    ))
}

/// 4:0:0 monochrome: replicate the luma plane across R, G, B.
fn finish_monochrome(
    dec: &hpvcd::DecodedYuv,
    w: u32,
    h: u32,
    high_bit: bool,
    is_12: bool,
) -> Result<DynamicImage, HeicError> {
    let img = if high_bit {
        let planes = dec
            .planes
            .as_u16()
            .ok_or_else(|| HeicError::Format("HEIC monochrome luma is not 10/12-bit".into()))?;
        let mut y = tight_plane(&planes.y);
        if let Some(alpha) = dec.alpha.as_ref().and_then(|a| a.as_u16()) {
            let alpha = tight_plane(alpha);
            if y.len() != alpha.len() {
                return Err(HeicError::Format(
                    "HEIC monochrome alpha dimensions do not match luma".into(),
                ));
            }
            let mut ya = vec![0u16; y.len() * 2];
            for (dst, (y, alpha)) in ya
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(y.into_iter().zip(alpha))
            {
                dst[0] = y;
                dst[1] = alpha;
            }
            expand_to_16bit(&mut ya, is_12);
            DynamicImage::ImageLumaA16(
                image::ImageBuffer::from_raw(w, h, ya)
                    .ok_or_else(|| HeicError::Format("HEIC LumaA16 mismatch".into()))?,
            )
        } else {
            expand_to_16bit(&mut y, is_12);
            DynamicImage::ImageLuma16(
                image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, y)
                    .ok_or_else(|| HeicError::Format("HEIC Luma16 mismatch".into()))?,
            )
        }
    } else {
        let planes = dec
            .planes
            .as_u8()
            .ok_or_else(|| HeicError::Format("HEIC monochrome luma is not 8-bit".into()))?;
        let y = tight_plane(&planes.y);
        if let Some(alpha) = dec.alpha.as_ref().and_then(|a| a.as_u8()) {
            let alpha = tight_plane(alpha);
            if y.len() != alpha.len() {
                return Err(HeicError::Format(
                    "HEIC monochrome alpha dimensions do not match luma".into(),
                ));
            }
            let mut ya = vec![0u8; y.len() * 2];
            for (dst, (y, alpha)) in ya
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(y.into_iter().zip(alpha))
            {
                dst[0] = y;
                dst[1] = alpha;
            }
            DynamicImage::ImageLumaA8(
                image::ImageBuffer::from_raw(w, h, ya)
                    .ok_or_else(|| HeicError::Format("HEIC LumaA8 mismatch".into()))?,
            )
        } else {
            DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(w, h, y)
                    .ok_or_else(|| HeicError::Format("HEIC Luma8 mismatch".into()))?,
            )
        }
    };
    Ok(apply_orientation(img, dec.orientation))
}

pub(crate) fn encode_hevc(
    img: &DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, anyhow::Error> {
    let chroma_fmt = match args.chroma.unwrap_or(Chroma::C420) {
        Chroma::C444 => ChromaFormat::Yuv444,
        Chroma::C422 => ChromaFormat::Yuv422,
        Chroma::C420 => ChromaFormat::Yuv420,
    };

    if args.lossless && chroma_fmt != ChromaFormat::Yuv444 {
        return Err(anyhow::anyhow!(HeicError::Lossless444(chroma_fmt)));
    }

    let mut cfg = hpvca::EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_threads(args.threads);

    if let Some(icc) = icc {
        cfg = cfg.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = exif {
        cfg = cfg.with_exif(exif.to_vec());
    }

    if args.lossless {
        cfg = cfg.with_lossless(true);
    }

    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;
    Ok(match (effective_depth, gray, alpha) {
        (Depth::D8, true, _) => {
            hpvca::encode_gray(img.to_luma8().as_raw(), img.width(), img.height(), &cfg)?
        }
        (Depth::D8, false, false) => {
            hpvca::encode_rgb(img.to_rgb8().as_raw(), img.width(), img.height(), &cfg)?
        }
        (Depth::D8, false, true) => {
            hpvca::encode_rgba_with_alpha(img.to_rgba8().as_raw(), img.width(), img.height(), &cfg)?
        }
        (Depth::D10, true, _) => hpvca::encode_gray10(
            &scale16_to_10(img.to_luma16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D10, false, false) => hpvca::encode_rgb10(
            &scale16_to_10(img.to_rgb16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D10, false, true) => hpvca::encode_rgba10_with_alpha(
            &scale16_to_10(img.to_rgba16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, true, _) => hpvca::encode_gray12(
            &scale16_to_12(img.to_luma16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, false, false) => hpvca::encode_rgb12(
            &scale16_to_12(img.to_rgb16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, false, true) => hpvca::encode_rgba12_with_alpha(
            &scale16_to_12(img.to_rgba16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
    })
}
