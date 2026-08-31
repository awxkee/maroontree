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
use crate::orientation::apply_orientation_vvc;
use crate::{Args, Chroma, Depth, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12};
use garnetash::{ChromaFormat, DecodedImage, MatrixCoefficients};
use image::{DynamicImage, Luma};
use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use thiserror::Error;

pub(crate) fn expand_to_16bit(buffer: &mut [u16], is12_bit: bool) {
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
pub enum VvcError {
    #[error("An error happened while decoding: heic`{0}`")]
    Format(String),
    #[error("Cannot read file due to an error:`{0}`")]
    Io(String),
}

/// Chroma subsampling factors `(SubWidthC, SubHeightC)` for a format.
fn sub_factors(chroma: ChromaFormat) -> (u32, u32) {
    match chroma {
        ChromaFormat::Yuv420 => (2, 2),
        ChromaFormat::Yuv422 => (2, 1),
        ChromaFormat::Yuv444 | ChromaFormat::Monochrome => (1, 1),
    }
}

/// Byte ranges of the Y, Cb and Cr planes inside garnetash's packed `planes`
/// buffer. garnetash stores the luma plane (`width × height`) first, then — for
/// non-monochrome — Cb and Cr at chroma resolution, with each sample one byte at
/// 8-bit and a little-endian `u16` at 10/12-bit. `bps` is the bytes-per-sample.
fn plane_ranges(
    width: u32,
    height: u32,
    chroma: ChromaFormat,
    bps: usize,
) -> (Range<usize>, Range<usize>, Range<usize>) {
    let (sub_w, sub_h) = sub_factors(chroma);
    let cw = width.div_ceil(sub_w) as usize;
    let ch = height.div_ceil(sub_h) as usize;
    let y_len = width as usize * height as usize * bps;
    let c_len = cw * ch * bps;
    (
        0..y_len,
        y_len..y_len + c_len,
        y_len + c_len..y_len + 2 * c_len,
    )
}

/// Reinterpret a little-endian `u16` byte plane (as garnetash packs >8-bit
/// samples) into a `Vec<u16>` the `yuv` crate can consume.
pub(crate) fn le_u16(bytes: &[u8]) -> Vec<u16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}

pub(crate) fn decode_heic_vvc_file_url(file: &PathBuf) -> Result<DynamicImage, VvcError> {
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

    // garnetash decodes the HEIF (or raw `.266`) directly to planar YCbCr.
    let dec = garnetash::decode(&fs::read(file).map_err(|x| VvcError::Io(x.to_string()))?)
        .map_err(|e| VvcError::Format(format!("garnetash: {e}")))?;

    let w = dec.width;
    let h = dec.height;
    let bit_depth = dec.bit_depth.bits() as u32;
    let high_bit = bit_depth > 8;
    let is_12 = bit_depth >= 12;

    // Chroma subsampling factors → tight plane strides.
    let (sub_w, sub_h) = sub_factors(dec.chroma);
    let y_stride = w;
    let c_stride = w.div_ceil(sub_w);

    let mut matrix = YuvStandardMatrix::Bt601;
    let mut range = YuvRange::Full;
    let mut is_ycgco = false;
    if let Some(enc) = &dec.color.cicp {
        matrix = match enc.matrix {
            MatrixCoefficients::Bt709 => YuvStandardMatrix::Bt709,
            MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => {
                YuvStandardMatrix::Bt2020
            }
            MatrixCoefficients::YCgCo => {
                is_ycgco = true;
                YuvStandardMatrix::Bt601
            }
            // Smpte170m / Bt470Bg / Fcc / Smpte240m / Identity / Unspecified / …
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
    let bps = if high_bit { 2 } else { 1 };
    let (yr, cbr, crr) = plane_ranges(w, h, dec.chroma, bps);

    let img = if high_bit {
        // Unpack the LE-u16 planes the `yuv` crate expects.
        let y16 = le_u16(&dec.planes[yr]);
        let cb16 = le_u16(&dec.planes[cbr]);
        let cr16 = le_u16(&dec.planes[crr]);
        let alpha16: Option<Vec<u16>> = dec.alpha.as_ref().map(|a| le_u16(&a.planes));

        if is_ycgco {
            if let Some(a) = alpha16.as_deref() {
                let yuva = YuvPlanarImageWithAlpha {
                    y_plane: &y16,
                    y_stride,
                    u_plane: &cb16,
                    u_stride: c_stride,
                    v_plane: &cr16,
                    v_stride: c_stride,
                    a_plane: a,
                    a_stride: y_stride,
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
                .map_err(|e| VvcError::Format(format!("icgc→rgba16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgba16_image(w, h, out)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: &y16,
                    y_stride,
                    u_plane: &cb16,
                    u_stride: c_stride,
                    v_plane: &cr16,
                    v_stride: c_stride,
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
                .map_err(|e| VvcError::Format(format!("icgc→rgb16: {e}")))?;
                expand_to_16bit(&mut out, is_12);
                rgb16_image(w, h, out)?
            }
        } else if let Some(a) = alpha16.as_deref() {
            let yuva = YuvPlanarImageWithAlpha {
                y_plane: &y16,
                y_stride,
                u_plane: &cb16,
                u_stride: c_stride,
                v_plane: &cr16,
                v_stride: c_stride,
                a_plane: a,
                a_stride: y_stride,
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
            .map_err(|e| VvcError::Format(format!("yuv→rgba16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgba16_image(w, h, out)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: &y16,
                y_stride,
                u_plane: &cb16,
                u_stride: c_stride,
                v_plane: &cr16,
                v_stride: c_stride,
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
            .map_err(|e| VvcError::Format(format!("yuv→rgb16: {e}")))?;
            expand_to_16bit(&mut out, is_12);
            rgb16_image(w, h, out)?
        }
    } else {
        // 8-bit planes can be sliced straight out of the packed buffer.
        let y8 = &dec.planes[yr];
        let cb8 = &dec.planes[cbr];
        let cr8 = &dec.planes[crr];
        let alpha8: Option<&[u8]> = dec.alpha.as_ref().map(|a| a.planes.as_slice());

        if is_ycgco {
            if let Some(a) = alpha8 {
                let yuv = YuvPlanarImageWithAlpha {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: c_stride,
                    v_plane: cr8,
                    v_stride: c_stride,
                    a_plane: a,
                    a_stride: y_stride,
                    width: w,
                    height: h,
                };
                let mut rgba = vec![0u8; (rgba_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    (2, 1) => ycgco422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    _ => ycgco444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                }
                .map_err(|e| VvcError::Format(format!("ycgco→rgba: {e}")))?;
                rgba8_image(w, h, rgba)?
            } else {
                let yuv = YuvPlanarImage {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: c_stride,
                    v_plane: cr8,
                    v_stride: c_stride,
                    width: w,
                    height: h,
                };
                let mut rgb = vec![0u8; (rgb_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    (2, 1) => ycgco422_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                    _ => ycgco444_to_rgb(&yuv, &mut rgb, rgb_stride, range),
                }
                .map_err(|e| VvcError::Format(format!("ycgco→rgb: {e}")))?;
                rgb8_image(w, h, rgb)?
            }
        } else if let Some(a) = alpha8 {
            let yuv = YuvPlanarImageWithAlpha {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: c_stride,
                v_plane: cr8,
                v_stride: c_stride,
                a_plane: a,
                a_stride: y_stride,
                width: w,
                height: h,
            };
            let mut rgba = vec![0u8; (rgba_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                (2, 1) => yuv422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                _ => yuv444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
            }
            .map_err(|e| VvcError::Format(format!("yuv→rgba: {e}")))?;
            rgba8_image(w, h, rgba)?
        } else {
            let yuv = YuvPlanarImage {
                y_plane: y8,
                y_stride,
                u_plane: cb8,
                u_stride: c_stride,
                v_plane: cr8,
                v_stride: c_stride,
                width: w,
                height: h,
            };
            let mut rgb = vec![0u8; (rgb_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                (2, 1) => yuv422_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
                _ => yuv444_to_rgb(&yuv, &mut rgb, rgb_stride, range, matrix),
            }
            .map_err(|e| VvcError::Format(format!("yuv→rgb: {e}")))?;
            rgb8_image(w, h, rgb)?
        }
    };

    Ok(apply_orientation_vvc(img, dec.orientation))
}

fn rgb8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, VvcError> {
    Ok(DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| VvcError::Format("HEIC RGB mismatch".into()))?,
    ))
}
fn rgba8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, VvcError> {
    Ok(DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(w, h, buf)
            .ok_or_else(|| VvcError::Format("HEIC RGBA mismatch".into()))?,
    ))
}
fn rgb16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, VvcError> {
    Ok(DynamicImage::ImageRgb16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| VvcError::Format("HEIC RGB16 mismatch".into()))?,
    ))
}
fn rgba16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, VvcError> {
    Ok(DynamicImage::ImageRgba16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| VvcError::Format("HEIC RGBA16 mismatch".into()))?,
    ))
}

/// 4:0:0 monochrome: the luma plane is the image. garnetash carries the luma in
/// `planes` (the whole buffer, no chroma) and any alpha as a separate monochrome
/// `DecodedImage` in `dec.alpha`.
fn finish_monochrome(
    dec: &DecodedImage,
    w: u32,
    h: u32,
    high_bit: bool,
    is_12: bool,
) -> Result<DynamicImage, VvcError> {
    let img = if high_bit {
        let mut y = le_u16(&dec.planes);
        expand_to_16bit(&mut y, is_12);
        DynamicImage::ImageLuma16(
            image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, y)
                .ok_or_else(|| VvcError::Format("HEIC Luma16 mismatch".into()))?,
        )
    } else if let Some(alpha) = dec.alpha.as_ref() {
        // Interleave luma + alpha into a LumaA8 image.
        let y = &dec.planes;
        let a = &alpha.planes;
        let mut ya = vec![0u8; w as usize * h as usize * 2];
        for ((dst, &yv), &av) in ya
            .as_chunks_mut::<2>()
            .0
            .iter_mut()
            .zip(y.iter())
            .zip(a.iter())
        {
            dst[0] = yv;
            dst[1] = av;
        }
        DynamicImage::ImageLumaA8(
            image::GrayAlphaImage::from_raw(w, h, ya)
                .ok_or_else(|| VvcError::Format("HEIC LumaA8 mismatch".into()))?,
        )
    } else {
        DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, dec.planes.clone())
                .ok_or_else(|| VvcError::Format("HEIC Luma8 mismatch".into()))?,
        )
    };
    Ok(apply_orientation_vvc(img, dec.orientation))
}

fn garnetash_cicp(cicp: crate::PngCicp) -> garnetash::Cicp {
    garnetash::Cicp {
        primaries: garnetash::Primaries::from_u16(cicp.color_primaries.into()),
        transfer: garnetash::TransferFunction::from_u16(cicp.transfer_function.into()),
        matrix: garnetash::MatrixCoefficients::Smpte170m,
        full_range: true,
    }
}

pub(crate) fn encode_vvc(
    img: &DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    png_cicp: Option<crate::PngCicp>,
) -> Result<Vec<u8>, anyhow::Error> {
    let chroma_fmt = match args.chroma.unwrap_or(Chroma::C420) {
        Chroma::C444 => garnetash::ChromaFormat::Yuv444,
        Chroma::C422 => garnetash::ChromaFormat::Yuv422,
        Chroma::C420 => garnetash::ChromaFormat::Yuv420,
    };

    let mut cfg = garnetash::EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_threads(args.threads)
        .with_aq(true)
        .with_mtt(true)
        .with_lfnst(true)
        .with_dep_quant(true)
        .with_mts(true)
        .with_dual_tree(true)
        .with_cclm(true)
        .with_deblocking(true);

    if let Some(cicp) = png_cicp {
        cfg = cfg.with_cicp(garnetash_cicp(cicp));
    }

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
            garnetash::encode_gray(img.to_luma8().as_raw(), img.width(), img.height(), &cfg)?
        }
        (Depth::D8, false, false) => {
            garnetash::encode_rgb(img.to_rgb8().as_raw(), img.width(), img.height(), &cfg)?
        }
        (Depth::D8, false, true) => garnetash::encode_rgba_with_alpha(
            img.to_rgba8().as_raw(),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D10, true, _) => garnetash::encode_gray10(
            &scale16_to_10(img.to_luma16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D10, false, false) => garnetash::encode_rgb10(
            &scale16_to_10(img.to_rgb16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D10, false, true) => garnetash::encode_rgba10_with_alpha(
            &scale16_to_10(img.to_rgba16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, true, _) => garnetash::encode_gray12(
            &scale16_to_12(img.to_luma16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, false, false) => garnetash::encode_rgb12(
            &scale16_to_12(img.to_rgb16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
        (Depth::D12, false, true) => garnetash::encode_rgba12_with_alpha(
            &scale16_to_12(img.to_rgba16().as_raw()),
            img.width(),
            img.height(),
            &cfg,
        )?,
    })
}
