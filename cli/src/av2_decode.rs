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
use crate::orientation::apply_orientation_tealdust;
use image::{DynamicImage, Luma};
use std::fs;
use std::hint::black_box;
use std::path::PathBuf;
use tealdust::{
    AvifImage, AvifSettings, ColorInfo, ColorPrimaries, MatrixCoefficients, Orientation,
    PixelLayout, TransferCharacteristics,
};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AvifError {
    #[error("An error happened while decoding: heic`{0}`")]
    Format(String),
    #[error("Cannot read file due to an error:`{0}`")]
    Io(String),
}

struct FinalizedView<T> {
    data: Vec<T>,
    width: usize,
    height: usize,
}

#[derive(Debug, Copy, Clone)]
enum Subsampling {
    Full,
    Sampled,
}

fn le_u16(bytes: &[u8], width: u32, stride: u32, sampling: Subsampling) -> Vec<u16> {
    bytes
        .chunks_exact(stride as usize)
        .flat_map(|row| {
            row[..match sampling {
                Subsampling::Full => width as usize,
                Subsampling::Sampled => (width as usize).div_ceil(2),
            } * 2]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
        })
        .collect()
}

fn finalize<T: Copy + Default, const N: usize>(
    mut data: Vec<T>,
    avif_image: &AvifImage,
) -> Result<FinalizedView<T>, AvifError> {
    let mut width = avif_image.width as usize;
    let mut height = avif_image.height as usize;

    // clap is defined in coded space, so crop BEFORE orientation (clap → irot → imir).
    if let Some(clap) = avif_image.clean_aperture.as_ref()
        && let Some((left, top, cw, ch)) = clap.to_crop_rect(avif_image.width, avif_image.height)
        && (cw as usize != width || ch as usize != height)
    {
        let (src_stride, dst_stride) = (width * N, cw as usize * N);
        let mut cropped = vec![T::default(); cw as usize * ch as usize * N];
        for row in 0..ch {
            let s = (top + row) as usize * src_stride + left as usize * N;
            let d = row as usize * dst_stride;
            cropped[d..d + dst_stride].copy_from_slice(&data[s..s + dst_stride]);
        }
        data = cropped;
        width = cw as usize;
        height = ch as usize;
    }

    Ok(FinalizedView {
        data,
        width,
        height,
    })
}

pub(crate) fn decode_av2_file_url(file: &PathBuf) -> Result<DynamicImage, AvifError> {
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

    let data_vec = fs::read(file).map_err(|x| AvifError::Io(x.to_string()))?;
    let mut settings = AvifSettings::default();
    settings.decoder_settings.frame_size_limit = 8096 * 8096;
    let mut decoder = tealdust::AvifDecoder::with_settings(&data_vec, settings)
        .map_err(|e| AvifError::Format(format!("garnetash: {e}")))?;

    let image_info = decoder
        .image_info()
        .map_err(|e| AvifError::Io(e.to_string()))?;
    let image = decoder.decode().map_err(|e| AvifError::Io(e.to_string()))?;
    let _ = black_box(decoder.decode().map_err(|e| AvifError::Io(e.to_string()))?);

    let w = image_info.width;
    let h = image_info.height;
    let (wu, hu) = (w as usize, h as usize);
    let bit_depth = image_info.bits_per_component as u32;
    let high_bit = bit_depth > 8;
    let is_12 = bit_depth >= 12;

    // Chroma subsampling factors → tight plane strides.
    let mut y_stride = image.strides[0] as u32;

    let mut is_ycgco = false;
    let cicp = image_info.color_info.unwrap_or(ColorInfo {
        color_primaries: ColorPrimaries::Bt709,
        matrix_coefficients: MatrixCoefficients::Smpte240,
        transfer_characteristics: TransferCharacteristics::Srgb,
        full_range: true,
    });
    let is_identity = cicp.matrix_coefficients == MatrixCoefficients::Identity;
    let matrix = match cicp.matrix_coefficients {
        MatrixCoefficients::Bt709 => YuvStandardMatrix::Bt709,
        MatrixCoefficients::Bt2020Ncl | MatrixCoefficients::Bt2020Cl => YuvStandardMatrix::Bt2020,
        MatrixCoefficients::YCgCo => {
            is_ycgco = true;
            YuvStandardMatrix::Bt601
        }
        _ => YuvStandardMatrix::Bt601,
    };
    let range = if cicp.full_range {
        YuvRange::Full
    } else {
        YuvRange::Limited
    };

    if image_info.pixel_layout == PixelLayout::I400 {
        return finish_monochrome(&image, w, h, high_bit, is_12);
    }

    let rgb_stride = w * 3;
    let rgba_stride = w * 4;
    let sub_w = if image_info.pixel_layout == PixelLayout::I420
        || image_info.pixel_layout == PixelLayout::I422
    {
        2
    } else {
        1
    };
    let sub_h = if image_info.pixel_layout == PixelLayout::I420 {
        2
    } else {
        1
    };
    let sampled_horizontally = image_info.pixel_layout == PixelLayout::I420
        || image_info.pixel_layout == PixelLayout::I422;

    let mut c_stride = image.strides[1] as u32;

    let img = if high_bit {
        let y16 = le_u16(&image.planes[0], w, y_stride, Subsampling::Full);
        let cb16 = le_u16(
            &image.planes[1],
            w,
            c_stride,
            if sampled_horizontally {
                Subsampling::Sampled
            } else {
                Subsampling::Full
            },
        );
        let cr16 = le_u16(
            &image.planes[2],
            w,
            c_stride,
            if sampled_horizontally {
                Subsampling::Sampled
            } else {
                Subsampling::Full
            },
        );
        let alpha16: Option<Vec<u16>> = image
            .alpha
            .as_ref()
            .map(|a| le_u16(&a.data, w, a.stride as u32, Subsampling::Full));
        c_stride = w.div_ceil(sub_w);
        y_stride = w;

        if is_identity {
            if image_info.pixel_layout != PixelLayout::I444 {
                return Err(AvifError::Format(
                    "identity matrix requires 4:4:4 chroma".into(),
                ));
            }
            if let Some(a) = alpha16.as_deref() {
                let mut out = Vec::with_capacity(wu * hu * 4);
                for (((&g, &b), &r), &alpha) in y16.iter().zip(&cb16).zip(&cr16).zip(a) {
                    out.extend_from_slice(&[r, g, b, alpha]);
                }
                let mut final_view = finalize::<u16, 4>(out, &image)?;
                crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
                rgba16_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
            } else {
                let mut out = Vec::with_capacity(wu * hu * 3);
                for ((&g, &b), &r) in y16.iter().zip(&cb16).zip(&cr16) {
                    out.extend_from_slice(&[r, g, b]);
                }
                let mut final_view = finalize::<u16, 3>(out, &image)?;
                crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
                rgb16_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
            }
        } else if is_ycgco {
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
                .map_err(|e| AvifError::Format(format!("icgc→rgba16: {e}")))?;
                let mut final_view = finalize::<u16, 4>(out, &image)?;
                crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
                rgba16_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
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
                .map_err(|e| AvifError::Format(format!("icgc→rgb16: {e}")))?;
                let mut final_view = finalize::<u16, 3>(out, &image)?;
                crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
                rgb16_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
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
            .map_err(|e| AvifError::Format(format!("yuv→rgba16: {e}")))?;
            let mut final_view = finalize::<u16, 4>(out, &image)?;
            crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
            rgba16_image(
                final_view.width as u32,
                final_view.height as u32,
                final_view.data,
            )?
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
            .map_err(|e| AvifError::Format(format!("yuv→rgb16: {e}")))?;
            let mut final_view = finalize::<u16, 3>(out, &image)?;
            crate::vvc::expand_to_16bit(&mut final_view.data, is_12);
            rgb16_image(
                final_view.width as u32,
                final_view.height as u32,
                final_view.data,
            )?
        }
    } else {
        // 8-bit planes can be sliced straight out of the packed buffer.
        let y8 = &image.planes[0];
        let cb8 = &image.planes[1];
        let cr8 = &image.planes[2];
        let alpha8: Option<&[u8]> = image.alpha.as_ref().map(|a| a.data.as_slice());

        if is_identity {
            if image_info.pixel_layout != PixelLayout::I444 {
                return Err(AvifError::Format(
                    "identity matrix requires 4:4:4 chroma".into(),
                ));
            }
            if let Some(a) = alpha8 {
                let a_stride = image.alpha.as_ref().map(|x| x.stride).unwrap_or(w as usize);
                let mut rgba = Vec::with_capacity(wu * hu * 4);
                for row in 0..hu {
                    let (yo, co, ao) = (
                        row * y_stride as usize,
                        row * c_stride as usize,
                        row * a_stride,
                    );
                    for x in 0..wu {
                        rgba.extend_from_slice(&[cr8[co + x], y8[yo + x], cb8[co + x], a[ao + x]]);
                    }
                }
                let final_view = finalize::<u8, 4>(rgba, &image)?;
                rgba8_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
            } else {
                let mut rgb = Vec::with_capacity(wu * hu * 3);
                for row in 0..hu {
                    let (yo, co) = (row * y_stride as usize, row * c_stride as usize);
                    for x in 0..wu {
                        rgb.extend_from_slice(&[cr8[co + x], y8[yo + x], cb8[co + x]]);
                    }
                }
                let final_view = finalize::<u8, 3>(rgb, &image)?;
                rgb8_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
            }
        } else if is_ycgco {
            if let Some(a) = alpha8 {
                let yuv = YuvPlanarImageWithAlpha {
                    y_plane: y8,
                    y_stride,
                    u_plane: cb8,
                    u_stride: c_stride,
                    v_plane: cr8,
                    v_stride: c_stride,
                    a_plane: a,
                    a_stride: image.alpha.as_ref().map(|x| x.stride as u32).unwrap_or(w),
                    width: w,
                    height: h,
                };
                let mut rgba = vec![0u8; (rgba_stride * h) as usize];
                match (sub_w, sub_h) {
                    (2, 2) => ycgco420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    (2, 1) => ycgco422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                    _ => ycgco444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range),
                }
                .map_err(|e| AvifError::Format(format!("ycgco→rgba: {e}")))?;
                let final_view = finalize::<u8, 4>(rgba, &image)?;
                rgba8_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
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
                .map_err(|e| AvifError::Format(format!("ycgco→rgb: {e}")))?;
                let final_view = finalize::<u8, 3>(rgb, &image)?;
                rgb8_image(
                    final_view.width as u32,
                    final_view.height as u32,
                    final_view.data,
                )?
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
                a_stride: image.alpha.as_ref().map(|x| x.stride as u32).unwrap_or(w),
                width: w,
                height: h,
            };
            let mut rgba = vec![0u8; (rgba_stride * h) as usize];
            match (sub_w, sub_h) {
                (2, 2) => yuv420_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                (2, 1) => yuv422_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
                _ => yuv444_alpha_to_rgba(&yuv, &mut rgba, rgba_stride, range, matrix, false),
            }
            .map_err(|e| AvifError::Format(format!("yuv→rgba: {e}")))?;
            let final_view = finalize::<u8, 4>(rgba, &image)?;
            rgba8_image(
                final_view.width as u32,
                final_view.height as u32,
                final_view.data,
            )?
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
            .map_err(|e| AvifError::Format(format!("yuv→rgb: {e}")))?;
            let final_view = finalize::<u8, 3>(rgb, &image)?;
            rgb8_image(
                final_view.width as u32,
                final_view.height as u32,
                final_view.data,
            )?
        }
    };

    Ok(apply_orientation_tealdust(
        img,
        image_info.orientation.unwrap_or(Orientation::Normal),
    ))
}

fn rgb8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, AvifError> {
    Ok(DynamicImage::ImageRgb8(
        image::RgbImage::from_raw(w, h, buf)
            .ok_or_else(|| AvifError::Format("HEIC RGB mismatch".into()))?,
    ))
}
fn rgba8_image(w: u32, h: u32, buf: Vec<u8>) -> Result<DynamicImage, AvifError> {
    Ok(DynamicImage::ImageRgba8(
        image::RgbaImage::from_raw(w, h, buf)
            .ok_or_else(|| AvifError::Format("HEIC RGBA mismatch".into()))?,
    ))
}
fn rgb16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, AvifError> {
    Ok(DynamicImage::ImageRgb16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| AvifError::Format("HEIC RGB16 mismatch".into()))?,
    ))
}
fn rgba16_image(w: u32, h: u32, buf: Vec<u16>) -> Result<DynamicImage, AvifError> {
    Ok(DynamicImage::ImageRgba16(
        image::ImageBuffer::from_raw(w, h, buf)
            .ok_or_else(|| AvifError::Format("HEIC RGBA16 mismatch".into()))?,
    ))
}

/// 4:0:0 monochrome: the luma plane is the image. garnetash carries the luma in
/// `planes` (the whole buffer, no chroma) and any alpha as a separate monochrome
/// `DecodedImage` in `dec.alpha`.
fn finish_monochrome(
    dec: &AvifImage,
    w: u32,
    h: u32,
    high_bit: bool,
    is_12: bool,
) -> Result<DynamicImage, AvifError> {
    let img = if high_bit {
        let mut y = le_u16(
            &dec.planes[0],
            dec.width,
            dec.strides[0] as u32,
            Subsampling::Full,
        );
        crate::vvc::expand_to_16bit(&mut y, is_12);
        DynamicImage::ImageLuma16(
            image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, y)
                .ok_or_else(|| AvifError::Format("HEIC Luma16 mismatch".into()))?,
        )
    } else if let Some(alpha) = dec.alpha.as_ref() {
        // Interleave luma + alpha into a LumaA8 image.
        let y = &dec.planes[0];
        let mut ya = vec![0u8; w as usize * h as usize * 2];
        for ((dst, y), a) in ya
            .chunks_exact_mut(w as usize * 2)
            .zip(y.chunks_exact(dec.strides[0]))
            .zip(alpha.data.chunks_exact(alpha.stride))
        {
            let y = &y[..w as usize];
            let a = &a[..w as usize];
            for ((dst, &yv), &av) in dst
                .as_chunks_mut::<2>()
                .0
                .iter_mut()
                .zip(y.iter())
                .zip(a.iter())
            {
                dst[0] = yv;
                dst[1] = av;
            }
        }
        DynamicImage::ImageLumaA8(
            image::GrayAlphaImage::from_raw(w, h, ya)
                .ok_or_else(|| AvifError::Format("HEIC LumaA8 mismatch".into()))?,
        )
    } else {
        DynamicImage::ImageLuma8(
            image::GrayImage::from_raw(w, h, dec.planes[0].clone())
                .ok_or_else(|| AvifError::Format("HEIC Luma8 mismatch".into()))?,
        )
    };
    Ok(apply_orientation_tealdust(img, dec.orientation))
}
