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
use crate::{Args, Depth, PngCicp, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12};
use image::DynamicImage;
use jxl::api::{JxlColorProfile, JxlColorType, JxlDataFormat};
use jxl::headers::extra_channels::ExtraChannel;
use std::fs;
use std::io::{BufReader, Cursor};
use std::path::PathBuf;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum JxlError {
    #[error("An error happened while decoding: heic`{0}`")]
    Format(String),
    #[error("Cannot read file due to an error:`{0}`")]
    Io(String),
}

fn color_encoding_from_cicp(cicp: PngCicp) -> Option<(jixel::ColorEncoding, Option<f32>)> {
    use jixel::{Primaries, RenderingIntent, TransferFunction, WhitePoint};

    // PNG sample values are RGB only when the matrix is identity and the full
    // component range is used. Other CICP layouts need a pixel conversion that
    // the current CLI raster path does not perform.
    if cicp.matrix_coefficients != 0 || !cicp.full_range {
        return None;
    }

    let (white_point, primaries) = match cicp.color_primaries {
        1 => (WhitePoint::D65, Primaries::Bt709),
        5 => (WhitePoint::D65, Primaries::Bt470Bg),
        6 => (WhitePoint::D65, Primaries::Bt601),
        7 => (WhitePoint::D65, Primaries::Smpte240),
        9 => (WhitePoint::D65, Primaries::Bt2020),
        10 => (WhitePoint::E, Primaries::Xyz),
        11 => (WhitePoint::Dci, Primaries::Smpte431),
        12 => (WhitePoint::D65, Primaries::Smpte432),
        22 => (WhitePoint::D65, Primaries::Ebu3213),
        _ => return None,
    };
    let transfer = match cicp.transfer_function {
        1 => TransferFunction::Bt709,
        4 => TransferFunction::Bt470M,
        5 => TransferFunction::Bt470Bg,
        6 => TransferFunction::Bt601,
        7 => TransferFunction::Smpte240,
        8 => TransferFunction::Linear,
        9 => TransferFunction::Log100,
        10 => TransferFunction::Log100sqrt10,
        11 => TransferFunction::Iec61966,
        12 => TransferFunction::Bt1361,
        13 => TransferFunction::Srgb,
        14 => TransferFunction::Bt202010bit,
        15 => TransferFunction::Bt202012bit,
        16 => TransferFunction::Smpte2084,
        17 => TransferFunction::Smpte428,
        18 => TransferFunction::Hlg,
        _ => return None,
    };
    let intensity_target = match transfer {
        TransferFunction::Smpte2084 => Some(10_000.0),
        TransferFunction::Hlg => Some(1_000.0),
        _ => None,
    };

    Some((
        jixel::ColorEncoding {
            white_point,
            primaries,
            transfer,
            rendering_intent: RenderingIntent::Perceptual,
        },
        intensity_target,
    ))
}

pub(crate) fn decode_jxl(file: &PathBuf) -> Result<(DynamicImage, Option<Vec<u8>>), JxlError> {
    use jxl::api::{
        Endianness, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult,
    };

    let input_src: Vec<u8> = fs::read(file).map_err(|x| JxlError::Io(x.to_string()))?;
    let mut reader = BufReader::new(Cursor::new(input_src));

    let mut decoder_with_image_info = match JxlDecoder::new(JxlDecoderOptions::default())
        .process(&mut reader, None)
        .map_err(|x| JxlError::Format(format!("jxl {x}")))?
    {
        ProcessingResult::Complete { result: d } => d,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(JxlError::Format("jxl: truncated before basic_info".into()));
        }
    };

    let info = decoder_with_image_info.basic_info();
    let (w, h) = info.size;
    let bits = info.bit_depth.bits_per_sample();
    let has_alpha = info
        .extra_channels
        .iter()
        .any(|ec| ec.ec_type == ExtraChannel::Alpha);
    let is_gray = matches!(
        decoder_with_image_info.current_pixel_format().color_type,
        JxlColorType::Grayscale | JxlColorType::GrayscaleAlpha
    );

    let color_type = match (is_gray, has_alpha) {
        (false, false) => JxlColorType::Rgb,
        (false, true) => JxlColorType::Rgba,
        (true, false) => JxlColorType::Grayscale,
        (true, true) => JxlColorType::GrayscaleAlpha,
    };

    let color_data_format = Some(match bits {
        0..=8 => JxlDataFormat::U8 {
            bit_depth: bits as u8,
        },
        9..=16 => JxlDataFormat::U16 {
            bit_depth: 16,
            endianness: Endianness::LittleEndian,
        },
        _ => JxlDataFormat::F32 {
            endianness: Endianness::LittleEndian,
        },
    });

    decoder_with_image_info.set_pixel_format(JxlPixelFormat {
        color_type,
        color_data_format,
        extra_channel_format: vec![None; info.extra_channels.len()],
    });
    let output_profile = match decoder_with_image_info.output_color_profile() {
        JxlColorProfile::Icc(v) => Some(v.to_vec()),
        JxlColorProfile::Simple(_) => None,
    };

    let decoder_with_frame_info = match decoder_with_image_info
        .process(&mut reader, None)
        .map_err(|x| JxlError::Format(format!("jxl {x}")))?
    {
        ProcessingResult::Complete { result: d } => d,
        ProcessingResult::NeedsMoreInput { .. } => {
            return Err(JxlError::Format("jxl: truncated before frame info".into()));
        }
    };

    macro_rules! decode_pixels {
        (u8, $channels:expr, $label:expr, $img:ident) => {{
            let stride = w * $channels;
            let mut buf = vec![0u8; h * stride];
            let mut out = [JxlOutputBuffer::new(buf.as_mut_slice(), h, stride)];
            decoder_with_frame_info
                .process(&mut reader, &mut out, None)
                .map_err(|x| JxlError::Format(format!("jxl {x}")))?;
            DynamicImage::$img(
                image::ImageBuffer::from_raw(w as u32, h as u32, buf)
                    .ok_or_else(|| JxlError::Format(format!("jxl {} buffer mismatch", $label)))?,
            )
        }};
        ($T:ty, $channels:expr, $label:expr, $img:ident) => {{
            let stride_bytes = w * $channels * size_of::<$T>();
            let mut buf = vec![0 as $T; w * h * $channels];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut reader, &mut out, None)
                .map_err(|x| JxlError::Format(format!("jxl {x}")))?;
            DynamicImage::$img(
                image::ImageBuffer::from_raw(w as u32, h as u32, buf)
                    .ok_or_else(|| JxlError::Format(format!("jxl {} buffer mismatch", $label)))?,
            )
        }};
    }

    let image = match (is_gray, has_alpha, bits) {
        (false, false, 0..=8) => decode_pixels!(u8, 3, "RGB8", ImageRgb8),
        (false, true, 0..=8) => decode_pixels!(u8, 4, "RGBA8", ImageRgba8),
        (true, false, 0..=8) => decode_pixels!(u8, 1, "Luma8", ImageLuma8),
        (true, true, 0..=8) => decode_pixels!(u8, 2, "LumaA8", ImageLumaA8),
        (false, false, 9..=16) => decode_pixels!(u16, 3, "RGB16", ImageRgb16),
        (false, true, 9..=16) => decode_pixels!(u16, 4, "RGBA16", ImageRgba16),
        (true, false, 9..=16) => decode_pixels!(u16, 1, "Luma16", ImageLuma16),
        (true, true, 9..=16) => decode_pixels!(u16, 2, "LumaA16", ImageLumaA16),
        (false, false, _) => decode_pixels!(f32, 3, "Rgb32F", ImageRgb32F),
        (false, true, _) => decode_pixels!(f32, 4, "Rgba32F", ImageRgba32F),
        // DynamicImage has no Luma32F/LumaA32F — expand gray to RGB(A)
        (true, false, _) => {
            let stride_bytes = w * size_of::<f32>();
            let mut buf = vec![0f32; w * h];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut reader, &mut out, None)
                .map_err(|x| JxlError::Format(format!("jxl {x}")))?;
            let rgb: Vec<f32> = buf.iter().flat_map(|&v| [v, v, v]).collect();
            DynamicImage::ImageRgb32F(
                image::ImageBuffer::from_raw(w as u32, h as u32, rgb).ok_or_else(|| {
                    JxlError::Format("jxl Gray to RGB buffer mismatch".to_string())
                })?,
            )
        }
        (true, true, _) => {
            let stride_bytes = w * 2 * size_of::<f32>();
            let mut buf = vec![0f32; w * h * 2];
            let mut out = [JxlOutputBuffer::new(
                bytemuck::cast_slice_mut(&mut buf),
                h,
                stride_bytes,
            )];
            decoder_with_frame_info
                .process(&mut reader, &mut out, None)
                .map_err(|x| JxlError::Format(format!("jxl {x}")))?;
            let rgba: Vec<f32> = buf
                .as_chunks::<2>()
                .0
                .iter()
                .flat_map(|&[g, a]| [g, g, g, a])
                .collect();
            DynamicImage::ImageRgba32F(
                image::ImageBuffer::from_raw(w as u32, h as u32, rgba).ok_or_else(|| {
                    JxlError::Format("jxl GrayA to RGBA buffer mismatch".to_string())
                })?,
            )
        }
    };

    Ok((image, output_profile))
}

pub(crate) fn encode_jxl(
    img: &DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    png_cicp: Option<PngCicp>,
) -> Result<Vec<u8>, anyhow::Error> {
    let mut cfg = jixel::EncodeConfig::default().with_quality(args.quality as f32);

    let cicp_encoding = png_cicp.and_then(color_encoding_from_cicp);
    if let Some((encoding, intensity_target)) = cicp_encoding {
        println!(
            "encoding CICP {:?} intensity_target {:?}",
            encoding, intensity_target
        );

        cfg = cfg.with_color_encoding(encoding);
        if let Some(intensity_target) = intensity_target {
            cfg = cfg.with_intensity_target(intensity_target);
        }
    } else if let Some(icc) = icc {
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
        (Depth::D8, true, _) => jixel::encode_image_gray(
            img.to_luma8().as_raw(),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D8, false, false) => jixel::encode_image(
            img.to_rgb8().as_raw(),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D8, false, true) => jixel::encode_image_with_alpha(
            img.to_rgba8().as_raw(),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D10, true, _) => jixel::encode_image_gray_10bit(
            &scale16_to_10(img.to_luma16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D10, false, false) => jixel::encode_image_10bit(
            &scale16_to_10(img.to_rgb16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D10, false, true) => jixel::encode_image_with_alpha_10bit(
            &scale16_to_10(img.to_rgba16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D12, true, _) => jixel::encode_image_gray_12bit(
            &scale16_to_12(img.to_luma16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D12, false, false) => jixel::encode_image_12bit(
            &scale16_to_12(img.to_rgb16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
        (Depth::D12, false, true) => jixel::encode_image_with_alpha_12bit(
            &scale16_to_12(img.to_rgba16().as_raw()),
            img.width() as usize,
            img.height() as usize,
            &cfg,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_png_rec2100_pq_cicp() {
        let (encoding, intensity_target) = color_encoding_from_cicp(PngCicp {
            color_primaries: 9,
            transfer_function: 16,
            matrix_coefficients: 0,
            full_range: true,
        })
        .expect("BT.2100 PQ must be supported");

        assert_eq!(encoding.white_point, jixel::WhitePoint::D65);
        assert_eq!(encoding.primaries, jixel::Primaries::Bt2020);
        assert_eq!(encoding.transfer, jixel::TransferFunction::Smpte2084);
        assert_eq!(intensity_target, Some(10_000.0));
    }

    #[test]
    fn rejects_non_rgb_png_cicp() {
        assert!(
            color_encoding_from_cicp(PngCicp {
                color_primaries: 9,
                transfer_function: 16,
                matrix_coefficients: 9,
                full_range: true,
            })
            .is_none()
        );
    }
}
