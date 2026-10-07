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

    // CICP must describe the RGB pixels supplied to this encoder. HEIF decode
    // replaces its original YCbCr matrix/range with identity and full range.
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
        // JXL has one P3 primaries enum (11); Display P3 is distinguished
        // from DCI-P3 by its D65 white point, not by CICP's primaries code 12.
        12 => (WhitePoint::D65, Primaries::Smpte431),
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

fn color_encoding_from_icc(icc: &[u8]) -> Option<(jixel::ColorEncoding, Option<f32>)> {
    use moxcms::{ColorProfile, DataColorSpace, ToneReprCurve, TransferCharacteristics, Xyzd};

    let profile = ColorProfile::new_from_slice(icc).ok()?;
    // Only matrix/TRC RGB profiles can be represented by this shortcut. A LUT
    // profile needs a CMS transform; its matrix tags need not describe its colors.
    if profile.color_space != DataColorSpace::Rgb
        || profile.pcs != DataColorSpace::Xyz
        || profile.lut_a_to_b_perceptual.is_some()
        || profile.lut_a_to_b_colorimetric.is_some()
        || profile.lut_a_to_b_saturation.is_some()
    {
        return None;
    }

    // Allow ICC fixed-point rounding and small differences in the D50
    // chromatic-adaptation constants (notably Apple's Display P3 profile).
    let close = |a: Xyzd, b: Xyzd| {
        (a.x - b.x).abs() <= 0.0005 && (a.y - b.y).abs() <= 0.0005 && (a.z - b.z).abs() <= 0.0005
    };
    // These are the RGB primary matrices implemented by jixel's input transform.
    // Match numerical colorants, never the human-readable profile description.
    let primaries = [
        (1, ColorProfile::new_srgb()),
        (12, ColorProfile::new_display_p3()),
        (9, ColorProfile::new_bt2020()),
    ]
    .into_iter()
    .find_map(|(code, reference)| {
        (close(profile.white_point, reference.white_point)
            && close(profile.red_colorant, reference.red_colorant)
            && close(profile.green_colorant, reference.green_colorant)
            && close(profile.blue_colorant, reference.blue_colorant))
        .then_some(code)
    })?;

    let curves = [
        profile.red_trc.as_ref()?.make_linear_evaluator().ok()?,
        profile.green_trc.as_ref()?.make_linear_evaluator().ok()?,
        profile.blue_trc.as_ref()?.make_linear_evaluator().ok()?,
    ];
    let samples = curves
        .map(|curve| std::array::from_fn::<_, 257, _>(|i| curve.evaluate_value(i as f32 / 256.0)));
    // Compare all three channels, including both endpoints. This accepts
    // sampled and parametric ICC curves without depending on their storage form.
    let transfer = [13u8, 8, 1, 4, 5, 7, 16, 17, 18]
        .into_iter()
        .find(|&code| {
            let Ok(reference) = TransferCharacteristics::try_from(code)
                .and_then(ToneReprCurve::make_cicp_linear_evaluator)
            else {
                return false;
            };
            (0..257).all(|i| {
                let expected = reference.evaluate_value(i as f32 / 256.0);
                samples.iter().all(|channel| {
                    channel[i].is_finite() && (channel[i] - expected).abs() <= 0.0002
                })
            })
        })?;
    let (mut encoding, intensity_target) = color_encoding_from_cicp(PngCicp {
        color_primaries: primaries,
        transfer_function: transfer,
        matrix_coefficients: 0,
        full_range: true,
    })?;
    encoding.rendering_intent = match profile.rendering_intent {
        moxcms::RenderingIntent::Perceptual => jixel::RenderingIntent::Perceptual,
        moxcms::RenderingIntent::RelativeColorimetric => jixel::RenderingIntent::Relative,
        moxcms::RenderingIntent::Saturation => jixel::RenderingIntent::Saturation,
        moxcms::RenderingIntent::AbsoluteColorimetric => jixel::RenderingIntent::Absolute,
    };
    Some((encoding, intensity_target))
}

fn can_signal_structured_color(encoding: jixel::ColorEncoding) -> bool {
    // JXL's built-in enum values are a subset of CICP. Other transfer
    // functions may be usable for jixel's input transform while still needing
    // an ICC (or custom gamma) to describe them in a conforming codestream.
    matches!(
        encoding.primaries,
        jixel::Primaries::Bt709 | jixel::Primaries::Bt2020 | jixel::Primaries::Smpte431
    ) && matches!(
        encoding.transfer,
        jixel::TransferFunction::Bt709
            | jixel::TransferFunction::Linear
            | jixel::TransferFunction::Srgb
            | jixel::TransferFunction::Smpte2084
            | jixel::TransferFunction::Smpte428
            | jixel::TransferFunction::Hlg
    )
}

pub(crate) fn decode_jxl(file: &PathBuf) -> Result<(DynamicImage, Option<Vec<u8>>), JxlError> {
    let input_src = fs::read(file).map_err(|x| JxlError::Io(x.to_string()))?;
    decode_jxl_bytes(input_src)
}

fn decode_jxl_bytes(input_src: Vec<u8>) -> Result<(DynamicImage, Option<Vec<u8>>), JxlError> {
    use jxl::api::{
        Endianness, JxlDecoder, JxlDecoderOptions, JxlOutputBuffer, JxlPixelFormat,
        ProcessingResult,
    };

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

    decoder_with_image_info
        .set_pixel_format(JxlPixelFormat {
            color_type,
            color_data_format,
            extra_channel_format: vec![None; info.extra_channels.len()],
        })
        .map_err(|x| JxlError::Format(format!("jxl pixel format: {x}")))?;
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

pub(crate) struct JxlMetadata<'a> {
    pub icc: Option<&'a [u8]>,
    pub exif: Option<&'a [u8]>,
    pub cicp: Option<PngCicp>,
    #[cfg(feature = "heic")]
    pub gain_map: Option<&'a crate::gainmap::ParsedGainMap>,
}

#[cfg(feature = "heic")]
fn jxl_gain_map(gain_map: &crate::gainmap::ParsedGainMap, lossless: bool) -> jixel::GainMap {
    let m = gain_map.metadata;
    // The two crates expose separate ISO metadata types. Copy the rationals
    // exactly so encoding does not introduce another float conversion.
    let metadata = jixel::IsoGainMap {
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
        backward_direction: m.backward_direction,
        use_base_color_space: m.use_base_color_space,
    };
    jixel::GainMap::gray16(
        gain_map.image.as_raw().clone(),
        16,
        gain_map.image.width() as usize,
        gain_map.image.height() as usize,
        metadata,
    )
    // Samples already contain normalized logarithmic gains, with no OETF.
    .with_color_encoding(jixel::ColorEncoding::srgb_linear())
    .with_lossless(lossless)
}

pub(crate) fn encode_jxl(
    img: &DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    metadata: JxlMetadata<'_>,
) -> Result<Vec<u8>, anyhow::Error> {
    let mut cfg = jixel::EncodeConfig::default().with_quality(args.quality as f32);
    #[cfg(feature = "heic")]
    if let Some(gain_map) = metadata.gain_map {
        cfg = cfg.with_gain_map(jxl_gain_map(gain_map, args.lossless));
        if args.verbose {
            eprintln!(
                "gainmap: encoding {}×{} ISO 21496-1 map ({:.3} stops) in JXL jhgm{}",
                gain_map.image.width(),
                gain_map.image.height(),
                gain_map.metadata.alternate_hdr_headroom(),
                if args.lossless { " (lossless)" } else { "" },
            );
        }
    }

    let cicp_encoding = metadata.cicp.and_then(color_encoding_from_cicp);
    let input_encoding = cicp_encoding.or_else(|| metadata.icc.and_then(color_encoding_from_icc));
    if let Some((encoding, intensity_target)) = input_encoding {
        cfg = cfg.with_color_encoding(encoding);
        if let Some(intensity_target) = intensity_target {
            cfg = cfg.with_intensity_target(intensity_target);
        }
        if args.verbose {
            eprintln!(
                "jxl    : input {:?} / {:?} (from {})",
                encoding.primaries,
                encoding.transfer,
                if cicp_encoding.is_some() {
                    "CICP"
                } else {
                    "ICC"
                },
            );
        }
    } else if metadata.icc.is_some() && !args.lossless {
        anyhow::bail!(
            "JXL: ICC profile cannot be matched to a supported input color space; \
             use --apply-icc to convert to sRGB, or --lossless to preserve the original pixels"
        );
    }
    // With XYB + ICC, ImageIO and decoders without a CMS can choose sRGB as
    // their output space. Signal recognized colors with JXL's built-in fields
    // so they can reconstruct P3/BT.2020 directly. Lossless keeps the exact ICC.
    let structured_color = !args.lossless
        && input_encoding.is_some_and(|(encoding, _)| can_signal_structured_color(encoding));
    if let Some(icc) = metadata.icc.filter(|_| !structured_color) {
        cfg = cfg.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = metadata.exif {
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

    #[test]
    fn identifies_icc_colorimetry_without_a_name_or_cicp_tag() {
        for (mut profile, primaries) in [
            (moxcms::ColorProfile::new_srgb(), jixel::Primaries::Bt709),
            (
                moxcms::ColorProfile::new_display_p3(),
                jixel::Primaries::Smpte431,
            ),
            (moxcms::ColorProfile::new_bt2020(), jixel::Primaries::Bt2020),
        ] {
            profile.description = None;
            profile.cicp = None;
            for (curve, transfer) in [
                (
                    moxcms::ColorProfile::new_srgb().red_trc.unwrap(),
                    jixel::TransferFunction::Srgb,
                ),
                (
                    moxcms::ToneReprCurve::Lut(vec![]),
                    jixel::TransferFunction::Linear,
                ),
                (
                    moxcms::ToneReprCurve::Parametric(vec![2.2]),
                    jixel::TransferFunction::Bt470M,
                ),
            ] {
                profile.red_trc = Some(curve.clone());
                profile.green_trc = Some(curve.clone());
                profile.blue_trc = Some(curve);
                let (encoding, _) = color_encoding_from_icc(&profile.encode().unwrap()).unwrap();
                assert_eq!(encoding.primaries, primaries);
                assert_eq!(encoding.transfer, transfer);
            }
        }
    }

    #[test]
    fn recognizes_apples_quantized_display_p3_colorants_and_curve() {
        use moxcms::{ToneReprCurve, Xyzd};
        let mut profile = moxcms::ColorProfile::new_display_p3();
        // Numeric ICC tags from Apple's 536-byte Display P3 profile. Its D50
        // adaptation differs slightly from moxcms's built-in reference.
        profile.white_point = Xyzd {
            x: 0.964202880859375,
            y: 1.0,
            z: 0.8249053955078125,
        };
        profile.red_colorant = Xyzd {
            x: 0.5151214599609375,
            y: 0.2411956787109375,
            z: -0.0010528564453125,
        };
        profile.green_colorant = Xyzd {
            x: 0.2919769287109375,
            y: 0.6922454833984375,
            z: 0.0418853759765625,
        };
        profile.blue_colorant = Xyzd {
            x: 0.1571044921875,
            y: 0.0665740966796875,
            z: 0.7840728759765625,
        };
        let curve = ToneReprCurve::Parametric(vec![
            2.399994,
            0.9478607,
            0.052139282,
            0.07739258,
            0.04045105,
        ]);
        profile.red_trc = Some(curve.clone());
        profile.green_trc = Some(curve.clone());
        profile.blue_trc = Some(curve);
        profile.cicp = None;
        profile.description = None;
        let (encoding, target) = color_encoding_from_icc(&profile.encode().unwrap()).unwrap();
        assert_eq!(encoding.primaries, jixel::Primaries::Smpte431);
        assert_eq!(encoding.transfer, jixel::TransferFunction::Srgb);
        assert_eq!(target, None);
    }

    #[test]
    fn rejects_icc_profiles_that_do_not_match_a_supported_transform() {
        assert!(color_encoding_from_icc(b"invalid ICC").is_none());
        // A familiar description is insufficient when a channel has a different TRC.
        let mut profile = moxcms::ColorProfile::new_display_p3();
        profile.green_trc = Some(moxcms::ToneReprCurve::Parametric(vec![1.8]));
        assert!(color_encoding_from_icc(&profile.encode().unwrap()).is_none());
        let mut profile = moxcms::ColorProfile::new_display_p3();
        profile.red_colorant.x += 0.05;
        assert!(color_encoding_from_icc(&profile.encode().unwrap()).is_none());
        assert!(
            color_encoding_from_icc(&moxcms::ColorProfile::new_adobe_rgb().encode().unwrap())
                .is_none()
        );
    }

    #[cfg(feature = "heic")]
    mod gain_maps {
        use super::*;
        use crate::gainmap::ParsedGainMap;

        fn args(lossless: bool) -> Args {
            Args {
                input: PathBuf::new(),
                output: PathBuf::new(),
                encoder: crate::Encoder::JpegXl,
                quality: 90,
                lossless,
                chroma: None,
                depth: None,
                threads: 1,
                no_alpha: false,
                no_exif: false,
                no_icc: false,
                apply_icc: false,
                verbose: false,
                speed: crate::EncodingEffort::Fast,
                qmatrix: None,
                updating_cdf: false,
                screen_content: false,
                intrabc: false,
                cdef: false,
                wiener: false,
            }
        }

        fn gain_map() -> ParsedGainMap {
            let metadata = jixel::IsoGainMap::from_floats(&jixel::GainMapFloats {
                max: [2.0; 3],
                base_offset: [0.0; 3],
                alternate_offset: [0.0; 3],
                alternate_hdr_headroom: 2.0,
                ..Default::default()
            })
            .unwrap();
            ParsedGainMap {
                image: image::ImageBuffer::from_raw(3, 2, vec![0, 10000, 30000, 65535, 42, 47000])
                    .unwrap(),
                metadata: gainforge::IsoGainMap::from_metadata(&metadata.to_metadata().unwrap())
                    .unwrap(),
            }
        }

        fn box_payload<'a>(mut data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
            while data.len() >= 8 {
                let size = u32::from_be_bytes(data[..4].try_into().unwrap()) as usize;
                let (size, header) = match size {
                    0 => (data.len(), 8),
                    1 => (
                        u64::from_be_bytes(data[8..16].try_into().unwrap()) as usize,
                        16,
                    ),
                    size => (size, 8),
                };
                assert!(size >= header && size <= data.len());
                if &data[4..8] == kind {
                    return Some(&data[header..size]);
                }
                data = &data[size..];
            }
            None
        }

        fn gain_bundle(data: &[u8]) -> (gainforge::IsoGainMap, &[u8]) {
            let bundle = box_payload(data, b"jhgm").expect("missing gain-map box");
            assert_eq!(bundle[0], 0); // JxlGainMapBundle version
            let metadata_end = 3 + u16::from_be_bytes(bundle[1..3].try_into().unwrap()) as usize;
            let metadata = gainforge::IsoGainMap::from_metadata(&bundle[3..metadata_end]).unwrap();
            let icc_offset = metadata_end + 1 + bundle[metadata_end] as usize;
            let image_offset = icc_offset
                + 4
                + u32::from_be_bytes(bundle[icc_offset..icc_offset + 4].try_into().unwrap())
                    as usize;
            (metadata, &bundle[image_offset..])
        }

        #[test]
        fn writes_decodable_gain_map_on_all_cli_pixel_paths() {
            let gain = gain_map();
            let rgb = DynamicImage::ImageRgb8(image::RgbImage::from_fn(8, 8, |x, y| {
                image::Rgb([(x * 31) as u8, (y * 31) as u8, 128])
            }));
            let images = [
                DynamicImage::ImageLuma8(rgb.to_luma8()),
                rgb.clone(),
                DynamicImage::ImageRgba8(rgb.to_rgba8()),
            ];
            for lossless in [false, true] {
                for depth in [Depth::D8, Depth::D10, Depth::D12] {
                    for primary in &images {
                        let encoded = encode_jxl(
                            primary,
                            &args(lossless),
                            primary.color(),
                            depth,
                            JxlMetadata {
                                icc: None,
                                exif: None,
                                cicp: None,
                                gain_map: Some(&gain),
                            },
                        )
                        .unwrap();
                        let (metadata, map_codestream) = gain_bundle(&encoded);
                        assert_eq!(metadata.map_min(), gain.metadata.map_min());
                        assert_eq!(metadata.map_max(), gain.metadata.map_max());
                        assert_eq!(metadata.gain_map_gamma(), gain.metadata.gain_map_gamma());
                        assert_eq!(metadata.map_base_offset(), gain.metadata.map_base_offset());
                        assert_eq!(
                            metadata.map_alternate_offset(),
                            gain.metadata.map_alternate_offset()
                        );
                        assert_eq!(
                            metadata.base_hdr_headroom(),
                            gain.metadata.base_hdr_headroom()
                        );
                        assert_eq!(
                            metadata.alternate_hdr_headroom(),
                            gain.metadata.alternate_hdr_headroom()
                        );
                        assert_eq!(
                            metadata.backward_direction,
                            gain.metadata.backward_direction
                        );
                        assert_eq!(
                            metadata.use_base_color_space,
                            gain.metadata.use_base_color_space
                        );
                        let (decoded_gain, _) = decode_jxl_bytes(map_codestream.to_vec()).unwrap();
                        assert_eq!(
                            (decoded_gain.width(), decoded_gain.height()),
                            gain.image.dimensions()
                        );
                        if lossless {
                            assert_eq!(decoded_gain.into_luma16(), gain.image);
                        }
                        let (decoded_primary, _) = decode_jxl_bytes(encoded).unwrap();
                        assert_eq!((decoded_primary.width(), decoded_primary.height()), (8, 8));
                    }
                }
            }
        }

        #[test]
        fn preserves_metadata_rationals_and_linear_sample_encoding() {
            let mut gain = gain_map();
            gain.metadata.gain_map_max_n = [1_234_567; 3];
            gain.metadata.gain_map_max_d = [1_000_000; 3];
            gain.metadata.alternate_hdr_headroom_n = 1_234_567;
            gain.metadata.alternate_hdr_headroom_d = 1_000_000;
            let encoded = jxl_gain_map(&gain, true);
            let parsed =
                gainforge::IsoGainMap::from_metadata(&encoded.metadata.to_metadata().unwrap())
                    .unwrap();
            assert_eq!(parsed.gain_map_max_n, gain.metadata.gain_map_max_n);
            assert_eq!(parsed.gain_map_max_d, gain.metadata.gain_map_max_d);
            assert_eq!(
                parsed.alternate_hdr_headroom_n,
                gain.metadata.alternate_hdr_headroom_n
            );
            assert_eq!(
                parsed.alternate_hdr_headroom_d,
                gain.metadata.alternate_hdr_headroom_d
            );
            assert_eq!(
                encoded.color_encoding.transfer,
                jixel::TransferFunction::Linear
            );
            assert!(encoded.lossless);
        }

        #[test]
        fn preserves_primary_profile_and_pixels_with_or_without_a_map() {
            let gain = gain_map();
            let primary = DynamicImage::ImageRgb8(image::RgbImage::from_pixel(
                8,
                8,
                image::Rgb([32, 128, 241]),
            ));
            let profile = moxcms::ColorProfile::new_display_p3().encode().unwrap();
            for map in [None, Some(&gain)] {
                let encoded = encode_jxl(
                    &primary,
                    &args(true),
                    primary.color(),
                    Depth::D8,
                    JxlMetadata {
                        icc: Some(&profile),
                        exif: Some(b"Exif\0\0test"),
                        cicp: None,
                        gain_map: map,
                    },
                )
                .unwrap();
                assert_eq!(box_payload(&encoded, b"jhgm").is_some(), map.is_some());
                assert!(box_payload(&encoded, b"Exif").is_some());
                let (decoded, icc) = decode_jxl_bytes(encoded).unwrap();
                assert_eq!(decoded.to_rgb8(), primary.to_rgb8());
                assert_eq!(icc.as_deref(), Some(profile.as_slice()));
            }
        }

        #[test]
        fn lossy_p3_uses_p3_for_input_and_default_decoder_output() {
            let primary = image::RgbImage::from_fn(8, 8, |x, y| {
                image::Rgb([(x * 31) as u8, (y * 31) as u8, 128])
            });
            let profile = moxcms::ColorProfile::new_display_p3().encode().unwrap();
            let cicp = PngCicp {
                color_primaries: 12,
                transfer_function: 13,
                matrix_coefficients: 0,
                full_range: true,
            };
            let cli_output = encode_jxl(
                &DynamicImage::ImageRgb8(primary.clone()),
                &args(false),
                image::ColorType::Rgb8,
                Depth::D8,
                JxlMetadata {
                    icc: Some(&profile),
                    exif: None,
                    cicp: Some(cicp),
                    gain_map: None,
                },
            )
            .unwrap();
            let expected_config = jixel::EncodeConfig::default()
                .with_quality(90.0)
                .with_color_encoding(jixel::ColorEncoding {
                    white_point: jixel::WhitePoint::D65,
                    primaries: jixel::Primaries::Smpte431,
                    transfer: jixel::TransferFunction::Srgb,
                    rendering_intent: jixel::RenderingIntent::Perceptual,
                });
            let reference = jixel::encode_image(primary.as_raw(), 8, 8, &expected_config).unwrap();
            assert_eq!(cli_output, reference);
            // Checking the original ICC alone missed ImageIO's sRGB fallback.
            // A separate decoder must also select P3 for its default pixels.
            use jxl::api::{
                JxlColorEncoding, JxlDecoder, JxlDecoderOptions, JxlPrimaries, JxlTransferFunction,
                JxlWhitePoint, ProcessingResult,
            };
            let mut reader = BufReader::new(Cursor::new(cli_output.as_slice()));
            let ProcessingResult::Complete { result: decoder } =
                JxlDecoder::new(JxlDecoderOptions::default())
                    .process(&mut reader, None)
                    .unwrap()
            else {
                panic!("complete codestream must include color information");
            };
            for profile in [
                decoder.embedded_color_profile(),
                decoder.output_color_profile(),
            ] {
                assert!(matches!(
                    profile,
                    JxlColorProfile::Simple(JxlColorEncoding::RgbColorSpace {
                        white_point: JxlWhitePoint::D65,
                        primaries: JxlPrimaries::P3,
                        transfer_function: JxlTransferFunction::SRGB,
                        ..
                    })
                ));
            }
            for cicp in [
                None,
                Some(PngCicp {
                    color_primaries: 2,
                    transfer_function: 2,
                    matrix_coefficients: 0,
                    full_range: true,
                }),
            ] {
                let without_cicp = encode_jxl(
                    &DynamicImage::ImageRgb8(primary.clone()),
                    &args(false),
                    image::ColorType::Rgb8,
                    Depth::D8,
                    JxlMetadata {
                        icc: Some(&profile),
                        exif: None,
                        cicp,
                        gain_map: None,
                    },
                )
                .unwrap();
                assert_eq!(
                    without_cicp, reference,
                    "ICC must supply missing input colorimetry"
                );
            }
            let srgb_input = jixel::encode_image(
                primary.as_raw(),
                8,
                8,
                &jixel::EncodeConfig::default()
                    .with_quality(90.0)
                    .with_icc_profile(profile),
            )
            .unwrap();
            assert_ne!(
                cli_output, srgb_input,
                "attaching a P3 ICC alone does not set the input primaries"
            );
        }

        #[test]
        fn fails_encode_when_gain_metadata_is_invalid() {
            let mut gain = gain_map();
            gain.metadata.gain_map_gamma_d[0] = 0;
            let primary = DynamicImage::new_rgb8(8, 8);
            assert!(
                encode_jxl(
                    &primary,
                    &args(false),
                    primary.color(),
                    Depth::D8,
                    JxlMetadata {
                        icc: None,
                        exif: None,
                        cicp: None,
                        gain_map: Some(&gain),
                    }
                )
                .is_err()
            );
        }
    }
}
