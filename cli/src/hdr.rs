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

//! Convert between ISO 21496-1 gain maps and full-range BT.2020 PQ or HLG.

use crate::{PngCicp, gainmap::ParsedGainMap, resize::resize_bilinear_u16};
use anyhow::{Context, Result, bail, ensure};
use gainforge::{IsoGainMap, TransferFunction};
use image::{DynamicImage, ImageBuffer, Rgb, Rgb32FImage};
use moxcms::{
    CicpColorPrimaries, CicpProfile, ColorProfile, DataColorSpace, Layout, MatrixCoefficients,
    ToneReprCurve, TransferCharacteristics, TransformOptions, Vector3f,
};
use std::borrow::Cow;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HdrConversion {
    ToPq,
    ToHlg,
    ToGainMap,
}

pub(crate) const PQ_CICP: PngCicp = PngCicp {
    color_primaries: 9,
    transfer_function: 16,
    matrix_coefficients: 0,
    full_range: true,
};

pub(crate) const HLG_CICP: PngCicp = PngCicp {
    transfer_function: 18,
    ..PQ_CICP
};

pub(crate) const SDR_CICP: PngCicp = PngCicp {
    transfer_function: 13,
    ..PQ_CICP
};

const SRGB_CICP: PngCicp = PngCicp {
    color_primaries: 1,
    ..SDR_CICP
};

// Normalize the linear gain-map math to BT.2408's 203 cd/m² reference white.
const PQ_PEAK_RELATIVE_TO_SDR_WHITE: f32 = 10_000.0 / 203.0;
const HLG_PEAK_RELATIVE_TO_SDR_WHITE: f32 = 1_000.0 / 203.0;
const HLG_SYSTEM_GAMMA: f32 = 1.2;
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892;
const HLG_C: f32 = 0.55991073;
const MAX_SAMPLE: f32 = u16::MAX as f32;

pub(crate) fn is_hdr(cicp: PngCicp) -> bool {
    matches!(cicp.transfer_function, 16 | 18)
}

fn bt2020_luminance([r, g, b]: [f32; 3]) -> f32 {
    0.2627 * r + 0.6780 * g + 0.0593 * b
}

// ITU-R BT.2100-3 Table 5. Use a 1,000 cd/m² reference display with zero
// black level. The OOTF applies the luminance-derived gain to all channels.
fn hlg_to_scene(signal: f32) -> f32 {
    if signal <= 0.5 {
        signal * signal / 3.0
    } else {
        (((signal - HLG_C) / HLG_A).exp() + HLG_B) / 12.0
    }
}

fn hlg_from_scene(scene: f32) -> f32 {
    if scene <= 1.0 / 12.0 {
        (3.0 * scene).sqrt()
    } else {
        HLG_A * (12.0 * scene - HLG_B).ln() + HLG_C
    }
}

fn encode_hlg(rgb: [f32; 3]) -> [u16; 3] {
    let display = rgb.map(|channel| channel.max(0.0) / HLG_PEAK_RELATIVE_TO_SDR_WHITE);
    let luminance = bt2020_luminance(display);
    if luminance == 0.0 {
        return [0; 3];
    }
    let inverse_ootf = luminance.powf(1.0 / HLG_SYSTEM_GAMMA - 1.0);
    let scene = display.map(|channel| channel * inverse_ootf);
    // Reduce out-of-range highlights to HLG's representable peak along the
    // same color direction, keeping channel ratios through the gamut limit.
    let scale = peak_channel(Rgb(scene)).max(1.0).recip();
    scene.map(|channel| encode_u16(hlg_from_scene(channel * scale)))
}

fn encode_pq(rgb: [f32; 3]) -> [u16; 3] {
    rgb.map(|channel| encode_u16(TransferFunction::PerceptualQuantizer.gamma(channel)))
}

fn hdr_pixel_encoder(cicp: PngCicp) -> Result<fn([f32; 3]) -> [u16; 3]> {
    match cicp {
        PQ_CICP => Ok(encode_pq),
        HLG_CICP => Ok(encode_hlg),
        _ => bail!("expected full-range BT.2020 PQ or HLG output"),
    }
}

fn linear_profile(primaries: u8) -> Result<ColorProfile> {
    let color_primaries = CicpColorPrimaries::try_from(primaries)?;
    ensure!(
        !matches!(
            color_primaries,
            CicpColorPrimaries::Reserved | CicpColorPrimaries::Unspecified
        ),
        "unsupported CICP color primaries {primaries}"
    );
    Ok(ColorProfile::new_from_cicp(CicpProfile {
        color_primaries,
        transfer_characteristics: TransferCharacteristics::Linear,
        matrix_coefficients: MatrixCoefficients::Identity,
        full_range: true,
    }))
}

fn linear_matrix_profile(profile: &ColorProfile) -> Result<ColorProfile> {
    let mut linear = linear_profile(SRGB_CICP.color_primaries)?;
    linear.red_colorant = profile.red_colorant;
    linear.green_colorant = profile.green_colorant;
    linear.blue_colorant = profile.blue_colorant;
    linear.chromatic_adaptation = profile.chromatic_adaptation;
    linear.white_point = profile.white_point;
    linear.cicp = None;
    Ok(linear)
}

fn linear_rgb_from_cicp(
    image: &DynamicImage,
    cicp: PngCicp,
) -> Result<(Rgb32FImage, ColorProfile)> {
    ensure!(cicp.matrix_coefficients == 0, "expected decoded RGB CICP");
    let transfer = match cicp.transfer_function {
        2 => TransferCharacteristics::Srgb,
        code => TransferCharacteristics::try_from(code)?,
    };
    let primaries = match cicp.color_primaries {
        2 if is_hdr(cicp) => PQ_CICP.color_primaries,
        2 => SRGB_CICP.color_primaries,
        code => code,
    };
    let profile = linear_profile(primaries)?;
    let (black, span) = if cicp.full_range {
        (0.0, 1.0)
    } else {
        // Limited-range RGB encodes black/white at 16/235, scaled to bit depth.
        let unit = match image.color() {
            image::ColorType::L16
            | image::ColorType::La16
            | image::ColorType::Rgb16
            | image::ColorType::Rgba16 => 256.0 / MAX_SAMPLE,
            _ => 1.0 / u8::MAX as f32,
        };
        (16.0 * unit, 219.0 * unit)
    };
    let mut rgb = image.to_rgb32f();
    let normalize = |sample: f32| ((sample - black) / span).clamp(0.0, 1.0);
    if transfer == TransferCharacteristics::Hlg {
        let to_bt2020 = profile
            .transform_matrix(&linear_profile(HLG_CICP.color_primaries)?)
            .to_f32();
        for pixel in rgb.pixels_mut() {
            let scene = pixel.0.map(|sample| hlg_to_scene(normalize(sample)));
            let luminance = bt2020_luminance(to_bt2020.mul_vector(Vector3f { v: scene }).v);
            let gain =
                HLG_PEAK_RELATIVE_TO_SDR_WHITE * luminance.max(0.0).powf(HLG_SYSTEM_GAMMA - 1.0);
            pixel.0 = scene.map(|channel| gain * channel);
        }
    } else if transfer == TransferCharacteristics::Smpte2084 {
        for sample in rgb.as_mut() {
            *sample = TransferFunction::PerceptualQuantizer
                .linearize(normalize(*sample), PQ_PEAK_RELATIVE_TO_SDR_WHITE);
        }
    } else {
        let curve = ToneReprCurve::make_cicp_linear_evaluator(transfer)?;
        for sample in rgb.as_mut() {
            *sample = curve.evaluate_value(normalize(*sample));
        }
    }
    Ok((rgb, profile))
}

fn linear_rgb_from_icc(
    image: &DynamicImage,
    icc: &[u8],
    math_profile: Option<&ColorProfile>,
    gain_uses_base_channels: bool,
) -> Result<(Rgb32FImage, ColorProfile)> {
    let source = ColorProfile::new_from_slice(icc).context("invalid base ICC profile")?;
    if let Some(cicp) = source.cicp {
        let cicp = PngCicp {
            color_primaries: cicp.color_primaries as u8,
            transfer_function: cicp.transfer_characteristics as u8,
            matrix_coefficients: 0,
            full_range: true,
        };
        if is_hdr(cicp) {
            return linear_rgb_from_cicp(image, cicp);
        }
    }
    let target = if let Some(profile) = math_profile {
        profile.clone()
    } else if source.is_matrix_shaper() {
        // Keep the base RGB axes for map math with per-channel gain or offsets.
        linear_matrix_profile(&source)?
    } else {
        ensure!(
            !gain_uses_base_channels,
            "per-channel gain map requires a base ICC profile with RGB primaries"
        );
        linear_profile(PQ_CICP.color_primaries)?
    };
    let (samples, layout) = match source.color_space {
        DataColorSpace::Rgb => (image.to_rgb32f().into_raw(), Layout::Rgb),
        DataColorSpace::Gray => (
            image
                .to_luma16()
                .into_raw()
                .into_iter()
                .map(|sample| sample as f32 / MAX_SAMPLE)
                .collect(),
            Layout::Gray,
        ),
        _ => bail!("unsupported color space in base ICC profile"),
    };
    let mut converted = Rgb32FImage::new(image.width(), image.height());
    source
        .create_transform_f32(layout, &target, Layout::Rgb, TransformOptions::default())?
        .transform(&samples, converted.as_mut())?;
    Ok((converted, target))
}

fn encode_u16(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * MAX_SAMPLE).round() as u16
}

fn encoded_image(
    source: &DynamicImage,
    pixels: impl Iterator<Item = [u16; 3]>,
) -> Result<DynamicImage> {
    if source.color().has_alpha() {
        let mut rgba = source.to_rgba16();
        for (out, rgb) in rgba.pixels_mut().zip(pixels) {
            out.0[..3].copy_from_slice(&rgb);
        }
        Ok(DynamicImage::ImageRgba16(rgba))
    } else {
        ImageBuffer::from_raw(source.width(), source.height(), pixels.flatten().collect())
            .map(DynamicImage::ImageRgb16)
            .context("HDR output dimensions do not match its samples")
    }
}

#[derive(Clone, Copy, PartialEq)]
struct GainCurve {
    min: f32,
    max: f32,
    gamma_inverse: f32,
    base_offset: f32,
    alternate_offset: f32,
}

fn gain_curves(metadata: &IsoGainMap) -> Result<[GainCurve; 3]> {
    let min = metadata.map_min().map(|v| v as f32);
    let max = metadata.map_max().map(|v| v as f32);
    let gamma = metadata.gain_map_gamma().map(|v| v as f32);
    let base_offset = metadata.map_base_offset().map(|v| v as f32);
    let alternate_offset = metadata.map_alternate_offset().map(|v| v as f32);
    for c in 0..3 {
        ensure!(
            min[c].is_finite()
                && max[c].is_finite()
                && max[c] >= min[c]
                && min[c].exp2() > 0.0
                && max[c].exp2().is_finite()
                && gamma[c].is_finite()
                && gamma[c] > 0.0
                && gamma[c].recip().is_finite()
                && base_offset[c].is_finite()
                && alternate_offset[c].is_finite(),
            "invalid gain map metadata for channel {c}"
        );
    }
    Ok(std::array::from_fn(|c| GainCurve {
        min: min[c],
        max: max[c],
        gamma_inverse: 1.0 / gamma[c],
        base_offset: base_offset[c],
        alternate_offset: alternate_offset[c],
    }))
}

pub(crate) fn gain_map_to_hdr(
    image: &DynamicImage,
    gain_map: &ParsedGainMap,
    cicp: Option<PngCicp>,
    icc: Option<&[u8]>,
    output_cicp: PngCicp,
) -> Result<DynamicImage> {
    let encode = hdr_pixel_encoder(output_cicp)?;
    let metadata = &gain_map.metadata;
    let base_headroom = metadata.base_hdr_headroom();
    let alternate_headroom = metadata.alternate_hdr_headroom();
    ensure!(
        base_headroom.is_finite() && alternate_headroom.is_finite(),
        "invalid gain map headroom"
    );
    // Render whichever rendition has the greater HDR headroom. A backward
    // map's HDR image is already stored as the base.
    let apply_gain = !metadata.backward_direction && alternate_headroom > base_headroom;
    let curves = gain_curves(metadata)?;
    let alternate_profile = if apply_gain && !metadata.use_base_color_space {
        match gain_map.alternate_icc.as_deref() {
            Some(icc) => {
                let profile = ColorProfile::new_from_slice(icc)
                    .context("invalid gain-map alternate ICC profile")?;
                ensure!(
                    profile.is_matrix_shaper(),
                    "gain-map math requires an alternate ICC profile with RGB primaries"
                );
                Some(linear_matrix_profile(&profile)?)
            }
            None => gain_map
                .alternate_cicp
                .filter(|cicp| cicp.color_primaries != 2)
                .map(|cicp| linear_profile(cicp.color_primaries))
                .transpose()?,
        }
    } else {
        None
    };
    let scalar_gain = gain_map.image.color() == image::ColorType::L16
        && curves.iter().all(|curve| *curve == curves[0]);
    let (rgb, source_profile) = match icc {
        Some(icc) => linear_rgb_from_icc(
            image,
            icc,
            alternate_profile.as_ref(),
            apply_gain && !scalar_gain,
        )?,
        None => linear_rgb_from_cicp(image, cicp.unwrap_or(SRGB_CICP))?,
    };
    let math_profile = alternate_profile.as_ref().unwrap_or(&source_profile);
    let to_math = source_profile.transform_matrix(math_profile).to_f32();
    let to_output = math_profile
        .transform_matrix(&linear_profile(output_cicp.color_primaries)?)
        .to_f32();
    let map = if apply_gain {
        let source = &gain_map.image;
        let size = rgb.dimensions();
        ensure!(source.width() > 0 && source.height() > 0, "empty gain map");
        let map = match source {
            DynamicImage::ImageLuma16(_) | DynamicImage::ImageRgb16(_)
                if (source.width(), source.height()) == size =>
            {
                Cow::Borrowed(source)
            }
            DynamicImage::ImageLuma16(gray) => Cow::Owned(DynamicImage::ImageLuma16(
                resize_bilinear_u16::<1, _>(gray, size).context("resizing grayscale gain map")?,
            )),
            DynamicImage::ImageRgb16(rgb) => Cow::Owned(DynamicImage::ImageRgb16(
                resize_bilinear_u16::<3, _>(rgb, size).context("resizing RGB gain map")?,
            )),
            _ => bail!("expected a 16-bit monochrome or RGB gain map"),
        };
        Some(map)
    } else {
        None
    };
    let pixels = rgb.enumerate_pixels().map(|(x, y, rgb)| {
        let mut linear = to_math.mul_vector(Vector3f { v: rgb.0 });
        if let Some(map) = &map {
            let samples = match map.as_ref() {
                DynamicImage::ImageLuma16(gray) => [gray.get_pixel(x, y).0[0]; 3],
                DynamicImage::ImageRgb16(rgb) => rgb.get_pixel(x, y).0,
                _ => unreachable!(),
            };
            for (rgb, (curve, gain)) in linear.v.iter_mut().zip(curves.into_iter().zip(samples)) {
                let gain = (gain as f32 / MAX_SAMPLE).powf(curve.gamma_inverse);
                let stops = curve.min * (1.0 - gain) + curve.max * gain;
                *rgb = (*rgb + curve.base_offset) * stops.exp2() - curve.alternate_offset;
            }
        }
        encode(to_output.mul_vector(linear).v)
    });
    encoded_image(image, pixels)
}

pub(crate) fn convert_hdr(
    image: &DynamicImage,
    source_cicp: PngCicp,
    output_cicp: PngCicp,
) -> Result<DynamicImage> {
    ensure!(
        is_hdr(source_cicp),
        "input does not signal a PQ or HLG transfer function"
    );
    let encode = hdr_pixel_encoder(output_cicp)?;
    if source_cicp == output_cicp {
        return Ok(if image.color().has_alpha() {
            DynamicImage::ImageRgba16(image.to_rgba16())
        } else {
            DynamicImage::ImageRgb16(image.to_rgb16())
        });
    }
    let (rgb, source_profile) = linear_rgb_from_cicp(image, source_cicp)?;
    let to_output = source_profile
        .transform_matrix(&linear_profile(output_cicp.color_primaries)?)
        .to_f32();
    let pixels = rgb
        .pixels()
        .map(|pixel| encode(to_output.mul_vector(Vector3f { v: pixel.0 }).v));
    encoded_image(image, pixels)
}

pub(crate) fn hdr_to_gain_map(
    image: &DynamicImage,
    cicp: PngCicp,
) -> Result<(DynamicImage, ParsedGainMap)> {
    ensure!(
        is_hdr(cicp),
        "input does not signal a PQ or HLG transfer function"
    );
    let (mut rgb, source_profile) = linear_rgb_from_cicp(image, cicp)?;
    let to_bt2020 = source_profile
        .transform_matrix(&linear_profile(SDR_CICP.color_primaries)?)
        .to_f32();
    let mut peak = 0.0f32;
    for pixel in rgb.pixels_mut() {
        pixel.0 = to_bt2020
            .mul_vector(Vector3f { v: pixel.0 })
            .v
            .map(|sample| sample.max(0.0));
        peak = peak.max(peak_channel(*pixel));
    }

    // Keep midtones unchanged and smoothly compress brighter values to the
    // SDR range. Scaling all channels together preserves linear RGB ratios.
    const KNEE: f32 = 0.75;
    let compression = if peak > 1.0 {
        1.0 / (1.0 - KNEE) - 1.0 / (peak - KNEE)
    } else {
        0.0
    };
    const DENOMINATOR: u32 = 1_000_000;
    // Keep a positive range for formats that require HDR headroom even when
    // the source never exceeds SDR white. In that case all gains remain zero.
    let gain_max_n = (f64::from(peak.max(1.0)).log2() * f64::from(DENOMINATOR))
        .ceil()
        .max(1.0) as i32;
    let max_stops = gain_max_n as f32 / DENOMINATOR as f32;
    let mut samples = Vec::with_capacity(rgb.width() as usize * rgb.height() as usize);
    let base = rgb.pixels().map(|&pixel| {
        let above_knee = (peak_channel(pixel) - KNEE).max(0.0);
        let gain = 1.0 + compression * above_knee;
        if above_knee == 0.0 || compression == 0.0 {
            samples.push(0);
            return pixel
                .0
                .map(|sample| encode_u16(TransferFunction::Srgb.gamma(sample)));
        }
        let mapped_peak = KNEE + above_knee / gain;
        let ratio = peak_channel(pixel) / mapped_peak;
        samples.push(encode_u16(ratio.log2() / max_stops));
        pixel
            .0
            .map(|sample| encode_u16(TransferFunction::Srgb.gamma(sample / ratio)))
    });
    let base = encoded_image(image, base)?;
    let map = ImageBuffer::from_raw(image.width(), image.height(), samples)
        .context("gain map dimensions do not match its samples")?;
    let metadata = IsoGainMap {
        gain_map_min_n: [0; 3],
        gain_map_min_d: [1; 3],
        gain_map_max_n: [gain_max_n; 3],
        gain_map_max_d: [DENOMINATOR; 3],
        gain_map_gamma_n: [1; 3],
        gain_map_gamma_d: [1; 3],
        base_offset_n: [0; 3],
        base_offset_d: [1; 3],
        alternate_offset_n: [0; 3],
        alternate_offset_d: [1; 3],
        base_hdr_headroom_n: 0,
        base_hdr_headroom_d: 1,
        alternate_hdr_headroom_n: gain_max_n as u32,
        alternate_hdr_headroom_d: DENOMINATOR,
        backward_direction: false,
        use_base_color_space: true,
    };
    Ok((
        base,
        ParsedGainMap {
            image: DynamicImage::ImageLuma16(map),
            metadata,
            alternate_cicp: Some(PngCicp {
                transfer_function: cicp.transfer_function,
                ..PQ_CICP
            }),
            alternate_icc: None,
        },
    ))
}

fn peak_channel(pixel: Rgb<f32>) -> f32 {
    pixel.0[0].max(pixel.0[1]).max(pixel.0[2])
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::Rgba;

    #[test]
    fn round_trips_pq_midtones_highlights_and_alpha() {
        // ST 2084 gray values at 100 and 1000 cd/m².
        let dark = encode_u16(0.5080784);
        let light = encode_u16(0.7518271);
        let image = DynamicImage::ImageRgba16(ImageBuffer::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([dark, dark, dark, 31415])
            } else {
                Rgba([light, dark, 0, 61235])
            }
        }));
        let (base, gain_map) = hdr_to_gain_map(&image, PQ_CICP).unwrap();
        let gains = gain_map.image.as_luma16().unwrap().as_raw();
        assert_eq!(gains, &[0, u16::MAX]);
        assert_eq!(gain_map.alternate_cicp, Some(PQ_CICP));
        let peak_gain = gain_map.metadata.alternate_hdr_headroom().exp2();
        assert!((peak_gain - 1000.0 / 203.0).abs() < 0.001);
        let reconstructed =
            gain_map_to_hdr(&base, &gain_map, Some(SDR_CICP), None, PQ_CICP).unwrap();
        for (original, output) in image
            .as_rgba16()
            .unwrap()
            .pixels()
            .zip(reconstructed.as_rgba16().unwrap().pixels())
        {
            for c in 0..3 {
                assert!(
                    original.0[c].abs_diff(output.0[c]) <= 3,
                    "{original:?} != {output:?}"
                );
            }
            assert_eq!(original.0[3], output.0[3]);
        }
    }

    #[test]
    fn limited_range_pq_expands_black_and_white() {
        let image = DynamicImage::ImageRgb16(ImageBuffer::from_fn(2, 1, |x, _| {
            Rgb([if x == 0 { 16 << 8 } else { 235 << 8 }; 3])
        }));
        let (base, gain_map) = hdr_to_gain_map(
            &image,
            PngCicp {
                full_range: false,
                ..PQ_CICP
            },
        )
        .unwrap();
        let reconstructed =
            gain_map_to_hdr(&base, &gain_map, Some(SDR_CICP), None, PQ_CICP).unwrap();
        assert_eq!(
            reconstructed.as_rgb16().unwrap().as_raw(),
            &[0, 0, 0, u16::MAX, u16::MAX, u16::MAX]
        );
    }

    #[test]
    fn reconstructs_an_rgb_map_with_gamma_and_offsets() {
        let base = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(2, 2, Rgb([32768; 3])));
        let (_, mut gain_map) = hdr_to_gain_map(&base, PQ_CICP).unwrap();
        gain_map.image =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([16384, u16::MAX, 0])));
        let metadata = &mut gain_map.metadata;
        metadata.gain_map_max_n = [2; 3];
        metadata.gain_map_max_d = [1; 3];
        metadata.gain_map_gamma_n = [2; 3];
        metadata.base_offset_n = [1; 3];
        metadata.base_offset_d = [4; 3];
        metadata.alternate_offset_n = [1; 3];
        metadata.alternate_offset_d = [8; 3];
        metadata.alternate_hdr_headroom_n = 2;
        metadata.alternate_hdr_headroom_d = 1;
        let pq = gain_map_to_hdr(
            &base,
            &gain_map,
            Some(PngCicp {
                transfer_function: 8,
                ..SDR_CICP
            }),
            None,
            PQ_CICP,
        )
        .unwrap();
        let expected = [1.375, 2.875, 0.625];
        for pixel in pq.as_rgb16().unwrap().pixels() {
            for (sample, expected) in pixel.0.into_iter().zip(expected) {
                let linear = TransferFunction::PerceptualQuantizer
                    .linearize(sample as f32 / MAX_SAMPLE, PQ_PEAK_RELATIVE_TO_SDR_WHITE);
                assert!((linear - expected).abs() <= 0.001, "{linear} != {expected}");
            }
        }
    }

    #[test]
    fn renders_display_p3_base_with_its_icc() {
        let base = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([u16::MAX, 0, 0])));
        let (_, gain_map) = hdr_to_gain_map(&base, PQ_CICP).unwrap();
        let via_icc = gain_map_to_hdr(
            &base,
            &gain_map,
            None,
            Some(&ColorProfile::new_display_p3().encode().unwrap()),
            PQ_CICP,
        )
        .unwrap();
        let via_cicp = gain_map_to_hdr(
            &base,
            &gain_map,
            Some(PngCicp {
                color_primaries: 12,
                ..SDR_CICP
            }),
            None,
            PQ_CICP,
        )
        .unwrap();
        for (&icc, &cicp) in via_icc
            .as_rgb16()
            .unwrap()
            .as_raw()
            .iter()
            .zip(via_cicp.as_rgb16().unwrap().as_raw())
        {
            assert!(icc.abs_diff(cicp) <= 20, "{icc} differs from {cicp}");
        }
    }

    #[test]
    fn uses_alternate_icc_primaries_for_rgb_gain() {
        let base =
            DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([42_000, 22_000, 1])));
        let mut gain_map = ParsedGainMap {
            image: DynamicImage::ImageRgb16(ImageBuffer::from_pixel(1, 1, Rgb([u16::MAX, 0, 0]))),
            metadata: IsoGainMap {
                gain_map_min_d: [1; 3],
                gain_map_max_n: [1; 3],
                gain_map_max_d: [1; 3],
                gain_map_gamma_n: [1; 3],
                gain_map_gamma_d: [1; 3],
                base_offset_d: [1; 3],
                alternate_offset_d: [1; 3],
                base_hdr_headroom_d: 1,
                alternate_hdr_headroom_n: 1,
                alternate_hdr_headroom_d: 1,
                ..IsoGainMap::default()
            },
            alternate_cicp: Some(PngCicp {
                color_primaries: 12,
                ..PQ_CICP
            }),
            alternate_icc: None,
        };
        let via_cicp = gain_map_to_hdr(&base, &gain_map, Some(SRGB_CICP), None, PQ_CICP).unwrap();
        gain_map.alternate_cicp = Some(PQ_CICP);
        gain_map.alternate_icc = Some(ColorProfile::new_display_p3().encode().unwrap());
        let via_icc = gain_map_to_hdr(&base, &gain_map, Some(SRGB_CICP), None, PQ_CICP).unwrap();
        for (&icc, &cicp) in via_icc
            .as_rgb16()
            .unwrap()
            .as_raw()
            .iter()
            .zip(via_cicp.as_rgb16().unwrap().as_raw())
        {
            assert!(icc.abs_diff(cicp) <= 20, "{icc} differs from {cicp}");
        }
    }

    #[test]
    fn maps_hlg_reference_white_and_peak_to_pq() {
        let hlg = DynamicImage::ImageRgb16(ImageBuffer::from_fn(4, 1, |x, _| {
            let patches = [[0.0; 3], [0.75; 3], [1.0; 3], [0.75, 0.5, 0.25]];
            Rgb(patches[x as usize].map(encode_u16))
        }));
        let pq = convert_hdr(&hlg, HLG_CICP, PQ_CICP).unwrap();
        let pq_rgb = pq.as_rgb16().unwrap();
        // 75% HLG is 203.15 cd/m² under the BT.2100 reference EOTF,
        // and 100% is 1,000 cd/m². The colored patch checks that the
        // reference OOTF derives a single gain from scene luminance.
        let expected = [[0; 3], [38_061; 3], [49_271; 3], [37_060, 29_471, 21_342]];
        for (pixel, expected) in pq_rgb.pixels().zip(expected) {
            for (channel, target) in pixel.0.into_iter().zip(expected) {
                assert!(channel.abs_diff(target) <= 3, "{channel} != {target}");
            }
        }
        let restored = convert_hdr(&pq, PQ_CICP, HLG_CICP).unwrap();
        for (a, b) in restored
            .as_rgb16()
            .unwrap()
            .as_raw()
            .iter()
            .zip(hlg.as_rgb16().unwrap().as_raw())
        {
            assert!(a.abs_diff(*b) <= 3, "{a} != {b}");
        }
    }

    #[test]
    fn round_trips_hlg_gain_map_with_colored_highlights_and_alpha() {
        let hlg = DynamicImage::ImageRgba16(ImageBuffer::from_fn(2, 1, |x, _| {
            if x == 0 {
                Rgba([32_768, 32_768, 32_768, 1])
            } else {
                Rgba([62_258, 47_185, 20_971, 42_417])
            }
        }));
        let (base, gain_map) = hdr_to_gain_map(&hlg, HLG_CICP).unwrap();
        assert_eq!(gain_map.alternate_cicp, Some(HLG_CICP));
        let restored = gain_map_to_hdr(&base, &gain_map, Some(SDR_CICP), None, HLG_CICP).unwrap();
        for (before, after) in hlg
            .as_rgba16()
            .unwrap()
            .pixels()
            .zip(restored.as_rgba16().unwrap().pixels())
        {
            assert_eq!(before.0[3], after.0[3]);
            for channel in 0..3 {
                assert!(
                    before.0[channel].abs_diff(after.0[channel]) <= 3,
                    "{before:?} differs from {after:?}"
                );
            }
        }
    }

    #[test]
    fn limits_pq_highlights_in_hlg_without_changing_rgb_ratios() {
        let pq = DynamicImage::ImageRgb16(ImageBuffer::from_pixel(
            1,
            1,
            Rgb(encode_pq([10.0, 5.0, 2.5])),
        ));
        let hlg = convert_hdr(&pq, PQ_CICP, HLG_CICP).unwrap();
        assert_eq!(hlg.as_rgb16().unwrap().as_raw()[0], u16::MAX);
        let (rgb, _) = linear_rgb_from_cicp(&hlg, HLG_CICP).unwrap();
        let [r, g, b] = rgb.get_pixel(0, 0).0;
        assert!((r / g - 2.0).abs() < 0.001);
        assert!((r / b - 4.0).abs() < 0.001);
    }
}
