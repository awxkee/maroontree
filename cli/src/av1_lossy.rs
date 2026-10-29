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
use crate::av1_lossless::encode_av1_lossless;
use crate::{
    Args, Chroma, Depth, PngCicp, Qmatrix, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12,
};
use maroontree::{
    BitDepth, ChromaFormat, Cicp, EncodeConfig, PlanarImage, encode_gray8, encode_gray10,
    encode_gray12, encode_rgb8, encode_rgb10, encode_rgb12, encode_rgba8_with_alpha,
    encode_rgba10_with_alpha, encode_rgba12_with_alpha,
};

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_av1(
    img: &image::DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
    png_cicp: Option<PngCicp>,
    gain_map: Option<maroontree::GainMap>,
) -> Result<Vec<u8>, anyhow::Error> {
    let chroma_fmt = match args.chroma.unwrap_or(Chroma::C420) {
        Chroma::C444 => ChromaFormat::Yuv444,
        Chroma::C422 => ChromaFormat::Yuv422,
        Chroma::C420 => ChromaFormat::Yuv420,
    };

    if args.lossless {
        return encode_av1_lossless(
            img,
            args,
            color_type,
            effective_depth,
            icc,
            exif,
            png_cicp,
            gain_map,
        );
    }

    let cicp = png_cicp
        .map(|c| c.to_maroontree(maroontree::MatrixCoefficients::Smpte170m, true))
        .unwrap_or_else(Cicp::srgb_ycbcr);
    let mut cfg = EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_cicp(cicp)
        .with_threads(args.threads)
        .with_speed(args.speed.to_maroontreee())
        .with_updating_cdf(args.updating_cdf)
        .with_screen_content(args.screen_content)
        .with_intrabc(args.intrabc)
        .with_cdef(args.cdef)
        .with_wiener(args.wiener);
    cfg = match args.qmatrix {
        Some(Qmatrix::Auto) => cfg.with_quantization_matrices(true),
        Some(Qmatrix::Level(level)) => cfg.with_qmatrix_level(level),
        None => cfg,
    };

    for option in &args.overrides {
        option.apply_av1(&mut cfg).map_err(anyhow::Error::msg)?;
    }

    if let Some(icc) = icc {
        cfg = cfg.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = exif {
        cfg = cfg.with_exif(exif.to_vec());
    }
    cfg.gain_map = gain_map.filter(|_| !args.no_gain_map);

    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    Ok(match (effective_depth, gray, alpha) {
        (Depth::D16, _, _) => anyhow::bail!("16-bit output is supported only by JXL"),
        (Depth::D8, true, _) => encode_gray8(
            &PlanarImage::from_luma(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Eight,
                img.to_luma8().as_raw(),
            )?,
            &cfg,
        )?,
        (Depth::D8, false, false) => encode_rgb8(
            &PlanarImage::from_interleaved_rgb(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Eight,
                img.to_rgb8().as_raw(),
            )?,
            &cfg,
        )?,
        (Depth::D8, false, true) => encode_rgba8_with_alpha(
            &PlanarImage::from_interleaved_rgba(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Eight,
                img.to_rgba8().as_raw(),
            )?,
            &cfg,
        )?,
        (Depth::D10, true, _) => encode_gray10(
            &PlanarImage::from_luma(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Ten,
                &scale16_to_10(img.to_luma16().as_raw()),
            )?,
            &cfg,
        )?,
        (Depth::D10, false, false) => encode_rgb10(
            &PlanarImage::from_interleaved_rgb(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Ten,
                &scale16_to_10(img.to_rgb16().as_raw()),
            )?,
            &cfg,
        )?,
        (Depth::D10, false, true) => encode_rgba10_with_alpha(
            &PlanarImage::from_interleaved_rgba(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Ten,
                &scale16_to_10(img.to_rgba16().as_raw()),
            )?,
            &cfg,
        )?,
        (Depth::D12, true, _) => encode_gray12(
            &PlanarImage::from_luma(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Twelve,
                &scale16_to_12(img.to_luma16().as_raw()),
            )?,
            &cfg,
        )?,
        (Depth::D12, false, false) => encode_rgb12(
            &PlanarImage::from_interleaved_rgb(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Twelve,
                &scale16_to_12(img.to_rgb16().as_raw()),
            )?,
            &cfg,
        )?,
        (Depth::D12, false, true) => encode_rgba12_with_alpha(
            &PlanarImage::from_interleaved_rgba(
                img.width() as usize,
                img.height() as usize,
                BitDepth::Twelve,
                &scale16_to_12(img.to_rgba16().as_raw()),
            )?,
            &cfg,
        )?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overrides_reach_lossy_and_lossless_av1_after_common_flags() {
        let image = image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(8, 8, |x, y| {
            image::Rgb([(x * 31) as u8, (y * 31) as u8, 128])
        }));
        for lossless in [false, true] {
            let mut options = vec![
                "cli",
                "-s",
                "slow",
                "-t",
                "2",
                "-q",
                "99",
                "in.png",
                "-O",
                "av1:speed=fast,threads=1,quality=60,updating-cdf=false",
            ];
            let mut reference = vec![
                "cli",
                "-s",
                "fast",
                "-t",
                "1",
                "-q",
                "60",
                "--no-cdf-update",
                "in.png",
            ];
            if lossless {
                options.push("--lossless");
                reference.push("--lossless");
            }
            let actual = crate::args::try_parse(&options).unwrap();
            let expected = crate::args::try_parse(&reference).unwrap();
            let encode = |args: &Args| {
                encode_av1(
                    &image,
                    args,
                    image.color(),
                    Depth::D8,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()
            };
            assert_eq!(encode(&actual), encode(&expected));
        }
    }
}
