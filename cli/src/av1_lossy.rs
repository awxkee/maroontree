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
use crate::{Args, Chroma, Depth, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12};
use maroontree::{
    BitDepth, ChromaFormat, Cicp, EncodeConfig, PlanarImage, encode_gray8, encode_gray10,
    encode_gray12, encode_rgb8, encode_rgb10, encode_rgb12, encode_rgba8_with_alpha,
    encode_rgba10_with_alpha, encode_rgba12_with_alpha,
};

pub(crate) fn encode_av1(
    img: &image::DynamicImage,
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

    if args.lossless {
        return encode_av1_lossless(img, args, color_type, effective_depth, icc, exif);
    }

    let mut cfg = EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_cicp(Cicp::srgb_ycbcr())
        .with_threads(args.threads);

    if let Some(icc) = icc {
        cfg = cfg.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = exif {
        cfg = cfg.with_exif(exif.to_vec());
    }

    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    Ok(match (effective_depth, gray, alpha) {
        (Depth::D8, true, _) => encode_gray8(
            &PlanarImage::from_interleaved_rgb(
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
