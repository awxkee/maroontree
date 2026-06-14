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
use crate::{Args, Chroma, Depth, has_alpha_channel, is_gray, scale16_to_10, scale16_to_12};
use maroontree::{
    BitDepth, ChromaFormat, ChromaSamplePosition, Cicp, EncodeConfig, MatrixCoefficients,
    PlanarImage, Primaries, TransferFunction, encode_lossless, encode_lossless_gray,
    encode_lossless_gray_alpha, encode_lossless_with_alpha,
};
use yuv::{
    YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, rgb_to_ycgco444, rgb10_to_icgc410,
    rgba_to_ycgco444, rgba12_to_icgc412,
};

pub(crate) fn encode_av1_lossless(
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
    if chroma_fmt != ChromaFormat::Yuv444 {
        return Err(anyhow::anyhow!(
            "Chroma format doesn't match yuv444 for lossless"
        ));
    }

    let color = Cicp {
        primaries: Primaries::Bt709,
        transfer: TransferFunction::Srgb,
        matrix: MatrixCoefficients::YCgCo,
        full_range: true,
        chroma_sample_position: ChromaSamplePosition::Unknown,
    };

    let mut cfg = EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_cicp(color)
        .with_threads(args.threads)
        .with_speed(args.speed.to_maroontreee());

    if let Some(icc) = icc {
        cfg = cfg.with_icc_profile(icc.to_vec());
    }
    if let Some(exif) = exif {
        cfg = cfg.with_exif(exif.to_vec());
    }

    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    Ok(match (effective_depth, gray, alpha) {
        (Depth::D8, true, alpha) => {
            if alpha {
                encode_lossless_gray_alpha(
                    &PlanarImage::from_interleaved_gray_alpha(
                        img.width() as usize,
                        img.height() as usize,
                        BitDepth::Eight,
                        &&img.to_luma8(),
                    )?,
                    &cfg,
                )?
            } else {
                encode_lossless_gray(
                    &PlanarImage::from_interleaved_gray_alpha(
                        img.width() as usize,
                        img.height() as usize,
                        BitDepth::Eight,
                        &img.to_luma_alpha8(),
                    )?,
                    &cfg,
                )?
            }
        }
        (Depth::D8, false, false) => {
            let rgb8 = img.to_rgb8();
            let mut planar_image =
                YuvPlanarImageMut::alloc(rgb8.width(), rgb8.height(), YuvChromaSubsampling::Yuv444);
            rgb_to_ycgco444(&mut planar_image, &rgb8, rgb8.width() * 3, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;
            let planar_image = PlanarImage {
                width: rgb8.width() as usize,
                height: rgb8.height() as usize,
                bit_depth: BitDepth::Eight,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    vec![],
                ],
            };
            encode_lossless(&planar_image, &cfg)?
        }
        (Depth::D8, false, true) => {
            let rgba8 = img.to_rgba8();
            let mut planar_image = YuvPlanarImageMut::alloc(
                rgba8.width(),
                rgba8.height(),
                YuvChromaSubsampling::Yuv444,
            );
            rgba_to_ycgco444(&mut planar_image, &rgba8, rgba8.width() * 4, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;

            let alpha_chan = rgba8
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| x[3])
                .collect::<Vec<_>>();

            let planar_image = PlanarImage {
                width: rgba8.width() as usize,
                height: rgba8.height() as usize,
                bit_depth: BitDepth::Eight,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    alpha_chan,
                ],
            };
            encode_lossless_with_alpha(&planar_image, &cfg)?
        }
        (Depth::D10, true, alpha) => {
            if alpha {
                encode_lossless_gray_alpha(
                    &PlanarImage::from_interleaved_gray_alpha(
                        img.width() as usize,
                        img.height() as usize,
                        BitDepth::Ten,
                        &scale16_to_10(&img.to_luma_alpha16()),
                    )?,
                    &cfg,
                )?
            } else {
                encode_lossless_gray(
                    &PlanarImage {
                        width: img.width() as usize,
                        height: img.height() as usize,
                        bit_depth: BitDepth::Ten,
                        planes: [
                            scale16_to_10(&img.to_luma16()).to_vec(),
                            vec![],
                            vec![],
                            vec![],
                        ],
                    },
                    &cfg,
                )?
            }
        }
        (Depth::D10, false, false) => {
            let rgb10 = img.to_rgb16();
            let src10 = scale16_to_10(&rgb10);
            let mut planar_image = YuvPlanarImageMut::alloc(
                rgb10.width(),
                rgb10.height(),
                YuvChromaSubsampling::Yuv444,
            );
            rgb10_to_icgc410(&mut planar_image, &src10, rgb10.width() * 3, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;
            let planar_image = PlanarImage {
                width: rgb10.width() as usize,
                height: rgb10.height() as usize,
                bit_depth: BitDepth::Ten,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    vec![],
                ],
            };
            encode_lossless(&planar_image, &cfg)?
        }
        (Depth::D10, false, true) => {
            let rgb10 = img.to_rgba16();
            let src10 = scale16_to_10(&rgb10);
            let mut planar_image = YuvPlanarImageMut::alloc(
                rgb10.width(),
                rgb10.height(),
                YuvChromaSubsampling::Yuv444,
            );
            rgb10_to_icgc410(&mut planar_image, &src10, rgb10.width() * 4, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;

            let alpha_chan = src10
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| x[3])
                .collect::<Vec<_>>();

            let planar_image = PlanarImage {
                width: rgb10.width() as usize,
                height: rgb10.height() as usize,
                bit_depth: BitDepth::Ten,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    alpha_chan,
                ],
            };
            encode_lossless_with_alpha(&planar_image, &cfg)?
        }
        (Depth::D12, true, alpha) => {
            if alpha {
                encode_lossless_gray_alpha(
                    &PlanarImage::from_interleaved_gray_alpha(
                        img.width() as usize,
                        img.height() as usize,
                        BitDepth::Twelve,
                        &scale16_to_12(img.to_luma_alpha16().as_raw()),
                    )?,
                    &cfg,
                )?
            } else {
                encode_lossless_gray(
                    &PlanarImage::from_luma(
                        img.width() as usize,
                        img.height() as usize,
                        BitDepth::Twelve,
                        &scale16_to_12(img.to_luma16().as_raw()),
                    )?,
                    &cfg,
                )?
            }
        }
        (Depth::D12, false, false) => {
            let rgb12 = img.to_rgb16();
            let src12 = scale16_to_12(&rgb12);
            let mut planar_image = YuvPlanarImageMut::alloc(
                rgb12.width(),
                rgb12.height(),
                YuvChromaSubsampling::Yuv444,
            );
            rgba12_to_icgc412(&mut planar_image, &src12, rgb12.width() * 4, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;
            let planar_image = PlanarImage {
                width: rgb12.width() as usize,
                height: rgb12.height() as usize,
                bit_depth: BitDepth::Twelve,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    vec![],
                ],
            };
            encode_lossless(&planar_image, &cfg)?
        }
        (Depth::D12, false, true) => {
            let rgb12 = img.to_rgba16();
            let src12 = scale16_to_12(&rgb12);
            let mut planar_image = YuvPlanarImageMut::alloc(
                rgb12.width(),
                rgb12.height(),
                YuvChromaSubsampling::Yuv444,
            );
            rgba12_to_icgc412(&mut planar_image, &src12, rgb12.width() * 4, YuvRange::Full)
                .map_err(|x| anyhow::anyhow!(x))?;

            let alpha_chan = src12
                .as_chunks::<4>()
                .0
                .iter()
                .map(|x| x[3])
                .collect::<Vec<_>>();

            let planar_image = PlanarImage {
                width: rgb12.width() as usize,
                height: rgb12.height() as usize,
                bit_depth: BitDepth::Twelve,
                planes: [
                    planar_image.y_plane.borrow().to_vec(),
                    planar_image.u_plane.borrow().to_vec(),
                    planar_image.v_plane.borrow().to_vec(),
                    alpha_chan,
                ],
            };
            encode_lossless_with_alpha(&planar_image, &cfg)?
        }
    })
}
