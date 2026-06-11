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
    Av2Encoder, BitDepth, ChromaSamplePosition, ColorEncoding, MatrixCoefficients, PlanarImage,
    Primaries, TransferFunction,
};
use yuv::{
    YuvChromaSubsampling, YuvPlanarImageMut, YuvRange, rgb_to_ycgco444, rgb10_to_icgc410,
    rgba12_to_icgc412,
};

fn extract_alpha<T: Copy + Default>(rgba: &[T], w: usize, h: usize) -> Vec<T> {
    let npx = w * h;
    let mut alpha = vec![T::default(); npx];
    for (px, alpha) in rgba.as_chunks::<4>().0.iter().zip(alpha.iter_mut()) {
        *alpha = px[3];
    }
    alpha
}

pub(crate) fn encode_av2_lossless_image(
    img: &image::DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    let chroma_choice = args.chroma.unwrap_or(Chroma::C444);
    let color = ColorEncoding {
        primaries: Primaries::Bt709,
        transfer: TransferFunction::Srgb,
        matrix: MatrixCoefficients::YCgCo,
        full_range: true,
        chroma_sample_position: ChromaSamplePosition::Unknown,
    };

    if chroma_choice != Chroma::C444 {
        return Err("Chroma mode is not supported for lossless".into());
    }

    let enc = Av2Encoder::new(0);
    let alpha_enc = Av2Encoder::new(0);

    if gray {
        let frame = match effective_depth {
            Depth::D8 => {
                let l = img.to_luma8().into_raw();
                enc.encode_image_400(
                    &PlanarImage::from_luma(w, h, BitDepth::Eight, &l)?,
                    &color,
                    args.threads,
                )?
            }
            Depth::D10 => {
                let l = scale16_to_10(img.to_luma16().as_raw());
                enc.encode_image_400(
                    &PlanarImage::from_luma(w, h, BitDepth::Ten, &l)?,
                    &color,
                    args.threads,
                )?
            }
            Depth::D12 => {
                let l = scale16_to_12(img.to_luma16().as_raw());
                enc.encode_image_400(
                    &PlanarImage::from_luma(w, h, BitDepth::Twelve, &l)?,
                    &color,
                    args.threads,
                )?
            }
        };
        return Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?);
    }

    let encode_color_8 = || -> Result<_, anyhow::Error> {
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
        enc.encode_image_444(&planar_image, &color, args.threads)
            .map_err(|x| anyhow::anyhow!(x))
    };

    let encode_color_16 = |bit_depth: u8| -> Result<_, anyhow::Error> {
        let diff = 16 - bit_depth;
        let mut planar_image: YuvPlanarImageMut<u16>;
        if bit_depth == 10 {
            planar_image =
                YuvPlanarImageMut::alloc(img.width(), img.height(), YuvChromaSubsampling::Yuv444);
            let rgb16 = img.to_rgb16();
            let rgb_data = rgb16.iter().map(|&x| x >> diff).collect::<Vec<_>>();
            rgb10_to_icgc410(
                &mut planar_image,
                &rgb_data,
                rgb16.width() * 3,
                YuvRange::Full,
            )
            .map_err(|x| anyhow::anyhow!(x))?;
        } else {
            planar_image =
                YuvPlanarImageMut::alloc(img.width(), img.height(), YuvChromaSubsampling::Yuv444);
            let rgb16 = img.to_rgba16();
            let rgb_data = rgb16.iter().map(|&x| x >> diff).collect::<Vec<_>>();
            rgba12_to_icgc412(
                &mut planar_image,
                &rgb_data,
                rgb16.width() * 4,
                YuvRange::Full,
            )
            .map_err(|x| anyhow::anyhow!(x))?;
        }
        let planar_image = PlanarImage {
            width: img.width() as usize,
            height: img.height() as usize,
            bit_depth: BitDepth::Eight,
            planes: [
                planar_image.y_plane.borrow().to_vec(),
                planar_image.u_plane.borrow().to_vec(),
                planar_image.v_plane.borrow().to_vec(),
                vec![],
            ],
        };
        enc.encode_image_444(&planar_image, &color, args.threads)
            .map_err(|x| anyhow::anyhow!(x))
    };

    if effective_depth == Depth::D8 {
        return if !alpha {
            let frame = encode_color_8()?;
            Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
        } else {
            let a = extract_alpha(&scale16_to_10(img.to_rgba16().as_raw()), w, h);
            let frame = encode_color_8()?;
            let alpha_frame = alpha_enc.encode_yuv400(
                &PlanarImage::from_luma(w, h, BitDepth::Eight, &a)?,
                &color,
                args.threads,
            )?;
            Ok(Av2Encoder::wrap_avif_alpha(
                &frame,
                &alpha_frame,
                icc,
                exif,
            )?)
        };
    }

    if effective_depth == Depth::D10 {
        return if !alpha {
            let frame = encode_color_16(10)?;
            Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
        } else {
            let a = extract_alpha(&scale16_to_10(img.to_rgba16().as_raw()), w, h);
            let frame = encode_color_16(10)?;
            let alpha_frame = alpha_enc.encode_yuv400(
                &PlanarImage::from_luma(w, h, BitDepth::Ten, &a)?,
                &color,
                args.threads,
            )?;
            Ok(Av2Encoder::wrap_avif_alpha(
                &frame,
                &alpha_frame,
                icc,
                exif,
            )?)
        };
    }

    // Depth::D12
    if !alpha {
        let frame = encode_color_16(12)?;
        Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
    } else {
        let a = extract_alpha(&scale16_to_12(img.to_rgba16().as_raw()), w, h);
        let frame = encode_color_16(12)?;
        let alpha_frame = alpha_enc.encode_yuv400(
            &PlanarImage::from_luma(w, h, BitDepth::Twelve, &a)?,
            &color,
            args.threads,
        )?;
        Ok(Av2Encoder::wrap_avif_alpha(
            &frame,
            &alpha_frame,
            icc,
            exif,
        )?)
    }
}
