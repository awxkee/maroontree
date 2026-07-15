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
    Av2Encoder, BitDepth, ChromaSamplePosition, Cicp, MatrixCoefficients, Orientation, PlanarImage,
    Primaries, TransferFunction, TxPart,
};

fn lossless_tiles(width: usize, height: usize, threads: usize) -> (usize, usize) {
    let workers = threads.max(1);
    let cols = width.div_ceil(256).clamp(1, workers);
    let rows = height.div_ceil(256).clamp(1, workers.div_ceil(cols));
    (cols, rows)
}

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
    let color = Cicp {
        primaries: Primaries::Bt709,
        transfer: TransferFunction::Srgb,
        matrix: MatrixCoefficients::Identity,
        full_range: true,
        chroma_sample_position: ChromaSamplePosition::Unknown,
    };

    if chroma_choice != Chroma::C444 {
        return Err("Chroma mode is not supported for lossless".into());
    }

    let (tile_cols, tile_rows) = lossless_tiles(w, h, args.threads);
    let enc = Av2Encoder::with_bit_depth(
        0,
        match effective_depth {
            Depth::D8 => 8,
            Depth::D10 => 10,
            Depth::D12 => 12,
        },
    )
    .with_tiles(tile_cols, tile_rows)
    .with_txpart(TxPart::ThreeWay)
    .with_speed(args.speed.to_maroontreee())
    .with_threads(args.threads);
    let alpha_enc = Av2Encoder::with_bit_depth(
        0,
        match effective_depth {
            Depth::D8 => 8,
            Depth::D10 => 10,
            Depth::D12 => 12,
        },
    )
    .with_tiles(tile_cols, tile_rows)
    .with_txpart(TxPart::ThreeWay)
    .with_speed(args.speed.to_maroontreee())
    .with_threads(args.threads);

    if gray {
        let frame = match effective_depth {
            Depth::D8 => {
                let l = img.to_luma8().into_raw();
                enc.encode_image_400(&PlanarImage::from_luma(w, h, BitDepth::Eight, &l)?, &color)?
            }
            Depth::D10 => {
                let l = scale16_to_10(img.to_luma16().as_raw());
                enc.encode_image_400(&PlanarImage::from_luma(w, h, BitDepth::Ten, &l)?, &color)?
            }
            Depth::D12 => {
                let l = scale16_to_12(img.to_luma16().as_raw());
                enc.encode_image_400(&PlanarImage::from_luma(w, h, BitDepth::Twelve, &l)?, &color)?
            }
        };
        return Ok(Av2Encoder::wrap_avif(
            &frame,
            icc,
            exif,
            Orientation::Normal,
            None,
        )?);
    }

    let encode_color_8 = || -> Result<_, anyhow::Error> {
        let rgb8 = img.to_rgb8();
        let planar_image = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, rgb8.as_raw())?;
        enc.encode_image_444(&planar_image, &color)
            .map_err(|x| anyhow::anyhow!(x))
    };

    let encode_color_16 = |bit_depth: u8| -> Result<_, anyhow::Error> {
        let diff = 16 - bit_depth;
        let rgb16 = img.to_rgb16();
        let rgb_data = rgb16.iter().map(|&x| x >> diff).collect::<Vec<_>>();
        let depth = if bit_depth == 10 {
            BitDepth::Ten
        } else {
            BitDepth::Twelve
        };
        let planar_image = PlanarImage::from_interleaved_rgb(w, h, depth, &rgb_data)?;
        enc.encode_image_444(&planar_image, &color)
            .map_err(|x| anyhow::anyhow!(x))
    };

    if effective_depth == Depth::D8 {
        return if !alpha {
            let frame = encode_color_8()?;
            Ok(Av2Encoder::wrap_avif(
                &frame,
                icc,
                exif,
                Orientation::Normal,
                None,
            )?)
        } else {
            let a = extract_alpha(img.to_rgba8().as_raw(), w, h);
            let frame = encode_color_8()?;
            let alpha_frame = alpha_enc
                .encode_yuv400(&PlanarImage::from_luma(w, h, BitDepth::Eight, &a)?, &color)?;
            Ok(Av2Encoder::wrap_avif_alpha(
                &frame,
                &alpha_frame,
                icc,
                exif,
                Orientation::Normal,
                None,
            )?)
        };
    }

    if effective_depth == Depth::D10 {
        return if !alpha {
            let frame = encode_color_16(10)?;
            Ok(Av2Encoder::wrap_avif(
                &frame,
                icc,
                exif,
                Orientation::Normal,
                None,
            )?)
        } else {
            let a = extract_alpha(&scale16_to_10(img.to_rgba16().as_raw()), w, h);
            let frame = encode_color_16(10)?;
            let alpha_frame = alpha_enc
                .encode_yuv400(&PlanarImage::from_luma(w, h, BitDepth::Ten, &a)?, &color)?;
            Ok(Av2Encoder::wrap_avif_alpha(
                &frame,
                &alpha_frame,
                icc,
                exif,
                Orientation::Normal,
                None,
            )?)
        };
    }

    // Depth::D12
    if !alpha {
        let frame = encode_color_16(12)?;
        Ok(Av2Encoder::wrap_avif(
            &frame,
            icc,
            exif,
            Orientation::Normal,
            None,
        )?)
    } else {
        let a = extract_alpha(&scale16_to_12(img.to_rgba16().as_raw()), w, h);
        let frame = encode_color_16(12)?;
        let alpha_frame = alpha_enc
            .encode_yuv400(&PlanarImage::from_luma(w, h, BitDepth::Twelve, &a)?, &color)?;
        Ok(Av2Encoder::wrap_avif_alpha(
            &frame,
            &alpha_frame,
            icc,
            exif,
            Orientation::Normal,
            None,
        )?)
    }
}
