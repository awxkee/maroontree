/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice, this
 *     list of conditions and the following disclaimer.
 *
 * 2.  Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 * 3.  Neither the name of the copyright holder nor the names of its contributors may
 *     be used to endorse or promote products derived from this software without
 *     specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
 * OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 */

//! # avif — CLI encoder
//!
//! ```text
//! Usage: avif [OPTIONS] <INPUT> [OUTPUT]
//!
//! Arguments:
//!   <INPUT>   Source image (PNG, JPEG, WebP, TIFF, GIF, BMP, …)
//!   [OUTPUT]  Destination .avif  [default: INPUT stem + ".avif"]
//!
//! Options:
//!   -e, --encoder <av1|av2>     Encoder backend                 [default: av1]
//!   -q, --quality <1-100>       Encode quality (higher = better) [default: 80(AV1)/60(AV2)]
//!       --lossless              Pixel-perfect lossless (AV2 only)
//!   -c, --chroma <444|422|420>  Chroma subsampling              [default: 420]
//!   -d, --depth <8|10|12>       Output bit depth (auto-detect from source)
//!   -t, --threads <N>           Worker threads; 0 = all cores   [default: all]
//!       --no-alpha              Discard alpha channel
//!       --no-exif               Strip EXIF from output
//!       --no-icc                Strip ICC profile from output
//!   -v, --verbose               Print dimensions, timing, file size
//!   -h, --help                  Print this help
//! ```

mod av1_lossless;
mod av2_lossless;

use crate::av1_lossless::encode_av1_lossless;
use crate::av2_lossless::encode_av2_lossless_image;
use img_parts::{ImageEXIF, ImageICC, jpeg::Jpeg, png::Png, webp::WebP};
use maroontree::{
    Av2Encoder, BitDepth, ChromaFormat, ColorEncoding, EncodeConfig, PlanarImage, TxPart,
    av2_map_quality, encode_gray8, encode_gray10, encode_gray12, encode_rgb8, encode_rgb10,
    encode_rgb12, encode_rgba8_with_alpha, encode_rgba10_with_alpha, encode_rgba12_with_alpha,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoder {
    Av1,
    Av2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Chroma {
    C444,
    C422,
    C420,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Depth {
    D8,
    D10,
    D12,
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    encoder: Encoder,
    quality: u8,
    lossless: bool,
    chroma: Option<Chroma>,
    depth: Option<Depth>,
    threads: usize,
    no_alpha: bool,
    no_exif: bool,
    no_icc: bool,
    verbose: bool,
}

fn usage() -> ! {
    eprintln!(
        "\
Usage: avif [OPTIONS] <INPUT> [OUTPUT]

Arguments:
  <INPUT>   Source image (PNG, JPEG, WebP, TIFF, …)
  [OUTPUT]  Output .avif  [default: input stem + \".avif\"]

Options:
  -e, --encoder <av1|av2>     Encoder backend                 [default: av1]
  -q, --quality <1-100>       Quality (higher = better)       [default: 80(AV1)/60(AV2)]
      --lossless              Pixel-perfect lossless (AV2 only)
  -c, --chroma <444|422|420>  Chroma subsampling              [default: 420]
  -d, --depth <8|10|12>       Output bit depth (auto from source)
  -t, --threads <N>           Worker threads; 0 = all cores   [default: all]
      --no-alpha              Discard alpha channel
      --no-exif               Strip EXIF metadata from output
      --no-icc                Strip ICC colour profile from output
  -v, --verbose               Print timing and file stats
  -h, --help                  Print this help"
    );
    std::process::exit(0);
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn basic_concurrency() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1).peekable();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut encoder = Encoder::Av1;
    let mut quality: Option<u8> = None;
    let mut lossless = false;
    let mut chroma: Option<Chroma> = None;
    let mut depth: Option<Depth> = None;
    let mut threads: usize = basic_concurrency();
    let mut no_alpha = false;
    let mut no_exif = false;
    let mut no_icc = false;
    let mut verbose = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-v" | "--verbose" => verbose = true,
            "--lossless" => lossless = true,
            "--no-alpha" => no_alpha = true,
            "--no-exif" => no_exif = true,
            "--no-icc" => no_icc = true,
            "-e" | "--encoder" => match args.next().unwrap_or_default().as_str() {
                "av1" => encoder = Encoder::Av1,
                "av2" => encoder = Encoder::Av2,
                other => die(format!("unknown encoder '{other}'; use av1 or av2")),
            },
            "-q" | "--quality" => {
                let v = args.next().unwrap_or_default();
                let new_quality = v
                    .parse::<u8>()
                    .unwrap_or_else(|_| die(format!("invalid quality '{v}'; expected 1-100")));
                if new_quality > 100 {
                    die("quality must be 1-100");
                }
                quality = Some(new_quality);
            }
            "-c" | "--chroma" => {
                chroma = Some(match args.next().unwrap_or_default().as_str() {
                    "444" => Chroma::C444,
                    "422" => Chroma::C422,
                    "420" => Chroma::C420,
                    other => die(format!("unknown chroma '{other}'; use 444, 422, or 420")),
                })
            }
            "-d" | "--depth" => {
                depth = Some(match args.next().unwrap_or_default().as_str() {
                    "8" => Depth::D8,
                    "10" => Depth::D10,
                    "12" => Depth::D12,
                    other => die(format!("unsupported depth '{other}'; use 8, 10, or 12")),
                })
            }
            "-t" | "--threads" => {
                let v = args
                    .next()
                    .unwrap_or_else(|| format!("{}", basic_concurrency()));
                threads = v
                    .parse()
                    .unwrap_or_else(|_| die(format!("invalid thread count '{v}'")));
                if threads == 0 {
                    threads = basic_concurrency();
                }
            }
            path => {
                if input.is_none() {
                    input = Some(PathBuf::from(path));
                } else if output.is_none() {
                    output = Some(PathBuf::from(path));
                } else {
                    die(format!("unexpected argument '{path}'"));
                }
            }
        }
    }

    let input = input.unwrap_or_else(|| die("no input file specified; use --help"));
    let output = output.unwrap_or_else(|| {
        let stem = input.file_stem().unwrap_or_default();
        let parent = input.parent().unwrap_or(Path::new("."));
        parent.join(stem).with_extension("avif")
    });

    Args {
        input,
        output,
        encoder,
        quality: match quality {
            None => match encoder {
                Encoder::Av1 => 80,
                Encoder::Av2 => 60,
            },
            Some(v) => v,
        },
        lossless,
        chroma: match chroma {
            None => Some(if lossless { Chroma::C444 } else { Chroma::C422 }),
            Some(v) => Some(v),
        },
        depth,
        threads,
        no_alpha,
        no_exif,
        no_icc,
        verbose,
    }
}

fn is_16bit(ct: image::ColorType) -> bool {
    use image::ColorType::*;
    matches!(ct, L16 | La16 | Rgb16 | Rgba16)
}

fn has_alpha_channel(ct: image::ColorType) -> bool {
    use image::ColorType::*;
    matches!(ct, La8 | Rgba8 | La16 | Rgba16)
}

fn is_gray(ct: image::ColorType) -> bool {
    use image::ColorType::*;
    matches!(ct, L8 | L16 | La8 | La16)
}

fn scale16_to_10(src: &[u16]) -> Vec<u16> {
    src.iter().map(|&v| v >> 6).collect()
}
fn scale16_to_12(src: &[u16]) -> Vec<u16> {
    src.iter().map(|&v| v >> 4).collect()
}

fn read_exif(path: &Path) -> Option<Vec<u8>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let raw = std::fs::read(path).ok()?;
    match ext.as_deref() {
        Some("jpg") | Some("jpeg") => Jpeg::from_bytes(raw.into())
            .ok()?
            .exif()
            .map(|b| b.to_vec()),
        Some("png") => Png::from_bytes(raw.into()).ok()?.exif().map(|b| b.to_vec()),
        Some("webp") => WebP::from_bytes(raw.into())
            .ok()?
            .exif()
            .map(|b| b.to_vec()),
        _ => None,
    }
}

fn read_icc(path: &Path) -> Option<Vec<u8>> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase());
    let raw = std::fs::read(path).ok()?;
    match ext.as_deref() {
        Some("jpg") | Some("jpeg") => Jpeg::from_bytes(raw.into())
            .ok()?
            .icc_profile()
            .map(|b| b.to_vec()),
        Some("png") => Png::from_bytes(raw.into())
            .ok()?
            .icc_profile()
            .map(|b| b.to_vec()),
        Some("webp") => WebP::from_bytes(raw.into())
            .ok()?
            .icc_profile()
            .map(|b| b.to_vec()),
        _ => None,
    }
}

fn encode_av1(
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
        .with_cicp(ColorEncoding::srgb_ycbcr())
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

fn deinterleave_rgba<T: Copy + Default>(rgba: &[T], w: usize, h: usize) -> (Vec<T>, Vec<T>) {
    let npx = w * h;
    let mut rgb = vec![T::default(); npx * 3];
    let mut alpha = vec![T::default(); npx];
    for ((px, rgb), alpha) in rgba
        .as_chunks::<4>()
        .0
        .iter()
        .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
        .zip(alpha.iter_mut())
    {
        rgb[0] = px[0];
        rgb[1] = px[1];
        rgb[2] = px[2];
        *alpha = px[3];
    }
    (rgb, alpha)
}

fn encode_av2(
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

    let base_q = if args.lossless {
        return encode_av2_lossless_image(img, args, color_type, effective_depth, icc, exif);
    } else {
        av2_map_quality(args.quality)
    };
    let chroma_choice = args
        .chroma
        .unwrap_or(if gray { Chroma::C444 } else { Chroma::C420 });
    let color = ColorEncoding::srgb_ycbcr();

    let enc = Av2Encoder::new(base_q)
        .with_tiles(8, 8)
        .with_txpart(TxPart::ThreeWay);
    let alpha_enc = Av2Encoder::new(0)
        .with_tiles(8, 8)
        .with_txpart(TxPart::ThreeWay);

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

    let encode_color_8 = |pimg: &PlanarImage<u8>| -> Result<_, Box<dyn std::error::Error>> {
        Ok(match (chroma_choice, args.lossless) {
            (_, true) | (Chroma::C444, _) => enc.encode_image_444(pimg, &color, args.threads)?,
            (Chroma::C420, false) => enc.encode_image_420(pimg, &color, args.threads)?,
            (Chroma::C422, false) => enc.encode_image_422(pimg, &color, args.threads)?,
        })
    };

    let encode_color_16 = |pimg: &PlanarImage<u16>| -> Result<_, Box<dyn std::error::Error>> {
        Ok(match (chroma_choice, args.lossless) {
            (_, true) | (Chroma::C444, _) => enc.encode_image_444(pimg, &color, args.threads)?,
            (Chroma::C420, false) => enc.encode_image_420(pimg, &color, args.threads)?,
            (Chroma::C422, false) => enc.encode_image_422(pimg, &color, args.threads)?,
        })
    };

    if effective_depth == Depth::D8 {
        return if !alpha {
            let rgb = img.to_rgb8().into_raw();
            let frame = encode_color_8(&PlanarImage::from_interleaved_rgb(
                w,
                h,
                BitDepth::Eight,
                &rgb,
            )?)?;
            Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
        } else {
            let (rgb, a) = deinterleave_rgba(&img.to_rgba8().into_raw(), w, h);
            let frame = encode_color_8(&PlanarImage::from_interleaved_rgb(
                w,
                h,
                BitDepth::Eight,
                &rgb,
            )?)?;
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
            let raw16 = scale16_to_10(img.to_rgb16().as_raw());
            let frame = encode_color_16(&PlanarImage::from_interleaved_rgb(
                w,
                h,
                BitDepth::Ten,
                &raw16,
            )?)?;
            Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
        } else {
            let (rgb, a) = deinterleave_rgba(&scale16_to_10(img.to_rgba16().as_raw()), w, h);
            let frame = encode_color_16(&PlanarImage::from_interleaved_rgb(
                w,
                h,
                BitDepth::Ten,
                &rgb,
            )?)?;
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
        let raw16 = scale16_to_12(img.to_rgb16().as_raw());
        let frame = encode_color_16(&PlanarImage::from_interleaved_rgb(
            w,
            h,
            BitDepth::Twelve,
            &raw16,
        )?)?;
        Ok(Av2Encoder::wrap_avif(&frame, icc, exif)?)
    } else {
        let (rgb, a) = deinterleave_rgba(&scale16_to_12(img.to_rgba16().as_raw()), w, h);
        let frame = encode_color_16(&PlanarImage::from_interleaved_rgb(
            w,
            h,
            BitDepth::Twelve,
            &rgb,
        )?)?;
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

fn main() {
    let args = parse_args();

    if args.lossless {
        if let Some(c) = args.chroma {
            if c != Chroma::C444 {
                die("--lossless requires --chroma 444");
            }
        }
    }

    let img = image::open(&args.input)
        .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", args.input.display())));
    let color_type = img.color();
    let effective_depth = args.depth.unwrap_or(if is_16bit(color_type) {
        Depth::D10
    } else {
        Depth::D8
    });

    let exif_bytes = (!args.no_exif).then(|| read_exif(&args.input)).flatten();
    let icc_bytes = (!args.no_icc).then(|| read_icc(&args.input)).flatten();

    if args.verbose {
        match &exif_bytes {
            Some(b) => eprintln!("exif   : {} bytes", b.len()),
            None => eprintln!("exif   : none"),
        }
        match &icc_bytes {
            Some(b) => eprintln!("icc    : {} bytes", b.len()),
            None => eprintln!("icc    : none"),
        }
        eprintln!(
            "input  : {} ({}×{}, {:?}, {:?})",
            args.input.display(),
            img.width(),
            img.height(),
            effective_depth,
            color_type
        );
        eprintln!(
            "output : {}  encoder={:?}  quality={}{}  chroma={:?}  threads={}",
            args.output.display(),
            args.encoder,
            args.quality,
            if args.lossless { " (lossless)" } else { "" },
            args.chroma.unwrap_or(Chroma::C420),
            args.threads
        );
    }

    let t0 = Instant::now();

    let avif_bytes = match args.encoder {
        Encoder::Av1 => encode_av1(
            &img,
            &args,
            color_type,
            effective_depth,
            icc_bytes.as_deref(),
            exif_bytes.as_deref(),
        )
        .unwrap_or_else(|e| die(format!("encode failed: {e}"))),
        Encoder::Av2 => encode_av2(
            &img,
            &args,
            color_type,
            effective_depth,
            icc_bytes.as_deref(),
            exif_bytes.as_deref(),
        )
        .unwrap_or_else(|e| die(format!("encode failed: {e}"))),
    };

    let elapsed = t0.elapsed();

    std::fs::write(&args.output, &avif_bytes)
        .unwrap_or_else(|e| die(format!("cannot write '{}': {e}", args.output.display())));

    if args.verbose {
        let src_bytes =
            img.width() as u64 * img.height() as u64 * img.color().channel_count() as u64;
        eprintln!(
            "done   : {:.2?}  {} → {} bytes  ({:.1}% of raw)",
            elapsed,
            src_bytes,
            avif_bytes.len(),
            avif_bytes.len() as f64 / src_bytes as f64 * 100.0
        );
    } else {
        println!("{}", args.output.display());
    }
}
