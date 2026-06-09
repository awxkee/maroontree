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
//! Encodes any image readable by `image-rs` to AVIF, dispatching to either the
//! AV1 or AV2 path in `maroontree`.
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
//!   -q, --quality <1-100>       Encode quality (higher = better) [default: 80]
//!       --lossless              Pixel-perfect lossless (AV2 only)
//!   -c, --chroma <444|422|420>  Chroma subsampling              [default: 420]
//!   -d, --depth <8|10>          Output bit depth (auto-detect from source)
//!   -t, --threads <N>           Worker threads; 0 = all cores   [default: 1]
//!       --no-alpha              Discard alpha, composite onto black
//!   -v, --verbose               Print dimensions, timing, file size
//!   -h, --help                  Print this help
//! ```

use maroontree::{
    Av2Encoder, BitDepth, ChromaFormat, ColorEncoding, EncodeConfig, PlanarImage, encode_gray8,
    encode_gray10, encode_rgb8, encode_rgb10, encode_rgba8_with_alpha, encode_rgba10_with_alpha,
};
use std::path::{Path, PathBuf};
use std::time::Instant;

// ── CLI args ─────────────────────────────────────────────────────────────────

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
}

#[derive(Debug)]
struct Args {
    input: PathBuf,
    output: PathBuf,
    encoder: Encoder,
    /// 1-100 for AV1; 1-100 mapped to base_q_idx for AV2; 0 triggers lossless
    quality: u8,
    lossless: bool,
    chroma: Option<Chroma>,
    depth: Option<Depth>,
    threads: usize,
    no_alpha: bool,
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
  -q, --quality <1-100>       Quality (higher = better)       [default: 80]
      --lossless              Pixel-perfect lossless (AV2 only)
  -c, --chroma <444|422|420>  Chroma subsampling              [default: 420]
  -d, --depth <8|10>          Output bit depth (auto from source)
  -t, --threads <N>           Worker threads; 0 = all cores   [default: 1]
      --no-alpha              Discard alpha channel
  -v, --verbose               Print timing and file stats
  -h, --help                  Print this help"
    );
    std::process::exit(0);
}

fn die(msg: impl std::fmt::Display) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(1);
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1).peekable();
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut encoder = Encoder::Av1;
    let mut quality: u8 = 80;
    let mut lossless = false;
    let mut chroma: Option<Chroma> = None;
    let mut depth: Option<Depth> = None;
    let mut threads: usize = 1;
    let mut no_alpha = false;
    let mut verbose = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-v" | "--verbose" => verbose = true,
            "--lossless" => lossless = true,
            "--no-alpha" => no_alpha = true,
            "-e" | "--encoder" => match args.next().unwrap_or_default().as_str() {
                "av1" => encoder = Encoder::Av1,
                "av2" => encoder = Encoder::Av2,
                other => die(format!("unknown encoder '{other}'; use av1 or av2")),
            },
            "-q" | "--quality" => {
                let v = args.next().unwrap_or_default();
                quality = v
                    .parse::<u8>()
                    .unwrap_or_else(|_| die(format!("invalid quality '{v}'; expected 1-100")));
                if quality > 100 {
                    die("quality must be 1-100");
                }
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
                    other => die(format!("unsupported depth '{other}'; use 8 or 10")),
                })
            }
            "-t" | "--threads" => {
                let v = args.next().unwrap_or_default();
                threads = v
                    .parse()
                    .unwrap_or_else(|_| die(format!("invalid thread count '{v}'")));
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
        quality,
        lossless,
        chroma,
        depth,
        threads,
        no_alpha,
        verbose,
    }
}

// ── Quality mapping ───────────────────────────────────────────────────────────

/// Maps CLI quality 1–100 to AV2 `base_q_idx` 1–254.
/// quality 100 → q≈3 (near-lossless), quality 60 → q≈100, quality 1 → q=254.
fn av2_base_q(quality: u8) -> u8 {
    debug_assert!(quality >= 1 && quality <= 100);
    ((100 - quality as u32) * 254 / 99).clamp(1, 254) as u8
}

// ── Source type helpers ───────────────────────────────────────────────────────

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

/// Scale 16-bit (0..65535) samples to 10-bit (0..1023) by right-shifting 6.
fn scale16_to_10(src: &[u16]) -> Vec<u16> {
    src.iter().map(|&v| v >> 6).collect()
}

// ── AV1 encode path ───────────────────────────────────────────────────────────

fn encode_av1(
    img: &image::DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let chroma_fmt = match args.chroma.unwrap_or(Chroma::C420) {
        Chroma::C444 => ChromaFormat::Yuv444,
        Chroma::C422 => ChromaFormat::Yuv422,
        Chroma::C420 => ChromaFormat::Yuv420,
    };
    let w = img.width();
    let h = img.height();
    let cfg = EncodeConfig::new()
        .with_quality(args.quality)
        .with_chroma(chroma_fmt)
        .with_cicp(ColorEncoding::srgb_ycbcr())
        .with_threads(args.threads);

    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    Ok(match (effective_depth, gray, alpha) {
        // ── 8-bit ──────────────────────────────────────────────────
        (Depth::D8, true, _) => {
            let px = img.to_luma8();
            encode_gray8(px.as_raw(), w, h, &cfg)?
        }
        (Depth::D8, false, false) => {
            let px = img.to_rgb8();
            encode_rgb8(px.as_raw(), w, h, &cfg)?
        }
        (Depth::D8, false, true) => {
            let px = img.to_rgba8();
            encode_rgba8_with_alpha(px.as_raw(), w, h, &cfg)?
        }
        // ── 10-bit (source was 16-bit, scaled down) ──────────────
        (Depth::D10, true, _) => {
            let px = scale16_to_10(img.to_luma16().as_raw());
            encode_gray10(&px, w, h, &cfg)?
        }
        (Depth::D10, false, false) => {
            let raw = scale16_to_10(img.to_rgb16().as_raw());
            encode_rgb10(&raw, w, h, &cfg)?
        }
        (Depth::D10, false, true) => {
            let raw = scale16_to_10(img.to_rgba16().as_raw());
            encode_rgba10_with_alpha(&raw, w, h, &cfg)?
        }
    })
}

fn encode_av2(
    img: &image::DynamicImage,
    args: &Args,
    color_type: image::ColorType,
    effective_depth: Depth,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let gray = is_gray(color_type);
    let alpha = has_alpha_channel(color_type) && !args.no_alpha;

    // For lossless: force 444 and the identity (RGB) matrix. For lossy with
    // chroma subsampling: 420/422 need base_q_idx != 0 (encoder will error on
    // q=0 for those paths). We guard that above before reaching here.
    let base_q: u8 = if args.lossless {
        0
    } else {
        av2_base_q(args.quality)
    };
    let chroma_choice = args
        .chroma
        .unwrap_or(if gray { Chroma::C444 } else { Chroma::C420 });
    let color = if args.lossless {
        ColorEncoding::identity_rgb()
    } else {
        ColorEncoding::srgb_ycbcr()
    };

    // Encoder is reused for colour and (when lossless) alpha.
    let enc = Av2Encoder::new(base_q);
    // Alpha is always encoded lossless to avoid fringing artefacts.
    let alpha_enc = Av2Encoder::new(0);

    let bit_depth = match effective_depth {
        Depth::D8 => BitDepth::Eight,
        Depth::D10 => BitDepth::Ten,
    };

    // ── Gray / mono path ─────────────────────────────────────────────────────
    if gray {
        let luma: Vec<u8>;
        let luma16: Vec<u16>;
        return match effective_depth {
            Depth::D8 => {
                luma = img.to_luma8().into_raw();
                let pimg = PlanarImage::from_luma(w, h, BitDepth::Eight, &luma);
                let frame = enc.encode_image_400(&pimg, &color, args.threads)?;
                Ok(Av2Encoder::wrap_avif(&frame, None, None)?)
            }
            Depth::D10 => {
                luma16 = scale16_to_10(img.to_luma16().as_raw());
                let pimg = PlanarImage::from_luma(w, h, BitDepth::Ten, &luma16);
                let frame = enc.encode_image_400(&pimg, &color, args.threads)?;
                Ok(Av2Encoder::wrap_avif(&frame, None, None)?)
            }
        };
    }

    // ── RGB / RGBA path ───────────────────────────────────────────────────────
    //
    // We always build a full-resolution RGB PlanarImage (planes[0]=G,1=B,2=R
    // per from_interleaved_rgb) and hand it to the encoder.  The encoder
    // applies the color matrix internally.

    // Extract color and (optionally) alpha plane.
    let rgb8: Vec<u8>;
    let rgb16: Vec<u16>;
    let a8: Vec<u8>;
    let a16: Vec<u16>;

    let (pimg, alpha_pimg): (PlanarImage<u8>, Option<PlanarImage<u8>>) =
        match (effective_depth, alpha) {
            (Depth::D8, false) => {
                rgb8 = img.to_rgb8().into_raw();
                (
                    PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb8),
                    None,
                )
            }
            (Depth::D8, true) => {
                let rgba = img.to_rgba8().into_raw();
                let mut r = vec![0u8; w * h * 3];
                let mut a = vec![0u8; w * h];
                for (i, px) in rgba.chunks_exact(4).enumerate() {
                    r[i * 3] = px[0];
                    r[i * 3 + 1] = px[1];
                    r[i * 3 + 2] = px[2];
                    a[i] = px[3];
                }
                rgb8 = r;
                a8 = a;
                let ap = PlanarImage::from_luma(w, h, BitDepth::Eight, &a8);
                (
                    PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb8),
                    Some(ap),
                )
            }
            // 10-bit: note the mismatch — encode functions for AV2 accept generic
            // PlanarImage<T: Pixel>, so u16 works but requires a separate branch.
            // We fall through to the u8 8-bit path after printing a note.
            (Depth::D10, _) => {
                if args.verbose {
                    eprintln!("note: AV2 10-bit from 16-bit source — encoding as 8-bit for now");
                }
                rgb8 = img.to_rgb8().into_raw();
                if alpha {
                    let rgba = img.to_rgba8().into_raw();
                    let mut r = vec![0u8; w * h * 3];
                    let mut a = vec![0u8; w * h];
                    for (i, px) in rgba.chunks_exact(4).enumerate() {
                        r[i * 3] = px[0];
                        r[i * 3 + 1] = px[1];
                        r[i * 3 + 2] = px[2];
                        a[i] = px[3];
                    }
                    a8 = a;
                    let ap = PlanarImage::from_luma(w, h, BitDepth::Eight, &a8);
                    (
                        PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb8),
                        Some(ap),
                    )
                } else {
                    (
                        PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb8),
                        None,
                    )
                }
            }
        };

    let color_frame = match (chroma_choice, args.lossless) {
        (_, true) | (Chroma::C444, _) => {
            // encode_image_444 uses the encoder's baked-in base_q.
            enc.encode_image_444(&pimg, &color, args.threads)?
        }
        (Chroma::C420, false) => enc.encode_image_420(&pimg, &color, args.threads)?,
        (Chroma::C422, false) => enc.encode_image_422(&pimg, &color, args.threads)?,
    };

    // ── Encode alpha (lossless mono) and mux ─────────────────────────────────
    match alpha_pimg {
        Some(ref ap) => {
            let alpha_frame = alpha_enc.encode_yuv400(ap, &color, args.threads)?;
            Ok(Av2Encoder::wrap_avif_alpha(
                &color_frame,
                &alpha_frame,
                None,
                None,
            )?)
        }
        None => Ok(Av2Encoder::wrap_avif(&color_frame, None, None)?),
    }
}

fn main() {
    let args = parse_args();

    // ── Load source image ────────────────────────────────────────────────────
    let img = image::open(&args.input)
        .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", args.input.display())));
    let color_type = img.color();

    // Resolve effective bit depth: prefer --depth, fall back to source depth.
    let effective_depth = args.depth.unwrap_or(if is_16bit(color_type) {
        Depth::D10
    } else {
        Depth::D8
    });

    // Guard: lossless only makes sense with AV2.
    if args.lossless && args.encoder == Encoder::Av1 {
        die("--lossless requires --encoder av2 (the AV1 path has no lossless entry point here)");
    }
    // Guard: AV2 lossless with 420/422 would be nonsensical — encoder returns InvalidQuality.
    if args.lossless {
        if let Some(c) = args.chroma {
            if c != Chroma::C444 {
                die("--lossless requires --chroma 444 (chroma subsampling loses information)");
            }
        }
    }

    if args.verbose {
        eprintln!(
            "input : {} ({}×{}, {:?}, color_type={:?})",
            args.input.display(),
            img.width(),
            img.height(),
            effective_depth,
            color_type
        );
        eprintln!(
            "output: {}  encoder={:?}  quality={}{}  chroma={:?}  threads={}",
            args.output.display(),
            args.encoder,
            args.quality,
            if args.lossless { " (lossless)" } else { "" },
            args.chroma.unwrap_or(Chroma::C420),
            args.threads,
        );
    }

    let t0 = Instant::now();

    let avif_bytes = match args.encoder {
        Encoder::Av1 => encode_av1(&img, &args, color_type, effective_depth),
        Encoder::Av2 => encode_av2(&img, &args, color_type, effective_depth),
    }
    .unwrap_or_else(|e| die(format!("encode failed: {e}")));

    let elapsed = t0.elapsed();

    std::fs::write(&args.output, &avif_bytes)
        .unwrap_or_else(|e| die(format!("cannot write '{}': {e}", args.output.display())));

    if args.verbose {
        let src_bytes =
            img.width() as u64 * img.height() as u64 * img.color().channel_count() as u64;
        let ratio = avif_bytes.len() as f64 / src_bytes as f64;
        eprintln!(
            "done   : {:.2?}  {src_bytes} → {} bytes  ({:.1}% of raw)",
            elapsed,
            avif_bytes.len(),
            ratio * 100.0,
        );
    } else {
        println!("{}", args.output.display());
    }
}
