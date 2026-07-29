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
//!   -e, --encoder <av1|av2|hevc|jxl,vvc>     Encoder backend                 [default: av1]
//!   -q, --quality <1-100>                    Encode quality (higher = better) [default: 80(AV1)/60(AV2)]
//!       --lossless                           Pixel-perfect lossless
//!   -c, --chroma <444|422|420>               Chroma subsampling              [default: 420]
//!   -d, --depth <8|10|12>                    Output bit depth (auto-detect from source)
//!   -t, --threads <N>                        Worker threads; 0 = all cores   [default: all]
//!       --no-alpha                           Discard alpha channel
//!       --no-screen-content                  Skip palette (screen-content) search
//!       --no-intrabc                         Veto lossy IntraBC (keeps loop filters)
//!       --no-exif                            Strip EXIF from output
//!       --no-icc                             Strip ICC profile from output
//!   -s, --speed                              Encoding effort (default = slow)
//!   -v, --verbose                            Print dimensions, timing, file size
//!   -h, --help                               Print this help
//! ```

mod av1_lossless;
mod av1_lossy;
mod av2_decode;
mod av2_lossless;
mod av2_lossy;
mod box_walker;
#[cfg(feature = "heic")]
mod heic;
mod jxl;
#[cfg(feature = "heic")]
mod orientation;
mod raster;
#[cfg(feature = "vvc")]
mod vvc;

use crate::av1_lossy::encode_av1;
use crate::av2_lossy::encode_av2;
use crate::box_walker::{ImageContainer, detect_image_container};
use crate::jxl::{decode_jxl, encode_jxl};
use image::{DynamicImage, Luma, Rgb, Rgba};
use img_parts::{ImageEXIF, ImageICC, jpeg::Jpeg, png::Png, webp::WebP};
use std::path::{Path, PathBuf};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoder {
    Av1,
    Av2,
    Hevc,
    JpegXl,
    Vvc,
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
    apply_icc: bool,
    verbose: bool,
    speed: EncodingEffort,
    qmatrix: Option<Qmatrix>,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
}

fn usage() -> ! {
    eprintln!(
        "\
Usage: avif [OPTIONS] <INPUT> [OUTPUT]

Arguments:
  <INPUT>   Source image (PNG, JPEG, WebP, TIFF, …)
  [OUTPUT]  Output .avif  [default: input stem + \".avif\"]

Options:
  -e, --encoder <av1|av2|hevc|jxl,vvc>  Encoder backend                 [default: av1]
  -q, --quality <1-100>                 Quality (higher = better)       [default: 80(AV1)/60(AV2)]
      --lossless                        Pixel-perfect lossless (AV2 only)
  -c, --chroma <444|422|420>            Chroma subsampling              [default: 420]
  -d, --depth <8|10|12>                 Output bit depth (auto from source)
  -t, --threads <N>                     Worker threads; 0 = all cores   [default: all]
      --no-alpha                        Discard alpha channel
      --no-screen-content               Skip palette (screen-content) search
      --no-intrabc                      Veto lossy IntraBC (keeps loop filters)
      --no-exif                         Strip EXIF metadata from output
      --no-icc                          Strip ICC color profile from output
      --apply-icc                       Apply ICC profile to pixels (convert to sRGB), then strip it
      --qm <auto|0-15>                  Enable AV1 quantization matrices
      --no-cdf-update                   Freeze AV1 entropy CDFs
  -s, --speed                           Encoding effort (default = slow)
  -v, --verbose                         Print timing and file stats
  -h, --help                            Print this help"
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

#[derive(Debug, Copy, Clone, Ord, PartialOrd, Eq, PartialEq)]
enum EncodingEffort {
    Slow,
    Medium,
    Fast,
}

#[derive(Debug, Copy, Clone)]
enum Qmatrix {
    Auto,
    Level(u8),
}

impl EncodingEffort {
    pub(crate) fn to_maroontreee(self) -> maroontree::Speed {
        match self {
            EncodingEffort::Slow => maroontree::Speed::Slow,
            EncodingEffort::Medium => maroontree::Speed::Medium,
            EncodingEffort::Fast => maroontree::Speed::Fast,
        }
    }
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
    let mut screen_content = true;
    let mut intrabc = true;
    let mut no_exif = false;
    let mut no_icc = false;
    let mut apply_icc = false;
    let mut verbose = false;
    let mut speed = EncodingEffort::Slow;
    let mut qmatrix = None;
    let mut updating_cdf = true;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => usage(),
            "-v" | "--verbose" => verbose = true,
            "--lossless" => lossless = true,
            "--no-alpha" => no_alpha = true,
            "--no-screen-content" => screen_content = false,
            "--no-intrabc" => intrabc = false,
            "--no-exif" => no_exif = true,
            "--no-icc" => no_icc = true,
            "--apply-icc" => apply_icc = true,
            "--no-cdf-update" => updating_cdf = false,
            "--qm" => {
                let value = args.next().unwrap_or_default();
                qmatrix = Some(if value == "auto" {
                    Qmatrix::Auto
                } else {
                    let level = value.parse::<u8>().unwrap_or_else(|_| {
                        die(format!(
                            "invalid qmatrix level '{value}'; expected auto or 0-15"
                        ))
                    });
                    if level > 15 {
                        die("qmatrix level must be 0-15");
                    }
                    Qmatrix::Level(level)
                });
            }
            "-e" | "--encoder" => match args.next().unwrap_or_default().as_str() {
                "av1" => encoder = Encoder::Av1,
                "av2" => encoder = Encoder::Av2,
                "hevc" | "heic" | "heif" => encoder = Encoder::Hevc,
                "jxl" | "jpegxl" | "jpeg-xl" => encoder = Encoder::JpegXl,
                "vvc" => encoder = Encoder::Vvc,
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
            "-s" | "--speed" => {
                speed = match args.next().unwrap_or_default().as_str() {
                    "slow" => EncodingEffort::Slow,
                    "medium" => EncodingEffort::Medium,
                    "fast" => EncodingEffort::Fast,
                    other => die(format!(
                        "unsupported speed '{other}'; use slow, medium, or fast"
                    )),
                }
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
                Encoder::Hevc => 80,
                Encoder::JpegXl => 70,
                Encoder::Vvc => 60,
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
        screen_content,
        intrabc,
        no_exif,
        no_icc,
        apply_icc,
        verbose,
        speed,
        qmatrix,
        updating_cdf,
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

fn apply_icc_to_image(
    img: &DynamicImage,
    effective_depth: Depth,
    icc: &[u8],
) -> Result<DynamicImage, anyhow::Error> {
    use moxcms::{ColorProfile, Layout, TransformOptions};

    let src = ColorProfile::new_from_slice(icc)
        .map_err(|e| anyhow::anyhow!("invalid ICC profile: {e:?}"))?;
    let dst = ColorProfile::new_srgb();
    let opts = TransformOptions::default();

    if has_alpha_channel(img.color()) {
        if effective_depth == Depth::D8 {
            if matches!(img, DynamicImage::ImageLumaA8(_)) {
                // gray image
                let rgba = img.to_luma_alpha8();
                let (w, h) = rgba.dimensions();
                let buf = rgba.into_raw();
                let mut dst_data = vec![0u8; buf.len()];
                src.create_transform_8bit(Layout::GrayAlpha, &dst, Layout::GrayAlpha, opts)
                    .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                    .transform(&buf, &mut dst_data)
                    .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
                return Ok(DynamicImage::ImageLumaA8(
                    image::GrayAlphaImage::from_raw(w, h, dst_data)
                        .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
                ));
            }
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            let buf = rgba.into_raw();
            let mut dst_data = vec![0u8; buf.len()];
            src.create_transform_8bit(Layout::Rgba, &dst, Layout::Rgba, opts)
                .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                .transform(&buf, &mut dst_data)
                .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
            Ok(DynamicImage::ImageRgba8(
                image::RgbaImage::from_raw(w, h, dst_data)
                    .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
            ))
        } else {
            let rgba = img.to_rgba16();
            let (w, h) = rgba.dimensions();
            let buf = rgba.into_raw();
            let mut dst_data = vec![0u16; buf.len()];
            src.create_transform_16bit(Layout::Rgba, &dst, Layout::Rgba, opts)
                .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                .transform(&buf, &mut dst_data)
                .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
            Ok(DynamicImage::ImageRgba16(
                image::ImageBuffer::<Rgba<u16>, Vec<u16>>::from_raw(w, h, dst_data)
                    .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
            ))
        }
    } else {
        if effective_depth == Depth::D8 {
            if matches!(img, DynamicImage::ImageLuma8(_)) {
                // gray image
                let rgba = img.to_luma8();
                let (w, h) = rgba.dimensions();
                let buf = rgba.into_raw();
                let mut dst_data = vec![0u8; buf.len()];
                src.create_transform_8bit(Layout::Gray, &dst, Layout::Gray, opts)
                    .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                    .transform(&buf, &mut dst_data)
                    .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
                return Ok(DynamicImage::ImageLuma8(
                    image::GrayImage::from_raw(w, h, dst_data)
                        .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
                ));
            }
            let rgba = img.to_rgb8();
            let (w, h) = rgba.dimensions();
            let buf = rgba.into_raw();
            let mut dst_data = vec![0u8; buf.len()];
            src.create_transform_8bit(Layout::Rgb, &dst, Layout::Rgb, opts)
                .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                .transform(&buf, &mut dst_data)
                .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
            Ok(DynamicImage::ImageRgb8(
                image::RgbImage::from_raw(w, h, dst_data)
                    .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
            ))
        } else {
            if matches!(img, DynamicImage::ImageLuma16(_)) {
                // gray image
                let rgba = img.to_luma16();
                let (w, h) = rgba.dimensions();
                let buf = rgba.into_raw();
                let mut dst_data = vec![0u16; buf.len()];
                src.create_transform_16bit(Layout::Gray, &dst, Layout::Gray, opts)
                    .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                    .transform(&buf, &mut dst_data)
                    .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
                return Ok(DynamicImage::ImageLuma16(
                    image::ImageBuffer::<Luma<u16>, Vec<u16>>::from_raw(w, h, dst_data)
                        .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
                ));
            }
            let rgba = img.to_rgb16();
            let (w, h) = rgba.dimensions();
            let buf = rgba.into_raw();
            let mut dst_data = vec![0u16; buf.len()];
            src.create_transform_16bit(Layout::Rgb, &dst, Layout::Rgb, opts)
                .map_err(|e| anyhow::anyhow!("ICC transform create: {e:?}"))?
                .transform(&buf, &mut dst_data)
                .map_err(|e| anyhow::anyhow!("ICC transform apply: {e:?}"))?;
            Ok(DynamicImage::ImageRgb16(
                image::ImageBuffer::<Rgb<u16>, Vec<u16>>::from_raw(w, h, dst_data)
                    .ok_or_else(|| anyhow::anyhow!("buffer size mismatch after ICC apply"))?,
            ))
        }
    }
}
fn is_heif_format(fmt: &str) -> bool {
    matches!(fmt.to_lowercase().as_str(), "heic" | "heif")
}

fn is_avif_format(fmt: &str) -> bool {
    matches!(fmt.to_lowercase().as_str(), "avif" | "avis")
}

fn is_jxl_format(fmt: &str) -> bool {
    matches!(fmt.to_lowercase().as_str(), "jxl" | "jpegxl")
}

fn load_image(path: &PathBuf) -> (DynamicImage, Option<Vec<u8>>) {
    let mut have_icc: Option<Vec<u8>> = None;

    let fmt = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    let mut _image_container = ImageContainer::Unknown;
    if let Some(ext) = fmt.as_ref()
        && (is_heif_format(ext) || is_avif_format(ext))
    {
        _image_container = detect_image_container(path);
    }

    println!(
        "Image container is {:?}, ext {:?}, is heif {:?}",
        _image_container,
        fmt,
        if let Some(ext) = fmt.as_ref() {
            is_heif_format(ext)
        } else {
            false
        }
    );

    let img = match fmt {
        #[cfg(feature = "heic")]
        Some(_) if _image_container == ImageContainer::Heic => {
            use crate::heic::decode_heic_file_url;
            decode_heic_file_url(path)
                .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display())))
        }
        #[cfg(feature = "vvc")]
        Some(_) if _image_container == ImageContainer::Vvc => {
            use crate::vvc::decode_heic_vvc_file_url;
            decode_heic_vvc_file_url(path)
                .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display())))
        }

        Some(_) if _image_container == ImageContainer::Av2 => {
            use crate::av2_decode::decode_av2_file_url;
            decode_av2_file_url(path)
                .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display())))
        }

        Some(fmt) if is_jxl_format(&fmt) => {
            let (decoded, icc) = decode_jxl(path)
                .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display())));
            have_icc = icc;
            decoded
        }

        _ => image::open(path)
            .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display()))),
    };

    (img, have_icc)
}

fn main() {
    let args = parse_args();

    if args.lossless
        && let Some(c) = args.chroma
        && c != Chroma::C444
    {
        die("--lossless requires --chroma 444");
    }

    let (img, have_icc) = load_image(&args.input);
    let color_type = img.color();
    let effective_depth = args.depth.unwrap_or(if is_16bit(color_type) {
        Depth::D10
    } else {
        Depth::D8
    });

    let exif_bytes = (!args.no_exif).then(|| read_exif(&args.input)).flatten();

    let mut raw_icc = if args.apply_icc || !args.no_icc {
        read_icc(&args.input)
    } else {
        None
    };
    if let Some(have_icc) = have_icc {
        raw_icc = Some(have_icc.to_vec());
    }

    // Apply ICC profile to pixel data via moxcms, then discard the profile so
    // it is not embedded in the output. The encoded image will be plain sRGB.
    let (img, icc_bytes) = if args.apply_icc {
        match raw_icc {
            Some(ref icc) => {
                let converted = apply_icc_to_image(&img, effective_depth, icc)
                    .unwrap_or_else(|e| die(format!("ICC apply failed: {e}")));
                (converted, None)
            }
            None => {
                if args.verbose {
                    eprintln!("apply-icc: no ICC profile found in source, skipping");
                }
                (img, None)
            }
        }
    } else {
        // --no-icc just strips without converting
        (img, if args.no_icc { None } else { raw_icc })
    };

    if args.verbose {
        match &exif_bytes {
            Some(b) => eprintln!("exif   : {} bytes", b.len()),
            None => eprintln!("exif   : none"),
        }
        match &icc_bytes {
            Some(b) => eprintln!("icc    : {} bytes (embedded)", b.len()),
            None if args.apply_icc => eprintln!("icc    : applied, not embedded"),
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

    let avif_bytes = if let Some(rf) = raster::raster_output_format(&args.output) {
        raster::encode_raster(&img, rf, &args, icc_bytes.as_deref(), exif_bytes.as_deref())
            .unwrap_or_else(|e| die(format!("encode failed: {e}")))
    } else {
        match args.encoder {
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
            Encoder::Hevc => {
                #[cfg(not(feature = "heic"))]
                {
                    die("to use heic container compile with heic support")
                }
                #[cfg(feature = "heic")]
                {
                    use crate::heic::encode_hevc;
                    encode_hevc(
                        &img,
                        &args,
                        color_type,
                        effective_depth,
                        icc_bytes.as_deref(),
                        exif_bytes.as_deref(),
                    )
                    .unwrap_or_else(|e| die(format!("encode failed: {e}")))
                }
            }
            Encoder::JpegXl => encode_jxl(
                &img,
                &args,
                color_type,
                effective_depth,
                icc_bytes.as_deref(),
                exif_bytes.as_deref(),
            )
            .unwrap_or_else(|e| die(format!("encode failed: {e}"))),
            Encoder::Vvc => {
                #[cfg(not(feature = "vvc"))]
                {
                    die("to use heif with vvc container compile with 'vvc' support")
                }
                #[cfg(feature = "vvc")]
                {
                    use crate::vvc::encode_vvc;
                    encode_vvc(
                        &img,
                        &args,
                        color_type,
                        effective_depth,
                        icc_bytes.as_deref(),
                        exif_bytes.as_deref(),
                    )
                    .unwrap_or_else(|e| die(format!("encode failed: {e}")))
                }
            }
        }
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
