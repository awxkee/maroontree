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

//! Image encoder CLI. Run `maroontree-cli --help` for options and examples.

mod args;
mod av1_lossless;
mod av1_lossy;
mod av2_decode;
mod av2_lossless;
mod av2_lossy;
mod box_walker;
#[cfg(feature = "heic")]
mod gainmap;
#[cfg(feature = "heic")]
mod heic;
mod jpeg;
mod jxl;
mod orientation;
mod overrides;
mod raster;
mod version;
#[cfg(feature = "vvc")]
mod vvc;

use crate::av1_lossy::encode_av1;
use crate::av2_lossy::encode_av2;
use crate::box_walker::{ImageContainer, detect_image_container};
use crate::jxl::{JxlMetadata, decode_jxl, encode_jxl};
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
    Jpeg,
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
    D16,
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
    no_gain_map: bool,
    apply_icc: bool,
    verbose: bool,
    speed: EncodingEffort,
    qmatrix: Option<Qmatrix>,
    updating_cdf: bool,
    screen_content: bool,
    intrabc: bool,
    cdef: bool,
    wiener: bool,
    overrides: Vec<overrides::EncoderOverride>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PngCicp {
    color_primaries: u8,
    transfer_function: u8,
    matrix_coefficients: u8,
    full_range: bool,
}

impl PngCicp {
    /// Adapt PNG RGB CICP to the matrix and range actually produced by a codec.
    /// Primaries and transfer remain properties of the source color space.
    fn to_maroontree(
        self,
        matrix: maroontree::MatrixCoefficients,
        full_range: bool,
    ) -> maroontree::Cicp {
        use maroontree::{ChromaSamplePosition, Primaries, TransferFunction};

        let primaries = match self.color_primaries {
            1 => Primaries::Bt709,
            4 => Primaries::Bt470M,
            5 => Primaries::Bt470Bg,
            6 => Primaries::Bt601,
            7 => Primaries::Smpte240,
            8 => Primaries::GenericFilm,
            9 => Primaries::Bt2020,
            10 => Primaries::Xyz,
            11 => Primaries::Smpte431,
            12 => Primaries::Smpte432,
            22 => Primaries::Ebu3213,
            _ => Primaries::Unspecified,
        };
        let transfer = match self.transfer_function {
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
            _ => TransferFunction::Unspecified,
        };

        maroontree::Cicp {
            primaries,
            transfer,
            matrix,
            full_range,
            chroma_sample_position: ChromaSamplePosition::Unknown,
        }
    }
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
    ExtraSlow,
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
            EncodingEffort::ExtraSlow => maroontree::Speed::ExtraSlow,
            EncodingEffort::Slow => maroontree::Speed::Slow,
            EncodingEffort::Medium => maroontree::Speed::Medium,
            EncodingEffort::Fast => maroontree::Speed::Fast,
        }
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

fn expand_to_16bit(buffer: &mut [u16], is12_bit: bool) {
    if is12_bit {
        for px in buffer.iter_mut() {
            *px = (*px << 4) | (*px >> 8);
        }
    } else {
        for px in buffer.iter_mut() {
            *px = (*px << 6) | (*px >> 4);
        }
    }
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

fn read_png_cicp(path: &Path) -> Option<PngCicp> {
    let is_png = path
        .extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("png"));
    if !is_png {
        return None;
    }

    let raw = std::fs::read(path).ok()?;
    let png = Png::from_bytes(raw.into()).ok()?;
    let contents = png.chunk_by_type(*b"cICP")?.contents();
    let &[
        color_primaries,
        transfer_function,
        matrix_coefficients,
        full_range,
    ] = contents.as_ref()
    else {
        return None;
    };

    Some(PngCicp {
        color_primaries,
        transfer_function,
        matrix_coefficients,
        full_range: full_range == 1,
    })
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

struct LoadedImage {
    image: DynamicImage,
    icc: Option<Vec<u8>>,
    #[cfg(feature = "heic")]
    cicp: Option<PngCicp>,
    #[cfg(feature = "heic")]
    gain_map: Option<gainmap::ParsedGainMap>,
}

fn load_image(path: &PathBuf, _load_gain_map: bool) -> LoadedImage {
    let mut have_icc: Option<Vec<u8>> = None;
    #[cfg(feature = "heic")]
    let mut gain_map = None;
    #[cfg(feature = "heic")]
    let mut have_cicp = None;

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

    let img = match fmt {
        #[cfg(feature = "heic")]
        Some(_) if _image_container == ImageContainer::Heic => {
            use crate::heic::decode_heic_file_url;
            let decoded = decode_heic_file_url(path, _load_gain_map)
                .unwrap_or_else(|e| die(format!("cannot open '{}': {e}", path.display())));
            have_icc = decoded.icc;
            have_cicp = decoded.cicp;
            gain_map = decoded.gain_map;
            decoded.image
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

    LoadedImage {
        image: img,
        icc: have_icc,
        #[cfg(feature = "heic")]
        cicp: have_cicp,
        #[cfg(feature = "heic")]
        gain_map,
    }
}

/// The source's HDR gain map as an AVIF gain map item, if it has one.
fn avif_gain_map(
    #[cfg(feature = "heic")] gain_map: Option<&gainmap::ParsedGainMap>,
    verbose: bool,
) -> Option<maroontree::GainMap> {
    #[cfg(feature = "heic")]
    if let Some(gain_map) = gain_map {
        match gainmap::avif_gain_map(gain_map) {
            Ok(avif) => {
                if verbose {
                    eprintln!(
                        "gainmap: encoding {}×{} ISO 21496-1 map ({:.3} stops) as AVIF tmap",
                        gain_map.image.width(),
                        gain_map.image.height(),
                        gain_map.metadata.alternate_hdr_headroom(),
                    );
                }
                return Some(avif);
            }
            Err(error) => eprintln!("gainmap: skipping gain map: {error:#}"),
        }
    }
    let _ = verbose;
    None
}

fn main() {
    let args = args::parse_args();

    // Raster output extensions take precedence over the selected encoder.
    let raster_format = raster::raster_output_format(&args.output);
    let LoadedImage {
        image: img,
        icc: have_icc,
        #[cfg(feature = "heic")]
            cicp: have_cicp,
        #[cfg(feature = "heic")]
        gain_map,
    } = load_image(
        &args.input,
        matches!(args.encoder, Encoder::JpegXl | Encoder::Av1 | Encoder::Hevc)
            && !args.no_gain_map
            && raster_format.is_none(),
    );
    let color_type = img.color();
    let effective_depth = args.depth.unwrap_or(if args.encoder == Encoder::Jpeg {
        Depth::D8
    } else if is_16bit(color_type) {
        if args.encoder == Encoder::JpegXl {
            Depth::D16
        } else {
            Depth::D10
        }
    } else {
        Depth::D8
    });

    let exif_bytes = (!args.no_exif).then(|| read_exif(&args.input)).flatten();

    let mut raw_icc = if args.apply_icc || !args.no_icc {
        read_icc(&args.input)
    } else {
        None
    };
    let png_cicp = (!args.apply_icc)
        .then(|| read_png_cicp(&args.input))
        .flatten();
    #[cfg(feature = "heic")]
    let png_cicp = png_cicp.or(have_cicp.filter(|_| !args.apply_icc));
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
        eprintln!("encoder: {}", version::encoder(args.encoder));
        for option in &args.overrides {
            eprintln!("override: {option}");
        }
        match &exif_bytes {
            Some(b) => eprintln!("exif   : {} bytes", b.len()),
            None => eprintln!("exif   : none"),
        }
        match &icc_bytes {
            Some(b) => eprintln!("icc    : {} bytes (source)", b.len()),
            None if args.apply_icc => eprintln!("icc    : applied, not embedded"),
            None => eprintln!("icc    : none"),
        }
        if let Some(cicp) = png_cicp {
            eprintln!(
                "cicp   : primaries={} transfer={} matrix={} full-range={}",
                cicp.color_primaries,
                cicp.transfer_function,
                cicp.matrix_coefficients,
                cicp.full_range
            );
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

    let avif_bytes = if let Some(rf) = raster_format {
        raster::encode_raster(
            &img,
            rf,
            &args,
            icc_bytes.as_deref(),
            exif_bytes.as_deref(),
            png_cicp,
        )
        .unwrap_or_else(|e| die(format!("encode failed: {e}")))
    } else {
        match args.encoder {
            Encoder::Jpeg => {
                jpeg::encode_jpeg(&img, &args, icc_bytes.as_deref(), exif_bytes.as_deref())
                    .unwrap_or_else(|e| die(format!("encode failed: {e}")))
            }
            Encoder::Av1 => encode_av1(
                &img,
                &args,
                color_type,
                effective_depth,
                icc_bytes.as_deref(),
                exif_bytes.as_deref(),
                png_cicp,
                avif_gain_map(
                    #[cfg(feature = "heic")]
                    gain_map.as_ref(),
                    args.verbose,
                ),
            )
            .unwrap_or_else(|e| die(format!("encode failed: {e}"))),
            Encoder::Av2 => encode_av2(
                &img,
                &args,
                color_type,
                effective_depth,
                icc_bytes.as_deref(),
                exif_bytes.as_deref(),
                png_cicp,
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
                        png_cicp,
                        gain_map.as_ref(),
                    )
                    .unwrap_or_else(|e| die(format!("encode failed: {e}")))
                }
            }
            Encoder::JpegXl => encode_jxl(
                &img,
                &args,
                color_type,
                effective_depth,
                JxlMetadata {
                    icc: icc_bytes.as_deref(),
                    exif: exif_bytes.as_deref(),
                    cicp: png_cicp,
                    #[cfg(feature = "heic")]
                    gain_map: gain_map.as_ref(),
                },
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
                        png_cicp,
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

#[cfg(test)]
mod cicp_tests {
    use super::*;

    #[test]
    fn maps_png_rec2100_pq_to_avif_cicp() {
        let cicp = PngCicp {
            color_primaries: 9,
            transfer_function: 16,
            matrix_coefficients: 0,
            full_range: true,
        }
        .to_maroontree(maroontree::MatrixCoefficients::Smpte170m, true);

        assert_eq!(cicp.primaries, maroontree::Primaries::Bt2020);
        assert_eq!(cicp.transfer, maroontree::TransferFunction::Smpte2084);
        assert_eq!(cicp.matrix, maroontree::MatrixCoefficients::Smpte170m);
        assert!(cicp.full_range);
    }

    #[test]
    fn unknown_png_cicp_codes_become_unspecified() {
        let cicp = PngCicp {
            color_primaries: u8::MAX,
            transfer_function: u8::MAX,
            matrix_coefficients: 0,
            full_range: true,
        }
        .to_maroontree(maroontree::MatrixCoefficients::Identity, true);

        assert_eq!(cicp.primaries, maroontree::Primaries::Unspecified);
        assert_eq!(cicp.transfer, maroontree::TransferFunction::Unspecified);
    }
}
