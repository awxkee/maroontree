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

use crate::{Args, is_gray};
use image::DynamicImage;
use img_parts::{ImageEXIF, ImageICC, jpeg::Jpeg, png::Png};
use std::io::Cursor;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RasterFormat {
    Png,
    Jpeg,
}

/// If `path`'s extension names a plain raster format we can write directly,
/// return it. These outputs skip the codec backends entirely.
pub(crate) fn raster_output_format(path: &Path) -> Option<RasterFormat> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Some(RasterFormat::Png),
        Some("jpg") | Some("jpeg") => Some(RasterFormat::Jpeg),
        _ => None,
    }
}

/// Encode `img` to PNG or JPEG bytes, re-attaching EXIF/ICC when present.
///
/// - **PNG** keeps the image's native depth (8 or 16-bit) and alpha channel.
/// - **JPEG** is 8-bit with no alpha, so the image is flattened to RGB8 (or
///   Luma8 when grayscale) and `args.quality` controls compression.
pub(crate) fn encode_raster(
    img: &DynamicImage,
    fmt: RasterFormat,
    args: &Args,
    icc: Option<&[u8]>,
    exif: Option<&[u8]>,
) -> Result<Vec<u8>, anyhow::Error> {
    match fmt {
        RasterFormat::Png => {
            let mut pixels = Vec::new();
            img.write_to(&mut Cursor::new(&mut pixels), image::ImageFormat::Png)?;
            if icc.is_none() && exif.is_none() {
                return Ok(pixels);
            }
            let mut png = Png::from_bytes(pixels.into())
                .map_err(|e| anyhow::anyhow!("re-parse PNG for metadata: {e}"))?;
            if let Some(icc) = icc {
                png.set_icc_profile(Some(icc.to_vec().into()));
            }
            if let Some(exif) = exif {
                png.set_exif(Some(exif.to_vec().into()));
            }
            let mut out = Vec::new();
            png.encoder()
                .write_to(&mut out)
                .map_err(|e| anyhow::anyhow!("write PNG: {e}"))?;
            Ok(out)
        }
        RasterFormat::Jpeg => {
            let mut pixels = Vec::new();
            {
                let q = args.quality.clamp(1, 100);
                let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut pixels, q);
                if is_gray(img.color()) {
                    enc.encode_image(&img.to_luma8())?;
                } else {
                    enc.encode_image(&img.to_rgb8())?;
                }
            }
            if icc.is_none() && exif.is_none() {
                return Ok(pixels);
            }
            let mut jpeg = Jpeg::from_bytes(pixels.into())
                .map_err(|e| anyhow::anyhow!("re-parse JPEG for metadata: {e}"))?;
            if let Some(icc) = icc {
                jpeg.set_icc_profile(Some(icc.to_vec().into()));
            }
            if let Some(exif) = exif {
                jpeg.set_exif(Some(exif.to_vec().into()));
            }
            let mut out = Vec::new();
            jpeg.encoder()
                .write_to(&mut out)
                .map_err(|e| anyhow::anyhow!("write JPEG: {e}"))?;
            Ok(out)
        }
    }
}
