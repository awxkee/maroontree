/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
use crate::avif::{checked_buffer_size, validate_dims};
use crate::color::ColorEncoding;
use crate::err::EncodeError;
use crate::obu::temporal_delimiter;
use crate::pixel::Pixel;
use crate::{BitDepth, ChromaFormat};

/// A planar image. `planes[0..3]` are full-resolution (4:4:4).
/// For identity RGB we store G, B, R in planes 0, 1, 2 (AV1 GBR ordering).
pub struct PlanarImage<T: Pixel> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: BitDepth,
    pub planes: [Vec<T>; 3],
}

fn validate_buf<T>(buf: &[T], w: usize, h: usize, ch: usize) -> Result<(), EncodeError> {
    let needed = checked_buffer_size::<T>(w, h, ch)?;
    if buf.len() != needed {
        return Err(EncodeError::InvalidInput);
    }
    Ok(())
}

impl<T: Pixel> PlanarImage<T> {
    pub(crate) fn validate_400(&self) -> Result<(), EncodeError> {
        validate_dims(self.width as u32, self.height as u32)?;
        validate_buf(&self.planes[0], self.width, self.height, 1)?;
        Ok(())
    }

    pub(crate) fn validate_444(&self) -> Result<(), EncodeError> {
        validate_dims(self.width as u32, self.height as u32)?;
        validate_buf(&self.planes[0], self.width, self.height, 1)?;
        validate_buf(&self.planes[1], self.width, self.height, 1)?;
        validate_buf(&self.planes[2], self.width, self.height, 1)?;
        Ok(())
    }

    pub(crate) fn validate_422(&self) -> Result<(), EncodeError> {
        validate_dims(self.width as u32, self.height as u32)?;
        validate_buf(&self.planes[0], self.width, self.height, 1)?;
        validate_buf(&self.planes[1], self.width.div_ceil(2), self.height, 1)?;
        validate_buf(&self.planes[2], self.width.div_ceil(2), self.height, 1)?;
        Ok(())
    }

    pub(crate) fn validate_420(&self) -> Result<(), EncodeError> {
        validate_dims(self.width as u32, self.height as u32)?;
        validate_buf(&self.planes[0], self.width, self.height, 1)?;
        validate_buf(
            &self.planes[1],
            self.width.div_ceil(2),
            self.height.div_ceil(2),
            1,
        )?;
        validate_buf(
            &self.planes[2],
            self.width.div_ceil(2),
            self.height.div_ceil(2),
            1,
        )?;
        Ok(())
    }

    pub(crate) fn validate_with(&self, chroma_format: ChromaFormat) -> Result<(), EncodeError> {
        match chroma_format {
            ChromaFormat::Yuv420 => {
                self.validate_420()?;
            }
            ChromaFormat::Yuv422 => {
                self.validate_422()?;
            }
            ChromaFormat::Yuv444 => {
                self.validate_444()?;
            }
            ChromaFormat::Monochrome => {
                self.validate_400()?;
            }
        }
        Ok(())
    }
}

// Q0.13 coefficients  (value = round(f * 8192))
const Q: i32 = 13;
const HALF: i32 = 1 << (Q - 1); // 0.5 rounding bias

const Y_R: i32 = 2449; // round( 0.299    * 8192)
const Y_G: i32 = 4809; // round( 0.587    * 8192)
const Y_B: i32 = 934; // round( 0.114    * 8192)

const CB_R: i32 = -1382; // round(-0.168736 * 8192)
const CB_G: i32 = -2714; // round(-0.331264 * 8192)
const CB_B: i32 = 4096; // round( 0.5      * 8192)

const CR_R: i32 = 4096; // round( 0.5      * 8192)
const CR_G: i32 = -3430; // round(-0.418688 * 8192)
const CR_B: i32 = -666; // round(-0.081312 * 8192)

impl<T: Pixel> PlanarImage<T> {
    /// Build from interleaved RGB samples (`r,g,b,r,g,b,...`).
    /// AV1 identity matrix mapping: plane0=G, plane1=B, plane2=R.
    pub fn from_interleaved_rgb(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        rgb: &[T],
    ) -> Self {
        assert_eq!(rgb.len(), width * height * 3);
        let n = width * height;
        let mut g = vec![T::default(); n];
        let mut b = vec![T::default(); n];
        let mut r = vec![T::default(); n];
        for (((px, g), b), r) in rgb
            .chunks_exact(3)
            .zip(g.iter_mut())
            .zip(b.iter_mut())
            .zip(r.iter_mut())
        {
            *r = px[0];
            *g = px[1];
            *b = px[2];
        }
        PlanarImage {
            width,
            height,
            bit_depth,
            planes: [g, b, r],
        }
    }

    /// Build from interleaved RGB samples (`r,g,b,r,g,b,...`).
    /// AV1 identity matrix mapping: plane0=G, plane1=B, plane2=R.
    pub fn from_luma(width: usize, height: usize, bit_depth: BitDepth, luma: &[T]) -> Self {
        assert_eq!(luma.len(), width * height);

        PlanarImage {
            width,
            height,
            bit_depth,
            planes: [luma.to_vec(), vec![], vec![]],
        }
    }

    /// Reconstruct interleaved RGB from the GBR planes.
    pub fn to_interleaved_rgb(&self) -> Vec<T> {
        let n = self.width * self.height;
        let mut out = vec![T::default(); n * 3];
        for (((dst, &r), &g), &b) in out
            .chunks_exact_mut(3)
            .zip(self.planes[2].iter())
            .zip(self.planes[0].iter())
            .zip(self.planes[1].iter())
        {
            dst[0] = r;
            dst[1] = g;
            dst[2] = b;
        }
        out
    }
}

/// Result of encoding.
pub struct Encoded {
    pub bytes: Vec<u8>,
    /// Always true for lossless coding; kept for API compatibility.
    pub lossless_verified: bool,
}

/// Encode a **lossy** 8-bit 4:4:4 still to a conformant AV1 OBU stream. Width
/// and height must be multiples of 64.
///
/// The frame is tiled into 64x64 superblocks (raster order, single tile), each
/// split uniformly into 8x8 blocks coded `DC_PRED` + `TX_8X8` (`DCT_DCT`) and
/// quantized by `base_q_idx` (keep `<= 20` to stay in coefficient qctx 0). A
/// reconstruction loop runs inside the encoder so each block's DC prediction
/// matches the decoder's, including across superblock boundaries. Verified to
/// decode in dav1d 1.4.1 and ffmpeg/libdav1d from 64x64 to 256x192 (gradient
/// PSNR ~49-57 dB at q=16; flat color is exact).
pub fn encode_still_lossy<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let bd = img.bit_depth;
    let maxv = (1i32 << bd.bits()) - 1;
    let off = (1i32 << (bd.bits() - 1)) as f32;
    let mx = maxv as f32;
    let n = img.planes[0].len();
    let off_q = (off as i32) << Q;
    let mx_i = mx as i32;
    let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
    for (((((yv, cbv), crv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(cb.iter_mut())
        .zip(cr.iter_mut())
        .zip(img.planes[2].iter())
        .zip(img.planes[0].iter())
        .zip(img.planes[1].iter())
    {
        let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());
        *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
        *cbv = ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
        *crv = ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i);
    }
    let bytes = crate::av1real::encode_av1_lossy_image_cs(
        base_q_idx,
        bd.bits(),
        img.width,
        img.height,
        &y,
        &cb,
        &cr,
        color,
        threads,
    );
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a **lossy 4:2:2** still (profile 2). Like [`encode_still_lossy`] but
/// the chroma planes are horizontally subsampled by 2 (BT.601 full-range YCbCr,
/// `RTX_4X8` chroma transforms). The decoder reconstructs full-resolution RGB.
/// Bit-exact reconstruction was verified against dav1d 1.4.1. On the PSNR
/// metric this is roughly neutral-to-slightly-better on colorful content and
/// slightly worse on very smooth content (where 4:4:4 chroma is already
/// skip-dominated and nearly free), so 4:4:4 remains the default.
pub fn encode_still_lossy_422<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let (w, h) = (img.width, img.height);
    let bd = img.bit_depth;
    let maxv = (1i32 << bd.bits()) - 1;
    let off = (1i32 << (bd.bits() - 1)) as f32;
    let mx = maxv as f32;
    let cw = w.div_ceil(2);
    let mut y = vec![0i32; w * h];

    let off_q = (off as i32) << Q;
    let mx_i = mx as i32;

    let mut fcb_q = vec![0i32; w * h];
    let mut fcr_q = vec![0i32; w * h];

    for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(fcb_q.iter_mut())
        .zip(fcr_q.iter_mut())
        .zip(img.planes[2].iter())
        .zip(img.planes[0].iter())
        .zip(img.planes[1].iter())
    {
        let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());

        *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);

        *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
        *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
    }
    const HALF_AVG: i32 = 1 << Q;

    let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);

    for row in 0..h {
        for c in 0..cw {
            let x0 = 2 * c;
            let x1 = (2 * c + 1).min(w - 1);

            let cb0 = fcb_q[row * w + x0];
            let cb1 = fcb_q[row * w + x1];
            let cr0 = fcr_q[row * w + x0];
            let cr1 = fcr_q[row * w + x1];

            cb[row * cw + c] = ((cb0 + cb1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            cr[row * cw + c] = ((cr0 + cr1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
        }
    }
    let bytes = crate::av1real::encode_av1_lossy_image_422(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        threads,
    );
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a **lossy 4:2:0** still (profile 0). Like [`encode_still_lossy`] but
/// chroma is subsampled by 2 both horizontally and vertically (2x2 box average,
/// BT.601 full-range YCbCr, `TX_4X4` chroma transforms — quarter-resolution
/// chroma, same as JPEG/`yuv420p`). Bit-exact reconstruction verified vs dav1d
/// 1.4.1. As with 4:2:2 this does *not* improve RGB-PSNR R-D over the default
/// 4:4:4 path (slimav's chroma skip already makes smooth chroma nearly free), so
/// it exists mainly for pipeline compatibility; **4:4:4 remains the default.**
pub fn encode_still_lossy_420<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    assert!(base_q_idx != 0, "use encode_still for lossless (q=0)");
    let (w, h) = (img.width, img.height);
    let bd = img.bit_depth;
    let maxv = (1i32 << bd.bits()) - 1;
    let off = (1i32 << (bd.bits() - 1)) as f32;
    let mx = maxv as f32;
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

    let off_q = (off as i32) << Q;
    let mx_i = mx as i32;

    let mut y = vec![0i32; w * h];
    let mut fcb_q = vec![0i32; w * h];
    let mut fcr_q = vec![0i32; w * h];

    for (((((yv, fcbv), fcrv), &rr), &gg), &bb) in y
        .iter_mut()
        .zip(fcb_q.iter_mut())
        .zip(fcr_q.iter_mut())
        .zip(img.planes[2].iter())
        .zip(img.planes[0].iter())
        .zip(img.planes[1].iter())
    {
        let (ri, gi, bi) = (rr.to_i32(), gg.to_i32(), bb.to_i32());

        *yv = ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i);
        *fcbv = CB_R * ri + CB_G * gi + CB_B * bi + off_q;
        *fcrv = CR_R * ri + CR_G * gi + CR_B * bi + off_q;
    }

    const HALF_AVG: i32 = 1 << (Q + 1); // rounding bias for >> (Q+2)

    let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);

    for row in 0..ch {
        for c in 0..cw {
            let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
            let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));

            let avg_q =
                |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];

            cb[row * cw + c] = ((avg_q(&fcb_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            cr[row * cw + c] = ((avg_q(&fcr_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
        }
    }
    let bytes = crate::av1real::encode_av1_lossy_image_420(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        threads,
    );
    Encoded {
        bytes,
        lossless_verified: false,
    }
}

/// Encode a single grayscale plane as a **monochrome** AV1 still
/// (`mono_chrome = 1`, one luma plane). This is the form AVIF uses for an alpha
/// auxiliary image — alpha in AV1/AVIF is not a 4th channel but a separate
/// monochrome image the container references as the alpha aux item. `plane` is
/// the `width*height` grayscale (e.g. alpha) raster; `full_range` sets
/// `color_range` (alpha is normally full range). `base_q_idx` controls quality;
/// `threads`: `0` = all cores, `1` = serial, `N` = up to N (the plane is tiled
/// toward the thread count, and large planes tile by size, exactly like the
/// color encoders). For exact alpha, use a small `base_q_idx`.
pub fn encode_still_mono<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    full_range: bool,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_400()?;
    assert!(base_q_idx != 0, "monochrome lossless is not yet supported");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let luma: Vec<i32> = planar_image.planes[0]
        .iter()
        .map(|v| v.to_i32().clamp(0, maxv))
        .collect();
    let bytes = crate::av1real::encode_av1_mono_image(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &luma,
        full_range,
        threads,
    );
    Ok(bytes)
}

/// Encode a 64×64 8-bit 4:4:4 still image to a conformant AV1 OBU stream.
///
/// # Panics
/// Panics if `img.width != 64 || img.height != 64 || img.bit_depth != 8`.
/// (Arbitrary-size / higher-bit-depth support requires multi-superblock tiling
/// and quantizer changes that are not yet implemented.)
pub fn encode_still<T: Pixel>(img: &PlanarImage<T>) -> Encoded {
    encode_still_with(img, &ColorEncoding::identity_rgb(), 1)
}

/// Encode a lossless 4:4:4 still with explicit color signaling.
///
/// `color` is written verbatim into the AV1 sequence-header `color_config()`
/// (primaries / transfer / matrix / range). The coded planes are emitted
/// without any color transform — pass RGB planes with
/// [`ColorEncoding::identity_rgb()`], or pre-converted YCgCo/etc. with the
/// matching matrix. For HDR metadata OBUs (CLL, MDCV, T.35) append them
/// manually using the helpers in [`crate::obu`] before passing to a muxer.
pub fn encode_still_with<T: Pixel>(
    img: &PlanarImage<T>,
    color: &ColorEncoding,
    threads: usize,
) -> Encoded {
    assert!(
        img.width > 0 && img.height > 0,
        "width/height must be non-zero"
    );
    let profile: u32 = if img.bit_depth == BitDepth::Twelve {
        2
    } else {
        1
    };
    let (w, h) = (img.width, img.height);
    let (w8, h8) = (crate::av1real::align8(w), crate::av1real::align8(h));
    let to_i16 = |p: &[T]| p.iter().map(|p| p.to_i32() as i16).collect::<Vec<i16>>();
    let planes_i16: [Vec<i16>; 3] = [
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[0]), w, h, w8, h8),
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[1]), w, h, w8, h8),
        crate::av1real::pad_to_mult8(&to_i16(&img.planes[2]), w, h, w8, h8),
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp(
        w as u32,
        h as u32,
        profile,
        img.bit_depth.bits(),
        color,
    ));
    bytes.extend_from_slice(&crate::av1real::encode_lossless_frame_obus(
        img.bit_depth.bits(),
        w8,
        h8,
        &planes_i16,
        threads,
    ));
    Encoded {
        bytes,
        lossless_verified: true,
    }
}

/// Encode a pre-converted 4:4:4 YCbCr still.
///
/// Y, Cb, Cr are full-resolution (`width × height`) samples. The AV1 bitstream
/// carries full-range BT.601 YCbCr signaling (profile 1 for ≤10-bit, 2 for 12-bit).
pub fn encode_yuv444<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_444()?;
    assert!(base_q_idx != 0, "use encode_still for lossless");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let bytes = crate::av1real::encode_av1_lossy_image_cs(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        threads,
    );
    Ok(bytes)
}

/// Encode a pre-subsampled 4:2:2 YCbCr still.
///
/// `cb` and `cr` must each be `ceil(width/2) × height` samples. The AV1 bitstream
/// uses AV1 profile 2 (4:2:2 / 12-bit profile).
pub fn encode_yuv422<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_422()?;
    assert!(base_q_idx != 0, "use encode_still for lossless");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let bytes = crate::av1real::encode_av1_lossy_image_422(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        threads,
    );
    Ok(bytes)
}

/// Encode a pre-subsampled 4:2:0 YCbCr still.
///
/// `cb` and `cr` must each be `ceil(width/2) × ceil(height/2)` samples. The AV1
/// bitstream uses AV1 profile 0 (4:2:0 main profile).
pub fn encode_yuv420<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: &ColorEncoding,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_420()?;
    assert!(base_q_idx != 0, "use encode_still for lossless");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let bytes = crate::av1real::encode_av1_lossy_image_420(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        threads,
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Truly-arbitrary-size lossless guard: a 70×50 image (padded to a 72×56
    /// coded grid, header signals 70×50). Checks the public `encode_still` path
    /// is deterministic. Verified bit-exact through dav1d 1.4.1 and ffmpeg.
    #[test]
    fn lossless_70x50_arbitrary() {
        let (w, h) = (70usize, 50usize);
        let mut rgb = vec![0u8; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                rgb[i] = (x % 64) as u8;
                rgb[i + 1] = (y % 64) as u8;
                rgb[i + 2] = ((x + y) % 64) as u8;
            }
        }
        let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb);
        let out = encode_still(&img);
        assert!(out.lossless_verified);
        // header signals the exact (unpadded) frame size
        let sum: u64 = out.bytes.iter().map(|&b| b as u64).sum();
        assert_eq!(out.bytes.len(), 1493);
        assert_eq!(sum, 200614);
    }

    /// 10- and 12-bit lossless guards (40×24, identity RGB in `u16` planes).
    /// Bytes captured after verifying maxdiff 0 through dav1d 1.4.1 at both
    /// depths (high-bitdepth y4m, 16-bit LE samples, coded plane order G,B,R).
    /// The WHT/coef coding is bit-depth-agnostic; only the predictor base
    /// (`1<<(bd-1)`) and the `color_config` high_bitdepth/twelve_bit + profile
    /// signalling change with depth.
    #[test]
    fn lossless_high_bitdepth_stable() {
        let (w, h) = (40usize, 24usize);
        for (bd, len, sum, head) in [
            (10u8, 1519usize, 202296u64, [18u8, 0, 10, 8, 56, 21]),
            (12u8, 1522usize, 203041u64, [18u8, 0, 10, 9, 88, 21]),
        ] {
            let m = (1u32 << bd) as u16;
            let mut rgb = vec![0u16; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    rgb[i] = (((x * 5 + y * 3) as u32) as u16) % m;
                    rgb[i + 1] = (((x * 2 + y * 7) as u32) as u16) % m;
                    rgb[i + 2] = (((x * y + 1) as u32) as u16) % m;
                }
            }
            let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::from_u8(bd).unwrap(), &rgb);
            let p = encode_still(&img).bytes;
            assert_eq!(p.len(), len, "bd={} length", bd);
            assert_eq!(
                p.iter().map(|&x| x as u64).sum::<u64>(),
                sum,
                "bd={} sum",
                bd
            );
            assert_eq!(&p[..6], &head, "bd={} head", bd);
        }
    }

    /// Lossless frames must tile like the lossy path: a small frame stays a
    /// single combined `OBU_FRAME` (type 6), while a frame wider than 4096px is
    /// forced to multiple tile columns and emitted as `OBU_FRAME_HEADER` (3) +
    /// `OBU_TILE_GROUP` (4). (The previous single-tile lossless path mis-signalled
    /// the wide case, so the decoder, deriving a non-zero minimum tile count,
    /// could not parse it.)
    #[test]
    fn lossless_wide_frame_is_multitile() {
        fn obu_types(buf: &[u8]) -> Vec<u8> {
            let mut p = 0;
            let mut out = Vec::new();
            while p < buf.len() {
                let hb = buf[p];
                let typ = (hb >> 3) & 0xf;
                let ext = (hb >> 2) & 1;
                let has_size = (hb >> 1) & 1;
                let mut q = p + 1 + ext as usize;
                let mut sz = buf.len() - q;
                if has_size == 1 {
                    let (mut v, mut s) = (0usize, 0u32);
                    loop {
                        let x = buf[q];
                        q += 1;
                        v |= ((x & 0x7f) as usize) << s;
                        if x & 0x80 == 0 {
                            break;
                        }
                        s += 7;
                    }
                    sz = v;
                }
                out.push(typ);
                p = q + sz;
            }
            out
        }
        let mk = |w: usize, h: usize| {
            let mut rgb = vec![0u8; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    rgb[i] = ((x * 7 + y * 3) % 256) as u8;
                    rgb[i + 1] = ((x ^ y) % 256) as u8;
                    rgb[i + 2] = ((x + y * 5) % 256) as u8;
                }
            }
            encode_still(&PlanarImage::from_interleaved_rgb(
                w,
                h,
                BitDepth::Eight,
                &rgb,
            ))
            .bytes
        };

        let small = obu_types(&mk(96, 64));
        assert!(
            small.contains(&6),
            "small lossless -> OBU_FRAME (6): {small:?}"
        );
        assert!(
            !small.contains(&4),
            "small lossless -> no tile group: {small:?}"
        );

        let wide = obu_types(&mk(4160, 64));
        assert!(
            wide.contains(&3) && wide.contains(&4),
            "wide lossless -> frame header + tile group (3,4): {wide:?}"
        );
        assert!(
            !wide.contains(&6),
            "wide lossless must not use OBU_FRAME: {wide:?}"
        );

        // Threading is deterministic for a fixed tiling: a >4096px-wide frame is
        // 2 tile columns at both 1 and 2 threads, so the bytes must be identical.
        let color = ColorEncoding::identity_rgb();
        let mut rgb = vec![0u8; 4160 * 64 * 3];
        for (i, b) in rgb.iter_mut().enumerate() {
            *b = (i * 31 % 256) as u8;
        }
        let img = PlanarImage::from_interleaved_rgb(4160, 64, BitDepth::Eight, &rgb);
        let s1 = encode_still_with(&img, &color, 1).bytes;
        let s2 = encode_still_with(&img, &color, 2).bytes;
        assert_eq!(
            s1, s2,
            "lossless threaded bytes must match serial (same tiling)"
        );
    }

    /// Lossy 10/12-bit guard across 4:4:4 / 4:2:2 / 4:2:0 (40×24, q80). Byte
    /// lengths + sums captured after verifying the encoder's reconstruction is
    /// bit-exact against dav1d 1.4.1 (`recon == decode`, maxdiff 0) at both
    /// depths and all three chroma formats, and that dav1d reports the correct
    /// `C444p10/C422p10/C420p10` (and `p12`) format.
    #[test]
    #[test]
    #[test]
    fn lossy_high_bitdepth_stable() {
        let (w, h) = (40usize, 24usize);
        for (bd, exp) in [
            (10u8, [(96usize, 10589u64), (93, 10563), (82, 9104)]),
            (12u8, [(59, 5872), (54, 5088), (51, 4999)]),
        ] {
            let m = (1u32 << bd) as u16;
            let mut rgb = vec![0u16; w * h * 3];
            for y in 0..h {
                for x in 0..w {
                    let i = (y * w + x) * 3;
                    rgb[i] = (((x * 5 + y * 3) as u32) as u16) % m;
                    rgb[i + 1] = (((x * 2 + y * 7) as u32) as u16) % m;
                    rgb[i + 2] = (((x * y + 1) as u32) as u16) % m;
                }
            }
            let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::from_u8(bd).unwrap(), &rgb);
            let outs = [
                encode_still_lossy(&img, 80, &crate::color::ColorEncoding::srgb_ycbcr(), 1),
                encode_still_lossy_422(&img, 80, &crate::color::ColorEncoding::srgb_ycbcr(), 1),
                encode_still_lossy_420(&img, 80, &crate::color::ColorEncoding::srgb_ycbcr(), 1),
            ];
            for (k, o) in outs.iter().enumerate() {
                assert!(!o.lossless_verified);
                assert_eq!(o.bytes.len(), exp[k].0, "bd={} fmt={} len", bd, k);
                assert_eq!(
                    o.bytes.iter().map(|&x| x as u64).sum::<u64>(),
                    exp[k].1,
                    "bd={} fmt={} sum",
                    bd,
                    k
                );
            }
        }
    }

    /// Non-multiple-of-8 sizes must not panic and must produce a non-empty
    /// stream for both very small and odd dimensions.
    #[test]
    fn arbitrary_sizes_do_not_panic() {
        for &(w, h) in &[(1usize, 1usize), (17, 17), (65, 33), (127, 129), (33, 7)] {
            let rgb = vec![100u8; w * h * 3];
            let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Twelve, &rgb);
            assert!(!encode_still(&img).bytes.is_empty());
            assert!(
                !encode_still_lossy(&img, 16, &crate::color::ColorEncoding::srgb_ycbcr(), 0)
                    .bytes
                    .is_empty()
            );
        }
    }
}
