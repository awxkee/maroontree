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
use crate::Speed;
use crate::avif::{
    checked_buffer_size, finalize_color, finalize_with_alpha, make_av1c, validate_dims,
};
use crate::color::Cicp;
use crate::err::EncodeError;
use crate::obu::temporal_delimiter;
use crate::pixel::Pixel;
use crate::{BitDepth, ChromaFormat, EncodeConfig, isobmff};

pub struct PlanarImage<T: Pixel> {
    pub width: usize,
    pub height: usize,
    pub bit_depth: BitDepth,
    pub planes: [Vec<T>; 4],
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
    /// AV1 identity matrix mapping: plane0=G, plane1=B, plane2=R. No alpha.
    pub fn from_interleaved_rgb(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        rgb: &[T],
    ) -> Result<Self, EncodeError> {
        if rgb.len() != width * height * 3 {
            return Err(EncodeError::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            });
        }
        let n = width * height;
        let mut g = vec![T::default(); n];
        let mut b = vec![T::default(); n];
        let mut r = vec![T::default(); n];
        for (((px, g), b), r) in rgb
            .as_chunks::<3>()
            .0
            .iter()
            .zip(g.iter_mut())
            .zip(b.iter_mut())
            .zip(r.iter_mut())
        {
            *r = px[0];
            *g = px[1];
            *b = px[2];
        }
        Ok(PlanarImage {
            width,
            height,
            bit_depth,
            planes: [g, b, r, Vec::new()],
        })
    }

    /// Build from interleaved RGBA samples (`r,g,b,a,r,g,b,a,...`) in a single
    /// pass: plane0=G, plane1=B, plane2=R, plane3=A. This is the deinterleave
    /// the `*_with_alpha` paths use — the colour planes and the alpha plane are
    /// split once, with no intermediate RGB buffer.
    pub fn from_interleaved_rgba(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        rgba: &[T],
    ) -> Result<Self, EncodeError> {
        if rgba.len() != width * height * 4 {
            return Err(EncodeError::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            });
        }
        let n = width * height;
        let mut g = vec![T::default(); n];
        let mut b = vec![T::default(); n];
        let mut r = vec![T::default(); n];
        let mut a = vec![T::default(); n];
        for ((((px, g), b), r), a) in rgba
            .as_chunks::<4>()
            .0
            .iter()
            .zip(g.iter_mut())
            .zip(b.iter_mut())
            .zip(r.iter_mut())
            .zip(a.iter_mut())
        {
            *r = px[0];
            *g = px[1];
            *b = px[2];
            *a = px[3];
        }
        Ok(PlanarImage {
            width,
            height,
            bit_depth,
            planes: [g, b, r, a],
        })
    }

    /// Build a monochrome image from a single luma plane. No alpha.
    pub fn from_luma(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        luma: &[T],
    ) -> Result<Self, EncodeError> {
        if luma.len() != width * height {
            return Err(EncodeError::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            });
        }
        Ok(PlanarImage {
            width,
            height,
            bit_depth,
            planes: [luma.to_vec(), Vec::new(), Vec::new(), Vec::new()],
        })
    }

    /// Build a monochrome-plus-alpha image from interleaved gray/alpha samples
    /// (`l,a,l,a,...`) in a single pass: plane0=luma, plane3=alpha (planes 1/2
    /// stay empty). Mirrors [`Self::from_interleaved_rgba`] for the 2-channel
    /// gray+alpha case.
    pub fn from_interleaved_gray_alpha(
        width: usize,
        height: usize,
        bit_depth: BitDepth,
        gray_alpha: &[T],
    ) -> Result<Self, EncodeError> {
        if gray_alpha.len() != width * height * 2 {
            return Err(EncodeError::InvalidDimensions {
                width: width as u32,
                height: height as u32,
            });
        }
        let n = width * height;
        let mut luma = vec![T::default(); n];
        let mut a = vec![T::default(); n];
        for ((px, luma), a) in gray_alpha
            .as_chunks::<2>()
            .0
            .iter()
            .zip(luma.iter_mut())
            .zip(a.iter_mut())
        {
            *luma = px[0];
            *a = px[1];
        }
        Ok(PlanarImage {
            width,
            height,
            bit_depth,
            planes: [luma, a, Vec::new(), Vec::new()],
        })
    }

    pub(crate) fn packed_3(&self) -> PlanarImage<T> {
        PlanarImage {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            planes: [
                self.planes[0].to_vec(),
                self.planes[1].to_vec(),
                self.planes[2].to_vec(),
                vec![],
            ],
        }
    }

    pub(crate) fn packed_alpha_4(&self) -> PlanarImage<T> {
        PlanarImage {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            planes: [self.planes[3].to_vec(), vec![], vec![], vec![]],
        }
    }

    pub(crate) fn packed_alpha_2(&self) -> PlanarImage<T> {
        PlanarImage {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            planes: [self.planes[1].to_vec(), vec![], vec![], vec![]],
        }
    }

    pub(crate) fn packed_1(&self) -> PlanarImage<T> {
        PlanarImage {
            width: self.width,
            height: self.height,
            bit_depth: self.bit_depth,
            planes: [self.planes[0].to_vec(), vec![], vec![], vec![]],
        }
    }

    /// Reconstruct interleaved RGB from the GBR planes (alpha is dropped).
    pub fn to_interleaved_rgb(&self) -> Vec<T> {
        let n = self.width * self.height;
        let mut out = vec![T::default(); n * 3];
        for (((dst, &r), &g), &b) in out
            .as_chunks_mut::<3>()
            .0
            .iter_mut()
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
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Vec<u8> {
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
    crate::av1real::encode_av1_lossy_image_cs(
        base_q_idx,
        bd.bits(),
        img.width,
        img.height,
        &y,
        &cb,
        &cr,
        color,
        threads,
        speed,
        aq,
        vb,
        cdef,
    )
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
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Vec<u8> {
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
    crate::av1real::encode_av1_lossy_image_422(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        threads,
        speed,
        aq,
        vb,
        cdef,
    )
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
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Vec<u8> {
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
        let cb_r = &mut cb[row * cw..row * cw + cw];
        let cr_r = &mut cr[row * cw..row * cw + cw];
        for (c, (cb, cr)) in cb_r.iter_mut().zip(cr_r.iter_mut()).enumerate() {
            let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
            let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));

            let avg_q =
                |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];

            *cb = ((avg_q(&fcb_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            *cr = ((avg_q(&fcr_q) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
        }
    }
    crate::av1real::encode_av1_lossy_image_420(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        threads,
        speed,
        aq,
        vb,
        cdef,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossy_gray_obu<T: Pixel>(
    img: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    full_range: bool,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    img.validate_400()?;
    let maxv = (1i32 << bit_depth.bits()) - 1;
    if base_q_idx == 0 {
        let luma: Vec<i16> = img.planes[0]
            .iter()
            .map(|v| v.to_i32().clamp(0, maxv) as i16)
            .collect();
        return Ok(crate::av1real::encode_av1_mono_lossless_image(
            bit_depth.bits(),
            img.width,
            img.height,
            &luma,
            full_range,
            threads,
        ));
    }
    let luma: Vec<i32> = img.planes[0]
        .iter()
        .map(|v| v.to_i32().clamp(0, maxv))
        .collect();
    let bytes = crate::av1real::encode_av1_mono_image(
        base_q_idx,
        bit_depth.bits(),
        img.width,
        img.height,
        &luma,
        full_range,
        threads,
        speed,
        aq,
        vb,
        cdef,
    );
    Ok(bytes)
}

pub fn encode_lossless_gray_obu<T: Pixel>(
    img: &PlanarImage<T>,
    full_range: bool,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    img.validate_400()?;
    encode_lossy_gray_obu(
        img,
        img.bit_depth,
        0,
        full_range,
        threads,
        Speed::Slow,
        false,
        crate::av1real::VarianceBoost::off(),
        false,
    )
}

/// Encode a lossless grayscale (monochrome) AVIF still.
pub fn encode_lossless_gray<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    img.validate_400()?;
    let obu = encode_lossy_gray_obu(
        img,
        img.bit_depth,
        0,
        true,
        cfg.threads,
        Speed::Slow,
        false,
        crate::av1real::VarianceBoost::off(),
        false,
    )?;
    finalize_color(
        obu,
        img.width as u32,
        img.height as u32,
        img.bit_depth.bits(),
        ChromaFormat::Monochrome,
        cfg,
    )
}

pub fn encode_lossless_gray_alpha<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    crate::avif::validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    crate::avif::validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    if cfg.chroma != ChromaFormat::Monochrome {
        return Err(EncodeError::UnsupportedChromaFormat(cfg.chroma));
    }
    let luma_obu = encode_lossless_gray(&img.packed_1(), cfg)?;
    let alpha_obu = encode_lossless_gray(&img.packed_alpha_2(), cfg)?;
    finalize_with_alpha(
        luma_obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        img.bit_depth.bits(),
        ChromaFormat::Monochrome,
        cfg,
    )
}

/// Encode a lossless 4:4:4 still with color signaling.
pub fn encode_lossless_obu<T: Pixel>(
    img: &PlanarImage<T>,
    color: Option<&Cicp>,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    img.validate_444()?;
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
    Ok(bytes)
}

/// Encode a lossless 4:4:4 AVIF still with color signaling.
pub fn encode_lossless<T: Pixel>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    img.validate_444()?;
    if cfg.chroma != ChromaFormat::Yuv444 {
        return Err(EncodeError::UnsupportedChromaFormat(cfg.chroma));
    }
    let obu = encode_lossless_obu(img, cfg.color_encoding.as_ref(), cfg.threads)?;
    let av1c = make_av1c(
        &obu,
        img.bit_depth.bits(),
        img.width as u32,
        img.height as u32,
        ChromaFormat::Yuv444,
    );
    isobmff::wrap_av1_image(
        &obu,
        img.width as u32,
        img.height as u32,
        img.bit_depth.bits(),
        3,
        &av1c,
        cfg.color_encoding.as_ref(),
        cfg.icc.as_deref(),
        &cfg.metadata,
    )
}

/// Encode a lossless 4:4:4 AVIF still with color signaling.
pub fn encode_lossless_with_alpha<T: Pixel + Copy>(
    img: &PlanarImage<T>,
    cfg: &EncodeConfig,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    cfg.validate()?;
    crate::avif::validate_buf(&img.planes[0], img.width as u32, img.height as u32, 1)?;
    crate::avif::validate_buf(&img.planes[1], img.width as u32, img.height as u32, 1)?;
    crate::avif::validate_buf(&img.planes[2], img.width as u32, img.height as u32, 1)?;
    crate::avif::validate_buf(&img.planes[3], img.width as u32, img.height as u32, 1)?;
    if cfg.chroma != ChromaFormat::Yuv444 {
        return Err(EncodeError::UnsupportedChromaFormat(cfg.chroma));
    }

    let obu = encode_lossless_obu(&img.packed_3(), cfg.color_encoding.as_ref(), cfg.threads)?;

    let alpha_obu = encode_lossy_gray_obu(
        &img.packed_alpha_4(),
        img.bit_depth,
        0,
        true,
        cfg.threads,
        Speed::Slow,
        false,
        crate::av1real::VarianceBoost::off(),
        false,
    )?;
    finalize_with_alpha(
        obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        img.bit_depth.bits(),
        cfg.chroma,
        cfg,
    )
}

/// Encode a pre-converted 4:4:4 YCbCr still.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_yuv444_obu<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_444()?;
    assert_ne!(base_q_idx, 0, "use encode_still for lossless");
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
        speed,
        aq,
        vb,
        cdef,
    );
    Ok(bytes)
}

/// Encode a pre-subsampled 4:2:2 YCbCr still.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_yuv422_obu<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_422()?;
    assert!(base_q_idx != 0, "4:2:2 doesn't support lossless encoding");
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
        speed,
        aq,
        vb,
        cdef,
    );
    Ok(bytes)
}

/// Encode a pre-subsampled 4:2:0 YCbCr still.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_yuv420_obu<T: Pixel>(
    planar_image: &PlanarImage<T>,
    bit_depth: BitDepth,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::av1real::VarianceBoost,
    cdef: bool,
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
        speed,
        aq,
        vb,
        cdef,
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1real::VarianceBoost;

    /// Non-multiple-of-8 sizes must not panic and must produce a non-empty
    /// stream for both very small and odd dimensions.
    #[test]
    fn arbitrary_sizes_do_not_panic() {
        for &(w, h) in &[(1usize, 1usize), (17, 17), (65, 33), (127, 129), (33, 7)] {
            let rgb = vec![100u8; w * h * 3];
            let img = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Twelve, &rgb).unwrap();
            assert!(!encode_lossless_obu(&img, None, 9).unwrap().is_empty());
            assert!(
                !encode_still_lossy(
                    &img,
                    16,
                    Some(&Cicp::srgb_ycbcr()),
                    0,
                    Speed::Slow,
                    false,
                    VarianceBoost::off(),
                    true
                )
                .is_empty()
            );
        }
    }
}
