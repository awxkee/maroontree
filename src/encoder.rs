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

/// Convert RGB rows to three output planes in parallel row bands. `f` maps one
/// (r, g, b) pixel to the three outputs; planes are zipped per band.
fn csc_rows_par<T: Pixel + Sync>(
    pool: &crate::par::Pool,
    w: usize,
    rgb: [&[T]; 3],
    out: [&mut [i32]; 3],
    f: impl Fn(i32, i32, i32) -> (i32, i32, i32) + Sync,
) {
    const BAND: usize = 64; // rows per work item
    let [o0, o1, o2] = out;
    #[allow(clippy::type_complexity)]
    let items: Vec<(usize, (&mut [i32], (&mut [i32], &mut [i32])))> = o0
        .chunks_mut(BAND * w)
        .zip(o1.chunks_mut(BAND * w).zip(o2.chunks_mut(BAND * w)))
        .enumerate()
        .collect();
    pool.for_each(pool.width(), items, |(bi, (b0, (b1, b2)))| {
        let base = bi * BAND * w;
        for (i, ((v0, v1), v2)) in b0
            .iter_mut()
            .zip(b1.iter_mut())
            .zip(b2.iter_mut())
            .enumerate()
        {
            let j = base + i;
            let (a, b, c) = f(rgb[0][j].to_i32(), rgb[1][j].to_i32(), rgb[2][j].to_i32());
            *v0 = a;
            *v1 = b;
            *v2 = c;
        }
    });
}

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
    /// the `*_with_alpha` paths use — the color planes and the alpha plane are
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

#[allow(clippy::too_many_arguments)]
pub fn encode_still_lossy<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
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
    let pool = crate::par::Pool::new(threads);
    let (mut y, mut cb, mut cr) = (vec![0i32; n], vec![0i32; n], vec![0i32; n]);
    csc_rows_par(
        &pool,
        img.width,
        [&img.planes[2], &img.planes[0], &img.planes[1]],
        [&mut y, &mut cb, &mut cr],
        |ri, gi, bi| {
            (
                ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i),
                ((CB_R * ri + CB_G * gi + CB_B * bi + off_q + HALF) >> Q).clamp(0, mx_i),
                ((CR_R * ri + CR_G * gi + CR_B * bi + off_q + HALF) >> Q).clamp(0, mx_i),
            )
        },
    );
    crate::dispatch::encode_lossy_444(
        base_q_idx,
        bd.bits(),
        img.width,
        img.height,
        &y,
        &cb,
        &cr,
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_still_lossy_422<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
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

    let pool = crate::par::Pool::new(threads);
    csc_rows_par(
        &pool,
        w,
        [&img.planes[2], &img.planes[0], &img.planes[1]],
        [&mut y, &mut fcb_q, &mut fcr_q],
        |ri, gi, bi| {
            (
                ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i),
                CB_R * ri + CB_G * gi + CB_B * bi + off_q,
                CR_R * ri + CR_G * gi + CR_B * bi + off_q,
            )
        },
    );
    const HALF_AVG: i32 = 1 << Q;

    let (mut cb, mut cr) = (vec![0i32; cw * h], vec![0i32; cw * h]);

    // Horizontal 2:1 averaging, parallel over disjoint output row bands.
    {
        #[allow(clippy::type_complexity)]
        let items: Vec<(usize, (&mut [i32], &mut [i32]))> = cb
            .chunks_mut(cw)
            .zip(cr.chunks_mut(cw))
            .enumerate()
            .collect();
        let (fcb, fcr) = (&fcb_q, &fcr_q);
        pool.for_each(pool.width(), items, |(row, (cbr, crr))| {
            for (c, (cb, cr)) in cbr.iter_mut().zip(crr.iter_mut()).enumerate() {
                let x0 = 2 * c;
                let x1 = (2 * c + 1).min(w - 1);
                let cb0 = fcb[row * w + x0];
                let cb1 = fcb[row * w + x1];
                let cr0 = fcr[row * w + x0];
                let cr1 = fcr[row * w + x1];
                *cb = ((cb0 + cb1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
                *cr = ((cr0 + cr1 + HALF_AVG) >> (Q + 1)).clamp(0, mx_i);
            }
        });
    }
    crate::dispatch::encode_lossy_422(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn encode_still_lossy_420<T: Pixel>(
    img: &PlanarImage<T>,
    base_q_idx: u8,
    color: Option<&Cicp>,
    threads: usize,
    speed: Speed,
    aq: bool,
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
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

    let pool = crate::par::Pool::new(threads);
    csc_rows_par(
        &pool,
        w,
        [&img.planes[2], &img.planes[0], &img.planes[1]],
        [&mut y, &mut fcb_q, &mut fcr_q],
        |ri, gi, bi| {
            (
                ((Y_R * ri + Y_G * gi + Y_B * bi + HALF) >> Q).clamp(0, mx_i),
                CB_R * ri + CB_G * gi + CB_B * bi + off_q,
                CR_R * ri + CR_G * gi + CR_B * bi + off_q,
            )
        },
    );

    const HALF_AVG: i32 = 1 << (Q + 1); // rounding bias for >> (Q + 2)

    let (mut cb, mut cr) = (vec![0i32; cw * ch], vec![0i32; cw * ch]);

    // 2x2 averaging, parallel over disjoint output row bands.
    {
        #[allow(clippy::type_complexity)]
        let items: Vec<(usize, (&mut [i32], &mut [i32]))> = cb
            .chunks_mut(cw)
            .zip(cr.chunks_mut(cw))
            .enumerate()
            .collect();
        let (fcb, fcr) = (&fcb_q, &fcr_q);
        pool.for_each(pool.width(), items, |(row, (cbr, crr))| {
            for (c, (cb, cr)) in cbr.iter_mut().zip(crr.iter_mut()).enumerate() {
                let (x0, x1) = (2 * c, (2 * c + 1).min(w - 1));
                let (y0, y1) = (2 * row, (2 * row + 1).min(h - 1));

                let avg_q =
                    |f: &[i32]| f[y0 * w + x0] + f[y0 * w + x1] + f[y1 * w + x0] + f[y1 * w + x1];

                *cb = ((avg_q(fcb) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
                *cr = ((avg_q(fcr) + HALF_AVG) >> (Q + 2)).clamp(0, mx_i);
            }
        });
    }
    crate::dispatch::encode_lossy_420(
        base_q_idx,
        bd.bits(),
        w,
        h,
        &y,
        &cb,
        &cr,
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
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
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Result<Vec<u8>, EncodeError> {
    validate_dims(img.width as u32, img.height as u32)?;
    img.validate_400()?;
    let maxv = (1i32 << bit_depth.bits()) - 1;
    if base_q_idx == 0 {
        let luma: Vec<i16> = img.planes[0]
            .iter()
            .map(|v| v.to_i32().clamp(0, maxv) as i16)
            .collect();
        return Ok(crate::dispatch::encode_lossless_monochrome(
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
    let bytes = crate::dispatch::encode_lossy_monochrome(
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
        wiener,
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
        crate::coder::VarianceBoost::off(),
        false,
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
        crate::coder::VarianceBoost::off(),
        false,
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

/// Use AV1's GBR identity matrix only when the caller supplied no CICP.
fn lossless_rgb_cicp(color: Option<&Cicp>) -> Cicp {
    color.copied().unwrap_or_else(Cicp::identity_rgb)
}

/// Encode a lossless 4:4:4 still with color signaling.
pub fn encode_lossless_obu<T: Pixel>(
    img: &PlanarImage<T>,
    color: Option<&Cicp>,
    threads: usize,
) -> Result<Vec<u8>, EncodeError> {
    img.validate_444()?;
    let effective_color = lossless_rgb_cicp(color);
    let profile: u32 = if img.bit_depth == BitDepth::Twelve {
        2
    } else {
        1
    };
    let (w, h) = (img.width, img.height);
    let (w8, h8) = (crate::coder::align8(w), crate::coder::align8(h));
    let to_i16 = |p: &[T]| p.iter().map(|p| p.to_i32() as i16).collect::<Vec<i16>>();
    let planes_i16: [Vec<i16>; 3] = [
        crate::coder::pad_to_mult8(&to_i16(&img.planes[0]), w, h, w8, h8),
        crate::coder::pad_to_mult8(&to_i16(&img.planes[1]), w, h, w8, h8),
        crate::coder::pad_to_mult8(&to_i16(&img.planes[2]), w, h, w8, h8),
    ];
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&temporal_delimiter());
    bytes.extend_from_slice(&crate::obu::sequence_header_cicp(
        w as u32,
        h as u32,
        profile,
        img.bit_depth.bits(),
        Some(&effective_color),
        false,
        false,
    ));
    bytes.extend_from_slice(&crate::coder::encode_lossless_frame_obus(
        img.bit_depth.bits(),
        w8,
        h8,
        w,
        h,
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
    let effective_color = lossless_rgb_cicp(cfg.color_encoding.as_ref());
    let obu = encode_lossless_obu(img, Some(&effective_color), cfg.threads)?;
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
        Some(&effective_color),
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

    let effective_color = lossless_rgb_cicp(cfg.color_encoding.as_ref());
    let obu = encode_lossless_obu(&img.packed_3(), Some(&effective_color), cfg.threads)?;

    let alpha_obu = encode_lossy_gray_obu(
        &img.packed_alpha_4(),
        img.bit_depth,
        0,
        true,
        cfg.threads,
        Speed::Slow,
        false,
        crate::coder::VarianceBoost::off(),
        false,
        false,
    )?;
    let mut effective_cfg = cfg.clone();
    effective_cfg.color_encoding = Some(effective_color);
    finalize_with_alpha(
        obu,
        alpha_obu,
        img.width as u32,
        img.height as u32,
        img.bit_depth.bits(),
        cfg.chroma,
        &effective_cfg,
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
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_444()?;
    assert_ne!(base_q_idx, 0, "use encode_still for lossless");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let pool = crate::par::Pool::new(threads);
    let bytes = crate::dispatch::encode_lossy_444(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
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
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_422()?;
    assert!(base_q_idx != 0, "4:2:2 doesn't support lossless encoding");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let pool = crate::par::Pool::new(threads);
    let bytes = crate::dispatch::encode_lossy_422(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
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
    vb: crate::coder::VarianceBoost,
    cdef: bool,
    wiener: bool,
) -> Result<Vec<u8>, EncodeError> {
    planar_image.validate_420()?;
    assert!(base_q_idx != 0, "use encode_still for lossless");
    let maxv = (1i32 << bit_depth.bits()) - 1;
    let to_i = |p: &[T]| {
        p.iter()
            .map(|v| v.to_i32().clamp(0, maxv))
            .collect::<Vec<i32>>()
    };
    let pool = crate::par::Pool::new(threads);
    let bytes = crate::dispatch::encode_lossy_420(
        base_q_idx,
        bit_depth.bits(),
        planar_image.width,
        planar_image.height,
        &to_i(&planar_image.planes[0]),
        &to_i(&planar_image.planes[1]),
        &to_i(&planar_image.planes[2]),
        color,
        &pool,
        speed,
        aq,
        vb,
        cdef,
        wiener,
    );
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coder::VarianceBoost;
    use std::path::PathBuf;
    use std::process::Command;

    #[test]
    fn lossless_rgb_preserves_explicit_cicp_and_defaults_to_identity() {
        let explicit = Cicp::srgb_ycbcr();
        assert_eq!(lossless_rgb_cicp(Some(&explicit)), explicit);
        assert_eq!(
            lossless_rgb_cicp(None).matrix,
            crate::color::MatrixCoefficients::Identity
        );
    }

    #[test]
    fn lossless_rgb_avif_preserves_explicit_or_defaults_identity() {
        let decoder = std::env::var_os("AVIFDEC").map(PathBuf::from).or_else(|| {
            let path = PathBuf::from("/opt/homebrew/bin/avifdec");
            path.is_file().then_some(path)
        });
        let Some(decoder) = decoder else {
            return;
        };
        let (w, h) = (8usize, 8usize);
        let rgb: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                [
                    ((i * 37) & 255) as u8,
                    ((i * 71) & 255) as u8,
                    ((i * 13) & 255) as u8,
                ]
            })
            .collect();
        let image = PlanarImage::from_interleaved_rgb(w, h, BitDepth::Eight, &rgb).unwrap();
        for (color, expected_matrix) in [(None, 0), (Some(Cicp::srgb_ycbcr()), 6)] {
            let mut cfg = EncodeConfig::new()
                .with_chroma(ChromaFormat::Yuv444)
                .with_threads(1);
            cfg = match color {
                Some(cicp) => cfg.with_cicp(cicp),
                None => cfg.without_cicp(),
            };
            let avif = encode_lossless(&image, &cfg).unwrap();
            let input = std::env::temp_dir().join(format!(
                "maroontree-lossless-rgb-{}-{expected_matrix}.avif",
                std::process::id()
            ));
            std::fs::write(&input, avif).unwrap();
            let output = Command::new(&decoder)
                .arg("--info")
                .arg(&input)
                .output()
                .unwrap();
            let _ = std::fs::remove_file(input);
            assert!(output.status.success());
            let info = String::from_utf8_lossy(&output.stdout);
            assert!(
                info.contains(&format!("Matrix Coeffs. : {expected_matrix}")),
                "lossless RGB signaled the wrong matrix:\n{info}"
            );
        }
    }

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
                    true,
                    false
                )
                .is_empty()
            );
        }
    }
}
