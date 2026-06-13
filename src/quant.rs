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

use crate::dct::{
    dct4x4_t, dct4x8_t, dct8x8, dct8x8_t, dct8x16_t, dct16x16, dct16x16_t, dct16x32_t, dct32x32,
    dct32x32_t,
};

/// DC dequant value `dav1d_dq_tbl[bd][q][0]` for bit_depth 8/10/12.
pub(crate) fn dc_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::DC_QLOOKUP_10,
        12 => &crate::coef_q::DC_QLOOKUP_12,
        _ => &crate::coef_q::DC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}
/// AC dequant value `dav1d_dq_tbl[bd][q][1]` for bit_depth 8/10/12.
pub(crate) fn ac_q(base_q_idx: u8, bd: u8) -> u16 {
    let t: &[u16; 256] = match bd {
        10 => &crate::coef_q::AC_QLOOKUP_10,
        12 => &crate::coef_q::AC_QLOOKUP_12,
        _ => &crate::coef_q::AC_QLOOKUP_8,
    };
    t[base_q_idx as usize]
}

pub(crate) fn chroma_dc_delta(base_q_idx: u8) -> i32 {
    let q = base_q_idx as i32;
    if q <= 160 {
        0
    } else {
        -((q - 160) / 3).min(32)
    }
}

/// Inverse-transform clip bounds for `bit_depth`, matching dav1d's `itx_tmpl.c`:
/// returns `(row_min, row_max, col_min, col_max, cf_max)`. 8-bit uses `INT16`
/// for both row and col; for higher depth the row clip is `±2^(bd+7)`, the
/// column clip is `±2^(bd+5)`, and `cf_max == row_max == ~(~127 << bd)`.
pub(crate) fn itx_clips(bd: u8) -> (i32, i32, i32, i32, i32) {
    if bd <= 8 {
        let (mn, mx) = (i16::MIN as i32, i16::MAX as i32);
        (mn, mx, mn, mx, 32767)
    } else {
        let row_max = (1i32 << (bd + 7)) - 1;
        let col_max = (1i32 << (bd + 5)) - 1;
        (!row_max, row_max, !col_max, col_max, row_max)
    }
}

/// Source of the dequant coefficients and inverse-transform clip bounds the
/// DCT/IDCT drivers need. Implemented by [`Quant`], which computes them once
/// per (base_q_idx, bit_depth) so the transforms read them from `self` instead
/// of indexing `dav1d_dq_tbl` and recomputing the clips on every block.
pub(crate) trait Dct {
    /// DC dequant step (`dav1d_dq_tbl[bd][q][0]`). Used by the inverse transform
    /// and as the trellis distortion weight.
    fn dc_q(&self) -> i32;
    /// AC dequant step (`dav1d_dq_tbl[bd][q][1]`).
    fn ac_q(&self) -> i32;
    /// Inverse-transform clips `(row_min, row_max, col_min, col_max, cf_max)`.
    fn clips(&self) -> (i32, i32, i32, i32, i32);
    /// Forward-quantisation multiplier for DC, `round(65536 / dc_q)`, so that the
    /// integer forward DCT's `mul_q16(coeff, q_mult_dc()) ≈ coeff / dc_q` (the
    /// inverse multiplies the level back by `dc_q`, so this round-trips).
    fn q_mult_dc(&self) -> i32;
    /// Forward-quantisation multiplier for AC, `round(65536 / ac_q)`.
    fn q_mult_ac(&self) -> i32;
}

/// Precomputed dequant coefficients + inverse-transform clips for one
/// (base_q_idx, bit_depth). Cheap to copy; build once and hand to the transforms.
#[derive(Clone, Copy)]
pub(crate) struct Quant {
    dc: i32,
    ac: i32,
    q_mult_dc: i32,
    q_mult_ac: i32,
    rmin: i32,
    rmax: i32,
    cmin: i32,
    cmax: i32,
    cf_max: i32,
}

impl Quant {
    pub(crate) fn new(base_q_idx: u8, bd: u8) -> Self {
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc = dc_q(base_q_idx, bd) as i32;
        let ac = ac_q(base_q_idx, bd) as i32;
        Quant {
            dc,
            ac,
            q_mult_dc: (65536.0_f64 / dc as f64).round() as i32,
            q_mult_ac: (65536.0_f64 / ac as f64).round() as i32,
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
        }
    }

    pub(crate) fn new_chroma(base_q_idx: u8, bd: u8) -> Self {
        let dc_idx = (base_q_idx as i32 + chroma_dc_delta(base_q_idx)).clamp(0, 255) as u8;
        let (rmin, rmax, cmin, cmax, cf_max) = itx_clips(bd);
        let dc = dc_q(dc_idx, bd) as i32;
        let ac = ac_q(base_q_idx, bd) as i32;
        Quant {
            dc,
            ac,
            q_mult_dc: (65536.0_f64 / dc as f64).round() as i32,
            q_mult_ac: (65536.0_f64 / ac as f64).round() as i32,
            rmin,
            rmax,
            cmin,
            cmax,
            cf_max,
        }
    }
}

impl Dct for Quant {
    #[inline]
    fn dc_q(&self) -> i32 {
        self.dc
    }
    #[inline]
    fn ac_q(&self) -> i32 {
        self.ac
    }
    #[inline]
    fn clips(&self) -> (i32, i32, i32, i32, i32) {
        (self.rmin, self.rmax, self.cmin, self.cmax, self.cf_max)
    }
    #[inline]
    fn q_mult_dc(&self) -> i32 {
        self.q_mult_dc
    }
    #[inline]
    fn q_mult_ac(&self) -> i32 {
        self.q_mult_ac
    }
}

/// Forward-DCT + quantize an 8x8 residual block into AV1 quantized coefficient
/// levels (raster order, for `encode_tx8_luma_coeffs`). The dav1d 8x8 inverse
/// DCT equals (1/8) x orthonormal DCT, so forward `cf = 8 * orthonormalDCT2(R)`,
/// quantized by dc_q (DC) / ac_q (AC), transposed (`rc = u*8 + v`) to dav1d's
/// coefficient layout. (Calibrated against dav1d: round-trip max error ~1 at q=16.)
pub(crate) fn forward_dct_quant_8x8(residual: &mut [i32; 64], q: &impl Dct) {
    dct8x8(residual, q)
}

/// Trellis (RDOQ) forward 8x8: the integer DCT levels plus the unrounded
/// per-coefficient targets. `.0` is bit-identical to `forward_dct_quant_8x8`.
pub(crate) fn forward_dct_quant_8x8_t(
    residual: &[i32; 64],
    q: &impl Dct,
) -> ([i32; 64], [f64; 64]) {
    dct8x8_t(residual, q)
}

/// `forward_dct_quant_8x8`: orthonormal float DCT (rows then cols), scaled, then
/// /q (dc_q for the (0,0) coefficient, ac_q otherwise). Output in dav1d order
/// `cf[u*16+v]`. The scale is calibrated so the round-trip through the exact
/// integer inverse recovers the residual; only the encoder uses this (recon is
/// the exact inverse), so its precision does not affect bit-exactness.
pub(crate) fn forward_dct_quant_16x16(residual: &mut [i32; 256], q: &impl Dct) {
    // forward_dct_quant_16x16_t(residual, q).0
    dct16x16(residual, q)
}

/// As [`forward_dct_quant_16x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_16x16_t(
    residual: &[i32; 256],
    q: &impl Dct,
) -> ([i32; 256], [f64; 256]) {
    dct16x16_t(residual, q)
}

/// Forward DCT + quantize a 32x32 residual via the shared integer DCT in
/// `crate::dct`. Recon is the exact integer inverse.
pub(crate) fn forward_dct_quant_32x32(residual: &mut [i32; 1024], q: &impl Dct) {
    dct32x32(residual, q)
}

pub(crate) fn forward_dct_quant_32x32_t(
    residual: &[i32; 1024],
    q: &impl Dct,
) -> ([i32; 1024], [f64; 1024]) {
    dct32x32_t(residual, q)
}

/// As [`forward_dct_quant_4x8`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_4x8_t(
    residual: &[i32; 32],
    q: &impl Dct,
) -> ([i32; 32], [f64; 32]) {
    dct4x8_t(residual, q)
}

/// As [`forward_dct_quant_8x16`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_8x16_t(
    residual: &[i32; 128],
    q: &impl Dct,
) -> ([i32; 128], [f64; 128]) {
    dct8x16_t(residual, q)
}

pub(crate) fn forward_dct_quant_16x32_t(
    residual: &[i32; 512],
    q: &impl Dct,
) -> ([i32; 512], [f64; 512]) {
    dct16x32_t(residual, q)
}

/// As [`forward_dct_quant_4x4`] but also returns the pre-round real targets.
pub(crate) fn forward_dct_quant_4x4_t(
    residual: &[i32; 16],
    q: &impl Dct,
) -> ([i32; 16], [f64; 16]) {
    dct4x4_t(residual, q)
}
