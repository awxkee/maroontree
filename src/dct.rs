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

// Coefficients in Q0.16 fixed-point (multiply float by 65536 and round)
// WC4[0] = 0.5411961  -> 35468
// WC4[1] = 1.3065630  -> 85627
// WC8[0] = 0.5097956  -> 33410
// WC8[1] = 0.6013449  -> 39393
// WC8[2] = 0.8999762  -> 58981
// WC8[3] = 2.5629156  -> 167982
// SQRT_2  = 1.4142136  -> 92682

use crate::av1real::Dct;
use std::sync::{Arc, OnceLock};

pub(crate) const WC4_0: i32 = 35468; // 0.541196  * 65536
pub(crate) const WC4_1: i32 = 85627; // 1.306563  * 65536
pub(crate) const WC8_0: i32 = 33410; // k=0: 0.5097956
pub(crate) const WC8_1: i32 = 39410; // k=1: 0.6013449  WAS 39393 ✗
pub(crate) const WC8_2: i32 = 58981; // k=2: 0.8999762
pub(crate) const WC8_3: i32 = 167963; // k=3: 2.5629154  WAS 167982 ✗
pub(crate) const SQRT2: i32 = 92682; // 1.414214  * 65536

/// Multiply a Q15 data value by a Q0.16 coefficient, returning Q15.
/// data is ~15-bit signed, coeff is Q0.16 (up to ~2.6 * 65536 = 170394).
/// Worst case: 32767 * 170394 = ~5.6e9, fits in i64, shift back by 16.
#[inline(always)]
fn mul_q16(data: i32, coeff: i32) -> i32 {
    (((data as i64) * (coeff as i64)) >> 16) as i32
}

/// fmla equivalent: a * SQRT2 + b, all in Q15 domain
#[inline(always)]
fn fmla_sqrt2(a: i32, b: i32) -> i32 {
    mul_q16(a, SQRT2) + b
}

#[inline(always)]
fn dct1d_2_i32(buf: &mut [i32]) {
    let a = buf[0];
    let b = buf[1];
    buf[0] = a + b;
    buf[1] = a - b;
}

#[inline(always)]
fn dct1d_4_i32(buf: &mut [i32; 4]) {
    let mut tmp = [0i32; 4];

    // Even part butterfly
    tmp[0] = buf[0] + buf[3];
    tmp[1] = buf[1] + buf[2];
    dct1d_2_i32(&mut tmp[0..2]);

    // Odd part: scale by WC4 coefficients
    tmp[2] = mul_q16(buf[0] - buf[3], WC4_0);
    tmp[3] = mul_q16(buf[1] - buf[2], WC4_1);
    dct1d_2_i32(&mut tmp[2..4]);

    // fmla: tmp[2] = tmp[2] * sqrt(2) + tmp[3]
    tmp[2] = fmla_sqrt2(tmp[2], tmp[3]);

    buf[0] = tmp[0];
    buf[2] = tmp[1];
    buf[1] = tmp[2];
    buf[3] = tmp[3];
}

#[inline(always)]
pub(crate) fn dct1d_8_i32(buf: &mut [i32; 8]) {
    let mut tmp = [0i32; 8];

    // Even part
    for i in 0..4 {
        tmp[i] = buf[i] + buf[7 - i];
    }
    dct1d_4_i32(<&mut [i32; 4]>::try_from(&mut tmp[..4]).unwrap());

    // Odd part: scale each difference by WC8[i]
    let wc8 = [WC8_0, WC8_1, WC8_2, WC8_3];
    for i in 0..4 {
        tmp[4 + i] = mul_q16(buf[i] - buf[7 - i], wc8[i]);
    }
    dct1d_4_i32(<&mut [i32; 4]>::try_from(&mut tmp[4..8]).unwrap());

    // Post-butterfly combine (mirrors float version exactly)
    tmp[4] = fmla_sqrt2(tmp[4], tmp[5]);
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];

    // Interleave even/odd outputs
    for i in 0..4 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[4 + i];
    }
}

// WC16 coefficients in Q0.16 (value * 65536, rounded)
// 0.5024193  -> 32924
// 0.5224986  -> 34236
// 0.5669440  -> 37144
// 0.6468218  -> 42391
// 0.7881546  -> 51638
// 1.0606777  -> 69496
// 1.7224471  -> 112863
// 5.1011486  -> 334233
pub(crate) const WC16_0: i32 = 32927; 
pub(crate) const WC16_1: i32 = 34242; 
pub(crate) const WC16_2: i32 = 37155; 
pub(crate) const WC16_3: i32 = 42390; 
pub(crate) const WC16_4: i32 = 51653; 
pub(crate) const WC16_5: i32 = 69513; 
pub(crate) const WC16_6: i32 = 112882;
pub(crate) const WC16_7: i32 = 334309;

#[inline(always)]
pub(crate) fn dct1d_16_i32(buf: &mut [i32; 16]) {
    let mut tmp = [0i32; 16];

    // Split into even and odd butterfly pairs
    for i in 0..8 {
        tmp[i] = buf[i] + buf[15 - i];
        tmp[8 + i] = buf[i] - buf[15 - i];
    }

    // Recurse on even half
    dct1d_8_i32(<&mut [i32; 8]>::try_from(&mut tmp[..8]).unwrap());

    // Scale odd half by WC16 then recurse
    let wc16 = [
        WC16_0, WC16_1, WC16_2, WC16_3, WC16_4, WC16_5, WC16_6, WC16_7,
    ];
    for i in 0..8 {
        tmp[8 + i] = mul_q16(tmp[8 + i], wc16[i]);
    }
    dct1d_8_i32(<&mut [i32; 8]>::try_from(&mut tmp[8..16]).unwrap());

    // Post-butterfly odd-half combine (mirrors float exactly)
    tmp[8] = fmla_sqrt2(tmp[8], tmp[9]);
    tmp[9] += tmp[10];
    tmp[10] += tmp[11];
    tmp[11] += tmp[12];
    tmp[12] += tmp[13];
    tmp[13] += tmp[14];
    tmp[14] += tmp[15];

    // Interleave even/odd outputs
    for i in 0..8 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[8 + i];
    }
}

pub(crate) fn dct8x8(input: &mut [i32; 64], quant: &impl Dct) {
    pub(crate) type Dct = dyn Fn(&mut [i32; 64], i32, i32) + Send + Sync;
    static WHT: OnceLock<Arc<Dct>> = OnceLock::new();
    let f = WHT.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::dct8x8_neon_i32;
            Arc::new(|input: &mut [i32; 64], dc_q: i32, ac_q: i32| unsafe {
                dct8x8_neon_i32(input, dc_q, ac_q)
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(dct8x8_scalar)
        }
    });
    f(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer DCT-8 transform core (no quantization). Output layout is
/// `out[r*8 + u] = F[row_freq=r][col_freq=u]`, DC at index 0. This is the single
/// forward transform reused by both the in-place quantizer ([`dct8x8_scalar`])
/// and the trellis/RDOQ variant ([`dct8x8_t`]).
#[inline]
fn dct8x8_coeffs(input: &[i32; 64]) -> [i32; 64] {
    let mut tmp = [0i32; 64];
    // Pass 1: column-wise DCT-8.
    for x in 0..8usize {
        let mut col = [0i32; 8];
        for r in 0..8 {
            col[r] = input[r * 8 + x];
        }
        dct1d_8_i32(&mut col);
        for r in 0..8 {
            tmp[r * 8 + x] = col[r];
        }
    }
    // Pass 2: row-wise DCT-8. Store transposed to the pipeline convention
    // `out[horiz_freq*8 + vert_freq]` (DC at index 0), matching the scan order,
    // the integer inverse, and the 16x16/32x32 cores.
    let mut out = [0i32; 64];
    for r in 0..8usize {
        let mut row: [i32; 8] = tmp[r * 8..r * 8 + 8].try_into().unwrap();
        dct1d_8_i32(&mut row);
        for u in 0..8 {
            out[u * 8 + r] = row[u];
        }
    }
    out
}

#[inline]
#[allow(unused)]
pub(crate) fn dct8x8_scalar(input: &mut [i32; 64], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x8_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = mul_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

#[inline]
pub(crate) fn dct16x16(input: &mut [i32; 256], quant: &impl Dct) {
    pub(crate) type Dct = dyn Fn(&mut [i32; 256], i32, i32) + Send + Sync;
    static WHT: OnceLock<Arc<Dct>> = OnceLock::new();
    let f = WHT.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::dct16x16_neon_i32;
            Arc::new(|input: &mut [i32; 256], dc_q: i32, ac_q: i32| unsafe {
                dct16x16_neon_i32(input, dc_q, ac_q)
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(dct16x16_scalar)
        }
    });
    f(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer DCT-16 transform core (no quantization). Output layout
/// `out[u*16 + v]`, DC at index 0. Reused by [`dct16x16_scalar`] and [`dct16x16_t`].
#[inline]
#[allow(unused)]
fn dct16x16_coeffs(input: &[i32; 256]) -> [i32; 256] {
    let mut tmp = [0i32; 256];
    // Column-wise 1D DCT
    for u in 0..16 {
        let mut col = [0i32; 16];
        for i in 0..16 {
            col[i] = input[i * 16 + u];
        }
        dct1d_16_i32(&mut col);
        for v in 0..16 {
            tmp[v * 16 + u] = col[v];
        }
    }
    // Row-wise 1D DCT
    let mut out = [0i32; 256];
    for v in 0..16 {
        let mut row: [i32; 16] = tmp[v * 16..v * 16 + 16].try_into().unwrap();
        dct1d_16_i32(&mut row);
        // Normalize the integer DCT-16 gain (sqrt(16) per pass -> 16x; the
        // pipeline expects the orthonormal*8 scale) by 1/2.
        for u in 0..16 {
            out[u * 16 + v] = mul_q16(row[u], 32768);
        }
    }
    out
}

#[allow(unused)]
pub(crate) fn dct16x16_scalar(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    let coeffs = dct16x16_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = mul_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

pub(crate) const WC32: [i32; 16] = [
    32808,  // k= 0: 0.5006030
    33127,  // k= 1: 0.5054710  WAS 33393 ✗
    33780,  // k= 2: 0.5154473  WAS 34236 ✗
    34802,  // k= 3: 0.5310426  WAS 35468 ✗
    36248,  // k= 4: 0.5531039  WAS 37144 ✗
    38203,  // k= 5: 0.5829350  WAS 39367 ✗
    40796,  // k= 6: 0.6225041  WAS 42391 ✗
    44224,  // k= 7: 0.6748083  WAS 46341 ✗
    48794,  // k= 8: 0.7445363  WAS 51638 ✗
    55008,  // k= 9: 0.8393496  WAS 58981 ✗
    63738,  // k=10: 0.9725682  WAS 69496 ✗
    76640,  // k=11: 1.1694399  WAS 85627 ✗
    97266,  // k=12: 1.4841646  WAS 112863 ✗
    134859, // k=13: 2.0577810  WAS 167982 ✗
    223321, // k=14: 3.4076084  WAS 334233 ✗
    667812, // k=15: 10.190008  WAS 667829 ✗
];

#[inline(always)]
pub(crate) fn dct1d_32_i32(buf: &mut [i32; 32]) {
    let mut tmp = [0i32; 32];

    // Butterfly split: even and odd halves
    for i in 0..16 {
        tmp[i] = buf[i] + buf[31 - i];
        tmp[16 + i] = buf[i] - buf[31 - i];
    }

    // Recurse on even half
    dct1d_16_i32(<&mut [i32; 16]>::try_from(&mut tmp[..16]).unwrap());

    // Scale odd half by WC32, then recurse
    for i in 0..16 {
        tmp[16 + i] = mul_q16(tmp[16 + i], WC32[i]);
    }
    dct1d_16_i32(<&mut [i32; 16]>::try_from(&mut tmp[16..32]).unwrap());

    // Post-butterfly odd-half combine chain
    tmp[16] = fmla_sqrt2(tmp[16], tmp[17]);
    tmp[17] += tmp[18];
    tmp[18] += tmp[19];
    tmp[19] += tmp[20];
    tmp[20] += tmp[21];
    tmp[21] += tmp[22];
    tmp[22] += tmp[23];
    tmp[23] += tmp[24];
    tmp[24] += tmp[25];
    tmp[25] += tmp[26];
    tmp[26] += tmp[27];
    tmp[27] += tmp[28];
    tmp[28] += tmp[29];
    tmp[29] += tmp[30];
    tmp[30] += tmp[31];

    // Interleave even/odd outputs
    for i in 0..16 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[16 + i];
    }
}

#[inline]
pub(crate) fn dct32x32(input: &mut [i32; 1024], quant: &impl Dct) {
    pub(crate) type Dct = dyn Fn(&mut [i32; 1024], i32, i32) + Send + Sync;
    static WHT: OnceLock<Arc<Dct>> = OnceLock::new();
    let f = WHT.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::dct32x32_neon_i32;
            Arc::new(|input: &mut [i32; 1024], dc_q: i32, ac_q: i32| unsafe {
                dct32x32_neon_i32(input, dc_q, ac_q)
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(dct32x32_scalar)
        }
    });
    f(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer DCT-32 transform core (no quantization). Output layout
/// `out[u*32 + v]`, DC at index 0. Reused by [`dct32x32_scalar`] and [`dct32x32_t`].
#[inline]
fn dct32x32_coeffs(input: &[i32; 1024]) -> [i32; 1024] {
    let mut tmp = [0i32; 1024];
    // Column-wise 1D DCT
    for u in 0..32 {
        let mut col = [0i32; 32];
        for i in 0..32 {
            col[i] = input[i * 32 + u];
        }
        dct1d_32_i32(&mut col);
        for v in 0..32 {
            tmp[v * 32 + u] = col[v];
        }
    }
    // Row-wise 1D DCT
    let mut out = [0i32; 1024];
    for v in 0..32 {
        let mut row: [i32; 32] = tmp[v * 32..v * 32 + 32].try_into().unwrap();
        dct1d_32_i32(&mut row);
        // Normalize the integer DCT-32 gain (32x) to orthonormal*8 by 1/4.
        for u in 0..32 {
            out[u * 32 + v] = mul_q16(row[u], 16384);
        }
    }
    out
}

#[allow(unused)]
pub(crate) fn dct32x32_scalar(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    let coeffs = dct32x32_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = mul_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

#[allow(unused)]
pub(crate) fn dct8x16_i32(input: &mut [i32; 128], quant: &impl Dct) {
    pub(crate) type Dct = dyn Fn(&mut [i32; 128], i32, i32) + Send + Sync;
    static WHT: OnceLock<Arc<Dct>> = OnceLock::new();
    let f = WHT.get_or_init(|| {
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            use crate::neon::dct8x16_neon_i32;
            Arc::new(|input: &mut [i32; 128], dc_q: i32, ac_q: i32| unsafe {
                dct8x16_neon_i32(input, dc_q, ac_q)
            })
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            Arc::new(crate::dct::dct8x16_i32_scalar)
        }
    });
    f(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer 8x16 transform core (8 wide x 16 tall; residual read as
/// `resid[row*8 + col]`). DCT-16 down each of the 8 columns, then DCT-8 across
/// each of the 16 rows. Output layout `out[horiz_freq*16 + vert_freq]`, DC at
/// index 0 — matching the float reference, the scan order and the inverse.
/// Reused by [`dct8x16_i32_scalar`] and [`dct8x16_t`].
#[inline]
fn dct8x16_coeffs(input: &[i32; 128]) -> [i32; 128] {
    // Pass 1: DCT-16 down each of the 8 columns (vertical).
    let mut tmp = [0i32; 128]; // tmp[fy*8 + col], fy = vertical freq
    for col in 0..8usize {
        let mut c = [0i32; 16];
        for row in 0..16 {
            c[row] = input[row * 8 + col];
        }
        dct1d_16_i32(&mut c);
        for fy in 0..16 {
            tmp[fy * 8 + col] = c[fy];
        }
    }
    // Pass 2: DCT-8 across each of the 16 rows (horizontal).
    let mut out = [0i32; 128];
    for fy in 0..16usize {
        let mut r: [i32; 8] = tmp[fy * 8..fy * 8 + 8].try_into().unwrap();
        dct1d_8_i32(&mut r);
        // Normalize the integer 8x16 gain sqrt(8*16)=sqrt(128) to orthonormal*8
        // by 1/sqrt(2) (round(65536/sqrt2) = 46341).
        for fx in 0..8 {
            out[fx * 16 + fy] = mul_q16(r[fx], 46341);
        }
    }
    out
}

#[allow(unused)]
pub(crate) fn dct8x16_i32_scalar(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x16_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = mul_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

#[inline]
fn quant_levels_and_targets<const N: usize>(
    coeffs: &[i32; N],
    q_mult_dc: i32,
    q_mult_ac: i32,
) -> ([i32; N], [f64; N]) {
    let mut cf = [0i32; N];
    let mut tf = [0.0f64; N];
    for i in 0..N {
        let m = if i == 0 { q_mult_dc } else { q_mult_ac };
        cf[i] = mul_q16(coeffs[i], m);
        tf[i] = coeffs[i] as f64 * m as f64 * (1. / 65536.0);
    }
    (cf, tf)
}

#[inline]
fn dct8x8_coeffs_sel(input: &[i32; 64]) -> [i32; 64] {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        unsafe { crate::neon::dct8x8_neon_coeffs(input) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        dct8x8_coeffs(input)
    }
}

#[inline]
fn dct16x16_coeffs_sel(input: &[i32; 256]) -> [i32; 256] {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        unsafe { crate::neon::dct16x16_neon_coeffs(input) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        dct16x16_coeffs(input)
    }
}

#[inline]
fn dct32x32_coeffs_sel(input: &[i32; 1024]) -> [i32; 1024] {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        unsafe { crate::neon::dct32x32_neon_coeffs(input) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        dct32x32_coeffs(input)
    }
}

#[inline]
fn dct8x16_coeffs_sel(input: &[i32; 128]) -> [i32; 128] {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        unsafe { crate::neon::dct8x16_neon_coeffs(input) }
    }
    #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
    {
        dct8x16_coeffs(input)
    }
}

pub(crate) fn dct8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f64; 64]) {
    let coeffs = dct8x8_coeffs_sel(residual);
    quant_levels_and_targets(&coeffs, quant.q_mult_dc(), quant.q_mult_ac())
}

pub(crate) fn dct16x16_t(residual: &[i32; 256], quant: &impl Dct) -> ([i32; 256], [f64; 256]) {
    let coeffs = dct16x16_coeffs_sel(residual);
    quant_levels_and_targets(&coeffs, quant.q_mult_dc(), quant.q_mult_ac())
}

pub(crate) fn dct32x32_t(residual: &[i32; 1024], quant: &impl Dct) -> ([i32; 1024], [f64; 1024]) {
    let coeffs = dct32x32_coeffs_sel(residual);
    quant_levels_and_targets(&coeffs, quant.q_mult_dc(), quant.q_mult_ac())
}

pub(crate) fn dct8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f64; 128]) {
    let coeffs = dct8x16_coeffs_sel(residual);
    quant_levels_and_targets(&coeffs, quant.q_mult_dc(), quant.q_mult_ac())
}

// ── Small / rectangular sizes without a SIMD path (scalar only) ───────────────
// Same construction as the larger cores: separable integer 1-D DCTs, canonical
// `out[horiz*H + vert]` layout (DC at index 0), then a per-size gain
// normalization to the orthonormal*8 scale (ratio = 8 / sqrt(W*H)).

/// 4x4: residual `resid[row*4+col]`. DCT-4 vertical then DCT-4 horizontal.
/// Returns native (orthonormal*sqrt(16)=*4) coefficients; the *8/sqrt(W*H) gain
/// normalization is folded into the trellis multiplier for full precision.
fn dct4x4_coeffs(input: &[i32; 16]) -> [i32; 16] {
    let mut tmp = [0i32; 16]; // tmp[fy*4 + col]
    for col in 0..4 {
        let mut c = [0i32; 4];
        for row in 0..4 {
            c[row] = input[row * 4 + col];
        }
        dct1d_4_i32(&mut c);
        for fy in 0..4 {
            tmp[fy * 4 + col] = c[fy];
        }
    }
    let mut out = [0i32; 16];
    for fy in 0..4 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            out[fx * 4 + fy] = r[fx];
        }
    }
    out
}

/// 4x8: residual `resid[row*4+col]` (8 tall x 4 wide). DCT-8 vertical, DCT-4
/// horizontal. Native (orthonormal*sqrt(32)) coefficients.
fn dct4x8_coeffs(input: &[i32; 32]) -> [i32; 32] {
    let mut tmp = [0i32; 32]; // tmp[fy*4 + col], fy in 0..8
    for col in 0..4 {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 4 + col];
        }
        dct1d_8_i32(&mut c);
        for fy in 0..8 {
            tmp[fy * 4 + col] = c[fy];
        }
    }
    let mut out = [0i32; 32];
    for fy in 0..8 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            out[fx * 8 + fy] = r[fx];
        }
    }
    out
}

/// 16x32: residual `resid[row*16+col]` (32 tall x 16 wide). DCT-32 vertical,
/// DCT-16 horizontal. Native (orthonormal*sqrt(512)) coefficients.
fn dct16x32_coeffs(input: &[i32; 512]) -> [i32; 512] {
    let mut tmp = [0i32; 512]; // tmp[fy*16 + col], fy in 0..32
    for col in 0..16 {
        let mut c = [0i32; 32];
        for row in 0..32 {
            c[row] = input[row * 16 + col];
        }
        dct1d_32_i32(&mut c);
        for fy in 0..32 {
            tmp[fy * 16 + col] = c[fy];
        }
    }
    let mut out = [0i32; 512];
    for fy in 0..32 {
        let mut r = [0i32; 16];
        for col in 0..16 {
            r[col] = tmp[fy * 16 + col];
        }
        dct1d_16_i32(&mut r);
        for fx in 0..16 {
            out[fx * 32 + fy] = r[fx];
        }
    }
    out
}

// Per-size gain ratios `8 / sqrt(W*H)` in Q0.16, folded into the quant
// multiplier so the native coefficients keep full precision through one rounding:
//   m = mul_q16(q_mult, ratio_q16);  level = mul_q16(coeff_native, m).
const RATIO_4X4_Q16: i32 = 131072; // 8/sqrt(16)  = 2
const RATIO_4X8_Q16: i32 = 92682; //  8/sqrt(32)  = sqrt(2)
const RATIO_16X32_Q16: i32 = 23170; // 8/sqrt(512) = 1/(2 sqrt2)

pub(crate) fn dct4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f64; 16]) {
    let coeffs = dct4x4_coeffs(residual);
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    quant_levels_and_targets(&coeffs, m_dc, m_ac)
}

pub(crate) fn dct4x8_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f64; 32]) {
    let coeffs = dct4x8_coeffs(residual);
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    quant_levels_and_targets(&coeffs, m_dc, m_ac)
}

pub(crate) fn dct16x32_t(residual: &[i32; 512], quant: &impl Dct) -> ([i32; 512], [f64; 512]) {
    let coeffs = dct16x32_coeffs(residual);
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_16X32_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_16X32_Q16);
    quant_levels_and_targets(&coeffs, m_dc, m_ac)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::av1real::Quant;

    fn pat(n: usize) -> Vec<i32> {
        (0..n).map(|i| ((i * 41 + 7) % 113) as i32 - 56).collect()
    }

    /// The trellis `_t` variants must emit levels bit-identical to the in-place
    /// integer DCT (so RDOQ refines exactly the levels the direct path codes).
    #[test]
    fn trellis_t_matches_inplace_levels() {
        let q = Quant::new(48, 8);
        let r: [i32; 64] = pat(64).try_into().unwrap();
        let mut inp = r;
        dct8x8(&mut inp, &q);
        assert_eq!(dct8x8_t(&r, &q).0, inp, "8x8");
        let r: [i32; 256] = pat(256).try_into().unwrap();
        let mut inp = r;
        dct16x16(&mut inp, &q);
        assert_eq!(dct16x16_t(&r, &q).0, inp, "16x16");
        let r: [i32; 1024] = pat(1024).try_into().unwrap();
        let mut inp = r;
        dct32x32(&mut inp, &q);
        assert_eq!(dct32x32_t(&r, &q).0.to_vec(), inp.to_vec(), "32x32");
        let r: [i32; 128] = pat(128).try_into().unwrap();
        let mut inp = r;
        dct8x16_i32(&mut inp, &q);
        assert_eq!(dct8x16_t(&r, &q).0, inp, "8x16");
    }

    // ── coefficient verifiers ─────────────────────────────────────────────────
    // Ground truth: 1/(2·cos((2k+1)·π/(2N))) computed in f64, rounded to Q0.16.

    fn q16(v: f64) -> i32 {
        (v * 65536.0_f64).round() as i32
    }

    fn wc(k: usize, n: usize) -> i32 {
        let angle = (2 * k + 1) as f64 * std::f64::consts::PI / (2 * n) as f64;
        q16(1.0 / (2.0 * angle.cos()))
    }

    #[test]
    fn verify_wc4_coefficients() {
        assert_eq!(WC4_0, wc(0, 4), "WC4_0");
        assert_eq!(WC4_1, wc(1, 4), "WC4_1");
    }

    #[test]
    fn verify_wc8_coefficients() {
        for k in 0..4 {
            let e = wc(k, 8);
            let a = [WC8_0, WC8_1, WC8_2, WC8_3][k];
            assert_eq!(e, a, "WC8[{k}]: expected {e} got {a}");
        }
    }

    #[test]
    fn verify_wc16_coefficients() {
        // WC16_4 must be 51653, not 51652:
        // cos(9π/32)=0.6343932842, 1/(2*that)=0.7881546235, *65536=51652.501 -> rounds to 51653
        let actual = [
            WC16_0, WC16_1, WC16_2, WC16_3, WC16_4, WC16_5, WC16_6, WC16_7,
        ];
        for k in 0..8 {
            let e = wc(k, 16);
            assert_eq!(e, actual[k], "WC16[{k}]: expected {e} got {}", actual[k]);
        }
    }

    #[test]
    fn verify_wc32_coefficients() {
        for k in 0..16 {
            let e = wc(k, 32);
            assert_eq!(e, WC32[k], "WC32[{k}]: expected {e} got {}", WC32[k]);
        }
    }

    #[test]
    fn verify_sqrt2() {
        let e = q16(std::f64::consts::SQRT_2);
        assert_eq!(SQRT2, e, "SQRT2: expected {e} got {SQRT2}");
    }

    // ── float Loeffler reference ──────────────────────────────────────────────
    //
    // IMPORTANT: the Loeffler factorization is a fast DCT-II but its output
    // has bin-specific implicit scale factors — it is NOT the same as a plain
    // unnormalized DCT-II. The fixed-point functions must be tested against
    // a float Loeffler reference, not against a naive sum-of-cosines DCT.

    fn fmla_f64(a: f64, b: f64) -> f64 {
        a * std::f64::consts::SQRT_2 + b
    }

    fn dct1d_2_f64(buf: &mut [f64]) {
        let (a, b) = (buf[0], buf[1]);
        buf[0] = a + b;
        buf[1] = a - b;
    }

    fn dct1d_4_f64(buf: &mut [f64]) {
        const WC4_F: [f64; 2] = [0.541_196_100_146_197, 1.306_562_964_876_376];
        let mut tmp = [0f64; 4];
        tmp[0] = buf[0] + buf[3];
        tmp[1] = buf[1] + buf[2];
        dct1d_2_f64(&mut tmp[0..2]);
        tmp[2] = (buf[0] - buf[3]) * WC4_F[0];
        tmp[3] = (buf[1] - buf[2]) * WC4_F[1];
        dct1d_2_f64(&mut tmp[2..4]);
        tmp[2] = fmla_f64(tmp[2], tmp[3]);
        buf[0] = tmp[0];
        buf[2] = tmp[1];
        buf[1] = tmp[2];
        buf[3] = tmp[3];
    }

    fn dct1d_8_f64(buf: &mut [f64]) {
        const WC8_F: [f64; 4] = [
            0.509_795_579_104_159,
            0.601_344_886_935_045,
            0.899_976_223_136_416,
            2.562_915_447_741_505,
        ];
        let mut tmp = [0f64; 8];
        for i in 0..4 {
            tmp[i] = buf[i] + buf[7 - i];
        }
        dct1d_4_f64(&mut tmp[0..4]);
        for i in 0..4 {
            tmp[4 + i] = (buf[i] - buf[7 - i]) * WC8_F[i];
        }
        dct1d_4_f64(&mut tmp[4..8]);
        tmp[4] = fmla_f64(tmp[4], tmp[5]);
        tmp[5] += tmp[6];
        tmp[6] += tmp[7];
        for i in 0..4 {
            buf[2 * i] = tmp[i];
            buf[2 * i + 1] = tmp[4 + i];
        }
    }

    fn dct1d_16_f64(buf: &mut [f64]) {
        const WC16_F: [f64; 8] = [
            0.502_419_286_188_155,
            0.522_498_614_939_688,
            0.566_944_034_816_358,
            0.646_821_783_359_990,
            0.788_154_623_451_250,
            1.060_677_685_990_347,
            1.722_447_098_238_488,
            5.101_148_618_689_155,
        ];
        let mut tmp = [0f64; 16];
        for i in 0..8 {
            tmp[i] = buf[i] + buf[15 - i];
            tmp[8 + i] = buf[i] - buf[15 - i];
        }
        dct1d_8_f64(&mut tmp[0..8]);
        for i in 0..8 {
            tmp[8 + i] *= WC16_F[i];
        }
        dct1d_8_f64(&mut tmp[8..16]);
        tmp[8] = fmla_f64(tmp[8], tmp[9]);
        tmp[9] += tmp[10];
        tmp[10] += tmp[11];
        tmp[11] += tmp[12];
        tmp[12] += tmp[13];
        tmp[13] += tmp[14];
        tmp[14] += tmp[15];
        for i in 0..8 {
            buf[2 * i] = tmp[i];
            buf[2 * i + 1] = tmp[8 + i];
        }
    }

    fn dct1d_32_f64(buf: &mut [f64]) {
        let wc32_f: [f64; 16] = std::array::from_fn(|k| {
            let angle = (2 * k + 1) as f64 * std::f64::consts::PI / 64.0;
            1.0 / (2.0 * angle.cos())
        });
        let mut tmp = [0f64; 32];
        for i in 0..16 {
            tmp[i] = buf[i] + buf[31 - i];
            tmp[16 + i] = buf[i] - buf[31 - i];
        }
        dct1d_16_f64(&mut tmp[0..16]);
        for i in 0..16 {
            tmp[16 + i] *= wc32_f[i];
        }
        dct1d_16_f64(&mut tmp[16..32]);
        tmp[16] = fmla_f64(tmp[16], tmp[17]);
        tmp[17] += tmp[18];
        tmp[18] += tmp[19];
        tmp[19] += tmp[20];
        tmp[20] += tmp[21];
        tmp[21] += tmp[22];
        tmp[22] += tmp[23];
        tmp[23] += tmp[24];
        tmp[24] += tmp[25];
        tmp[25] += tmp[26];
        tmp[26] += tmp[27];
        tmp[27] += tmp[28];
        tmp[28] += tmp[29];
        tmp[29] += tmp[30];
        tmp[30] += tmp[31];
        for i in 0..16 {
            buf[2 * i] = tmp[i];
            buf[2 * i + 1] = tmp[16 + i];
        }
    }

    // Scale used throughout: small enough to give headroom through WC32[15] ~10x.
    const SCALE: f64 = 128.0;

    // ── dct1d_8 ───────────────────────────────────────────────────────────────

    #[test]
    fn test_dct1d_8_zero() {
        let mut buf = [0i32; 8];
        dct1d_8_i32(&mut buf);
        assert_eq!(buf, [0i32; 8]);
    }

    #[test]
    fn test_dct1d_8_dc_only() {
        let mut buf = [256i32; 8];
        dct1d_8_i32(&mut buf);
        assert!(buf[0].abs() > 0, "DC must be nonzero");
        for (i, &v) in buf[1..].iter().enumerate() {
            assert!(v.abs() <= 2, "AC bin {i} should be ~0, got {v}");
        }
    }

    #[test]
    fn test_dct1d_8_matches_reference() {
        let signal: Vec<f64> = (0..8).map(|i| (i as f64 * 0.7 + 0.3).sin()).collect();
        let fixed: Vec<i32> = signal.iter().map(|&v| (v * SCALE).round() as i32).collect();

        // Reference: float Loeffler on the same integer-valued input
        let mut ref_buf: Vec<f64> = fixed.iter().map(|&v| v as f64).collect();
        dct1d_8_f64(&mut ref_buf);

        let mut got: [i32; 8] = fixed.as_slice().try_into().unwrap();
        dct1d_8_i32(&mut got);

        for k in 0..8 {
            let exp = ref_buf[k].round() as i32;
            assert!(
                (got[k] - exp).abs() <= 5,
                "dct1d_8 bin {k}: got {} expected {exp} (diff {})",
                got[k],
                (got[k] - exp).abs()
            );
        }
    }

    #[test]
    fn test_dct1d_8_impulse_cosine_envelope() {
        // DCT-II of x[0]=A, rest 0 → X[k] = A·cos(k·π/(2N))
        // Verify by running the float reference on the same input.
        let a = 512i32;
        let mut ref_buf = [0f64; 8];
        ref_buf[0] = a as f64;
        dct1d_8_f64(&mut ref_buf);

        let mut got = [0i32; 8];
        got[0] = a;
        dct1d_8_i32(&mut got);

        for k in 0..8 {
            let exp = ref_buf[k].round() as i32;
            assert!(
                (got[k] - exp).abs() <= 4,
                "impulse bin {k}: got {} expected {exp} (diff {})",
                got[k],
                (got[k] - exp).abs()
            );
        }
    }

    // ── dct1d_16 ──────────────────────────────────────────────────────────────

    #[test]
    fn test_dct1d_16_zero() {
        let mut buf = [0i32; 16];
        dct1d_16_i32(&mut buf);
        assert_eq!(buf, [0i32; 16]);
    }

    #[test]
    fn test_dct1d_16_dc_only() {
        let mut buf = [128i32; 16];
        dct1d_16_i32(&mut buf);
        assert!(buf[0].abs() > 0, "DC must be nonzero");
        for (i, &v) in buf[1..].iter().enumerate() {
            assert!(v.abs() <= 4, "AC bin {i} should be ~0, got {v}");
        }
    }

    #[test]
    fn test_dct1d_16_matches_reference() {
        let signal: Vec<f64> = (0..16).map(|i| (i as f64 * 0.4 + 1.0).cos()).collect();
        let fixed: Vec<i32> = signal.iter().map(|&v| (v * SCALE).round() as i32).collect();

        let mut ref_buf: Vec<f64> = fixed.iter().map(|&v| v as f64).collect();
        dct1d_16_f64(&mut ref_buf);

        let mut got: [i32; 16] = fixed.as_slice().try_into().unwrap();
        dct1d_16_i32(&mut got);

        for k in 0..16 {
            let exp = ref_buf[k].round() as i32;
            assert!(
                (got[k] - exp).abs() <= 13,
                "dct1d_16 bin {k}: got {} expected {exp} (diff {})",
                got[k],
                (got[k] - exp).abs()
            );
        }
    }

    #[test]
    fn test_dct1d_32_zero() {
        let mut buf = [0i32; 32];
        dct1d_32_i32(&mut buf);
        assert_eq!(buf, [0i32; 32]);
    }

    #[test]
    fn test_dct1d_32_dc_only() {
        let mut buf = [32i32; 32];
        dct1d_32_i32(&mut buf);
        assert!(buf[0].abs() > 0, "DC must be nonzero");
        for (i, &v) in buf[1..].iter().enumerate() {
            assert!(v.abs() <= 8, "AC bin {i} should be ~0, got {v}");
        }
    }

    #[test]
    fn test_dct1d_32_matches_reference() {
        let signal: Vec<f64> = (0..32).map(|i| (i as f64 * 0.3).sin()).collect();
        let fixed: Vec<i32> = signal.iter().map(|&v| (v * SCALE).round() as i32).collect();

        let mut ref_buf: Vec<f64> = fixed.iter().map(|&v| v as f64).collect();
        dct1d_32_f64(&mut ref_buf);

        let mut got: [i32; 32] = fixed.as_slice().try_into().unwrap();
        dct1d_32_i32(&mut got);

        for k in 0..32 {
            let exp = ref_buf[k].round() as i32;
            assert!(
                (got[k] - exp).abs() <= 32,
                "dct1d_32 bin {k}: got {} expected {exp} (diff {})",
                got[k],
                (got[k] - exp).abs()
            );
        }
    }
}
