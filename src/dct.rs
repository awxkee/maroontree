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
#![allow(clippy::needless_range_loop)]
use crate::quant::Dct;
use crate::util::FastRound;
use std::sync::OnceLock;

pub(crate) const WC4_0: i32 = 35468; // 0.541196  * 65536
pub(crate) const WC4_1: i32 = 85627; // 1.306563  * 65536
pub(crate) const WC8_0: i32 = 33410; // k=0: 0.5097956
pub(crate) const WC8_1: i32 = 39410; // k=1: 0.6013449
pub(crate) const WC8_2: i32 = 58981; // k=2: 0.8999762
pub(crate) const WC8_3: i32 = 167963; // k=3: 2.5629154
pub(crate) const SQRT2: i32 = 92682; // 1.414214  * 65536

/// Multiply a Q15 data value by a Q0.16 coefficient, returning Q15.
/// data is ~15-bit signed, coeff is Q0.16 (up to ~2.6 * 65536 = 170394).
/// Worst case: 32767 * 170394 = ~5.6e9, fits in i64, shift back by 16.
#[inline(always)]
fn mul_q16(data: i32, coeff: i32) -> i32 {
    (((data as i64) * (coeff as i64)) >> 16) as i32
}

/// Quantize a transform coefficient: `round(data * coeff / 65536)` with
/// magnitude-symmetric round-to-nearest. Unlike a bare `>> 16` (truncation
/// toward -inf), this keeps the reconstruction error zero-mean, so per-
/// coefficient error does not accumulate at the block's top-left corner (where
/// every DCT basis function is positive and in phase) into a dark dot.
#[inline(always)]
fn quant_q16(data: i32, coeff: i32) -> i32 {
    let prod = (data as i64) * (coeff as i64);
    let mag = prod.unsigned_abs();
    if mag < 65536 {
        return 0;
    } // dead-zone: |coeff/step| < 1.0 -> zero (preserves compression)
    let lvl = ((mag + 32768) >> 16) as i32; // round-to-nearest for kept coefficients (de-biases the corner)
    if prod >= 0 { lvl } else { -lvl }
}

/// Apply AV1's forward quantization-matrix reciprocal to already normalized
/// transform targets. Level 15 is flat, so the normal SIMD path remains
/// untouched unless matrices are enabled.
#[inline]
fn apply_qmatrix<const N: usize>(
    mut quantized: ([i32; N], [f32; N]),
    quant: &impl Dct,
    w: usize,
    h: usize,
) -> ([i32; N], [f32; N]) {
    if !quant.has_qmatrix() {
        return quantized;
    }
    for rc in 0..N {
        let target = quantized.1[rc] * quant.forward_qm_weight(rc, w, h) as f32 / 32.0;
        quantized.1[rc] = target;
        quantized.0[rc] = if target.abs() < 1.0 {
            0
        } else {
            target.fast_round() as i32
        };
    }
    quantized
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

/// Shared 2-D integer DCT-8 transform core (no quantization). Output layout is
/// `out[r*8 + u] = F[row_freq=r][col_freq=u]`, DC at index 0. This is the single
/// forward transform reused by both the in-place quantizer ([`dct8x8_scalar`])
/// and the trellis/RDOQ variant ([`dct8x8_t`]).
#[inline]
pub(crate) fn dct8x8_coeffs(input: &[i32; 64]) -> [i32; 64] {
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
fn dct8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
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

    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for r in 0..8usize {
        let mut row: [i32; 8] = tmp[r * 8..r * 8 + 8].try_into().unwrap();
        dct1d_8_i32(&mut row);
        for u in 0..8 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 8 + r, row[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

/// Forward ADST8 (1-D), Q12 matrix derived as the gain-matched transpose of
/// dav1d's integer inverse ADST8 (`inv_adst8_1d`). Calibrated so the forward/
/// inverse round-trip gain equals the DCT pair's, so the same quant multipliers
/// and TX_8X8 inverse shifts apply unchanged. Used only as the encoder's
/// analysis transform; reconstruction is dav1d-exact via `iadst_dequant_8x8`.
static ADST8_FWD_Q12: [[i32; 8]; 8] = [
    [567, 1682, 2730, 3675, 4477, 5109, 5544, 5766],
    [1682, 4480, 5766, 5109, 2732, -569, -3675, -5545],
    [2732, 5766, 3675, -1682, -5544, -4479, 569, 5109],
    [3675, 5108, -1681, -5764, -569, 5542, 2732, -4479],
    [4479, 2732, -5542, -569, 5764, -1681, -5108, 3675],
    [5109, -569, -4479, 5544, -1682, -3675, 5766, -2732],
    [5545, -3675, 569, 2732, -5109, 5766, -4480, 1682],
    [5766, -5544, 5109, -4477, 3675, -2730, 1682, -567],
];

#[inline]
fn fwd_adst8_1d(inp: &[i32; 8]) -> [i32; 8] {
    let mut out = [0i32; 8];
    for i in 0..8 {
        let mut acc = 0i64;
        for j in 0..8 {
            acc += ADST8_FWD_Q12[i][j] as i64 * inp[j] as i64;
        }
        out[i] = ((acc + 2048) >> 12) as i32;
    }
    out
}

/// Trellis (RDOQ) forward ADST8: levels + unrounded targets, mirroring
/// `dct8x8_t`. Ratio is 1.0 for 8x8, so the quant multipliers apply directly.
pub(crate) fn adst8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix(
        adst8x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        8,
        8,
    )
}

pub(crate) fn adstdct8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix(
        adstdct8x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        8,
        8,
    )
}

pub(crate) fn dctadst8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix(
        dctadst8x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        8,
        8,
    )
}

pub(crate) static ADST16_FWD_Q12: [[i16; 16]; 16] = [
    [
        284, 850, 1408, 1951, 2477, 2978, 3451, 3891, 4294, 4653, 4969, 5236, 5455, 5619, 5731,
        5788,
    ],
    [
        850, 2479, 3892, 4970, 5619, 5788, 5457, 4654, 3452, 1951, 284, -1409, -2978, -4294, -5239,
        -5732,
    ],
    [
        1408, 3892, 5455, 5731, 4653, 2477, -284, -2979, -4970, -5788, -5237, -3451, -850, 1952,
        4294, 5621,
    ],
    [
        1952, 4970, 5731, 3891, 286, -3451, -5621, -5239, -2477, 1408, 4654, 5786, 4294, 852,
        -2979, -5457,
    ],
    [
        2477, 5621, 4654, 284, -4294, -5731, -2979, 1952, 5455, 4970, 852, -3891, -5788, -3451,
        1408, 5239,
    ],
    [
        2979, 5788, 2477, -3452, -5731, -1951, 3891, 5621, 1408, -4294, -5454, -850, 4654, 5237,
        284, -4970,
    ],
    [
        3452, 5457, -284, -5621, -2979, 3891, 5239, -850, -5731, -2477, 4294, 4970, -1408, -5788,
        -1952, 4654,
    ],
    [
        3892, 4654, -2979, -5239, 1952, 5621, -850, -5788, -284, 5731, 1408, -5455, -2477, 4970,
        3452, -4294,
    ],
    [
        4294, 3452, -4970, -2477, 5455, 1408, -5731, -284, 5788, -850, -5621, 1952, 5239, -2979,
        -4654, 3892,
    ],
    [
        4654, 1952, -5788, 1408, 4970, -4294, -2477, 5731, -850, -5239, 3891, 2979, -5621, 284,
        5457, -3452,
    ],
    [
        4970, 284, -5237, 4654, 850, -5454, 4294, 1408, -5621, 3891, 1951, -5731, 3452, 2477,
        -5788, 2979,
    ],
    [
        5239, -1408, -3451, 5788, -3891, -852, 4970, -5455, 1952, 2979, -5731, 4294, 284, -4654,
        5621, -2477,
    ],
    [
        5457, -2979, -852, 4294, -5786, 4654, -1408, -2477, 5239, -5621, 3451, 286, -3891, 5731,
        -4970, 1952,
    ],
    [
        5621, -4294, 1952, 850, -3451, 5237, -5788, 4970, -2979, 284, 2477, -4653, 5731, -5455,
        3892, -1408,
    ],
    [
        5732, -5239, 4294, -2978, 1409, 284, -1951, 3452, -4654, 5457, -5788, 5619, -4970, 3892,
        -2479, 850,
    ],
    [
        5788, -5731, 5619, -5455, 5236, -4969, 4653, -4294, 3891, -3451, 2978, -2477, 1951, -1408,
        850, -284,
    ],
];

#[inline]
fn fwd_adst16_1d(inp: &[i32; 16]) -> [i32; 16] {
    let mut out = [0i32; 16];
    for i in 0..16 {
        let mut acc = 0i64;
        for j in 0..16 {
            acc += ADST16_FWD_Q12[i][j] as i64 * inp[j] as i64;
        }
        out[i] = ((acc + 2048) >> 12) as i32;
    }
    out
}

pub(crate) fn adst16x16_t(residual: &[i32; 256], quant: &impl Dct) -> ([i32; 256], [f32; 256]) {
    apply_qmatrix(
        resolve_adst16x16_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        16,
        16,
    )
}

pub(crate) fn adstdct16x16_t(residual: &[i32; 256], quant: &impl Dct) -> ([i32; 256], [f32; 256]) {
    apply_qmatrix(
        resolve_adstdct16x16_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        16,
        16,
    )
}

pub(crate) fn dctadst16x16_t(residual: &[i32; 256], quant: &impl Dct) -> ([i32; 256], [f32; 256]) {
    apply_qmatrix(
        resolve_dctadst16x16_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        16,
        16,
    )
}

#[inline]
#[allow(unused)]
pub(crate) fn dct8x8_scalar(input: &mut [i32; 64], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x8_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = quant_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

type Dct16x16Fn = fn(&mut [i32; 256], i32, i32);
static DCT16X16: OnceLock<Dct16x16Fn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct16x16_neon_i32_wrap(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    unsafe { crate::neon::dct16x16_neon_i32(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct16x16_avx2_i32_wrap(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    unsafe { crate::avx::dct16x16_avx2_i32(input, dc_q, ac_q) }
}

#[inline]
fn resolve_dct16x16() -> Dct16x16Fn {
    *DCT16X16.get_or_init(|| {
        let mut _f: Dct16x16Fn = dct16x16_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct16x16_neon_i32_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct16x16_avx2_i32_wrap;
            }
        }
        _f
    })
}

#[inline]
pub(crate) fn dct16x16(input: &mut [i32; 256], quant: &impl Dct) {
    resolve_dct16x16()(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer DCT-16 transform core (no quantization). Output layout
/// `out[u*16 + v]`, DC at index 0. Reused by [`dct16x16_scalar`] and [`dct16x16_t`].
#[inline]
#[allow(unused)]
pub(crate) fn dct16x16_coeffs(input: &[i32; 256]) -> [i32; 256] {
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

#[inline]
fn dct16x16_quant_t_direct(input: &[i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [f32; 256]) {
    let mut tmp = [0i32; 256];
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

    let mut cf = [0i32; 256];
    let mut tf = [0.0f32; 256];
    for v in 0..16 {
        let mut row: [i32; 16] = tmp[v * 16..v * 16 + 16].try_into().unwrap();
        dct1d_16_i32(&mut row);
        for u in 0..16 {
            let coeff = mul_q16(row[u], 32768);
            store_quant_target_scalar(&mut cf, &mut tf, u * 16 + v, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[allow(unused)]
pub(crate) fn dct16x16_scalar(input: &mut [i32; 256], dc_q: i32, ac_q: i32) {
    let coeffs = dct16x16_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = quant_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

pub(crate) const WC32: [i32; 16] = [
    32808, 33127, 33780, 34802, 36248, 38203, 40796, 44224, 48794, 55008, 63738, 76640, 97266,
    134859, 223321, 667812,
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

type Dct32x32Fn = fn(&mut [i32; 1024], i32, i32);
static DCT32X32: OnceLock<Dct32x32Fn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct32x32_neon_i32_wrap(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    unsafe { crate::neon::dct32x32_neon_i32(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct32x32_avx2_i32_wrap(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    unsafe { crate::avx::dct32x32_avx2_i32(input, dc_q, ac_q) }
}

#[inline]
fn resolve_dct32x32() -> Dct32x32Fn {
    *DCT32X32.get_or_init(|| {
        let mut _f: Dct32x32Fn = dct32x32_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct32x32_neon_i32_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct32x32_avx2_i32_wrap;
            }
        }
        _f
    })
}

#[inline]
#[allow(unused)]
pub(crate) fn dct32x32(input: &mut [i32; 1024], quant: &impl Dct) {
    resolve_dct32x32()(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer DCT-32 transform core (no quantization). Output layout
/// `out[u*32 + v]`, DC at index 0. Reused by [`dct32x32_scalar`] and [`dct32x32_t`].
#[inline]
pub(crate) fn dct32x32_coeffs(input: &[i32; 1024]) -> [i32; 1024] {
    const B: i32 = 6;
    let mut tmp = [0i32; 1024];
    // Column-wise 1D DCT (on the B-bit-headroom residual)
    for u in 0..32 {
        let mut col = [0i32; 32];
        for i in 0..32 {
            col[i] = input[i * 32 + u] << B;
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
        for u in 0..32 {
            let prod = (row[u] as i64) * 16384;
            out[u * 32 + v] = ((prod + (1i64 << (15 + B))) >> (16 + B)) as i32;
        }
    }
    out
}

#[inline]
fn dct32x32_quant_t_direct(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    const B: i32 = 6;
    let mut tmp = [0i32; 1024];
    for u in 0..32 {
        let mut col = [0i32; 32];
        for i in 0..32 {
            col[i] = input[i * 32 + u] << B;
        }
        dct1d_32_i32(&mut col);
        for v in 0..32 {
            tmp[v * 32 + u] = col[v];
        }
    }

    let mut cf = [0i32; 1024];
    let mut tf = [0.0f32; 1024];
    for v in 0..32 {
        let mut row: [i32; 32] = tmp[v * 32..v * 32 + 32].try_into().unwrap();
        dct1d_32_i32(&mut row);
        for u in 0..32 {
            let prod = (row[u] as i64) * 16384;
            let coeff = ((prod + (1i64 << (15 + B))) >> (16 + B)) as i32;
            store_quant_target_scalar(&mut cf, &mut tf, u * 32 + v, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[allow(unused)]
pub(crate) fn dct32x32_scalar(input: &mut [i32; 1024], dc_q: i32, ac_q: i32) {
    let coeffs = dct32x32_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = quant_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

type Dct8x16Fn = fn(&mut [i32; 128], i32, i32);
static DCT8X16: OnceLock<Dct8x16Fn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct8x16_neon_i32_wrap(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    unsafe { crate::neon::dct8x16_neon_i32(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct8x16_avx2_i32_wrap(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    unsafe { crate::avx::dct8x16_avx2_i32(input, dc_q, ac_q) }
}

#[inline]
fn resolve_dct8x16() -> Dct8x16Fn {
    *DCT8X16.get_or_init(|| {
        let mut _f: Dct8x16Fn = dct8x16_i32_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct8x16_neon_i32_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct8x16_avx2_i32_wrap;
            }
        }
        _f
    })
}

#[allow(unused)]
pub(crate) fn dct8x16_i32(input: &mut [i32; 128], quant: &impl Dct) {
    resolve_dct8x16()(input, quant.q_mult_dc(), quant.q_mult_ac());
}

/// Shared 2-D integer 8x16 transform core (8 wide x 16 tall; residual read as
/// `resid[row*8 + col]`). DCT-16 down each of the 8 columns, then DCT-8 across
/// each of the 16 rows. Output layout `out[horiz_freq*16 + vert_freq]`, DC at
/// index 0 — matching the float reference, the scan order and the inverse.
/// Reused by [`dct8x16_i32_scalar`] and [`dct8x16_t`].
#[inline]
pub(crate) fn dct8x16_coeffs(input: &[i32; 128]) -> [i32; 128] {
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

#[inline]
fn dct8x16_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
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

    let mut cf = [0i32; 128];
    let mut tf = [0.0f32; 128];
    for fy in 0..16usize {
        let mut r: [i32; 8] = tmp[fy * 8..fy * 8 + 8].try_into().unwrap();
        dct1d_8_i32(&mut r);
        for fx in 0..8 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 16 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[allow(unused)]
pub(crate) fn dct8x16_i32_scalar(input: &mut [i32; 128], dc_q: i32, ac_q: i32) {
    let coeffs = dct8x16_coeffs(input);
    for (i, dst) in input.iter_mut().enumerate() {
        *dst = quant_q16(coeffs[i], if i == 0 { dc_q } else { ac_q });
    }
}

#[inline(always)]
fn store_quant_target_scalar<const N: usize>(
    cf: &mut [i32; N],
    tf: &mut [f32; N],
    idx: usize,
    coeff: i32,
    q_mult_dc: i32,
    q_mult_ac: i32,
) {
    let m = if idx == 0 { q_mult_dc } else { q_mult_ac };
    cf[idx] = quant_q16(coeff, m);
    tf[idx] = coeff as f32 * m as f32 * (1.0 / 65536.0);
}

#[inline]
#[allow(unused)]
fn quant_levels_and_targets<const N: usize>(
    coeffs: &[i32; N],
    q_mult_dc: i32,
    q_mult_ac: i32,
) -> ([i32; N], [f32; N]) {
    let mut cf = [0i32; N];
    let mut tf = [0.0f32; N];
    for (i, &coeff) in coeffs.iter().enumerate() {
        store_quant_target_scalar(&mut cf, &mut tf, i, coeff, q_mult_dc, q_mult_ac);
    }
    (cf, tf)
}

#[inline]
fn adst8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
    for x in 0..8usize {
        let mut col = [0i32; 8];
        for r in 0..8 {
            col[r] = input[r * 8 + x];
        }
        let c = fwd_adst8_1d(&col);
        for r in 0..8 {
            tmp[r * 8 + x] = c[r];
        }
    }
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for r in 0..8usize {
        let row: [i32; 8] = tmp[r * 8..r * 8 + 8].try_into().unwrap();
        let rr = fwd_adst8_1d(&row);
        for u in 0..8 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 8 + r, rr[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn adstdct8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
    for x in 0..8usize {
        let mut col = [0i32; 8];
        for r in 0..8 {
            col[r] = input[r * 8 + x];
        }
        let c = fwd_adst8_1d(&col);
        for r in 0..8 {
            tmp[r * 8 + x] = c[r];
        }
    }
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for r in 0..8usize {
        let mut row: [i32; 8] = tmp[r * 8..r * 8 + 8].try_into().unwrap();
        dct1d_8_i32(&mut row);
        for u in 0..8 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 8 + r, row[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn dctadst8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
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
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for r in 0..8usize {
        let row: [i32; 8] = tmp[r * 8..r * 8 + 8].try_into().unwrap();
        let rr = fwd_adst8_1d(&row);
        for u in 0..8 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 8 + r, rr[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn adst16x16_quant_t_direct(input: &[i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [f32; 256]) {
    let mut tmp = [0i32; 256];
    for u in 0..16 {
        let mut col = [0i32; 16];
        for i in 0..16 {
            col[i] = input[i * 16 + u];
        }
        let c = fwd_adst16_1d(&col);
        for v in 0..16 {
            tmp[v * 16 + u] = c[v];
        }
    }
    let mut cf = [0i32; 256];
    let mut tf = [0.0f32; 256];
    for v in 0..16 {
        let row: [i32; 16] = tmp[v * 16..v * 16 + 16].try_into().unwrap();
        let rr = fwd_adst16_1d(&row);
        for u in 0..16 {
            store_quant_target_scalar(
                &mut cf,
                &mut tf,
                u * 16 + v,
                mul_q16(rr[u], 32768),
                dc_q,
                ac_q,
            );
        }
    }
    (cf, tf)
}

#[inline]
fn adstdct16x16_quant_t_direct(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp = [0i32; 256];
    for u in 0..16 {
        let mut col = [0i32; 16];
        for i in 0..16 {
            col[i] = input[i * 16 + u];
        }
        let c = fwd_adst16_1d(&col);
        for v in 0..16 {
            tmp[v * 16 + u] = c[v];
        }
    }
    let mut cf = [0i32; 256];
    let mut tf = [0.0f32; 256];
    for v in 0..16 {
        let mut row: [i32; 16] = tmp[v * 16..v * 16 + 16].try_into().unwrap();
        dct1d_16_i32(&mut row);
        for u in 0..16 {
            store_quant_target_scalar(
                &mut cf,
                &mut tf,
                u * 16 + v,
                mul_q16(row[u], 32768),
                dc_q,
                ac_q,
            );
        }
    }
    (cf, tf)
}

#[inline]
fn dctadst16x16_quant_t_direct(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    let mut tmp = [0i32; 256];
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
    let mut cf = [0i32; 256];
    let mut tf = [0.0f32; 256];
    for v in 0..16 {
        let row: [i32; 16] = tmp[v * 16..v * 16 + 16].try_into().unwrap();
        let rr = fwd_adst16_1d(&row);
        for u in 0..16 {
            store_quant_target_scalar(
                &mut cf,
                &mut tf,
                u * 16 + v,
                mul_q16(rr[u], 32768),
                dc_q,
                ac_q,
            );
        }
    }
    (cf, tf)
}

#[inline]
fn adst4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    let mut tmp = [0i32; 16];
    for x in 0..4usize {
        let mut col = [0i32; 4];
        for r in 0..4 {
            col[r] = input[r * 4 + x];
        }
        let c = fwd_adst4_1d(&col);
        for r in 0..4 {
            tmp[r * 4 + x] = c[r];
        }
    }
    let mut cf = [0i32; 16];
    let mut tf = [0.0f32; 16];
    for r in 0..4usize {
        let row: [i32; 4] = tmp[r * 4..r * 4 + 4].try_into().unwrap();
        let rr = fwd_adst4_1d(&row);
        for u in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 4 + r, rr[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn adstdct4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    let mut tmp = [0i32; 16];
    for x in 0..4usize {
        let mut col = [0i32; 4];
        for r in 0..4 {
            col[r] = input[r * 4 + x];
        }
        let c = fwd_adst4_1d(&col);
        for r in 0..4 {
            tmp[r * 4 + x] = c[r];
        }
    }
    let mut cf = [0i32; 16];
    let mut tf = [0.0f32; 16];
    for r in 0..4usize {
        let mut row: [i32; 4] = tmp[r * 4..r * 4 + 4].try_into().unwrap();
        dct1d_4_i32(&mut row);
        for u in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 4 + r, row[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn dctadst4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    let mut tmp = [0i32; 16];
    for x in 0..4usize {
        let mut col = [0i32; 4];
        for r in 0..4 {
            col[r] = input[r * 4 + x];
        }
        dct1d_4_i32(&mut col);
        for r in 0..4 {
            tmp[r * 4 + x] = col[r];
        }
    }
    let mut cf = [0i32; 16];
    let mut tf = [0.0f32; 16];
    for r in 0..4usize {
        let row: [i32; 4] = tmp[r * 4..r * 4 + 4].try_into().unwrap();
        let rr = fwd_adst4_1d(&row);
        for u in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, u * 4 + r, rr[u], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn adst4x8_quant_t_direct(input: &[i32; 32], dc_q: i32, ac_q: i32) -> ([i32; 32], [f32; 32]) {
    let mut tmp = [0i32; 32];
    for col in 0..4 {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 4 + col];
        }
        let cc = fwd_adst8_1d(&c);
        for fy in 0..8 {
            tmp[fy * 4 + col] = cc[fy];
        }
    }
    let mut cf = [0i32; 32];
    let mut tf = [0.0f32; 32];
    for fy in 0..8 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        let rr = fwd_adst4_1d(&r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, rr[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn adstdct4x8_quant_t_direct(input: &[i32; 32], dc_q: i32, ac_q: i32) -> ([i32; 32], [f32; 32]) {
    let mut tmp = [0i32; 32];
    for col in 0..4 {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 4 + col];
        }
        let cc = fwd_adst8_1d(&c);
        for fy in 0..8 {
            tmp[fy * 4 + col] = cc[fy];
        }
    }
    let mut cf = [0i32; 32];
    let mut tf = [0.0f32; 32];
    for fy in 0..8 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, r[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn dctadst4x8_quant_t_direct(input: &[i32; 32], dc_q: i32, ac_q: i32) -> ([i32; 32], [f32; 32]) {
    let mut tmp = [0i32; 32];
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
    let mut cf = [0i32; 32];
    let mut tf = [0.0f32; 32];
    for fy in 0..8 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        let rr = fwd_adst4_1d(&r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, rr[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

type Dct8x8QuantTFn = fn(&[i32; 64], i32, i32) -> ([i32; 64], [f32; 64]);
type Dct8x16QuantTFn = fn(&[i32; 128], i32, i32) -> ([i32; 128], [f32; 128]);
type Dct16x16QuantTFn = fn(&[i32; 256], i32, i32) -> ([i32; 256], [f32; 256]);
type Tx16x16QuantTFn = fn(&[i32; 256], i32, i32) -> ([i32; 256], [f32; 256]);
type Dct32x32QuantTFn = fn(&[i32; 1024], i32, i32) -> ([i32; 1024], [f32; 1024]);
type Dct16x32QuantTFn = fn(&[i32; 512], i32, i32) -> ([i32; 512], [f32; 512]);
type Dct32x16QuantTFn = fn(&[i32; 512], i32, i32) -> ([i32; 512], [f32; 512]);

static DCT8X8_QUANT_T: OnceLock<Dct8x8QuantTFn> = OnceLock::new();
static DCT8X16_QUANT_T: OnceLock<Dct8x16QuantTFn> = OnceLock::new();
static DCT16X16_QUANT_T: OnceLock<Dct16x16QuantTFn> = OnceLock::new();
static ADST16X16_QUANT_T: OnceLock<Tx16x16QuantTFn> = OnceLock::new();
static ADSTDCT16X16_QUANT_T: OnceLock<Tx16x16QuantTFn> = OnceLock::new();
static DCTADST16X16_QUANT_T: OnceLock<Tx16x16QuantTFn> = OnceLock::new();
static DCT32X32_QUANT_T: OnceLock<Dct32x32QuantTFn> = OnceLock::new();
static DCT16X32_QUANT_T: OnceLock<Dct16x32QuantTFn> = OnceLock::new();
static DCT32X16_QUANT_T: OnceLock<Dct32x16QuantTFn> = OnceLock::new();

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct8x8_neon_quant_t_wrap(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    unsafe { crate::neon::dct8x8_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct16x16_neon_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::neon::dct16x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn adst16x16_neon_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::neon::adst16x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn adstdct16x16_neon_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::neon::adstdct16x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dctadst16x16_neon_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::neon::dctadst16x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct32x32_neon_quant_t_wrap(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    unsafe { crate::neon::dct32x32_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct8x16_neon_quant_t_wrap(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    unsafe { crate::neon::dct8x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct16x32_neon_quant_t_wrap(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    unsafe { crate::neon::dct16x32_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn dct32x16_neon_quant_t_wrap(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    unsafe { crate::neon::dct32x16_neon_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct8x8_avx2_quant_t_wrap(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    unsafe { crate::avx::dct8x8_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct16x16_avx2_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::avx::dct16x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn adst16x16_avx2_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::avx::adst16x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn adstdct16x16_avx2_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::avx::adstdct16x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dctadst16x16_avx2_quant_t_wrap(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    unsafe { crate::avx::dctadst16x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct32x32_avx2_quant_t_wrap(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    unsafe { crate::avx::dct32x32_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct8x16_avx2_quant_t_wrap(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    unsafe { crate::avx::dct8x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct16x32_avx2_quant_t_wrap(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    unsafe { crate::avx::dct16x32_avx2_quant_t(input, dc_q, ac_q) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn dct32x16_avx2_quant_t_wrap(
    input: &[i32; 512],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 512], [f32; 512]) {
    unsafe { crate::avx::dct32x16_avx2_quant_t(input, dc_q, ac_q) }
}

#[inline]
fn dct8x8_quant_t_scalar(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    dct8x8_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dct8x16_quant_t_scalar(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    dct8x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dct16x16_quant_t_scalar(input: &[i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [f32; 256]) {
    dct16x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn adst16x16_quant_t_scalar(input: &[i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [f32; 256]) {
    adst16x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn adstdct16x16_quant_t_scalar(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    adstdct16x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dctadst16x16_quant_t_scalar(
    input: &[i32; 256],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 256], [f32; 256]) {
    dctadst16x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dct32x32_quant_t_scalar(
    input: &[i32; 1024],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 1024], [f32; 1024]) {
    dct32x32_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dct16x32_quant_t_scalar(input: &[i32; 512], dc_q: i32, ac_q: i32) -> ([i32; 512], [f32; 512]) {
    dct16x32_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn dct32x16_quant_t_scalar(input: &[i32; 512], dc_q: i32, ac_q: i32) -> ([i32; 512], [f32; 512]) {
    dct32x16_quant_t_direct(input, dc_q, ac_q)
}

#[inline]
fn resolve_dct8x8_quant_t() -> Dct8x8QuantTFn {
    *DCT8X8_QUANT_T.get_or_init(|| {
        let mut _f: Dct8x8QuantTFn = dct8x8_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct8x8_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct8x8_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dct8x16_quant_t() -> Dct8x16QuantTFn {
    *DCT8X16_QUANT_T.get_or_init(|| {
        let mut _f: Dct8x16QuantTFn = dct8x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct8x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct8x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dct16x16_quant_t() -> Dct16x16QuantTFn {
    *DCT16X16_QUANT_T.get_or_init(|| {
        let mut _f: Dct16x16QuantTFn = dct16x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct16x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct16x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_adst16x16_quant_t() -> Tx16x16QuantTFn {
    *ADST16X16_QUANT_T.get_or_init(|| {
        let mut _f: Tx16x16QuantTFn = adst16x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = adst16x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = adst16x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_adstdct16x16_quant_t() -> Tx16x16QuantTFn {
    *ADSTDCT16X16_QUANT_T.get_or_init(|| {
        let mut _f: Tx16x16QuantTFn = adstdct16x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = adstdct16x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = adstdct16x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dctadst16x16_quant_t() -> Tx16x16QuantTFn {
    *DCTADST16X16_QUANT_T.get_or_init(|| {
        let mut _f: Tx16x16QuantTFn = dctadst16x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dctadst16x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dctadst16x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dct32x32_quant_t() -> Dct32x32QuantTFn {
    *DCT32X32_QUANT_T.get_or_init(|| {
        let mut _f: Dct32x32QuantTFn = dct32x32_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct32x32_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct32x32_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dct16x32_quant_t() -> Dct16x32QuantTFn {
    *DCT16X32_QUANT_T.get_or_init(|| {
        let mut _f: Dct16x32QuantTFn = dct16x32_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct16x32_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct16x32_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

#[inline]
fn resolve_dct32x16_quant_t() -> Dct32x16QuantTFn {
    *DCT32X16_QUANT_T.get_or_init(|| {
        let mut _f: Dct32x16QuantTFn = dct32x16_quant_t_scalar;
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            _f = dct32x16_neon_quant_t_wrap;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        {
            if std::is_x86_feature_detected!("avx2") {
                _f = dct32x16_avx2_quant_t_wrap;
            }
        }
        _f
    })
}

pub(crate) fn dct8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix(
        resolve_dct8x8_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        8,
        8,
    )
}

/// Forward TX_8X8 **IDTX** (identity): produce quantized levels + unquantized
/// targets that pair with `iidentity_dequant_8x8`. The inverse has uniform gain
/// 1/8 (dequant->residual), so the forward level at raster position `y + x*8`
/// (the inverse transposes `coeff[y + x*8]` into pixel `(y,x)`) is
/// `round(residual[y,x] * 8 / q)` with the same full-step dead-zone as
/// `quant_q16`. Bit-exactness with dav1d is carried entirely by the inverse;
/// this only decides which levels get coded.
pub(crate) fn fidentity8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    let (dc_q, ac_q) = (quant.dc_q(), quant.ac_q());
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let rc = y + x * 8;
            let qd = if rc == 0 { dc_q } else { ac_q };
            let num = residual[y * 8 + x] * 8;
            tf[rc] = num as f32 / qd as f32;
            let am = num.unsigned_abs() as i32;
            cf[rc] = if am < qd {
                0
            } else {
                let l = (am + qd / 2) / qd;
                if num < 0 { -l } else { l }
            };
        }
    }
    (cf, tf)
}

pub(crate) fn dct16x16_t(residual: &[i32; 256], quant: &impl Dct) -> ([i32; 256], [f32; 256]) {
    apply_qmatrix(
        resolve_dct16x16_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        16,
        16,
    )
}

pub(crate) fn dct32x32_t(residual: &[i32; 1024], quant: &impl Dct) -> ([i32; 1024], [f32; 1024]) {
    apply_qmatrix(
        resolve_dct32x32_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        32,
        32,
    )
}

pub(crate) fn dct8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix(
        resolve_dct8x16_quant_t()(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        8,
        16,
    )
}

#[inline]
fn dct16x8_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
    for col in 0..16usize {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 16 + col];
        }
        dct1d_8_i32(&mut c);
        for fy in 0..8 {
            tmp[fy * 16 + col] = c[fy];
        }
    }

    let mut cf = [0i32; 128];
    let mut tf = [0.0f32; 128];
    for fy in 0..8usize {
        let mut r: [i32; 16] = tmp[fy * 16..fy * 16 + 16].try_into().unwrap();
        dct1d_16_i32(&mut r);
        for fx in 0..16 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

pub(crate) fn dct16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix(
        dct16x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac()),
        quant,
        16,
        8,
    )
}

#[inline]
fn dct4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    let mut tmp = [0i32; 16];
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

    let mut cf = [0i32; 16];
    let mut tf = [0.0f32; 16];
    for fy in 0..4 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 4 + fy, r[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn dct4x8_quant_t_direct(input: &[i32; 32], dc_q: i32, ac_q: i32) -> ([i32; 32], [f32; 32]) {
    let mut tmp = [0i32; 32];
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

    let mut cf = [0i32; 32];
    let mut tf = [0.0f32; 32];
    for fy in 0..8 {
        let mut r = [0i32; 4];
        for col in 0..4 {
            r[col] = tmp[fy * 4 + col];
        }
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, r[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[inline]
fn dct16x32_quant_t_direct(input: &[i32; 512], dc_q: i32, ac_q: i32) -> ([i32; 512], [f32; 512]) {
    const B: i32 = 6;
    let mut tmp = [0i32; 512];
    for col in 0..16 {
        let mut c = [0i32; 32];
        for row in 0..32 {
            c[row] = input[row * 16 + col] << B;
        }
        dct1d_32_i32(&mut c);
        for fy in 0..32 {
            tmp[fy * 16 + col] = c[fy];
        }
    }

    let mut cf = [0i32; 512];
    let mut tf = [0.0f32; 512];
    for fy in 0..32 {
        let mut r = [0i32; 16];
        for col in 0..16 {
            r[col] = tmp[fy * 16 + col];
        }
        dct1d_16_i32(&mut r);
        for fx in 0..16 {
            let coeff = (r[fx] + (1 << (B - 1))) >> B;
            store_quant_target_scalar(&mut cf, &mut tf, fx * 32 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

// Per-size gain ratios `8 / sqrt(W*H)` in Q0.16, folded into the quant
// multiplier so the native coefficients keep full precision through one rounding:
//   m = mul_q16(q_mult, ratio_q16);  level = mul_q16(coeff_native, m).
const RATIO_4X4_Q16: i32 = 131072; // 8/sqrt(16)  = 2
const RATIO_4X8_Q16: i32 = 92682; //  8/sqrt(32)  = sqrt(2)
const RATIO_16X32_Q16: i32 = 23170; // 8/sqrt(512) = 1/(2 sqrt2)

/// Forward ADST-4 1D matrix in Q12, derived as the inverse of dav1d's exact
/// `inv_adst4_1d` and rescaled so its row norm (2.0) matches the codebase's
/// forward DCT-4 (which is likewise non-orthonormal and corrected by
/// `RATIO_4X4_Q16`). This pairing makes ADST_ADST 4x4 reconstruct at the same
/// intrinsic scale as DCT_DCT 4x4, so `RATIO_4X4_Q16` and the inverse
/// orchestration carry over unchanged. Byte-exactness is validated against
/// aomdec/dav1d.
static ADST4_FWD_Q12: [[i32; 4]; 4] = [
    [1868, 3510, 4730, 5379],
    [4730, 4730, 0, -4730],
    [5379, -1868, -4730, 3510],
    [3510, -5379, 4730, -1868],
];

#[inline]
fn fwd_adst4_1d(inp: &[i32; 4]) -> [i32; 4] {
    let mut out = [0i32; 4];
    for i in 0..4 {
        let mut acc = 0i64;
        for j in 0..4 {
            acc += ADST4_FWD_Q12[i][j] as i64 * inp[j] as i64;
        }
        out[i] = ((acc + 2048) >> 12) as i32;
    }
    out
}

/// Trellis (RDOQ) forward ADST_ADST 4x4: levels + unrounded targets, mirroring
/// `dct4x4_t` (same `RATIO_4X4_Q16` quant scaling).
pub(crate) fn adst4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix(adst4x4_quant_t_direct(residual, m_dc, m_ac), quant, 4, 4)
}

pub(crate) fn adstdct4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix(adstdct4x4_quant_t_direct(residual, m_dc, m_ac), quant, 4, 4)
}

pub(crate) fn dctadst4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix(dctadst4x4_quant_t_direct(residual, m_dc, m_ac), quant, 4, 4)
}

pub(crate) fn dct4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix(dct4x4_quant_t_direct(residual, m_dc, m_ac), quant, 4, 4)
}

#[inline]
fn dct8x4_quant_t_direct(input: &[i32; 32], dc_q: i32, ac_q: i32) -> ([i32; 32], [f32; 32]) {
    let mut tmp = [0i32; 32];
    for row in 0..4 {
        let mut c = [0i32; 8];
        for col in 0..8 {
            c[col] = input[row * 8 + col];
        }
        dct1d_8_i32(&mut c);
        for fx in 0..8 {
            tmp[fx * 4 + row] = c[fx];
        }
    }

    let mut cf = [0i32; 32];
    let mut tf = [0.0f32; 32];
    for fx in 0..8 {
        let mut r = [0i32; 4];
        for row in 0..4 {
            r[row] = tmp[fx * 4 + row];
        }
        dct1d_4_i32(&mut r);
        for fy in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 4 + fy, r[fy], dc_q, ac_q);
        }
    }
    (cf, tf)
}

pub(crate) fn dct8x4_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f32; 32]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    apply_qmatrix(dct8x4_quant_t_direct(residual, m_dc, m_ac), quant, 8, 4)
}

pub(crate) fn dct4x8_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f32; 32]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    apply_qmatrix(dct4x8_quant_t_direct(residual, m_dc, m_ac), quant, 4, 8)
}

pub(crate) fn adst4x8_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f32; 32]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    apply_qmatrix(adst4x8_quant_t_direct(residual, m_dc, m_ac), quant, 4, 8)
}

pub(crate) fn adstdct4x8_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f32; 32]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    apply_qmatrix(adstdct4x8_quant_t_direct(residual, m_dc, m_ac), quant, 4, 8)
}

pub(crate) fn dctadst4x8_t(residual: &[i32; 32], quant: &impl Dct) -> ([i32; 32], [f32; 32]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
    apply_qmatrix(dctadst4x8_quant_t_direct(residual, m_dc, m_ac), quant, 4, 8)
}

#[inline]
fn dct32x16_quant_t_direct(input: &[i32; 512], dc_q: i32, ac_q: i32) -> ([i32; 512], [f32; 512]) {
    const B: i32 = 6;
    let mut tmp = [0i32; 512];
    for row in 0..16 {
        let mut c = [0i32; 32];
        for col in 0..32 {
            c[col] = input[row * 32 + col] << B;
        }
        dct1d_32_i32(&mut c);
        for fx in 0..32 {
            tmp[fx * 16 + row] = c[fx];
        }
    }

    let mut cf = [0i32; 512];
    let mut tf = [0.0f32; 512];
    for fx in 0..32 {
        let mut r = [0i32; 16];
        for row in 0..16 {
            r[row] = tmp[fx * 16 + row];
        }
        dct1d_16_i32(&mut r);
        for fy in 0..16 {
            let coeff = (r[fy] + (1 << (B - 1))) >> B;
            store_quant_target_scalar(&mut cf, &mut tf, fx * 16 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

pub(crate) fn dct32x16_t(residual: &[i32; 512], quant: &impl Dct) -> ([i32; 512], [f32; 512]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_16X32_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_16X32_Q16);
    apply_qmatrix(
        resolve_dct32x16_quant_t()(residual, m_dc, m_ac),
        quant,
        32,
        16,
    )
}

pub(crate) fn dct16x32_t(residual: &[i32; 512], quant: &impl Dct) -> ([i32; 512], [f32; 512]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_16X32_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_16X32_Q16);
    apply_qmatrix(
        resolve_dct16x32_quant_t()(residual, m_dc, m_ac),
        quant,
        16,
        32,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Quant;

    fn pat(n: usize) -> Vec<i32> {
        (0..n).map(|i| ((i * 41 + 7) % 113) as i32 - 56).collect()
    }

    /// The 16x8 (wide) forward+inverse pair must reconstruct a residual to
    /// within transform rounding (DC and low-frequency content recovered).
    #[test]
    fn dct16x8_pairs_with_inverse() {
        use crate::idct::idct_dequant_16x8;
        let q = Quant::new(32, 8);
        // residual laid out 8 tall x 16 wide: rw[row*16 + col]
        let rw: [i32; 128] = pat(128).try_into().unwrap();
        let (lw, _) = dct16x8_t(&rw, &q);
        let rec = idct_dequant_16x8(&lw, &q);
        // The mean should be preserved closely and the reconstruction should
        // correlate strongly with the input (lossy quant, so not exact).
        let mean_in: f64 = rw.iter().map(|&v| v as f64).sum::<f64>() / 128.0;
        let mean_out: f64 = rec.iter().map(|&v| v as f64).sum::<f64>() / 128.0;
        assert!(
            (mean_in - mean_out).abs() < 2.0,
            "mean drift {mean_in} vs {mean_out}"
        );
        // Energy of the error should be far below the energy of the signal.
        let sig: f64 = rw.iter().map(|&v| (v as f64).powi(2)).sum();
        let err: f64 = rw
            .iter()
            .zip(rec.iter())
            .map(|(&a, &b)| ((a - b) as f64).powi(2))
            .sum();
        assert!(err < sig * 0.5, "error energy {err} vs signal {sig}");
    }

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

#[cfg(all(target_arch = "aarch64", feature = "neon", test))]
mod neon_consistency {
    use super::*;
    use crate::neon::dct16x16_neon_i32;
    fn next(seed: &mut u64) -> i32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as i32).rem_euclid(960) - 480
    }

    #[test]
    fn neon_matches_scalar_16x16() {
        let mut seed = 0xD1B5_4A32_D192_ED03u64;
        for &(dcq, acq) in &[(26i32, 30i32), (52, 58), (210, 240)] {
            for _ in 0..64 {
                let mut a = [0i32; 256];
                for v in a.iter_mut() {
                    *v = next(&mut seed);
                }
                let mut b = a;
                unsafe { dct16x16_neon_i32(&mut a, dcq, acq) };
                dct16x16_scalar(&mut b, dcq, acq);
                assert_eq!(a, b, "NEON 16x16 forward+quant diverges from scalar");
            }
        }
    }
}
