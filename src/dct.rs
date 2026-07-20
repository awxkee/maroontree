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
use crate::quant::{Dct, qm_row};
use crate::util::FastRound;
use std::sync::OnceLock;

#[cfg(test)]
macro_rules! neon_fwd_tx {
    ($neon:ident, $scalar:ident, $input:expr, $dc_q:expr, $ac_q:expr) => {{
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            unsafe { crate::neon::$neon($input, $dc_q, $ac_q) }
        }
        #[cfg(not(all(target_arch = "aarch64", feature = "neon")))]
        {
            $scalar($input, $dc_q, $ac_q)
        }
    }};
}

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
    // Rounding, not truncation: every butterfly stage truncated toward -inf,
    // and the bias accumulated coherently across ~5 stages x 2 passes. Only
    // the 32x32 kernels escaped it (they prescale by B=6 and normalise with
    // shr_round); 4/8/16 carried a measurable negative bias into `cf` AND the
    // `tf` the trellis prices against.
    (((data as i64) * (coeff as i64) + 32768) >> 16) as i32
}

#[inline(always)]
fn quant_q16(data: i32, coeff: i32) -> i32 {
    let prod = (data as i64) * (coeff as i64);
    let mag = prod.unsigned_abs();
    if mag < 32768 {
        return 0;
    } // dead-zone == round-to-nearest half-step; trellis (RDOQ) makes the R-D call
    let lvl = ((mag + 32768) >> 16) as i32; // round-to-nearest for kept coefficients (de-biases the corner)
    if prod >= 0 { lvl } else { -lvl }
}

#[inline(never)]
fn quant_flat(coeffs: &[i32], dc_q: i32, ac_q: i32, out: &mut [i32]) {
    debug_assert_eq!(coeffs.len(), out.len());
    let Some((&dc, ac)) = coeffs.split_first() else {
        return;
    };
    let (dc_out, ac_out) = out.split_first_mut().unwrap();
    *dc_out = quant_q16(dc, dc_q);
    for (&coeff, dst) in ac.iter().zip(ac_out.iter_mut()) {
        *dst = quant_q16(coeff, ac_q);
    }
}

const fn forward_qm_scales() -> [f32; 256] {
    let mut scales = [0.0; 256];
    let mut iwt = 1usize;
    while iwt < scales.len() {
        let weight = (1024 + iwt / 2) / iwt;
        scales[iwt] = weight as f32 / 32.0;
        iwt += 1;
    }
    scales
}

pub(crate) const FORWARD_QM_SCALES: [f32; 256] = forward_qm_scales();

/// Applies one inverse-QM row to dynamically-sized quantized coefficient
/// buffers. Keeping the hot operation slice-based lets one SIMD function serve
/// every AV1 transform size.
pub(crate) type ApplyQmatrixFn = fn(&mut [i32], &mut [f32], &[u8]);

pub(crate) fn apply_qmatrix(levels: &mut [i32], targets: &mut [f32], inverse_weights: &[u8]) {
    assert_eq!(levels.len(), targets.len());
    assert_eq!(levels.len(), inverse_weights.len());
    for ((level, target), &iwt) in levels
        .iter_mut()
        .zip(targets.iter_mut())
        .zip(inverse_weights)
    {
        let target_value = *target * FORWARD_QM_SCALES[iwt as usize];
        *target = target_value;
        // Round-to-nearest half-step dead-zone (matches `quant_q16`); the trellis
        // makes the actual R-D keep/drop decision on the surviving levels.
        *level = if target_value.abs() < 0.5 {
            0
        } else {
            target_value.fast_round() as i32
        };
    }
}

#[inline]
fn apply_qmatrix_result<const N: usize>(
    mut quantized: ([i32; N], [f32; N]),
    quant: &impl Dct,
    w: usize,
    h: usize,
    apply: ApplyQmatrixFn,
) -> ([i32; N], [f32; N]) {
    if let Some(inverse_weights) = qm_row(quant.qm_level(), quant.qm_chroma(), w, h) {
        apply(&mut quantized.0, &mut quantized.1, inverse_weights);
    }
    quantized
}

#[cfg(test)]
#[inline]
fn apply_qmatrix_result_scalar<const N: usize>(
    quantized: ([i32; N], [f32; N]),
    quant: &impl Dct,
    w: usize,
    h: usize,
) -> ([i32; N], [f32; N]) {
    apply_qmatrix_result(quantized, quant, w, h, apply_qmatrix)
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

pub(crate) static ADST8_FWD_Q12: [[i32; 8]; 8] = [
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

fn f1ddct8x8_quant_t_direct(
    residual: &[i32; 64],
    qm_dc: i32,
    qm_ac: i32,
    vertical: bool,
) -> ([i32; 64], [f32; 64]) {
    let mut f = [0i32; 64];
    if vertical {
        // V_DCT: DCT along columns, identity along rows.
        for x in 0..8 {
            let mut col = [0i32; 8];
            for (y, c) in col.iter_mut().enumerate() {
                *c = residual[y * 8 + x];
            }
            dct1d_8_i32(&mut col);
            for (y, &v) in col.iter().enumerate() {
                f[y * 8 + x] = v * 2;
            }
        }
    } else {
        // H_DCT: DCT along rows, identity along columns.
        for y in 0..8 {
            let mut row = [0i32; 8];
            row.copy_from_slice(&residual[y * 8..y * 8 + 8]);
            dct1d_8_i32(&mut row);
            for (x, &v) in row.iter().enumerate() {
                f[y * 8 + x] = v * 2;
            }
        }
    }
    let mut cf = [0i32; 64];
    let mut tf = [0f32; 64];
    for y in 0..8 {
        for x in 0..8 {
            let rc = y + x * 8; // transposed storage, matching the inverses
            let m = if rc == 0 { qm_dc } else { qm_ac };
            quant_fullstep(&mut cf, &mut tf, rc, f[y * 8 + x], m);
        }
    }
    (cf, tf)
}

/// Full-step dead-zone quantizer in Q16: zero below one whole step, else
/// round-to-nearest. The half-step `quant_q16` is for trellis-refined paths.
#[inline]
fn quant_fullstep<const N: usize>(
    cf: &mut [i32; N],
    tf: &mut [f32; N],
    idx: usize,
    coeff: i32,
    q_mult: i32,
) {
    let prod = (coeff as i64) * (q_mult as i64);
    tf[idx] = prod as f32 * (1.0 / 65536.0);
    let mag = prod.unsigned_abs();
    cf[idx] = if mag < 65536 {
        0
    } else {
        let l = ((mag + 32768) >> 16) as i32;
        if prod >= 0 { l } else { -l }
    };
}

fn f1ddct4x4_quant_t_direct(
    residual: &[i32; 16],
    qm_dc: i32,
    qm_ac: i32,
    vertical: bool,
) -> ([i32; 16], [f32; 16]) {
    let mut f = [0i32; 16];
    if vertical {
        // V_DCT: DCT along columns, identity along rows.
        for x in 0..4 {
            let mut col = [0i32; 4];
            for (y, c) in col.iter_mut().enumerate() {
                *c = residual[y * 4 + x];
            }
            dct1d_4_i32(&mut col);
            for (y, &v) in col.iter().enumerate() {
                f[y * 4 + x] = mul_q16(v * 2, SQRT2);
            }
        }
    } else {
        // H_DCT: DCT along rows, identity along columns.
        for y in 0..4 {
            let mut row = [0i32; 4];
            row.copy_from_slice(&residual[y * 4..y * 4 + 4]);
            dct1d_4_i32(&mut row);
            for (x, &v) in row.iter().enumerate() {
                f[y * 4 + x] = mul_q16(v * 2, SQRT2);
            }
        }
    }
    let mut cf = [0i32; 16];
    let mut tf = [0f32; 16];
    for y in 0..4 {
        for x in 0..4 {
            let rc = y + x * 4; // transposed storage, matching the inverses
            let m = if rc == 0 { qm_dc } else { qm_ac };
            quant_fullstep(&mut cf, &mut tf, rc, f[y * 4 + x], m);
        }
    }
    (cf, tf)
}

#[cfg(test)]
pub(crate) fn fvdct8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct8x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), true)
}

#[cfg(test)]
pub(crate) fn fhdct8x8_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct8x8_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), false)
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
    quant_flat(&coeffs, dc_q, ac_q, input);
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
    quant_flat(&coeffs, dc_q, ac_q, input);
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

fn adst8x16_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
    for col in 0..8usize {
        let mut c = [0i32; 16];
        for row in 0..16 {
            c[row] = input[row * 8 + col];
        }
        let c = fwd_adst16_1d(&c);
        for fy in 0..16 {
            tmp[fy * 8 + col] = c[fy];
        }
    }
    let mut cf = [0i32; 128];
    let mut tf = [0.0f32; 128];
    for fy in 0..16usize {
        let r: [i32; 8] = tmp[fy * 8..fy * 8 + 8].try_into().unwrap();
        let r = fwd_adst8_1d(&r);
        for fx in 0..8 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 16 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

fn adst16x8_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
    for col in 0..16usize {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 16 + col];
        }
        let c = fwd_adst8_1d(&c);
        for fy in 0..8 {
            tmp[fy * 16 + col] = c[fy];
        }
    }
    let mut cf = [0i32; 128];
    let mut tf = [0.0f32; 128];
    for fy in 0..8usize {
        let r: [i32; 16] = tmp[fy * 16..fy * 16 + 16].try_into().unwrap();
        let r = fwd_adst16_1d(&r);
        for fx in 0..16 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

fn adstdct16x8_quant_t_direct(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
    for col in 0..16usize {
        let mut c = [0i32; 8];
        for row in 0..8 {
            c[row] = input[row * 16 + col];
        }
        let c = fwd_adst8_1d(&c);
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

/// DCT_ADST for RTX_16X8 (vertical DCT-8, horizontal ADST-16).
fn dctadst16x8_quant_t_direct(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
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
        let r: [i32; 16] = tmp[fy * 16..fy * 16 + 16].try_into().unwrap();
        let r = fwd_adst16_1d(&r);
        for fx in 0..16 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 8 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

/// ADST_DCT for RTX_8X16 (vertical ADST-16, horizontal DCT-8).
fn adstdct8x16_quant_t_direct(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
    let mut tmp = [0i32; 128];
    for col in 0..8usize {
        let mut c = [0i32; 16];
        for row in 0..16 {
            c[row] = input[row * 8 + col];
        }
        let c = fwd_adst16_1d(&c);
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

fn dctadst8x16_quant_t_direct(
    input: &[i32; 128],
    dc_q: i32,
    ac_q: i32,
) -> ([i32; 128], [f32; 128]) {
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
        let r: [i32; 8] = tmp[fy * 8..fy * 8 + 8].try_into().unwrap();
        let r = fwd_adst8_1d(&r);
        for fx in 0..8 {
            let coeff = mul_q16(r[fx], 46341);
            store_quant_target_scalar(&mut cf, &mut tf, fx * 16 + fy, coeff, dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[cfg(test)]
pub(crate) fn adstdct16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            adstdct16x8_neon_quant_t,
            adstdct16x8_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        16,
        8,
    )
}

#[cfg(test)]
pub(crate) fn dctadst16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dctadst16x8_neon_quant_t,
            dctadst16x8_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        16,
        8,
    )
}

#[cfg(test)]
pub(crate) fn adstdct8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            adstdct8x16_neon_quant_t,
            adstdct8x16_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        8,
        16,
    )
}

#[cfg(test)]
pub(crate) fn dctadst8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dctadst8x16_neon_quant_t,
            dctadst8x16_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        8,
        16,
    )
}

fn f1ddct_rect_quant_t_direct(
    residual: &[i32; 128],
    qm_dc: i32,
    qm_ac: i32,
    w: usize,
    vertical: bool,
) -> ([i32; 128], [f32; 128]) {
    let h = 128 / w;
    let mut f = [0i32; 128];
    if vertical {
        // V_DCT: DCT along columns (height axis), identity along rows.
        for x in 0..w {
            if h == 8 {
                let mut col = [0i32; 8];
                for (y, c) in col.iter_mut().enumerate() {
                    *c = residual[y * w + x];
                }
                dct1d_8_i32(&mut col);
                for (y, &v) in col.iter().enumerate() {
                    f[y * w + x] = v * 2;
                }
            } else {
                let mut col = [0i32; 16];
                for (y, c) in col.iter_mut().enumerate() {
                    *c = residual[y * w + x];
                }
                dct1d_16_i32(&mut col);
                for (y, &v) in col.iter().enumerate() {
                    f[y * w + x] = mul_q16(v, SQRT2);
                }
            }
        }
    } else {
        // H_DCT: DCT along rows (width axis), identity along columns.
        for y in 0..h {
            if w == 16 {
                let mut row = [0i32; 16];
                row.copy_from_slice(&residual[y * w..y * w + 16]);
                dct1d_16_i32(&mut row);
                for (x, &v) in row.iter().enumerate() {
                    f[y * w + x] = mul_q16(v, SQRT2);
                }
            } else {
                let mut row = [0i32; 8];
                row.copy_from_slice(&residual[y * w..y * w + 8]);
                dct1d_8_i32(&mut row);
                for (x, &v) in row.iter().enumerate() {
                    f[y * w + x] = v * 2;
                }
            }
        }
    }
    let mut cf = [0i32; 128];
    let mut tf = [0f32; 128];
    for y in 0..h {
        for x in 0..w {
            let rc = y + x * h; // transposed storage, matching the inverses
            let m = if rc == 0 { qm_dc } else { qm_ac };
            quant_fullstep(&mut cf, &mut tf, rc, f[y * w + x], m);
        }
    }
    (cf, tf)
}

#[cfg(test)]
pub(crate) fn fvdct16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct_rect_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), 16, true)
}

#[cfg(test)]
pub(crate) fn fhdct16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct_rect_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), 16, false)
}

#[cfg(test)]
pub(crate) fn fvdct8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct_rect_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), 8, true)
}

#[cfg(test)]
pub(crate) fn fhdct8x16_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    f1ddct_rect_quant_t_direct(residual, quant.q_mult_dc(), quant.q_mult_ac(), 8, false)
}

#[cfg(test)]
pub(crate) fn fidentity_rect_t(
    residual: &[i32; 128],
    quant: &impl Dct,
    w: usize,
) -> ([i32; 128], [f32; 128]) {
    let quant = &crate::quant::FlatDct(quant); // identity family: QM not applied
    let h = 128 / w;
    let (dc_q, ac_q) = (quant.dc_q(), quant.ac_q());
    let mut cf = [0i32; 128];
    let mut tf = [0.0f32; 128];
    for y in 0..h {
        for x in 0..w {
            let rc = y + x * h;
            let qd = if rc == 0 { dc_q } else { ac_q };
            let num = residual[y * w + x] * 8;
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
    quant_flat(&coeffs, dc_q, ac_q, input);
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

fn fvdct4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    f1ddct4x4_quant_t_direct(input, dc_q, ac_q, true)
}

fn fhdct4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    f1ddct4x4_quant_t_direct(input, dc_q, ac_q, false)
}

fn fvdct8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    f1ddct8x8_quant_t_direct(input, dc_q, ac_q, true)
}

fn fhdct8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    f1ddct8x8_quant_t_direct(input, dc_q, ac_q, false)
}

fn fvdct16x8_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    f1ddct_rect_quant_t_direct(input, dc_q, ac_q, 16, true)
}

fn fhdct16x8_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    f1ddct_rect_quant_t_direct(input, dc_q, ac_q, 16, false)
}

fn fvdct8x16_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    f1ddct_rect_quant_t_direct(input, dc_q, ac_q, 8, true)
}

fn fhdct8x16_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    f1ddct_rect_quant_t_direct(input, dc_q, ac_q, 8, false)
}

fn identity_quant_t_direct<const N: usize>(
    residual: &[i32; N],
    dc_q: i32,
    ac_q: i32,
    w: usize,
    h: usize,
) -> ([i32; N], [f32; N]) {
    let mut cf = [0i32; N];
    let mut tf = [0.0f32; N];
    for y in 0..h {
        for x in 0..w {
            let rc = y + x * h;
            let qd = if rc == 0 { dc_q } else { ac_q };
            let num = residual[y * w + x] * 8;
            tf[rc] = num as f32 / qd as f32;
            let am = num.unsigned_abs() as i32;
            cf[rc] = if am < qd {
                0
            } else {
                let level = (am + qd / 2) / qd;
                if num < 0 { -level } else { level }
            };
        }
    }
    (cf, tf)
}

fn idtx4x4_quant_t_direct(input: &[i32; 16], dc_q: i32, ac_q: i32) -> ([i32; 16], [f32; 16]) {
    identity_quant_t_direct(input, dc_q, ac_q, 4, 4)
}

fn idtx8x8_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    identity_quant_t_direct(input, dc_q, ac_q, 8, 8)
}

fn idtx8x16_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    identity_quant_t_direct(input, dc_q, ac_q, 8, 16)
}

fn idtx16x8_quant_t_direct(input: &[i32; 128], dc_q: i32, ac_q: i32) -> ([i32; 128], [f32; 128]) {
    identity_quant_t_direct(input, dc_q, ac_q, 16, 8)
}

fn idtx16x16_quant_t_direct(input: &[i32; 256], dc_q: i32, ac_q: i32) -> ([i32; 256], [f32; 256]) {
    identity_quant_t_direct(input, dc_q, ac_q, 16, 16)
}

pub(crate) type DctFn<const N: usize> = fn(&[i32; N], i32, i32) -> ([i32; N], [f32; N]);
type Dct8x8QuantTFn = DctFn<64>;
type Dct8x16QuantTFn = DctFn<128>;
type Dct16x16QuantTFn = DctFn<256>;
type Tx16x16QuantTFn = DctFn<256>;
type Dct32x32QuantTFn = DctFn<1024>;
type Dct16x32QuantTFn = DctFn<512>;
type Dct32x16QuantTFn = DctFn<512>;

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn apply_qmatrix_neon_wrap(levels: &mut [i32], targets: &mut [f32], inverse_weights: &[u8]) {
    unsafe { crate::neon::apply_qmatrix_neon(levels, targets, inverse_weights) }
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn apply_qmatrix_avx2_wrap(levels: &mut [i32], targets: &mut [f32], inverse_weights: &[u8]) {
    unsafe { crate::avx::apply_qmatrix_avx2(levels, targets, inverse_weights) }
}

pub(crate) fn selected_apply_qmatrix() -> ApplyQmatrixFn {
    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    {
        apply_qmatrix_neon_wrap
    }
    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    {
        if std::is_x86_feature_detected!("avx2") {
            apply_qmatrix_avx2_wrap
        } else {
            apply_qmatrix
        }
    }
    #[cfg(not(any(
        all(target_arch = "aarch64", feature = "neon"),
        all(target_arch = "x86_64", feature = "avx")
    )))]
    {
        apply_qmatrix
    }
}

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

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
macro_rules! small_neon_wrap {
    ($name:ident, $neon:ident, $n:literal) => {
        fn $name(input: &[i32; $n], dc_q: i32, ac_q: i32) -> ([i32; $n], [f32; $n]) {
            unsafe { crate::neon::$neon(input, dc_q, ac_q) }
        }
    };
}

#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct4x4_neon_quant_t_wrap, dct4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct4x8_neon_quant_t_wrap, dct4x8_neon_quant_t, 32);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct8x4_neon_quant_t_wrap, dct8x4_neon_quant_t, 32);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct4x16_neon_quant_t_wrap, dct4x16_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct16x4_neon_quant_t_wrap, dct16x4_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dct16x8_neon_quant_t_wrap, dct16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adst4x4_neon_quant_t_wrap, adst4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adstdct4x4_neon_quant_t_wrap, adstdct4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dctadst4x4_neon_quant_t_wrap, dctadst4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adst4x8_neon_quant_t_wrap, adst4x8_neon_quant_t, 32);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adstdct4x8_neon_quant_t_wrap, adstdct4x8_neon_quant_t, 32);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dctadst4x8_neon_quant_t_wrap, dctadst4x8_neon_quant_t, 32);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adst8x8_neon_quant_t_wrap, adst8x8_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adstdct8x8_neon_quant_t_wrap, adstdct8x8_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dctadst8x8_neon_quant_t_wrap, dctadst8x8_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adst8x16_neon_quant_t_wrap, adst8x16_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adstdct8x16_neon_quant_t_wrap, adstdct8x16_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dctadst8x16_neon_quant_t_wrap, dctadst8x16_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adst16x8_neon_quant_t_wrap, adst16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(adstdct16x8_neon_quant_t_wrap, adstdct16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(dctadst16x8_neon_quant_t_wrap, dctadst16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fvdct4x4_neon_quant_t_wrap, fvdct4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fhdct4x4_neon_quant_t_wrap, fhdct4x4_neon_quant_t, 16);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fvdct8x8_neon_quant_t_wrap, fvdct8x8_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fhdct8x8_neon_quant_t_wrap, fhdct8x8_neon_quant_t, 64);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fvdct16x8_neon_quant_t_wrap, fvdct16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fhdct16x8_neon_quant_t_wrap, fhdct16x8_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fvdct8x16_neon_quant_t_wrap, fvdct8x16_neon_quant_t, 128);
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
small_neon_wrap!(fhdct8x16_neon_quant_t_wrap, fhdct8x16_neon_quant_t, 128);

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

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
macro_rules! small_avx2_wrap {
    ($name:ident, $avx2:ident, $n:literal) => {
        fn $name(input: &[i32; $n], dc_q: i32, ac_q: i32) -> ([i32; $n], [f32; $n]) {
            unsafe { crate::avx::$avx2(input, dc_q, ac_q) }
        }
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct4x4_avx2_quant_t_wrap, dct4x4_avx2_quant_t, 16);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct4x8_avx2_quant_t_wrap, dct4x8_avx2_quant_t, 32);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct8x4_avx2_quant_t_wrap, dct8x4_avx2_quant_t, 32);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct4x16_avx2_quant_t_wrap, dct4x16_avx2_quant_t, 64);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct16x4_avx2_quant_t_wrap, dct16x4_avx2_quant_t, 64);
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
small_avx2_wrap!(dct16x8_avx2_quant_t_wrap, dct16x8_avx2_quant_t, 128);

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
macro_rules! avx2_wrap_set {
    ($(($wrap:ident, $func:ident, $n:literal)),* $(,)?) => {
        $(small_avx2_wrap!($wrap, $func, $n);)*
    };
}

#[cfg(all(target_arch = "x86_64", feature = "avx"))]
avx2_wrap_set!(
    (adst4x4_avx2_quant_t_wrap, adst4x4_avx2_quant_t, 16),
    (adstdct4x4_avx2_quant_t_wrap, adstdct4x4_avx2_quant_t, 16),
    (dctadst4x4_avx2_quant_t_wrap, dctadst4x4_avx2_quant_t, 16),
    (adst4x8_avx2_quant_t_wrap, adst4x8_avx2_quant_t, 32),
    (adstdct4x8_avx2_quant_t_wrap, adstdct4x8_avx2_quant_t, 32),
    (dctadst4x8_avx2_quant_t_wrap, dctadst4x8_avx2_quant_t, 32),
    (adst8x8_avx2_quant_t_wrap, adst8x8_avx2_quant_t, 64),
    (adstdct8x8_avx2_quant_t_wrap, adstdct8x8_avx2_quant_t, 64),
    (dctadst8x8_avx2_quant_t_wrap, dctadst8x8_avx2_quant_t, 64),
    (adst8x16_avx2_quant_t_wrap, adst8x16_avx2_quant_t, 128),
    (adstdct8x16_avx2_quant_t_wrap, adstdct8x16_avx2_quant_t, 128),
    (dctadst8x16_avx2_quant_t_wrap, dctadst8x16_avx2_quant_t, 128),
    (adst16x8_avx2_quant_t_wrap, adst16x8_avx2_quant_t, 128),
    (adstdct16x8_avx2_quant_t_wrap, adstdct16x8_avx2_quant_t, 128),
    (dctadst16x8_avx2_quant_t_wrap, dctadst16x8_avx2_quant_t, 128),
    (fvdct4x4_avx2_quant_t_wrap, fvdct4x4_avx2_quant_t, 16),
    (fhdct4x4_avx2_quant_t_wrap, fhdct4x4_avx2_quant_t, 16),
    (fvdct8x8_avx2_quant_t_wrap, fvdct8x8_avx2_quant_t, 64),
    (fhdct8x8_avx2_quant_t_wrap, fhdct8x8_avx2_quant_t, 64),
    (fvdct16x8_avx2_quant_t_wrap, fvdct16x8_avx2_quant_t, 128),
    (fhdct16x8_avx2_quant_t_wrap, fhdct16x8_avx2_quant_t, 128),
    (fvdct8x16_avx2_quant_t_wrap, fvdct8x16_avx2_quant_t, 128),
    (fhdct8x16_avx2_quant_t_wrap, fhdct8x16_avx2_quant_t, 128),
);

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

macro_rules! selected_forward_tx {
    ($name:ident, $scalar:ident, $neon_wrap:ident, $avx2_wrap:ident, $n:literal) => {
        fn $name() -> DctFn<$n> {
            #[cfg(all(target_arch = "aarch64", feature = "neon"))]
            {
                $neon_wrap
            }
            #[cfg(all(target_arch = "x86_64", feature = "avx"))]
            {
                if std::is_x86_feature_detected!("avx2") {
                    $avx2_wrap
                } else {
                    $scalar
                }
            }
            #[cfg(not(any(
                all(target_arch = "aarch64", feature = "neon"),
                all(target_arch = "x86_64", feature = "avx")
            )))]
            {
                $scalar
            }
        }
    };
}

selected_forward_tx!(
    selected_dct4x4,
    dct4x4_quant_t_direct,
    dct4x4_neon_quant_t_wrap,
    dct4x4_avx2_quant_t_wrap,
    16
);
selected_forward_tx!(
    selected_dct4x8,
    dct4x8_quant_t_direct,
    dct4x8_neon_quant_t_wrap,
    dct4x8_avx2_quant_t_wrap,
    32
);
selected_forward_tx!(
    selected_dct8x4,
    dct8x4_quant_t_direct,
    dct8x4_neon_quant_t_wrap,
    dct8x4_avx2_quant_t_wrap,
    32
);
selected_forward_tx!(
    selected_dct4x16,
    dct4x16_quant_t_direct,
    dct4x16_neon_quant_t_wrap,
    dct4x16_avx2_quant_t_wrap,
    64
);
selected_forward_tx!(
    selected_dct16x4,
    dct16x4_quant_t_direct,
    dct16x4_neon_quant_t_wrap,
    dct16x4_avx2_quant_t_wrap,
    64
);
selected_forward_tx!(
    selected_dct16x8,
    dct16x8_quant_t_direct,
    dct16x8_neon_quant_t_wrap,
    dct16x8_avx2_quant_t_wrap,
    128
);

macro_rules! select_forward_set {
    ($(($name:ident, $scalar:ident, $neon:ident, $avx2:ident, $n:literal)),* $(,)?) => {
        $(selected_forward_tx!($name, $scalar, $neon, $avx2, $n);)*
    };
}

select_forward_set!(
    (
        selected_adst4x4,
        adst4x4_quant_t_direct,
        adst4x4_neon_quant_t_wrap,
        adst4x4_avx2_quant_t_wrap,
        16
    ),
    (
        selected_adstdct4x4,
        adstdct4x4_quant_t_direct,
        adstdct4x4_neon_quant_t_wrap,
        adstdct4x4_avx2_quant_t_wrap,
        16
    ),
    (
        selected_dctadst4x4,
        dctadst4x4_quant_t_direct,
        dctadst4x4_neon_quant_t_wrap,
        dctadst4x4_avx2_quant_t_wrap,
        16
    ),
    (
        selected_adst4x8,
        adst4x8_quant_t_direct,
        adst4x8_neon_quant_t_wrap,
        adst4x8_avx2_quant_t_wrap,
        32
    ),
    (
        selected_adstdct4x8,
        adstdct4x8_quant_t_direct,
        adstdct4x8_neon_quant_t_wrap,
        adstdct4x8_avx2_quant_t_wrap,
        32
    ),
    (
        selected_dctadst4x8,
        dctadst4x8_quant_t_direct,
        dctadst4x8_neon_quant_t_wrap,
        dctadst4x8_avx2_quant_t_wrap,
        32
    ),
    (
        selected_adst8x8,
        adst8x8_quant_t_direct,
        adst8x8_neon_quant_t_wrap,
        adst8x8_avx2_quant_t_wrap,
        64
    ),
    (
        selected_adstdct8x8,
        adstdct8x8_quant_t_direct,
        adstdct8x8_neon_quant_t_wrap,
        adstdct8x8_avx2_quant_t_wrap,
        64
    ),
    (
        selected_dctadst8x8,
        dctadst8x8_quant_t_direct,
        dctadst8x8_neon_quant_t_wrap,
        dctadst8x8_avx2_quant_t_wrap,
        64
    ),
    (
        selected_adst8x16,
        adst8x16_quant_t_direct,
        adst8x16_neon_quant_t_wrap,
        adst8x16_avx2_quant_t_wrap,
        128
    ),
    (
        selected_adstdct8x16,
        adstdct8x16_quant_t_direct,
        adstdct8x16_neon_quant_t_wrap,
        adstdct8x16_avx2_quant_t_wrap,
        128
    ),
    (
        selected_dctadst8x16,
        dctadst8x16_quant_t_direct,
        dctadst8x16_neon_quant_t_wrap,
        dctadst8x16_avx2_quant_t_wrap,
        128
    ),
    (
        selected_adst16x8,
        adst16x8_quant_t_direct,
        adst16x8_neon_quant_t_wrap,
        adst16x8_avx2_quant_t_wrap,
        128
    ),
    (
        selected_adstdct16x8,
        adstdct16x8_quant_t_direct,
        adstdct16x8_neon_quant_t_wrap,
        adstdct16x8_avx2_quant_t_wrap,
        128
    ),
    (
        selected_dctadst16x8,
        dctadst16x8_quant_t_direct,
        dctadst16x8_neon_quant_t_wrap,
        dctadst16x8_avx2_quant_t_wrap,
        128
    ),
    (
        selected_fvdct4x4,
        fvdct4x4_quant_t_direct,
        fvdct4x4_neon_quant_t_wrap,
        fvdct4x4_avx2_quant_t_wrap,
        16
    ),
    (
        selected_fhdct4x4,
        fhdct4x4_quant_t_direct,
        fhdct4x4_neon_quant_t_wrap,
        fhdct4x4_avx2_quant_t_wrap,
        16
    ),
    (
        selected_fvdct8x8,
        fvdct8x8_quant_t_direct,
        fvdct8x8_neon_quant_t_wrap,
        fvdct8x8_avx2_quant_t_wrap,
        64
    ),
    (
        selected_fhdct8x8,
        fhdct8x8_quant_t_direct,
        fhdct8x8_neon_quant_t_wrap,
        fhdct8x8_avx2_quant_t_wrap,
        64
    ),
    (
        selected_fvdct16x8,
        fvdct16x8_quant_t_direct,
        fvdct16x8_neon_quant_t_wrap,
        fvdct16x8_avx2_quant_t_wrap,
        128
    ),
    (
        selected_fhdct16x8,
        fhdct16x8_quant_t_direct,
        fhdct16x8_neon_quant_t_wrap,
        fhdct16x8_avx2_quant_t_wrap,
        128
    ),
    (
        selected_fvdct8x16,
        fvdct8x16_quant_t_direct,
        fvdct8x16_neon_quant_t_wrap,
        fvdct8x16_avx2_quant_t_wrap,
        128
    ),
    (
        selected_fhdct8x16,
        fhdct8x16_quant_t_direct,
        fhdct8x16_neon_quant_t_wrap,
        fhdct8x16_avx2_quant_t_wrap,
        128
    ),
);

/// Per-encode AV1 forward-DCT dispatch. The selected pointers are copied into
/// each tile so block and transform-search loops never touch the module
/// `OnceLock`s.
#[derive(Clone, Copy)]
pub(crate) struct DctDispatch {
    pub(crate) apply_qmatrix: ApplyQmatrixFn,
    pub(crate) dct4x4: DctFn<16>,
    pub(crate) dct4x8: DctFn<32>,
    pub(crate) dct8x4: DctFn<32>,
    pub(crate) dct4x16: DctFn<64>,
    pub(crate) dct16x4: DctFn<64>,
    pub(crate) dct8x8: DctFn<64>,
    pub(crate) dct8x16: DctFn<128>,
    pub(crate) dct16x8: DctFn<128>,
    pub(crate) dct16x16: DctFn<256>,
    pub(crate) dct16x32: DctFn<512>,
    pub(crate) dct32x16: DctFn<512>,
    pub(crate) dct32x32: DctFn<1024>,

    pub(crate) adst4x4: DctFn<16>,
    pub(crate) adstdct4x4: DctFn<16>,
    pub(crate) dctadst4x4: DctFn<16>,
    pub(crate) adst4x8: DctFn<32>,
    pub(crate) adstdct4x8: DctFn<32>,
    pub(crate) dctadst4x8: DctFn<32>,
    pub(crate) adst8x8: DctFn<64>,
    pub(crate) adstdct8x8: DctFn<64>,
    pub(crate) dctadst8x8: DctFn<64>,
    pub(crate) adst8x16: DctFn<128>,
    pub(crate) adstdct8x16: DctFn<128>,
    pub(crate) dctadst8x16: DctFn<128>,
    pub(crate) adst16x8: DctFn<128>,
    pub(crate) adstdct16x8: DctFn<128>,
    pub(crate) dctadst16x8: DctFn<128>,
    pub(crate) adst16x16: DctFn<256>,
    pub(crate) adstdct16x16: DctFn<256>,
    pub(crate) dctadst16x16: DctFn<256>,

    pub(crate) fvdct4x4: DctFn<16>,
    pub(crate) fhdct4x4: DctFn<16>,
    pub(crate) fvdct8x8: DctFn<64>,
    pub(crate) fhdct8x8: DctFn<64>,
    pub(crate) fvdct8x16: DctFn<128>,
    pub(crate) fhdct8x16: DctFn<128>,
    pub(crate) fvdct16x8: DctFn<128>,
    pub(crate) fhdct16x8: DctFn<128>,
    pub(crate) idtx4x4: DctFn<16>,
    pub(crate) idtx8x8: DctFn<64>,
    pub(crate) idtx8x16: DctFn<128>,
    pub(crate) idtx16x8: DctFn<128>,
    pub(crate) idtx16x16: DctFn<256>,
}

macro_rules! dispatch_qm_ratio_methods {
    ($(($method:ident, $field:ident, $n:literal, $w:literal, $h:literal, $ratio:expr)),* $(,)?) => {
        $(
            pub(crate) fn $method(
                &self,
                residual: &[i32; $n],
                quant: &impl Dct,
            ) -> ([i32; $n], [f32; $n]) {
                // Transform-gain compensation for the 4-wide family — the
                // pre-dispatch wrappers applied this and the refactor dropped
                // it (bisected 2026-07-26: +1.77% BD, x_fractal +5.70 — the
                // mis-scaled forward made ADST-4x4/4x8 candidates falsely
                // cheap).
                let dc = mul_q16(quant.q_mult_dc(), $ratio);
                let ac = mul_q16(quant.q_mult_ac(), $ratio);
                apply_qmatrix_result(
                    (self.$field)(residual, dc, ac),
                    quant,
                    $w,
                    $h,
                    self.apply_qmatrix,
                )
            }
        )*
    };
}

macro_rules! dispatch_qm_methods {
    ($(($method:ident, $field:ident, $n:literal, $w:literal, $h:literal)),* $(,)?) => {
        $(
            pub(crate) fn $method(
                &self,
                residual: &[i32; $n],
                quant: &impl Dct,
            ) -> ([i32; $n], [f32; $n]) {
                apply_qmatrix_result(
                    (self.$field)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
                    quant,
                    $w,
                    $h,
                    self.apply_qmatrix,
                )
            }
        )*
    };
}

macro_rules! dispatch_flat_methods {
    ($(($method:ident, $field:ident, $n:literal)),* $(,)?) => {
        $(
            pub(crate) fn $method(
                &self,
                residual: &[i32; $n],
                quant: &impl Dct,
            ) -> ([i32; $n], [f32; $n]) {
                let quant = &crate::quant::FlatDct(quant);
                (self.$field)(residual, quant.q_mult_dc(), quant.q_mult_ac())
            }
        )*
    };
}

impl DctDispatch {
    pub(crate) fn selected() -> Self {
        Self {
            apply_qmatrix: selected_apply_qmatrix(),
            dct4x4: selected_dct4x4(),
            dct4x8: selected_dct4x8(),
            dct8x4: selected_dct8x4(),
            dct4x16: selected_dct4x16(),
            dct16x4: selected_dct16x4(),
            dct8x8: resolve_dct8x8_quant_t(),
            dct8x16: resolve_dct8x16_quant_t(),
            dct16x8: selected_dct16x8(),
            dct16x16: resolve_dct16x16_quant_t(),
            dct16x32: resolve_dct16x32_quant_t(),
            dct32x16: resolve_dct32x16_quant_t(),
            dct32x32: resolve_dct32x32_quant_t(),
            adst4x4: selected_adst4x4(),
            adstdct4x4: selected_adstdct4x4(),
            dctadst4x4: selected_dctadst4x4(),
            adst4x8: selected_adst4x8(),
            adstdct4x8: selected_adstdct4x8(),
            dctadst4x8: selected_dctadst4x8(),
            adst8x8: selected_adst8x8(),
            adstdct8x8: selected_adstdct8x8(),
            dctadst8x8: selected_dctadst8x8(),
            adst8x16: selected_adst8x16(),
            adstdct8x16: selected_adstdct8x16(),
            dctadst8x16: selected_dctadst8x16(),
            adst16x8: selected_adst16x8(),
            adstdct16x8: selected_adstdct16x8(),
            dctadst16x8: selected_dctadst16x8(),
            adst16x16: resolve_adst16x16_quant_t(),
            adstdct16x16: resolve_adstdct16x16_quant_t(),
            dctadst16x16: resolve_dctadst16x16_quant_t(),
            fvdct4x4: selected_fvdct4x4(),
            fhdct4x4: selected_fhdct4x4(),
            fvdct8x8: selected_fvdct8x8(),
            fhdct8x8: selected_fhdct8x8(),
            fvdct8x16: selected_fvdct8x16(),
            fhdct8x16: selected_fhdct8x16(),
            fvdct16x8: selected_fvdct16x8(),
            fhdct16x8: selected_fhdct16x8(),
            idtx4x4: idtx4x4_quant_t_direct,
            idtx8x8: idtx8x8_quant_t_direct,
            idtx8x16: idtx8x16_quant_t_direct,
            idtx16x8: idtx16x8_quant_t_direct,
            idtx16x16: idtx16x16_quant_t_direct,
        }
    }

    pub(crate) fn scalar() -> Self {
        Self {
            apply_qmatrix,
            dct4x4: dct4x4_quant_t_direct,
            dct4x8: dct4x8_quant_t_direct,
            dct8x4: dct8x4_quant_t_direct,
            dct4x16: dct4x16_quant_t_direct,
            dct16x4: dct16x4_quant_t_direct,
            dct8x8: dct8x8_quant_t_direct,
            dct8x16: dct8x16_quant_t_direct,
            dct16x8: dct16x8_quant_t_direct,
            dct16x16: dct16x16_quant_t_direct,
            dct16x32: dct16x32_quant_t_direct,
            dct32x16: dct32x16_quant_t_direct,
            dct32x32: dct32x32_quant_t_direct,
            adst4x4: adst4x4_quant_t_direct,
            adstdct4x4: adstdct4x4_quant_t_direct,
            dctadst4x4: dctadst4x4_quant_t_direct,
            adst4x8: adst4x8_quant_t_direct,
            adstdct4x8: adstdct4x8_quant_t_direct,
            dctadst4x8: dctadst4x8_quant_t_direct,
            adst8x8: adst8x8_quant_t_direct,
            adstdct8x8: adstdct8x8_quant_t_direct,
            dctadst8x8: dctadst8x8_quant_t_direct,
            adst8x16: adst8x16_quant_t_direct,
            adstdct8x16: adstdct8x16_quant_t_direct,
            dctadst8x16: dctadst8x16_quant_t_direct,
            adst16x8: adst16x8_quant_t_direct,
            adstdct16x8: adstdct16x8_quant_t_direct,
            dctadst16x8: dctadst16x8_quant_t_direct,
            adst16x16: adst16x16_quant_t_direct,
            adstdct16x16: adstdct16x16_quant_t_direct,
            dctadst16x16: dctadst16x16_quant_t_direct,
            fvdct4x4: fvdct4x4_quant_t_direct,
            fhdct4x4: fhdct4x4_quant_t_direct,
            fvdct8x8: fvdct8x8_quant_t_direct,
            fhdct8x8: fhdct8x8_quant_t_direct,
            fvdct8x16: fvdct8x16_quant_t_direct,
            fhdct8x16: fhdct8x16_quant_t_direct,
            fvdct16x8: fvdct16x8_quant_t_direct,
            fhdct16x8: fhdct16x8_quant_t_direct,
            idtx4x4: idtx4x4_quant_t_direct,
            idtx8x8: idtx8x8_quant_t_direct,
            idtx8x16: idtx8x16_quant_t_direct,
            idtx16x8: idtx16x8_quant_t_direct,
            idtx16x16: idtx16x16_quant_t_direct,
        }
    }

    pub(crate) fn dct4x4_t(
        &self,
        residual: &[i32; 16],
        quant: &impl Dct,
    ) -> ([i32; 16], [f32; 16]) {
        let dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
        let ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
        apply_qmatrix_result(
            (self.dct4x4)(residual, dc, ac),
            quant,
            4,
            4,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct4x8_t(
        &self,
        residual: &[i32; 32],
        quant: &impl Dct,
    ) -> ([i32; 32], [f32; 32]) {
        let dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
        let ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
        apply_qmatrix_result(
            (self.dct4x8)(residual, dc, ac),
            quant,
            4,
            8,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct8x4_t(
        &self,
        residual: &[i32; 32],
        quant: &impl Dct,
    ) -> ([i32; 32], [f32; 32]) {
        let dc = mul_q16(quant.q_mult_dc(), RATIO_4X8_Q16);
        let ac = mul_q16(quant.q_mult_ac(), RATIO_4X8_Q16);
        apply_qmatrix_result(
            (self.dct8x4)(residual, dc, ac),
            quant,
            8,
            4,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct4x16_t(
        &self,
        residual: &[i32; 64],
        quant: &impl Dct,
    ) -> ([i32; 64], [f32; 64]) {
        apply_qmatrix_result(
            (self.dct4x16)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            4,
            16,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct16x4_t(
        &self,
        residual: &[i32; 64],
        quant: &impl Dct,
    ) -> ([i32; 64], [f32; 64]) {
        apply_qmatrix_result(
            (self.dct16x4)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            16,
            4,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct8x8_t(
        &self,
        residual: &[i32; 64],
        quant: &impl Dct,
    ) -> ([i32; 64], [f32; 64]) {
        apply_qmatrix_result(
            (self.dct8x8)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            8,
            8,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct8x16_t(
        &self,
        residual: &[i32; 128],
        quant: &impl Dct,
    ) -> ([i32; 128], [f32; 128]) {
        apply_qmatrix_result(
            (self.dct8x16)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            8,
            16,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct16x8_t(
        &self,
        residual: &[i32; 128],
        quant: &impl Dct,
    ) -> ([i32; 128], [f32; 128]) {
        apply_qmatrix_result(
            (self.dct16x8)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            16,
            8,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct16x16_t(
        &self,
        residual: &[i32; 256],
        quant: &impl Dct,
    ) -> ([i32; 256], [f32; 256]) {
        apply_qmatrix_result(
            (self.dct16x16)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            16,
            16,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct16x32_t(
        &self,
        residual: &[i32; 512],
        quant: &impl Dct,
    ) -> ([i32; 512], [f32; 512]) {
        let dc = mul_q16(quant.q_mult_dc(), RATIO_16X32_Q16);
        let ac = mul_q16(quant.q_mult_ac(), RATIO_16X32_Q16);
        apply_qmatrix_result(
            (self.dct16x32)(residual, dc, ac),
            quant,
            16,
            32,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct32x16_t(
        &self,
        residual: &[i32; 512],
        quant: &impl Dct,
    ) -> ([i32; 512], [f32; 512]) {
        let dc = mul_q16(quant.q_mult_dc(), RATIO_16X32_Q16);
        let ac = mul_q16(quant.q_mult_ac(), RATIO_16X32_Q16);
        apply_qmatrix_result(
            (self.dct32x16)(residual, dc, ac),
            quant,
            32,
            16,
            self.apply_qmatrix,
        )
    }

    pub(crate) fn dct32x32_t(
        &self,
        residual: &[i32; 1024],
        quant: &impl Dct,
    ) -> ([i32; 1024], [f32; 1024]) {
        apply_qmatrix_result(
            (self.dct32x32)(residual, quant.q_mult_dc(), quant.q_mult_ac()),
            quant,
            32,
            32,
            self.apply_qmatrix,
        )
    }

    dispatch_qm_ratio_methods!(
        (adst4x4_t, adst4x4, 16, 4, 4, RATIO_4X4_Q16),
        (adstdct4x4_t, adstdct4x4, 16, 4, 4, RATIO_4X4_Q16),
        (dctadst4x4_t, dctadst4x4, 16, 4, 4, RATIO_4X4_Q16),
        (adst4x8_t, adst4x8, 32, 4, 8, RATIO_4X8_Q16),
        (adstdct4x8_t, adstdct4x8, 32, 4, 8, RATIO_4X8_Q16),
        (dctadst4x8_t, dctadst4x8, 32, 4, 8, RATIO_4X8_Q16),
    );
    dispatch_qm_methods!(
        (adst8x8_t, adst8x8, 64, 8, 8),
        (adstdct8x8_t, adstdct8x8, 64, 8, 8),
        (dctadst8x8_t, dctadst8x8, 64, 8, 8),
        (adst8x16_t, adst8x16, 128, 8, 16),
        (adstdct8x16_t, adstdct8x16, 128, 8, 16),
        (dctadst8x16_t, dctadst8x16, 128, 8, 16),
        (adst16x8_t, adst16x8, 128, 16, 8),
        (adstdct16x8_t, adstdct16x8, 128, 16, 8),
        (dctadst16x8_t, dctadst16x8, 128, 16, 8),
        (adst16x16_t, adst16x16, 256, 16, 16),
        (adstdct16x16_t, adstdct16x16, 256, 16, 16),
        (dctadst16x16_t, dctadst16x16, 256, 16, 16),
    );

    dispatch_flat_methods!(
        (fvdct4x4_t, fvdct4x4, 16),
        (fhdct4x4_t, fhdct4x4, 16),
        (fvdct8x8_t, fvdct8x8, 64),
        (fhdct8x8_t, fhdct8x8, 64),
        (fvdct8x16_t, fvdct8x16, 128),
        (fhdct8x16_t, fhdct8x16, 128),
        (fvdct16x8_t, fvdct16x8, 128),
        (fhdct16x8_t, fhdct16x8, 128),
    );

    pub(crate) fn idtx4x4_t(
        &self,
        residual: &[i32; 16],
        quant: &impl Dct,
    ) -> ([i32; 16], [f32; 16]) {
        (self.idtx4x4)(residual, quant.dc_q(), quant.ac_q())
    }

    pub(crate) fn idtx8x8_t(
        &self,
        residual: &[i32; 64],
        quant: &impl Dct,
    ) -> ([i32; 64], [f32; 64]) {
        (self.idtx8x8)(residual, quant.dc_q(), quant.ac_q())
    }

    pub(crate) fn idtx8x16_t(
        &self,
        residual: &[i32; 128],
        quant: &impl Dct,
    ) -> ([i32; 128], [f32; 128]) {
        (self.idtx8x16)(residual, quant.dc_q(), quant.ac_q())
    }

    pub(crate) fn idtx16x8_t(
        &self,
        residual: &[i32; 128],
        quant: &impl Dct,
    ) -> ([i32; 128], [f32; 128]) {
        (self.idtx16x8)(residual, quant.dc_q(), quant.ac_q())
    }

    pub(crate) fn idtx16x16_t(
        &self,
        residual: &[i32; 256],
        quant: &impl Dct,
    ) -> ([i32; 256], [f32; 256]) {
        (self.idtx16x16)(residual, quant.dc_q(), quant.ac_q())
    }
}

#[inline]
fn dct16x4_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
    for col in 0..16usize {
        let mut c = [0i32; 4];
        for row in 0..4 {
            c[row] = input[row * 16 + col];
        }
        dct1d_4_i32(&mut c);
        for fy in 0..4 {
            tmp[fy * 16 + col] = c[fy];
        }
    }
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for fy in 0..4usize {
        let mut r: [i32; 16] = tmp[fy * 16..fy * 16 + 16].try_into().unwrap();
        dct1d_16_i32(&mut r);
        for fx in 0..16 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 4 + fy, r[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

/// DCT_DCT for RTX_4X16.
fn dct4x16_quant_t_direct(input: &[i32; 64], dc_q: i32, ac_q: i32) -> ([i32; 64], [f32; 64]) {
    let mut tmp = [0i32; 64];
    for col in 0..4usize {
        let mut c = [0i32; 16];
        for row in 0..16 {
            c[row] = input[row * 4 + col];
        }
        dct1d_16_i32(&mut c);
        for fy in 0..16 {
            tmp[fy * 4 + col] = c[fy];
        }
    }
    let mut cf = [0i32; 64];
    let mut tf = [0.0f32; 64];
    for fy in 0..16usize {
        let mut r: [i32; 4] = tmp[fy * 4..fy * 4 + 4].try_into().unwrap();
        dct1d_4_i32(&mut r);
        for fx in 0..4 {
            store_quant_target_scalar(&mut cf, &mut tf, fx * 16 + fy, r[fx], dc_q, ac_q);
        }
    }
    (cf, tf)
}

#[cfg(test)]
pub(crate) fn dct16x4_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dct16x4_neon_quant_t,
            dct16x4_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        16,
        4,
    )
}

#[cfg(test)]
pub(crate) fn dct4x16_t(residual: &[i32; 64], quant: &impl Dct) -> ([i32; 64], [f32; 64]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dct4x16_neon_quant_t,
            dct4x16_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
        quant,
        4,
        16,
    )
}

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

#[cfg(test)]
pub(crate) fn dct16x8_t(residual: &[i32; 128], quant: &impl Dct) -> ([i32; 128], [f32; 128]) {
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dct16x8_neon_quant_t,
            dct16x8_quant_t_direct,
            residual,
            quant.q_mult_dc(),
            quant.q_mult_ac()
        ),
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
pub(crate) static ADST4_FWD_Q12: [[i32; 4]; 4] = [
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

#[cfg(test)]
pub(crate) fn adst4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            adst4x4_neon_quant_t,
            adst4x4_quant_t_direct,
            residual,
            m_dc,
            m_ac
        ),
        quant,
        4,
        4,
    )
}

#[cfg(test)]
pub(crate) fn adstdct4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            adstdct4x4_neon_quant_t,
            adstdct4x4_quant_t_direct,
            residual,
            m_dc,
            m_ac
        ),
        quant,
        4,
        4,
    )
}

#[cfg(test)]
pub(crate) fn dctadst4x4_t(residual: &[i32; 16], quant: &impl Dct) -> ([i32; 16], [f32; 16]) {
    let m_dc = mul_q16(quant.q_mult_dc(), RATIO_4X4_Q16);
    let m_ac = mul_q16(quant.q_mult_ac(), RATIO_4X4_Q16);
    apply_qmatrix_result_scalar(
        neon_fwd_tx!(
            dctadst4x4_neon_quant_t,
            dctadst4x4_quant_t_direct,
            residual,
            m_dc,
            m_ac
        ),
        quant,
        4,
        4,
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::Quant;

    fn pat(n: usize) -> Vec<i32> {
        (0..n).map(|i| ((i * 41 + 7) % 113) as i32 - 56).collect()
    }

    fn forward_parity_inputs<const N: usize>() -> [(&'static str, [i32; N]); 7] {
        let mut seed = 0x9E37_79B9u32;
        let random = std::array::from_fn(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((seed >> 16) as i32 & 0x3ff) - 512
        });
        let random_12b = std::array::from_fn(|_| {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed % 8191) as i32 - 4095
        });
        [
            ("zeros", [0; N]),
            ("constant", [1023; N]),
            (
                "impulses",
                std::array::from_fn(|i| {
                    if i == 0 {
                        4095
                    } else if i + 1 == N {
                        -4095
                    } else {
                        0
                    }
                }),
            ),
            (
                "ramp",
                std::array::from_fn(|i| ((i * 73 + i * i * 19 + 11) % 511) as i32 - 255),
            ),
            (
                "alternating-12b",
                std::array::from_fn(|i| if i & 1 == 0 { 4095 } else { -4095 }),
            ),
            ("random", random),
            ("random-12b", random_12b),
        ]
    }

    fn assert_selected_matches_scalar<const N: usize>(
        name: &str,
        selected_fn: DctFn<N>,
        scalar_fn: DctFn<N>,
    ) {
        const QUANT_PAIRS: [(i32, i32); 4] = [
            (65536, 65536),
            (65536, 46341),
            (32768, 32768),
            (17341, 9017),
        ];
        for (case, input) in forward_parity_inputs() {
            for (dc_q, ac_q) in QUANT_PAIRS {
                let scalar = scalar_fn(&input, dc_q, ac_q);
                let selected = selected_fn(&input, dc_q, ac_q);
                assert_eq!(
                    scalar.0, selected.0,
                    "{name} {case}: levels differ at dc_q={dc_q}, ac_q={ac_q}"
                );
                for (i, (&s, &v)) in scalar.1.iter().zip(&selected.1).enumerate() {
                    let tolerance = 1.0e-5 * s.abs().max(1.0);
                    assert!(
                        (s - v).abs() <= tolerance,
                        "{name} {case}: target {i} differs at dc_q={dc_q}, ac_q={ac_q}: \
                         scalar={s}, selected={v}"
                    );
                }
            }
        }
    }

    fn assert_qmatrix_matches_scalar(name: &str, simd: ApplyQmatrixFn) {
        for len in [1, 3, 4, 7, 8, 15, 16, 31, 64, 128, 1024] {
            let inverse_weights: Vec<u8> =
                (0..len).map(|i| ((i * 37 + 1) % 255 + 1) as u8).collect();
            let targets: Vec<f32> = (0..len)
                .map(|i| match i % 8 {
                    0 => 0.0,
                    1 => 0.499,
                    2 => -0.499,
                    3 => 0.5,
                    4 => -0.5,
                    5 => 31.25,
                    6 => -63.75,
                    _ => ((i * 977 + 13) % 8191) as f32 / 17.0 - 240.0,
                })
                .collect();
            let mut scalar_targets = targets.clone();
            let mut scalar_levels = vec![i32::MIN; len];
            apply_qmatrix(&mut scalar_levels, &mut scalar_targets, &inverse_weights);

            let mut simd_targets = targets;
            let mut simd_levels = vec![i32::MIN; len];
            simd(&mut simd_levels, &mut simd_targets, &inverse_weights);

            assert_eq!(
                scalar_levels, simd_levels,
                "{name}: levels differ at len={len}"
            );
            assert_eq!(
                scalar_targets, simd_targets,
                "{name}: targets differ at len={len}"
            );
        }
    }

    #[test]
    fn selected_qmatrix_matches_scalar_for_dynamic_lengths() {
        assert_qmatrix_matches_scalar("selected", selected_apply_qmatrix());
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_qmatrix_matches_scalar_for_dynamic_lengths() {
        assert_qmatrix_matches_scalar("neon", apply_qmatrix_neon_wrap);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_qmatrix_matches_scalar_for_dynamic_lengths() {
        if std::is_x86_feature_detected!("avx2") {
            assert_qmatrix_matches_scalar("avx2", apply_qmatrix_avx2_wrap);
        }
    }

    #[test]
    fn selected_dct_dispatch_matches_scalar() {
        let selected = DctDispatch::selected();
        let scalar = DctDispatch::scalar();
        assert_selected_matches_scalar("dct4x4", selected.dct4x4, scalar.dct4x4);
        assert_selected_matches_scalar("dct4x8", selected.dct4x8, scalar.dct4x8);
        assert_selected_matches_scalar("dct8x4", selected.dct8x4, scalar.dct8x4);
        assert_selected_matches_scalar("dct4x16", selected.dct4x16, scalar.dct4x16);
        assert_selected_matches_scalar("dct16x4", selected.dct16x4, scalar.dct16x4);
        assert_selected_matches_scalar("dct8x8", selected.dct8x8, scalar.dct8x8);
        assert_selected_matches_scalar("dct8x16", selected.dct8x16, scalar.dct8x16);
        assert_selected_matches_scalar("dct16x8", selected.dct16x8, scalar.dct16x8);
        assert_selected_matches_scalar("dct16x16", selected.dct16x16, scalar.dct16x16);
        assert_selected_matches_scalar("dct16x32", selected.dct16x32, scalar.dct16x32);
        assert_selected_matches_scalar("dct32x16", selected.dct32x16, scalar.dct32x16);
        assert_selected_matches_scalar("dct32x32", selected.dct32x32, scalar.dct32x32);
        assert_selected_matches_scalar("adst4x4", selected.adst4x4, scalar.adst4x4);
        assert_selected_matches_scalar("adstdct4x4", selected.adstdct4x4, scalar.adstdct4x4);
        assert_selected_matches_scalar("dctadst4x4", selected.dctadst4x4, scalar.dctadst4x4);
        assert_selected_matches_scalar("adst4x8", selected.adst4x8, scalar.adst4x8);
        assert_selected_matches_scalar("adstdct4x8", selected.adstdct4x8, scalar.adstdct4x8);
        assert_selected_matches_scalar("dctadst4x8", selected.dctadst4x8, scalar.dctadst4x8);
        assert_selected_matches_scalar("adst8x8", selected.adst8x8, scalar.adst8x8);
        assert_selected_matches_scalar("adstdct8x8", selected.adstdct8x8, scalar.adstdct8x8);
        assert_selected_matches_scalar("dctadst8x8", selected.dctadst8x8, scalar.dctadst8x8);
        assert_selected_matches_scalar("adst8x16", selected.adst8x16, scalar.adst8x16);
        assert_selected_matches_scalar("adstdct8x16", selected.adstdct8x16, scalar.adstdct8x16);
        assert_selected_matches_scalar("dctadst8x16", selected.dctadst8x16, scalar.dctadst8x16);
        assert_selected_matches_scalar("adst16x8", selected.adst16x8, scalar.adst16x8);
        assert_selected_matches_scalar("adstdct16x8", selected.adstdct16x8, scalar.adstdct16x8);
        assert_selected_matches_scalar("dctadst16x8", selected.dctadst16x8, scalar.dctadst16x8);
        assert_selected_matches_scalar("adst16x16", selected.adst16x16, scalar.adst16x16);
        assert_selected_matches_scalar("adstdct16x16", selected.adstdct16x16, scalar.adstdct16x16);
        assert_selected_matches_scalar("dctadst16x16", selected.dctadst16x16, scalar.dctadst16x16);
        assert_selected_matches_scalar("fvdct4x4", selected.fvdct4x4, scalar.fvdct4x4);
        assert_selected_matches_scalar("fhdct4x4", selected.fhdct4x4, scalar.fhdct4x4);
        assert_selected_matches_scalar("fvdct8x8", selected.fvdct8x8, scalar.fvdct8x8);
        assert_selected_matches_scalar("fhdct8x8", selected.fhdct8x8, scalar.fhdct8x8);
        assert_selected_matches_scalar("fvdct8x16", selected.fvdct8x16, scalar.fvdct8x16);
        assert_selected_matches_scalar("fhdct8x16", selected.fhdct8x16, scalar.fhdct8x16);
        assert_selected_matches_scalar("fvdct16x8", selected.fvdct16x8, scalar.fvdct16x8);
        assert_selected_matches_scalar("fhdct16x8", selected.fhdct16x8, scalar.fhdct16x8);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    fn assert_forward_neon_matches_scalar<const N: usize>(
        name: &str,
        scalar_fn: fn(&[i32; N], i32, i32) -> ([i32; N], [f32; N]),
        neon_fn: unsafe fn(&[i32; N], i32, i32) -> ([i32; N], [f32; N]),
    ) {
        const QUANT_PAIRS: [(i32, i32); 4] = [
            (65536, 65536),
            (65536, 46341),
            (32768, 32768),
            (17341, 9017),
        ];
        for (case, input) in forward_parity_inputs() {
            for (dc_q, ac_q) in QUANT_PAIRS {
                let scalar = scalar_fn(&input, dc_q, ac_q);
                let neon = unsafe { neon_fn(&input, dc_q, ac_q) };
                assert_eq!(
                    scalar.0, neon.0,
                    "{name} {case}: levels differ at dc_q={dc_q}, ac_q={ac_q}"
                );
                for (i, (&s, &n)) in scalar.1.iter().zip(&neon.1).enumerate() {
                    let tolerance = 1.0e-5 * s.abs().max(1.0);
                    assert!(
                        (s - n).abs() <= tolerance,
                        "{name} {case}: target {i} differs at dc_q={dc_q}, ac_q={ac_q}: \
                         scalar={s}, neon={n}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    fn assert_forward_avx2_matches_scalar<const N: usize>(
        name: &str,
        scalar_fn: DctFn<N>,
        avx2_fn: unsafe fn(&[i32; N], i32, i32) -> ([i32; N], [f32; N]),
    ) {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        static QUANT_PAIRS: [(i32, i32); 4] = [
            (65536, 65536),
            (65536, 46341),
            (32768, 32768),
            (17341, 9017),
        ];
        for (case, input) in forward_parity_inputs() {
            for (dc_q, ac_q) in QUANT_PAIRS {
                let scalar = scalar_fn(&input, dc_q, ac_q);
                let avx2 = unsafe { avx2_fn(&input, dc_q, ac_q) };
                assert_eq!(
                    scalar.0, avx2.0,
                    "{name} {case}: levels differ at dc_q={dc_q}, ac_q={ac_q}"
                );
                for (i, (&s, &v)) in scalar.1.iter().zip(&avx2.1).enumerate() {
                    let tolerance = 1.0e-5 * s.abs().max(1.0);
                    assert!(
                        (s - v).abs() <= tolerance,
                        "{name} {case}: target {i} differs at dc_q={dc_q}, ac_q={ac_q}: \
                         scalar={s}, avx2={v}"
                    );
                }
            }
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_rectangular_dct_forward_transforms_match_scalar() {
        assert_forward_avx2_matches_scalar(
            "dct4x8",
            dct4x8_quant_t_direct,
            crate::avx::dct4x8_avx2_quant_t,
        );
        assert_forward_avx2_matches_scalar(
            "dct8x4",
            dct8x4_quant_t_direct,
            crate::avx::dct8x4_avx2_quant_t,
        );
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_dct_forward_transforms_match_scalar() {
        assert_forward_neon_matches_scalar(
            "dct4x4",
            dct4x4_quant_t_direct,
            crate::neon::dct4x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dct4x8",
            dct4x8_quant_t_direct,
            crate::neon::dct4x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dct8x4",
            dct8x4_quant_t_direct,
            crate::neon::dct8x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dct4x16",
            dct4x16_quant_t_direct,
            crate::neon::dct4x16_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dct16x4",
            dct16x4_quant_t_direct,
            crate::neon::dct16x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dct16x8",
            dct16x8_quant_t_direct,
            crate::neon::dct16x8_neon_quant_t,
        );
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_square_adst_forward_transforms_match_scalar() {
        assert_forward_neon_matches_scalar(
            "adst4x4",
            adst4x4_quant_t_direct,
            crate::neon::adst4x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adstdct4x4",
            adstdct4x4_quant_t_direct,
            crate::neon::adstdct4x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dctadst4x4",
            dctadst4x4_quant_t_direct,
            crate::neon::dctadst4x4_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adst8x8",
            adst8x8_quant_t_direct,
            crate::neon::adst8x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adstdct8x8",
            adstdct8x8_quant_t_direct,
            crate::neon::adstdct8x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dctadst8x8",
            dctadst8x8_quant_t_direct,
            crate::neon::dctadst8x8_neon_quant_t,
        );
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_rectangular_adst_forward_transforms_match_scalar() {
        assert_forward_neon_matches_scalar(
            "adst4x8",
            adst4x8_quant_t_direct,
            crate::neon::adst4x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adstdct4x8",
            adstdct4x8_quant_t_direct,
            crate::neon::adstdct4x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dctadst4x8",
            dctadst4x8_quant_t_direct,
            crate::neon::dctadst4x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adst8x16",
            adst8x16_quant_t_direct,
            crate::neon::adst8x16_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adstdct8x16",
            adstdct8x16_quant_t_direct,
            crate::neon::adstdct8x16_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dctadst8x16",
            dctadst8x16_quant_t_direct,
            crate::neon::dctadst8x16_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adst16x8",
            adst16x8_quant_t_direct,
            crate::neon::adst16x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "adstdct16x8",
            adstdct16x8_quant_t_direct,
            crate::neon::adstdct16x8_neon_quant_t,
        );
        assert_forward_neon_matches_scalar(
            "dctadst16x8",
            dctadst16x8_quant_t_direct,
            crate::neon::dctadst16x8_neon_quant_t,
        );
    }

    /// The new 4:1 pairs must round-trip a residual within quant rounding at a
    /// fine quantizer — pins the forward gain against the inverse shift chain.
    #[test]
    fn dct16x4_and_4x16_pair_with_inverse() {
        use crate::idct::{idct_dequant_4x16, idct_dequant_16x4};
        let q = Quant::new(8, 8);
        let r: [i32; 64] = pat(64).try_into().unwrap();
        for wide in [true, false] {
            let (cf, rec) = if wide {
                let (cf, _) = dct16x4_t(&r, &q);
                (cf, idct_dequant_16x4(&cf, &q))
            } else {
                let (cf, _) = dct4x16_t(&r, &q);
                (cf, idct_dequant_4x16(&cf, &q))
            };
            assert!(cf.iter().any(|&c| c != 0), "all-zero coeffs at fine q");
            let sig: f64 = r.iter().map(|&v| (v as f64).powi(2)).sum();
            let err: f64 = r
                .iter()
                .zip(rec.iter())
                .map(|(&a, &b)| ((a - b) as f64).powi(2))
                .sum();
            assert!(err < sig * 0.05, "wide={wide}: err {err} vs sig {sig}");
        }
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
