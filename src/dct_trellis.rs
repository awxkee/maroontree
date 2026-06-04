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
use crate::av1real::{Dct, forward_dct_quant_8x8};

#[cfg(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
))]
#[inline(always)]
#[allow(unused)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    f32::mul_add(a, b, c)
}

#[cfg(not(any(
    all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "fma"
    ),
    target_arch = "aarch64"
)))]
#[inline(always)]
#[allow(unused)]
pub(crate) fn fmla(a: f32, b: f32, c: f32) -> f32 {
    a * b + c
}

pub(crate) const WC4: [f32; 2] = [0.541_196_1, 1.306_563];

pub(crate) const WC8: [f32; 4] = [0.509_795_6, 0.601_344_9, 0.899_976_2, 2.562_915_6];

#[allow(unused)]
#[inline(always)]
fn dct1d_2(buf: &mut [f32]) {
    let a = buf[0];
    let b = buf[1];
    buf[0] = a + b;
    buf[1] = a - b;
}

#[allow(unused)]
#[inline(always)]
fn dct1d_4(buf: &mut [f32; 4]) {
    let mut tmp = [0.0f32; 4];
    tmp[0] = buf[0] + buf[3];
    tmp[1] = buf[1] + buf[2];
    dct1d_2(&mut tmp[0..2]);
    tmp[2] = buf[0] - buf[3];
    tmp[3] = buf[1] - buf[2];
    tmp[2] *= WC4[0];
    tmp[3] *= WC4[1];
    dct1d_2(&mut tmp[2..4]);
    tmp[2] = fmla(tmp[2], std::f32::consts::SQRT_2, tmp[3]);
    buf[0] = tmp[0];
    buf[2] = tmp[1];
    buf[1] = tmp[2];
    buf[3] = tmp[3];
}

#[inline(always)]
#[allow(unused)]
fn dct1d_8(buf: &mut [f32]) {
    let mut tmp = [0.0f32; 8];
    for i in 0..4 {
        tmp[i] = buf[i] + buf[7 - i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[..4]).unwrap());
    for i in 0..4 {
        tmp[4 + i] = (buf[i] - buf[7 - i]) * WC8[i];
    }
    dct1d_4(<&mut [f32; 4]>::try_from(&mut tmp[4..8]).unwrap());
    tmp[4] = fmla(tmp[4], std::f32::consts::SQRT_2, tmp[5]);
    tmp[5] += tmp[6];
    tmp[6] += tmp[7];
    for i in 0..4 {
        buf[2 * i] = tmp[i];
        buf[2 * i + 1] = tmp[4 + i];
    }
}

/// As [`forward_dct_quant_8x8`] but also returns the pre-round real targets
/// (`c/dq` per coefficient) for the trellis quantizer. `.0` is bit-identical to
/// the wrapper above.
pub fn forward_dct_quant_8x8_t(residual: &[i32; 64], q: &impl Dct) -> ([i32; 64], [f64; 64]) {
    let mut m = [[0.0f64; 8]; 8];
    for k in 0..8 {
        let s: f64 = ((if k == 0 { 0.5f64 } else { 1.0 }) * 2.0 / 8.0).sqrt();
        for n in 0..8 {
            m[k][n] = (std::f64::consts::PI * (2 * n + 1) as f64 * k as f64 / 16.0).cos() * s;
        }
    }
    let mut tmp = [[0.0f64; 8]; 8]; // tmp[v][x] = sum_y M[v][y] * R[y][x]
    for v in 0..8 {
        for x in 0..8 {
            let mut acc = 0.0;
            for y in 0..8 {
                acc += m[v][y] * residual[y * 8 + x] as f64;
            }
            tmp[v][x] = acc;
        }
    }
    let (dc_q, ac_q) = (q.dc_q() as f64, q.ac_q() as f64);
    let mut cf = [0i32; 64];
    let mut tf = [0.0f64; 64];
    for v in 0..8 {
        for u in 0..8 {
            let mut c = 0.0;
            for x in 0..8 {
                c += m[u][x] * tmp[v][x];
            }
            c *= 8.0;
            let dq = if v == 0 && u == 0 { dc_q } else { ac_q };
            let q = c / dq;
            tf[u * 8 + v] = q;
            cf[u * 8 + v] = q.round() as i32;
        }
    }
    (cf, tf)
}
