/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

use crate::av2::itx::{
    ADST4_KERNEL, ADST8_KERNEL, ADST16_KERNEL, DCT8_KERNEL, DCT16_DENSE_KERNEL, DCT32_DENSE_KERNEL,
    DDT8_KERNEL, DDT16_KERNEL, DIM, FLIPADST4_KERNEL, FLIPADST16_KERNEL, TXSH,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn zero() -> __m256i {
    _mm256_setzero_si256()
}

#[inline]
#[target_feature(enable = "avx2")]
fn add(a: __m256i, b: __m256i) -> __m256i {
    _mm256_add_epi32(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
fn sub(a: __m256i, b: __m256i) -> __m256i {
    _mm256_sub_epi32(a, b)
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_n(a: __m256i, k: i32) -> __m256i {
    if k == 0 {
        zero()
    } else {
        _mm256_mullo_epi32(a, _mm256_set1_epi32(k))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn mul_add_n(acc: __m256i, x: __m256i, k: i32) -> __m256i {
    if k == 0 { acc } else { add(acc, mul_n(x, k)) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn from_array(a: [i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(a.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn to_array(v: __m256i) -> [i32; 8] {
    let mut a = [0i32; 8];
    unsafe { _mm256_storeu_si256(a.as_mut_ptr().cast::<__m256i>(), v) };
    a
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_row_lanes(tmp: &[i32], sw: usize, row: usize, x: usize, lanes: usize) -> __m256i {
    debug_assert!(lanes == 4 || lanes == 8);
    let mut a = [0i32; 8];
    for lane in 0..lanes {
        a[lane] = tmp[(row + lane) * sw + x];
    }
    from_array(a)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_row_lanes(tmp: &mut [i32], sw: usize, row: usize, x: usize, lanes: usize, v: __m256i) {
    debug_assert!(lanes == 4 || lanes == 8);
    let a = to_array(v);
    for lane in 0..lanes {
        tmp[(row + lane) * sw + x] = a[lane];
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_col_lanes(row: &[i32], x: usize, lanes: usize) -> __m256i {
    debug_assert!(lanes == 4 || lanes == 8);
    if lanes == 8 {
        unsafe { _mm256_loadu_si256(row.as_ptr().add(x).cast::<__m256i>()) }
    } else {
        let mut a = [0i32; 8];
        a[..4].copy_from_slice(&row[x..x + 4]);
        from_array(a)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_col_lanes(row: &mut [i32], x: usize, lanes: usize, v: __m256i) {
    debug_assert!(lanes == 4 || lanes == 8);
    if lanes == 8 {
        unsafe { _mm256_storeu_si256(row.as_mut_ptr().add(x).cast::<__m256i>(), v) };
    } else {
        let a = to_array(v);
        row[x..x + 4].copy_from_slice(&a[..4]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dot(mat: &[i8], v: &[__m256i]) -> __m256i {
    debug_assert!(!mat.is_empty() && mat.len() <= v.len());
    let mut acc = mul_n(v[0], mat[0] as i32);
    for i in 1..mat.len() {
        acc = mul_add_n(acc, v[i], mat[i] as i32);
    }
    acc
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct4(c: &mut [__m256i]) {
    debug_assert!(c.len() >= 4);
    let c0 = c[0];
    let c1 = c[1];
    let c2 = c[2];
    let c3 = c[3];
    let a0 = add(mul_n(c0, 64), mul_n(c2, 64));
    let a1 = sub(mul_n(c0, 64), mul_n(c2, 64));
    let b0 = add(mul_n(c1, 83), mul_n(c3, 35));
    let b1 = sub(mul_n(c1, 35), mul_n(c3, 83));
    c[0] = add(a0, b0);
    c[1] = add(a1, b1);
    c[2] = sub(a1, b1);
    c[3] = sub(a0, b0);
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct8(c: &mut [__m256i]) {
    debug_assert!(c.len() >= 8);
    let mut even = [zero(); 8];
    let mut odd = [zero(); 8];
    for i in 0..4 {
        even[i] = c[2 * i];
        odd[i] = c[2 * i + 1];
    }
    dct4(&mut even[..4]);
    for i in 0..4 {
        let b = dot(&DCT8_KERNEL[i * 4..i * 4 + 4], &odd[..4]);
        c[i] = add(even[i], b);
        c[7 - i] = sub(even[i], b);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct16(c: &mut [__m256i]) {
    debug_assert!(c.len() >= 16);
    let mut s = [zero(); 16];
    s.copy_from_slice(&c[..16]);
    let k = |j: usize, m: usize| DCT16_DENSE_KERNEL[j * 16 + m] as i32;

    let mut b = [zero(); 8];
    for m in 0..8 {
        let mut acc = zero();
        let mut j = 1;
        while j < 16 {
            acc = mul_add_n(acc, s[j], k(j, m));
            j += 2;
        }
        b[m] = acc;
    }

    let mut d = [zero(); 4];
    for m in 0..4 {
        let mut acc = zero();
        let mut j = 2;
        while j < 16 {
            acc = mul_add_n(acc, s[j], k(j, m));
            j += 4;
        }
        d[m] = acc;
    }

    let f = [
        mul_add_n(mul_n(s[4], k(4, 0)), s[12], k(12, 0)),
        mul_add_n(mul_n(s[4], k(4, 1)), s[12], k(12, 1)),
    ];
    let g = [
        mul_add_n(mul_n(s[0], k(0, 0)), s[8], k(8, 0)),
        mul_add_n(mul_n(s[0], k(0, 1)), s[8], k(8, 1)),
    ];
    let mut cc = [zero(); 4];
    for kk in 0..2 {
        cc[kk] = add(g[kk], f[kk]);
        cc[kk + 2] = sub(g[1 - kk], f[1 - kk]);
    }
    let mut a = [zero(); 8];
    for kk in 0..4 {
        a[kk] = add(cc[kk], d[kk]);
        a[kk + 4] = sub(cc[3 - kk], d[3 - kk]);
    }
    for kk in 0..8 {
        c[kk] = add(a[kk], b[kk]);
        c[kk + 8] = sub(a[7 - kk], b[7 - kk]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dct32(c: &mut [__m256i]) {
    debug_assert!(c.len() >= 32);
    let mut s = [zero(); 32];
    s.copy_from_slice(&c[..32]);
    let k = |j: usize, m: usize| DCT32_DENSE_KERNEL[j * 32 + m] as i32;

    let mut b = [zero(); 16];
    for m in 0..16 {
        let mut acc = zero();
        let mut j = 1;
        while j < 32 {
            acc = mul_add_n(acc, s[j], k(j, m));
            j += 2;
        }
        b[m] = acc;
    }

    let mut d = [zero(); 8];
    for m in 0..8 {
        let mut acc = zero();
        let mut j = 2;
        while j < 32 {
            acc = mul_add_n(acc, s[j], k(j, m));
            j += 4;
        }
        d[m] = acc;
    }

    let mut f = [zero(); 4];
    for m in 0..4 {
        f[m] = mul_add_n(
            mul_add_n(
                mul_add_n(mul_n(s[4], k(4, m)), s[12], k(12, m)),
                s[20],
                k(20, m),
            ),
            s[28],
            k(28, m),
        );
    }
    let h = [
        mul_add_n(mul_n(s[8], k(8, 0)), s[24], k(24, 0)),
        mul_add_n(mul_n(s[8], k(8, 1)), s[24], k(24, 1)),
    ];
    let g = [
        mul_add_n(mul_n(s[0], k(0, 0)), s[16], k(16, 0)),
        mul_add_n(mul_n(s[0], k(0, 1)), s[16], k(16, 1)),
    ];
    let e = [
        add(g[0], h[0]),
        add(g[1], h[1]),
        sub(g[1], h[1]),
        sub(g[0], h[0]),
    ];
    let mut cc = [zero(); 8];
    for kk in 0..4 {
        cc[kk] = add(e[kk], f[kk]);
        cc[kk + 4] = sub(e[3 - kk], f[3 - kk]);
    }
    let mut a = [zero(); 16];
    for kk in 0..8 {
        a[kk] = add(cc[kk], d[kk]);
        a[kk + 8] = sub(cc[7 - kk], d[7 - kk]);
    }
    for kk in 0..16 {
        c[kk] = add(a[kk], b[kk]);
        c[kk + 16] = sub(a[15 - kk], b[15 - kk]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dst(c: &mut [__m256i], mat: &[i8], flip: bool) {
    let n = c.len();
    debug_assert!(n <= 16);
    let mut sums = [zero(); 16];
    for i in 0..n {
        sums[i] = dot(&mat[i * n..i * n + n], c);
    }
    if flip {
        for i in 0..n {
            c[n - 1 - i] = sums[i];
        }
    } else {
        c[..n].copy_from_slice(&sums[..n]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn id4(c: &mut [__m256i]) {
    for v in c.iter_mut() {
        *v = mul_n(*v, 128);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn id8(c: &mut [__m256i]) {
    for v in c.iter_mut() {
        *v = mul_n(*v, 181);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn id16(c: &mut [__m256i]) {
    for v in c.iter_mut() {
        *v = mul_n(*v, 256);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn id32(c: &mut [__m256i]) {
    for v in c.iter_mut() {
        *v = mul_n(*v, 362);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn adst4(c: &mut [__m256i]) {
    dst(c, &ADST4_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn adst8(c: &mut [__m256i]) {
    dst(c, &ADST8_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn adst16(c: &mut [__m256i]) {
    dst(c, &ADST16_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn flipadst4(c: &mut [__m256i]) {
    dst(c, &FLIPADST4_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn flipadst8(c: &mut [__m256i]) {
    dst(c, &ADST8_KERNEL, true);
}
#[inline]
#[target_feature(enable = "avx2")]
fn flipadst16(c: &mut [__m256i]) {
    dst(c, &FLIPADST16_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn ddt8(c: &mut [__m256i]) {
    dst(c, &DDT8_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn ddt16(c: &mut [__m256i]) {
    dst(c, &DDT16_KERNEL, false);
}
#[inline]
#[target_feature(enable = "avx2")]
fn flipddt8(c: &mut [__m256i]) {
    dst(c, &DDT8_KERNEL, true);
}
#[inline]
#[target_feature(enable = "avx2")]
fn flipddt16(c: &mut [__m256i]) {
    dst(c, &DDT16_KERNEL, true);
}

#[inline]
#[target_feature(enable = "avx2")]
fn transform_1d(sz: usize, ty: usize, c: &mut [__m256i]) -> bool {
    match (sz, ty) {
        (0, 0) => dct4(c),
        (1, 0) => dct8(c),
        (2, 0) => dct16(c),
        (3, 0) | (4, 0) => dct32(c),
        (0, 1) => id4(c),
        (1, 1) => id8(c),
        (2, 1) => id16(c),
        (3, 1) => id32(c),
        (0, 2) => adst4(c),
        (1, 2) => adst8(c),
        (2, 2) => adst16(c),
        (0, 3) => flipadst4(c),
        (1, 3) => flipadst8(c),
        (2, 3) => flipadst16(c),
        (1, 4) => ddt8(c),
        (2, 4) => ddt16(c),
        (1, 5) => flipddt8(c),
        (2, 5) => flipddt16(c),
        _ => return false,
    }
    true
}

#[inline]
#[target_feature(enable = "avx2")]
fn lane_group(n: usize) -> usize {
    debug_assert!(n == 4 || n == 8 || n == 16 || n == 32);
    if n >= 8 { 8 } else { 4 }
}

#[target_feature(enable = "avx2")]
pub(crate) fn inv_txfm_passes_avx2(
    tmp_arr: &mut [i32; 32 * 32],
    coeff: &[i32],
    txtp: usize,
    tx: usize,
    bd: i32,
) -> (usize, usize, usize, usize, i32) {
    debug_assert!(tx < DIM.len());
    debug_assert!((1..=16).contains(&bd));

    let (tw, th, lw, lh) = DIM[tx];
    let (s0, s1) = TXSH[tx];
    let (w, h) = (4 * tw, 4 * th);
    let is_rect2 = (lw + lh) & 1 != 0;
    let coef_clip_min = -(1 << (bd + 7));
    let coef_clip_max = (1 << (bd + 7)) - 1;
    let row_clip_min = coef_clip_min;
    let row_clip_max = coef_clip_max;
    let hor_ty = txtp & 7;
    let ver_ty = (txtp >> 5) & 7;
    let (sw, sh) = (w.min(32), h.min(32));
    let area = sw * sh;
    debug_assert!(area <= tmp_arr.len());
    debug_assert!(coeff.len() >= area);
    debug_assert!(sw == 4 || sw == 8 || sw == 16 || sw == 32);
    debug_assert!(sh == 4 || sh == 8 || sh == 16 || sh == 32);

    let coeff = &coeff[..area];
    let tmp = &mut tmp_arr[..area];

    for (col, row) in tmp.chunks_exact_mut(sw).enumerate() {
        for (slot, &v) in row.iter_mut().zip(coeff.iter().skip(col).step_by(sh)) {
            let v = if is_rect2 { (v * 181 + 128) >> 8 } else { v };
            *slot = v.clamp(coef_clip_min, coef_clip_max);
        }
    }

    let mut lanes = [zero(); 32];
    let row_lanes = lane_group(sh);
    for row in (0..sh).step_by(row_lanes) {
        for x in 0..sw {
            lanes[x] = load_row_lanes(tmp, sw, row, x, row_lanes);
        }
        assert!(
            transform_1d(lw, hor_ty, &mut lanes[..sw]),
            "invalid horizontal tx"
        );
        for x in 0..sw {
            store_row_lanes(tmp, sw, row, x, row_lanes, lanes[x]);
        }
    }

    let rnd0 = (1 << s0) >> 1;
    for v in tmp.iter_mut() {
        *v = ((*v + rnd0) >> s0).clamp(row_clip_min, row_clip_max);
    }

    let col_lanes = lane_group(sw);
    for x in (0..sw).step_by(col_lanes) {
        for (lane, row) in lanes[..sh].iter_mut().zip(tmp.chunks_exact(sw)) {
            *lane = load_col_lanes(row, x, col_lanes);
        }
        assert!(
            transform_1d(lh, ver_ty, &mut lanes[..sh]),
            "invalid vertical tx"
        );
        for (lane, row) in lanes[..sh].iter().zip(tmp.chunks_exact_mut(sw)) {
            store_col_lanes(row, x, col_lanes, *lane);
        }
    }

    (sw, sh, w, h, s1)
}
