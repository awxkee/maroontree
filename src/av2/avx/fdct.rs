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

use crate::av2::fdct::{
    DCT4_FWD_KERNEL, DCT8_FWD_KERNEL, DCT16_FWD_KERNEL, DCT32_FWD_KERNEL, DCT64_FWD_KERNEL,
    fwd_shifts,
};
use std::cell::RefCell;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

struct FdctScratch {
    buf: Box<[i32; 4096]>,
    coeff: Box<[i32; 4096]>,
}

thread_local! {
    static FDCT_SCRATCH: RefCell<FdctScratch> = RefCell::new(FdctScratch {
        buf: Box::new([0; 4096]),
        coeff: Box::new([0; 4096]),
    });
}

#[inline]
#[target_feature(enable = "avx2")]
fn round_shift(v: __m256i, shift: i32) -> __m256i {
    if shift <= 0 {
        v
    } else {
        let add = _mm256_set1_epi32(1i32 << (shift - 1));
        let count = _mm256_set1_epi32(shift);
        _mm256_srav_epi32(_mm256_add_epi32(v, add), count)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load8(src: &[i32], line: usize, j: usize, k: usize) -> __m256i {
    unsafe { _mm256_loadu_si256(src.as_ptr().add(k * line + j).cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store8(dst: &mut [i32], n: usize, j: usize, k: usize, v: __m256i) {
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), v) };
    for lane in 0..8 {
        dst[(j + lane) * n + k] = lanes[lane];
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn madd_n(acc: __m256i, v: __m256i, c: i32) -> __m256i {
    if c == 0 {
        acc
    } else {
        _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(c)))
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn dot_i8<const N: usize>(v: &[__m256i; N], kernel: &[i8], row: usize, width: usize) -> __m256i {
    let mut acc = _mm256_setzero_si256();
    let row = &kernel[row * width..row * width + N];
    for i in 0..N {
        acc = madd_n(acc, v[i], row[i] as i32);
    }
    acc
}

#[inline]
#[target_feature(enable = "avx2")]
#[allow(clippy::too_many_arguments)]
fn store_dot<const N: usize>(
    dst: &mut [i32],
    stride: usize,
    j: usize,
    row: usize,
    shift: i32,
    v: &[__m256i; N],
    kernel: &[i8],
    width: usize,
) {
    let acc = dot_i8(v, kernel, row, width);
    let acc = round_shift(acc, shift);
    store8(dst, stride, j, row, acc);
}

#[inline]
fn round_shift_scalar(v: i32, shift: i32) -> i32 {
    if shift <= 0 {
        v
    } else {
        (v + (1 << (shift - 1))) >> shift
    }
}

#[inline]
fn dot_i8_scalar(v: &[i32], kernel: &[i8], row: usize, width: usize) -> i32 {
    let row = &kernel[row * width..row * width + v.len()];
    v.iter().zip(row.iter()).map(|(&x, &c)| x * c as i32).sum()
}

#[inline]
#[allow(clippy::too_many_arguments)]
fn store_dot_scalar(
    dst: &mut [i32],
    stride: usize,
    j: usize,
    row: usize,
    shift: i32,
    v: &[i32],
    kernel: &[i8],
    width: usize,
) {
    dst[j * stride + row] = round_shift_scalar(dot_i8_scalar(v, kernel, row, width), shift);
}

fn fdct4_1d_scalar_one(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s = |k: usize| src[k * line + j];
    let a = [s(0) + s(3), s(1) + s(2)];
    let b = [s(0) - s(3), s(1) - s(2)];
    store_dot_scalar(dst, 4, j, 0, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot_scalar(dst, 4, j, 2, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot_scalar(dst, 4, j, 1, shift, &b, &DCT4_FWD_KERNEL, 4);
    store_dot_scalar(dst, 4, j, 3, shift, &b, &DCT4_FWD_KERNEL, 4);
}

fn fdct8_1d_scalar_one(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s = |k: usize| src[k * line + j];
    let mut a = [0i32; 4];
    let mut b = [0i32; 4];
    for k in 0..4 {
        a[k] = s(k) + s(7 - k);
        b[k] = s(k) - s(7 - k);
    }
    let c = [a[0] + a[3], a[1] + a[2]];
    let d = [a[0] - a[3], a[1] - a[2]];
    store_dot_scalar(dst, 8, j, 0, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot_scalar(dst, 8, j, 4, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot_scalar(dst, 8, j, 2, shift, &d, &DCT8_FWD_KERNEL, 8);
    store_dot_scalar(dst, 8, j, 6, shift, &d, &DCT8_FWD_KERNEL, 8);
    let mut row = 1;
    while row < 8 {
        store_dot_scalar(dst, 8, j, row, shift, &b, &DCT8_FWD_KERNEL, 8);
        row += 2;
    }
}

fn fdct16_1d_scalar_one(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s = |k: usize| src[k * line + j];
    let mut a = [0i32; 8];
    let mut b = [0i32; 8];
    for k in 0..8 {
        a[k] = s(k) + s(15 - k);
        b[k] = s(k) - s(15 - k);
    }
    let mut c = [0i32; 4];
    let mut d = [0i32; 4];
    for k in 0..4 {
        c[k] = a[k] + a[7 - k];
        d[k] = a[k] - a[7 - k];
    }
    let e = [c[0] + c[3], c[1] + c[2]];
    let f = [c[0] - c[3], c[1] - c[2]];
    store_dot_scalar(dst, 16, j, 0, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot_scalar(dst, 16, j, 8, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot_scalar(dst, 16, j, 4, shift, &f, &DCT16_FWD_KERNEL, 16);
    store_dot_scalar(dst, 16, j, 12, shift, &f, &DCT16_FWD_KERNEL, 16);
    let mut row = 2;
    while row < 16 {
        store_dot_scalar(dst, 16, j, row, shift, &d, &DCT16_FWD_KERNEL, 16);
        row += 4;
    }
    let mut row = 1;
    while row < 16 {
        store_dot_scalar(dst, 16, j, row, shift, &b, &DCT16_FWD_KERNEL, 16);
        row += 2;
    }
}

fn fdct32_1d_scalar_one(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s = |k: usize| src[k * line + j];
    let mut a = [0i32; 16];
    let mut b = [0i32; 16];
    for k in 0..16 {
        a[k] = s(k) + s(31 - k);
        b[k] = s(k) - s(31 - k);
    }
    let mut c = [0i32; 8];
    let mut d = [0i32; 8];
    for k in 0..8 {
        c[k] = a[k] + a[15 - k];
        d[k] = a[k] - a[15 - k];
    }
    let mut e = [0i32; 4];
    let mut f = [0i32; 4];
    for k in 0..4 {
        e[k] = c[k] + c[7 - k];
        f[k] = c[k] - c[7 - k];
    }
    let g = [e[0] + e[3], e[1] + e[2]];
    let h = [e[0] - e[3], e[1] - e[2]];
    store_dot_scalar(dst, 32, j, 0, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot_scalar(dst, 32, j, 16, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot_scalar(dst, 32, j, 8, shift, &h, &DCT32_FWD_KERNEL, 32);
    store_dot_scalar(dst, 32, j, 24, shift, &h, &DCT32_FWD_KERNEL, 32);
    let mut row = 4;
    while row < 32 {
        store_dot_scalar(dst, 32, j, row, shift, &f, &DCT32_FWD_KERNEL, 32);
        row += 8;
    }
    let mut row = 2;
    while row < 32 {
        store_dot_scalar(dst, 32, j, row, shift, &d, &DCT32_FWD_KERNEL, 32);
        row += 4;
    }
    let mut row = 1;
    while row < 32 {
        store_dot_scalar(dst, 32, j, row, shift, &b, &DCT32_FWD_KERNEL, 32);
        row += 2;
    }
}

fn fdct64_1d_scalar_one(
    src: &[i32],
    dst: &mut [i32],
    shift: i32,
    line: usize,
    zero_line: usize,
    j: usize,
) {
    let top = if zero_line != 0 { 32 } else { 64 };
    let s = |k: usize| src[k * line + j];
    let mut a = [0i32; 32];
    let mut b = [0i32; 32];
    for k in 0..32 {
        a[k] = s(k) + s(63 - k);
        b[k] = s(k) - s(63 - k);
    }
    let mut c = [0i32; 16];
    let mut d = [0i32; 16];
    for k in 0..16 {
        c[k] = a[k] + a[31 - k];
        d[k] = a[k] - a[31 - k];
    }
    let mut e = [0i32; 8];
    let mut f = [0i32; 8];
    for k in 0..8 {
        e[k] = c[k] + c[15 - k];
        f[k] = c[k] - c[15 - k];
    }
    let mut g = [0i32; 4];
    let mut h = [0i32; 4];
    for k in 0..4 {
        g[k] = e[k] + e[7 - k];
        h[k] = e[k] - e[7 - k];
    }
    let i0 = [g[0] + g[3], g[1] + g[2]];
    let u0 = [g[0] - g[3], g[1] - g[2]];
    store_dot_scalar(dst, 64, j, 0, shift, &i0, &DCT64_FWD_KERNEL, 64);
    store_dot_scalar(dst, 64, j, 16, shift, &u0, &DCT64_FWD_KERNEL, 64);
    if top == 64 {
        store_dot_scalar(dst, 64, j, 32, shift, &i0, &DCT64_FWD_KERNEL, 64);
        store_dot_scalar(dst, 64, j, 48, shift, &u0, &DCT64_FWD_KERNEL, 64);
    }
    let mut row = 8;
    while row < top {
        store_dot_scalar(dst, 64, j, row, shift, &h, &DCT64_FWD_KERNEL, 64);
        row += 16;
    }
    let mut row = 4;
    while row < top {
        store_dot_scalar(dst, 64, j, row, shift, &f, &DCT64_FWD_KERNEL, 64);
        row += 8;
    }
    let mut row = 2;
    while row < top {
        store_dot_scalar(dst, 64, j, row, shift, &d, &DCT64_FWD_KERNEL, 64);
        row += 4;
    }
    let mut row = 1;
    while row < top {
        store_dot_scalar(dst, 64, j, row, shift, &b, &DCT64_FWD_KERNEL, 64);
        row += 2;
    }
}

#[target_feature(enable = "avx2")]
fn fdct4_1d_8(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s0 = unsafe { load8(src, line, j, 0) };
    let s1 = unsafe { load8(src, line, j, 1) };
    let s2 = unsafe { load8(src, line, j, 2) };
    let s3 = unsafe { load8(src, line, j, 3) };

    let a = [_mm256_add_epi32(s0, s3), _mm256_add_epi32(s1, s2)];
    let b = [_mm256_sub_epi32(s0, s3), _mm256_sub_epi32(s1, s2)];

    store_dot(dst, 4, j, 0, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 2, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 1, shift, &b, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 3, shift, &b, &DCT4_FWD_KERNEL, 4);
}

#[target_feature(enable = "avx2")]
fn fdct8_1d_8(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let z = _mm256_setzero_si256();
    let mut a = [z; 4];
    let mut b = [z; 4];
    for k in 0..4 {
        let lo = unsafe { load8(src, line, j, k) };
        let hi = unsafe { load8(src, line, j, 7 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let c = [_mm256_add_epi32(a[0], a[3]), _mm256_add_epi32(a[1], a[2])];
    let d = [_mm256_sub_epi32(a[0], a[3]), _mm256_sub_epi32(a[1], a[2])];

    store_dot(dst, 8, j, 0, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 4, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 2, shift, &d, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 6, shift, &d, &DCT8_FWD_KERNEL, 8);
    let mut row = 1;
    while row < 8 {
        store_dot(dst, 8, j, row, shift, &b, &DCT8_FWD_KERNEL, 8);
        row += 2;
    }
}

#[target_feature(enable = "avx2")]
fn fdct16_1d_8(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let z = _mm256_setzero_si256();
    let mut a = [z; 8];
    let mut b = [z; 8];
    for k in 0..8 {
        let lo = unsafe { load8(src, line, j, k) };
        let hi = unsafe { load8(src, line, j, 15 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let z = _mm256_setzero_si256();
    let mut c = [z; 4];
    let mut d = [z; 4];
    for k in 0..4 {
        c[k] = _mm256_add_epi32(a[k], a[7 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[7 - k]);
    }
    let e = [_mm256_add_epi32(c[0], c[3]), _mm256_add_epi32(c[1], c[2])];
    let f = [_mm256_sub_epi32(c[0], c[3]), _mm256_sub_epi32(c[1], c[2])];

    store_dot(dst, 16, j, 0, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 8, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 4, shift, &f, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 12, shift, &f, &DCT16_FWD_KERNEL, 16);
    let mut row = 2;
    while row < 16 {
        store_dot(dst, 16, j, row, shift, &d, &DCT16_FWD_KERNEL, 16);
        row += 4;
    }
    let mut row = 1;
    while row < 16 {
        store_dot(dst, 16, j, row, shift, &b, &DCT16_FWD_KERNEL, 16);
        row += 2;
    }
}

#[target_feature(enable = "avx2")]
fn fdct32_1d_8(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let z = _mm256_setzero_si256();
    let mut a = [z; 16];
    let mut b = [z; 16];
    for k in 0..16 {
        let lo = unsafe { load8(src, line, j, k) };
        let hi = unsafe { load8(src, line, j, 31 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let z = _mm256_setzero_si256();
    let mut c = [z; 8];
    let mut d = [z; 8];
    for k in 0..8 {
        c[k] = _mm256_add_epi32(a[k], a[15 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[15 - k]);
    }
    let z = _mm256_setzero_si256();
    let mut e = [z; 4];
    let mut f = [z; 4];
    for k in 0..4 {
        e[k] = _mm256_add_epi32(c[k], c[7 - k]);
        f[k] = _mm256_sub_epi32(c[k], c[7 - k]);
    }
    let g = [_mm256_add_epi32(e[0], e[3]), _mm256_add_epi32(e[1], e[2])];
    let h = [_mm256_sub_epi32(e[0], e[3]), _mm256_sub_epi32(e[1], e[2])];

    store_dot(dst, 32, j, 0, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 16, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 8, shift, &h, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 24, shift, &h, &DCT32_FWD_KERNEL, 32);
    let mut row = 4;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &f, &DCT32_FWD_KERNEL, 32);
        row += 8;
    }
    let mut row = 2;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &d, &DCT32_FWD_KERNEL, 32);
        row += 4;
    }
    let mut row = 1;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &b, &DCT32_FWD_KERNEL, 32);
        row += 2;
    }
}

#[target_feature(enable = "avx2")]
fn fdct64_1d_8(src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero_line: usize, j: usize) {
    let top = if zero_line != 0 { 32 } else { 64 };
    let z = _mm256_setzero_si256();
    let mut a = [z; 32];
    let mut b = [z; 32];
    for k in 0..32 {
        let lo = unsafe { load8(src, line, j, k) };
        let hi = unsafe { load8(src, line, j, 63 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let z = _mm256_setzero_si256();
    let mut c = [z; 16];
    let mut d = [z; 16];
    for k in 0..16 {
        c[k] = _mm256_add_epi32(a[k], a[31 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[31 - k]);
    }
    let z = _mm256_setzero_si256();
    let mut e = [z; 8];
    let mut f = [z; 8];
    for k in 0..8 {
        e[k] = _mm256_add_epi32(c[k], c[15 - k]);
        f[k] = _mm256_sub_epi32(c[k], c[15 - k]);
    }
    let z = _mm256_setzero_si256();
    let mut g = [z; 4];
    let mut h = [z; 4];
    for k in 0..4 {
        g[k] = _mm256_add_epi32(e[k], e[7 - k]);
        h[k] = _mm256_sub_epi32(e[k], e[7 - k]);
    }
    let i0 = [_mm256_add_epi32(g[0], g[3]), _mm256_add_epi32(g[1], g[2])];
    let u0 = [_mm256_sub_epi32(g[0], g[3]), _mm256_sub_epi32(g[1], g[2])];

    store_dot(dst, 64, j, 0, shift, &i0, &DCT64_FWD_KERNEL, 64);
    store_dot(dst, 64, j, 16, shift, &u0, &DCT64_FWD_KERNEL, 64);
    if top == 64 {
        store_dot(dst, 64, j, 32, shift, &i0, &DCT64_FWD_KERNEL, 64);
        store_dot(dst, 64, j, 48, shift, &u0, &DCT64_FWD_KERNEL, 64);
    }
    let mut row = 8;
    while row < top {
        store_dot(dst, 64, j, row, shift, &h, &DCT64_FWD_KERNEL, 64);
        row += 16;
    }
    let mut row = 4;
    while row < top {
        store_dot(dst, 64, j, row, shift, &f, &DCT64_FWD_KERNEL, 64);
        row += 8;
    }
    let mut row = 2;
    while row < top {
        store_dot(dst, 64, j, row, shift, &d, &DCT64_FWD_KERNEL, 64);
        row += 4;
    }
    let mut row = 1;
    while row < top {
        store_dot(dst, 64, j, row, shift, &b, &DCT64_FWD_KERNEL, 64);
        row += 2;
    }
}

#[target_feature(enable = "avx2")]
fn fdct_1d_n(n: usize, src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero: usize) {
    let mut j = 0usize;
    while j + 8 <= line {
        match n {
            4 => fdct4_1d_8(src, dst, shift, line, j),
            8 => fdct8_1d_8(src, dst, shift, line, j),
            16 => fdct16_1d_8(src, dst, shift, line, j),
            32 => fdct32_1d_8(src, dst, shift, line, j),
            64 => fdct64_1d_8(src, dst, shift, line, zero, j),
            _ => unreachable!("unsupported 1D size {n}"),
        }
        j += 8;
    }
    while j < line {
        match n {
            4 => fdct4_1d_scalar_one(src, dst, shift, line, j),
            8 => fdct8_1d_scalar_one(src, dst, shift, line, j),
            16 => fdct16_1d_scalar_one(src, dst, shift, line, j),
            32 => fdct32_1d_scalar_one(src, dst, shift, line, j),
            64 => fdct64_1d_scalar_one(src, dst, shift, line, zero, j),
            _ => unreachable!("unsupported 1D size {n}"),
        }
        j += 1;
    }
}

#[inline]
fn scale_rect2_in_place(out: &mut [i32]) {
    for v in out.iter_mut() {
        *v = (((*v as i64) * 5793 + 2048) >> 12) as i32;
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn fdct_rect_avx2(resid: &[i32], w: usize, h: usize, out: &mut [i32]) -> usize {
    let (s1, s2) = fwd_shifts(w, h);
    let zh = if h > 32 { 32 } else { 0 };
    let zw = if w > 32 { 32 } else { 0 };
    let (cw, ch) = (w.min(32), h.min(32));

    FDCT_SCRATCH.with(|cell| {
        let s = &mut *cell.borrow_mut();
        fdct_1d_n(h, resid, &mut s.buf[..w * h], s1, w, zh);
        fdct_1d_n(w, &s.buf[..w * h], &mut s.coeff[..w * h], s2, h, zw);
        for vf in 0..ch {
            let src = &s.coeff[vf * w..vf * w + cw];
            let dst = &mut out[vf * cw..vf * cw + cw];
            dst.copy_from_slice(src);
        }
    });

    let n = cw * ch;
    if (w.trailing_zeros() + h.trailing_zeros()) & 1 == 1 {
        scale_rect2_in_place(&mut out[..n]);
    }
    n
}
