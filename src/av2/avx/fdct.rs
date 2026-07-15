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

/// Load one sample from each of `LANES` adjacent transform lines.
///
/// AV2 transform dimensions are multiples of four, so an AVX2 group is either
/// eight full lanes or a final four-lane group. Masked loads keep the latter in
/// the same SIMD pipeline without reading beyond the row.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load8<const LANES: usize>(src: &[i32], line: usize, j: usize, k: usize) -> __m256i {
    let ptr = unsafe { src.as_ptr().add(k * line + j) };
    if LANES == 8 {
        unsafe { _mm256_loadu_si256(ptr.cast::<__m256i>()) }
    } else {
        debug_assert_eq!(LANES, 4);
        let mask = _mm256_set_epi32(0, 0, 0, 0, -1, -1, -1, -1);
        unsafe { _mm256_maskload_epi32(ptr, mask) }
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
fn dot_shift<const N: usize>(
    shift: i32,
    v: &[__m256i; N],
    kernel: &[i8],
    row: usize,
    width: usize,
) -> __m256i {
    round_shift(dot_i8(v, kernel, row, width), shift)
}

/// Transpose eight frequency vectors whose lanes hold eight adjacent transform
/// lines. Afterwards each vector is one complete contiguous 8-coefficient row.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_i32(c: &mut [__m256i; 8]) {
    let t0 = _mm256_unpacklo_epi32(c[0], c[1]);
    let t1 = _mm256_unpackhi_epi32(c[0], c[1]);
    let t2 = _mm256_unpacklo_epi32(c[2], c[3]);
    let t3 = _mm256_unpackhi_epi32(c[2], c[3]);
    let t4 = _mm256_unpacklo_epi32(c[4], c[5]);
    let t5 = _mm256_unpackhi_epi32(c[4], c[5]);
    let t6 = _mm256_unpacklo_epi32(c[6], c[7]);
    let t7 = _mm256_unpackhi_epi32(c[6], c[7]);

    let u0 = _mm256_unpacklo_epi64(t0, t2);
    let u1 = _mm256_unpackhi_epi64(t0, t2);
    let u2 = _mm256_unpacklo_epi64(t1, t3);
    let u3 = _mm256_unpackhi_epi64(t1, t3);
    let u4 = _mm256_unpacklo_epi64(t4, t6);
    let u5 = _mm256_unpackhi_epi64(t4, t6);
    let u6 = _mm256_unpacklo_epi64(t5, t7);
    let u7 = _mm256_unpackhi_epi64(t5, t7);

    c[0] = _mm256_permute2x128_si256::<0x20>(u0, u4);
    c[1] = _mm256_permute2x128_si256::<0x20>(u1, u5);
    c[2] = _mm256_permute2x128_si256::<0x20>(u2, u6);
    c[3] = _mm256_permute2x128_si256::<0x20>(u3, u7);
    c[4] = _mm256_permute2x128_si256::<0x31>(u0, u4);
    c[5] = _mm256_permute2x128_si256::<0x31>(u1, u5);
    c[6] = _mm256_permute2x128_si256::<0x31>(u2, u6);
    c[7] = _mm256_permute2x128_si256::<0x31>(u3, u7);
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_i32(r: [__m128i; 4]) -> [__m128i; 4] {
    let t0 = _mm_unpacklo_epi32(r[0], r[1]);
    let t1 = _mm_unpackhi_epi32(r[0], r[1]);
    let t2 = _mm_unpacklo_epi32(r[2], r[3]);
    let t3 = _mm_unpackhi_epi32(r[2], r[3]);
    [
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    ]
}

/// Store a complete SIMD batch after a register transpose. No coefficient is
/// spilled and scattered: every memory write is a contiguous 128- or 256-bit
/// store into the normal transposed scratch layout expected by the next pass.
#[inline]
#[target_feature(enable = "avx2")]
fn transpose_store<const N: usize, const LANES: usize>(
    dst: &mut [i32],
    j: usize,
    coeff: &[__m256i; N],
    valid: usize,
) {
    debug_assert!(valid <= N && valid.is_multiple_of(4));
    let dst = dst.as_mut_ptr();

    if LANES == 8 && valid >= 8 {
        let mut freq = 0usize;
        while freq + 8 <= valid {
            let mut tile = [
                coeff[freq],
                coeff[freq + 1],
                coeff[freq + 2],
                coeff[freq + 3],
                coeff[freq + 4],
                coeff[freq + 5],
                coeff[freq + 6],
                coeff[freq + 7],
            ];
            transpose_8x8_i32(&mut tile);
            for lane in 0..8 {
                unsafe {
                    _mm256_storeu_si256(
                        dst.add((j + lane) * N + freq).cast::<__m256i>(),
                        tile[lane],
                    );
                }
            }
            freq += 8;
        }
        debug_assert_eq!(freq, valid);
    } else {
        debug_assert!(LANES == 4 || (LANES == 8 && N == 4));
        let mut freq = 0usize;
        while freq < valid {
            let lo = transpose_4x4_i32([
                _mm256_castsi256_si128(coeff[freq]),
                _mm256_castsi256_si128(coeff[freq + 1]),
                _mm256_castsi256_si128(coeff[freq + 2]),
                _mm256_castsi256_si128(coeff[freq + 3]),
            ]);
            for lane in 0..4 {
                unsafe {
                    _mm_storeu_si128(dst.add((j + lane) * N + freq).cast::<__m128i>(), lo[lane]);
                }
            }

            if LANES == 8 {
                let hi = transpose_4x4_i32([
                    _mm256_extracti128_si256::<1>(coeff[freq]),
                    _mm256_extracti128_si256::<1>(coeff[freq + 1]),
                    _mm256_extracti128_si256::<1>(coeff[freq + 2]),
                    _mm256_extracti128_si256::<1>(coeff[freq + 3]),
                ]);
                for lane in 0..4 {
                    unsafe {
                        _mm_storeu_si128(
                            dst.add((j + 4 + lane) * N + freq).cast::<__m128i>(),
                            hi[lane],
                        );
                    }
                }
            }
            freq += 4;
        }
    }
}

#[target_feature(enable = "avx2")]
fn fdct4_1d<const LANES: usize>(src: &[i32], shift: i32, line: usize, j: usize) -> [__m256i; 4] {
    let s0 = unsafe { load8::<LANES>(src, line, j, 0) };
    let s1 = unsafe { load8::<LANES>(src, line, j, 1) };
    let s2 = unsafe { load8::<LANES>(src, line, j, 2) };
    let s3 = unsafe { load8::<LANES>(src, line, j, 3) };
    let a = [_mm256_add_epi32(s0, s3), _mm256_add_epi32(s1, s2)];
    let b = [_mm256_sub_epi32(s0, s3), _mm256_sub_epi32(s1, s2)];
    let mut out = [_mm256_setzero_si256(); 4];
    out[0] = dot_shift(shift, &a, &DCT4_FWD_KERNEL, 0, 4);
    out[2] = dot_shift(shift, &a, &DCT4_FWD_KERNEL, 2, 4);
    out[1] = dot_shift(shift, &b, &DCT4_FWD_KERNEL, 1, 4);
    out[3] = dot_shift(shift, &b, &DCT4_FWD_KERNEL, 3, 4);
    out
}

#[target_feature(enable = "avx2")]
fn fdct8_1d<const LANES: usize>(src: &[i32], shift: i32, line: usize, j: usize) -> [__m256i; 8] {
    let z = _mm256_setzero_si256();
    let mut a = [z; 4];
    let mut b = [z; 4];
    for k in 0..4 {
        let lo = unsafe { load8::<LANES>(src, line, j, k) };
        let hi = unsafe { load8::<LANES>(src, line, j, 7 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let c = [_mm256_add_epi32(a[0], a[3]), _mm256_add_epi32(a[1], a[2])];
    let d = [_mm256_sub_epi32(a[0], a[3]), _mm256_sub_epi32(a[1], a[2])];
    let mut out = [z; 8];
    out[0] = dot_shift(shift, &c, &DCT8_FWD_KERNEL, 0, 8);
    out[4] = dot_shift(shift, &c, &DCT8_FWD_KERNEL, 4, 8);
    out[2] = dot_shift(shift, &d, &DCT8_FWD_KERNEL, 2, 8);
    out[6] = dot_shift(shift, &d, &DCT8_FWD_KERNEL, 6, 8);
    for row in (1..8).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT8_FWD_KERNEL, row, 8);
    }
    out
}

#[target_feature(enable = "avx2")]
fn fdct16_1d<const LANES: usize>(src: &[i32], shift: i32, line: usize, j: usize) -> [__m256i; 16] {
    let z = _mm256_setzero_si256();
    let mut a = [z; 8];
    let mut b = [z; 8];
    for k in 0..8 {
        let lo = unsafe { load8::<LANES>(src, line, j, k) };
        let hi = unsafe { load8::<LANES>(src, line, j, 15 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let mut c = [z; 4];
    let mut d = [z; 4];
    for k in 0..4 {
        c[k] = _mm256_add_epi32(a[k], a[7 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[7 - k]);
    }
    let e = [_mm256_add_epi32(c[0], c[3]), _mm256_add_epi32(c[1], c[2])];
    let f = [_mm256_sub_epi32(c[0], c[3]), _mm256_sub_epi32(c[1], c[2])];
    let mut out = [z; 16];
    out[0] = dot_shift(shift, &e, &DCT16_FWD_KERNEL, 0, 16);
    out[8] = dot_shift(shift, &e, &DCT16_FWD_KERNEL, 8, 16);
    out[4] = dot_shift(shift, &f, &DCT16_FWD_KERNEL, 4, 16);
    out[12] = dot_shift(shift, &f, &DCT16_FWD_KERNEL, 12, 16);
    for row in (2..16).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT16_FWD_KERNEL, row, 16);
    }
    for row in (1..16).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT16_FWD_KERNEL, row, 16);
    }
    out
}

#[target_feature(enable = "avx2")]
fn fdct32_1d<const LANES: usize>(src: &[i32], shift: i32, line: usize, j: usize) -> [__m256i; 32] {
    let z = _mm256_setzero_si256();
    let mut a = [z; 16];
    let mut b = [z; 16];
    for k in 0..16 {
        let lo = unsafe { load8::<LANES>(src, line, j, k) };
        let hi = unsafe { load8::<LANES>(src, line, j, 31 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let mut c = [z; 8];
    let mut d = [z; 8];
    for k in 0..8 {
        c[k] = _mm256_add_epi32(a[k], a[15 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[15 - k]);
    }
    let mut e = [z; 4];
    let mut f = [z; 4];
    for k in 0..4 {
        e[k] = _mm256_add_epi32(c[k], c[7 - k]);
        f[k] = _mm256_sub_epi32(c[k], c[7 - k]);
    }
    let g = [_mm256_add_epi32(e[0], e[3]), _mm256_add_epi32(e[1], e[2])];
    let h = [_mm256_sub_epi32(e[0], e[3]), _mm256_sub_epi32(e[1], e[2])];
    let mut out = [z; 32];
    out[0] = dot_shift(shift, &g, &DCT32_FWD_KERNEL, 0, 32);
    out[16] = dot_shift(shift, &g, &DCT32_FWD_KERNEL, 16, 32);
    out[8] = dot_shift(shift, &h, &DCT32_FWD_KERNEL, 8, 32);
    out[24] = dot_shift(shift, &h, &DCT32_FWD_KERNEL, 24, 32);
    for row in (4..32).step_by(8) {
        out[row] = dot_shift(shift, &f, &DCT32_FWD_KERNEL, row, 32);
    }
    for row in (2..32).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT32_FWD_KERNEL, row, 32);
    }
    for row in (1..32).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT32_FWD_KERNEL, row, 32);
    }
    out
}

#[target_feature(enable = "avx2")]
fn fdct64_1d<const LANES: usize>(
    src: &[i32],
    shift: i32,
    line: usize,
    zero_line: usize,
    j: usize,
) -> [__m256i; 64] {
    let top = if zero_line != 0 { 32 } else { 64 };
    let z = _mm256_setzero_si256();
    let mut a = [z; 32];
    let mut b = [z; 32];
    for k in 0..32 {
        let lo = unsafe { load8::<LANES>(src, line, j, k) };
        let hi = unsafe { load8::<LANES>(src, line, j, 63 - k) };
        a[k] = _mm256_add_epi32(lo, hi);
        b[k] = _mm256_sub_epi32(lo, hi);
    }
    let mut c = [z; 16];
    let mut d = [z; 16];
    for k in 0..16 {
        c[k] = _mm256_add_epi32(a[k], a[31 - k]);
        d[k] = _mm256_sub_epi32(a[k], a[31 - k]);
    }
    let mut e = [z; 8];
    let mut f = [z; 8];
    for k in 0..8 {
        e[k] = _mm256_add_epi32(c[k], c[15 - k]);
        f[k] = _mm256_sub_epi32(c[k], c[15 - k]);
    }
    let mut g = [z; 4];
    let mut h = [z; 4];
    for k in 0..4 {
        g[k] = _mm256_add_epi32(e[k], e[7 - k]);
        h[k] = _mm256_sub_epi32(e[k], e[7 - k]);
    }
    let i0 = [_mm256_add_epi32(g[0], g[3]), _mm256_add_epi32(g[1], g[2])];
    let u0 = [_mm256_sub_epi32(g[0], g[3]), _mm256_sub_epi32(g[1], g[2])];
    let mut out = [z; 64];
    out[0] = dot_shift(shift, &i0, &DCT64_FWD_KERNEL, 0, 64);
    out[16] = dot_shift(shift, &u0, &DCT64_FWD_KERNEL, 16, 64);
    if top == 64 {
        out[32] = dot_shift(shift, &i0, &DCT64_FWD_KERNEL, 32, 64);
        out[48] = dot_shift(shift, &u0, &DCT64_FWD_KERNEL, 48, 64);
    }
    for row in (8..top).step_by(16) {
        out[row] = dot_shift(shift, &h, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (4..top).step_by(8) {
        out[row] = dot_shift(shift, &f, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (2..top).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (1..top).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT64_FWD_KERNEL, row, 64);
    }
    out
}

#[inline]
#[target_feature(enable = "avx2")]
fn fdct_batch<const LANES: usize>(
    n: usize,
    src: &[i32],
    dst: &mut [i32],
    shift: i32,
    line: usize,
    zero: usize,
    j: usize,
) {
    let valid = if n == 64 && zero != 0 { 32 } else { n };
    match n {
        4 => transpose_store::<4, LANES>(dst, j, &fdct4_1d::<LANES>(src, shift, line, j), valid),
        8 => transpose_store::<8, LANES>(dst, j, &fdct8_1d::<LANES>(src, shift, line, j), valid),
        16 => transpose_store::<16, LANES>(dst, j, &fdct16_1d::<LANES>(src, shift, line, j), valid),
        32 => transpose_store::<32, LANES>(dst, j, &fdct32_1d::<LANES>(src, shift, line, j), valid),
        64 => transpose_store::<64, LANES>(
            dst,
            j,
            &fdct64_1d::<LANES>(src, shift, line, zero, j),
            valid,
        ),
        _ => unreachable!("unsupported 1D size {n}"),
    }
}

#[target_feature(enable = "avx2")]
fn fdct_1d_n(n: usize, src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero: usize) {
    debug_assert!(line.is_multiple_of(4));
    let mut j = 0usize;
    while j + 8 <= line {
        fdct_batch::<8>(n, src, dst, shift, line, zero, j);
        j += 8;
    }
    if j < line {
        debug_assert_eq!(line - j, 4);
        fdct_batch::<4>(n, src, dst, shift, line, zero, j);
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
        for (dst_row, src_row) in out[..cw * ch]
            .chunks_exact_mut(cw)
            .zip(s.coeff.chunks_exact(w).take(ch))
        {
            dst_row.copy_from_slice(&src_row[..cw]);
        }
    });

    let n = cw * ch;
    if (w.trailing_zeros() + h.trailing_zeros()) & 1 == 1 {
        scale_rect2_in_place(&mut out[..n]);
    }
    n
}
