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

#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x8(src: &[i32; 8]) -> __m256i {
    unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x8(dst: &mut [i32; 8], v: __m256i) {
    unsafe { _mm256_storeu_si256(dst.as_mut_ptr().cast::<__m256i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_i32x4(src: &[i32; 4]) -> __m128i {
    unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x4(dst: &mut [i32; 4], v: __m128i) {
    unsafe { _mm_storeu_si128(dst.as_mut_ptr().cast::<__m128i>(), v) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn square_i32x8_to_i64(v: __m256i) -> __m256i {
    let even = _mm256_mul_epi32(v, v);
    let odd_src = _mm256_srli_epi64::<32>(v);
    let odd = _mm256_mul_epi32(odd_src, odd_src);
    _mm256_add_epi64(even, odd)
}

#[inline]
#[target_feature(enable = "avx2")]
fn square_i32x4_to_i64(v: __m128i) -> __m128i {
    let even = _mm_mul_epi32(v, v);
    let odd_src = _mm_srli_epi64::<32>(v);
    let odd = _mm_mul_epi32(odd_src, odd_src);
    _mm_add_epi64(even, odd)
}

#[inline]
#[target_feature(enable = "avx2")]
fn reduce_i64x4(v: __m256i) -> i64 {
    let lo = _mm256_castsi256_si128(v);
    let hi = _mm256_extracti128_si256::<1>(v);
    let sum2 = _mm_add_epi64(lo, hi);
    let hi64 = _mm_unpackhi_epi64(sum2, sum2);
    let sum1 = _mm_add_epi64(sum2, hi64);
    _mm_cvtsi128_si64(sum1)
}

#[inline]
#[target_feature(enable = "avx2")]
fn widen_i64x2_to_i64x4(v: __m128i) -> __m256i {
    _mm256_inserti128_si256::<0>(_mm256_setzero_si256(), v)
}

#[inline]
fn block_row_mut(dst: &mut [i32], row: usize, w: usize) -> &mut [i32] {
    let off = row * w;
    &mut dst[off..off + w]
}

#[inline]
fn block_row(src: &[i32], row: usize, w: usize) -> &[i32] {
    let off = row * w;
    &src[off..off + w]
}

#[inline]
fn image_row(src: &[i32], stride: usize, px: usize, py: usize, row: usize, w: usize) -> &[i32] {
    let off = (py + row) * stride + px;
    &src[off..off + w]
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_pred_avx2(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    for ry in 0..h {
        let dst_row = block_row_mut(dst, ry, w);
        let pred_row = block_row(pred, ry, w);
        let src_row = image_row(src, stride, px, py, ry, w);

        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();
        let (pred8, pred_tail) = pred_row.as_chunks::<8>();
        let (src8, src_tail) = src_row.as_chunks::<8>();

        for ((d, s), p) in dst8.iter_mut().zip(src8).zip(pred8) {
            let s = load_i32x8(s);
            let p = load_i32x8(p);
            let r = _mm256_sub_epi32(s, p);
            store_i32x8(d, r);
        }

        let (dst4, dst_tail) = dst_tail.as_chunks_mut::<4>();
        let (pred4, pred_tail) = pred_tail.as_chunks::<4>();
        let (src4, src_tail) = src_tail.as_chunks::<4>();

        for ((d, s), p) in dst4.iter_mut().zip(src4).zip(pred4) {
            let s = load_i32x4(s);
            let p = load_i32x4(p);
            let r = _mm_sub_epi32(s, p);
            store_i32x4(d, r);
        }

        for ((d, &s), &p) in dst_tail.iter_mut().zip(src_tail).zip(pred_tail) {
            *d = s - p;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_dc_avx2(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    let dc8 = _mm256_set1_epi32(dc);
    let dc4 = _mm_set1_epi32(dc);

    for ry in 0..h {
        let dst_row = block_row_mut(dst, ry, w);
        let src_row = image_row(src, stride, px, py, ry, w);

        let (dst8, dst_tail) = dst_row.as_chunks_mut::<8>();
        let (src8, src_tail) = src_row.as_chunks::<8>();

        for (d, s) in dst8.iter_mut().zip(src8) {
            let s = load_i32x8(s);
            let r = _mm256_sub_epi32(s, dc8);
            store_i32x8(d, r);
        }

        let (dst4, dst_tail) = dst_tail.as_chunks_mut::<4>();
        let (src4, src_tail) = src_tail.as_chunks::<4>();

        for (d, s) in dst4.iter_mut().zip(src4) {
            let s = load_i32x4(s);
            let r = _mm_sub_epi32(s, dc4);
            store_i32x4(d, r);
        }

        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = s - dc;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn sse_recon_avx2(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    let zero8 = _mm256_setzero_si256();
    let max8 = _mm256_set1_epi32(maxv);

    let zero4 = _mm_setzero_si128();
    let max4 = _mm_set1_epi32(maxv);

    let mut acc = _mm256_setzero_si256();
    let mut scalar = 0i64;

    for ry in 0..h {
        let pred_row = block_row(pred, ry, w);
        let resid_row = block_row(resid, ry, w);
        let src_row = image_row(src, stride, px, py, ry, w);

        let (pred8, pred_tail) = pred_row.as_chunks::<8>();
        let (resid8, resid_tail) = resid_row.as_chunks::<8>();
        let (src8, src_tail) = src_row.as_chunks::<8>();

        for ((s, p), e) in src8.iter().zip(pred8).zip(resid8) {
            let s = load_i32x8(s);
            let p = load_i32x8(p);
            let e = load_i32x8(e);

            let r = _mm256_add_epi32(p, e);
            let r = _mm256_max_epi32(r, zero8);
            let r = _mm256_min_epi32(r, max8);

            let d = _mm256_sub_epi32(s, r);
            acc = _mm256_add_epi64(acc, square_i32x8_to_i64(d));
        }

        let (pred4, pred_tail) = pred_tail.as_chunks::<4>();
        let (resid4, resid_tail) = resid_tail.as_chunks::<4>();
        let (src4, src_tail) = src_tail.as_chunks::<4>();

        for ((s, p), e) in src4.iter().zip(pred4).zip(resid4) {
            let s = load_i32x4(s);
            let p = load_i32x4(p);
            let e = load_i32x4(e);

            let r = _mm_add_epi32(p, e);
            let r = _mm_max_epi32(r, zero4);
            let r = _mm_min_epi32(r, max4);

            let d = _mm_sub_epi32(s, r);
            acc = _mm256_add_epi64(acc, widen_i64x2_to_i64x4(square_i32x4_to_i64(d)));
        }

        for ((&s, &p), &e) in src_tail.iter().zip(pred_tail).zip(resid_tail) {
            let r = (p + e).clamp(0, maxv);
            let d = (s - r) as i64;
            scalar += d * d;
        }
    }

    reduce_i64x4(acc) + scalar
}
