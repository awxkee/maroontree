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
fn image_row(src: &[u16], stride: usize, px: usize, py: usize, row: usize, w: usize) -> &[u16] {
    let off = (py + row) * stride + px;
    &src[off..off + w]
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8_as_i32(src: &[u16; 8]) -> __m256i {
    unsafe { _mm256_cvtepu16_epi32(_mm_loadu_si128(src.as_ptr().cast::<__m128i>())) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x4_as_i32(src: &[u16; 4]) -> __m128i {
    unsafe { _mm_cvtepu16_epi32(_mm_loadl_epi64(src.as_ptr().cast::<__m128i>())) }
}

#[target_feature(enable = "avx2")]
pub(crate) fn sum_i32_avx2(values: &[i32]) -> i32 {
    let (chunks, tail) = values.as_chunks::<8>();
    let mut acc = _mm256_setzero_si256();
    for chunk in chunks {
        acc = _mm256_add_epi32(acc, load_i32x8(chunk));
    }
    let mut lanes = [0i32; 8];
    store_i32x8(&mut lanes, acc);
    lanes.iter().chain(tail).copied().sum()
}

#[target_feature(enable = "avx2")]
pub(crate) fn sum_u16_avx2(values: &[u16]) -> i32 {
    let (chunks, tail) = values.as_chunks::<8>();
    let mut acc = _mm256_setzero_si256();
    for chunk in chunks {
        acc = _mm256_add_epi32(acc, load_u16x8_as_i32(chunk));
    }
    let mut lanes = [0i32; 8];
    store_i32x8(&mut lanes, acc);
    lanes.iter().copied().sum::<i32>() + tail.iter().map(|&value| i32::from(value)).sum::<i32>()
}

#[target_feature(enable = "avx2")]
pub(crate) fn sum_u16_strided_avx2(values: &[u16], stride: usize, len: usize) -> i32 {
    let mut packed = [0u16; 8];
    let mut acc = _mm256_setzero_si256();
    let mut index = 0;
    while index + 8 <= len {
        for lane in 0..8 {
            packed[lane] = values[(index + lane) * stride];
        }
        acc = _mm256_add_epi32(acc, load_u16x8_as_i32(&packed));
        index += 8;
    }
    let mut lanes = [0i32; 8];
    store_i32x8(&mut lanes, acc);
    lanes.iter().copied().sum::<i32>()
        + (index..len)
            .map(|lane| i32::from(values[lane * stride]))
            .sum::<i32>()
}

#[target_feature(enable = "avx2")]
pub(crate) fn all_zero_i32_avx2(values: &[i32]) -> bool {
    let (chunks, tail) = values.as_chunks::<8>();
    let mut bits = _mm256_setzero_si256();
    for chunk in chunks {
        bits = _mm256_or_si256(bits, load_i32x8(chunk));
    }
    _mm256_testz_si256(bits, bits) != 0 && tail.iter().all(|&value| value == 0)
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_pred_avx2(
    dst: &mut [i32],
    pred: &[i32],
    src: &[u16],
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
            let s = load_u16x8_as_i32(s);
            let p = load_i32x8(p);
            let r = _mm256_sub_epi32(s, p);
            store_i32x8(d, r);
        }

        let (dst4, dst_tail) = dst_tail.as_chunks_mut::<4>();
        let (pred4, pred_tail) = pred_tail.as_chunks::<4>();
        let (src4, src_tail) = src_tail.as_chunks::<4>();

        for ((d, s), p) in dst4.iter_mut().zip(src4).zip(pred4) {
            let s = load_u16x4_as_i32(s);
            let p = load_i32x4(p);
            let r = _mm_sub_epi32(s, p);
            store_i32x4(d, r);
        }

        for ((d, &s), &p) in dst_tail.iter_mut().zip(src_tail).zip(pred_tail) {
            *d = s as i32 - p;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn residual_dc_avx2(
    dst: &mut [i32],
    src: &[u16],
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
            let s = load_u16x8_as_i32(s);
            let r = _mm256_sub_epi32(s, dc8);
            store_i32x8(d, r);
        }

        let (dst4, dst_tail) = dst_tail.as_chunks_mut::<4>();
        let (src4, src_tail) = src_tail.as_chunks::<4>();

        for (d, s) in dst4.iter_mut().zip(src4) {
            let s = load_u16x4_as_i32(s);
            let r = _mm_sub_epi32(s, dc4);
            store_i32x4(d, r);
        }

        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = s as i32 - dc;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn reconstruct_avx2(
    dst: &mut [u16],
    dst_stride: usize,
    mirror: &mut [u16],
    mirror_stride: usize,
    pred: &[i32],
    resid: &[i32],
    w: usize,
    h: usize,
    maxv: i32,
) {
    let zero8 = _mm256_setzero_si256();
    let max8 = _mm256_set1_epi32(maxv);
    let zero4 = _mm_setzero_si128();
    let max4 = _mm_set1_epi32(maxv);
    let mirrored = !mirror.is_empty();

    for ry in 0..h {
        let dst_row = &mut dst[ry * dst_stride..][..w];
        let pred_row = &pred[ry * w..][..w];
        let resid_row = (!resid.is_empty()).then(|| &resid[ry * w..][..w]);
        let (pred8, pred_tail) = pred_row.as_chunks::<8>();

        for (chunk, prediction) in pred8.iter().enumerate() {
            let reconstruction = if let Some(residual) = resid_row {
                let residual = residual[chunk * 8..].first_chunk::<8>().unwrap();
                _mm256_add_epi32(load_i32x8(prediction), load_i32x8(residual))
            } else {
                load_i32x8(prediction)
            };
            let reconstruction = _mm256_min_epi32(_mm256_max_epi32(reconstruction, zero8), max8);
            let lo = _mm256_castsi256_si128(reconstruction);
            let hi = _mm256_extracti128_si256::<1>(reconstruction);
            let reconstruction = _mm_packus_epi32(lo, hi);
            let x = chunk * 8;
            unsafe {
                _mm_storeu_si128(dst_row[x..].as_mut_ptr().cast::<__m128i>(), reconstruction);
                if mirrored {
                    _mm_storeu_si128(
                        mirror[ry * mirror_stride + x..]
                            .as_mut_ptr()
                            .cast::<__m128i>(),
                        reconstruction,
                    );
                }
            }
        }

        let tail_x = pred8.len() * 8;
        let (pred4, pred_tail) = pred_tail.as_chunks::<4>();
        for (chunk, prediction) in pred4.iter().enumerate() {
            let reconstruction = if let Some(residual) = resid_row {
                let x = tail_x + chunk * 4;
                let residual = residual[x..].first_chunk::<4>().unwrap();
                _mm_add_epi32(load_i32x4(prediction), load_i32x4(residual))
            } else {
                load_i32x4(prediction)
            };
            let reconstruction = _mm_min_epi32(_mm_max_epi32(reconstruction, zero4), max4);
            let reconstruction = _mm_packus_epi32(reconstruction, reconstruction);
            let x = tail_x + chunk * 4;
            unsafe {
                _mm_storel_epi64(dst_row[x..].as_mut_ptr().cast::<__m128i>(), reconstruction);
                if mirrored {
                    _mm_storel_epi64(
                        mirror[ry * mirror_stride + x..]
                            .as_mut_ptr()
                            .cast::<__m128i>(),
                        reconstruction,
                    );
                }
            }
        }

        let scalar_x = tail_x + pred4.len() * 4;
        for (lane, &prediction) in pred_tail.iter().enumerate() {
            let residual = resid_row.map_or(0, |row| row[scalar_x + lane]);
            let reconstruction = (prediction + residual).clamp(0, maxv) as u16;
            dst_row[scalar_x + lane] = reconstruction;
            if mirrored {
                mirror[ry * mirror_stride + scalar_x + lane] = reconstruction;
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn sse_recon_avx2(
    pred: &[i32],
    resid: &[i32],
    src: &[u16],
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
            let s = load_u16x8_as_i32(s);
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
            let s = load_u16x4_as_i32(s);
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
            let d = (s as i32 - r) as i64;
            scalar += d * d;
        }
    }

    reduce_i64x4(acc) + scalar
}

#[target_feature(enable = "avx2")]
pub(crate) fn chroma_sse_avx2(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> i64 {
    let zero8 = _mm256_setzero_si256();
    let max8 = _mm256_set1_epi32(max_value);
    let dc8 = _mm256_set1_epi32(dc);
    let zero4 = _mm_setzero_si128();
    let max4 = _mm_set1_epi32(max_value);
    let dc4 = _mm_set1_epi32(dc);
    let mut acc = _mm256_setzero_si256();
    let mut scalar = 0i64;

    for y in 0..h {
        let src_row = image_row(src, stride, px, py, y, w);
        let pred_row = (!pred.is_empty()).then(|| block_row(pred, y, w));
        let residual_row = (!residual.is_empty()).then(|| block_row(residual, y, w));
        let (src8, src_tail) = src_row.as_chunks::<8>();

        for (chunk, source) in src8.iter().enumerate() {
            let prediction = pred_row.map_or(dc8, |row| {
                load_i32x8(row[chunk * 8..].first_chunk::<8>().unwrap())
            });
            let reconstruction = residual_row.map_or(prediction, |row| {
                _mm256_add_epi32(
                    prediction,
                    load_i32x8(row[chunk * 8..].first_chunk::<8>().unwrap()),
                )
            });
            let reconstruction = _mm256_min_epi32(_mm256_max_epi32(reconstruction, zero8), max8);
            let delta = _mm256_sub_epi32(load_u16x8_as_i32(source), reconstruction);
            acc = _mm256_add_epi64(acc, square_i32x8_to_i64(delta));
        }

        let x4 = src8.len() * 8;
        let (src4, src_tail) = src_tail.as_chunks::<4>();
        for (chunk, source) in src4.iter().enumerate() {
            let x = x4 + chunk * 4;
            let prediction =
                pred_row.map_or(dc4, |row| load_i32x4(row[x..].first_chunk::<4>().unwrap()));
            let reconstruction = residual_row.map_or(prediction, |row| {
                _mm_add_epi32(prediction, load_i32x4(row[x..].first_chunk::<4>().unwrap()))
            });
            let reconstruction = _mm_min_epi32(_mm_max_epi32(reconstruction, zero4), max4);
            let delta = _mm_sub_epi32(load_u16x4_as_i32(source), reconstruction);
            acc = _mm256_add_epi64(acc, widen_i64x2_to_i64x4(square_i32x4_to_i64(delta)));
        }

        let scalar_x = x4 + src4.len() * 4;
        for (lane, &source) in src_tail.iter().enumerate() {
            let x = scalar_x + lane;
            let prediction = pred_row.map_or(dc, |row| row[x]);
            let reconstruction = prediction + residual_row.map_or(0, |row| row[x]);
            let delta = (i32::from(source) - reconstruction.clamp(0, max_value)) as i64;
            scalar += delta * delta;
        }
    }

    reduce_i64x4(acc) + scalar
}

#[target_feature(enable = "avx2")]
pub(crate) fn sse_u16_avx2(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: &[u16],
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut acc = _mm256_setzero_si256();
    let mut scalar = 0i64;
    for row in 0..h {
        let src_row = &src[(src_y + row) * src_stride + src_x..][..w];
        let ref_row = &reference[(ref_y + row) * ref_stride + ref_x..][..w];
        let (src8, src_tail) = src_row.as_chunks::<8>();
        let (ref8, ref_tail) = ref_row.as_chunks::<8>();
        for (src, reference) in src8.iter().zip(ref8) {
            let diff = _mm256_sub_epi32(load_u16x8_as_i32(src), load_u16x8_as_i32(reference));
            acc = _mm256_add_epi64(acc, square_i32x8_to_i64(diff));
        }
        let (src4, src_tail) = src_tail.as_chunks::<4>();
        let (ref4, ref_tail) = ref_tail.as_chunks::<4>();
        for (src, reference) in src4.iter().zip(ref4) {
            let diff = _mm_sub_epi32(load_u16x4_as_i32(src), load_u16x4_as_i32(reference));
            acc = _mm256_add_epi64(acc, widen_i64x2_to_i64x4(square_i32x4_to_i64(diff)));
        }
        for (&src, &reference) in src_tail.iter().zip(ref_tail) {
            let diff = i64::from(src) - i64::from(reference);
            scalar += diff * diff;
        }
    }
    reduce_i64x4(acc) + scalar
}

#[inline]
#[target_feature(enable = "avx2")]
fn had4_butterfly_x4(
    d0: __m128i,
    d1: __m128i,
    d2: __m128i,
    d3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let e = _mm_add_epi32(d0, d2);
    let f = _mm_sub_epi32(d0, d2);
    let g = _mm_add_epi32(d1, d3);
    let h = _mm_sub_epi32(d1, d3);
    (
        _mm_add_epi32(e, g),
        _mm_add_epi32(f, h),
        _mm_sub_epi32(f, h),
        _mm_sub_epi32(e, g),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4_x4(
    t0: __m128i,
    t1: __m128i,
    t2: __m128i,
    t3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let a = _mm_unpacklo_epi32(t0, t1);
    let b = _mm_unpackhi_epi32(t0, t1);
    let c = _mm_unpacklo_epi32(t2, t3);
    let d = _mm_unpackhi_epi32(t2, t3);
    (
        _mm_unpacklo_epi64(a, c),
        _mm_unpackhi_epi64(a, c),
        _mm_unpacklo_epi64(b, d),
        _mm_unpackhi_epi64(b, d),
    )
}

/// abs(i32x4) widened and accumulated into an i64x4 accumulator.
#[inline]
#[target_feature(enable = "avx2")]
fn abs_acc_i32x4(acc: __m256i, v: __m128i) -> __m256i {
    _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(_mm_abs_epi32(v)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn satd_4x4_accumulate(mut acc: __m256i, error: [__m128i; 4]) -> __m256i {
    let (t0, t1, t2, t3) = had4_butterfly_x4(error[0], error[1], error[2], error[3]);
    let (r0, r1, r2, r3) = transpose_4x4_x4(t0, t1, t2, t3);
    let (u0, u1, u2, u3) = had4_butterfly_x4(r0, r1, r2, r3);
    acc = abs_acc_i32x4(acc, u0);
    acc = abs_acc_i32x4(acc, u1);
    acc = abs_acc_i32x4(acc, u2);
    abs_acc_i32x4(acc, u3)
}

#[target_feature(enable = "avx2")]
pub(crate) fn luma_satd_avx2(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> u64 {
    let zero = _mm_setzero_si128();
    let max_value = _mm_set1_epi32(max_value);
    let dc = _mm_set1_epi32(dc);
    let mut satd = _mm256_setzero_si256();
    for ty in (0..h).step_by(4) {
        let src_rows: [&[[u16; 4]]; 4] =
            std::array::from_fn(|row| src[(py + ty + row) * stride + px..][..w].as_chunks::<4>().0);
        let pred_rows: Option<[&[[i32; 4]]; 4]> = (!pred.is_empty())
            .then(|| std::array::from_fn(|row| pred[(ty + row) * w..][..w].as_chunks::<4>().0));
        let residual_rows: Option<[&[[i32; 4]]; 4]> = (!residual.is_empty())
            .then(|| std::array::from_fn(|row| residual[(ty + row) * w..][..w].as_chunks::<4>().0));

        for chunk in 0..w / 4 {
            let error = std::array::from_fn(|row| {
                let source = load_u16x4_as_i32(&src_rows[row][chunk]);
                let prediction = if let Some(rows) = &pred_rows {
                    load_i32x4(&rows[row][chunk])
                } else {
                    dc
                };
                let reconstruction = if let Some(rows) = &residual_rows {
                    _mm_add_epi32(prediction, load_i32x4(&rows[row][chunk]))
                } else {
                    prediction
                };
                let reconstruction = _mm_min_epi32(_mm_max_epi32(reconstruction, zero), max_value);
                _mm_sub_epi32(source, reconstruction)
            });
            satd = satd_4x4_accumulate(satd, error);
        }
    }
    reduce_i64x4(satd) as u64
}

#[target_feature(enable = "avx2")]
pub(crate) fn satd_sad_proxy_avx2(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sad_acc = _mm256_setzero_si256();
    let mut satd_acc = _mm256_setzero_si256();
    for ty in (0..h).step_by(4) {
        let src_rows: [&[u16]; 4] =
            std::array::from_fn(|r| &src[(ty + r) * src_stride..(ty + r) * src_stride + w]);
        let pred_rows: [&[i32]; 4] =
            std::array::from_fn(|r| &pred[(ty + r) * pred_stride..(ty + r) * pred_stride + w]);
        let s: [&[[u16; 4]]; 4] = std::array::from_fn(|r| src_rows[r].as_chunks::<4>().0);
        let p: [&[[i32; 4]]; 4] = std::array::from_fn(|r| pred_rows[r].as_chunks::<4>().0);
        for i in 0..w / 4 {
            let d0 = _mm_sub_epi32(load_u16x4_as_i32(&s[0][i]), load_i32x4(&p[0][i]));
            let d1 = _mm_sub_epi32(load_u16x4_as_i32(&s[1][i]), load_i32x4(&p[1][i]));
            let d2 = _mm_sub_epi32(load_u16x4_as_i32(&s[2][i]), load_i32x4(&p[2][i]));
            let d3 = _mm_sub_epi32(load_u16x4_as_i32(&s[3][i]), load_i32x4(&p[3][i]));

            sad_acc = abs_acc_i32x4(sad_acc, d0);
            sad_acc = abs_acc_i32x4(sad_acc, d1);
            sad_acc = abs_acc_i32x4(sad_acc, d2);
            sad_acc = abs_acc_i32x4(sad_acc, d3);

            satd_acc = satd_4x4_accumulate(satd_acc, [d0, d1, d2, d3]);
        }
    }
    let sad = reduce_i64x4(sad_acc);
    let satd = reduce_i64x4(satd_acc);
    (sad as u64) + ((satd as u64) >> 2)
}
