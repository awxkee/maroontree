/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1.  Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
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

use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn load_u16x8(src: *const u16) -> __m128i {
    unsafe { _mm_loadu_si128(src.cast()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn interleave_uv(u: __m128i, v: __m128i) -> __m256i {
    let lo = _mm_unpacklo_epi16(u, v);
    let hi = _mm_unpackhi_epi16(u, v);
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_indices8(dst: *mut u8, indices: __m256i) {
    let lo = _mm256_castsi256_si128(indices);
    let hi = _mm256_extracti128_si256::<1>(indices);
    let indices16 = _mm_packus_epi32(lo, hi);
    let indices8 = _mm_packus_epi16(indices16, _mm_setzero_si128());
    unsafe { _mm_storel_epi64(dst.cast(), indices8) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_indices16(dst: *mut u8, indices: __m256i) {
    let lo = _mm256_castsi256_si128(indices);
    let hi = _mm256_extracti128_si256::<1>(indices);
    unsafe { _mm_storeu_si128(dst.cast(), _mm_packus_epi16(lo, hi)) };
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn uv_nearest_indices_avx2(
    src_u: &[u16],
    src_v: &[u16],
    stride: usize,
    cx: usize,
    cy: usize,
    w: usize,
    h: usize,
    centers: &[(i32, i32)],
    out: &mut [u8],
) {
    debug_assert!((1..=8).contains(&centers.len()));
    debug_assert_eq!(out.len(), w * h);
    let k = centers.len();
    assert!(k <= 8);
    let mut center_vec = [_mm256_setzero_si256(); 8];
    for j in 0..k {
        let (u, v) = centers[j];
        debug_assert!((0..=i16::MAX as i32).contains(&u));
        debug_assert!((0..=i16::MAX as i32).contains(&v));
        let uv = u | (v << 16);
        center_vec[j] = _mm256_set1_epi32(uv);
    }
    for ((u_row, v_row), out_row) in src_u
        .chunks_exact(stride)
        .skip(cy)
        .take(h)
        .zip(src_v.chunks_exact(stride).skip(cy).take(h))
        .zip(out.chunks_exact_mut(w))
    {
        let u_row = &u_row[cx..cx + w];
        let v_row = &v_row[cx..cx + w];
        let (u_chunks, u_tail) = u_row.as_chunks::<8>();
        let (v_chunks, v_tail) = v_row.as_chunks::<8>();
        let (out_chunks, out_tail) = out_row.as_chunks_mut::<8>();
        for ((u_chunk, v_chunk), out_chunk) in u_chunks
            .iter()
            .zip(v_chunks.iter())
            .zip(out_chunks.iter_mut())
        {
            let u = load_u16x8(u_chunk.as_ptr());
            let v = load_u16x8(v_chunk.as_ptr());
            let samples = interleave_uv(u, v);
            let mut best_dist = _mm256_set1_epi32(i32::MAX);
            let mut best_idx = _mm256_setzero_si256();
            #[allow(clippy::needless_range_loop)]
            for j in 0..k {
                let delta = _mm256_sub_epi16(samples, center_vec[j]);
                let dist = _mm256_madd_epi16(delta, delta);
                // Strict `<` preserves the first center on an equal-distance tie.
                let replace = _mm256_cmpgt_epi32(best_dist, dist);
                best_idx = _mm256_blendv_epi8(best_idx, _mm256_set1_epi32(j as i32), replace);
                best_dist = _mm256_min_epi32(best_dist, dist);
            }
            store_indices8(out_chunk.as_mut_ptr(), best_idx);
        }

        for ((&u, &v), out_idx) in u_tail.iter().zip(v_tail.iter()).zip(out_tail.iter_mut()) {
            let u = u as i32;
            let v = v as i32;
            let mut best_dist = i64::MAX;
            let mut best_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..k {
                let (cu, cv) = centers[j];
                let du = i64::from(u - cu);
                let dv = i64::from(v - cv);
                let dist = du * du + dv * dv;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            *out_idx = best_idx as u8;
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn luma_nearest_indices_avx2(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    centers: &[i32],
    out: &mut [u8],
) {
    debug_assert!((1..=8).contains(&centers.len()));
    debug_assert_eq!(out.len(), w * h);
    let k = centers.len();
    assert!(k <= 8);
    let mut center_vec = [_mm256_setzero_si256(); 8];
    for j in 0..k {
        let center = centers[j];
        debug_assert!((0..=i16::MAX as i32).contains(&center));
        center_vec[j] = _mm256_set1_epi16(center as i16);
    }

    for (row, out_row) in src
        .chunks_exact(stride)
        .skip(py)
        .take(h)
        .zip(out.chunks_exact_mut(w))
    {
        let row = &row[px..px + w];
        let (chunks, tail) = row.as_chunks::<16>();
        let (out_chunks, out_tail) = out_row.as_chunks_mut::<16>();
        for (chunk, out_chunk) in chunks.iter().zip(out_chunks.iter_mut()) {
            let samples = unsafe { _mm256_loadu_si256(chunk.as_ptr().cast()) };
            let mut best_dist = _mm256_set1_epi16(i16::MAX);
            let mut best_idx = _mm256_setzero_si256();
            #[allow(clippy::needless_range_loop)]
            for j in 0..k {
                // Squaring is monotonic for non-negative values, so comparing
                // absolute deltas selects exactly the same center.
                let dist = _mm256_abs_epi16(_mm256_sub_epi16(samples, center_vec[j]));
                let replace = _mm256_cmpgt_epi16(best_dist, dist);
                best_idx = _mm256_blendv_epi8(best_idx, _mm256_set1_epi16(j as i16), replace);
                best_dist = _mm256_min_epi16(best_dist, dist);
            }
            store_indices16(out_chunk.as_mut_ptr(), best_idx);
        }

        for (&sample, out_idx) in tail.iter().zip(out_tail.iter_mut()) {
            let sample = sample as i32;
            let mut best_dist = i64::MAX;
            let mut best_idx = 0;
            #[allow(clippy::needless_range_loop)]
            for j in 0..k {
                let delta = i64::from(sample - centers[j]);
                let dist = delta * delta;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = j;
                }
            }
            *out_idx = best_idx as u8;
        }
    }
}
