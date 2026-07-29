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

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::intrapred::{DR_EDGE_ORIGIN, DR_ZONE1, DR_ZONE2, DR_ZONE3, DrPrediction, sm_weights};

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
fn load_i32x8_as_i16(src: &[i32; 8]) -> __m128i {
    let value = load_i32x8(src);
    _mm_packs_epi32(
        _mm256_castsi256_si128(value),
        _mm256_extracti128_si256::<1>(value),
    )
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i16x8_as_i32(dst: &mut [i32; 8], value: __m128i) {
    store_i32x8(dst, _mm256_cvtepi16_epi32(value));
}

#[inline]
#[target_feature(enable = "avx2")]
fn accumulate_i32x4_i64(acc: __m256i, value: __m128i) -> __m256i {
    _mm256_add_epi64(acc, _mm256_cvtepi32_epi64(value))
}

#[inline]
#[target_feature(enable = "avx2")]
fn accumulate_i32x8_i64(mut acc: __m256i, value: __m256i) -> __m256i {
    acc = accumulate_i32x4_i64(acc, _mm256_castsi256_si128(value));
    accumulate_i32x4_i64(acc, _mm256_extracti128_si256::<1>(value))
}

#[inline]
#[target_feature(enable = "avx2")]
fn sum_i64x4(value: __m256i) -> i64 {
    let mut lanes = [0i64; 4];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), value) };
    lanes.iter().sum()
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_cfl_q3_8(dst: &mut [i32; 8], q3: __m256i, sum: __m256i) -> __m256i {
    store_i32x8(dst, q3);
    accumulate_i32x8_i64(sum, q3)
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_cfl_q3_4(dst: &mut [i32; 4], q3: __m128i, sum: __m256i) -> __m256i {
    store_i32x4(dst, q3);
    accumulate_i32x4_i64(sum, q3)
}

#[inline]
#[target_feature(enable = "avx2")]
fn subtract_cfl_mean(ac: &mut [i32], sum: u64, log2sz: u32) {
    let mean = ((sum + ((1u64 << log2sz) >> 1)) >> log2sz) as i32;
    let (ac8, rest) = ac.as_chunks_mut::<8>();
    let mean8 = _mm256_set1_epi32(mean);
    for chunk in ac8 {
        store_i32x8(chunk, _mm256_sub_epi32(load_i32x8(chunk), mean8));
    }
    let (ac4, ac_tail) = rest.as_chunks_mut::<4>();
    let mean4 = _mm_set1_epi32(mean);
    for chunk in ac4 {
        store_i32x4(chunk, _mm_sub_epi32(load_i32x4(chunk), mean4));
    }
    for value in ac_tail {
        *value -= mean;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn cfl_predict_i16(
    ac: __m128i,
    dc: __m128i,
    max: __m128i,
    alpha_sign: __m128i,
    abs_alpha_q12: i16,
) -> __m128i {
    let magnitude = _mm_mulhrs_epi16(_mm_abs_epi16(ac), _mm_set1_epi16(abs_alpha_q12));
    let sign = _mm_srai_epi16::<15>(_mm_xor_si128(ac, alpha_sign));
    let signed = _mm_sub_epi16(_mm_xor_si128(magnitude, sign), sign);
    _mm_min_epi16(
        _mm_max_epi16(_mm_add_epi16(dc, signed), _mm_setzero_si128()),
        max,
    )
}

/// AVX2 has no integer FMA; multiply-accumulate is mullo + add.
#[inline]
#[target_feature(enable = "avx2")]
fn mla8_n(acc: __m256i, v: __m256i, k: i32) -> __m256i {
    _mm256_add_epi32(acc, _mm256_mullo_epi32(v, _mm256_set1_epi32(k)))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mla4_n(acc: __m128i, v: __m128i, k: i32) -> __m128i {
    _mm_add_epi32(acc, _mm_mullo_epi32(v, _mm_set1_epi32(k)))
}

/// Lane mask for `x <= y`: `max(x, y) == y` holds exactly then (no
/// all-ones constant needed, unlike negating `cmpgt`).
#[inline]
#[target_feature(enable = "avx2")]
fn le8(x: __m256i, y: __m256i) -> __m256i {
    _mm256_cmpeq_epi32(_mm256_max_epi32(x, y), y)
}

#[inline]
#[target_feature(enable = "avx2")]
fn le4(x: __m128i, y: __m128i) -> __m128i {
    _mm_cmpeq_epi32(_mm_max_epi32(x, y), y)
}

#[inline]
#[target_feature(enable = "avx2")]
fn absdiff8(a: __m256i, b: __m256i) -> __m256i {
    _mm256_abs_epi32(_mm256_sub_epi32(a, b))
}

#[inline]
#[target_feature(enable = "avx2")]
fn absdiff4(a: __m128i, b: __m128i) -> __m128i {
    _mm_abs_epi32(_mm_sub_epi32(a, b))
}

/// Chunked views of `src` shifted by `0..N` samples: `views[j][i]` is the
/// `W`-sample window at `src[W * i + j]`.
#[inline]
fn shifted_chunks<const N: usize, const W: usize>(src: &[i32]) -> [&[[i32; W]]; N] {
    std::array::from_fn(|j| src[j..].as_chunks::<W>().0)
}

/// Row-aligned 8-wide and 4-wide views of a `bw`-long edge/weight array, so
/// each view chunks identically to the output row.
#[inline]
fn row_views(src: &[i32], bw: usize) -> (&[[i32; 8]], &[[i32; 4]]) {
    let (v8, rest) = src[..bw].as_chunks::<8>();
    (v8, rest.as_chunks::<4>().0)
}

#[target_feature(enable = "avx2")]
pub(crate) fn dc_pred_avx2(
    recon: &[u16],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let mut edges = [0u16; 64];
    let mut len = 0;
    if have_top {
        edges[..width].copy_from_slice(&recon[(oy - 1) * stride + ox..][..width]);
        len = width;
    }
    if have_left {
        for y in 0..height {
            edges[len + y] = recon[(oy + y) * stride + ox - 1];
        }
        len += height;
    }
    let (chunks, tail) = edges[..len].as_chunks::<8>();
    let mut acc = _mm256_setzero_si256();
    for chunk in chunks {
        let packed = unsafe { _mm_loadu_si128(chunk.as_ptr().cast::<__m128i>()) };
        acc = _mm256_add_epi32(acc, _mm256_cvtepu16_epi32(packed));
    }
    let mut lanes = [0i32; 8];
    unsafe { _mm256_storeu_si256(lanes.as_mut_ptr().cast::<__m256i>(), acc) };
    let sum = lanes.iter().sum::<i32>() + tail.iter().map(|&v| i32::from(v)).sum::<i32>();
    crate::intrapred::dc_pred_from_sum(sum, width, height, have_top, have_left, bit_depth)
}

#[target_feature(enable = "avx2")]
pub(crate) fn smooth_avx2(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let (wv, wh) = (sm_weights(bh), sm_weights(bw));
    let (right, bottom) = (top[bw - 1], left[bh - 1]);
    let (top8, top4) = row_views(top, bw);
    let (wh8, wh4) = row_views(wh, bw);
    let c256_8 = _mm256_set1_epi32(256);
    let c256_4 = _mm_set1_epi32(256);
    let rnd8 = _mm256_set1_epi32(256);
    let rnd4 = _mm_set1_epi32(256);
    for ((row, &wvy), &lv) in out
        .chunks_exact_mut(bw)
        .zip(wv.iter())
        .zip(left.iter())
        .take(bh)
    {
        let base = (256 - wvy) * bottom;
        let base8 = _mm256_set1_epi32(base);
        let base4 = _mm_set1_epi32(base);
        let (o8, rest) = row.as_chunks_mut::<8>();
        for ((o, t), w) in o8.iter_mut().zip(top8).zip(wh8) {
            let whx = load_i32x8(w);
            let mut acc = mla8_n(base8, load_i32x8(t), wvy); // base + top*wvy
            acc = mla8_n(acc, whx, lv); // + wh*left[y]
            acc = mla8_n(acc, _mm256_sub_epi32(c256_8, whx), right); // + (256-wh)*right
            store_i32x8(o, _mm256_srai_epi32::<9>(_mm256_add_epi32(acc, rnd8)));
        }
        let (o4, tail) = rest.as_chunks_mut::<4>();
        for ((o, t), w) in o4.iter_mut().zip(top4).zip(wh4) {
            let whx = load_i32x4(w);
            let mut acc = mla4_n(base4, load_i32x4(t), wvy);
            acc = mla4_n(acc, whx, lv);
            acc = mla4_n(acc, _mm_sub_epi32(c256_4, whx), right);
            store_i32x4(o, _mm_srai_epi32::<9>(_mm_add_epi32(acc, rnd4)));
        }
        let done = o8.len() * 8 + o4.len() * 4;
        for (x, o) in tail.iter_mut().enumerate() {
            let x = done + x;
            let pred = wvy * top[x] + (256 - wvy) * bottom + wh[x] * lv + (256 - wh[x]) * right;
            *o = (pred + 256) >> 9;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn smooth_v_avx2(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wv = sm_weights(bh);
    let bottom = left[bh - 1];
    let (top8, top4) = row_views(top, bw);
    let rnd8 = _mm256_set1_epi32(128);
    let rnd4 = _mm_set1_epi32(128);
    for (row, &wvy) in out.chunks_exact_mut(bw).zip(wv.iter()).take(bh) {
        let base = (256 - wvy) * bottom;
        let base8 = _mm256_set1_epi32(base);
        let base4 = _mm_set1_epi32(base);
        let (o8, rest) = row.as_chunks_mut::<8>();
        for (o, t) in o8.iter_mut().zip(top8) {
            let acc = mla8_n(base8, load_i32x8(t), wvy);
            store_i32x8(o, _mm256_srai_epi32::<8>(_mm256_add_epi32(acc, rnd8)));
        }
        let (o4, tail) = rest.as_chunks_mut::<4>();
        for (o, t) in o4.iter_mut().zip(top4) {
            let acc = mla4_n(base4, load_i32x4(t), wvy);
            store_i32x4(o, _mm_srai_epi32::<8>(_mm_add_epi32(acc, rnd4)));
        }
        let done = o8.len() * 8 + o4.len() * 4;
        for (x, o) in tail.iter_mut().enumerate() {
            *o = (wvy * top[done + x] + (256 - wvy) * bottom + 128) >> 8;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn smooth_h_avx2(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wh = sm_weights(bw);
    let right = top[bw - 1];
    let (wh8, wh4) = row_views(wh, bw);
    let c256_8 = _mm256_set1_epi32(256);
    let c256_4 = _mm_set1_epi32(256);
    let rnd8 = _mm256_set1_epi32(128);
    let rnd4 = _mm_set1_epi32(128);
    let zero8 = _mm256_setzero_si256();
    let zero4 = _mm_setzero_si128();
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let (o8, rest) = row.as_chunks_mut::<8>();
        for (o, w) in o8.iter_mut().zip(wh8) {
            let whx = load_i32x8(w);
            let mut acc = mla8_n(zero8, _mm256_sub_epi32(c256_8, whx), right); // (256-wh)*right
            acc = mla8_n(acc, whx, lv); // + wh*left[y]
            store_i32x8(o, _mm256_srai_epi32::<8>(_mm256_add_epi32(acc, rnd8)));
        }
        let (o4, tail) = rest.as_chunks_mut::<4>();
        for (o, w) in o4.iter_mut().zip(wh4) {
            let whx = load_i32x4(w);
            let mut acc = mla4_n(zero4, _mm_sub_epi32(c256_4, whx), right);
            acc = mla4_n(acc, whx, lv);
            store_i32x4(o, _mm_srai_epi32::<8>(_mm_add_epi32(acc, rnd4)));
        }
        let done = o8.len() * 8 + o4.len() * 4;
        for (x, o) in tail.iter_mut().enumerate() {
            let whx = wh[done + x];
            *o = (whx * lv + (256 - whx) * right + 128) >> 8;
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn vertical_avx2(bw: usize, bh: usize, top: &[i32], _left: &[i32], out: &mut [i32]) {
    let (top8, rest) = top[..bw].as_chunks::<8>();
    let (top4, top_tail) = rest.as_chunks::<4>();
    for row in out.chunks_exact_mut(bw).take(bh) {
        let (out8, rest) = row.as_chunks_mut::<8>();
        for (o, t) in out8.iter_mut().zip(top8) {
            store_i32x8(o, load_i32x8(t));
        }
        let (out4, out_tail) = rest.as_chunks_mut::<4>();
        for (o, t) in out4.iter_mut().zip(top4) {
            store_i32x4(o, load_i32x4(t));
        }
        out_tail.copy_from_slice(top_tail);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn horizontal_avx2(bw: usize, bh: usize, _top: &[i32], left: &[i32], out: &mut [i32]) {
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let (out8, rest) = row.as_chunks_mut::<8>();
        let v8 = _mm256_set1_epi32(lv);
        for o in out8 {
            store_i32x8(o, v8);
        }
        let (out4, out_tail) = rest.as_chunks_mut::<4>();
        let v4 = _mm_set1_epi32(lv);
        for o in out4 {
            store_i32x4(o, v4);
        }
        out_tail.fill(lv);
    }
}

/// 5-tap edge-smoothing convolution over the clamp-free middle run:
/// `out[t] = (Σ_j k[j] * win[t + j] + 8) >> 4`. `win` must hold
/// `out.len() + 4` samples.
#[target_feature(enable = "avx2")]
pub(crate) fn edge_conv5_avx2(win: &[i32], k: &[i32; 5], out: &mut [i32]) {
    let rnd8 = _mm256_set1_epi32(8);
    let rnd4 = _mm_set1_epi32(8);
    let (out8, rest) = out.as_chunks_mut::<8>();
    let taps8 = shifted_chunks::<5, 8>(win);
    for (i, o) in out8.iter_mut().enumerate() {
        let mut acc = rnd8;
        for (t, &kj) in taps8.iter().zip(k.iter()) {
            acc = mla8_n(acc, load_i32x8(&t[i]), kj);
        }
        store_i32x8(o, _mm256_srai_epi32::<4>(acc));
    }
    let done8 = out8.len() * 8;
    let (out4, out_tail) = rest.as_chunks_mut::<4>();
    let taps4 = shifted_chunks::<5, 4>(&win[done8..]);
    for (i, o) in out4.iter_mut().enumerate() {
        let mut acc = rnd4;
        for (t, &kj) in taps4.iter().zip(k.iter()) {
            acc = mla4_n(acc, load_i32x4(&t[i]), kj);
        }
        store_i32x4(o, _mm_srai_epi32::<4>(acc));
    }
    let done = done8 + out4.len() * 4;
    for (o, t) in out_tail.iter_mut().zip(done..) {
        let sum: i32 = k.iter().enumerate().map(|(j, &kj)| kj * win[t + j]).sum();
        *o = (sum + 8) >> 4;
    }
}

/// The 4x2-cell recursive filter-intra pass over `buf` (33x33, row 0 and
/// column 0 hold the references). One cell's eight outputs are a single
/// i32x8 accumulated from seven broadcast samples times transposed taps.
#[target_feature(enable = "avx2")]
pub(crate) fn filter_intra_cells_avx2(
    buf: &mut [[i32; 33]; 33],
    taps: &[[i8; 7]; 8],
    width: usize,
    height: usize,
    max_sample: i32,
) {
    // Transpose taps to per-input vectors across the eight cell outputs
    // (outputs 0..4 are row r, 4..8 are row r + 1).
    let mut tv = [[0i32; 8]; 7];
    for (k, filter) in taps.iter().enumerate() {
        for (j, &tap) in filter.iter().enumerate() {
            tv[j][k] = tap as i32;
        }
    }
    let tvv: [__m256i; 7] = std::array::from_fn(|j| load_i32x8(&tv[j]));
    let rnd = _mm256_set1_epi32(8);
    let zero = _mm256_setzero_si256();
    let maxv = _mm256_set1_epi32(max_sample);
    let shift = _mm_cvtsi32_si128(crate::tables::INTRA_FILTER_SCALE_BITS as i32);
    for r in (1..=height).step_by(2) {
        let (above, rest) = buf.split_at_mut(r);
        let (row_lo, row_hi) = rest.split_at_mut(1);
        let p_above = &above[r - 1];
        // A cell's left reference is the last column the previous cell wrote
        // (the recursion); carry it forward instead of re-reading the row.
        let (mut left_lo, mut left_hi) = (row_lo[0][0], row_hi[0][0]);
        let (cells_lo, _) = row_lo[0][1..].as_chunks_mut::<4>();
        let (cells_hi, _) = row_hi[0][1..].as_chunks_mut::<4>();
        for (i, (clo, chi)) in cells_lo
            .iter_mut()
            .zip(cells_hi.iter_mut())
            .take(width / 4)
            .enumerate()
        {
            let c = 1 + 4 * i;
            let p = [
                p_above[c - 1],
                p_above[c],
                p_above[c + 1],
                p_above[c + 2],
                p_above[c + 3],
                left_lo,
                left_hi,
            ];
            let mut acc = rnd;
            for (&pj, &tj) in p.iter().zip(tvv.iter()) {
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(tj, _mm256_set1_epi32(pj)));
            }
            let v = _mm256_sra_epi32(acc, shift);
            let v = _mm256_min_epi32(_mm256_max_epi32(v, zero), maxv);
            store_i32x4(clo, _mm256_castsi256_si128(v));
            store_i32x4(chi, _mm256_extracti128_si256::<1>(v));
            (left_lo, left_hi) = (clo[3], chi[3]);
        }
    }
}

/// Interpolate one contiguous output run. `e` starts at its first base sample
/// and must hold `(out.len() - 1) * step + 2` samples.
#[target_feature(enable = "avx2")]
fn dr_interp_avx2(e: &[i32], step: usize, shift: i32, out: &mut [i32]) {
    let (shv8, ishv8) = (_mm256_set1_epi32(shift), _mm256_set1_epi32(32 - shift));
    let (shv4, ishv4) = (_mm_set1_epi32(shift), _mm_set1_epi32(32 - shift));
    let (rnd8, rnd4) = (_mm256_set1_epi32(16), _mm_set1_epi32(16));
    let done = if step == 1 {
        // Consecutive samples: taps at offsets 0 and 1.
        let (out8, rest) = out.as_chunks_mut::<8>();
        let taps8 = shifted_chunks::<2, 8>(e);
        for (i, o) in out8.iter_mut().enumerate() {
            let a = _mm256_mullo_epi32(load_i32x8(&taps8[0][i]), ishv8);
            let b = _mm256_mullo_epi32(load_i32x8(&taps8[1][i]), shv8);
            let v = _mm256_add_epi32(_mm256_add_epi32(a, b), rnd8);
            store_i32x8(o, _mm256_srai_epi32::<5>(v));
        }
        let d = out8.len() * 8;
        let (out4, _) = rest.as_chunks_mut::<4>();
        let taps4 = shifted_chunks::<2, 4>(&e[d..]);
        for (i, o) in out4.iter_mut().enumerate() {
            let a = _mm_mullo_epi32(load_i32x4(&taps4[0][i]), ishv4);
            let b = _mm_mullo_epi32(load_i32x4(&taps4[1][i]), shv4);
            let v = _mm_add_epi32(_mm_add_epi32(a, b), rnd4);
            store_i32x4(o, _mm_srai_epi32::<5>(v));
        }
        d + out4.len() * 4
    } else {
        // Upsampled edge: four outputs consume eight samples. De-interleave
        // the pair of 4-wide loads into evens (base) and odds (base + 1).
        let (out4, _) = out.as_chunks_mut::<4>();
        let (e8, _) = e.as_chunks::<8>();
        for (o, src) in out4.iter_mut().zip(e8) {
            let (lo, hi) = src.split_at(4);
            let a = load_i32x4(lo.first_chunk::<4>().unwrap());
            let b = load_i32x4(hi.first_chunk::<4>().unwrap());
            let (af, bf) = (_mm_castsi128_ps(a), _mm_castsi128_ps(b));
            let ev = _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(af, bf));
            let od = _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(af, bf));
            let v = _mm_add_epi32(
                _mm_add_epi32(_mm_mullo_epi32(ev, ishv4), _mm_mullo_epi32(od, shv4)),
                rnd4,
            );
            store_i32x4(o, _mm_srai_epi32::<5>(v));
        }
        out4.len() * 4
    };
    for (o, j) in out[done..].iter_mut().zip(done..) {
        let b = j * step;
        *o = (e[b] * (32 - shift) + e[b + 1] * shift + 16) >> 5;
    }
}

#[inline]
fn dr_edge(edge: &[i32], index: i32) -> i32 {
    edge[(index + DR_EDGE_ORIGIN) as usize]
}

#[inline]
#[target_feature(enable = "avx2")]
fn dr_interp_gather8(
    edge: &[i32],
    a_index: [i32; 8],
    b_index: [i32; 8],
    shift: [i32; 8],
) -> __m256i {
    let a = unsafe { _mm256_i32gather_epi32::<4>(edge.as_ptr(), load_i32x8(&a_index)) };
    let b = unsafe { _mm256_i32gather_epi32::<4>(edge.as_ptr(), load_i32x8(&b_index)) };
    let shift = load_i32x8(&shift);
    let inverse = _mm256_sub_epi32(_mm256_set1_epi32(32), shift);
    _mm256_srai_epi32::<5>(_mm256_add_epi32(
        _mm256_add_epi32(_mm256_mullo_epi32(a, inverse), _mm256_mullo_epi32(b, shift)),
        _mm256_set1_epi32(16),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
fn dr_interp_gather4(
    edge: &[i32],
    a_index: [i32; 4],
    b_index: [i32; 4],
    shift: [i32; 4],
) -> __m128i {
    let a = unsafe { _mm_i32gather_epi32::<4>(edge.as_ptr(), load_i32x4(&a_index)) };
    let b = unsafe { _mm_i32gather_epi32::<4>(edge.as_ptr(), load_i32x4(&b_index)) };
    let shift = load_i32x4(&shift);
    let inverse = _mm_sub_epi32(_mm_set1_epi32(32), shift);
    _mm_srai_epi32::<5>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a, inverse), _mm_mullo_epi32(b, shift)),
        _mm_set1_epi32(16),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
fn dr_interp_column4(
    edge: &[i32],
    base: i32,
    step: usize,
    shift: i32,
    max_base: i32,
    fill: i32,
) -> __m128i {
    let last_base = base + 3 * step as i32;
    let (a, b) = if last_base < max_base {
        let start = (base + DR_EDGE_ORIGIN) as usize;
        if step == 1 {
            (
                load_i32x4(edge[start..].first_chunk::<4>().unwrap()),
                load_i32x4(edge[start + 1..].first_chunk::<4>().unwrap()),
            )
        } else {
            let src = edge[start..].first_chunk::<8>().unwrap();
            let (lo, hi) = src.split_at(4);
            let a = load_i32x4(lo.first_chunk::<4>().unwrap());
            let b = load_i32x4(hi.first_chunk::<4>().unwrap());
            let (a, b) = (_mm_castsi128_ps(a), _mm_castsi128_ps(b));
            (
                _mm_castps_si128(_mm_shuffle_ps::<0b10_00_10_00>(a, b)),
                _mm_castps_si128(_mm_shuffle_ps::<0b11_01_11_01>(a, b)),
            )
        }
    } else {
        let mut a = [fill; 4];
        let mut b = [fill; 4];
        for (lane, (a, b)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            let index = base + lane as i32 * step as i32;
            if index < max_base {
                *a = dr_edge(edge, index);
                *b = dr_edge(edge, index + 1);
            }
        }
        (load_i32x4(&a), load_i32x4(&b))
    };
    let shift = _mm_set1_epi32(shift);
    let inverse = _mm_sub_epi32(_mm_set1_epi32(32), shift);
    _mm_srai_epi32::<5>(_mm_add_epi32(
        _mm_add_epi32(_mm_mullo_epi32(a, inverse), _mm_mullo_epi32(b, shift)),
        _mm_set1_epi32(16),
    ))
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4(c0: __m128i, c1: __m128i, c2: __m128i, c3: __m128i) -> [__m128i; 4] {
    let a = _mm_unpacklo_epi32(c0, c1);
    let b = _mm_unpackhi_epi32(c0, c1);
    let c = _mm_unpacklo_epi32(c2, c3);
    let d = _mm_unpackhi_epi32(c2, c3);
    [
        _mm_unpacklo_epi64(a, c),
        _mm_unpackhi_epi64(a, c),
        _mm_unpacklo_epi64(b, d),
        _mm_unpackhi_epi64(b, d),
    ]
}

#[target_feature(enable = "avx2")]
pub(crate) fn dr_predict_avx2(p: DrPrediction, above: &[i32], left: &[i32], out: &mut [i32]) {
    debug_assert!(out.len() >= p.bw * p.bh);
    match p.zone {
        DR_ZONE1 => {
            let max_base = (p.edge_len as i32 - 1) << p.up_above;
            let frac_bits = 6 - p.up_above;
            let step = 1usize << p.up_above;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let xpos = p.dx * (y as i32 + 1);
                let shift = ((xpos << p.up_above) & 0x3f) >> 1;
                let base = xpos >> frac_bits;
                let n_fit = if base >= max_base {
                    0
                } else {
                    ((max_base - base + step as i32 - 1) / step as i32).min(p.bw as i32) as usize
                };
                dr_interp_avx2(
                    &above[(base + DR_EDGE_ORIGIN) as usize..],
                    step,
                    shift,
                    &mut row[..n_fit],
                );
                row[n_fit..].fill(dr_edge(above, max_base));
            }
        }
        DR_ZONE3 => {
            let max_base = (p.edge_len as i32 - 1) << p.up_left;
            let frac_bits = 6 - p.up_left;
            let step = 1usize << p.up_left;
            let fill = dr_edge(left, max_base);
            debug_assert_eq!(p.bw & 3, 0);
            debug_assert_eq!(p.bh & 3, 0);
            for (y4, block) in out[..p.bw * p.bh].chunks_exact_mut(p.bw * 4).enumerate() {
                let (row0, block) = block.split_at_mut(p.bw);
                let (row1, block) = block.split_at_mut(p.bw);
                let (row2, row3) = block.split_at_mut(p.bw);
                let (row0, _) = row0.as_chunks_mut::<4>();
                let (row1, _) = row1.as_chunks_mut::<4>();
                let (row2, _) = row2.as_chunks_mut::<4>();
                let (row3, _) = row3.as_chunks_mut::<4>();
                for (x4, (((dst0, dst1), dst2), dst3)) in
                    row0.iter_mut().zip(row1).zip(row2).zip(row3).enumerate()
                {
                    let y = y4 * 4;
                    let x = x4 * 4;
                    let mut columns = [_mm_setzero_si128(); 4];
                    for (lane, column) in columns.iter_mut().enumerate() {
                        let ypos = p.dy * (x as i32 + lane as i32 + 1);
                        let base = (ypos >> frac_bits) + y as i32 * step as i32;
                        let shift = ((ypos << p.up_left) & 0x3f) >> 1;
                        *column = dr_interp_column4(left, base, step, shift, max_base, fill);
                    }
                    let [v0, v1, v2, v3] =
                        transpose_4x4(columns[0], columns[1], columns[2], columns[3]);
                    store_i32x4(dst0, v0);
                    store_i32x4(dst1, v1);
                    store_i32x4(dst2, v2);
                    store_i32x4(dst3, v3);
                }
            }
        }
        DR_ZONE2 => {
            let frac_bits_y = 6 - p.up_left;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let t = (y as i32 + 1) * p.dx - 64;
                let x0 = if t <= 0 {
                    0
                } else {
                    (((t + 63) >> 6) as usize).min(p.bw)
                };
                let (left8, rest) = row[..x0].as_chunks_mut::<8>();
                for (chunk, dst) in left8.iter_mut().enumerate() {
                    let mut ai = [0i32; 8];
                    let mut bi = [0i32; 8];
                    let mut shifts = [0i32; 8];
                    for lane in 0..8 {
                        let x = chunk * 8 + lane;
                        let ypos = ((y as i32) << 6) - (x as i32 + 1) * p.dy;
                        let base = ypos >> frac_bits_y;
                        ai[lane] = base + DR_EDGE_ORIGIN;
                        bi[lane] = base + DR_EDGE_ORIGIN + 1;
                        shifts[lane] = ((ypos * (1 << p.up_left)) & 0x3f) >> 1;
                    }
                    store_i32x8(dst, dr_interp_gather8(left, ai, bi, shifts));
                }
                let done8 = left8.len() * 8;
                let (left4, left_tail) = rest.as_chunks_mut::<4>();
                for (chunk, dst) in left4.iter_mut().enumerate() {
                    let mut ai = [0i32; 4];
                    let mut bi = [0i32; 4];
                    let mut shifts = [0i32; 4];
                    for lane in 0..4 {
                        let x = done8 + chunk * 4 + lane;
                        let ypos = ((y as i32) << 6) - (x as i32 + 1) * p.dy;
                        let base = ypos >> frac_bits_y;
                        ai[lane] = base + DR_EDGE_ORIGIN;
                        bi[lane] = base + DR_EDGE_ORIGIN + 1;
                        shifts[lane] = ((ypos * (1 << p.up_left)) & 0x3f) >> 1;
                    }
                    store_i32x4(dst, dr_interp_gather4(left, ai, bi, shifts));
                }
                let done = done8 + left4.len() * 4;
                for (lane, dst) in left_tail.iter_mut().enumerate() {
                    let x = done + lane;
                    let ypos = ((y as i32) << 6) - (x as i32 + 1) * p.dy;
                    let base = ypos >> frac_bits_y;
                    let shift = ((ypos * (1 << p.up_left)) & 0x3f) >> 1;
                    *dst =
                        (dr_edge(left, base) * (32 - shift) + dr_edge(left, base + 1) * shift + 16)
                            >> 5;
                }
                if x0 < p.bw {
                    let xpos = ((x0 as i32) << 6) - (y as i32 + 1) * p.dx;
                    let base = xpos >> (6 - p.up_above);
                    let shift = ((xpos * (1 << p.up_above)) & 0x3f) >> 1;
                    dr_interp_avx2(
                        &above[(base + DR_EDGE_ORIGIN) as usize..],
                        1usize << p.up_above,
                        shift,
                        &mut row[x0..],
                    );
                }
            }
        }
        _ => unreachable!("invalid directional prediction zone"),
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn paeth_avx2(
    bw: usize,
    bh: usize,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [i32],
) {
    let (cn8, cn4) = (_mm256_set1_epi32(corner), _mm_set1_epi32(corner));
    let (top8, top4) = row_views(top, bw);
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let (lvv8, lvv4) = (_mm256_set1_epi32(lv), _mm_set1_epi32(lv));
        let (lmc8, lmc4) = (_mm256_set1_epi32(lv - corner), _mm_set1_epi32(lv - corner));
        let (o8, rest) = row.as_chunks_mut::<8>();
        for (o, t) in o8.iter_mut().zip(top8) {
            let tv = load_i32x8(t);
            let b = _mm256_add_epi32(tv, lmc8); // lv + tv - corner
            let (ld, td, cd) = (absdiff8(lvv8, b), absdiff8(tv, b), absdiff8(cn8, b));
            let m_tv = le8(td, cd); // td <= cd
            let m_lv = _mm256_and_si256(le8(ld, td), le8(ld, cd)); // ld<=td && ld<=cd
            let res = _mm256_blendv_epi8(cn8, tv, m_tv); // td<=cd ? tv : corner
            store_i32x8(o, _mm256_blendv_epi8(res, lvv8, m_lv)); // ld<=..? lv : res
        }
        let (o4, tail) = rest.as_chunks_mut::<4>();
        for (o, t) in o4.iter_mut().zip(top4) {
            let tv = load_i32x4(t);
            let b = _mm_add_epi32(tv, lmc4);
            let (ld, td, cd) = (absdiff4(lvv4, b), absdiff4(tv, b), absdiff4(cn4, b));
            let m_tv = le4(td, cd);
            let m_lv = _mm_and_si128(le4(ld, td), le4(ld, cd));
            let res = _mm_blendv_epi8(cn4, tv, m_tv);
            store_i32x4(o, _mm_blendv_epi8(res, lvv4, m_lv));
        }
        let done = o8.len() * 8 + o4.len() * 4;
        for (x, o) in tail.iter_mut().enumerate() {
            let tv = top[done + x];
            let b = lv + tv - corner;
            let (ld, td, cd) = ((lv - b).abs(), (tv - b).abs(), (corner - b).abs());
            *o = if ld <= td && ld <= cd {
                lv
            } else if td <= cd {
                tv
            } else {
                corner
            };
        }
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_ac_444_u16_avx2(luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
    let n = w * h;
    debug_assert!(luma_rec.len() >= n);
    debug_assert!(ac.len() >= n);
    debug_assert!(luma_rec[..n].iter().all(|&value| value <= 4095));

    let (src16, src_tail16) = luma_rec[..n].as_chunks::<16>();
    let (dst16, dst_tail16) = ac[..n].as_chunks_mut::<16>();
    let mut sum_v = _mm256_setzero_si256();
    for (src, dst) in src16.iter().zip(dst16) {
        let pixels = unsafe { _mm256_loadu_si256(src.as_ptr().cast::<__m256i>()) };
        let (dst8, _) = dst.as_chunks_mut::<8>();
        sum_v = store_cfl_q3_8(
            &mut dst8[0],
            _mm256_slli_epi32::<3>(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(pixels))),
            sum_v,
        );
        sum_v = store_cfl_q3_8(
            &mut dst8[1],
            _mm256_slli_epi32::<3>(_mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(pixels))),
            sum_v,
        );
    }

    let (src8, src_tail) = src_tail16.as_chunks::<8>();
    let (dst8, dst_tail) = dst_tail16.as_chunks_mut::<8>();
    for (src, dst) in src8.iter().zip(dst8) {
        let pixels = unsafe { _mm_loadu_si128(src.as_ptr().cast::<__m128i>()) };
        sum_v = store_cfl_q3_8(
            dst,
            _mm256_slli_epi32::<3>(_mm256_cvtepu16_epi32(pixels)),
            sum_v,
        );
    }
    let mut sum = sum_i64x4(sum_v) as u64;
    for (&src, dst) in src_tail.iter().zip(dst_tail) {
        *dst = i32::from(src) << 3;
        sum += *dst as u64;
    }

    subtract_cfl_mean(&mut ac[..n], sum, w.trailing_zeros() + h.trailing_zeros());
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "avx2")]
pub(crate) fn cfl_ac_sub_u16_avx2(
    luma_rec: &[u16],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    let n = cw * ch;
    let sx = usize::from(ss_hor);
    let sy = usize::from(ss_ver);
    let lw = cw << sx;
    let lh = ch << sy;
    debug_assert!(cw > 0 && ch > 0);
    debug_assert!(lstride >= lw);
    debug_assert!(luma_rec.len() >= (lh - 1) * lstride + lw);
    debug_assert!(ac.len() >= n);
    debug_assert!(
        luma_rec[..(lh - 1) * lstride + lw]
            .iter()
            .all(|&value| value <= 4095)
    );

    let shift = 1 + u32::from(!ss_ver) + u32::from(!ss_hor);
    let ones8 = _mm256_set1_epi16(1);
    let ones4 = _mm_set1_epi16(1);
    let mut sum_v = _mm256_setzero_si256();
    let mut scalar_sum = 0u64;
    for y in 0..ch {
        let ly = y << sy;
        let top = &luma_rec[ly * lstride..ly * lstride + lw];
        let out = &mut ac[y * cw..(y + 1) * cw];

        let done = match (ss_hor, ss_ver) {
            (true, true) => {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                let (out8, rest) = out.as_chunks_mut::<8>();
                let (top16, _) = top.as_chunks::<16>();
                let (bottom16, _) = bottom.as_chunks::<16>();
                for ((dst, top), bottom) in out8.iter_mut().zip(top16).zip(bottom16) {
                    let top_pairs = _mm256_madd_epi16(
                        unsafe { _mm256_loadu_si256(top.as_ptr().cast::<__m256i>()) },
                        ones8,
                    );
                    let bottom_pairs = _mm256_madd_epi16(
                        unsafe { _mm256_loadu_si256(bottom.as_ptr().cast::<__m256i>()) },
                        ones8,
                    );
                    sum_v = store_cfl_q3_8(
                        dst,
                        _mm256_slli_epi32::<1>(_mm256_add_epi32(top_pairs, bottom_pairs)),
                        sum_v,
                    );
                }
                let done8 = out8.len() * 8;
                let (out4, _) = rest.as_chunks_mut::<4>();
                let (top8, _) = top[done8 * 2..].as_chunks::<8>();
                let (bottom8, _) = bottom[done8 * 2..].as_chunks::<8>();
                for ((dst, top), bottom) in out4.iter_mut().zip(top8).zip(bottom8) {
                    let top_pairs = _mm_madd_epi16(
                        unsafe { _mm_loadu_si128(top.as_ptr().cast::<__m128i>()) },
                        ones4,
                    );
                    let bottom_pairs = _mm_madd_epi16(
                        unsafe { _mm_loadu_si128(bottom.as_ptr().cast::<__m128i>()) },
                        ones4,
                    );
                    sum_v = store_cfl_q3_4(
                        dst,
                        _mm_slli_epi32::<1>(_mm_add_epi32(top_pairs, bottom_pairs)),
                        sum_v,
                    );
                }
                done8 + out4.len() * 4
            }
            (true, false) => {
                let (out8, rest) = out.as_chunks_mut::<8>();
                let (top16, _) = top.as_chunks::<16>();
                for (dst, top) in out8.iter_mut().zip(top16) {
                    let pairs = _mm256_madd_epi16(
                        unsafe { _mm256_loadu_si256(top.as_ptr().cast::<__m256i>()) },
                        ones8,
                    );
                    sum_v = store_cfl_q3_8(dst, _mm256_slli_epi32::<2>(pairs), sum_v);
                }
                let done8 = out8.len() * 8;
                let (out4, _) = rest.as_chunks_mut::<4>();
                let (top8, _) = top[done8 * 2..].as_chunks::<8>();
                for (dst, top) in out4.iter_mut().zip(top8) {
                    let pairs = _mm_madd_epi16(
                        unsafe { _mm_loadu_si128(top.as_ptr().cast::<__m128i>()) },
                        ones4,
                    );
                    sum_v = store_cfl_q3_4(dst, _mm_slli_epi32::<2>(pairs), sum_v);
                }
                done8 + out4.len() * 4
            }
            (false, true) => {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                let (out8, rest) = out.as_chunks_mut::<8>();
                let (top8, _) = top.as_chunks::<8>();
                let (bottom8, _) = bottom.as_chunks::<8>();
                for ((dst, top), bottom) in out8.iter_mut().zip(top8).zip(bottom8) {
                    let top_pixels = _mm256_cvtepu16_epi32(unsafe {
                        _mm_loadu_si128(top.as_ptr().cast::<__m128i>())
                    });
                    let bottom_pixels = _mm256_cvtepu16_epi32(unsafe {
                        _mm_loadu_si128(bottom.as_ptr().cast::<__m128i>())
                    });
                    sum_v = store_cfl_q3_8(
                        dst,
                        _mm256_slli_epi32::<2>(_mm256_add_epi32(top_pixels, bottom_pixels)),
                        sum_v,
                    );
                }
                let done8 = out8.len() * 8;
                let (out4, _) = rest.as_chunks_mut::<4>();
                let (top4, _) = top[done8..].as_chunks::<4>();
                let (bottom4, _) = bottom[done8..].as_chunks::<4>();
                for ((dst, top), bottom) in out4.iter_mut().zip(top4).zip(bottom4) {
                    let top_pixels = _mm_cvtepu16_epi32(unsafe {
                        _mm_loadl_epi64(top.as_ptr().cast::<__m128i>())
                    });
                    let bottom_pixels = _mm_cvtepu16_epi32(unsafe {
                        _mm_loadl_epi64(bottom.as_ptr().cast::<__m128i>())
                    });
                    sum_v = store_cfl_q3_4(
                        dst,
                        _mm_slli_epi32::<2>(_mm_add_epi32(top_pixels, bottom_pixels)),
                        sum_v,
                    );
                }
                done8 + out4.len() * 4
            }
            (false, false) => {
                let (out8, rest) = out.as_chunks_mut::<8>();
                let (top8, _) = top.as_chunks::<8>();
                for (dst, top) in out8.iter_mut().zip(top8) {
                    let pixels = _mm256_cvtepu16_epi32(unsafe {
                        _mm_loadu_si128(top.as_ptr().cast::<__m128i>())
                    });
                    sum_v = store_cfl_q3_8(dst, _mm256_slli_epi32::<3>(pixels), sum_v);
                }
                let done8 = out8.len() * 8;
                let (out4, _) = rest.as_chunks_mut::<4>();
                let (top4, _) = top[done8..].as_chunks::<4>();
                for (dst, top) in out4.iter_mut().zip(top4) {
                    let pixels = _mm_cvtepu16_epi32(unsafe {
                        _mm_loadl_epi64(top.as_ptr().cast::<__m128i>())
                    });
                    sum_v = store_cfl_q3_4(dst, _mm_slli_epi32::<3>(pixels), sum_v);
                }
                done8 + out4.len() * 4
            }
        };

        for (x, dst) in out[done..].iter_mut().enumerate() {
            let x = done + x;
            let lx = x << sx;
            let mut value = i32::from(top[lx]);
            if ss_hor {
                value += i32::from(top[lx + 1]);
            }
            if ss_ver {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                value += i32::from(bottom[lx]);
                if ss_hor {
                    value += i32::from(bottom[lx + 1]);
                }
            }
            *dst = value << shift;
            scalar_sum += *dst as u64;
        }
    }

    subtract_cfl_mean(
        &mut ac[..n],
        sum_i64x4(sum_v) as u64 + scalar_sum,
        cw.trailing_zeros() + ch.trailing_zeros(),
    );
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_pred_avx2(dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
    debug_assert!(ac.len() >= dst.len());
    debug_assert!((-16..=16).contains(&alpha));
    debug_assert!((8..=12).contains(&bd));
    debug_assert!((0..=(1 << bd) - 1).contains(&dc));
    debug_assert!(
        ac[..dst.len()]
            .iter()
            .all(|&value| i16::try_from(value).is_ok())
    );

    let len = dst.len();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    let (ac8, ac_tail) = ac[..len].as_chunks::<8>();
    let dc_v = _mm_set1_epi16(dc as i16);
    let max_v = _mm_set1_epi16(((1 << bd) - 1) as i16);
    let alpha_sign = _mm_set1_epi16(alpha as i16);
    let abs_alpha_q12 = (alpha.abs() << 9) as i16;
    for (out, input) in dst8.iter_mut().zip(ac8) {
        store_i16x8_as_i32(
            out,
            cfl_predict_i16(
                load_i32x8_as_i16(input),
                dc_v,
                max_v,
                alpha_sign,
                abs_alpha_q12,
            ),
        );
    }
    for (out, &input) in dst_tail.iter_mut().zip(ac_tail) {
        *out = crate::intrapred::cfl_pred_pixel(dc, input, alpha, bd);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_best_alpha_u16_avx2(ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
    debug_assert!(ac.len() >= n);
    debug_assert!(src.len() >= n);
    debug_assert!((8..=12).contains(&bd));
    let max = (1 << bd) - 1;
    debug_assert!((0..=max).contains(&dc));
    debug_assert!(ac[..n].iter().all(|&value| i16::try_from(value).is_ok()));
    debug_assert!(src[..n].iter().all(|&value| i32::from(value) <= max));

    let (ac8, ac_tail) = ac[..n].as_chunks::<8>();
    let (src8, src_tail) = src[..n].as_chunks::<8>();
    let dc_v = _mm_set1_epi16(dc as i16);
    let mut numerator_v = _mm256_setzero_si256();
    let mut denominator_v = _mm256_setzero_si256();
    // CfL AC is bounded by +/-32760 at 12-bit. PMADDWD's largest
    // denominator lane is therefore 2 * 32760^2 = 2_146_435_200, below
    // i32::MAX; each four-lane result is widened to i64 immediately.
    for (ac_chunk, src_chunk) in ac8.iter().zip(src8) {
        let ac_v = load_i32x8_as_i16(ac_chunk);
        let src_v = unsafe { _mm_loadu_si128(src_chunk.as_ptr().cast::<__m128i>()) };
        numerator_v = accumulate_i32x4_i64(
            numerator_v,
            _mm_madd_epi16(_mm_sub_epi16(src_v, dc_v), ac_v),
        );
        denominator_v = accumulate_i32x4_i64(denominator_v, _mm_madd_epi16(ac_v, ac_v));
    }
    let mut numerator = sum_i64x4(numerator_v);
    let mut denominator = sum_i64x4(denominator_v);
    for (&ac, &src) in ac_tail.iter().zip(src_tail) {
        numerator += i64::from(i32::from(src) - dc) * i64::from(ac);
        denominator += i64::from(ac) * i64::from(ac);
    }
    if denominator == 0 {
        return 0;
    }

    let a0 = ((64 * numerator + (denominator >> 1) * numerator.signum()) / denominator)
        .clamp(-16, 16) as i32;
    let max_v = _mm_set1_epi16(max as i16);
    let mut best_alpha = 0;
    let mut best_error = i64::MAX;
    for alpha in (a0 - 3)..=(a0 + 3) {
        if !(-16..=16).contains(&alpha) {
            continue;
        }
        let alpha_sign = _mm_set1_epi16(alpha as i16);
        let abs_alpha_q12 = (alpha.abs() << 9) as i16;
        let mut error_v = _mm256_setzero_si256();
        for (ac_chunk, src_chunk) in ac8.iter().zip(src8) {
            let pred = cfl_predict_i16(
                load_i32x8_as_i16(ac_chunk),
                dc_v,
                max_v,
                alpha_sign,
                abs_alpha_q12,
            );
            let src_v = unsafe { _mm_loadu_si128(src_chunk.as_ptr().cast::<__m128i>()) };
            let residual = _mm_sub_epi16(src_v, pred);
            error_v = accumulate_i32x4_i64(error_v, _mm_madd_epi16(residual, residual));
        }
        let mut error = sum_i64x4(error_v);
        for (&ac, &src) in ac_tail.iter().zip(src_tail) {
            let residual =
                i64::from(i32::from(src) - crate::intrapred::cfl_pred_pixel(dc, ac, alpha, bd));
            error += residual * residual;
        }
        if error < best_error {
            best_error = error;
            best_alpha = alpha;
        }
    }
    best_alpha
}
