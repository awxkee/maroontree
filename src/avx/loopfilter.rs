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

use crate::loopfilter::{WIDE6_WEIGHTS, WIDE8_WEIGHTS, WIDE16_WEIGHTS};
use core::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2")]
fn load_sample(
    dst: &[u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
) -> __m128i {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        let packed = unsafe { _mm_loadl_epi64(dst.as_ptr().add(pos).cast()) };
        _mm_cvtepu16_epi32(packed)
    } else {
        _mm_setr_epi32(
            dst[(base as isize + offset * stride_b) as usize] as i32,
            dst[(base as isize + stride_a + offset * stride_b) as usize] as i32,
            dst[(base as isize + 2 * stride_a + offset * stride_b) as usize] as i32,
            dst[(base as isize + 3 * stride_a + offset * stride_b) as usize] as i32,
        )
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_sample(
    dst: &mut [u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
    value: __m128i,
) {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        let packed = _mm_packus_epi32(value, _mm_setzero_si128());
        unsafe { _mm_storel_epi64(dst.as_mut_ptr().add(pos).cast(), packed) };
    } else {
        let pos = base as isize + offset * stride_b;
        dst[pos as usize] = _mm_extract_epi32::<0>(value) as u16;
        dst[(pos + stride_a) as usize] = _mm_extract_epi32::<1>(value) as u16;
        dst[(pos + 2 * stride_a) as usize] = _mm_extract_epi32::<2>(value) as u16;
        dst[(pos + 3 * stride_a) as usize] = _mm_extract_epi32::<3>(value) as u16;
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_vertical_4_lines(dst: &[u16], base: usize, stride: isize, offset: isize) -> [__m128i; 4] {
    let src = unsafe { dst.as_ptr().add((base as isize + offset) as usize) };
    let rows: [__m128i; 4] = std::array::from_fn(|row| unsafe {
        _mm_loadl_epi64(src.offset(row as isize * stride).cast())
    });
    let a0 = _mm_unpacklo_epi16(rows[0], rows[1]);
    let a1 = _mm_unpacklo_epi16(rows[2], rows[3]);
    let b0 = _mm_unpacklo_epi32(a0, a1);
    let b1 = _mm_unpackhi_epi32(a0, a1);
    [
        _mm_cvtepu16_epi32(b0),
        _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b0)),
        _mm_cvtepu16_epi32(b1),
        _mm_cvtepu16_epi32(_mm_srli_si128::<8>(b1)),
    ]
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_vertical_edge(dst: &[u16], base: usize, stride: isize, wd: i32) -> [__m128i; 14] {
    let mut samples = [_mm_setzero_si128(); 14];
    match wd {
        4 => samples[5..9].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -2)),
        6 => {
            samples[4..8].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -3));
            samples[6..10].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -1));
        }
        8 => {
            samples[3..7].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -4));
            samples[7..11].copy_from_slice(&load_vertical_4_lines(dst, base, stride, 0));
        }
        16 => {
            samples[0..4].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -7));
            samples[4..8].copy_from_slice(&load_vertical_4_lines(dst, base, stride, -3));
            samples[8..12].copy_from_slice(&load_vertical_4_lines(dst, base, stride, 1));
            samples[10..14].copy_from_slice(&load_vertical_4_lines(dst, base, stride, 3));
        }
        _ => unreachable!("unsupported loop-filter width"),
    }
    samples
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_vertical_4_lines(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[__m128i],
) {
    let a0 = _mm_unpacklo_epi32(values[0], values[1]);
    let a1 = _mm_unpackhi_epi32(values[0], values[1]);
    let a2 = _mm_unpacklo_epi32(values[2], values[3]);
    let a3 = _mm_unpackhi_epi32(values[2], values[3]);
    let rows = [
        _mm_unpacklo_epi64(a0, a2),
        _mm_unpackhi_epi64(a0, a2),
        _mm_unpacklo_epi64(a1, a3),
        _mm_unpackhi_epi64(a1, a3),
    ];
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for (row, value) in rows.iter().enumerate() {
        let packed = _mm_packus_epi32(*value, *value);
        unsafe { _mm_storel_epi64(dst.offset(row as isize * stride).cast(), packed) };
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_vertical_2_lines(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[__m128i],
) {
    let zero = _mm_setzero_si128();
    let a = _mm_packus_epi32(values[0], zero);
    let b = _mm_packus_epi32(values[1], zero);
    let pairs = _mm_unpacklo_epi16(a, b);
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for row in 0..4 {
        let value = match row {
            0 => _mm_extract_epi32::<0>(pairs),
            1 => _mm_extract_epi32::<1>(pairs),
            2 => _mm_extract_epi32::<2>(pairs),
            _ => _mm_extract_epi32::<3>(pairs),
        };
        unsafe {
            dst.offset(row as isize * stride)
                .cast::<i32>()
                .write_unaligned(value);
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn abs_diff(a: __m128i, b: __m128i) -> __m128i {
    _mm_abs_epi32(_mm_sub_epi32(a, b))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mask_not(a: __m128i) -> __m128i {
    _mm_xor_si128(a, _mm_set1_epi32(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mask_any(a: __m128i) -> bool {
    _mm_testz_si128(a, a) == 0
}

#[inline]
#[target_feature(enable = "avx2")]
fn less_equal(a: __m128i, b: __m128i) -> __m128i {
    mask_not(_mm_cmpgt_epi32(a, b))
}

#[inline]
#[target_feature(enable = "avx2")]
fn select(mask: __m128i, yes: __m128i, no: __m128i) -> __m128i {
    _mm_blendv_epi8(no, yes, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip(value: __m128i, lo: __m128i, hi: __m128i) -> __m128i {
    _mm_max_epi32(lo, _mm_min_epi32(value, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn weighted(samples: &[__m128i], weights: &[i32], bias: i32, shift: i32) -> __m128i {
    let mut sum = _mm_set1_epi32(bias);
    for (&sample, &weight) in samples.iter().zip(weights) {
        if weight != 0 {
            sum = _mm_add_epi32(sum, _mm_mullo_epi32(sample, _mm_set1_epi32(weight)));
        }
    }
    _mm_sra_epi32(sum, _mm_cvtsi32_si128(shift))
}

/// Four-lane dav1d-style loop filter. Each lane is one line crossing the edge;
/// filter/HEV/flat decisions are masks and all selected widths are branchless
/// within the four-line segment.
#[target_feature(enable = "avx2")]
pub(crate) fn loop_filter_avx2(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    let s = if stride_a == 1 {
        let mut samples = [_mm_setzero_si128(); 14];
        for offset in -2isize..=1 {
            samples[(offset + 7) as usize] = load_sample(dst, base, stride_a, stride_b, offset);
        }
        if wd > 4 {
            samples[4] = load_sample(dst, base, stride_a, stride_b, -3);
            samples[9] = load_sample(dst, base, stride_a, stride_b, 2);
        }
        if wd > 6 {
            samples[3] = load_sample(dst, base, stride_a, stride_b, -4);
            samples[10] = load_sample(dst, base, stride_a, stride_b, 3);
        }
        if wd >= 16 {
            for offset in -7isize..=-5 {
                samples[(offset + 7) as usize] = load_sample(dst, base, stride_a, stride_b, offset);
            }
            for offset in 4isize..=6 {
                samples[(offset + 7) as usize] = load_sample(dst, base, stride_a, stride_b, offset);
            }
        }
        samples
    } else {
        load_vertical_edge(dst, base, stride_a, wd)
    };

    let scale = 1i32 << (bd as i32 - 8);
    let e = _mm_set1_epi32(e * scale);
    let i_lim = _mm_set1_epi32(i_lim * scale);
    let h_thresh = _mm_set1_epi32(h_thresh * scale);
    let flat_limit = _mm_set1_epi32(scale);
    let zero = _mm_setzero_si128();
    let clip_lo = _mm_set1_epi32(-128 * scale);
    let clip_hi = _mm_set1_epi32(128 * scale - 1);
    let pixel_hi = _mm_set1_epi32((1 << bd) - 1);

    let p1 = s[5];
    let p0 = s[6];
    let q0 = s[7];
    let q1 = s[8];

    let mut fm = _mm_and_si128(
        less_equal(abs_diff(p1, p0), i_lim),
        less_equal(abs_diff(q1, q0), i_lim),
    );
    let edge_metric = _mm_add_epi32(
        _mm_slli_epi32::<1>(abs_diff(p0, q0)),
        _mm_srai_epi32::<1>(abs_diff(p1, q1)),
    );
    fm = _mm_and_si128(fm, less_equal(edge_metric, e));
    if wd > 4 {
        fm = _mm_and_si128(fm, less_equal(abs_diff(s[4], p1), i_lim));
        fm = _mm_and_si128(fm, less_equal(abs_diff(s[9], q1), i_lim));
    }
    if wd > 6 {
        fm = _mm_and_si128(fm, less_equal(abs_diff(s[3], s[4]), i_lim));
        fm = _mm_and_si128(fm, less_equal(abs_diff(s[10], s[9]), i_lim));
    }
    if !mask_any(fm) {
        return;
    }

    let mut flat_in = _mm_setzero_si128();
    if wd >= 6 {
        flat_in = less_equal(abs_diff(s[4], p0), flat_limit);
        flat_in = _mm_and_si128(flat_in, less_equal(abs_diff(p1, p0), flat_limit));
        flat_in = _mm_and_si128(flat_in, less_equal(abs_diff(q1, q0), flat_limit));
        flat_in = _mm_and_si128(flat_in, less_equal(abs_diff(s[9], q0), flat_limit));
    }
    if wd >= 8 {
        flat_in = _mm_and_si128(flat_in, less_equal(abs_diff(s[3], p0), flat_limit));
        flat_in = _mm_and_si128(flat_in, less_equal(abs_diff(s[10], q0), flat_limit));
    }

    let mut flat_out = _mm_setzero_si128();
    if wd >= 16 {
        flat_out = less_equal(abs_diff(s[0], p0), flat_limit);
        flat_out = _mm_and_si128(flat_out, less_equal(abs_diff(s[1], p0), flat_limit));
        flat_out = _mm_and_si128(flat_out, less_equal(abs_diff(s[2], p0), flat_limit));
        flat_out = _mm_and_si128(flat_out, less_equal(abs_diff(s[11], q0), flat_limit));
        flat_out = _mm_and_si128(flat_out, less_equal(abs_diff(s[12], q0), flat_limit));
        flat_out = _mm_and_si128(flat_out, less_equal(abs_diff(s[13], q0), flat_limit));
    }

    let wide16 = if wd >= 16 {
        _mm_and_si128(fm, _mm_and_si128(flat_in, flat_out))
    } else {
        _mm_setzero_si128()
    };
    let wide8 = if wd >= 8 {
        _mm_and_si128(_mm_and_si128(fm, flat_in), mask_not(wide16))
    } else {
        _mm_setzero_si128()
    };
    let wide6 = if wd == 6 {
        _mm_and_si128(fm, flat_in)
    } else {
        _mm_setzero_si128()
    };
    let wide = _mm_or_si128(_mm_or_si128(wide16, wide8), wide6);
    let short = _mm_and_si128(fm, mask_not(wide));

    let hev = _mm_or_si128(
        _mm_cmpgt_epi32(abs_diff(p1, p0), h_thresh),
        _mm_cmpgt_epi32(abs_diff(q1, q0), h_thresh),
    );
    let delta = _mm_sub_epi32(q0, p0);
    let triple = _mm_add_epi32(delta, _mm_slli_epi32::<1>(delta));
    let fv_plain = clip(triple, clip_lo, clip_hi);
    let fv_hev = clip(
        _mm_add_epi32(triple, clip(_mm_sub_epi32(p1, q1), clip_lo, clip_hi)),
        clip_lo,
        clip_hi,
    );
    let fv = select(hev, fv_hev, fv_plain);
    let f1 = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(fv, _mm_set1_epi32(4)), clip_hi));
    let f2 = _mm_srai_epi32::<3>(_mm_min_epi32(_mm_add_epi32(fv, _mm_set1_epi32(3)), clip_hi));

    let mut out = s;
    out[6] = select(short, clip(_mm_add_epi32(p0, f2), zero, pixel_hi), out[6]);
    out[7] = select(short, clip(_mm_sub_epi32(q0, f1), zero, pixel_hi), out[7]);
    let outer_short = _mm_and_si128(short, mask_not(hev));
    let f = _mm_srai_epi32::<1>(_mm_add_epi32(f1, _mm_set1_epi32(1)));
    out[5] = select(
        outer_short,
        clip(_mm_add_epi32(p1, f), zero, pixel_hi),
        out[5],
    );
    out[8] = select(
        outer_short,
        clip(_mm_sub_epi32(q1, f), zero, pixel_hi),
        out[8],
    );

    let has_wide6 = mask_any(wide6);
    let has_wide8 = mask_any(wide8);
    let has_wide16 = mask_any(wide16);
    if has_wide6 {
        for (j, weights) in WIDE6_WEIGHTS.iter().enumerate() {
            out[j + 5] = select(wide6, weighted(&s[4..10], weights, 4, 3), out[j + 5]);
        }
    }
    if has_wide8 {
        for (j, weights) in WIDE8_WEIGHTS.iter().enumerate() {
            out[j + 4] = select(wide8, weighted(&s[3..11], weights, 4, 3), out[j + 4]);
        }
    }
    if has_wide16 {
        for (j, weights) in WIDE16_WEIGHTS.iter().enumerate() {
            out[j + 1] = select(wide16, weighted(&s, weights, 8, 4), out[j + 1]);
        }
    }

    let (first, last) = if has_wide16 {
        (1usize, 12usize)
    } else if has_wide8 {
        (4, 9)
    } else {
        (5, 8)
    };
    if stride_a == 1 {
        #[allow(clippy::needless_range_loop)]
        for index in first..=last {
            store_sample(
                dst,
                base,
                stride_a,
                stride_b,
                index as isize - 7,
                out[index],
            );
        }
    } else if has_wide16 {
        store_vertical_4_lines(dst, base, stride_a, -6, &out[1..5]);
        store_vertical_4_lines(dst, base, stride_a, -2, &out[5..9]);
        store_vertical_4_lines(dst, base, stride_a, 2, &out[9..13]);
    } else if has_wide8 {
        store_vertical_4_lines(dst, base, stride_a, -3, &out[4..8]);
        store_vertical_2_lines(dst, base, stride_a, 1, &out[8..10]);
    } else {
        store_vertical_4_lines(dst, base, stride_a, -2, &out[5..9]);
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_sample_batch(
    dst: &[u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
) -> __m256i {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        unsafe { _mm256_loadu_si256(dst.as_ptr().add(pos).cast()) }
    } else {
        let pos = base as isize + offset * stride_b;
        let src = dst.as_ptr();
        unsafe {
            _mm256_setr_epi16(
                *src.offset(pos) as i16,
                *src.offset(pos + stride_a) as i16,
                *src.offset(pos + 2 * stride_a) as i16,
                *src.offset(pos + 3 * stride_a) as i16,
                *src.offset(pos + 4 * stride_a) as i16,
                *src.offset(pos + 5 * stride_a) as i16,
                *src.offset(pos + 6 * stride_a) as i16,
                *src.offset(pos + 7 * stride_a) as i16,
                *src.offset(pos + 8 * stride_a) as i16,
                *src.offset(pos + 9 * stride_a) as i16,
                *src.offset(pos + 10 * stride_a) as i16,
                *src.offset(pos + 11 * stride_a) as i16,
                *src.offset(pos + 12 * stride_a) as i16,
                *src.offset(pos + 13 * stride_a) as i16,
                *src.offset(pos + 14 * stride_a) as i16,
                *src.offset(pos + 15 * stride_a) as i16,
            )
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_8x8_u16(values: &mut [__m128i; 8]) {
    let a0 = _mm_unpacklo_epi16(values[0], values[1]);
    let a1 = _mm_unpackhi_epi16(values[0], values[1]);
    let a2 = _mm_unpacklo_epi16(values[2], values[3]);
    let a3 = _mm_unpackhi_epi16(values[2], values[3]);
    let a4 = _mm_unpacklo_epi16(values[4], values[5]);
    let a5 = _mm_unpackhi_epi16(values[4], values[5]);
    let a6 = _mm_unpacklo_epi16(values[6], values[7]);
    let a7 = _mm_unpackhi_epi16(values[6], values[7]);

    let b0 = _mm_unpacklo_epi32(a0, a2);
    let b1 = _mm_unpackhi_epi32(a0, a2);
    let b2 = _mm_unpacklo_epi32(a1, a3);
    let b3 = _mm_unpackhi_epi32(a1, a3);
    let b4 = _mm_unpacklo_epi32(a4, a6);
    let b5 = _mm_unpackhi_epi32(a4, a6);
    let b6 = _mm_unpacklo_epi32(a5, a7);
    let b7 = _mm_unpackhi_epi32(a5, a7);

    values[0] = _mm_unpacklo_epi64(b0, b4);
    values[1] = _mm_unpackhi_epi64(b0, b4);
    values[2] = _mm_unpacklo_epi64(b1, b5);
    values[3] = _mm_unpackhi_epi64(b1, b5);
    values[4] = _mm_unpacklo_epi64(b2, b6);
    values[5] = _mm_unpackhi_epi64(b2, b6);
    values[6] = _mm_unpacklo_epi64(b3, b7);
    values[7] = _mm_unpackhi_epi64(b3, b7);
}

#[inline]
#[target_feature(enable = "avx2")]
fn combine_128(lo: __m128i, hi: __m128i) -> __m256i {
    _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_vertical<const N: usize>(
    dst: &[u16],
    base: usize,
    stride: isize,
    offset: isize,
) -> [__m256i; 8] {
    let src = unsafe { dst.as_ptr().add((base as isize + offset) as usize) };
    let mut lo = [_mm_setzero_si128(); 8];
    let mut hi = [_mm_setzero_si128(); 8];
    for row in 0..8 {
        let lo_src = unsafe { src.offset(row as isize * stride) };
        let hi_src = unsafe { src.offset((row + 8) as isize * stride) };
        if N == 4 {
            lo[row] = unsafe { _mm_loadl_epi64(lo_src.cast()) };
            hi[row] = unsafe { _mm_loadl_epi64(hi_src.cast()) };
        } else {
            lo[row] = unsafe { _mm_loadu_si128(lo_src.cast()) };
            hi[row] = unsafe { _mm_loadu_si128(hi_src.cast()) };
        }
    }
    transpose_8x8_u16(&mut lo);
    transpose_8x8_u16(&mut hi);
    std::array::from_fn(|i| combine_128(lo[i], hi[i]))
}

#[inline]
#[target_feature(enable = "avx2")]
fn load_vertical_batch(dst: &[u16], base: usize, stride: isize, wd: i32) -> [__m256i; 14] {
    let mut samples = [_mm256_setzero_si256(); 14];
    match wd {
        4 => samples[5..9].copy_from_slice(&load_vertical::<4>(dst, base, stride, -2)[..4]),
        6 => {
            samples[4..8].copy_from_slice(&load_vertical::<4>(dst, base, stride, -3)[..4]);
            samples[6..10].copy_from_slice(&load_vertical::<4>(dst, base, stride, -1)[..4]);
        }
        8 => samples[3..11].copy_from_slice(&load_vertical::<8>(dst, base, stride, -4)),
        16 => {
            samples[0..8].copy_from_slice(&load_vertical::<8>(dst, base, stride, -7));
            samples[6..14].copy_from_slice(&load_vertical::<8>(dst, base, stride, -1));
        }
        _ => unreachable!("unsupported loop-filter width"),
    }
    samples
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_vertical_2(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[__m256i],
) {
    let a = [
        _mm256_castsi256_si128(values[0]),
        _mm256_extracti128_si256::<1>(values[0]),
    ];
    let b = [
        _mm256_castsi256_si128(values[1]),
        _mm256_extracti128_si256::<1>(values[1]),
    ];
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for half in 0..2 {
        let pairs = [
            _mm_unpacklo_epi16(a[half], b[half]),
            _mm_unpackhi_epi16(a[half], b[half]),
        ];
        for (quarter, pair) in pairs.iter().enumerate() {
            for lane in 0..4 {
                let value = match lane {
                    0 => _mm_extract_epi32::<0>(*pair),
                    1 => _mm_extract_epi32::<1>(*pair),
                    2 => _mm_extract_epi32::<2>(*pair),
                    _ => _mm_extract_epi32::<3>(*pair),
                };
                let row = half * 8 + quarter * 4 + lane;
                unsafe {
                    dst.offset(row as isize * stride)
                        .cast::<i32>()
                        .write_unaligned(value);
                }
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_vertical_transposed<const N: usize>(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[__m256i],
) {
    let mut lo = [_mm_setzero_si128(); 8];
    let mut hi = [_mm_setzero_si128(); 8];
    for i in 0..N {
        lo[i] = _mm256_castsi256_si128(values[i]);
        hi[i] = _mm256_extracti128_si256::<1>(values[i]);
    }
    transpose_8x8_u16(&mut lo);
    transpose_8x8_u16(&mut hi);

    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for row in 0..8 {
        let lo_dst = unsafe { dst.offset(row as isize * stride) };
        let hi_dst = unsafe { dst.offset((row + 8) as isize * stride) };
        unsafe {
            if N == 4 {
                _mm_storel_epi64(lo_dst.cast(), lo[row]);
                _mm_storel_epi64(hi_dst.cast(), hi[row]);
            } else {
                _mm_storeu_si128(lo_dst.cast(), lo[row]);
                _mm_storeu_si128(hi_dst.cast(), hi[row]);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_sample_batch(
    dst: &mut [u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
    value: __m256i,
) {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        unsafe { _mm256_storeu_si256(dst.as_mut_ptr().add(pos).cast(), value) };
    } else {
        let pos = base as isize + offset * stride_b;
        let dst = dst.as_mut_ptr();
        unsafe {
            *dst.offset(pos) = _mm256_extract_epi16::<0>(value) as u16;
            *dst.offset(pos + stride_a) = _mm256_extract_epi16::<1>(value) as u16;
            *dst.offset(pos + 2 * stride_a) = _mm256_extract_epi16::<2>(value) as u16;
            *dst.offset(pos + 3 * stride_a) = _mm256_extract_epi16::<3>(value) as u16;
            *dst.offset(pos + 4 * stride_a) = _mm256_extract_epi16::<4>(value) as u16;
            *dst.offset(pos + 5 * stride_a) = _mm256_extract_epi16::<5>(value) as u16;
            *dst.offset(pos + 6 * stride_a) = _mm256_extract_epi16::<6>(value) as u16;
            *dst.offset(pos + 7 * stride_a) = _mm256_extract_epi16::<7>(value) as u16;
            *dst.offset(pos + 8 * stride_a) = _mm256_extract_epi16::<8>(value) as u16;
            *dst.offset(pos + 9 * stride_a) = _mm256_extract_epi16::<9>(value) as u16;
            *dst.offset(pos + 10 * stride_a) = _mm256_extract_epi16::<10>(value) as u16;
            *dst.offset(pos + 11 * stride_a) = _mm256_extract_epi16::<11>(value) as u16;
            *dst.offset(pos + 12 * stride_a) = _mm256_extract_epi16::<12>(value) as u16;
            *dst.offset(pos + 13 * stride_a) = _mm256_extract_epi16::<13>(value) as u16;
            *dst.offset(pos + 14 * stride_a) = _mm256_extract_epi16::<14>(value) as u16;
            *dst.offset(pos + 15 * stride_a) = _mm256_extract_epi16::<15>(value) as u16;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn less_equal_u16(a: __m256i, b: __m256i) -> __m256i {
    _mm256_xor_si256(_mm256_cmpgt_epi16(a, b), _mm256_set1_epi16(-1))
}

#[inline]
#[target_feature(enable = "avx2")]
fn select_u16(mask: __m256i, yes: __m256i, no: __m256i) -> __m256i {
    _mm256_blendv_epi8(no, yes, mask)
}

#[inline]
#[target_feature(enable = "avx2")]
fn clip_s16(value: __m256i, lo: __m256i, hi: __m256i) -> __m256i {
    _mm256_max_epi16(lo, _mm256_min_epi16(value, hi))
}

#[inline]
#[target_feature(enable = "avx2")]
fn mask_any_u16(value: __m256i) -> bool {
    _mm256_testz_si256(value, value) == 0
}

#[inline]
#[target_feature(enable = "avx2")]
fn weighted_u16(samples: &[__m256i], weights: &[i32], bias: i16, shift: i32) -> __m256i {
    let mut sum = _mm256_set1_epi16(bias);
    for (&sample, &weight) in samples.iter().zip(weights) {
        if weight != 0 {
            sum = _mm256_add_epi16(
                sum,
                _mm256_mullo_epi16(sample, _mm256_set1_epi16(weight as i16)),
            );
        }
    }
    match shift {
        3 => _mm256_srli_epi16::<3>(sum),
        4 => _mm256_srli_epi16::<4>(sum),
        _ => unreachable!("unsupported loop-filter shift"),
    }
}

/// Sixteen-line loop filter using 16-bit lanes. AV1's 12-bit extrema fit:
/// edge metric <= 10237, signed filter accumulator <= 14333, and the widest
/// unsigned convolution including bias <= 65528.
#[target_feature(enable = "avx2")]
pub(crate) fn loop_filter_batch_avx2(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    let s = if stride_a == 1 {
        let mut samples = [_mm256_setzero_si256(); 14];
        for offset in -2isize..=1 {
            samples[(offset + 7) as usize] =
                load_sample_batch(dst, base, stride_a, stride_b, offset);
        }
        if wd > 4 {
            samples[4] = load_sample_batch(dst, base, stride_a, stride_b, -3);
            samples[9] = load_sample_batch(dst, base, stride_a, stride_b, 2);
        }
        if wd > 6 {
            samples[3] = load_sample_batch(dst, base, stride_a, stride_b, -4);
            samples[10] = load_sample_batch(dst, base, stride_a, stride_b, 3);
        }
        if wd >= 16 {
            for offset in -7isize..=-5 {
                samples[(offset + 7) as usize] =
                    load_sample_batch(dst, base, stride_a, stride_b, offset);
            }
            for offset in 4isize..=6 {
                samples[(offset + 7) as usize] =
                    load_sample_batch(dst, base, stride_a, stride_b, offset);
            }
        }
        samples
    } else {
        load_vertical_batch(dst, base, stride_a, wd)
    };

    let scale = 1i16 << (bd - 8);
    let e = _mm256_set1_epi16(e as i16 * scale);
    let i_lim = _mm256_set1_epi16(i_lim as i16 * scale);
    let h_thresh = _mm256_set1_epi16(h_thresh as i16 * scale);
    let flat_limit = _mm256_set1_epi16(scale);
    let p1 = s[5];
    let p0 = s[6];
    let q0 = s[7];
    let q1 = s[8];

    let abs_diff = |a, b| _mm256_abs_epi16(_mm256_sub_epi16(a, b));
    let mut fm = _mm256_and_si256(
        less_equal_u16(abs_diff(p1, p0), i_lim),
        less_equal_u16(abs_diff(q1, q0), i_lim),
    );
    let edge_metric = _mm256_add_epi16(
        _mm256_slli_epi16::<1>(abs_diff(p0, q0)),
        _mm256_srli_epi16::<1>(abs_diff(p1, q1)),
    );
    fm = _mm256_and_si256(fm, less_equal_u16(edge_metric, e));
    if wd > 4 {
        fm = _mm256_and_si256(fm, less_equal_u16(abs_diff(s[4], p1), i_lim));
        fm = _mm256_and_si256(fm, less_equal_u16(abs_diff(s[9], q1), i_lim));
    }
    if wd > 6 {
        fm = _mm256_and_si256(fm, less_equal_u16(abs_diff(s[3], s[4]), i_lim));
        fm = _mm256_and_si256(fm, less_equal_u16(abs_diff(s[10], s[9]), i_lim));
    }
    if !mask_any_u16(fm) {
        return;
    }

    let mut flat_in = _mm256_setzero_si256();
    if wd >= 6 {
        flat_in = less_equal_u16(abs_diff(s[4], p0), flat_limit);
        flat_in = _mm256_and_si256(flat_in, less_equal_u16(abs_diff(p1, p0), flat_limit));
        flat_in = _mm256_and_si256(flat_in, less_equal_u16(abs_diff(q1, q0), flat_limit));
        flat_in = _mm256_and_si256(flat_in, less_equal_u16(abs_diff(s[9], q0), flat_limit));
    }
    if wd >= 8 {
        flat_in = _mm256_and_si256(flat_in, less_equal_u16(abs_diff(s[3], p0), flat_limit));
        flat_in = _mm256_and_si256(flat_in, less_equal_u16(abs_diff(s[10], q0), flat_limit));
    }

    let mut flat_out = _mm256_setzero_si256();
    if wd >= 16 {
        flat_out = less_equal_u16(abs_diff(s[0], p0), flat_limit);
        for &sample in &s[1..3] {
            flat_out = _mm256_and_si256(flat_out, less_equal_u16(abs_diff(sample, p0), flat_limit));
        }
        for &sample in &s[11..14] {
            flat_out = _mm256_and_si256(flat_out, less_equal_u16(abs_diff(sample, q0), flat_limit));
        }
    }

    let zero = _mm256_setzero_si256();
    let all = _mm256_set1_epi16(-1);
    let wide16 = if wd >= 16 {
        _mm256_and_si256(fm, _mm256_and_si256(flat_in, flat_out))
    } else {
        zero
    };
    let wide8 = if wd >= 8 {
        _mm256_and_si256(_mm256_and_si256(fm, flat_in), _mm256_xor_si256(wide16, all))
    } else {
        zero
    };
    let wide6 = if wd == 6 {
        _mm256_and_si256(fm, flat_in)
    } else {
        zero
    };
    let wide = _mm256_or_si256(_mm256_or_si256(wide16, wide8), wide6);
    let short = _mm256_and_si256(fm, _mm256_xor_si256(wide, all));

    let mut out = s;
    if mask_any_u16(short) {
        let hev = _mm256_or_si256(
            _mm256_cmpgt_epi16(abs_diff(p1, p0), h_thresh),
            _mm256_cmpgt_epi16(abs_diff(q1, q0), h_thresh),
        );
        let clip_lo = _mm256_set1_epi16(-128 * scale);
        let clip_hi = _mm256_set1_epi16(128 * scale - 1);
        let pixel_hi = _mm256_set1_epi16((1i16 << bd) - 1);
        let delta = _mm256_sub_epi16(q0, p0);
        let triple = _mm256_add_epi16(delta, _mm256_slli_epi16::<1>(delta));
        let fv_plain = clip_s16(triple, clip_lo, clip_hi);
        let fv_hev = clip_s16(
            _mm256_add_epi16(triple, clip_s16(_mm256_sub_epi16(p1, q1), clip_lo, clip_hi)),
            clip_lo,
            clip_hi,
        );
        let fv = select_u16(hev, fv_hev, fv_plain);
        let f1 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
            _mm256_add_epi16(fv, _mm256_set1_epi16(4)),
            clip_hi,
        ));
        let f2 = _mm256_srai_epi16::<3>(_mm256_min_epi16(
            _mm256_add_epi16(fv, _mm256_set1_epi16(3)),
            clip_hi,
        ));

        out[6] = select_u16(
            short,
            clip_s16(_mm256_add_epi16(p0, f2), zero, pixel_hi),
            out[6],
        );
        out[7] = select_u16(
            short,
            clip_s16(_mm256_sub_epi16(q0, f1), zero, pixel_hi),
            out[7],
        );
        let outer_short = _mm256_and_si256(short, _mm256_xor_si256(hev, all));
        let f = _mm256_srai_epi16::<1>(_mm256_add_epi16(f1, _mm256_set1_epi16(1)));
        out[5] = select_u16(
            outer_short,
            clip_s16(_mm256_add_epi16(p1, f), zero, pixel_hi),
            out[5],
        );
        out[8] = select_u16(
            outer_short,
            clip_s16(_mm256_sub_epi16(q1, f), zero, pixel_hi),
            out[8],
        );
    }

    let has_wide6 = mask_any_u16(wide6);
    let has_wide8 = mask_any_u16(wide8);
    let has_wide16 = mask_any_u16(wide16);
    if has_wide6 {
        for (j, weights) in WIDE6_WEIGHTS.iter().enumerate() {
            out[j + 5] = select_u16(wide6, weighted_u16(&s[4..10], weights, 4, 3), out[j + 5]);
        }
    }
    if has_wide8 {
        for (j, weights) in WIDE8_WEIGHTS.iter().enumerate() {
            out[j + 4] = select_u16(wide8, weighted_u16(&s[3..11], weights, 4, 3), out[j + 4]);
        }
    }
    if has_wide16 {
        for (j, weights) in WIDE16_WEIGHTS.iter().enumerate() {
            out[j + 1] = select_u16(wide16, weighted_u16(&s, weights, 8, 4), out[j + 1]);
        }
    }

    let (first, last) = if has_wide16 {
        (1usize, 12usize)
    } else if has_wide8 {
        (4, 9)
    } else {
        (5, 8)
    };
    if stride_a == 1 {
        for (index, &value) in (first..=last).zip(out[first..=last].iter()) {
            store_sample_batch(dst, base, stride_a, stride_b, index as isize - 7, value);
        }
    } else if has_wide16 {
        store_vertical_transposed::<8>(dst, base, stride_a, -6, &out[1..9]);
        store_vertical_transposed::<4>(dst, base, stride_a, 2, &out[9..13]);
    } else if has_wide8 {
        store_vertical_transposed::<4>(dst, base, stride_a, -3, &out[4..8]);
        store_vertical_2(dst, base, stride_a, 1, &out[8..10]);
    } else {
        store_vertical_transposed::<4>(dst, base, stride_a, -2, &out[5..9]);
    }
}
