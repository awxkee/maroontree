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

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "avx2,fma")]
fn avx2_log2p1_f32(x: __m256) -> __m256 {
    let y = _mm256_add_ps(x, _mm256_set1_ps(1.0));

    // Decompose y = m * 2^e, m in [1, 2).
    let bits = _mm256_castps_si256(y);
    let exp_u = _mm256_srli_epi32::<23>(bits);
    let exp_i = _mm256_sub_epi32(exp_u, _mm256_set1_epi32(127));
    let e = _mm256_cvtepi32_ps(exp_i);

    let mant_bits = _mm256_or_si256(
        _mm256_and_si256(bits, _mm256_set1_epi32(0x007f_ffff)),
        _mm256_set1_epi32(0x3f80_0000u32 as i32),
    );
    let m = _mm256_castsi256_ps(mant_bits);
    let t = _mm256_sub_ps(m, _mm256_set1_ps(1.0));

    let c0 = _mm256_set1_ps(1.4426934719085693359375);
    let c1 = _mm256_set1_ps(-0.721179187297821044921875);
    let c2 = _mm256_set1_ps(0.477900087833404541015625);
    let c3 = _mm256_set1_ps(-0.340080082416534423828125);
    let c4 = _mm256_set1_ps(0.21719777584075927734375);
    let c5 = _mm256_set1_ps(-9.749893844127655029296875e-2);
    let c6 = _mm256_set1_ps(2.096841670572757720947265625e-2);

    let mut p = c6;
    p = _mm256_fmadd_ps(t, p, c5);
    p = _mm256_fmadd_ps(t, p, c4);
    p = _mm256_fmadd_ps(t, p, c3);
    p = _mm256_fmadd_ps(t, p, c2);
    p = _mm256_fmadd_ps(t, p, c1);
    p = _mm256_fmadd_ps(t, p, c0);

    _mm256_add_ps(e, _mm256_mul_ps(t, p))
}

#[inline]
#[target_feature(enable = "avx2")]
fn hsum_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps::<0x55>(sum64, sum64));
    _mm_cvtss_f32(sum32)
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn coeff_rate_f32_avx2(lev: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let zero = _mm256_setzero_ps();
    let two = _mm256_set1_ps(2.0);

    let chunks = lev.as_chunks::<8>();
    let rem = chunks.1;

    for chunk in chunks.0.iter() {
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        let a = _mm256_and_ps(v, sign_mask);
        let nz = _mm256_cmp_ps::<_CMP_GT_OQ>(a, zero);
        let cost = _mm256_add_ps(two, _mm256_mul_ps(two, avx2_log2p1_f32(a)));
        sum = _mm256_add_ps(sum, _mm256_and_ps(cost, nz));
    }

    let mut out = hsum_ps(sum);
    for &v in rem {
        if v != 0.0 {
            out += 2.0 + 2.0 * crate::av2::helpers::log2p1_approx_f32(v.abs());
        }
    }
    out
}

#[target_feature(enable = "avx2")]
pub(crate) fn coeff_abs_rate_f32_avx2(lev: &[f32]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));

    let chunks = lev.as_chunks::<8>();
    let rem = chunks.1;

    for chunk in chunks.0.iter() {
        let v = unsafe { _mm256_loadu_ps(chunk.as_ptr()) };
        sum = _mm256_add_ps(sum, _mm256_and_ps(v, sign_mask));
    }

    let mut out = hsum_ps(sum);
    for &v in rem {
        out += v.abs();
    }
    out
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
fn hsum_i64x4(v: __m256i) -> i64 {
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

#[target_feature(enable = "avx2")]
pub(crate) fn sad_u8_avx2(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_si256();
    let (a32, a_tail) = a.as_chunks::<32>();
    let (b32, b_tail) = b.as_chunks::<32>();
    for (aa, bb) in a32.iter().zip(b32) {
        let av = unsafe { _mm256_loadu_si256(aa.as_ptr().cast()) };
        let bv = unsafe { _mm256_loadu_si256(bb.as_ptr().cast()) };
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(av, bv));
    }
    let mut out = hsum_i64x4(acc).clamp(0, u32::MAX as i64) as u32;
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        out = out.saturating_add((x as i32 - y as i32).unsigned_abs());
    }
    out
}

/// Strided SAD for integer-valued f32 video planes.
#[target_feature(enable = "avx2")]
pub(crate) fn sad_f32_avx2(
    src: &[f32],
    src_stride: usize,
    pred: &[f32],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    let sign_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let mut acc = _mm256_setzero_ps();
    let mut scalar = 0.0f32;
    for row in 0..height {
        let src = &src[row * src_stride..];
        let pred = &pred[row * pred_stride..];
        let mut x = 0;
        while x + 8 <= width {
            let delta = _mm256_sub_ps(unsafe { _mm256_loadu_ps(src[x..].as_ptr()) }, unsafe {
                _mm256_loadu_ps(pred[x..].as_ptr())
            });
            acc = _mm256_add_ps(acc, _mm256_and_ps(delta, sign_mask));
            x += 8;
        }
        while x < width {
            scalar += (src[x] - pred[x]).abs();
            x += 1;
        }
    }
    (hsum_ps(acc) + scalar) as u64
}

#[target_feature(enable = "avx2")]
pub(crate) fn scaled_residual_f32_avx2(
    dst: &mut [f32],
    src: &[f32],
    pred: &[f32],
    spec: crate::av2::metrics::ResidualSpec,
) {
    let crate::av2::metrics::ResidualSpec {
        src_stride,
        pred_stride,
        width,
        height,
        scale,
    } = spec;
    let scale_vector = _mm256_set1_ps(scale);
    for y in 0..height {
        let mut x = 0;
        while x + 8 <= width {
            let s = unsafe { _mm256_loadu_ps(src[y * src_stride + x..].as_ptr()) };
            let p = unsafe { _mm256_loadu_ps(pred[y * pred_stride + x..].as_ptr()) };
            unsafe {
                _mm256_storeu_ps(
                    dst[y * width + x..].as_mut_ptr(),
                    _mm256_mul_ps(_mm256_sub_ps(s, p), scale_vector),
                )
            };
            x += 8;
        }
        while x < width {
            dst[y * width + x] = (src[y * src_stride + x] - pred[y * pred_stride + x]) * scale;
            x += 1;
        }
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn had4_i32_avx2(v: __m128i) -> __m128i {
    let reverse = _mm_shuffle_epi32::<0x1b>(v);
    let sum = _mm_add_epi32(v, reverse);
    let difference = _mm_sub_epi32(v, reverse);
    let sum_lane1 = _mm_shuffle_epi32::<0x55>(sum);
    let difference_lane1 = _mm_shuffle_epi32::<0x55>(difference);
    let out0 = _mm_add_epi32(sum, sum_lane1);
    let out1 = _mm_add_epi32(difference, difference_lane1);
    let out2 = _mm_sub_epi32(sum, sum_lane1);
    let out3 = _mm_sub_epi32(difference, difference_lane1);
    let lo01 = _mm_unpacklo_epi32(out0, out1);
    let lo23 = _mm_unpacklo_epi32(out2, out3);
    _mm_unpacklo_epi64(lo01, lo23)
}

/// Strided 4x4-Hadamard SATD for integer-valued f32 video planes.
#[target_feature(enable = "avx2")]
pub(crate) fn satd_f32_avx2(
    src: &[f32],
    src_stride: usize,
    pred: &[f32],
    pred_stride: usize,
    width: usize,
    height: usize,
) -> u64 {
    debug_assert_eq!(width & 3, 0);
    debug_assert_eq!(height & 3, 0);
    let mut acc = _mm_setzero_si128();
    let mut y = 0;
    while y < height {
        let mut x = 0;
        while x < width {
            let rows: [__m128i; 4] = std::array::from_fn(|row| {
                let src = unsafe { _mm_loadu_ps(src[(y + row) * src_stride + x..].as_ptr()) };
                let pred = unsafe { _mm_loadu_ps(pred[(y + row) * pred_stride + x..].as_ptr()) };
                had4_i32_avx2(_mm_cvtps_epi32(_mm_sub_ps(src, pred)))
            });
            let e = _mm_add_epi32(rows[0], rows[2]);
            let f = _mm_sub_epi32(rows[0], rows[2]);
            let g = _mm_add_epi32(rows[1], rows[3]);
            let h = _mm_sub_epi32(rows[1], rows[3]);
            for transformed in [
                _mm_add_epi32(e, g),
                _mm_add_epi32(f, h),
                _mm_sub_epi32(f, h),
                _mm_sub_epi32(e, g),
            ] {
                acc = _mm_add_epi32(acc, _mm_abs_epi32(transformed));
            }
            x += 4;
        }
        y += 4;
    }
    let mut lanes = [0i32; 4];
    unsafe { _mm_storeu_si128(lanes.as_mut_ptr().cast(), acc) };
    lanes.into_iter().map(|value| value as u64).sum()
}

#[target_feature(enable = "avx2")]
pub(crate) fn pixel_sse_f32_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_ps();
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    for (aa, bb) in a8.iter().zip(b8) {
        let d = _mm256_sub_ps(unsafe { _mm256_loadu_ps(aa.as_ptr()) }, unsafe {
            _mm256_loadu_ps(bb.as_ptr())
        });
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    let mut out = hsum_ps(acc);
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        let d = x - y;
        out += d * d;
    }
    out
}

#[target_feature(enable = "avx2")]
pub(crate) fn pixel_sse_f32_u16_avx2(a: &[f32], b: &[u16]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = _mm256_setzero_ps();
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    for (aa, bb) in a8.iter().zip(b8) {
        let av = unsafe { _mm256_loadu_ps(aa.as_ptr()) };
        let bv16 = unsafe { _mm_loadu_si128(bb.as_ptr().cast()) };
        let bv = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(bv16));
        let d = _mm256_sub_ps(av, bv);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    let mut out = hsum_ps(acc);
    for (&x, &y) in a_tail.iter().zip(b_tail) {
        let d = x - y as f32;
        out += d * d;
    }
    out
}

#[target_feature(enable = "avx2")]
pub(crate) fn weighted_pixel_sse_f32_avx2(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), weights.len());
    let mut acc = _mm256_setzero_ps();
    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    let (w8, w_tail) = weights.as_chunks::<8>();
    for ((aa, bb), ww) in a8.iter().zip(b8).zip(w8) {
        let d = _mm256_sub_ps(unsafe { _mm256_loadu_ps(aa.as_ptr()) }, unsafe {
            _mm256_loadu_ps(bb.as_ptr())
        });
        let wv = unsafe { _mm256_loadu_ps(ww.as_ptr()) };
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_mul_ps(d, d), wv));
    }
    let mut out = hsum_ps(acc);
    for ((&x, &y), &w) in a_tail.iter().zip(b_tail).zip(w_tail) {
        let d = x - y;
        out += d * d * w;
    }
    out
}

#[target_feature(enable = "avx2")]
pub(crate) fn sum_sumsq_f32_avx2(v: &[f32]) -> (f32, f32) {
    let mut sum_v = _mm256_setzero_ps();
    let mut sq_v = _mm256_setzero_ps();
    let (v8, tail) = v.as_chunks::<8>();
    for x in v8 {
        let xv = unsafe { _mm256_loadu_ps(x.as_ptr()) };
        sum_v = _mm256_add_ps(sum_v, xv);
        sq_v = _mm256_add_ps(sq_v, _mm256_mul_ps(xv, xv));
    }
    let mut sum = hsum_ps(sum_v);
    let mut sumsq = hsum_ps(sq_v);
    for &x in tail {
        sum += x;
        sumsq += x * x;
    }
    (sum, sumsq)
}

#[target_feature(enable = "avx2")]
pub(crate) fn cfl_sse_i32_avx2(
    src: &[i32],
    ac: &[i32],
    alpha_q3: i32,
    dc_value: i32,
    maxv: i32,
) -> f32 {
    debug_assert_eq!(src.len(), ac.len());
    let alpha = _mm256_set1_epi32(alpha_q3);
    let dc = _mm256_set1_epi32(dc_value);
    let zero = _mm256_setzero_si256();
    let maxv_v = _mm256_set1_epi32(maxv);
    let round = _mm256_set1_epi32(1 << 10);
    let mut acc = _mm256_setzero_si256();
    let (src8, src_tail) = src.as_chunks::<8>();
    let (ac8, ac_tail) = ac.as_chunks::<8>();
    for (ss, aa) in src8.iter().zip(ac8) {
        let s = unsafe { _mm256_loadu_si256(ss.as_ptr().cast()) };
        let a = unsafe { _mm256_loadu_si256(aa.as_ptr().cast()) };
        let p = _mm256_mullo_epi32(a, alpha);
        let sign = _mm256_srai_epi32::<31>(p);
        let abs = _mm256_sub_epi32(_mm256_xor_si256(p, sign), sign);
        let q = _mm256_srli_epi32::<11>(_mm256_add_epi32(abs, round));
        let scaled = _mm256_sub_epi32(_mm256_xor_si256(q, sign), sign);
        let pred = _mm256_max_epi32(zero, _mm256_min_epi32(maxv_v, _mm256_add_epi32(dc, scaled)));
        acc = _mm256_add_epi64(acc, square_i32x8_to_i64(_mm256_sub_epi32(s, pred)));
    }
    let mut out = hsum_i64x4(acc) as f32;
    for (&s, &a) in src_tail.iter().zip(ac_tail) {
        let p = alpha_q3 * a;
        let sign = p >> 31;
        let abs = (p ^ sign) - sign;
        let q = (abs + (1 << 10)) >> 11;
        let scaled = (q ^ sign) - sign;
        let pred = (dc_value + scaled).clamp(0, maxv);
        out += crate::av2::helpers::sq_diff_f32(s, pred);
    }
    out
}

#[target_feature(enable = "avx2")]
pub(crate) fn pixel_sse_rounded_avx2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());

    let half = _mm256_set1_ps(0.5);
    let half4 = _mm_set1_ps(0.5);
    let mut acc = _mm256_setzero_si256();
    let mut scalar = 0.0f32;

    let (a8, a_tail) = a.as_chunks::<8>();
    let (b8, b_tail) = b.as_chunks::<8>();
    for (aa, bb) in a8.iter().zip(b8.iter()) {
        let ax = unsafe { _mm256_loadu_ps(aa.as_ptr()) };
        let bx = unsafe { _mm256_loadu_ps(bb.as_ptr()) };
        let ai = _mm256_cvttps_epi32(_mm256_add_ps(ax, half));
        let bi = _mm256_cvttps_epi32(_mm256_add_ps(bx, half));
        let d = _mm256_sub_epi32(ai, bi);
        acc = _mm256_add_epi64(acc, square_i32x8_to_i64(d));
    }

    let (a4, a_tail) = a_tail.as_chunks::<4>();
    let (b4, b_tail) = b_tail.as_chunks::<4>();
    for (aa, bb) in a4.iter().zip(b4.iter()) {
        let ax = unsafe { _mm_loadu_ps(aa.as_ptr()) };
        let bx = unsafe { _mm_loadu_ps(bb.as_ptr()) };
        let ai = _mm_cvttps_epi32(_mm_add_ps(ax, half4));
        let bi = _mm_cvttps_epi32(_mm_add_ps(bx, half4));
        let d = _mm_sub_epi32(ai, bi);
        acc = _mm256_add_epi64(acc, widen_i64x2_to_i64x4(square_i32x4_to_i64(d)));
    }

    for (&x, &y) in a_tail.iter().zip(b_tail.iter()) {
        scalar += crate::av2::helpers::sq_diff_f32(
            crate::av2::helpers::pixel_to_i32(x),
            crate::av2::helpers::pixel_to_i32(y),
        );
    }

    hsum_i64x4(acc) as f32 + scalar
}

#[target_feature(enable = "avx2")]
pub(crate) fn pixel_sse_rounded_const_avx2(a: &[f32], v: f32) -> f32 {
    let half = _mm256_set1_ps(0.5);
    let half4 = _mm_set1_ps(0.5);
    let vi8 = _mm256_cvttps_epi32(_mm256_add_ps(_mm256_set1_ps(v), half));
    let vi4 = _mm_cvttps_epi32(_mm_add_ps(_mm_set1_ps(v), half4));
    let mut acc = _mm256_setzero_si256();
    let mut scalar = 0.0f32;

    let (a8, a_tail) = a.as_chunks::<8>();
    for aa in a8.iter() {
        let ax = unsafe { _mm256_loadu_ps(aa.as_ptr()) };
        let ai = _mm256_cvttps_epi32(_mm256_add_ps(ax, half));
        let d = _mm256_sub_epi32(ai, vi8);
        acc = _mm256_add_epi64(acc, square_i32x8_to_i64(d));
    }

    let (a4, a_tail) = a_tail.as_chunks::<4>();
    for aa in a4.iter() {
        let ax = unsafe { _mm_loadu_ps(aa.as_ptr()) };
        let ai = _mm_cvttps_epi32(_mm_add_ps(ax, half4));
        let d = _mm_sub_epi32(ai, vi4);
        acc = _mm256_add_epi64(acc, widen_i64x2_to_i64x4(square_i32x4_to_i64(d)));
    }

    let vi = crate::av2::helpers::pixel_to_i32(v);
    for &x in a_tail.iter() {
        scalar += crate::av2::helpers::sq_diff_f32(crate::av2::helpers::pixel_to_i32(x), vi);
    }

    hsum_i64x4(acc) as f32 + scalar
}
