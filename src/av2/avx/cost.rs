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
