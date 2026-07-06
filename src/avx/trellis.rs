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

use core::arch::x86_64::*;

use crate::cost::{COEF_RATE_LUT, coef_rate_bits, rate_cost};

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_scan8(scan: &[u32], i: usize) -> __m256i {
    debug_assert!(i + 8 <= scan.len());
    unsafe { _mm256_loadu_si256(scan.as_ptr().add(i).cast::<__m256i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_scan4(scan: &[u32], i: usize) -> __m128i {
    debug_assert!(i + 4 <= scan.len());
    unsafe { _mm_loadu_si128(scan.as_ptr().add(i).cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn q2_for_scan_chunk8(i: usize, dc_q2: f32, ac_q2: f32) -> __m256 {
    if i == 0 {
        _mm256_set_ps(ac_q2, ac_q2, ac_q2, ac_q2, ac_q2, ac_q2, ac_q2, dc_q2)
    } else {
        _mm256_set1_ps(ac_q2)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn q2_for_scan_chunk4(i: usize, dc_q2: f32, ac_q2: f32) -> __m128 {
    if i == 0 {
        _mm_set_ps(ac_q2, ac_q2, ac_q2, dc_q2)
    } else {
        _mm_set1_ps(ac_q2)
    }
}

#[inline]
#[target_feature(enable = "avx2")]
fn abs_i32x4(v: __m128i) -> __m128i {
    let sign = _mm_srai_epi32::<31>(v);
    _mm_sub_epi32(_mm_xor_si128(v, sign), sign)
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn round_down_cost8(
    tf: &[f32],
    cf: &[i32],
    idx: __m256i,
    i: usize,
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) -> Option<i32> {
    let c = unsafe { _mm256_i32gather_epi32::<4>(cf.as_ptr(), idx) };
    let l_i = _mm256_abs_epi32(c);

    // Keep the SIMD path LUT-only. Higher magnitudes are rare here and need the
    // closed-form Golomb tail from coef_rate_bits()
    let ge64 = _mm256_cmpgt_epi32(l_i, _mm256_set1_epi32(63));
    if _mm256_movemask_epi8(ge64) != 0 {
        return None;
    }

    let at = _mm256_and_ps(
        unsafe { _mm256_i32gather_ps::<4>(tf.as_ptr(), idx) },
        _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff)),
    );
    let l_dn_i = _mm256_max_epi32(
        _mm256_sub_epi32(l_i, _mm256_set1_epi32(1)),
        _mm256_setzero_si256(),
    );
    let l = _mm256_cvtepi32_ps(l_i);
    let l_dn = _mm256_cvtepi32_ps(l_dn_i);
    let rate_l = unsafe { _mm256_i32gather_ps::<4>(COEF_RATE_LUT.as_ptr(), l_i) };
    let rate_dn = unsafe { _mm256_i32gather_ps::<4>(COEF_RATE_LUT.as_ptr(), l_dn_i) };
    let q2 = q2_for_scan_chunk8(i, dc_q2, ac_q2);
    let e_l = _mm256_sub_ps(at, l);
    let e_dn = _mm256_sub_ps(at, l_dn);
    let cost_l = _mm256_fmadd_ps(
        q2,
        _mm256_mul_ps(e_l, e_l),
        _mm256_mul_ps(_mm256_set1_ps(lambda), rate_l),
    );
    let cost_dn = _mm256_fmadd_ps(
        q2,
        _mm256_mul_ps(e_dn, e_dn),
        _mm256_mul_ps(_mm256_set1_ps(lambda), rate_dn),
    );
    let better = _mm256_cmp_ps::<_CMP_LT_OS>(cost_dn, cost_l);
    Some(_mm256_movemask_ps(better))
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn round_down_cost4(
    tf: &[f32],
    cf: &[i32],
    idx: __m128i,
    i: usize,
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) -> Option<i32> {
    let c = unsafe { _mm_i32gather_epi32::<4>(cf.as_ptr(), idx) };
    let l_i = abs_i32x4(c);

    let ge64 = _mm_cmpgt_epi32(l_i, _mm_set1_epi32(63));
    if _mm_movemask_epi8(ge64) != 0 {
        return None;
    }

    let at = _mm_and_ps(
        unsafe { _mm_i32gather_ps::<4>(tf.as_ptr(), idx) },
        _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff)),
    );
    let l_dn_i = _mm_max_epi32(_mm_sub_epi32(l_i, _mm_set1_epi32(1)), _mm_setzero_si128());
    let l = _mm_cvtepi32_ps(l_i);
    let l_dn = _mm_cvtepi32_ps(l_dn_i);
    let rate_l = unsafe { _mm_i32gather_ps::<4>(COEF_RATE_LUT.as_ptr(), l_i) };
    let rate_dn = unsafe { _mm_i32gather_ps::<4>(COEF_RATE_LUT.as_ptr(), l_dn_i) };
    let q2 = q2_for_scan_chunk4(i, dc_q2, ac_q2);
    let e_l = _mm_sub_ps(at, l);
    let e_dn = _mm_sub_ps(at, l_dn);
    let cost_l = _mm_fmadd_ps(
        q2,
        _mm_mul_ps(e_l, e_l),
        _mm_mul_ps(_mm_set1_ps(lambda), rate_l),
    );
    let cost_dn = _mm_fmadd_ps(
        q2,
        _mm_mul_ps(e_dn, e_dn),
        _mm_mul_ps(_mm_set1_ps(lambda), rate_dn),
    );
    let better = _mm_cmplt_ps(cost_dn, cost_l);
    Some(_mm_movemask_ps(better))
}

#[inline]
fn round_down_scalar_one(
    cf: &mut [i32],
    tf: &[f32],
    rc: usize,
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    let c = cf[rc];
    if c == 0 {
        return;
    }
    let l = c.unsigned_abs();
    let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
    let at = tf[rc].abs();
    let e_l = at - l as f32;
    let e_dn = at - (l - 1) as f32;
    let cost_l = dq2 * (e_l * e_l) + rate_cost(lambda, coef_rate_bits(l));
    let cost_dn = dq2 * (e_dn * e_dn) + rate_cost(lambda, coef_rate_bits(l - 1));
    if cost_dn < cost_l {
        cf[rc] = if c < 0 { -(l as i32 - 1) } else { l as i32 - 1 };
    }
}

#[target_feature(enable = "avx2,fma")]
pub(crate) fn trellis_round_down_scan_avx2(
    cf: &mut [i32],
    tf: &[f32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
    lambda: f32,
) {
    let mut i = 0usize;
    let n = scan.len();
    while i + 8 <= n {
        let idx = unsafe { load_scan8(scan, i) };
        match round_down_cost8(tf, cf, idx, i, dc_q2, ac_q2, lambda) {
            Some(mut mask) => {
                while mask != 0 {
                    let lane = mask.trailing_zeros() as usize;
                    let rc = unsafe { *scan.get_unchecked(i + lane) } as usize;
                    let c = unsafe { *cf.get_unchecked(rc) };
                    if c != 0 {
                        let l = c.unsigned_abs();
                        unsafe {
                            *cf.get_unchecked_mut(rc) =
                                if c < 0 { -(l as i32 - 1) } else { l as i32 - 1 };
                        }
                    }
                    mask &= mask - 1;
                }
            }
            None => {
                for &rc32 in scan[i..i + 8].iter() {
                    round_down_scalar_one(cf, tf, rc32 as usize, dc_q2, ac_q2, lambda);
                }
            }
        }
        i += 8;
    }
    while i + 4 <= n {
        let idx = unsafe { load_scan4(scan, i) };
        match round_down_cost4(tf, cf, idx, i, dc_q2, ac_q2, lambda) {
            Some(mut mask) => {
                while mask != 0 {
                    let lane = mask.trailing_zeros() as usize;
                    let rc = unsafe { *scan.get_unchecked(i + lane) } as usize;
                    let c = unsafe { *cf.get_unchecked(rc) };
                    if c != 0 {
                        let l = c.unsigned_abs();
                        unsafe {
                            *cf.get_unchecked_mut(rc) =
                                if c < 0 { -(l as i32 - 1) } else { l as i32 - 1 };
                        }
                    }
                    mask &= mask - 1;
                }
            }
            None => {
                for &rc32 in scan[i..i + 4].iter() {
                    round_down_scalar_one(cf, tf, rc32 as usize, dc_q2, ac_q2, lambda);
                }
            }
        }
        i += 4;
    }
    for &rc32 in unsafe { scan.get_unchecked(i..n) } {
        round_down_scalar_one(cf, tf, rc32 as usize, dc_q2, ac_q2, lambda);
    }
}

#[target_feature(enable = "avx2")]
pub(crate) fn trellis_dist_current_zero_scan_avx2(
    dst_cur: &mut [f32],
    dst_zero: &mut [f32],
    tf: &[f32],
    cf: &[i32],
    scan: &[u32],
    dc_q2: f32,
    ac_q2: f32,
) {
    let mut i = 0usize;
    let n = scan.len();
    let abs_mask8 = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fff_ffff));
    let abs_mask4 = _mm_castsi128_ps(_mm_set1_epi32(0x7fff_ffff));
    while i + 8 <= n {
        let idx = unsafe { load_scan8(scan, i) };
        let t = unsafe { _mm256_i32gather_ps::<4>(tf.as_ptr(), idx) };
        let lev_i = _mm256_abs_epi32(unsafe { _mm256_i32gather_epi32::<4>(cf.as_ptr(), idx) });
        let lev = _mm256_cvtepi32_ps(lev_i);
        let q2 = q2_for_scan_chunk8(i, dc_q2, ac_q2);
        let at = _mm256_and_ps(t, abs_mask8);
        let e = _mm256_sub_ps(at, lev);
        let cur = _mm256_mul_ps(q2, _mm256_mul_ps(e, e));
        let zero = _mm256_mul_ps(q2, _mm256_mul_ps(at, at));
        unsafe {
            _mm256_storeu_ps(dst_cur.as_mut_ptr().add(i), cur);
            _mm256_storeu_ps(dst_zero.as_mut_ptr().add(i), zero);
        }
        i += 8;
    }
    while i + 4 <= n {
        let idx = unsafe { load_scan4(scan, i) };
        let t = unsafe { _mm_i32gather_ps::<4>(tf.as_ptr(), idx) };
        let lev_i = abs_i32x4(unsafe { _mm_i32gather_epi32::<4>(cf.as_ptr(), idx) });
        let lev = _mm_cvtepi32_ps(lev_i);
        let q2 = q2_for_scan_chunk4(i, dc_q2, ac_q2);
        let at = _mm_and_ps(t, abs_mask4);
        let e = _mm_sub_ps(at, lev);
        let cur = _mm_mul_ps(q2, _mm_mul_ps(e, e));
        let zero = _mm_mul_ps(q2, _mm_mul_ps(at, at));
        unsafe {
            _mm_storeu_ps(dst_cur.as_mut_ptr().add(i), cur);
            _mm_storeu_ps(dst_zero.as_mut_ptr().add(i), zero);
        }
        i += 4;
    }
    for ((out_cur, out_zero), &rc32) in dst_cur[i..n]
        .iter_mut()
        .zip(dst_zero[i..n].iter_mut())
        .zip(scan[i..n].iter())
    {
        let rc = rc32 as usize;
        let dq2 = if rc == 0 { dc_q2 } else { ac_q2 };
        let at = unsafe { (*tf.get_unchecked(rc)).abs() };
        let e = at - unsafe { (*cf.get_unchecked(rc)).unsigned_abs() } as f32;
        *out_cur = dq2 * (e * e);
        *out_zero = dq2 * (at * at);
    }
}
