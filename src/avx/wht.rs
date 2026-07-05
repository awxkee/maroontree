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
#[target_feature(enable = "avx2")]
fn load_i32x4(ptr: *const i32) -> __m128i {
    unsafe { _mm_loadu_si128(ptr.cast::<__m128i>()) }
}

#[inline]
#[target_feature(enable = "avx2")]
fn store_i32x4(ptr: *mut i32, v: __m128i) {
    unsafe { _mm_storeu_si128(ptr.cast::<__m128i>(), v) };
}

#[inline]
#[target_feature(enable = "avx2")]
fn wht4(
    a0: __m128i,
    b0: __m128i,
    c0: __m128i,
    d0: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let mut a1 = _mm_add_epi32(a0, b0);
    let mut d1 = _mm_sub_epi32(d0, c0);
    let e1 = _mm_srai_epi32::<1>(_mm_sub_epi32(a1, d1));
    let b1 = _mm_sub_epi32(e1, b0);
    let c1 = _mm_sub_epi32(e1, c0);
    a1 = _mm_sub_epi32(a1, c1);
    d1 = _mm_add_epi32(d1, b1);
    (a1, c1, d1, b1)
}

#[inline]
#[target_feature(enable = "avx2")]
fn transpose_4x4(
    r0: __m128i,
    r1: __m128i,
    r2: __m128i,
    r3: __m128i,
) -> (__m128i, __m128i, __m128i, __m128i) {
    let t0 = _mm_unpacklo_epi32(r0, r1);
    let t1 = _mm_unpackhi_epi32(r0, r1);
    let t2 = _mm_unpacklo_epi32(r2, r3);
    let t3 = _mm_unpackhi_epi32(r2, r3);
    (
        _mm_unpacklo_epi64(t0, t2),
        _mm_unpackhi_epi64(t0, t2),
        _mm_unpacklo_epi64(t1, t3),
        _mm_unpackhi_epi64(t1, t3),
    )
}

#[target_feature(enable = "avx2")]
pub(crate) fn fwht_raw_avx2(input: &mut [i32; 16]) {
    let a = load_i32x4(input.as_ptr());
    let b = load_i32x4(unsafe { input.as_ptr().add(4) });
    let c = load_i32x4(unsafe { input.as_ptr().add(8) });
    let d = load_i32x4(unsafe { input.as_ptr().add(12) });

    let (r0, r1, r2, r3) = wht4(a, b, c, d);
    let (r0, r1, r2, r3) = transpose_4x4(r0, r1, r2, r3);
    let (o0, o1, o2, o3) = wht4(r0, r1, r2, r3);

    store_i32x4(input.as_mut_ptr(), o0);
    store_i32x4(unsafe { input.as_mut_ptr().add(4) }, o1);
    store_i32x4(unsafe { input.as_mut_ptr().add(8) }, o2);
    store_i32x4(unsafe { input.as_mut_ptr().add(12) }, o3);
}

#[cfg(test)]
mod avx2_vs_scalar {
    use super::*;

    fn fwht_scalar(input: &mut [i32; 16]) {
        let mut mid = [0i32; 16];
        for i in 0..4 {
            let (a0, b0, c0, d0) = (input[i], input[4 + i], input[8 + i], input[12 + i]);
            let mut a1 = a0 + b0;
            let mut d1 = d0 - c0;
            let e1 = (a1 - d1) >> 1;
            let b1 = e1 - b0;
            let c1 = e1 - c0;
            a1 -= c1;
            d1 += b1;
            mid[i] = a1;
            mid[4 + i] = c1;
            mid[8 + i] = d1;
            mid[12 + i] = b1;
        }
        for i in 0..4 {
            let base = i * 4;
            let (a0, b0, c0, d0) = (mid[base], mid[base + 1], mid[base + 2], mid[base + 3]);
            let mut a1 = a0 + b0;
            let mut d1 = d0 - c0;
            let e1 = (a1 - d1) >> 1;
            let b1 = e1 - b0;
            let c1 = e1 - c0;
            a1 -= c1;
            d1 += b1;
            input[base] = a1;
            input[base + 1] = c1;
            input[base + 2] = d1;
            input[base + 3] = b1;
        }
        let mut t = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                t[c * 4 + r] = input[r * 4 + c];
            }
        }
        *input = t;
    }

    fn lcg(state: &mut u32) -> i32 {
        *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        ((*state >> 16) as i32 & 0xff) - 128
    }

    #[test]
    fn fwht_matches_scalar() {
        if !std::is_x86_feature_detected!("avx2") {
            return;
        }
        for seed in [0x1234_5678, 0xdead_beef, 0xabcd_1234] {
            let mut state = seed;
            let mut input = [0i32; 16];
            for v in input.iter_mut() {
                *v = lcg(&mut state);
            }
            let mut s = input;
            let mut a = input;
            fwht_scalar(&mut s);
            unsafe { fwht_raw_avx2(&mut a) };
            assert_eq!(s, a, "WHT mismatch seed={seed:#x}");
        }
    }
}
