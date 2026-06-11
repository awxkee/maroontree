/*
 * Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
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
use std::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn wht4(
    a0: int32x4_t,
    b0: int32x4_t,
    c0: int32x4_t,
    d0: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let mut a1 = vaddq_s32(a0, b0);
    let mut d1 = vsubq_s32(d0, c0);

    let e1 = vshrq_n_s32::<1>(vsubq_s32(a1, d1));

    let b1 = vsubq_s32(e1, b0);
    let c1 = vsubq_s32(e1, c0);

    a1 = vsubq_s32(a1, c1);
    d1 = vaddq_s32(d1, b1);

    (a1, c1, d1, b1)
}

pub(crate) fn fwht_raw_neon(input: &mut [i32; 16]) {
    unsafe { fwht_raw_neon_impl(input) }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(
    r0: int32x4_t,
    r1: int32x4_t,
    r2: int32x4_t,
    r3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let v0 = vtrn1q_s32(r0, r1);
    let v1 = vtrn2q_s32(r0, r1);
    let v2 = vtrn1q_s32(r2, r3);
    let v3 = vtrn2q_s32(r2, r3);
    let c0 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(v0),
        vreinterpretq_s64_s32(v2),
    ));
    let c1 = vreinterpretq_s32_s64(vtrn1q_s64(
        vreinterpretq_s64_s32(v1),
        vreinterpretq_s64_s32(v3),
    ));
    let c2 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(v0),
        vreinterpretq_s64_s32(v2),
    ));
    let c3 = vreinterpretq_s32_s64(vtrn2q_s64(
        vreinterpretq_s64_s32(v1),
        vreinterpretq_s64_s32(v3),
    ));
    (c0, c1, c2, c3)
}

#[inline]
#[target_feature(enable = "neon")]
fn fwht_raw_neon_impl(input: &mut [i32; 16]) {
    let a = unsafe { vld1q_s32(input.as_ptr()) };
    let b = unsafe { vld1q_s32(input[4..].as_ptr()) };
    let c = unsafe { vld1q_s32(input[8..].as_ptr()) };
    let d = unsafe { vld1q_s32(input[12..].as_ptr()) };

    let (r0, r1, r2, r3) = wht4(a, b, c, d);

    let (r0, r1, r2, r3) = transpose_4x4(r0, r1, r2, r3);
    let (o0, o1, o2, o3) = wht4(r0, r1, r2, r3);

    unsafe {
        vst1q_s32(input.as_mut_ptr(), o0);
        vst1q_s32(input[4..].as_mut_ptr(), o1);
        vst1q_s32(input[8..].as_mut_ptr(), o2);
        vst1q_s32(input[12..].as_mut_ptr(), o3);
    }
}
