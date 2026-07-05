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
#![allow(clippy::needless_range_loop)]

use crate::av2::fdct::{
    DCT4_FWD_KERNEL, DCT8_FWD_KERNEL, DCT16_FWD_KERNEL, DCT32_FWD_KERNEL, DCT64_FWD_KERNEL,
    fwd_shifts,
};
use std::arch::aarch64::*;
use std::cell::RefCell;

struct FdctScratch {
    buf: Box<[i32; 4096]>,
    coeff: Box<[i32; 4096]>,
}

thread_local! {
    static FDCT_SCRATCH: RefCell<FdctScratch> = RefCell::new(FdctScratch {
        buf: Box::new([0; 4096]),
        coeff: Box::new([0; 4096]),
    });
}

#[inline]
#[target_feature(enable = "neon")]
fn round_shift(v: int32x4_t, shift: i32) -> int32x4_t {
    if shift <= 0 {
        v
    } else {
        let add = vdupq_n_s32(1i32 << (shift - 1));
        let rshift = vdupq_n_s32(-shift);
        vshlq_s32(vaddq_s32(v, add), rshift)
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load4(src: &[i32], line: usize, j: usize, k: usize) -> int32x4_t {
    unsafe { vld1q_s32(src.as_ptr().add(k * line + j)) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store4(dst: &mut [i32], n: usize, j: usize, k: usize, v: int32x4_t) {
    let base = j * n + k;
    dst[base] = vgetq_lane_s32::<0>(v);
    dst[base + n] = vgetq_lane_s32::<1>(v);
    dst[base + 2 * n] = vgetq_lane_s32::<2>(v);
    dst[base + 3 * n] = vgetq_lane_s32::<3>(v);
}

#[inline]
#[target_feature(enable = "neon")]
fn madd_n(acc: int32x4_t, v: int32x4_t, c: i32) -> int32x4_t {
    if c == 0 {
        acc
    } else {
        vaddq_s32(acc, vmulq_n_s32(v, c))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn dot_i8<const N: usize>(
    v: &[int32x4_t; N],
    kernel: &[i8],
    row: usize,
    width: usize,
) -> int32x4_t {
    let mut acc = vdupq_n_s32(0);
    let row = &kernel[row * width..row * width + N];
    for i in 0..N {
        acc = madd_n(acc, v[i], row[i] as i32);
    }
    acc
}

#[inline]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
fn store_dot<const N: usize>(
    dst: &mut [i32],
    stride: usize,
    j: usize,
    row: usize,
    shift: i32,
    v: &[int32x4_t; N],
    kernel: &[i8],
    width: usize,
) {
    let acc = dot_i8(v, kernel, row, width);
    store4(dst, stride, j, row, round_shift(acc, shift));
}

#[target_feature(enable = "neon")]
fn fdct4_1d_4(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let s0 = load4(src, line, j, 0);
    let s1 = load4(src, line, j, 1);
    let s2 = load4(src, line, j, 2);
    let s3 = load4(src, line, j, 3);

    let a = [vaddq_s32(s0, s3), vaddq_s32(s1, s2)];
    let b = [vsubq_s32(s0, s3), vsubq_s32(s1, s2)];

    store_dot(dst, 4, j, 0, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 2, shift, &a, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 1, shift, &b, &DCT4_FWD_KERNEL, 4);
    store_dot(dst, 4, j, 3, shift, &b, &DCT4_FWD_KERNEL, 4);
}

#[target_feature(enable = "neon")]
fn fdct8_1d_4(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let mut a = [vdupq_n_s32(0); 4];
    let mut b = [vdupq_n_s32(0); 4];
    for k in 0..4 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 7 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let c = [vaddq_s32(a[0], a[3]), vaddq_s32(a[1], a[2])];
    let d = [vsubq_s32(a[0], a[3]), vsubq_s32(a[1], a[2])];

    store_dot(dst, 8, j, 0, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 4, shift, &c, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 2, shift, &d, &DCT8_FWD_KERNEL, 8);
    store_dot(dst, 8, j, 6, shift, &d, &DCT8_FWD_KERNEL, 8);
    let mut row = 1;
    while row < 8 {
        store_dot(dst, 8, j, row, shift, &b, &DCT8_FWD_KERNEL, 8);
        row += 2;
    }
}

#[target_feature(enable = "neon")]
fn fdct16_1d_4(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let mut a = [vdupq_n_s32(0); 8];
    let mut b = [vdupq_n_s32(0); 8];
    for k in 0..8 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 15 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [vdupq_n_s32(0); 4];
    let mut d = [vdupq_n_s32(0); 4];
    for k in 0..4 {
        c[k] = vaddq_s32(a[k], a[7 - k]);
        d[k] = vsubq_s32(a[k], a[7 - k]);
    }
    let e = [vaddq_s32(c[0], c[3]), vaddq_s32(c[1], c[2])];
    let f = [vsubq_s32(c[0], c[3]), vsubq_s32(c[1], c[2])];

    store_dot(dst, 16, j, 0, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 8, shift, &e, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 4, shift, &f, &DCT16_FWD_KERNEL, 16);
    store_dot(dst, 16, j, 12, shift, &f, &DCT16_FWD_KERNEL, 16);
    let mut row = 2;
    while row < 16 {
        store_dot(dst, 16, j, row, shift, &d, &DCT16_FWD_KERNEL, 16);
        row += 4;
    }
    let mut row = 1;
    while row < 16 {
        store_dot(dst, 16, j, row, shift, &b, &DCT16_FWD_KERNEL, 16);
        row += 2;
    }
}

#[target_feature(enable = "neon")]
fn fdct32_1d_4(src: &[i32], dst: &mut [i32], shift: i32, line: usize, j: usize) {
    let mut a = [vdupq_n_s32(0); 16];
    let mut b = [vdupq_n_s32(0); 16];
    for k in 0..16 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 31 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [vdupq_n_s32(0); 8];
    let mut d = [vdupq_n_s32(0); 8];
    for k in 0..8 {
        c[k] = vaddq_s32(a[k], a[15 - k]);
        d[k] = vsubq_s32(a[k], a[15 - k]);
    }
    let mut e = [vdupq_n_s32(0); 4];
    let mut f = [vdupq_n_s32(0); 4];
    for k in 0..4 {
        e[k] = vaddq_s32(c[k], c[7 - k]);
        f[k] = vsubq_s32(c[k], c[7 - k]);
    }
    let g = [vaddq_s32(e[0], e[3]), vaddq_s32(e[1], e[2])];
    let h = [vsubq_s32(e[0], e[3]), vsubq_s32(e[1], e[2])];

    store_dot(dst, 32, j, 0, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 16, shift, &g, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 8, shift, &h, &DCT32_FWD_KERNEL, 32);
    store_dot(dst, 32, j, 24, shift, &h, &DCT32_FWD_KERNEL, 32);
    let mut row = 4;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &f, &DCT32_FWD_KERNEL, 32);
        row += 8;
    }
    let mut row = 2;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &d, &DCT32_FWD_KERNEL, 32);
        row += 4;
    }
    let mut row = 1;
    while row < 32 {
        store_dot(dst, 32, j, row, shift, &b, &DCT32_FWD_KERNEL, 32);
        row += 2;
    }
}

#[target_feature(enable = "neon")]
fn fdct64_1d_4(src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero_line: usize, j: usize) {
    let top = if zero_line != 0 { 32 } else { 64 };
    let mut a = [vdupq_n_s32(0); 32];
    let mut b = [vdupq_n_s32(0); 32];
    for k in 0..32 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 63 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [vdupq_n_s32(0); 16];
    let mut d = [vdupq_n_s32(0); 16];
    for k in 0..16 {
        c[k] = vaddq_s32(a[k], a[31 - k]);
        d[k] = vsubq_s32(a[k], a[31 - k]);
    }
    let mut e = [vdupq_n_s32(0); 8];
    let mut f = [vdupq_n_s32(0); 8];
    for k in 0..8 {
        e[k] = vaddq_s32(c[k], c[15 - k]);
        f[k] = vsubq_s32(c[k], c[15 - k]);
    }
    let mut g = [vdupq_n_s32(0); 4];
    let mut h = [vdupq_n_s32(0); 4];
    for k in 0..4 {
        g[k] = vaddq_s32(e[k], e[7 - k]);
        h[k] = vsubq_s32(e[k], e[7 - k]);
    }
    let i0 = [vaddq_s32(g[0], g[3]), vaddq_s32(g[1], g[2])];
    let u0 = [vsubq_s32(g[0], g[3]), vsubq_s32(g[1], g[2])];

    store_dot(dst, 64, j, 0, shift, &i0, &DCT64_FWD_KERNEL, 64);
    store_dot(dst, 64, j, 16, shift, &u0, &DCT64_FWD_KERNEL, 64);
    if top == 64 {
        store_dot(dst, 64, j, 32, shift, &i0, &DCT64_FWD_KERNEL, 64);
        store_dot(dst, 64, j, 48, shift, &u0, &DCT64_FWD_KERNEL, 64);
    }
    let mut row = 8;
    while row < top {
        store_dot(dst, 64, j, row, shift, &h, &DCT64_FWD_KERNEL, 64);
        row += 16;
    }
    let mut row = 4;
    while row < top {
        store_dot(dst, 64, j, row, shift, &f, &DCT64_FWD_KERNEL, 64);
        row += 8;
    }
    let mut row = 2;
    while row < top {
        store_dot(dst, 64, j, row, shift, &d, &DCT64_FWD_KERNEL, 64);
        row += 4;
    }
    let mut row = 1;
    while row < top {
        store_dot(dst, 64, j, row, shift, &b, &DCT64_FWD_KERNEL, 64);
        row += 2;
    }
}

#[target_feature(enable = "neon")]
fn fdct_1d_n(n: usize, src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero: usize) {
    debug_assert!(line.is_multiple_of(4));
    let mut j = 0usize;
    while j + 4 <= line {
        match n {
            4 => fdct4_1d_4(src, dst, shift, line, j),
            8 => fdct8_1d_4(src, dst, shift, line, j),
            16 => fdct16_1d_4(src, dst, shift, line, j),
            32 => fdct32_1d_4(src, dst, shift, line, j),
            64 => fdct64_1d_4(src, dst, shift, line, zero, j),
            _ => unreachable!("unsupported 1D size {n}"),
        }
        j += 4;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn scale_rect2_in_place(out: &mut [i32]) {
    let coeff = vdup_n_s32(5793);
    let add = vdupq_n_s64(2048);
    let chunks = out.as_chunks_mut::<4>();
    for chunk in chunks.0.iter_mut() {
        unsafe {
            let v = vld1q_s32(chunk.as_ptr());
            let lo = vaddq_s64(vmull_s32(vget_low_s32(v), coeff), add);
            let hi = vaddq_s64(vmull_s32(vget_high_s32(v), coeff), add);
            let r = vcombine_s32(vshrn_n_s64::<12>(lo), vshrn_n_s64::<12>(hi));
            vst1q_s32(chunk.as_mut_ptr(), r);
        }
    }
    for v in chunks.1.iter_mut() {
        *v = (((*v as i64) * 5793 + 2048) >> 12) as i32;
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn fdct_rect_neon(resid: &[i32], w: usize, h: usize, out: &mut [i32]) -> usize {
    let (s1, s2) = fwd_shifts(w, h);
    let zh = if h > 32 { 32 } else { 0 };
    let zw = if w > 32 { 32 } else { 0 };
    let (cw, ch) = (w.min(32), h.min(32));

    FDCT_SCRATCH.with(|cell| {
        let s = &mut *cell.borrow_mut();
        fdct_1d_n(h, resid, &mut s.buf[..w * h], s1, w, zh);
        fdct_1d_n(w, &s.buf[..w * h], &mut s.coeff[..w * h], s2, h, zw);
        for vf in 0..ch {
            let src = &s.coeff[vf * w..vf * w + cw];
            let dst = &mut out[vf * cw..vf * cw + cw];
            dst.copy_from_slice(src);
        }
    });

    let n = cw * ch;
    if (w.trailing_zeros() + h.trailing_zeros()) & 1 == 1 {
        scale_rect2_in_place(&mut out[..n]);
    }
    n
}
