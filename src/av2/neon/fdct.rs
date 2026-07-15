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
fn dot_shift<const N: usize>(
    shift: i32,
    v: &[int32x4_t; N],
    kernel: &[i8],
    row: usize,
    width: usize,
) -> int32x4_t {
    round_shift(dot_i8(v, kernel, row, width), shift)
}

/// Transpose four frequency vectors whose lanes hold four adjacent transform
/// lines. The returned vectors are complete contiguous coefficient rows.
#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_i32(r: [int32x4_t; 4]) -> [int32x4_t; 4] {
    let t0 = vtrn1q_s32(r[0], r[1]);
    let t1 = vtrn2q_s32(r[0], r[1]);
    let t2 = vtrn1q_s32(r[2], r[3]);
    let t3 = vtrn2q_s32(r[2], r[3]);
    [
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(t0),
            vreinterpretq_s64_s32(t2),
        )),
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(t1),
            vreinterpretq_s64_s32(t3),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(t0),
            vreinterpretq_s64_s32(t2),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(t1),
            vreinterpretq_s64_s32(t3),
        )),
    ]
}

/// Store a complete four-line SIMD batch after a register transpose. Every
/// write is one contiguous 128-bit store into the transposed scratch consumed
/// by the next 1-D pass.
#[inline]
#[target_feature(enable = "neon")]
fn transpose_store<const N: usize>(
    dst: &mut [i32],
    j: usize,
    coeff: &[int32x4_t; N],
    valid: usize,
) {
    debug_assert!(valid <= N && valid.is_multiple_of(4));
    let dst = dst.as_mut_ptr();
    for freq in (0..valid).step_by(4) {
        let rows = transpose_4x4_i32([
            coeff[freq],
            coeff[freq + 1],
            coeff[freq + 2],
            coeff[freq + 3],
        ]);
        for lane in 0..4 {
            unsafe { vst1q_s32(dst.add((j + lane) * N + freq), rows[lane]) };
        }
    }
}

#[target_feature(enable = "neon")]
fn fdct4_1d(src: &[i32], shift: i32, line: usize, j: usize) -> [int32x4_t; 4] {
    let s0 = load4(src, line, j, 0);
    let s1 = load4(src, line, j, 1);
    let s2 = load4(src, line, j, 2);
    let s3 = load4(src, line, j, 3);
    let a = [vaddq_s32(s0, s3), vaddq_s32(s1, s2)];
    let b = [vsubq_s32(s0, s3), vsubq_s32(s1, s2)];
    let mut out = [vdupq_n_s32(0); 4];
    out[0] = dot_shift(shift, &a, &DCT4_FWD_KERNEL, 0, 4);
    out[2] = dot_shift(shift, &a, &DCT4_FWD_KERNEL, 2, 4);
    out[1] = dot_shift(shift, &b, &DCT4_FWD_KERNEL, 1, 4);
    out[3] = dot_shift(shift, &b, &DCT4_FWD_KERNEL, 3, 4);
    out
}

#[target_feature(enable = "neon")]
fn fdct8_1d(src: &[i32], shift: i32, line: usize, j: usize) -> [int32x4_t; 8] {
    let z = vdupq_n_s32(0);
    let mut a = [z; 4];
    let mut b = [z; 4];
    for k in 0..4 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 7 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let c = [vaddq_s32(a[0], a[3]), vaddq_s32(a[1], a[2])];
    let d = [vsubq_s32(a[0], a[3]), vsubq_s32(a[1], a[2])];
    let mut out = [z; 8];
    out[0] = dot_shift(shift, &c, &DCT8_FWD_KERNEL, 0, 8);
    out[4] = dot_shift(shift, &c, &DCT8_FWD_KERNEL, 4, 8);
    out[2] = dot_shift(shift, &d, &DCT8_FWD_KERNEL, 2, 8);
    out[6] = dot_shift(shift, &d, &DCT8_FWD_KERNEL, 6, 8);
    for row in (1..8).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT8_FWD_KERNEL, row, 8);
    }
    out
}

#[target_feature(enable = "neon")]
fn fdct16_1d(src: &[i32], shift: i32, line: usize, j: usize) -> [int32x4_t; 16] {
    let z = vdupq_n_s32(0);
    let mut a = [z; 8];
    let mut b = [z; 8];
    for k in 0..8 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 15 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [z; 4];
    let mut d = [z; 4];
    for k in 0..4 {
        c[k] = vaddq_s32(a[k], a[7 - k]);
        d[k] = vsubq_s32(a[k], a[7 - k]);
    }
    let e = [vaddq_s32(c[0], c[3]), vaddq_s32(c[1], c[2])];
    let f = [vsubq_s32(c[0], c[3]), vsubq_s32(c[1], c[2])];
    let mut out = [z; 16];
    out[0] = dot_shift(shift, &e, &DCT16_FWD_KERNEL, 0, 16);
    out[8] = dot_shift(shift, &e, &DCT16_FWD_KERNEL, 8, 16);
    out[4] = dot_shift(shift, &f, &DCT16_FWD_KERNEL, 4, 16);
    out[12] = dot_shift(shift, &f, &DCT16_FWD_KERNEL, 12, 16);
    for row in (2..16).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT16_FWD_KERNEL, row, 16);
    }
    for row in (1..16).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT16_FWD_KERNEL, row, 16);
    }
    out
}

#[target_feature(enable = "neon")]
fn fdct32_1d(src: &[i32], shift: i32, line: usize, j: usize) -> [int32x4_t; 32] {
    let z = vdupq_n_s32(0);
    let mut a = [z; 16];
    let mut b = [z; 16];
    for k in 0..16 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 31 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [z; 8];
    let mut d = [z; 8];
    for k in 0..8 {
        c[k] = vaddq_s32(a[k], a[15 - k]);
        d[k] = vsubq_s32(a[k], a[15 - k]);
    }
    let mut e = [z; 4];
    let mut f = [z; 4];
    for k in 0..4 {
        e[k] = vaddq_s32(c[k], c[7 - k]);
        f[k] = vsubq_s32(c[k], c[7 - k]);
    }
    let g = [vaddq_s32(e[0], e[3]), vaddq_s32(e[1], e[2])];
    let h = [vsubq_s32(e[0], e[3]), vsubq_s32(e[1], e[2])];
    let mut out = [z; 32];
    out[0] = dot_shift(shift, &g, &DCT32_FWD_KERNEL, 0, 32);
    out[16] = dot_shift(shift, &g, &DCT32_FWD_KERNEL, 16, 32);
    out[8] = dot_shift(shift, &h, &DCT32_FWD_KERNEL, 8, 32);
    out[24] = dot_shift(shift, &h, &DCT32_FWD_KERNEL, 24, 32);
    for row in (4..32).step_by(8) {
        out[row] = dot_shift(shift, &f, &DCT32_FWD_KERNEL, row, 32);
    }
    for row in (2..32).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT32_FWD_KERNEL, row, 32);
    }
    for row in (1..32).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT32_FWD_KERNEL, row, 32);
    }
    out
}

#[target_feature(enable = "neon")]
fn fdct64_1d(src: &[i32], shift: i32, line: usize, zero_line: usize, j: usize) -> [int32x4_t; 64] {
    let top = if zero_line != 0 { 32 } else { 64 };
    let z = vdupq_n_s32(0);
    let mut a = [z; 32];
    let mut b = [z; 32];
    for k in 0..32 {
        let lo = load4(src, line, j, k);
        let hi = load4(src, line, j, 63 - k);
        a[k] = vaddq_s32(lo, hi);
        b[k] = vsubq_s32(lo, hi);
    }
    let mut c = [z; 16];
    let mut d = [z; 16];
    for k in 0..16 {
        c[k] = vaddq_s32(a[k], a[31 - k]);
        d[k] = vsubq_s32(a[k], a[31 - k]);
    }
    let mut e = [z; 8];
    let mut f = [z; 8];
    for k in 0..8 {
        e[k] = vaddq_s32(c[k], c[15 - k]);
        f[k] = vsubq_s32(c[k], c[15 - k]);
    }
    let mut g = [z; 4];
    let mut h = [z; 4];
    for k in 0..4 {
        g[k] = vaddq_s32(e[k], e[7 - k]);
        h[k] = vsubq_s32(e[k], e[7 - k]);
    }
    let i0 = [vaddq_s32(g[0], g[3]), vaddq_s32(g[1], g[2])];
    let u0 = [vsubq_s32(g[0], g[3]), vsubq_s32(g[1], g[2])];
    let mut out = [z; 64];
    out[0] = dot_shift(shift, &i0, &DCT64_FWD_KERNEL, 0, 64);
    out[16] = dot_shift(shift, &u0, &DCT64_FWD_KERNEL, 16, 64);
    if top == 64 {
        out[32] = dot_shift(shift, &i0, &DCT64_FWD_KERNEL, 32, 64);
        out[48] = dot_shift(shift, &u0, &DCT64_FWD_KERNEL, 48, 64);
    }
    for row in (8..top).step_by(16) {
        out[row] = dot_shift(shift, &h, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (4..top).step_by(8) {
        out[row] = dot_shift(shift, &f, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (2..top).step_by(4) {
        out[row] = dot_shift(shift, &d, &DCT64_FWD_KERNEL, row, 64);
    }
    for row in (1..top).step_by(2) {
        out[row] = dot_shift(shift, &b, &DCT64_FWD_KERNEL, row, 64);
    }
    out
}

#[inline]
#[target_feature(enable = "neon")]
fn fdct_batch(
    n: usize,
    src: &[i32],
    dst: &mut [i32],
    shift: i32,
    line: usize,
    zero: usize,
    j: usize,
) {
    let valid = if n == 64 && zero != 0 { 32 } else { n };
    match n {
        4 => transpose_store(dst, j, &fdct4_1d(src, shift, line, j), valid),
        8 => transpose_store(dst, j, &fdct8_1d(src, shift, line, j), valid),
        16 => transpose_store(dst, j, &fdct16_1d(src, shift, line, j), valid),
        32 => transpose_store(dst, j, &fdct32_1d(src, shift, line, j), valid),
        64 => transpose_store(dst, j, &fdct64_1d(src, shift, line, zero, j), valid),
        _ => unreachable!("unsupported 1D size {n}"),
    }
}

#[target_feature(enable = "neon")]
fn fdct_1d_n(n: usize, src: &[i32], dst: &mut [i32], shift: i32, line: usize, zero: usize) {
    debug_assert!(line.is_multiple_of(4));
    for j in (0..line).step_by(4) {
        fdct_batch(n, src, dst, shift, line, zero, j);
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
        for (dst_row, src_row) in out[..cw * ch]
            .chunks_exact_mut(cw)
            .zip(s.coeff.chunks_exact(w).take(ch))
        {
            dst_row.copy_from_slice(&src_row[..cw]);
        }
    });

    let n = cw * ch;
    if (w.trailing_zeros() + h.trailing_zeros()) & 1 == 1 {
        scale_rect2_in_place(&mut out[..n]);
    }
    n
}
