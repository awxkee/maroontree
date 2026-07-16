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

use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x4(src: &[i32; 4]) -> int32x4_t {
    unsafe { vld1q_s32(src.as_ptr()) }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_i32x4(dst: &mut [i32; 4], v: int32x4_t) {
    unsafe { vst1q_s32(dst.as_mut_ptr(), v) }
}

#[inline]
#[target_feature(enable = "neon")]
fn square_acc_i32x4(acc0: int64x2_t, acc1: int64x2_t, d: int32x4_t) -> (int64x2_t, int64x2_t) {
    let lo = vget_low_s32(d);
    let hi = vget_high_s32(d);

    (
        vaddq_s64(acc0, vmull_s32(lo, lo)),
        vaddq_s64(acc1, vmull_s32(hi, hi)),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn reduce_i64x2x2(acc0: int64x2_t, acc1: int64x2_t) -> i64 {
    vaddvq_s64(vaddq_s64(acc0, acc1))
}

#[inline]
fn src_rows(
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> impl Iterator<Item = &[i32]> {
    src[py * stride..]
        .chunks_exact(stride)
        .take(h)
        .map(move |row| &row[px..px + w])
}

#[target_feature(enable = "neon")]
pub(crate) fn residual_pred_neon(
    dst: &mut [i32],
    pred: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }

    let dst_rows = dst.chunks_exact_mut(w).take(h);
    let pred_rows = pred.chunks_exact(w).take(h);
    let src_rows = src_rows(src, stride, px, py, w, h);

    for ((dst_row, pred_row), src_row) in dst_rows.zip(pred_rows).zip(src_rows) {
        let (dst4, dst_tail) = dst_row.as_chunks_mut::<4>();
        let (pred4, pred_tail) = pred_row.as_chunks::<4>();
        let (src4, src_tail) = src_row.as_chunks::<4>();

        for ((d, s), p) in dst4.iter_mut().zip(src4).zip(pred4) {
            let s = load_i32x4(s);
            let p = load_i32x4(p);
            store_i32x4(d, vsubq_s32(s, p));
        }

        for ((d, &s), &p) in dst_tail.iter_mut().zip(src_tail).zip(pred_tail) {
            *d = s - p;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn residual_dc_neon(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    dc: i32,
) {
    if w == 0 || h == 0 {
        return;
    }

    let dc_v = vdupq_n_s32(dc);

    let dst_rows = dst.chunks_exact_mut(w).take(h);
    let src_rows = src_rows(src, stride, px, py, w, h);

    for (dst_row, src_row) in dst_rows.zip(src_rows) {
        let (dst4, dst_tail) = dst_row.as_chunks_mut::<4>();
        let (src4, src_tail) = src_row.as_chunks::<4>();

        for (d, s) in dst4.iter_mut().zip(src4) {
            let s = load_i32x4(s);
            store_i32x4(d, vsubq_s32(s, dc_v));
        }

        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = s - dc;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn sse_recon_neon(
    pred: &[i32],
    resid: &[i32],
    src: &[i32],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    maxv: i32,
) -> i64 {
    if w == 0 || h == 0 {
        return 0;
    }

    let zero = vdupq_n_s32(0);
    let maxv_v = vdupq_n_s32(maxv);

    let mut acc0 = vdupq_n_s64(0);
    let mut acc1 = vdupq_n_s64(0);
    let mut scalar = 0i64;

    let pred_rows = pred.chunks_exact(w).take(h);
    let resid_rows = resid.chunks_exact(w).take(h);
    let src_rows = src_rows(src, stride, px, py, w, h);

    for ((pred_row, resid_row), src_row) in pred_rows.zip(resid_rows).zip(src_rows) {
        let (pred4, pred_tail) = pred_row.as_chunks::<4>();
        let (resid4, resid_tail) = resid_row.as_chunks::<4>();
        let (src4, src_tail) = src_row.as_chunks::<4>();

        for ((s, p), e) in src4.iter().zip(pred4).zip(resid4) {
            let s = load_i32x4(s);
            let p = load_i32x4(p);
            let e = load_i32x4(e);

            let r = vaddq_s32(p, e);
            let r = vmaxq_s32(r, zero);
            let r = vminq_s32(r, maxv_v);

            let d = vsubq_s32(s, r);
            (acc0, acc1) = square_acc_i32x4(acc0, acc1, d);
        }

        for ((&s, &p), &e) in src_tail.iter().zip(pred_tail).zip(resid_tail) {
            let r = (p + e).clamp(0, maxv);
            let d = (s - r) as i64;
            scalar += d * d;
        }
    }

    reduce_i64x2x2(acc0, acc1) + scalar
}
