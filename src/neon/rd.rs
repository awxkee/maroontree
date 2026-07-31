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
fn load_u16x4_as_i32(src: &[u16; 4]) -> int32x4_t {
    unsafe { vreinterpretq_s32_u32(vmovl_u16(vld1_u16(src.as_ptr()))) }
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
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
) -> impl Iterator<Item = &[u16]> {
    src[py * stride..]
        .chunks_exact(stride)
        .take(h)
        .map(move |row| &row[px..px + w])
}

#[target_feature(enable = "neon")]
pub(crate) fn sum_i32_neon(values: &[i32]) -> i32 {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut acc = vdupq_n_s32(0);
    for chunk in chunks {
        acc = vaddq_s32(acc, load_i32x4(chunk));
    }
    vaddvq_s32(acc) + tail.iter().copied().sum::<i32>()
}

#[target_feature(enable = "neon")]
pub(crate) fn sum_u16_neon(values: &[u16]) -> i32 {
    let (chunks, tail) = values.as_chunks::<8>();
    let mut sum = 0u32;
    for chunk in chunks {
        sum += unsafe { vaddlvq_u16(vld1q_u16(chunk.as_ptr())) };
    }
    (sum + tail.iter().map(|&value| u32::from(value)).sum::<u32>()) as i32
}

#[target_feature(enable = "neon")]
pub(crate) fn sum_u16_strided_neon(values: &[u16], stride: usize, len: usize) -> i32 {
    let mut packed = [0u16; 8];
    let mut sum = 0u32;
    let mut index = 0;
    while index + 8 <= len {
        for lane in 0..8 {
            packed[lane] = values[(index + lane) * stride];
        }
        sum += unsafe { vaddlvq_u16(vld1q_u16(packed.as_ptr())) };
        index += 8;
    }
    for lane in index..len {
        sum += u32::from(values[lane * stride]);
    }
    sum as i32
}

#[target_feature(enable = "neon")]
pub(crate) fn all_zero_i32_neon(values: &[i32]) -> bool {
    let (chunks, tail) = values.as_chunks::<4>();
    let mut bits = vdupq_n_u32(0);
    for chunk in chunks {
        bits = vorrq_u32(bits, vreinterpretq_u32_s32(load_i32x4(chunk)));
    }
    vmaxvq_u32(bits) == 0 && tail.iter().all(|&value| value == 0)
}

#[target_feature(enable = "neon")]
pub(crate) fn residual_pred_neon(
    dst: &mut [i32],
    pred: &[i32],
    src: &[u16],
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
            let s = load_u16x4_as_i32(s);
            let p = load_i32x4(p);
            store_i32x4(d, vsubq_s32(s, p));
        }

        for ((d, &s), &p) in dst_tail.iter_mut().zip(src_tail).zip(pred_tail) {
            *d = s as i32 - p;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn residual_dc_neon(
    dst: &mut [i32],
    src: &[u16],
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
            let s = load_u16x4_as_i32(s);
            store_i32x4(d, vsubq_s32(s, dc_v));
        }

        for (d, &s) in dst_tail.iter_mut().zip(src_tail) {
            *d = s as i32 - dc;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn reconstruct_neon(
    dst: &mut [u16],
    dst_stride: usize,
    mirror: &mut [u16],
    mirror_stride: usize,
    pred: &[i32],
    resid: &[i32],
    w: usize,
    h: usize,
    maxv: i32,
) {
    let zero = vdupq_n_s32(0);
    let maxv = vdupq_n_s32(maxv);
    let mirrored = !mirror.is_empty();

    for ry in 0..h {
        let dst_row = &mut dst[ry * dst_stride..][..w];
        let pred_row = &pred[ry * w..][..w];
        let resid_row = (!resid.is_empty()).then(|| &resid[ry * w..][..w]);
        let (pred4, pred_tail) = pred_row.as_chunks::<4>();

        for (chunk, prediction) in pred4.iter().enumerate() {
            let reconstruction = if let Some(residual) = resid_row {
                let residual = residual[chunk * 4..].first_chunk::<4>().unwrap();
                vaddq_s32(load_i32x4(prediction), load_i32x4(residual))
            } else {
                load_i32x4(prediction)
            };
            let reconstruction = vminq_s32(vmaxq_s32(reconstruction, zero), maxv);
            let reconstruction = vqmovun_s32(reconstruction);
            let x = chunk * 4;
            unsafe {
                vst1_u16(dst_row[x..].as_mut_ptr(), reconstruction);
                if mirrored {
                    vst1_u16(
                        mirror[ry * mirror_stride + x..].as_mut_ptr(),
                        reconstruction,
                    );
                }
            }
        }

        let tail_x = pred4.len() * 4;
        for (lane, &prediction) in pred_tail.iter().enumerate() {
            let residual = resid_row.map_or(0, |row| row[tail_x + lane]);
            let reconstruction = (prediction + residual).clamp(0, vgetq_lane_s32::<0>(maxv)) as u16;
            dst_row[tail_x + lane] = reconstruction;
            if mirrored {
                mirror[ry * mirror_stride + tail_x + lane] = reconstruction;
            }
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn sse_recon_neon(
    pred: &[i32],
    resid: &[i32],
    src: &[u16],
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
            let s = load_u16x4_as_i32(s);
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
            let d = (s as i32 - r) as i64;
            scalar += d * d;
        }
    }

    reduce_i64x2x2(acc0, acc1) + scalar
}

#[target_feature(enable = "neon")]
pub(crate) fn chroma_sse_neon(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> i64 {
    let zero = vdupq_n_s32(0);
    let max_value_v = vdupq_n_s32(max_value);
    let dc_v = vdupq_n_s32(dc);
    let mut acc0 = vdupq_n_s64(0);
    let mut acc1 = vdupq_n_s64(0);
    let mut scalar = 0i64;

    for y in 0..h {
        let src_row = &src[(py + y) * stride + px..][..w];
        let pred_row = (!pred.is_empty()).then(|| &pred[y * w..][..w]);
        let residual_row = (!residual.is_empty()).then(|| &residual[y * w..][..w]);
        let (src4, src_tail) = src_row.as_chunks::<4>();

        for (chunk, source) in src4.iter().enumerate() {
            let x = chunk * 4;
            let prediction =
                pred_row.map_or(dc_v, |row| load_i32x4(row[x..].first_chunk::<4>().unwrap()));
            let reconstruction = residual_row.map_or(prediction, |row| {
                vaddq_s32(prediction, load_i32x4(row[x..].first_chunk::<4>().unwrap()))
            });
            let reconstruction = vminq_s32(vmaxq_s32(reconstruction, zero), max_value_v);
            let delta = vsubq_s32(load_u16x4_as_i32(source), reconstruction);
            (acc0, acc1) = square_acc_i32x4(acc0, acc1, delta);
        }

        let scalar_x = src4.len() * 4;
        for (lane, &source) in src_tail.iter().enumerate() {
            let x = scalar_x + lane;
            let prediction = pred_row.map_or(dc, |row| row[x]);
            let reconstruction = prediction + residual_row.map_or(0, |row| row[x]);
            let delta = (i32::from(source) - reconstruction.clamp(0, max_value)) as i64;
            scalar += delta * delta;
        }
    }

    reduce_i64x2x2(acc0, acc1) + scalar
}

#[target_feature(enable = "neon")]
pub(crate) fn sse_u16_neon(
    src: &[u16],
    src_stride: usize,
    src_x: usize,
    src_y: usize,
    reference: &[u16],
    ref_stride: usize,
    ref_x: usize,
    ref_y: usize,
    w: usize,
    h: usize,
) -> i64 {
    let mut acc0 = vdupq_n_s64(0);
    let mut acc1 = vdupq_n_s64(0);
    let mut scalar = 0i64;
    for row in 0..h {
        let src_row = &src[(src_y + row) * src_stride + src_x..][..w];
        let ref_row = &reference[(ref_y + row) * ref_stride + ref_x..][..w];
        let (src4, src_tail) = src_row.as_chunks::<4>();
        let (ref4, ref_tail) = ref_row.as_chunks::<4>();
        for (src, reference) in src4.iter().zip(ref4) {
            let diff = vsubq_s32(load_u16x4_as_i32(src), load_u16x4_as_i32(reference));
            (acc0, acc1) = square_acc_i32x4(acc0, acc1, diff);
        }
        for (&src, &reference) in src_tail.iter().zip(ref_tail) {
            let diff = i64::from(src) - i64::from(reference);
            scalar += diff * diff;
        }
    }
    reduce_i64x2x2(acc0, acc1) + scalar
}

/// Lane-wise 4-point Hadamard butterfly — the exact integer network of the
/// scalar `had4(a, b, c, d)` applied to four vectors at once.
#[inline]
#[target_feature(enable = "neon")]
fn had4_butterfly(
    d0: int32x4_t,
    d1: int32x4_t,
    d2: int32x4_t,
    d3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let e = vaddq_s32(d0, d2);
    let f = vsubq_s32(d0, d2);
    let g = vaddq_s32(d1, d3);
    let h = vsubq_s32(d1, d3);
    (
        vaddq_s32(e, g),
        vaddq_s32(f, h),
        vsubq_s32(f, h),
        vsubq_s32(e, g),
    )
}

/// 4x4 int32 transpose (vtrn on 32-bit lanes, then on 64-bit lanes).
#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(
    t0: int32x4_t,
    t1: int32x4_t,
    t2: int32x4_t,
    t3: int32x4_t,
) -> (int32x4_t, int32x4_t, int32x4_t, int32x4_t) {
    let a = vtrn1q_s32(t0, t1);
    let b = vtrn2q_s32(t0, t1);
    let c = vtrn1q_s32(t2, t3);
    let d = vtrn2q_s32(t2, t3);
    (
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(a),
            vreinterpretq_s64_s32(c),
        )),
        vreinterpretq_s32_s64(vtrn1q_s64(
            vreinterpretq_s64_s32(b),
            vreinterpretq_s64_s32(d),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(a),
            vreinterpretq_s64_s32(c),
        )),
        vreinterpretq_s32_s64(vtrn2q_s64(
            vreinterpretq_s64_s32(b),
            vreinterpretq_s64_s32(d),
        )),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn satd_4x4_accumulate(mut acc: int64x2_t, error: [int32x4_t; 4]) -> int64x2_t {
    let (t0, t1, t2, t3) = had4_butterfly(error[0], error[1], error[2], error[3]);
    let (r0, r1, r2, r3) = transpose_4x4(t0, t1, t2, t3);
    let (u0, u1, u2, u3) = had4_butterfly(r0, r1, r2, r3);
    acc = vpadalq_s32(acc, vabsq_s32(u0));
    acc = vpadalq_s32(acc, vabsq_s32(u1));
    acc = vpadalq_s32(acc, vabsq_s32(u2));
    vpadalq_s32(acc, vabsq_s32(u3))
}

#[target_feature(enable = "neon")]
pub(crate) fn luma_satd_neon(
    src: &[u16],
    stride: usize,
    px: usize,
    py: usize,
    w: usize,
    h: usize,
    max_value: i32,
    pred: &[i32],
    dc: i32,
    residual: &[i32],
) -> u64 {
    let zero = vdupq_n_s32(0);
    let max_value = vdupq_n_s32(max_value);
    let dc = vdupq_n_s32(dc);
    let mut satd = vdupq_n_s64(0);
    for ty in (0..h).step_by(4) {
        let src_rows: [&[[u16; 4]]; 4] =
            std::array::from_fn(|row| src[(py + ty + row) * stride + px..][..w].as_chunks::<4>().0);
        let pred_rows: Option<[&[[i32; 4]]; 4]> = (!pred.is_empty())
            .then(|| std::array::from_fn(|row| pred[(ty + row) * w..][..w].as_chunks::<4>().0));
        let residual_rows: Option<[&[[i32; 4]]; 4]> = (!residual.is_empty())
            .then(|| std::array::from_fn(|row| residual[(ty + row) * w..][..w].as_chunks::<4>().0));

        for chunk in 0..w / 4 {
            let error = std::array::from_fn(|row| {
                let source = load_u16x4_as_i32(&src_rows[row][chunk]);
                let prediction = if let Some(rows) = &pred_rows {
                    load_i32x4(&rows[row][chunk])
                } else {
                    dc
                };
                let reconstruction = if let Some(rows) = &residual_rows {
                    vaddq_s32(prediction, load_i32x4(&rows[row][chunk]))
                } else {
                    prediction
                };
                let reconstruction = vminq_s32(vmaxq_s32(reconstruction, zero), max_value);
                vsubq_s32(source, reconstruction)
            });
            satd = satd_4x4_accumulate(satd, error);
        }
    }
    vpaddd_s64(satd) as u64
}

#[target_feature(enable = "neon")]
pub(crate) fn satd_sad_proxy_neon(
    src: &[u16],
    src_stride: usize,
    pred: &[i32],
    pred_stride: usize,
    w: usize,
    h: usize,
) -> u64 {
    let mut sad_acc = vdupq_n_s64(0);
    let mut satd_acc = vdupq_n_s64(0);
    for ty in (0..h).step_by(4) {
        let src_rows: [&[u16]; 4] =
            std::array::from_fn(|r| &src[(ty + r) * src_stride..(ty + r) * src_stride + w]);
        let pred_rows: [&[i32]; 4] =
            std::array::from_fn(|r| &pred[(ty + r) * pred_stride..(ty + r) * pred_stride + w]);
        let s: [&[[u16; 4]]; 4] = std::array::from_fn(|r| src_rows[r].as_chunks::<4>().0);
        let p: [&[[i32; 4]]; 4] = std::array::from_fn(|r| pred_rows[r].as_chunks::<4>().0);
        for i in 0..w / 4 {
            let d0 = vsubq_s32(load_u16x4_as_i32(&s[0][i]), load_i32x4(&p[0][i]));
            let d1 = vsubq_s32(load_u16x4_as_i32(&s[1][i]), load_i32x4(&p[1][i]));
            let d2 = vsubq_s32(load_u16x4_as_i32(&s[2][i]), load_i32x4(&p[2][i]));
            let d3 = vsubq_s32(load_u16x4_as_i32(&s[3][i]), load_i32x4(&p[3][i]));

            sad_acc = vpadalq_s32(sad_acc, vabsq_s32(d0));
            sad_acc = vpadalq_s32(sad_acc, vabsq_s32(d1));
            sad_acc = vpadalq_s32(sad_acc, vabsq_s32(d2));
            sad_acc = vpadalq_s32(sad_acc, vabsq_s32(d3));

            satd_acc = satd_4x4_accumulate(satd_acc, [d0, d1, d2, d3]);
        }
    }
    let sad = vpaddd_s64(sad_acc);
    let satd = vpaddd_s64(satd_acc);
    (sad as u64) + ((satd as u64) >> 2)
}
