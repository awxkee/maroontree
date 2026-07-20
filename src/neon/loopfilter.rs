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
use core::arch::aarch64::*;

#[inline]
#[target_feature(enable = "neon")]
fn load_sample(
    dst: &[u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
) -> int32x4_t {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        vreinterpretq_s32_u32(vmovl_u16(unsafe { vld1_u16(dst.as_ptr().add(pos)) }))
    } else {
        let pos = base as isize + offset * stride_b;
        let mut packed = vdup_n_u16(0);
        packed = vset_lane_u16::<0>(dst[pos as usize], packed);
        packed = vset_lane_u16::<1>(dst[(pos + stride_a) as usize], packed);
        packed = vset_lane_u16::<2>(dst[(pos + 2 * stride_a) as usize], packed);
        packed = vset_lane_u16::<3>(dst[(pos + 3 * stride_a) as usize], packed);
        vreinterpretq_s32_u32(vmovl_u16(packed))
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_sample(
    dst: &mut [u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
    value: int32x4_t,
) {
    let packed = vqmovun_s32(value);
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        unsafe { vst1_u16(dst.as_mut_ptr().add(pos), packed) };
    } else {
        let pos = base as isize + offset * stride_b;
        dst[pos as usize] = vget_lane_u16::<0>(packed);
        dst[(pos + stride_a) as usize] = vget_lane_u16::<1>(packed);
        dst[(pos + 2 * stride_a) as usize] = vget_lane_u16::<2>(packed);
        dst[(pos + 3 * stride_a) as usize] = vget_lane_u16::<3>(packed);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4_u16(values: &mut [uint16x4_t; 4]) {
    let a0 = vtrn1_u16(values[0], values[1]);
    let a1 = vtrn2_u16(values[0], values[1]);
    let a2 = vtrn1_u16(values[2], values[3]);
    let a3 = vtrn2_u16(values[2], values[3]);
    values[0] = vreinterpret_u16_u32(vtrn1_u32(
        vreinterpret_u32_u16(a0),
        vreinterpret_u32_u16(a2),
    ));
    values[1] = vreinterpret_u16_u32(vtrn1_u32(
        vreinterpret_u32_u16(a1),
        vreinterpret_u32_u16(a3),
    ));
    values[2] = vreinterpret_u16_u32(vtrn2_u32(
        vreinterpret_u32_u16(a0),
        vreinterpret_u32_u16(a2),
    ));
    values[3] = vreinterpret_u16_u32(vtrn2_u32(
        vreinterpret_u32_u16(a1),
        vreinterpret_u32_u16(a3),
    ));
}

#[inline]
#[target_feature(enable = "neon")]
fn load_vertical_4_lines(dst: &[u16], base: usize, stride: isize, offset: isize) -> [int32x4_t; 4] {
    let src = unsafe { dst.as_ptr().add((base as isize + offset) as usize) };
    let mut rows = [vdup_n_u16(0); 4];
    for (row, value) in rows.iter_mut().enumerate() {
        *value = unsafe { vld1_u16(src.offset(row as isize * stride)) };
    }
    transpose_4x4_u16(&mut rows);
    std::array::from_fn(|i| vreinterpretq_s32_u32(vmovl_u16(rows[i])))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_vertical_edge(dst: &[u16], base: usize, stride: isize, wd: i32) -> [int32x4_t; 14] {
    let mut samples = [vdupq_n_s32(0); 14];
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
#[target_feature(enable = "neon")]
fn store_vertical_4_lines(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[int32x4_t],
) {
    let mut rows = std::array::from_fn(|i| vqmovun_s32(values[i]));
    transpose_4x4_u16(&mut rows);
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for (row, value) in rows.iter().enumerate() {
        unsafe { vst1_u16(dst.offset(row as isize * stride), *value) };
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_vertical_2_lines(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[int32x4_t],
) {
    let a = vqmovun_s32(values[0]);
    let b = vqmovun_s32(values[1]);
    let lo = vreinterpret_u32_u16(vzip1_u16(a, b));
    let hi = vreinterpret_u32_u16(vzip2_u16(a, b));
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    unsafe {
        vst1_lane_u32::<0>(dst.cast(), lo);
        vst1_lane_u32::<1>(dst.offset(stride).cast(), lo);
        vst1_lane_u32::<0>(dst.offset(2 * stride).cast(), hi);
        vst1_lane_u32::<1>(dst.offset(3 * stride).cast(), hi);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn abs_diff(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    vabdq_s32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
fn mask_and(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    vandq_u32(a, b)
}

#[inline]
#[target_feature(enable = "neon")]
fn mask_not(a: uint32x4_t) -> uint32x4_t {
    vmvnq_u32(a)
}

#[inline]
#[target_feature(enable = "neon")]
fn mask_any(a: uint32x4_t) -> bool {
    vmaxvq_u32(a) != 0
}

#[inline]
#[target_feature(enable = "neon")]
fn select(mask: uint32x4_t, yes: int32x4_t, no: int32x4_t) -> int32x4_t {
    vbslq_s32(mask, yes, no)
}

#[inline]
#[target_feature(enable = "neon")]
fn clip(value: int32x4_t, lo: int32x4_t, hi: int32x4_t) -> int32x4_t {
    vmaxq_s32(lo, vminq_s32(value, hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn weighted(samples: &[int32x4_t], weights: &[i32], bias: i32, shift: i32) -> int32x4_t {
    let mut sum = vdupq_n_s32(bias);
    for (&sample, &weight) in samples.iter().zip(weights) {
        if weight != 0 {
            sum = vmlaq_n_s32(sum, sample, weight);
        }
    }
    vshlq_s32(sum, vdupq_n_s32(-shift))
}

/// Four-lane dav1d-style loop filter. Each lane is one line crossing the edge;
/// filter/HEV/flat decisions are masks and all selected widths are branchless
/// within the four-line segment.
#[target_feature(enable = "neon")]
pub(crate) fn loop_filter_neon(
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
        let mut samples = [vdupq_n_s32(0); 14];
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
    let e = vdupq_n_s32(e * scale);
    let i_lim = vdupq_n_s32(i_lim * scale);
    let h_thresh = vdupq_n_s32(h_thresh * scale);
    let flat_limit = vdupq_n_s32(scale);
    let zero = vdupq_n_s32(0);
    let clip_lo = vdupq_n_s32(-128 * scale);
    let clip_hi = vdupq_n_s32(128 * scale - 1);
    let pixel_hi = vdupq_n_s32((1 << bd) - 1);

    let p1 = s[5];
    let p0 = s[6];
    let q0 = s[7];
    let q1 = s[8];

    let mut fm = mask_and(
        vcleq_s32(abs_diff(p1, p0), i_lim),
        vcleq_s32(abs_diff(q1, q0), i_lim),
    );
    let edge_metric = vaddq_s32(
        vshlq_n_s32::<1>(abs_diff(p0, q0)),
        vshrq_n_s32::<1>(abs_diff(p1, q1)),
    );
    fm = mask_and(fm, vcleq_s32(edge_metric, e));
    if wd > 4 {
        fm = mask_and(fm, vcleq_s32(abs_diff(s[4], p1), i_lim));
        fm = mask_and(fm, vcleq_s32(abs_diff(s[9], q1), i_lim));
    }
    if wd > 6 {
        fm = mask_and(fm, vcleq_s32(abs_diff(s[3], s[4]), i_lim));
        fm = mask_and(fm, vcleq_s32(abs_diff(s[10], s[9]), i_lim));
    }
    if !mask_any(fm) {
        return;
    }

    let mut flat_in = vdupq_n_u32(0);
    if wd >= 6 {
        flat_in = vcleq_s32(abs_diff(s[4], p0), flat_limit);
        flat_in = mask_and(flat_in, vcleq_s32(abs_diff(p1, p0), flat_limit));
        flat_in = mask_and(flat_in, vcleq_s32(abs_diff(q1, q0), flat_limit));
        flat_in = mask_and(flat_in, vcleq_s32(abs_diff(s[9], q0), flat_limit));
    }
    if wd >= 8 {
        flat_in = mask_and(flat_in, vcleq_s32(abs_diff(s[3], p0), flat_limit));
        flat_in = mask_and(flat_in, vcleq_s32(abs_diff(s[10], q0), flat_limit));
    }

    let mut flat_out = vdupq_n_u32(0);
    if wd >= 16 {
        flat_out = vcleq_s32(abs_diff(s[0], p0), flat_limit);
        flat_out = mask_and(flat_out, vcleq_s32(abs_diff(s[1], p0), flat_limit));
        flat_out = mask_and(flat_out, vcleq_s32(abs_diff(s[2], p0), flat_limit));
        flat_out = mask_and(flat_out, vcleq_s32(abs_diff(s[11], q0), flat_limit));
        flat_out = mask_and(flat_out, vcleq_s32(abs_diff(s[12], q0), flat_limit));
        flat_out = mask_and(flat_out, vcleq_s32(abs_diff(s[13], q0), flat_limit));
    }

    let wide16 = if wd >= 16 {
        mask_and(fm, mask_and(flat_in, flat_out))
    } else {
        vdupq_n_u32(0)
    };
    let wide8 = if wd >= 8 {
        mask_and(mask_and(fm, flat_in), mask_not(wide16))
    } else {
        vdupq_n_u32(0)
    };
    let wide6 = if wd == 6 {
        mask_and(fm, flat_in)
    } else {
        vdupq_n_u32(0)
    };
    let wide = vorrq_u32(vorrq_u32(wide16, wide8), wide6);
    let short = mask_and(fm, mask_not(wide));

    let hev = vorrq_u32(
        vcgtq_s32(abs_diff(p1, p0), h_thresh),
        vcgtq_s32(abs_diff(q1, q0), h_thresh),
    );
    let delta = vsubq_s32(q0, p0);
    let triple = vaddq_s32(delta, vshlq_n_s32::<1>(delta));
    let fv_plain = clip(triple, clip_lo, clip_hi);
    let fv_hev = clip(
        vaddq_s32(triple, clip(vsubq_s32(p1, q1), clip_lo, clip_hi)),
        clip_lo,
        clip_hi,
    );
    let fv = select(hev, fv_hev, fv_plain);
    let f1 = vshrq_n_s32::<3>(vminq_s32(vaddq_s32(fv, vdupq_n_s32(4)), clip_hi));
    let f2 = vshrq_n_s32::<3>(vminq_s32(vaddq_s32(fv, vdupq_n_s32(3)), clip_hi));

    let mut out = s;
    out[6] = select(short, clip(vaddq_s32(p0, f2), zero, pixel_hi), out[6]);
    out[7] = select(short, clip(vsubq_s32(q0, f1), zero, pixel_hi), out[7]);
    let outer_short = mask_and(short, mask_not(hev));
    let f = vshrq_n_s32::<1>(vaddq_s32(f1, vdupq_n_s32(1)));
    out[5] = select(outer_short, clip(vaddq_s32(p1, f), zero, pixel_hi), out[5]);
    out[8] = select(outer_short, clip(vsubq_s32(q1, f), zero, pixel_hi), out[8]);

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
        for (index, &value) in (first..=last).zip(out[first..=last].iter()) {
            store_sample(dst, base, stride_a, stride_b, index as isize - 7, value);
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
#[target_feature(enable = "neon")]
fn load_sample_batch(
    dst: &[u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
) -> uint16x8_t {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        unsafe { vld1q_u16(dst.as_ptr().add(pos)) }
    } else {
        let pos = base as isize + offset * stride_b;
        let src = dst.as_ptr();
        let mut value = vdupq_n_u16(0);
        unsafe {
            value = vsetq_lane_u16::<0>(*src.offset(pos), value);
            value = vsetq_lane_u16::<1>(*src.offset(pos + stride_a), value);
            value = vsetq_lane_u16::<2>(*src.offset(pos + 2 * stride_a), value);
            value = vsetq_lane_u16::<3>(*src.offset(pos + 3 * stride_a), value);
            value = vsetq_lane_u16::<4>(*src.offset(pos + 4 * stride_a), value);
            value = vsetq_lane_u16::<5>(*src.offset(pos + 5 * stride_a), value);
            value = vsetq_lane_u16::<6>(*src.offset(pos + 6 * stride_a), value);
            vsetq_lane_u16::<7>(*src.offset(pos + 7 * stride_a), value)
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_8x8_u16(values: &mut [uint16x8_t; 8]) {
    let mut half = [values[0]; 8];
    for i in 0..4 {
        half[2 * i] = vtrn1q_u16(values[2 * i], values[2 * i + 1]);
        half[2 * i + 1] = vtrn2q_u16(values[2 * i], values[2 * i + 1]);
    }

    let mut word = [values[0]; 8];
    for i in 0..2 {
        let j = 4 * i;
        word[j] = vreinterpretq_u16_u32(vtrn1q_u32(
            vreinterpretq_u32_u16(half[j]),
            vreinterpretq_u32_u16(half[j + 2]),
        ));
        word[j + 2] = vreinterpretq_u16_u32(vtrn2q_u32(
            vreinterpretq_u32_u16(half[j]),
            vreinterpretq_u32_u16(half[j + 2]),
        ));
        word[j + 1] = vreinterpretq_u16_u32(vtrn1q_u32(
            vreinterpretq_u32_u16(half[j + 1]),
            vreinterpretq_u32_u16(half[j + 3]),
        ));
        word[j + 3] = vreinterpretq_u16_u32(vtrn2q_u32(
            vreinterpretq_u32_u16(half[j + 1]),
            vreinterpretq_u32_u16(half[j + 3]),
        ));
    }

    for i in 0..4 {
        values[i] = vreinterpretq_u16_u64(vtrn1q_u64(
            vreinterpretq_u64_u16(word[i]),
            vreinterpretq_u64_u16(word[i + 4]),
        ));
        values[i + 4] = vreinterpretq_u16_u64(vtrn2q_u64(
            vreinterpretq_u64_u16(word[i]),
            vreinterpretq_u64_u16(word[i + 4]),
        ));
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn load_vertical_4(dst: &[u16], base: usize, stride: isize, offset: isize) -> [uint16x8_t; 4] {
    let src = unsafe { dst.as_ptr().add((base as isize + offset) as usize) };
    let zero = vdup_n_u16(0);
    let mut rows = [vdupq_n_u16(0); 8];
    for (row, value) in rows.iter_mut().enumerate() {
        *value = vcombine_u16(unsafe { vld1_u16(src.offset(row as isize * stride)) }, zero);
    }
    transpose_8x8_u16(&mut rows);
    [rows[0], rows[1], rows[2], rows[3]]
}

#[inline]
#[target_feature(enable = "neon")]
fn load_vertical_8(dst: &[u16], base: usize, stride: isize, offset: isize) -> [uint16x8_t; 8] {
    let src = unsafe { dst.as_ptr().add((base as isize + offset) as usize) };
    let mut rows = [vdupq_n_u16(0); 8];
    for (row, value) in rows.iter_mut().enumerate() {
        *value = unsafe { vld1q_u16(src.offset(row as isize * stride)) };
    }
    transpose_8x8_u16(&mut rows);
    rows
}

#[inline]
#[target_feature(enable = "neon")]
fn load_vertical_batch(dst: &[u16], base: usize, stride: isize, wd: i32) -> [uint16x8_t; 14] {
    let zero = vdupq_n_u16(0);
    let mut samples = [zero; 14];
    match wd {
        4 => samples[5..9].copy_from_slice(&load_vertical_4(dst, base, stride, -2)),
        6 => {
            samples[4..8].copy_from_slice(&load_vertical_4(dst, base, stride, -3));
            samples[6..10].copy_from_slice(&load_vertical_4(dst, base, stride, -1));
        }
        8 => samples[3..11].copy_from_slice(&load_vertical_8(dst, base, stride, -4)),
        16 => {
            samples[0..8].copy_from_slice(&load_vertical_8(dst, base, stride, -7));
            samples[6..14].copy_from_slice(&load_vertical_8(dst, base, stride, -1));
        }
        _ => unreachable!("unsupported loop-filter width"),
    }
    samples
}

#[inline]
#[target_feature(enable = "neon")]
fn store_vertical_2(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[uint16x8_t],
) {
    let lo = vreinterpretq_u32_u16(vzip1q_u16(values[0], values[1]));
    let hi = vreinterpretq_u32_u16(vzip2q_u16(values[0], values[1]));
    let pairs = [
        vget_low_u32(lo),
        vget_high_u32(lo),
        vget_low_u32(hi),
        vget_high_u32(hi),
    ];
    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for (pair, rows) in pairs.iter().enumerate() {
        unsafe {
            vst1_lane_u32::<0>(
                dst.offset((2 * pair) as isize * stride).cast::<u32>(),
                *rows,
            );
            vst1_lane_u32::<1>(
                dst.offset((2 * pair + 1) as isize * stride).cast::<u32>(),
                *rows,
            );
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_vertical_transposed<const N: usize>(
    dst: &mut [u16],
    base: usize,
    stride: isize,
    offset: isize,
    values: &[uint16x8_t],
) {
    let zero = vdupq_n_u16(0);
    let mut rows = [zero; 8];
    rows[..N].copy_from_slice(&values[..N]);
    transpose_8x8_u16(&mut rows);

    let dst = unsafe { dst.as_mut_ptr().add((base as isize + offset) as usize) };
    for (row, value) in rows.iter().enumerate() {
        unsafe {
            if N == 4 {
                vst1_u16(dst.offset(row as isize * stride), vget_low_u16(*value));
            } else {
                vst1q_u16(dst.offset(row as isize * stride), *value);
            }
        }
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn store_sample_batch(
    dst: &mut [u16],
    base: usize,
    stride_a: isize,
    stride_b: isize,
    offset: isize,
    value: uint16x8_t,
) {
    if stride_a == 1 {
        let pos = (base as isize + offset * stride_b) as usize;
        unsafe { vst1q_u16(dst.as_mut_ptr().add(pos), value) };
    } else {
        let pos = base as isize + offset * stride_b;
        dst[pos as usize] = vgetq_lane_u16::<0>(value);
        dst[(pos + stride_a) as usize] = vgetq_lane_u16::<1>(value);
        dst[(pos + 2 * stride_a) as usize] = vgetq_lane_u16::<2>(value);
        dst[(pos + 3 * stride_a) as usize] = vgetq_lane_u16::<3>(value);
        dst[(pos + 4 * stride_a) as usize] = vgetq_lane_u16::<4>(value);
        dst[(pos + 5 * stride_a) as usize] = vgetq_lane_u16::<5>(value);
        dst[(pos + 6 * stride_a) as usize] = vgetq_lane_u16::<6>(value);
        dst[(pos + 7 * stride_a) as usize] = vgetq_lane_u16::<7>(value);
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn select_u16(mask: uint16x8_t, yes: uint16x8_t, no: uint16x8_t) -> uint16x8_t {
    vbslq_u16(mask, yes, no)
}

#[inline]
#[target_feature(enable = "neon")]
fn clip_s16(value: int16x8_t, lo: int16x8_t, hi: int16x8_t) -> int16x8_t {
    vmaxq_s16(lo, vminq_s16(value, hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn weighted_u16(samples: &[uint16x8_t], weights: &[i32], bias: u16, shift: i32) -> uint16x8_t {
    let mut sum = vdupq_n_u16(bias);
    for (&sample, &weight) in samples.iter().zip(weights) {
        if weight != 0 {
            sum = vaddq_u16(sum, vmulq_u16(sample, vdupq_n_u16(weight as u16)));
        }
    }
    match shift {
        3 => vshrq_n_u16::<3>(sum),
        4 => vshrq_n_u16::<4>(sum),
        _ => unreachable!("unsupported loop-filter shift"),
    }
}

/// Eight-line loop filter using 16-bit lanes. AV1's 12-bit extrema fit:
/// edge metric <= 10237, signed filter accumulator <= 14333, and the widest
/// unsigned convolution including bias <= 65528.
#[target_feature(enable = "neon")]
pub(crate) fn loop_filter_batch_neon(
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
        let mut samples = [vdupq_n_u16(0); 14];
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

    let scale = 1u16 << (bd - 8);
    let e = vdupq_n_u16(e as u16 * scale);
    let i_lim = vdupq_n_u16(i_lim as u16 * scale);
    let h_thresh = vdupq_n_u16(h_thresh as u16 * scale);
    let flat_limit = vdupq_n_u16(scale);
    let p1 = s[5];
    let p0 = s[6];
    let q0 = s[7];
    let q1 = s[8];

    let mut fm = vandq_u16(
        vcleq_u16(vabdq_u16(p1, p0), i_lim),
        vcleq_u16(vabdq_u16(q1, q0), i_lim),
    );
    let edge_metric = vaddq_u16(
        vshlq_n_u16::<1>(vabdq_u16(p0, q0)),
        vshrq_n_u16::<1>(vabdq_u16(p1, q1)),
    );
    fm = vandq_u16(fm, vcleq_u16(edge_metric, e));
    if wd > 4 {
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(s[4], p1), i_lim));
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(s[9], q1), i_lim));
    }
    if wd > 6 {
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(s[3], s[4]), i_lim));
        fm = vandq_u16(fm, vcleq_u16(vabdq_u16(s[10], s[9]), i_lim));
    }
    if vmaxvq_u16(fm) == 0 {
        return;
    }

    let mut flat_in = vdupq_n_u16(0);
    if wd >= 6 {
        flat_in = vcleq_u16(vabdq_u16(s[4], p0), flat_limit);
        flat_in = vandq_u16(flat_in, vcleq_u16(vabdq_u16(p1, p0), flat_limit));
        flat_in = vandq_u16(flat_in, vcleq_u16(vabdq_u16(q1, q0), flat_limit));
        flat_in = vandq_u16(flat_in, vcleq_u16(vabdq_u16(s[9], q0), flat_limit));
    }
    if wd >= 8 {
        flat_in = vandq_u16(flat_in, vcleq_u16(vabdq_u16(s[3], p0), flat_limit));
        flat_in = vandq_u16(flat_in, vcleq_u16(vabdq_u16(s[10], q0), flat_limit));
    }

    let mut flat_out = vdupq_n_u16(0);
    if wd >= 16 {
        flat_out = vcleq_u16(vabdq_u16(s[0], p0), flat_limit);
        flat_out = vandq_u16(flat_out, vcleq_u16(vabdq_u16(s[1], p0), flat_limit));
        flat_out = vandq_u16(flat_out, vcleq_u16(vabdq_u16(s[2], p0), flat_limit));
        flat_out = vandq_u16(flat_out, vcleq_u16(vabdq_u16(s[11], q0), flat_limit));
        flat_out = vandq_u16(flat_out, vcleq_u16(vabdq_u16(s[12], q0), flat_limit));
        flat_out = vandq_u16(flat_out, vcleq_u16(vabdq_u16(s[13], q0), flat_limit));
    }

    let wide16 = if wd >= 16 {
        vandq_u16(fm, vandq_u16(flat_in, flat_out))
    } else {
        vdupq_n_u16(0)
    };
    let wide8 = if wd >= 8 {
        vandq_u16(vandq_u16(fm, flat_in), vmvnq_u16(wide16))
    } else {
        vdupq_n_u16(0)
    };
    let wide6 = if wd == 6 {
        vandq_u16(fm, flat_in)
    } else {
        vdupq_n_u16(0)
    };
    let wide = vorrq_u16(vorrq_u16(wide16, wide8), wide6);
    let short = vandq_u16(fm, vmvnq_u16(wide));

    let mut out = s;
    if vmaxvq_u16(short) != 0 {
        let hev = vorrq_u16(
            vcgtq_u16(vabdq_u16(p1, p0), h_thresh),
            vcgtq_u16(vabdq_u16(q1, q0), h_thresh),
        );
        let p1s = vreinterpretq_s16_u16(p1);
        let p0s = vreinterpretq_s16_u16(p0);
        let q0s = vreinterpretq_s16_u16(q0);
        let q1s = vreinterpretq_s16_u16(q1);
        let zero = vdupq_n_s16(0);
        let clip_lo = vdupq_n_s16(-(128i16 * scale as i16));
        let clip_hi = vdupq_n_s16(128i16 * scale as i16 - 1);
        let pixel_hi = vdupq_n_s16((1i16 << bd) - 1);
        let delta = vsubq_s16(q0s, p0s);
        let triple = vaddq_s16(delta, vshlq_n_s16::<1>(delta));
        let fv_plain = clip_s16(triple, clip_lo, clip_hi);
        let fv_hev = clip_s16(
            vaddq_s16(triple, clip_s16(vsubq_s16(p1s, q1s), clip_lo, clip_hi)),
            clip_lo,
            clip_hi,
        );
        let fv = vbslq_s16(hev, fv_hev, fv_plain);
        let f1 = vshrq_n_s16::<3>(vminq_s16(vaddq_s16(fv, vdupq_n_s16(4)), clip_hi));
        let f2 = vshrq_n_s16::<3>(vminq_s16(vaddq_s16(fv, vdupq_n_s16(3)), clip_hi));

        out[6] = select_u16(
            short,
            vreinterpretq_u16_s16(clip_s16(vaddq_s16(p0s, f2), zero, pixel_hi)),
            out[6],
        );
        out[7] = select_u16(
            short,
            vreinterpretq_u16_s16(clip_s16(vsubq_s16(q0s, f1), zero, pixel_hi)),
            out[7],
        );
        let outer_short = vandq_u16(short, vmvnq_u16(hev));
        let f = vshrq_n_s16::<1>(vaddq_s16(f1, vdupq_n_s16(1)));
        out[5] = select_u16(
            outer_short,
            vreinterpretq_u16_s16(clip_s16(vaddq_s16(p1s, f), zero, pixel_hi)),
            out[5],
        );
        out[8] = select_u16(
            outer_short,
            vreinterpretq_u16_s16(clip_s16(vsubq_s16(q1s, f), zero, pixel_hi)),
            out[8],
        );
    }

    let has_wide6 = vmaxvq_u16(wide6) != 0;
    let has_wide8 = vmaxvq_u16(wide8) != 0;
    let has_wide16 = vmaxvq_u16(wide16) != 0;
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
