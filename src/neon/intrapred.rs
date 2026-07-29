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

use crate::intrapred::{DR_EDGE_ORIGIN, DR_ZONE1, DR_ZONE2, DR_ZONE3, DrPrediction, sm_weights};

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
fn load_i32x8(src: &[i32; 8]) -> (int32x4_t, int32x4_t) {
    let (src4, _) = src.as_chunks::<4>();
    (load_i32x4(&src4[0]), load_i32x4(&src4[1]))
}

#[inline]
#[target_feature(enable = "neon")]
fn load_i32x8_as_s16(src: &[i32; 8]) -> int16x8_t {
    let (lo, hi) = load_i32x8(src);
    vcombine_s16(vqmovn_s32(lo), vqmovn_s32(hi))
}

#[inline]
#[target_feature(enable = "neon")]
fn store_s16x8_as_i32(dst: &mut [i32; 8], value: int16x8_t) {
    let (dst4, _) = dst.as_chunks_mut::<4>();
    store_i32x4(&mut dst4[0], vmovl_s16(vget_low_s16(value)));
    store_i32x4(&mut dst4[1], vmovl_high_s16(value));
}

#[inline]
#[target_feature(enable = "neon")]
fn accumulate_dot_i32(acc: int64x2_t, lhs: int32x4_t, rhs: int32x4_t) -> int64x2_t {
    vaddq_s64(
        vaddq_s64(acc, vmull_s32(vget_low_s32(lhs), vget_low_s32(rhs))),
        vmull_high_s32(lhs, rhs),
    )
}

#[inline]
#[target_feature(enable = "neon")]
fn store_cfl_q3(dst: &mut [i32; 4], q3: uint32x4_t, sum: uint64x2_t) -> uint64x2_t {
    store_i32x4(dst, vreinterpretq_s32_u32(q3));
    vpadalq_u32(sum, q3)
}

#[inline]
#[target_feature(enable = "neon")]
fn subtract_cfl_mean(ac: &mut [i32], sum: u64, log2sz: u32) {
    let mean = ((sum + ((1u64 << log2sz) >> 1)) >> log2sz) as i32;
    let mean_v = vdupq_n_s32(mean);
    let (ac4, ac_tail) = ac.as_chunks_mut::<4>();
    for chunk in ac4 {
        store_i32x4(chunk, vsubq_s32(load_i32x4(chunk), mean_v));
    }
    for value in ac_tail {
        *value -= mean;
    }
}

#[inline]
#[target_feature(enable = "neon")]
fn cfl_predict_s16(
    ac: int16x8_t,
    dc: int16x8_t,
    max: int16x8_t,
    alpha_sign: int16x8_t,
    abs_alpha_q12: i16,
) -> int16x8_t {
    let magnitude = vqrdmulhq_n_s16(vabsq_s16(ac), abs_alpha_q12);
    let sign = vshrq_n_s16::<15>(veorq_s16(ac, alpha_sign));
    let signed = vsubq_s16(veorq_s16(magnitude, sign), sign);
    vminq_s16(vmaxq_s16(vaddq_s16(dc, signed), vdupq_n_s16(0)), max)
}

/// De-interleaved load of eight consecutive samples into (evens, odds).
#[inline]
#[target_feature(enable = "neon")]
fn load2_i32x4(src: &[i32; 8]) -> (int32x4_t, int32x4_t) {
    let p = unsafe { vld2q_s32(src.as_ptr()) };
    (p.0, p.1)
}

#[inline]
#[target_feature(enable = "neon")]
fn mla_n(acc: int32x4_t, v: int32x4_t, k: i32) -> int32x4_t {
    vmlaq_s32(acc, v, vdupq_n_s32(k))
}

/// Chunked views of `src` shifted by `0..N` samples: `views[j][i]` is the
/// 4-sample window at `src[4 * i + j]`. Lets the sliding-window kernels index
/// whole vectors instead of re-slicing (and re-checking) per tap.
#[inline]
fn shifted_chunks<const N: usize>(src: &[i32]) -> [&[[i32; 4]]; N] {
    std::array::from_fn(|j| src[j..].as_chunks::<4>().0)
}

#[target_feature(enable = "neon")]
pub(crate) fn dc_pred_neon(
    recon: &[u16],
    stride: usize,
    ox: usize,
    oy: usize,
    width: usize,
    height: usize,
    bit_depth: i32,
) -> i32 {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let mut edges = [0u16; 64];
    let mut len = 0;
    if have_top {
        edges[..width].copy_from_slice(&recon[(oy - 1) * stride + ox..][..width]);
        len = width;
    }
    if have_left {
        for y in 0..height {
            edges[len + y] = recon[(oy + y) * stride + ox - 1];
        }
        len += height;
    }
    let (chunks, tail) = edges[..len].as_chunks::<8>();
    let mut sum = 0u32;
    for chunk in chunks {
        sum += unsafe { vaddlvq_u16(vld1q_u16(chunk.as_ptr())) };
    }
    sum += tail.iter().map(|&v| u32::from(v)).sum::<u32>();
    crate::intrapred::dc_pred_from_sum(sum as i32, width, height, have_top, have_left, bit_depth)
}

#[target_feature(enable = "neon")]
pub(crate) fn smooth_neon(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let (wv, wh) = (sm_weights(bh), sm_weights(bw));
    let (right, bottom) = (top[bw - 1], left[bh - 1]);
    let (top4, _) = top.as_chunks::<4>();
    let (wh4, _) = wh.as_chunks::<4>();
    let c256 = vdupq_n_s32(256);
    let rnd = vdupq_n_s32(256);
    for ((row, &wvy), &lv) in out
        .chunks_exact_mut(bw)
        .zip(wv.iter())
        .zip(left.iter())
        .take(bh)
    {
        let base = vdupq_n_s32((256 - wvy) * bottom);
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        for ((o, t), w) in out4.iter_mut().zip(top4).zip(wh4) {
            let tv = load_i32x4(t);
            let whx = load_i32x4(w);
            let w2 = vsubq_s32(c256, whx);
            let mut acc = mla_n(base, tv, wvy); // base + top*wvy
            acc = mla_n(acc, whx, lv); // + wh*left[y]
            acc = mla_n(acc, w2, right); // + (256-wh)*right
            store_i32x4(o, vshrq_n_s32::<9>(vaddq_s32(acc, rnd)));
        }
        let done = out4.len() * 4;
        for (x, o) in out_tail.iter_mut().enumerate() {
            let x = done + x;
            let pred = wvy * top[x] + (256 - wvy) * bottom + wh[x] * lv + (256 - wh[x]) * right;
            *o = (pred + 256) >> 9;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn smooth_v_neon(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wv = sm_weights(bh);
    let bottom = left[bh - 1];
    let (top4, _) = top.as_chunks::<4>();
    let rnd = vdupq_n_s32(128);
    for (row, &wvy) in out.chunks_exact_mut(bw).zip(wv.iter()).take(bh) {
        let base = vdupq_n_s32((256 - wvy) * bottom);
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        for (o, t) in out4.iter_mut().zip(top4) {
            let acc = mla_n(base, load_i32x4(t), wvy);
            store_i32x4(o, vshrq_n_s32::<8>(vaddq_s32(acc, rnd)));
        }
        let done = out4.len() * 4;
        for (x, o) in out_tail.iter_mut().enumerate() {
            *o = (wvy * top[done + x] + (256 - wvy) * bottom + 128) >> 8;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn smooth_h_neon(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    let wh = sm_weights(bw);
    let right = top[bw - 1];
    let (wh4, _) = wh.as_chunks::<4>();
    let c256 = vdupq_n_s32(256);
    let rnd = vdupq_n_s32(128);
    let zero = vdupq_n_s32(0);
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        for (o, w) in out4.iter_mut().zip(wh4) {
            let whx = load_i32x4(w);
            let w2 = vsubq_s32(c256, whx);
            let mut acc = mla_n(zero, w2, right); // (256-wh)*right
            acc = mla_n(acc, whx, lv); // + wh*left[y]
            store_i32x4(o, vshrq_n_s32::<8>(vaddq_s32(acc, rnd)));
        }
        let done = out4.len() * 4;
        for (x, o) in out_tail.iter_mut().enumerate() {
            let whx = wh[done + x];
            *o = (whx * lv + (256 - whx) * right + 128) >> 8;
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn vertical_neon(bw: usize, bh: usize, top: &[i32], _left: &[i32], out: &mut [i32]) {
    let (top4, top_tail) = top[..bw].as_chunks::<4>();
    for row in out.chunks_exact_mut(bw).take(bh) {
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        for (o, t) in out4.iter_mut().zip(top4) {
            store_i32x4(o, load_i32x4(t));
        }
        out_tail.copy_from_slice(top_tail);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn horizontal_neon(bw: usize, bh: usize, _top: &[i32], left: &[i32], out: &mut [i32]) {
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        let v = vdupq_n_s32(lv);
        for o in out4 {
            store_i32x4(o, v);
        }
        out_tail.fill(lv);
    }
}

/// 5-tap edge-smoothing convolution over the clamp-free middle run:
/// `out[t] = (Σ_j k[j] * win[t + j] + 8) >> 4`. `win` must hold
/// `out.len() + 4` samples.
#[target_feature(enable = "neon")]
pub(crate) fn edge_conv5_neon(win: &[i32], k: &[i32; 5], out: &mut [i32]) {
    let rnd = vdupq_n_s32(8);
    let (out4, out_tail) = out.as_chunks_mut::<4>();
    let taps = shifted_chunks::<5>(win);
    for (i, o) in out4.iter_mut().enumerate() {
        let mut acc = rnd;
        for (t, &kj) in taps.iter().zip(k.iter()) {
            acc = mla_n(acc, load_i32x4(&t[i]), kj);
        }
        store_i32x4(o, vshrq_n_s32::<4>(acc));
    }
    let done = out4.len() * 4;
    for (o, t) in out_tail.iter_mut().zip(done..) {
        let sum: i32 = k.iter().enumerate().map(|(j, &kj)| kj * win[t + j]).sum();
        *o = (sum + 8) >> 4;
    }
}

/// The 4x2-cell recursive filter-intra pass over `buf` (33x33, row 0 and
/// column 0 hold the references). Bit-exact with the scalar tap loop: the
/// eight outputs of a cell are two i32x4 lanes accumulated from seven
/// broadcast samples times transposed tap vectors.
#[target_feature(enable = "neon")]
pub(crate) fn filter_intra_cells_neon(
    buf: &mut [[i32; 33]; 33],
    taps: &[[i8; 7]; 8],
    width: usize,
    height: usize,
    max_sample: i32,
) {
    // Transpose taps into per-input vectors, split by output row: outputs
    // 0..4 land in row r, outputs 4..8 in row r + 1.
    let (mut tlo, mut thi) = ([[0i32; 4]; 7], [[0i32; 4]; 7]);
    for (k, filter) in taps.iter().enumerate() {
        for (j, &tap) in filter.iter().enumerate() {
            if k < 4 {
                tlo[j][k] = tap as i32;
            } else {
                thi[j][k - 4] = tap as i32;
            }
        }
    }
    let tv_lo: [int32x4_t; 7] = std::array::from_fn(|j| load_i32x4(&tlo[j]));
    let tv_hi: [int32x4_t; 7] = std::array::from_fn(|j| load_i32x4(&thi[j]));
    let rnd = vdupq_n_s32(8);
    let zero = vdupq_n_s32(0);
    let maxv = vdupq_n_s32(max_sample);
    let shift = vdupq_n_s32(-(crate::tables::INTRA_FILTER_SCALE_BITS as i32));
    for r in (1..=height).step_by(2) {
        let (above, rest) = buf.split_at_mut(r);
        let (row_lo, row_hi) = rest.split_at_mut(1);
        let p_above = &above[r - 1];
        // A cell's left reference is the last column the previous cell wrote
        // (the recursion); carry it forward instead of re-reading the row,
        // so the row stays exclusively borrowed as 4-wide cells.
        let (mut left_lo, mut left_hi) = (row_lo[0][0], row_hi[0][0]);
        // Cells cover columns 1 + 4i, so the row prefix `[1..]` chunks into
        // exactly those cells.
        let (cells_lo, _) = row_lo[0][1..].as_chunks_mut::<4>();
        let (cells_hi, _) = row_hi[0][1..].as_chunks_mut::<4>();
        for (i, (clo, chi)) in cells_lo
            .iter_mut()
            .zip(cells_hi.iter_mut())
            .take(width / 4)
            .enumerate()
        {
            let c = 1 + 4 * i;
            let p = [
                p_above[c - 1],
                p_above[c],
                p_above[c + 1],
                p_above[c + 2],
                p_above[c + 3],
                left_lo,
                left_hi,
            ];
            let mut lo = rnd;
            let mut hi = rnd;
            for ((&pj, &tl), &th) in p.iter().zip(tv_lo.iter()).zip(tv_hi.iter()) {
                let pv = vdupq_n_s32(pj);
                lo = vmlaq_s32(lo, tl, pv);
                hi = vmlaq_s32(hi, th, pv);
            }
            let lo = vminq_s32(vmaxq_s32(vshlq_s32(lo, shift), zero), maxv);
            let hi = vminq_s32(vmaxq_s32(vshlq_s32(hi, shift), zero), maxv);
            store_i32x4(clo, lo);
            store_i32x4(chi, hi);
            (left_lo, left_hi) = (clo[3], chi[3]);
        }
    }
}

/// Interpolate one contiguous output run. `e` starts at its first base sample
/// and must hold `(out.len() - 1) * step + 2` samples.
#[target_feature(enable = "neon")]
fn dr_interp_neon(e: &[i32], step: usize, shift: i32, out: &mut [i32]) {
    let shv = vdupq_n_s32(shift);
    let ishv = vdupq_n_s32(32 - shift);
    let rnd = vdupq_n_s32(16);
    let (out4, out_tail) = out.as_chunks_mut::<4>();
    if step == 1 {
        // Consecutive samples: taps at offsets 0 and 1.
        let taps = shifted_chunks::<2>(e);
        for (i, o) in out4.iter_mut().enumerate() {
            let a = load_i32x4(&taps[0][i]);
            let b = load_i32x4(&taps[1][i]);
            let v = vmlaq_s32(vmlaq_s32(rnd, a, ishv), b, shv);
            store_i32x4(o, vshrq_n_s32::<5>(v));
        }
    } else {
        // Upsampled edge: four outputs consume eight samples, de-interleaved.
        let (e8, _) = e.as_chunks::<8>();
        for (o, src) in out4.iter_mut().zip(e8) {
            let (a, b) = load2_i32x4(src);
            let v = vmlaq_s32(vmlaq_s32(rnd, a, ishv), b, shv);
            store_i32x4(o, vshrq_n_s32::<5>(v));
        }
    }
    let done = out4.len() * 4;
    for (o, j) in out_tail.iter_mut().zip(done..) {
        let b = j * step;
        *o = (e[b] * (32 - shift) + e[b + 1] * shift + 16) >> 5;
    }
}

#[inline]
fn dr_edge(edge: &[i32], index: i32) -> i32 {
    edge[(index + DR_EDGE_ORIGIN) as usize]
}

#[inline]
#[target_feature(enable = "neon")]
fn dr_interp_column4(
    edge: &[i32],
    base: i32,
    step: usize,
    shift: i32,
    max_base: i32,
    fill: i32,
) -> int32x4_t {
    let last_base = base + 3 * step as i32;
    let (a, b) = if last_base < max_base {
        let start = (base + DR_EDGE_ORIGIN) as usize;
        if step == 1 {
            (
                load_i32x4(edge[start..].first_chunk::<4>().unwrap()),
                load_i32x4(edge[start + 1..].first_chunk::<4>().unwrap()),
            )
        } else {
            load2_i32x4(edge[start..].first_chunk::<8>().unwrap())
        }
    } else {
        let mut a = [fill; 4];
        let mut b = [fill; 4];
        for (lane, (a, b)) in a.iter_mut().zip(b.iter_mut()).enumerate() {
            let index = base + lane as i32 * step as i32;
            if index < max_base {
                *a = dr_edge(edge, index);
                *b = dr_edge(edge, index + 1);
            }
        }
        (load_i32x4(&a), load_i32x4(&b))
    };
    let shift = vdupq_n_s32(shift);
    let inverse = vsubq_s32(vdupq_n_s32(32), shift);
    let sum = vmlaq_s32(vmlaq_s32(vdupq_n_s32(16), a, inverse), b, shift);
    vshrq_n_s32::<5>(sum)
}

#[inline]
#[target_feature(enable = "neon")]
fn transpose_4x4(c0: int32x4_t, c1: int32x4_t, c2: int32x4_t, c3: int32x4_t) -> [int32x4_t; 4] {
    let a = vtrn1q_s32(c0, c1);
    let b = vtrn2q_s32(c0, c1);
    let c = vtrn1q_s32(c2, c3);
    let d = vtrn2q_s32(c2, c3);
    [
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
    ]
}

#[target_feature(enable = "neon")]
pub(crate) fn dr_predict_neon(p: DrPrediction, above: &[i32], left: &[i32], out: &mut [i32]) {
    debug_assert!(out.len() >= p.bw * p.bh);
    match p.zone {
        DR_ZONE1 => {
            let max_base = (p.edge_len as i32 - 1) << p.up_above;
            let frac_bits = 6 - p.up_above;
            let step = 1usize << p.up_above;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let xpos = p.dx * (y as i32 + 1);
                let shift = ((xpos << p.up_above) & 0x3f) >> 1;
                let base = xpos >> frac_bits;
                let n_fit = if base >= max_base {
                    0
                } else {
                    ((max_base - base + step as i32 - 1) / step as i32).min(p.bw as i32) as usize
                };
                dr_interp_neon(
                    &above[(base + DR_EDGE_ORIGIN) as usize..],
                    step,
                    shift,
                    &mut row[..n_fit],
                );
                row[n_fit..].fill(dr_edge(above, max_base));
            }
        }
        DR_ZONE3 => {
            let max_base = (p.edge_len as i32 - 1) << p.up_left;
            let step = 1usize << p.up_left;
            let fill = dr_edge(left, max_base);
            let frac_bits = 6 - p.up_left;
            debug_assert_eq!(p.bw & 3, 0);
            debug_assert_eq!(p.bh & 3, 0);
            for (y4, block) in out[..p.bw * p.bh].chunks_exact_mut(p.bw * 4).enumerate() {
                let (row0, block) = block.split_at_mut(p.bw);
                let (row1, block) = block.split_at_mut(p.bw);
                let (row2, row3) = block.split_at_mut(p.bw);
                let (row0, _) = row0.as_chunks_mut::<4>();
                let (row1, _) = row1.as_chunks_mut::<4>();
                let (row2, _) = row2.as_chunks_mut::<4>();
                let (row3, _) = row3.as_chunks_mut::<4>();
                for (x4, (((dst0, dst1), dst2), dst3)) in
                    row0.iter_mut().zip(row1).zip(row2).zip(row3).enumerate()
                {
                    let y = y4 * 4;
                    let x = x4 * 4;
                    let mut columns = [vdupq_n_s32(0); 4];
                    for (lane, column) in columns.iter_mut().enumerate() {
                        let ypos = p.dy * (x as i32 + lane as i32 + 1);
                        let base = (ypos >> frac_bits) + y as i32 * step as i32;
                        let shift = ((ypos << p.up_left) & 0x3f) >> 1;
                        *column = dr_interp_column4(left, base, step, shift, max_base, fill);
                    }
                    let [v0, v1, v2, v3] =
                        transpose_4x4(columns[0], columns[1], columns[2], columns[3]);
                    store_i32x4(dst0, v0);
                    store_i32x4(dst1, v1);
                    store_i32x4(dst2, v2);
                    store_i32x4(dst3, v3);
                }
            }
        }
        DR_ZONE2 => {
            let frac_bits_y = 6 - p.up_left;
            for (y, row) in out.chunks_exact_mut(p.bw).take(p.bh).enumerate() {
                let t = (y as i32 + 1) * p.dx - 64;
                let x0 = if t <= 0 {
                    0
                } else {
                    (((t + 63) >> 6) as usize).min(p.bw)
                };
                for (x, dst) in row[..x0].iter_mut().enumerate() {
                    let ypos = ((y as i32) << 6) - (x as i32 + 1) * p.dy;
                    let base = ypos >> frac_bits_y;
                    let shift = ((ypos * (1 << p.up_left)) & 0x3f) >> 1;
                    *dst =
                        (dr_edge(left, base) * (32 - shift) + dr_edge(left, base + 1) * shift + 16)
                            >> 5;
                }
                if x0 < p.bw {
                    let xpos = ((x0 as i32) << 6) - (y as i32 + 1) * p.dx;
                    let base = xpos >> (6 - p.up_above);
                    let shift = ((xpos * (1 << p.up_above)) & 0x3f) >> 1;
                    dr_interp_neon(
                        &above[(base + DR_EDGE_ORIGIN) as usize..],
                        1usize << p.up_above,
                        shift,
                        &mut row[x0..],
                    );
                }
            }
        }
        _ => unreachable!("invalid directional prediction zone"),
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn paeth_neon(
    bw: usize,
    bh: usize,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [i32],
) {
    let cn = vdupq_n_s32(corner);
    let (top4, _) = top.as_chunks::<4>();
    for (row, &lv) in out.chunks_exact_mut(bw).zip(left.iter()).take(bh) {
        let lvv = vdupq_n_s32(lv);
        let lmc = vdupq_n_s32(lv - corner);
        let (out4, out_tail) = row.as_chunks_mut::<4>();
        for (o, t) in out4.iter_mut().zip(top4) {
            let tv = load_i32x4(t);
            let b = vaddq_s32(tv, lmc); // lv + tv - corner
            let ld = vabdq_s32(lvv, b);
            let td = vabdq_s32(tv, b);
            let cd = vabdq_s32(cn, b);
            let m_tv = vcleq_s32(td, cd); // td <= cd
            let m_lv = vandq_u32(vcleq_s32(ld, td), vcleq_s32(ld, cd)); // ld<=td && ld<=cd
            let mut res = vbslq_s32(m_tv, tv, cn); // td<=cd ? tv : corner
            res = vbslq_s32(m_lv, lvv, res); // ld<=...? lv : res
            store_i32x4(o, res);
        }
        let done = out4.len() * 4;
        for (x, o) in out_tail.iter_mut().enumerate() {
            let tv = top[done + x];
            let b = lv + tv - corner;
            let (ld, td, cd) = ((lv - b).abs(), (tv - b).abs(), (corner - b).abs());
            *o = if ld <= td && ld <= cd {
                lv
            } else if td <= cd {
                tv
            } else {
                corner
            };
        }
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_ac_444_u16_neon(luma_rec: &[u16], w: usize, h: usize, ac: &mut [i32]) {
    let n = w * h;
    debug_assert!(luma_rec.len() >= n);
    debug_assert!(ac.len() >= n);

    let (src8, src_tail8) = luma_rec[..n].as_chunks::<8>();
    let (dst8, dst_tail8) = ac[..n].as_chunks_mut::<8>();
    let mut sum_v = vdupq_n_u64(0);
    for (src, dst) in src8.iter().zip(dst8) {
        let pixels = unsafe { vld1q_u16(src.as_ptr()) };
        let (dst4, _) = dst.as_chunks_mut::<4>();
        sum_v = store_cfl_q3(
            &mut dst4[0],
            vshlq_n_u32::<3>(vmovl_u16(vget_low_u16(pixels))),
            sum_v,
        );
        sum_v = store_cfl_q3(
            &mut dst4[1],
            vshlq_n_u32::<3>(vmovl_high_u16(pixels)),
            sum_v,
        );
    }

    let (src4, src_tail) = src_tail8.as_chunks::<4>();
    let (dst4, dst_tail) = dst_tail8.as_chunks_mut::<4>();
    for (src, dst) in src4.iter().zip(dst4) {
        let pixels = unsafe { vld1_u16(src.as_ptr()) };
        sum_v = store_cfl_q3(dst, vshlq_n_u32::<3>(vmovl_u16(pixels)), sum_v);
    }
    let mut sum = vaddvq_u64(sum_v);
    for (&src, dst) in src_tail.iter().zip(dst_tail) {
        *dst = i32::from(src) << 3;
        sum += *dst as u64;
    }

    subtract_cfl_mean(&mut ac[..n], sum, w.trailing_zeros() + h.trailing_zeros());
}

#[allow(clippy::too_many_arguments)]
#[target_feature(enable = "neon")]
pub(crate) fn cfl_ac_sub_u16_neon(
    luma_rec: &[u16],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    let n = cw * ch;
    let sx = usize::from(ss_hor);
    let sy = usize::from(ss_ver);
    let lw = cw << sx;
    let lh = ch << sy;
    debug_assert!(cw > 0 && ch > 0);
    debug_assert!(lstride >= lw);
    debug_assert!(luma_rec.len() >= (lh - 1) * lstride + lw);
    debug_assert!(ac.len() >= n);

    let shift = 1 + u32::from(!ss_ver) + u32::from(!ss_hor);
    let mut sum_v = vdupq_n_u64(0);
    let mut scalar_sum = 0u64;
    for y in 0..ch {
        let ly = y << sy;
        let top = &luma_rec[ly * lstride..ly * lstride + lw];
        let out = &mut ac[y * cw..(y + 1) * cw];
        let (out4, out_tail) = out.as_chunks_mut::<4>();

        match (ss_hor, ss_ver) {
            (true, true) => {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                let (top8, _) = top.as_chunks::<8>();
                let (bottom8, _) = bottom.as_chunks::<8>();
                for ((dst, top), bottom) in out4.iter_mut().zip(top8).zip(bottom8) {
                    let top_pairs = vpaddlq_u16(unsafe { vld1q_u16(top.as_ptr()) });
                    let bottom_pairs = vpaddlq_u16(unsafe { vld1q_u16(bottom.as_ptr()) });
                    sum_v = store_cfl_q3(
                        dst,
                        vshlq_n_u32::<1>(vaddq_u32(top_pairs, bottom_pairs)),
                        sum_v,
                    );
                }
            }
            (true, false) => {
                let (top8, _) = top.as_chunks::<8>();
                for (dst, top) in out4.iter_mut().zip(top8) {
                    let pairs = vpaddlq_u16(unsafe { vld1q_u16(top.as_ptr()) });
                    sum_v = store_cfl_q3(dst, vshlq_n_u32::<2>(pairs), sum_v);
                }
            }
            (false, true) => {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                let (top4, _) = top.as_chunks::<4>();
                let (bottom4, _) = bottom.as_chunks::<4>();
                for ((dst, top), bottom) in out4.iter_mut().zip(top4).zip(bottom4) {
                    let top_pixels = vmovl_u16(unsafe { vld1_u16(top.as_ptr()) });
                    let bottom_pixels = vmovl_u16(unsafe { vld1_u16(bottom.as_ptr()) });
                    sum_v = store_cfl_q3(
                        dst,
                        vshlq_n_u32::<2>(vaddq_u32(top_pixels, bottom_pixels)),
                        sum_v,
                    );
                }
            }
            (false, false) => {
                let (top4, _) = top.as_chunks::<4>();
                for (dst, top) in out4.iter_mut().zip(top4) {
                    let pixels = vmovl_u16(unsafe { vld1_u16(top.as_ptr()) });
                    sum_v = store_cfl_q3(dst, vshlq_n_u32::<3>(pixels), sum_v);
                }
            }
        }

        let x0 = out4.len() * 4;
        for (tail_x, dst) in out_tail.iter_mut().enumerate() {
            let x = x0 + tail_x;
            let lx = x << sx;
            let mut value = i32::from(top[lx]);
            if ss_hor {
                value += i32::from(top[lx + 1]);
            }
            if ss_ver {
                let bottom = &luma_rec[(ly + 1) * lstride..(ly + 1) * lstride + lw];
                value += i32::from(bottom[lx]);
                if ss_hor {
                    value += i32::from(bottom[lx + 1]);
                }
            }
            *dst = value << shift;
            scalar_sum += *dst as u64;
        }
    }

    subtract_cfl_mean(
        &mut ac[..n],
        vaddvq_u64(sum_v) + scalar_sum,
        cw.trailing_zeros() + ch.trailing_zeros(),
    );
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_pred_neon(dst: &mut [i32], ac: &[i32], dc: i32, alpha: i32, bd: u8) {
    debug_assert!(ac.len() >= dst.len());
    debug_assert!((-16..=16).contains(&alpha));
    debug_assert!((8..=12).contains(&bd));
    debug_assert!((0..=(1 << bd) - 1).contains(&dc));
    debug_assert!(
        ac[..dst.len()]
            .iter()
            .all(|&value| i16::try_from(value).is_ok())
    );

    // Every CfL subsampling mode normalizes reconstructed luma to Q3. At
    // 12-bit that is 0..32760 before subtracting a mean from the same range,
    // hence AC remains in -32760..32760. With |alpha| <= 16 the rounded
    // contribution is at most 8190, and adding a 12-bit DC stays within
    // -8190..12285: signed 16-bit is sufficient before the final clamp.
    let len = dst.len();
    let (dst8, dst_tail) = dst.as_chunks_mut::<8>();
    let (ac8, ac_tail) = ac[..len].as_chunks::<8>();
    let dc_v = vdupq_n_s16(dc as i16);
    let max_v = vdupq_n_s16(((1 << bd) - 1) as i16);
    let alpha_sign = vdupq_n_s16(alpha as i16);
    let abs_alpha_q12 = (alpha.abs() << 9) as i16;
    for (out, input) in dst8.iter_mut().zip(ac8) {
        let ac_v = load_i32x8_as_s16(input);
        store_s16x8_as_i32(
            out,
            cfl_predict_s16(ac_v, dc_v, max_v, alpha_sign, abs_alpha_q12),
        );
    }
    for (out, &input) in dst_tail.iter_mut().zip(ac_tail) {
        *out = crate::intrapred::cfl_pred_pixel(dc, input, alpha, bd);
    }
}

#[target_feature(enable = "neon")]
pub(crate) fn cfl_best_alpha_u16_neon(ac: &[i32], src: &[u16], dc: i32, n: usize, bd: u8) -> i32 {
    debug_assert!(ac.len() >= n);
    debug_assert!(src.len() >= n);
    debug_assert!((8..=12).contains(&bd));
    let max = (1 << bd) - 1;
    debug_assert!((0..=max).contains(&dc));
    debug_assert!(ac[..n].iter().all(|&value| i16::try_from(value).is_ok()));
    debug_assert!(src[..n].iter().all(|&value| i32::from(value) <= max));

    let (ac8, ac_tail) = ac[..n].as_chunks::<8>();
    let (src8, src_tail) = src[..n].as_chunks::<8>();
    let dc_i32 = vdupq_n_s32(dc);
    let mut numerator_v = vdupq_n_s64(0);
    let mut denominator_v = vdupq_n_s64(0);
    for (ac_chunk, src_chunk) in ac8.iter().zip(src8) {
        let (ac_lo, ac_hi) = load_i32x8(ac_chunk);
        let src_v = unsafe { vld1q_u16(src_chunk.as_ptr()) };
        let src_lo = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(src_v)));
        let src_hi = vreinterpretq_s32_u32(vmovl_high_u16(src_v));
        numerator_v = accumulate_dot_i32(numerator_v, vsubq_s32(src_lo, dc_i32), ac_lo);
        numerator_v = accumulate_dot_i32(numerator_v, vsubq_s32(src_hi, dc_i32), ac_hi);
        denominator_v = accumulate_dot_i32(denominator_v, ac_lo, ac_lo);
        denominator_v = accumulate_dot_i32(denominator_v, ac_hi, ac_hi);
    }
    let mut numerator = vaddvq_s64(numerator_v);
    let mut denominator = vaddvq_s64(denominator_v);
    for (&ac, &src) in ac_tail.iter().zip(src_tail) {
        numerator += i64::from(i32::from(src) - dc) * i64::from(ac);
        denominator += i64::from(ac) * i64::from(ac);
    }
    if denominator == 0 {
        return 0;
    }

    let a0 = ((64 * numerator + (denominator >> 1) * numerator.signum()) / denominator)
        .clamp(-16, 16) as i32;
    let dc_v = vdupq_n_s16(dc as i16);
    let max_v = vdupq_n_s16(max as i16);
    let mut best_alpha = 0;
    let mut best_error = i64::MAX;
    for alpha in (a0 - 3)..=(a0 + 3) {
        if !(-16..=16).contains(&alpha) {
            continue;
        }
        let alpha_sign = vdupq_n_s16(alpha as i16);
        let abs_alpha_q12 = (alpha.abs() << 9) as i16;
        let mut error_v = vdupq_n_u64(0);
        for (ac_chunk, src_chunk) in ac8.iter().zip(src8) {
            let pred = cfl_predict_s16(
                load_i32x8_as_s16(ac_chunk),
                dc_v,
                max_v,
                alpha_sign,
                abs_alpha_q12,
            );
            let src_v = vreinterpretq_s16_u16(unsafe { vld1q_u16(src_chunk.as_ptr()) });
            let residual = vsubq_s16(src_v, pred);
            error_v = vpadalq_u32(
                error_v,
                vreinterpretq_u32_s32(vmull_s16(vget_low_s16(residual), vget_low_s16(residual))),
            );
            error_v = vpadalq_u32(
                error_v,
                vreinterpretq_u32_s32(vmull_high_s16(residual, residual)),
            );
        }
        let mut error = vaddvq_u64(error_v) as i64;
        for (&ac, &src) in ac_tail.iter().zip(src_tail) {
            let residual =
                i64::from(i32::from(src) - crate::intrapred::cfl_pred_pixel(dc, ac, alpha, bd));
            error += residual * residual;
        }
        if error < best_error {
            best_error = error;
            best_alpha = alpha;
        }
    }
    best_alpha
}
