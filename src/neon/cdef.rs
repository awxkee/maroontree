/*
 * Copyright (c) Radzivon Bartoshyk 10/2026. All rights reserved.
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
//! NEON port of the CDEF encoder-search kernel: every signaled strength pair
//! filtered at once for one interior 8x8 block (see
//! `cdef::cdef_block_candidates`, the scalar reference it must match exactly).
//!
//! All quantities fit 16-bit lanes: pixels and `CDEF_VERY_LARGE` (0x4000) are
//! below 2^15, a constrained tap is bounded by its strength (<= 64 at 12-bit),
//! and the weighted tap sums stay below 2^11. The direction and sum kernels
//! additionally require REAL pixels (`< 1 << bd`): their projections and
//! per-lane products are sized for that, not for `CDEF_VERY_LARGE`.
#![allow(clippy::too_many_arguments)]

use crate::cdef::{
    CAND_PRI, CAND_SEC, CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS, CDEF_VERY_LARGE, N_CAND,
    adjust_pri, log2_floor,
};
use core::arch::aarch64::*;

/// `constrain(d, strength, shift)` on eight lanes: `sign(d) * clamp(strength -
/// (|d| >> shift), 0, |d|)`. `nsh` holds `-shift` (a negative `vshl` count is a
/// right shift).
#[inline]
#[target_feature(enable = "neon")]
unsafe fn constrain8(d: int16x8_t, strength: int16x8_t, nsh: int16x8_t) -> int16x8_t {
    let a = vabsq_s16(d);
    let x = vsubq_s16(strength, vshlq_s16(a, nsh));
    let cl = vminq_s16(vmaxq_s16(x, vdupq_n_s16(0)), a);
    let neg = vcltq_s16(d, vdupq_n_s16(0));
    vbslq_s16(neg, vnegq_s16(cl), cl)
}

/// Interior 8x8 [`crate::cdef::cdef_block_candidates`]: the caller guarantees
/// `x >= 2`, `y >= 2`, `x + 10 <= stride` and `y + 10 <= rows` so every tap is a
/// plain in-plane load.
#[target_feature(enable = "neon")]
pub(crate) fn cdef_block_candidates_8x8_neon(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    dir: usize,
    var: Option<i32>,
    damping: i32,
    bd: u8,
    out: &mut [[i32; 64]; N_CAND],
) {
    unsafe {
        debug_assert!(x >= 2 && y >= 2 && x + 10 <= stride && y + 10 <= src.len() / stride);
        let coeff_shift = (bd - 8) as i32;
        let maxv = (1i32 << bd) - 1;
        let damp = damping.max(1);
        let shift = |s: i32| {
            if s == 0 {
                0
            } else {
                (damp - log2_floor(s)).max(0)
            }
        };
        let mut pri_eff = [0i32; 3];
        for (pe, &p) in pri_eff.iter_mut().zip(&CAND_PRI[1..]) {
            *pe = match var {
                Some(v) => adjust_pri(p << coeff_shift, v),
                None => p << coeff_shift,
            };
        }
        let pri_w = pri_eff.map(|pe| CDEF_PRI_TAPS[((pe >> coeff_shift) & 1) as usize]);
        let sec_eff = [CAND_SEC[1] << coeff_shift, CAND_SEC[2] << coeff_shift];

        let zero = vdupq_n_s16(0);
        let vl = vdupq_n_s16(CDEF_VERY_LARGE as i16);
        let mxv = vdupq_n_s16(maxv as i16);
        let eight = vdupq_n_s16(8);
        let pri_s = pri_eff.map(|s| vdupq_n_s16(s as i16));
        let pri_nsh = pri_eff.map(|s| vdupq_n_s16(-(shift(s) as i16)));
        let sec_s = sec_eff.map(|s| vdupq_n_s16(s as i16));
        let sec_nsh = sec_eff.map(|s| vdupq_n_s16(-(shift(s) as i16)));

        let base = src.as_ptr();
        let s = stride as isize;
        for i in 0..8 {
            let row = ((y + i) * stride + x) as isize;
            let load = |dr: i32, dc: i32| -> int16x8_t {
                let p = base.offset(row + dr as isize * s + dc as isize);
                vreinterpretq_s16_u16(vld1q_u16(p))
            };
            let c = load(0, 0);
            let mut psum = [zero; 3];
            let mut ssum = [zero; 2]; // secondary taps along `dir`
            let mut ssum0 = [zero; 2]; // secondary taps along direction 0
            let mut mn = c;
            let mut mx = c;
            for k in 0..2usize {
                let (pdr, pdc) = CDEF_DIRECTIONS[dir][k];
                for sgn in [1i32, -1] {
                    let t = load(sgn * pdr, sgn * pdc);
                    let valid = vmvnq_u16(vceqq_s16(t, vl));
                    let vmask = vreinterpretq_s16_u16(valid);
                    let d = vsubq_s16(t, c);
                    for q in 0..3 {
                        if pri_eff[q] != 0 {
                            let con = vandq_s16(constrain8(d, pri_s[q], pri_nsh[q]), vmask);
                            psum[q] = vmlaq_n_s16(psum[q], con, pri_w[q][k] as i16);
                        }
                    }
                    mn = vminq_s16(mn, vbslq_s16(valid, t, mn));
                    mx = vmaxq_s16(mx, vbslq_s16(valid, t, mx));
                }
                for doff in [2usize, 6] {
                    let (sdr, sdc) = CDEF_DIRECTIONS[(dir + doff) & 7][k];
                    for sgn in [1i32, -1] {
                        let t = load(sgn * sdr, sgn * sdc);
                        let valid = vmvnq_u16(vceqq_s16(t, vl));
                        let vmask = vreinterpretq_s16_u16(valid);
                        let d = vsubq_s16(t, c);
                        for q in 0..2 {
                            let con = vandq_s16(constrain8(d, sec_s[q], sec_nsh[q]), vmask);
                            ssum[q] = vmlaq_n_s16(ssum[q], con, CDEF_SEC_TAPS[k] as i16);
                        }
                        mn = vminq_s16(mn, vbslq_s16(valid, t, mn));
                        mx = vmaxq_s16(mx, vbslq_s16(valid, t, mx));
                    }
                    if dir != 0 {
                        let (sdr, sdc) = CDEF_DIRECTIONS[doff & 7][k];
                        for sgn in [1i32, -1] {
                            let t = load(sgn * sdr, sgn * sdc);
                            let vmask = vreinterpretq_s16_u16(vmvnq_u16(vceqq_s16(t, vl)));
                            let d = vsubq_s16(t, c);
                            for q in 0..2 {
                                let con = vandq_s16(constrain8(d, sec_s[q], sec_nsh[q]), vmask);
                                ssum0[q] = vmlaq_n_s16(ssum0[q], con, CDEF_SEC_TAPS[k] as i16);
                            }
                        }
                    }
                }
            }
            if dir == 0 {
                ssum0 = ssum;
            }
            let c_is_vl = vceqq_s16(c, vl);
            for si in 0..3usize {
                for pi in 0..4usize {
                    let en_p = pi != 0 && pri_eff[pi - 1] != 0;
                    let en_s = si != 0;
                    let sum = if pi == 0 {
                        if en_s { ssum0[si - 1] } else { zero }
                    } else {
                        let a = if en_p { psum[pi - 1] } else { zero };
                        if en_s { vaddq_s16(a, ssum[si - 1]) } else { a }
                    };
                    // centre + ((8 + sum - (sum < 0)) >> 4), arithmetic shift.
                    let adj = vaddq_s16(vaddq_s16(sum, vshrq_n_s16::<15>(sum)), eight);
                    let mut v = vaddq_s16(c, vshrq_n_s16::<4>(adj));
                    if en_p && en_s {
                        v = vminq_s16(vmaxq_s16(v, mn), mx);
                    }
                    v = vminq_s16(vmaxq_s16(v, zero), mxv);
                    v = vbslq_s16(c_is_vl, c, v);
                    let o = out[si * 4 + pi].as_mut_ptr().add(i * 8);
                    vst1q_s32(o, vmovl_s16(vget_low_s16(v)));
                    vst1q_s32(o.add(4), vmovl_s16(vget_high_s16(v)));
                }
            }
        }
    }
}

/// Partial-sum accumulation of `cdef::cdef_direction` for one interior 8x8
/// block (`x + 8 <= stride`, `y + 8 <= rows`): the eight line-projection
/// arrays of the spec's `cdef_find_dir`, built with unaligned 16-bit lane
/// adds instead of 512 scalar scatter-adds. The cost/argmax step is shared
/// with the scalar path (`cdef::cdef_direction_cost`).
#[target_feature(enable = "neon")]
pub(crate) fn cdef_direction_partials_8x8_neon(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    bd: u8,
) -> [[i32; 15]; 8] {
    unsafe {
        debug_assert!(x + 8 <= stride && y + 8 <= src.len() / stride);
        let nsh = vdupq_n_s16(-((bd - 8) as i16));
        let c128 = vdupq_n_s16(128);
        // Sixteen lanes of slack so every unaligned add stays in bounds.
        let mut acc = [[0i16; 24]; 8];
        let mut rowsum = [0i16; 8];
        let mut col = vdupq_n_s16(0);
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            let p = src.as_ptr().add((y + i) * stride + x);
            let v = vsubq_s16(vshlq_s16(vreinterpretq_s16_u16(vld1q_u16(p)), nsh), c128);
            let rev = {
                let r = vrev64q_s16(v);
                vcombine_s16(vget_high_s16(r), vget_low_s16(r))
            };
            let pairs = vget_low_s16(vpaddq_s16(v, v));
            let rpairs = vrev64_s16(pairs);
            let add8 = |a: &mut [i16; 24], at: usize, w: int16x8_t| {
                let q = a.as_mut_ptr().add(at);
                vst1q_s16(q, vaddq_s16(vld1q_s16(q), w));
            };
            let add4 = |a: &mut [i16; 24], at: usize, w: int16x4_t| {
                let q = a.as_mut_ptr().add(at);
                vst1_s16(q, vadd_s16(vld1_s16(q), w));
            };
            add8(&mut acc[0], i, v); // partial[0][i + j]
            add4(&mut acc[1], i, pairs); // partial[1][i + j/2]
            rowsum[i] = vaddvq_s16(v); // partial[2][i]
            add4(&mut acc[3], i, rpairs); // partial[3][3 + i - j/2]
            add8(&mut acc[4], i, rev); // partial[4][7 + i - j]
            add8(&mut acc[5], 3 - i / 2, v); // partial[5][3 - i/2 + j]
            col = vaddq_s16(col, v); // partial[6][j]
            add8(&mut acc[7], i / 2, v); // partial[7][i/2 + j]
        }
        let mut colv = [0i16; 8];
        vst1q_s16(colv.as_mut_ptr(), col);
        let mut out = [[0i32; 15]; 8];
        for d in [0usize, 1, 3, 4, 5, 7] {
            for k in 0..15 {
                out[d][k] = acc[d][k] as i32;
            }
        }
        for (k, (&r, &c)) in rowsum.iter().zip(colv.iter()).enumerate() {
            out[2][k] = r as i32;
            out[6][k] = c as i32;
        }
        out
    }
}

/// Per-candidate `(sum d, sum d^2, sum s*d)` over a full 8x8 block for every
/// entry of a `cdef_block_candidates` table — the inputs of
/// `cdef::cdef_dist_from_sums` (the source-only sums are the caller's).
#[target_feature(enable = "neon")]
pub(crate) fn cdef_cand_sums_8x8_neon(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    cand: &[[i32; 64]; N_CAND],
    out: &mut [[i64; 3]; N_CAND],
) {
    unsafe {
        debug_assert!(x + 8 <= stride && y + 8 <= src.len() / stride);
        let mut srows = [vdupq_n_s32(0); 16];
        for i in 0..8 {
            let p = src.as_ptr().add((y + i) * stride + x);
            let v = vld1q_u16(p);
            srows[2 * i] = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(v)));
            srows[2 * i + 1] = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(v)));
        }
        for (cb, o) in cand.iter().zip(out.iter_mut()) {
            let mut sd = vdupq_n_s32(0);
            let mut sd2 = vdupq_n_s64(0);
            let mut ssd = vdupq_n_s64(0);
            for i in 0..8 {
                let dlo = vld1q_s32(cb.as_ptr().add(i * 8));
                let dhi = vld1q_s32(cb.as_ptr().add(i * 8 + 4));
                sd = vaddq_s32(sd, vaddq_s32(dlo, dhi));
                // Per-row products stay below 2^31 (8 * 4095^2), so one pairwise
                // widening accumulate per row is exact.
                let d2 = vaddq_s32(vmulq_s32(dlo, dlo), vmulq_s32(dhi, dhi));
                sd2 = vpadalq_s32(sd2, d2);
                let sx = vaddq_s32(
                    vmulq_s32(srows[2 * i], dlo),
                    vmulq_s32(srows[2 * i + 1], dhi),
                );
                ssd = vpadalq_s32(ssd, sx);
            }
            o[0] = vaddvq_s32(sd) as i64;
            o[1] = vaddvq_s64(sd2);
            o[2] = vaddvq_s64(ssd);
        }
    }
}
