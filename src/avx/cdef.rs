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
#![allow(clippy::too_many_arguments)]

use crate::cdef::{
    CAND_PRI, CAND_SEC, CDEF_DIRECTIONS, CDEF_PRI_TAPS, CDEF_SEC_TAPS, CDEF_VERY_LARGE, N_CAND,
    adjust_pri, log2_floor,
};
use std::arch::x86_64::*;

/// Two 8-sample rows into one vector: row 0 in the low lane, row 1 in the high.
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn load_rows2(p0: *const u16, p1: *const u16) -> __m256i {
    unsafe {
        let lo = _mm_loadu_si128(p0.cast());
        let hi = _mm_loadu_si128(p1.cast());
        _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi)
    }
}

/// `constrain(d, strength, shift)` on sixteen lanes: `sign(d) * clamp(strength
/// - (|d| >> shift), 0, |d|)`; `sh` holds the shift count.
#[inline]
#[target_feature(enable = "avx2")]
fn constrain16(d: __m256i, strength: __m256i, sh: __m128i) -> __m256i {
    let a = _mm256_abs_epi16(d);
    let x = _mm256_sub_epi16(strength, _mm256_srl_epi16(a, sh));
    let cl = _mm256_min_epi16(_mm256_max_epi16(x, _mm256_setzero_si256()), a);
    _mm256_sign_epi16(cl, d)
}

/// Interior 8x8 [`crate::cdef::cdef_block_candidates`]: the caller guarantees
/// `x >= 2`, `y >= 2`, `x + 10 <= stride` and `y + 10 <= rows`.
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_block_candidates_8x8_avx2(
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

        let zero = _mm256_setzero_si256();
        let vl = _mm256_set1_epi16(CDEF_VERY_LARGE as i16);
        let mxv = _mm256_set1_epi16(maxv as i16);
        let eight = _mm256_set1_epi16(8);
        let pri_s = pri_eff.map(|s| _mm256_set1_epi16(s as i16));
        let pri_sh = pri_eff.map(|s| _mm_cvtsi32_si128(shift(s)));
        let pri_wv = pri_w.map(|w| w.map(|t| _mm256_set1_epi16(t as i16)));
        let sec_s = sec_eff.map(|s| _mm256_set1_epi16(s as i16));
        let sec_sh = sec_eff.map(|s| _mm_cvtsi32_si128(shift(s)));
        let sec_wv = CDEF_SEC_TAPS.map(|t| _mm256_set1_epi16(t as i16));

        let base = src.as_ptr();
        let s = stride as isize;
        for rp in 0..4usize {
            let i0 = 2 * rp;
            let row0 = ((y + i0) * stride + x) as isize;
            let row1 = row0 + s;
            let load = |dr: i32, dc: i32| -> __m256i {
                let off = dr as isize * s + dc as isize;
                load_rows2(base.offset(row0 + off), base.offset(row1 + off))
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
                    let invalid = _mm256_cmpeq_epi16(t, vl);
                    let d = _mm256_sub_epi16(t, c);
                    for q in 0..3 {
                        if pri_eff[q] != 0 {
                            let con =
                                _mm256_andnot_si256(invalid, constrain16(d, pri_s[q], pri_sh[q]));
                            psum[q] =
                                _mm256_add_epi16(psum[q], _mm256_mullo_epi16(con, pri_wv[q][k]));
                        }
                    }
                    mn = _mm256_min_epi16(mn, _mm256_blendv_epi8(t, mn, invalid));
                    mx = _mm256_max_epi16(mx, _mm256_blendv_epi8(t, mx, invalid));
                }
                for doff in [2usize, 6] {
                    let (sdr, sdc) = CDEF_DIRECTIONS[(dir + doff) & 7][k];
                    for sgn in [1i32, -1] {
                        let t = load(sgn * sdr, sgn * sdc);
                        let invalid = _mm256_cmpeq_epi16(t, vl);
                        let d = _mm256_sub_epi16(t, c);
                        for q in 0..2 {
                            let con =
                                _mm256_andnot_si256(invalid, constrain16(d, sec_s[q], sec_sh[q]));
                            ssum[q] = _mm256_add_epi16(ssum[q], _mm256_mullo_epi16(con, sec_wv[k]));
                        }
                        mn = _mm256_min_epi16(mn, _mm256_blendv_epi8(t, mn, invalid));
                        mx = _mm256_max_epi16(mx, _mm256_blendv_epi8(t, mx, invalid));
                    }
                    if dir != 0 {
                        let (sdr, sdc) = CDEF_DIRECTIONS[doff & 7][k];
                        for sgn in [1i32, -1] {
                            let t = load(sgn * sdr, sgn * sdc);
                            let invalid = _mm256_cmpeq_epi16(t, vl);
                            let d = _mm256_sub_epi16(t, c);
                            for q in 0..2 {
                                let con = _mm256_andnot_si256(
                                    invalid,
                                    constrain16(d, sec_s[q], sec_sh[q]),
                                );
                                ssum0[q] =
                                    _mm256_add_epi16(ssum0[q], _mm256_mullo_epi16(con, sec_wv[k]));
                            }
                        }
                    }
                }
            }
            if dir == 0 {
                ssum0 = ssum;
            }
            let c_is_vl = _mm256_cmpeq_epi16(c, vl);
            for si in 0..3usize {
                for pi in 0..4usize {
                    let en_p = pi != 0 && pri_eff[pi - 1] != 0;
                    let en_s = si != 0;
                    let sum = if pi == 0 {
                        if en_s { ssum0[si - 1] } else { zero }
                    } else {
                        let a = if en_p { psum[pi - 1] } else { zero };
                        if en_s {
                            _mm256_add_epi16(a, ssum[si - 1])
                        } else {
                            a
                        }
                    };
                    // centre + ((8 + sum - (sum < 0)) >> 4), arithmetic shift.
                    let adj = _mm256_add_epi16(
                        _mm256_add_epi16(sum, _mm256_srai_epi16::<15>(sum)),
                        eight,
                    );
                    let mut v = _mm256_add_epi16(c, _mm256_srai_epi16::<4>(adj));
                    if en_p && en_s {
                        v = _mm256_min_epi16(_mm256_max_epi16(v, mn), mx);
                    }
                    v = _mm256_min_epi16(_mm256_max_epi16(v, zero), mxv);
                    v = _mm256_blendv_epi8(v, c, c_is_vl);
                    let o = out[si * 4 + pi].as_mut_ptr().add(i0 * 8);
                    _mm256_storeu_si256(o.cast(), _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v)));
                    _mm256_storeu_si256(
                        o.add(8).cast(),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(v)),
                    );
                }
            }
        }
    }
}

/// Partial-sum accumulation of `cdef::cdef_direction` for one interior 8x8
/// block (`x + 8 <= stride`, `y + 8 <= rows`); see `neon::cdef` for the layout.
#[target_feature(enable = "avx2")]
pub(crate) unsafe fn cdef_direction_partials_8x8_avx2(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    bd: u8,
) -> [[i32; 15]; 8] {
    unsafe {
        debug_assert!(x + 8 <= stride && y + 8 <= src.len() / stride);
        let shcnt = _mm_cvtsi32_si128((bd - 8) as i32);
        let c128 = _mm_set1_epi16(128);
        // Word-reversal shuffle: lane j <- lane 7 - j.
        let rev_words = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
        // Sixteen lanes of slack so every unaligned add stays in bounds.
        let mut acc = [[0i16; 24]; 8];
        let mut rowsum = [0i16; 8];
        let mut col = _mm_setzero_si128();
        let add8 = |a: &mut [i16; 24], at: usize, w: __m128i| {
            let q = a.as_mut_ptr().add(at).cast::<__m128i>();
            _mm_storeu_si128(q, _mm_add_epi16(_mm_loadu_si128(q), w));
        };
        let add4 = |a: &mut [i16; 24], at: usize, w: __m128i| {
            let q = a.as_mut_ptr().add(at).cast::<__m128i>();
            _mm_storel_epi64(q, _mm_add_epi16(_mm_loadl_epi64(q), w));
        };
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            let p = src.as_ptr().add((y + i) * stride + x);
            let v = _mm_sub_epi16(_mm_srl_epi16(_mm_loadu_si128(p.cast()), shcnt), c128);
            let rev = _mm_shuffle_epi8(v, rev_words);
            let pairs = _mm_hadd_epi16(v, v);
            let rpairs = _mm_shufflelo_epi16::<0x1b>(pairs);
            add8(&mut acc[0], i, v); // partial[0][i + j]
            add4(&mut acc[1], i, pairs); // partial[1][i + j/2]
            let mut lanes = [0i16; 8];
            _mm_storeu_si128(lanes.as_mut_ptr().cast(), v);
            rowsum[i] = lanes.iter().sum(); // partial[2][i]
            add4(&mut acc[3], i, rpairs); // partial[3][3 + i - j/2]
            add8(&mut acc[4], i, rev); // partial[4][7 + i - j]
            add8(&mut acc[5], 3 - i / 2, v); // partial[5][3 - i/2 + j]
            col = _mm_add_epi16(col, v); // partial[6][j]
            add8(&mut acc[7], i / 2, v); // partial[7][i/2 + j]
        }
        let mut colv = [0i16; 8];
        _mm_storeu_si128(colv.as_mut_ptr().cast(), col);
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

/// Per-candidate `(sum d, sum d^2, sum s*d)` over a full 8x8 block; the
/// per-lane 32-bit accumulators hold at most `8 * 4095^2`.
#[target_feature(enable = "avx2")]
pub(crate) fn cdef_cand_sums_8x8_avx2(
    src: &[u16],
    stride: usize,
    x: usize,
    y: usize,
    cand: &[[i32; 64]; N_CAND],
    out: &mut [[i64; 3]; N_CAND],
) {
    unsafe {
        debug_assert!(x + 8 <= stride && y + 8 <= src.len() / stride);
        let mut srows = [_mm256_setzero_si256(); 8];
        for (i, r) in srows.iter_mut().enumerate() {
            let p = src.as_ptr().add((y + i) * stride + x);
            *r = _mm256_cvtepu16_epi32(_mm_loadu_si128(p.cast()));
        }
        let hsum = |v: __m256i| -> i64 {
            let mut lanes = [0i32; 8];
            _mm256_storeu_si256(lanes.as_mut_ptr().cast(), v);
            lanes.iter().map(|&l| l as i64).sum()
        };
        for (cb, o) in cand.iter().zip(out.iter_mut()) {
            let mut sd = _mm256_setzero_si256();
            let mut sd2 = _mm256_setzero_si256();
            let mut ssd = _mm256_setzero_si256();
            for (i, sr) in srows.iter().enumerate() {
                let d = _mm256_loadu_si256(cb.as_ptr().add(i * 8).cast());
                sd = _mm256_add_epi32(sd, d);
                sd2 = _mm256_add_epi32(sd2, _mm256_mullo_epi32(d, d));
                ssd = _mm256_add_epi32(ssd, _mm256_mullo_epi32(*sr, d));
            }
            o[0] = hsum(sd);
            o[1] = hsum(sd2);
            o[2] = hsum(ssd);
        }
    }
}
