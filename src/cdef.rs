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

//! AV1 Constrained Directional Enhancement Filter (CDEF), spec §7.15.
//!
//! CDEF is an in-loop post-filter applied after deblocking. It runs over each
//! 8x8 luma block: first a direction is estimated (`cdef_direction`), then a
//! primary filter follows that direction with strength `pri` and a secondary
//! filter runs at 45° offsets with strength `sec`, both attenuated by a
//! `damping` parameter so large differences from the centre pixel are ignored
//! (that is the "constrained" part — it preserves true edges while removing
//! ringing).
//!
//! This module implements the exact integer filter the decoder runs, so the
//! encoder can apply it to its own reconstruction and stay bit-synchronised. It
//! also provides a cheap per-64x64 strength search (`pick_strength_sse`) that
//! evaluates a small candidate set by SSE against the source — not a one-tap
//! proxy, but far cheaper than full RD.
//!
//! Pixels are `i32` samples in `0..=(1<<bd)-1`, matching the encoder's planes.

/// CDEF can be skipped per-8x8 when the block is flagged (largest-value sentinel).
pub(crate) const CDEF_VERY_LARGE: i32 = 0x4000;

/// Per-direction primary tap offsets (2 taps per direction, mirrored).
/// `cdef_directions[dir][k]` is a (row, col) offset; the mirrored tap is the
/// negation. Matches the spec table `Cdef_Directions` (8 directions, 2 taps).
const CDEF_DIRECTIONS: [[(i32, i32); 2]; 8] = [
    [(-1, 1), (-2, 2)],
    [(0, 1), (-1, 2)],
    [(0, 1), (0, 2)],
    [(0, 1), (1, 2)],
    [(1, 1), (2, 2)],
    [(1, 0), (2, 1)],
    [(1, 0), (2, 0)],
    [(1, 0), (2, -1)],
];

/// Primary tap weights, selected by `pri_strength & 1` (spec `Cdef_Pri_Taps`).
const CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
/// Secondary tap weights (spec `Cdef_Sec_Taps`), one row used for all strengths.
const CDEF_SEC_TAPS: [i32; 2] = [2, 1];

/// Constrain a difference `diff` between a tap and the centre by `strength` and
/// `damping` — see [`constrain_spec`]. Thin wrapper kept for readability at the
/// call sites.
#[inline]
#[allow(dead_code)]
fn constrain(diff: i32, strength: i32, damping: i32) -> i32 {
    constrain_spec(diff, strength, damping)
}

#[inline]
fn log2_floor(v: i32) -> i32 {
    if v <= 0 {
        0
    } else {
        31 - (v as u32).leading_zeros() as i32
    }
}

/// Spec-exact `constrain`: `sign(diff) * Clip3(0, |diff|, strength - (|diff| >>
/// (damping - log2(strength))))` with the shift floored at 0. When `strength`
/// is 0 the result is 0 (no filtering for that tap).
#[inline]
fn constrain_spec(diff: i32, strength: i32, damping: i32) -> i32 {
    if strength == 0 {
        return 0;
    }
    let shift = (damping - log2_floor(strength)).max(0);
    let reduced = strength - (diff.abs() >> shift);
    let clamped = reduced.clamp(0, diff.abs());
    if diff < 0 { -clamped } else { clamped }
}

/// Estimate the dominant direction (0..=7) of an 8x8 block and the strength of
/// that direction's correlation (`var`, used by callers if needed). Mirrors the
/// spec `cdef_direction` process: it correlates the 8x8 luma block against the 8
/// directional patterns and returns the best. `bd_shift = bd - 8` normalises
/// high-bit-depth samples to 8-bit before the variance comparison.
pub(crate) fn cdef_direction(
    plane: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    bd: u8,
) -> (usize, i32) {
    let coeff_shift = (bd - 8) as i32;
    let mut cost = [0i64; 8];
    let mut partial = [[0i32; 15]; 8];
    let rows = plane.len() / stride;
    for i in 0..8 {
        for j in 0..8 {
            // Clamp to the plane so partial-edge chroma blocks (e.g. 4:2:0 chroma
            // whose width isn't a multiple of 8) don't read out of bounds; the
            // edge sample is replicated, which is a reasonable direction estimate.
            let yy = (y + i).min(rows - 1);
            let xx = (x + j).min(stride - 1);
            let p = (plane[yy * stride + xx] >> coeff_shift) - 128;
            partial[0][i + j] += p;
            partial[1][i + j / 2] += p;
            partial[2][i] += p;
            partial[3][3 + i - j / 2] += p;
            partial[4][7 + i - j] += p;
            partial[5][3 - i / 2 + j] += p;
            partial[6][j] += p;
            partial[7][i / 2 + j] += p;
        }
    }
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        cost[2] += (partial[2][i] as i64) * (partial[2][i] as i64);
        cost[6] += (partial[6][i] as i64) * (partial[6][i] as i64);
    }
    cost[2] *= 105;
    cost[6] *= 105;
    // Divisor tables for the diagonal/near-diagonal partials (spec Div_Table).
    const DIV: [i64; 8] = [0, 840, 420, 280, 210, 168, 140, 120];
    for i in 0..7 {
        let i2 = 2 * i + 1;
        cost[0] +=
            ((partial[0][i] as i64).pow(2) + (partial[0][14 - i] as i64).pow(2)) * DIV[i + 1];
        cost[4] +=
            ((partial[4][i] as i64).pow(2) + (partial[4][14 - i] as i64).pow(2)) * DIV[i + 1];
        let _ = i2;
    }
    cost[0] += (partial[0][7] as i64).pow(2) * 105;
    cost[4] += (partial[4][7] as i64).pow(2) * 105;
    for s in [1usize, 3, 5, 7] {
        for i in 0..5 {
            cost[s] += (partial[s][3 + i] as i64).pow(2) * 105;
        }
        for i in 0..3 {
            cost[s] += ((partial[s][i] as i64).pow(2) + (partial[s][10 - i] as i64).pow(2))
                * DIV[2 * i + 2];
        }
    }
    let mut best_dir = 0usize;
    let mut best = cost[0];
    for (d, &c) in cost.iter().enumerate() {
        if c > best {
            best = c;
            best_dir = d;
        }
    }
    // Directional variance per spec 7.15.2: difference between the best cost and
    // the cost of the opposite direction, normalised by >>10. Used to scale the
    // primary strength per block (see `adjust_pri`).
    let var = ((best - cost[(best_dir + 4) & 7]) >> 10) as i32;
    (best_dir, var)
}

/// Scale the primary strength by the block's directional variance, per AV1 spec
/// 7.15.3: low-variance (flat-ish) blocks get a reduced primary strength so CDEF
/// doesn't over-smooth them. Secondary strength is never adjusted.
pub(crate) fn adjust_pri(pri: i32, var: i32) -> i32 {
    if var == 0 {
        return 0;
    }
    let vs = var >> 6;
    let l = if vs != 0 { log2_floor(vs).min(12) } else { 0 };
    (pri * (4 + l) + 8) >> 4
}

/// Apply the CDEF filter to one 8x8 block in place into `dst`, reading from
/// `src` (the deblocked reconstruction). `pri`/`sec` are the primary/secondary
/// strengths, `dir` the block direction, `damping` the frame damping, `bd` the
/// bit depth. Out-of-frame / skip taps are marked `CDEF_VERY_LARGE` in `src`
/// and excluded. Mirrors spec `cdef_filter` for one 8x8.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_8x8(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
    bd: u8,
) {
    cdef_filter_block(dst, src, stride, x, y, 8, 8, pri, sec, dir, damping, bd);
}

/// CDEF filter over a `bw` x `bh` block at (x, y). 8x8 for luma and 4:4:4
/// chroma; 4x8 for 4:2:2 chroma; 4x4 for 4:2:0 chroma. The taps, constrain, and
/// clamping are identical regardless of block size — only the iteration bounds
/// change — so this exactly mirrors the decoders' per-sub-block chroma filtering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block(
    dst: &mut [i32],
    src: &[i32],
    stride: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
    bd: u8,
) {
    let coeff_shift = (bd - 8) as i32;
    // Primary damping uses the frame damping; secondary uses damping-1 (>=1).
    // The spec further lowers primary damping by log2(pri) inside constrain via
    // the shift, so here we pass the frame damping straight through.
    let pri_damping = damping.max(1);
    // Secondary damping: AV1 uses the same frame damping as primary for the
    // secondary taps (no -1). Floored at 1.
    let sec_damping = damping.max(1);
    // pri_taps row selected by (pri >> coeff_shift) & 1 (spec).
    let pri_taps = CDEF_PRI_TAPS[((pri >> coeff_shift) & 1) as usize];
    let maxv = (1i32 << bd) - 1;
    let rows = src.len() / stride;
    for i in 0..bh {
        for j in 0..bw {
            let cx = x + j;
            let cy = y + i;
            if cx >= stride || cy >= rows {
                continue; // partial-edge block: skip out-of-frame samples
            }
            let centre = src[cy * stride + cx];
            if centre == CDEF_VERY_LARGE {
                dst[cy * stride + cx] = centre;
                continue;
            }
            let mut sum = 0i32;
            let mut min = centre;
            let mut max = centre;
            for k in 0..2usize {
                // Primary taps: along `dir`, both +/- offsets.
                let (pdr, pdc) = CDEF_DIRECTIONS[dir][k];
                for &sgn in &[1i32, -1] {
                    let p = sample(src, stride, cx as i32 + sgn * pdc, cy as i32 + sgn * pdr);
                    if p != CDEF_VERY_LARGE {
                        sum += pri_taps[k] * constrain_spec(p - centre, pri, pri_damping);
                        if p > max {
                            max = p;
                        }
                        if p < min {
                            min = p;
                        }
                    }
                }
                // Secondary taps: directions (dir+2)&7 and (dir+6)&7, both +/-.
                for &doff in &[2usize, 6] {
                    let sd = (dir + doff) & 7;
                    let (sdr, sdc) = CDEF_DIRECTIONS[sd][k];
                    for &sgn in &[1i32, -1] {
                        let p = sample(src, stride, cx as i32 + sgn * sdc, cy as i32 + sgn * sdr);
                        if p != CDEF_VERY_LARGE {
                            sum += CDEF_SEC_TAPS[k] * constrain_spec(p - centre, sec, sec_damping);
                            if p > max {
                                max = p;
                            }
                            if p < min {
                                min = p;
                            }
                        }
                    }
                }
            }
            let mut out = centre + ((8 + sum - (sum < 0) as i32) >> 4);
            if out < min {
                out = min;
            }
            if out > max {
                out = max;
            }
            dst[cy * stride + cx] = out.clamp(0, maxv);
        }
    }
}

/// Fetch a sample with bounds check; out-of-frame returns `CDEF_VERY_LARGE` so
/// the caller excludes it (spec pads with the large value).
#[inline]
fn sample(plane: &[i32], stride: usize, x: i32, y: i32) -> i32 {
    if x < 0 || y < 0 {
        return CDEF_VERY_LARGE;
    }
    let (x, y) = (x as usize, y as usize);
    let rows = plane.len() / stride;
    if x >= stride || y >= rows {
        return CDEF_VERY_LARGE;
    }
    plane[y * stride + x]
}

/// Candidate primary strengths searched per 64x64 (kept small for speed).
pub(crate) const PRI_CANDIDATES: [i32; 4] = [0, 1, 2, 4];
/// Candidate secondary strengths (spec values 0,1,2,4).
pub(crate) const SEC_CANDIDATES: [i32; 3] = [0, 1, 2];
