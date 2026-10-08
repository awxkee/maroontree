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

/// CDEF can be skipped per-8x8 when the block is flagged (largest-value sentinel).
pub(crate) const CDEF_VERY_LARGE: i32 = 0x4000;

/// Per-direction primary tap offsets (2 taps per direction, mirrored).
/// `cdef_directions[dir][k]` is a (row, col) offset; the mirrored tap is the
/// negation. Matches the spec table `Cdef_Directions` (8 directions, 2 taps).
static CDEF_DIRECTIONS: [[(i32, i32); 2]; 8] = [
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
static CDEF_PRI_TAPS: [[i32; 2]; 2] = [[4, 2], [3, 3]];
/// Secondary tap weights (spec `Cdef_Sec_Taps`), one row used for all strengths.
static CDEF_SEC_TAPS: [i32; 2] = [2, 1];

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
/// directional patterns and returns the best. `bd_shift = bd - 8` normalizes
/// high-bit-depth samples to 8-bit before the variance comparison.
pub(crate) fn cdef_direction<P: crate::intrapred::Pel>(
    plane: &[P],
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
            let p = (plane[yy * stride + xx].widen() >> coeff_shift) - 128;
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
    // the cost of the opposite direction, normalized by >>10. Used to scale the
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
pub(crate) fn cdef_filter_8x8<P: crate::intrapred::Pel>(
    dst: &mut [P],
    src: &[P],
    stride: usize,
    x: usize,
    y: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    damping: i32,
    bd: u8,
) {
    cdef_filter_block(dst, 0, src, stride, x, y, 8, 8, pri, sec, dir, damping, bd);
}

/// CDEF filter over a `bw` x `bh` block at (x, y). 8x8 for luma and 4:4:4
/// chroma; 4x8 for 4:2:2 chroma; 4x4 for 4:2:0 chroma. The taps, constrain, and
/// clamping are identical regardless of block size — only the iteration bounds
/// change — so this exactly mirrors the decoders' per-sub-block chroma filtering.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_filter_block<P: crate::intrapred::Pel>(
    dst: &mut [P],
    dst_y0: usize,
    src: &[P],
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
    // libaom/AVM dispatch (`cdef_filter_16_{0..3}` / `cdef_filter_block_internal`):
    // a zero (variance-adjusted) primary strength disables the primary tap loop
    // entirely, a zero secondary strength the secondary loop, and the min/max
    // output clamp runs ONLY when both are enabled (`clipping_required`).
    let enable_pri = pri != 0;
    let enable_sec = sec != 0;
    let clipping = enable_pri && enable_sec;
    let maxv = (1i32 << bd) - 1;
    let rows = src.len() / stride;
    // Every tap lies within 2 samples of the centre, so a block with a 2-sample
    // in-plane margin never reaches the out-of-frame branch of `sample`.
    if x >= 2 && y >= 2 && x + bw + 2 <= stride && y + bh + 2 <= rows {
        let s = stride as isize;
        let off = |(dr, dc): (i32, i32)| dr as isize * s + dc as isize;
        let pri_off = [off(CDEF_DIRECTIONS[dir][0]), off(CDEF_DIRECTIONS[dir][1])];
        let (s2, s6) = ((dir + 2) & 7, (dir + 6) & 7);
        let sec_off = [
            [off(CDEF_DIRECTIONS[s2][0]), off(CDEF_DIRECTIONS[s6][0])],
            [off(CDEF_DIRECTIONS[s2][1]), off(CDEF_DIRECTIONS[s6][1])],
        ];
        // `constrain_spec` with the per-strength shift hoisted out of the taps.
        let shift = |strength: i32, damping: i32| {
            if strength == 0 {
                0
            } else {
                (damping - log2_floor(strength)).max(0)
            }
        };
        let (pri_shift, sec_shift) = (shift(pri, pri_damping), shift(sec, sec_damping));
        let constrain = |diff: i32, strength: i32, shift: i32| -> i32 {
            let a = diff.abs();
            let clamped = (strength - (a >> shift)).clamp(0, a);
            if diff < 0 { -clamped } else { clamped }
        };
        for i in 0..bh {
            let row = (y + i) * stride + x;
            let drow = (y + i - dst_y0) * stride + x;
            for j in 0..bw {
                let idx = (row + j) as isize;
                let centre = src[idx as usize].widen();
                if centre == CDEF_VERY_LARGE {
                    continue;
                }
                let tap = |o: isize| src[(idx + o) as usize].widen();
                let mut sum = 0i32;
                let mut min = centre;
                let mut max = centre;
                for k in 0..2usize {
                    if enable_pri {
                        for p in [tap(pri_off[k]), tap(-pri_off[k])] {
                            if p != CDEF_VERY_LARGE {
                                sum += pri_taps[k] * constrain(p - centre, pri, pri_shift);
                                if clipping {
                                    max = max.max(p);
                                    min = min.min(p);
                                }
                            }
                        }
                    }
                    if enable_sec {
                        for o in sec_off[k] {
                            for p in [tap(o), tap(-o)] {
                                if p != CDEF_VERY_LARGE {
                                    sum += CDEF_SEC_TAPS[k] * constrain(p - centre, sec, sec_shift);
                                    if clipping {
                                        max = max.max(p);
                                        min = min.min(p);
                                    }
                                }
                            }
                        }
                    }
                }
                let mut out = centre + ((8 + sum - (sum < 0) as i32) >> 4);
                if clipping {
                    out = out.clamp(min, max);
                }
                dst[drow + j] = P::narrow(out.clamp(0, maxv));
            }
        }
        return;
    }
    cdef_filter_block_generic(
        dst,
        dst_y0,
        src,
        stride,
        x,
        y,
        bw,
        bh,
        pri,
        sec,
        dir,
        pri_damping,
        sec_damping,
        bd,
    );
}

/// Per-sample bounds-checked [`cdef_filter_block`] body: any block position,
/// including frame edges, and the exactness reference for the fast paths.
#[allow(clippy::too_many_arguments)]
fn cdef_filter_block_generic<P: crate::intrapred::Pel>(
    dst: &mut [P],
    dst_y0: usize,
    src: &[P],
    stride: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    pri: i32,
    sec: i32,
    dir: usize,
    pri_damping: i32,
    sec_damping: i32,
    bd: u8,
) {
    let coeff_shift = (bd - 8) as i32;
    let pri_taps = CDEF_PRI_TAPS[((pri >> coeff_shift) & 1) as usize];
    let enable_pri = pri != 0;
    let enable_sec = sec != 0;
    let clipping = enable_pri && enable_sec;
    let maxv = (1i32 << bd) - 1;
    let rows = src.len() / stride;
    for i in 0..bh {
        for j in 0..bw {
            let cx = x + j;
            let cy = y + i;
            if cx >= stride || cy >= rows {
                continue; // partial-edge block: skip out-of-frame samples
            }
            let centre = src[cy * stride + cx].widen();
            if centre == CDEF_VERY_LARGE {
                // Out-of-frame centre (marked snapshot): the decoder has no such
                // pixel, so leave `dst` untouched.
                continue;
            }
            let mut sum = 0i32;
            let mut min = centre;
            let mut max = centre;
            for k in 0..2usize {
                // Primary taps: along `dir`, both +/- offsets.
                if enable_pri {
                    let (pdr, pdc) = CDEF_DIRECTIONS[dir][k];
                    for &sgn in &[1i32, -1] {
                        let p = sample(src, stride, cx as i32 + sgn * pdc, cy as i32 + sgn * pdr);
                        if p != CDEF_VERY_LARGE {
                            sum += pri_taps[k] * constrain_spec(p - centre, pri, pri_damping);
                            if clipping {
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
                // Secondary taps: directions (dir+2)&7 and (dir+6)&7, both +/-.
                if enable_sec {
                    for &doff in &[2usize, 6] {
                        let sd = (dir + doff) & 7;
                        let (sdr, sdc) = CDEF_DIRECTIONS[sd][k];
                        for &sgn in &[1i32, -1] {
                            let p =
                                sample(src, stride, cx as i32 + sgn * sdc, cy as i32 + sgn * sdr);
                            if p != CDEF_VERY_LARGE {
                                sum +=
                                    CDEF_SEC_TAPS[k] * constrain_spec(p - centre, sec, sec_damping);
                                if clipping {
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
                }
            }
            let mut out = centre + ((8 + sum - (sum < 0) as i32) >> 4);
            if clipping {
                out = out.clamp(min, max);
            }
            dst[(cy - dst_y0) * stride + cx] = P::narrow(out.clamp(0, maxv));
        }
    }
}

/// Fetch a sample with bounds check; out-of-frame returns `CDEF_VERY_LARGE` so
/// the caller excludes it (spec pads with the large value).
#[inline]
fn sample<P: crate::intrapred::Pel>(plane: &[P], stride: usize, x: i32, y: i32) -> i32 {
    if x < 0 || y < 0 {
        return CDEF_VERY_LARGE;
    }
    let (x, y) = (x as usize, y as usize);
    let rows = plane.len() / stride;
    if x >= stride || y >= rows {
        return CDEF_VERY_LARGE;
    }
    plane[y * stride + x].widen()
}

/// Signaled strengths [`cdef_block_candidates`] evaluates together: candidate
/// `si * 4 + pi` is (`CAND_PRI[pi]`, `CAND_SEC[si]`); index 0 is unfiltered.
pub(crate) const CAND_PRI: [i32; 4] = [0, 1, 2, 4];
pub(crate) const CAND_SEC: [i32; 3] = [0, 1, 2];
pub(crate) const N_CAND: usize = 12;

/// Index of signaled `(pri, sec)` in [`cdef_block_candidates`]' output.
pub(crate) fn cand_index(pri: i32, sec: i32) -> usize {
    let pi = CAND_PRI
        .iter()
        .position(|&p| p == pri)
        .expect("CDEF pri candidate");
    let si = CAND_SEC
        .iter()
        .position(|&s| s == sec)
        .expect("CDEF sec candidate");
    si * 4 + pi
}

/// Filter one `bw` x `bh` block (at most 8x8) at every signaled strength pair
/// in [`CAND_PRI`] x [`CAND_SEC`] at once. `out[c][i * bw + j]` equals what
/// [`cdef_filter_block`] writes for candidate `c` with the decoders' inputs:
/// primary strength `adjust_pri(pri << (bd - 8), var)` when `var` is given
/// (luma) or `pri << (bd - 8)` (chroma), secondary `sec << (bd - 8)`, and
/// direction `dir`, or 0 when the signaled primary is 0. Samples whose centre
/// is `CDEF_VERY_LARGE` (which that filter leaves untouched) get the centre.
///
/// Each tap's constrained contribution is computed once per strength and the
/// twelve outputs are then combined from the shared sums. All arithmetic is
/// integer, so the result is exact whatever the summation order.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cdef_block_candidates<P: crate::intrapred::Pel>(
    src: &[P],
    stride: usize,
    x: usize,
    y: usize,
    bw: usize,
    bh: usize,
    dir: usize,
    var: Option<i32>,
    damping: i32,
    bd: u8,
    out: &mut [[i32; 64]; N_CAND],
) {
    debug_assert!(bw <= 8 && bh <= 8);
    let n = bw * bh;
    let coeff_shift = (bd - 8) as i32;
    let maxv = (1i32 << bd) - 1;
    let rows = src.len() / stride;
    let interior = x >= 2 && y >= 2 && x + bw + 2 <= stride && y + bh + 2 <= rows;
    let gather = |dr: i32, dc: i32, buf: &mut [i32; 64]| {
        for i in 0..bh {
            if interior {
                let start = ((y + i) as isize + dr as isize) as usize * stride
                    + (x as isize + dc as isize) as usize;
                for (b, &p) in buf[i * bw..(i + 1) * bw]
                    .iter_mut()
                    .zip(&src[start..start + bw])
                {
                    *b = p.widen();
                }
            } else {
                for j in 0..bw {
                    buf[i * bw + j] = sample(src, stride, (x + j) as i32 + dc, (y + i) as i32 + dr);
                }
            }
        }
    };
    let mut c = [0i32; 64];
    gather(0, 0, &mut c);

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
    let pri_shift = pri_eff.map(shift);
    let pri_w = pri_eff.map(|pe| CDEF_PRI_TAPS[((pe >> coeff_shift) & 1) as usize]);
    let sec_eff = [CAND_SEC[1] << coeff_shift, CAND_SEC[2] << coeff_shift];
    let sec_shift = sec_eff.map(shift);

    #[inline(always)]
    fn constrain(diff: i32, strength: i32, shift: i32) -> i32 {
        let a = diff.abs();
        let clamped = (strength - (a >> shift)).clamp(0, a);
        if diff < 0 { -clamped } else { clamped }
    }

    let mut psum = [[0i32; 64]; 3];
    let mut ssum = [[0i32; 64]; 2]; // secondary taps along `dir`
    let mut ssum0 = [[0i32; 64]; 2]; // secondary taps along direction 0
    let mut mn = c;
    let mut mx = c;
    let mut t = [0i32; 64];
    for k in 0..2usize {
        let (pdr, pdc) = CDEF_DIRECTIONS[dir][k];
        for sgn in [1i32, -1] {
            gather(sgn * pdr, sgn * pdc, &mut t);
            for p in 0..n {
                let tp = t[p];
                if tp == CDEF_VERY_LARGE {
                    continue;
                }
                let d = tp - c[p];
                for q in 0..3 {
                    if pri_eff[q] != 0 {
                        psum[q][p] += pri_w[q][k] * constrain(d, pri_eff[q], pri_shift[q]);
                    }
                }
                mn[p] = mn[p].min(tp);
                mx[p] = mx[p].max(tp);
            }
        }
        for (sec_dir, sums, minmax) in [(dir, &mut ssum, true), (0, &mut ssum0, false)] {
            if !minmax && dir == 0 {
                continue; // identical to the `dir` sums; copied below
            }
            for doff in [2usize, 6] {
                let (sdr, sdc) = CDEF_DIRECTIONS[(sec_dir + doff) & 7][k];
                for sgn in [1i32, -1] {
                    gather(sgn * sdr, sgn * sdc, &mut t);
                    for p in 0..n {
                        let tp = t[p];
                        if tp == CDEF_VERY_LARGE {
                            continue;
                        }
                        let d = tp - c[p];
                        for q in 0..2 {
                            sums[q][p] += CDEF_SEC_TAPS[k] * constrain(d, sec_eff[q], sec_shift[q]);
                        }
                        if minmax {
                            mn[p] = mn[p].min(tp);
                            mx[p] = mx[p].max(tp);
                        }
                    }
                }
            }
        }
    }
    if dir == 0 {
        ssum0 = ssum;
    }

    for si in 0..3usize {
        for pi in 0..4usize {
            let o = &mut out[si * 4 + pi];
            let (en_p, en_s) = (pi != 0 && pri_eff[pi - 1] != 0, si != 0);
            let clip = en_p && en_s;
            for p in 0..n {
                let centre = c[p];
                if centre == CDEF_VERY_LARGE {
                    o[p] = centre;
                    continue;
                }
                let sum = if pi == 0 {
                    if en_s { ssum0[si - 1][p] } else { 0 }
                } else {
                    (if en_p { psum[pi - 1][p] } else { 0 })
                        + (if en_s { ssum[si - 1][p] } else { 0 })
                };
                let mut v = centre + ((8 + sum - (sum < 0) as i32) >> 4);
                if clip {
                    v = v.clamp(mn[p], mx[p]);
                }
                o[p] = v.clamp(0, maxv);
            }
        }
    }
}

/// Candidate primary strengths searched per 64x64 (kept small for speed).
pub(crate) static PRI_CANDIDATES: [i32; 4] = [0, 1, 2, 4];
/// Candidate secondary strengths (spec values 0,1,2,4).
pub(crate) static SEC_CANDIDATES: [i32; 3] = [0, 1, 2];

/// SSE over one (possibly edge-clipped) 8x8 block of a plane.
pub(crate) fn block_sse_8x8<P: crate::intrapred::Pel>(
    a: &[P],
    b: &[P],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
) -> i64 {
    let mut s = 0i64;
    let yh = (y + 8).min(h);
    let xw = (x + 8).min(w);
    for yy in y..yh {
        for xx in x..xw {
            let d = (a[yy * w + xx].widen() as i64) - (b[yy * w + xx].widen() as i64);
            s += d * d;
        }
    }
    s
}

/// libaom's perceptual CDEF distortion for one 8x8 block (`dist_8x8_16bit`): the
/// SSE weighted by `0.5 * (svar + dvar + c1) / sqrt(c2 + svar*dvar)`, where `svar`
/// / `dvar` are the (x64) source / reconstructed variances. The weight up-weights
/// error in flat blocks (where ringing is visible) and down-weights it in
/// high-variance textured blocks (masking) — so filtering, which cuts a block's
/// variance, raises its weight and only wins where the SSE drop is real ringing.
/// `dst` is the candidate (filtered/unfiltered) recon, `src` the source. Partial
/// edge blocks fall back to plain SSE (the ×64 variance calibration needs a full
/// 8x8). This is an encoder-side decision metric, not part of the bitstream.
pub(crate) fn cdef_dist_8x8<P: crate::intrapred::Pel>(
    src: &[P],
    dst: &[P],
    w: usize,
    h: usize,
    x: usize,
    y: usize,
    coeff_shift: u32,
) -> i64 {
    if x + 8 > w || y + 8 > h {
        return block_sse_8x8(dst, src, w, h, x, y);
    }
    let (mut ss, mut sd, mut ss2, mut sd2, mut ssd) = (0i64, 0i64, 0i64, 0i64, 0i64);
    for yy in y..y + 8 {
        for xx in x..x + 8 {
            let s = src[yy * w + xx].widen() as i64;
            let d = dst[yy * w + xx].widen() as i64;
            ss += s;
            sd += d;
            ss2 += s * s;
            sd2 += d * d;
            ssd += s * d;
        }
    }
    cdef_dist_from_sums(ss, sd, ss2, sd2, ssd, coeff_shift)
}

/// [`cdef_dist_8x8`] from the block's 64-sample sums (source, candidate,
/// their squares and cross product).
pub(crate) fn cdef_dist_from_sums(
    ss: i64,
    sd: i64,
    ss2: i64,
    sd2: i64,
    ssd: i64,
    coeff_shift: u32,
) -> i64 {
    let svar = ss2 - ((ss * ss + 32) >> 6);
    let dvar = sd2 - ((sd * sd + 32) >> 6);
    let sse = (sd2 + ss2 - 2 * ssd) as f64;
    let c1 = (400i64 << (2 * coeff_shift)) as f64;
    let c2 = (20000i64 << (4 * coeff_shift)) as f64;
    let w = 0.5 * (svar as f64 + dvar as f64 + c1) / (c2 + svar as f64 * dvar as f64).sqrt();
    (0.5 + sse * w) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fast_paths_match_generic_filter() {
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = |m: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % m
        };
        for bd in [8u8, 10, 12] {
            let (stride, rows) = (40usize, 24usize);
            for trial in 0..400 {
                let maxv = (1u64 << bd) - 1;
                // Smooth-ish base with sharp outliers so every constrain branch fires.
                let base = rnd(maxv + 1);
                let src: Vec<u16> = (0..stride * rows)
                    .map(|_| {
                        if trial % 5 == 0 && rnd(23) == 0 {
                            CDEF_VERY_LARGE as u16
                        } else if rnd(4) == 0 {
                            rnd(maxv + 1) as u16
                        } else {
                            (base as i64 + rnd(33) as i64 - 16).clamp(0, maxv as i64) as u16
                        }
                    })
                    .collect();
                let (bw, bh) = [(8, 8), (4, 8), (4, 4)][rnd(3) as usize];
                let x = rnd((stride - bw + 1) as u64) as usize;
                let y = rnd((rows - bh + 1) as u64) as usize;
                let dir = rnd(8) as usize;
                let damping = 3 + rnd(4) as i32 + (bd as i32 - 8);
                let var = [0, 7, 100, 5000, 1 << 20][rnd(5) as usize];
                let luma = rnd(2) == 0;
                let mut cand = [[0i32; 64]; N_CAND];
                cdef_block_candidates(
                    &src,
                    stride,
                    x,
                    y,
                    bw,
                    bh,
                    dir,
                    luma.then_some(var),
                    damping,
                    bd,
                    &mut cand,
                );
                for (c, got) in cand.iter().enumerate() {
                    let (pri, sec) = (CAND_PRI[c % 4], CAND_SEC[c / 4]);
                    let sh = bd - 8;
                    let p = if luma {
                        adjust_pri(pri << sh, var)
                    } else {
                        pri << sh
                    };
                    let d = if pri == 0 { 0 } else { dir };
                    let mut fast = src.clone();
                    let mut slow = src.clone();
                    cdef_filter_block(
                        &mut fast,
                        0,
                        &src,
                        stride,
                        x,
                        y,
                        bw,
                        bh,
                        p,
                        sec << sh,
                        d,
                        damping,
                        bd,
                    );
                    let dm = damping.max(1);
                    cdef_filter_block_generic(
                        &mut slow,
                        0,
                        &src,
                        stride,
                        x,
                        y,
                        bw,
                        bh,
                        p,
                        sec << sh,
                        d,
                        dm,
                        dm,
                        bd,
                    );
                    assert_eq!(fast, slow, "fast path bd={bd} trial={trial} cand={c}");
                    for i in 0..bh {
                        for j in 0..bw {
                            let at = (y + i) * stride + x + j;
                            if src[at] as i32 == CDEF_VERY_LARGE {
                                continue;
                            }
                            assert_eq!(
                                got[i * bw + j],
                                slow[at] as i32,
                                "candidates bd={bd} trial={trial} cand={c} ({pri},{sec}) px={j},{i}"
                            );
                        }
                    }
                }
            }
        }
    }
}
