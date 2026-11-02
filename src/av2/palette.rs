/*
 * Copyright (c) Radzivon Bartoshyk 11/2026. All rights reserved.
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

//! AV2 luma palette for lossy screen-content frames.
//!
//! Syntax (AVM): a DC_PRED luma block of 8x8..64x64 (area >= 64) in a frame with
//! `allow_screen_content_tools` reads `palette_y_mode`; a palette block then reads
//! its size (2..8), the colors (a neighbour color cache of per-entry flags, then
//! ascending literals) and, after the rest of the mode info, the color-index map:
//! an optional transpose bit, then per scan line a 3-ary flag (0 = indices coded,
//! 1 = one coded index repeated along the line, 2 = copy of the previous line).
//! The decoder sorts the colors, so their order carries no information: the
//! encoder levers are the color values and the map.
//!
//! The encoder side picks the colors with an exact 1-D DP (optimal contiguous
//! clustering of the block's sorted luma histogram, every size at once) and the map
//! with a line-by-line RD walk over the three line flags and both scan directions.
//! All RD pricing uses the default CDFs (like every other still decision), so it
//! is identical in the wavefront decide pass and the serial emit pass; the color
//! cache, which depends on emitted neighbours, only lowers the real cost.

use crate::av2::entropy::RangeEncoder;
use std::cell::Cell;
use std::sync::OnceLock;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
}

/// Palette candidates for the SB decision in scope (screen-content frames): the
/// shared luma coders and the partition probe consult [`enabled`]. Restores the
/// previous state on drop.
pub(crate) struct Scope(bool);

impl Scope {
    pub(crate) fn new(on: bool) -> Self {
        Scope(ON.with(|c| c.replace(on)))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        ON.with(|c| c.set(self.0));
    }
}

#[inline]
pub(crate) fn enabled() -> bool {
    ON.with(|c| c.get())
}

/// Replay token mode for "DC_PRED + palette" (luma modes use 0..=12).
pub(crate) const PAL_MODE: usize = 15;
pub(crate) const MAX_COLORS: usize = 8;

/// AVM default_palette_y_mode_cdf (ICDF) and its PARA.
pub(crate) const Y_MODE_ICDF: u16 = 2723;
pub(crate) const Y_MODE_PARA: (u8, u8, u8) = (0, 1, 2);
/// AVM default_palette_y_size_cdf (7 sizes).
pub(crate) const Y_SIZE_ICDF: [u16; 6] = [23989, 17673, 11991, 7865, 4845, 2365];
pub(crate) const Y_SIZE_PARA: (u8, u8, u8) = (2, 3, 3);
/// AVM default_identity_row_cdf_y[ctx] (3 symbols); ctx 3 = first line.
pub(crate) const ROW_ICDF: [[u16; 2]; 4] =
    [[10253, 7017], [28754, 27535], [29220, 28605], [19769, 12]];
pub(crate) const ROW_PARA: [(u8, u8, u8); 4] = [(1, 3, 4), (1, 2, 3), (1, 2, 5), (0, 2, 3)];
/// AVM default_palette_y_color_index_cdf[size - 2][ctx] (zero padded).
pub(crate) const COLOR_ICDF: [[[u16; 7]; 5]; 7] = [
    [
        [4628, 0, 0, 0, 0, 0, 0],
        [16384, 0, 0, 0, 0, 0, 0],
        [24186, 0, 0, 0, 0, 0, 0],
        [5355, 0, 0, 0, 0, 0, 0],
        [2339, 0, 0, 0, 0, 0, 0],
    ],
    [
        [7418, 3742, 0, 0, 0, 0, 0],
        [21405, 7495, 0, 0, 0, 0, 0],
        [25927, 4189, 0, 0, 0, 0, 0],
        [11418, 6756, 0, 0, 0, 0, 0],
        [2195, 1122, 0, 0, 0, 0, 0],
    ],
    [
        [9062, 5806, 3708, 0, 0, 0, 0],
        [22792, 10252, 5386, 0, 0, 0, 0],
        [26077, 7308, 3534, 0, 0, 0, 0],
        [13859, 8843, 4365, 0, 0, 0, 0],
        [2460, 1692, 950, 0, 0, 0, 0],
    ],
    [
        [8652, 5811, 4282, 2827, 0, 0, 0],
        [23200, 12296, 8474, 3826, 0, 0, 0],
        [27062, 7525, 4728, 2362, 0, 0, 0],
        [12663, 9786, 5744, 3857, 0, 0, 0],
        [1871, 1426, 1002, 569, 0, 0, 0],
    ],
    [
        [11944, 8541, 6842, 5309, 3502, 0, 0],
        [24627, 13779, 11169, 6586, 4192, 0, 0],
        [27516, 8428, 6318, 4330, 2143, 0, 0],
        [13249, 10073, 7181, 5796, 4345, 0, 0],
        [2385, 1878, 1521, 1115, 618, 0, 0],
    ],
    [
        [11140, 8256, 6895, 5714, 4637, 3229, 0],
        [24740, 14504, 12155, 7344, 5656, 3862, 0],
        [26279, 10526, 8307, 6374, 4418, 2258, 0],
        [10720, 8339, 5778, 4824, 4351, 3194, 0],
        [1967, 1563, 1296, 1040, 763, 463, 0],
    ],
    [
        [10297, 7685, 6784, 5875, 5114, 4018, 2865],
        [25226, 15711, 13617, 9218, 7309, 5702, 3964],
        [25186, 12331, 10040, 8146, 6253, 4189, 2136],
        [10666, 8624, 5852, 4617, 3922, 3556, 2615],
        [2244, 1881, 1612, 1375, 1142, 857, 487],
    ],
];
pub(crate) const COLOR_PARA: [[(u8, u8, u8); 5]; 7] = [
    [(3, 4, 4), (2, 3, 4), (2, 2, 3), (1, 2, 2), (3, 4, 5)],
    [(3, 4, 4), (2, 2, 2), (2, 3, 3), (2, 2, 3), (3, 4, 5)],
    [(3, 4, 5), (3, 4, 4), (3, 3, 4), (2, 3, 4), (4, 5, 6)],
    [(3, 4, 5), (4, 3, 4), (4, 4, 4), (2, 3, 4), (4, 5, 6)],
    [(4, 4, 5), (4, 4, 5), (4, 4, 5), (3, 3, 4), (4, 4, 6)],
    [(4, 3, 5), (4, 5, 5), (4, 4, 5), (4, 4, 4), (4, 5, 6)],
    [(4, 5, 6), (4, 5, 5), (4, 5, 5), (3, 4, 5), (4, 5, 6)],
];

/// Whether a luma block of `bw4` x `bh4` (4-px units) reads `palette_y_mode` when
/// DC-predicted (AVM av2_allow_palette: <= 64 per side, BLOCK_8X8 and the larger
/// enum entries, i.e. every size but 4x4 / 4x8 / 8x4).
#[inline]
pub(crate) fn block_allowed(bw4: usize, bh4: usize) -> bool {
    bw4 <= 16 && bh4 <= 16 && bw4 * bh4 >= 4
}

/// A luma palette block: ascending distinct colors, the row-major index map and
/// the per-line flags of the chosen scan direction.
#[derive(Clone, Debug)]
pub(crate) struct Palette {
    pub(crate) colors: Vec<u16>,
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) map: Vec<u8>,
    pub(crate) transposed: bool,
    pub(crate) flags: Vec<u8>,
    pub(crate) bd: u32,
    /// Default-CDF bits of the palette syntax (mode-on flag, size, colors without
    /// cache, map), for the leaf RD.
    pub(crate) bits: f32,
}

impl Palette {
    /// The palette prediction (row-major `w` x `h`).
    pub(crate) fn pred(&self) -> Vec<f32> {
        self.map
            .iter()
            .map(|&i| self.colors[i as usize] as f32)
            .collect()
    }
}

/// AVM derive_color_index_ctx: the context and the color order (neighbour colors
/// first, then the remaining indices ascending) at (y, x) of a `stride`-wide map.
pub(crate) fn color_ctx(
    map: &[u8],
    stride: usize,
    y: usize,
    x: usize,
    size: usize,
) -> (usize, [u8; 8]) {
    let mut order = [0u8; 8];
    let mut used = [false; 8];
    let mut count = 0usize;
    macro_rules! push {
        ($color:expr) => {{
            let color = $color;
            if !used[color as usize] {
                order[count] = color;
                used[color as usize] = true;
                count += 1;
            }
        }};
    }
    let ctx = if y == 0 {
        push!(map[x - 1]);
        0
    } else if x == 0 {
        push!(map[(y - 1) * stride]);
        0
    } else {
        let left = map[y * stride + x - 1];
        let diag = map[(y - 1) * stride + x - 1];
        let top = map[(y - 1) * stride + x];
        if left == diag && left == top {
            push!(left);
            4
        } else if left == top {
            push!(left);
            push!(diag);
            3
        } else if left == diag {
            push!(left);
            push!(top);
            2
        } else if diag == top {
            push!(top);
            push!(left);
            2
        } else {
            // This AVM palette-simplification revision keeps neighbor order
            // (left, top, top-left); it does not sort left/top here.
            push!(left);
            push!(top);
            push!(diag);
            1
        }
    };
    for color in 0..size as u8 {
        push!(color);
    }
    debug_assert_eq!(count, size);
    (ctx, order)
}

struct Bits {
    mode: [f32; 2],
    size: [f32; 7],
    row: [[f32; 3]; 4],
    color: [[[f32; 8]; 5]; 7],
}

fn bits() -> &'static Bits {
    static B: OnceLock<Bits> = OnceLock::new();
    B.get_or_init(|| Bits {
        mode: std::array::from_fn(|s| RangeEncoder::symbol_bits(&[Y_MODE_ICDF], s, 2)),
        size: std::array::from_fn(|s| RangeEncoder::symbol_bits(&Y_SIZE_ICDF, s, 7)),
        row: std::array::from_fn(|c| {
            std::array::from_fn(|s| RangeEncoder::symbol_bits(&ROW_ICDF[c], s, 3))
        }),
        color: std::array::from_fn(|n| {
            std::array::from_fn(|c| {
                std::array::from_fn(|s| {
                    if s < n + 2 {
                        RangeEncoder::symbol_bits(&COLOR_ICDF[n][c], s, n + 2)
                    } else {
                        0.0
                    }
                })
            })
        }),
    })
}

fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

fn uniform_bits(v: usize, n: usize) -> f32 {
    let l = 32 - (n as u32).leading_zeros();
    let m = (1usize << l) - n;
    if v < m { (l - 1) as f32 } else { l as f32 }
}

/// Bits of the ascending literal colors (`palette_colors` + `bits` + deltas).
fn literal_bits(colors: &[u16], bd: u32) -> u32 {
    if colors.is_empty() {
        return 0;
    }
    let mut total = bd;
    if colors.len() > 1 {
        let min_bits = bd - 3;
        let max_delta = colors
            .windows(2)
            .map(|v| (v[1] - v[0]) as u32)
            .max()
            .unwrap();
        let mut b = ceil_log2(max_delta).max(min_bits);
        total += 2;
        let mut range = (1u32 << bd) - colors[0] as u32 - 1;
        for v in colors.windows(2) {
            let d = (v[1] - v[0]) as u32;
            total += b;
            range -= d;
            b = b.min(ceil_log2(range));
        }
    }
    total
}

fn emit_literals(enc: &mut RangeEncoder, colors: &[u16], bd: u32) {
    if colors.is_empty() {
        return;
    }
    enc.encode_bypass(colors[0] as u32, bd);
    if colors.len() == 1 {
        return;
    }
    let min_bits = bd - 3;
    let max_delta = colors
        .windows(2)
        .map(|v| (v[1] - v[0]) as u32)
        .max()
        .unwrap();
    let mut b = ceil_log2(max_delta).max(min_bits);
    enc.encode_bypass(b - min_bits, 2);
    let mut range = (1u32 << bd) - colors[0] as u32 - 1;
    for v in colors.windows(2) {
        let d = (v[1] - v[0]) as u32;
        enc.encode_bypass(d - 1, b);
        range -= d;
        b = b.min(ceil_log2(range));
    }
}

/// Emit `palette_y_size` and the colors against the neighbour `cache` (AVM
/// av2_get_palette_cache order: entries are flagged while colors remain; the
/// colors not taken from the cache follow as ascending literals).
pub(crate) fn emit_colors(enc: &mut RangeEncoder, p: &Palette, cache: &[u16], bd: u32) {
    let n = p.colors.len();
    enc.sym_palette_y_size(n);
    let mut taken = [false; MAX_COLORS];
    let mut idx = 0usize;
    for &c in cache {
        if idx >= n {
            break;
        }
        let hit = p.colors.iter().position(|&v| v == c).filter(|&k| !taken[k]);
        enc.encode_bypass(hit.is_some() as u32, 1);
        if let Some(k) = hit {
            taken[k] = true;
            idx += 1;
        }
    }
    let lits: Vec<u16> = p
        .colors
        .iter()
        .zip(taken)
        .filter(|&(_, t)| !t)
        .map(|(&c, _)| c)
        .collect();
    emit_literals(enc, &lits, bd);
}

/// Emit the color-index map (transpose bit, line flags, indices).
pub(crate) fn emit_map(enc: &mut RangeEncoder, p: &Palette) {
    let (w, h, n) = (p.w, p.h, p.colors.len());
    if w < 64 && h < 64 {
        enc.encode_bypass(p.transposed as u32, 1);
    }
    let (lines, len) = if p.transposed { (w, h) } else { (h, w) };
    let mut prev = 0usize;
    for l in 0..lines {
        let f = p.flags[l] as usize;
        enc.sym_palette_row(if l == 0 { 3 } else { prev }, f);
        if f != 2 {
            for a in 0..if f == 1 { 1 } else { len } {
                let (y, x) = if p.transposed { (a, l) } else { (l, a) };
                let color = p.map[y * w + x];
                if l == 0 && a == 0 {
                    enc.encode_uniform(color as u32, n as u32);
                } else {
                    let (ctx, order) = color_ctx(&p.map, w, y, x, n);
                    let sym = order[..n].iter().position(|&v| v == color).unwrap();
                    enc.sym_palette_y_color(n, ctx, sym);
                }
            }
        }
        prev = f;
    }
}

/// Optimal contiguous clusterings of a sorted weighted histogram: for every
/// k in 1..=kmax the cluster means (rounded) and the optimal SSE, via
/// F(k, j) = min F(k-1, i) + C(i, j).
fn dp_centers(hist: &[(f64, f64, f64)], kmax: usize, maxv: f64) -> Vec<(Vec<u16>, f64)> {
    // At most 64 distinct values (`search` bails beyond), at most MAX_COLORS sizes.
    let n = hist.len();
    assert!(n <= 64 && kmax <= MAX_COLORS);
    let mut pw = [0f64; 65];
    let mut ps = [0f64; 65];
    let mut pq = [0f64; 65];
    for (i, &(w, s, q)) in hist.iter().enumerate() {
        pw[i + 1] = pw[i] + w;
        ps[i + 1] = ps[i] + s;
        pq[i + 1] = pq[i] + q;
    }
    // Cost of bins i..j (half-open).
    let cost = |i: usize, j: usize| -> f64 {
        let w = pw[j] - pw[i];
        let s = ps[j] - ps[i];
        (pq[j] - pq[i]) - s * s / w
    };
    let kmax = kmax.min(n);
    let mut f = [[f64::INFINITY; 65]; MAX_COLORS + 1];
    let mut arg = [[0usize; 65]; MAX_COLORS + 1];
    f[0][0] = 0.0;
    for k in 1..=kmax {
        for j in k..=n {
            let mut best = f64::INFINITY;
            let mut bi = k - 1;
            for (i, &fi) in f[k - 1][..=n].iter().enumerate().take(j).skip(k - 1) {
                let c = fi + cost(i, j);
                if c < best {
                    best = c;
                    bi = i;
                }
            }
            f[k][j] = best;
            arg[k][j] = bi;
        }
    }
    (1..=kmax)
        .map(|k| {
            let mut centers = Vec::with_capacity(k);
            let mut j = n;
            for kk in (1..=k).rev() {
                let i = arg[kk][j];
                let w = pw[j] - pw[i];
                centers.push(((ps[j] - ps[i]) / w).round().clamp(0.0, maxv) as u16);
                j = i;
            }
            centers.reverse();
            centers.dedup();
            (centers, f[k][n].max(0.0))
        })
        .collect()
}

/// RD map for fixed colors in one scan direction: line by line, the cheapest of
/// copy-previous-line, one repeated index, or per-pixel greedy indices. `None`
/// once the running cost `sse + lambda * (head + bits)` reaches `bound`: it only
/// grows, so the map could not beat the palette costing `bound`.
#[allow(clippy::too_many_arguments)]
fn build_map(
    src: &[f32],
    w: usize,
    h: usize,
    colors: &[u16],
    transposed: bool,
    lambda: f32,
    head: f32,
    bound: f32,
) -> Option<(Vec<u8>, Vec<u8>, f32, f32)> {
    let b = bits();
    let n = colors.len();
    let cb = &b.color[n - 2];
    let (lines, len) = if transposed { (w, h) } else { (h, w) };
    let at = |l: usize, a: usize| if transposed { a * w + l } else { l * w + a };
    let d = |p: usize, c: usize| {
        let e = src[p] - colors[c] as f32;
        e * e
    };
    let mut map = vec![0u8; w * h];
    let mut flags = vec![0u8; lines];
    let (mut sse, mut nbits) = (0f32, 0f32);
    let mut line_buf = [0u8; 64];
    let line = &mut line_buf[..len];
    let mut prev = 0usize;
    for l in 0..lines {
        let rctx = if l == 0 { 3 } else { prev };
        let rb = &b.row[rctx];
        // First-index cost of a line (uniform at the block origin).
        let first_cost = |map: &[u8], c: usize| -> f32 {
            if l == 0 {
                uniform_bits(c, n)
            } else {
                let (y, x) = if transposed { (0, l) } else { (l, 0) };
                let (ctx, order) = color_ctx(map, w, y, x, n);
                cb[ctx][order[..n].iter().position(|&v| v as usize == c).unwrap()]
            }
        };
        // flag 2: copy the previous line.
        let mut best = (f32::INFINITY, 0f32, 0f32, 2u8);
        if l > 0 {
            let dist: f32 = (0..len)
                .map(|a| d(at(l, a), map[at(l - 1, a)] as usize))
                .sum();
            best = (dist + lambda * rb[2], dist, rb[2], 2);
        }
        // flag 1: one index for the whole line.
        for c in 0..n {
            let dist: f32 = (0..len).map(|a| d(at(l, a), c)).sum();
            let r = rb[1] + first_cost(&map, c);
            let j = dist + lambda * r;
            if j < best.0 {
                best = (j, dist, r, 1);
                line.fill(c as u8);
            }
        }
        // flag 0: greedy per-pixel indices.
        let (mut dist0, mut r0) = (0f32, rb[0]);
        for a in 0..len {
            let p = at(l, a);
            let (y, x) = (p / w, p % w);
            let mut bc = (f32::INFINITY, 0usize, 0f32, 0f32);
            if l == 0 && a == 0 {
                for c in 0..n {
                    let (dd, r) = (d(p, c), uniform_bits(c, n));
                    let j = dd + lambda * r;
                    if j < bc.0 {
                        bc = (j, c, dd, r);
                    }
                }
            } else {
                let (ctx, order) = color_ctx(&map, w, y, x, n);
                for (pos, &c) in order[..n].iter().enumerate() {
                    let (dd, r) = (d(p, c as usize), cb[ctx][pos]);
                    let j = dd + lambda * r;
                    if j < bc.0 {
                        bc = (j, c as usize, dd, r);
                    }
                }
            }
            map[p] = bc.1 as u8;
            dist0 += bc.2;
            r0 += bc.3;
        }
        if dist0 + lambda * r0 < best.0 {
            best = (dist0 + lambda * r0, dist0, r0, 0);
        } else {
            for a in 0..len {
                // `line` still holds the flag-1 winner's index.
                map[at(l, a)] = if best.3 == 1 {
                    line[a]
                } else {
                    map[at(l - 1, a)]
                };
            }
        }
        flags[l] = best.3;
        sse += best.1;
        nbits += best.2;
        if sse + lambda * (head + nbits) >= bound {
            return None;
        }
        prev = best.3 as usize;
    }
    Some((map, flags, sse, nbits))
}

/// The RD-best palette for a `w` x `h` luma source block (row-major), or None for
/// a flat block or one with more than 64 distinct values (as aom: such blocks
/// are natural texture, never palette winners). `lambda` is in pixel-SSE per bit,
/// the leaf RD's units.
pub(crate) fn search(src: &[f32], w: usize, h: usize, bd: u32, lambda: f32) -> Option<Palette> {
    // 8x8 palettes never pay once their bits are weighted (below): measured
    // neutral to drop, -13% instructions on screen content.
    if w.max(h) < 16 {
        return None;
    }
    let maxv = ((1u32 << bd) - 1) as f64;
    let mut vals_buf = [0u16; 64 * 64];
    let vals = &mut vals_buf[..src.len()];
    for (d, &v) in vals.iter_mut().zip(src) {
        *d = v.round().clamp(0.0, maxv as f32) as u16;
    }
    vals.sort_unstable();
    // Weighted histogram of the distinct values.
    let mut hist_buf = [(0f64, 0f64, 0f64); 64];
    let mut nh = 0usize;
    let pald = (crate::av2::rd::knob_or("PALD", 32.0) as usize).min(64);
    let mut cur = u16::MAX;
    for &v in vals.iter() {
        let f = v as f64;
        if v != cur || nh == 0 {
            if nh == pald {
                return None;
            }
            nh += 1;
            cur = v;
        }
        let e = &mut hist_buf[nh - 1];
        e.0 += 1.0;
        e.1 += f;
        e.2 += f * f;
    }
    let hist = &hist_buf[..nh];
    if hist.len() < 2 {
        return None;
    }
    let b = bits();
    let allow_t = w < 64 && h < 64;
    let lines = h.min(if allow_t { w } else { h });
    // Branch and bound over the sizes: the DP SSE plus the header and the cheapest
    // possible line flags bound a size's RD cost from below.
    let mut sizes: Vec<(f32, Vec<u16>, f32)> = dp_centers(hist, MAX_COLORS, maxv)
        .into_iter()
        .filter(|(c, _)| c.len() >= 2)
        .map(|(colors, sse)| {
            let head = b.mode[1] - b.mode[0]
                + b.size[colors.len() - 2]
                + literal_bits(&colors, bd) as f32
                + allow_t as u32 as f32;
            let min_line = b.row.iter().flatten().fold(f32::INFINITY, |m, &v| m.min(v));
            let lb = sse as f32 + lambda * (head + lines as f32 * min_line);
            (lb, colors, head)
        })
        .collect();
    sizes.sort_by(|x, y| x.0.total_cmp(&y.0));
    sizes.dedup_by(|x, y| x.1 == y.1);
    let mut best: Option<(f32, Palette)> = None;
    for (lb, colors, head) in sizes {
        if best.as_ref().is_some_and(|(bj, _)| lb >= *bj) {
            break;
        }
        for transposed in [false, true] {
            if transposed && !allow_t {
                continue;
            }
            let bound = match &best {
                Some((bj, _)) if crate::av2::rd::knob_or("PALBB", 1.0) > 0.0 => *bj,
                _ => f32::INFINITY,
            };
            let Some((map, flags, sse, mbits)) =
                build_map(src, w, h, &colors, transposed, lambda, head, bound)
            else {
                continue;
            };
            let j = sse + lambda * (head + mbits);
            if best.as_ref().is_none_or(|(bj, _)| j < *bj) {
                best = Some((
                    j,
                    Palette {
                        colors: colors.clone(),
                        w,
                        h,
                        map,
                        transposed,
                        flags,
                        bd,
                        bits: head + mbits,
                    },
                ));
            }
        }
    }
    // The leaf RD sees the palette bits x3: on flat-colour illustrations the
    // palette's SSE wins come from posterising anti-aliased edges, which SS2
    // punishes far more than SSE does (x1: mascot +5.2 / poster +3.3 SS2 BD vs no
    // palette; x2.5..x4 plateau at about -4.5 vs x1; x7 starts to cost).
    best.map(|(_, mut p)| {
        p.bits *= 3.0;
        p
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dp_centers_split_two_levels() {
        let mut hist = Vec::new();
        for &(v, n) in &[
            (10.0f64, 30.0f64),
            (11.0, 2.0),
            (200.0, 20.0),
            (201.0, 10.0),
        ] {
            hist.push((n, v * n, v * v * n));
        }
        let c = dp_centers(&hist, 4, 255.0);
        assert_eq!(c[1].0, vec![10, 200]);
        assert_eq!(c[3].0, vec![10, 11, 200, 201]);
        assert!(c[3].1 < 1e-9);
    }

    #[test]
    fn search_exact_two_color_block() {
        let src: Vec<f32> = (0..256)
            .map(|i| if (i % 16) < 3 { 20.0 } else { 220.0 })
            .collect();
        let p = search(&src, 16, 16, 8, 10.0).unwrap();
        assert_eq!(p.colors, vec![20, 220]);
        assert_eq!(p.pred(), src);
    }
}
