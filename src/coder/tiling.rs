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

// Frame padding, tile planning (`Tiling`) and tile crop / stitch helpers.

pub(crate) fn align8(n: usize) -> usize {
    (n + 7) & !7
}

/// Pad a `w`×`h` plane to `w8`×`h8` (≥ originals) by replicating the last
/// in-frame row/column. AV1's coded block grid is always 8-pixel aligned
/// (`MiCols = ((w+7)>>3)<<1`), so frames whose dimensions are not multiples of 8
/// are coded on the padded grid and the decoder crops back to the signaled
/// frame size. Edge replication keeps the (cropped-away) padding cheap to code.
pub(crate) fn pad_to_mult8<T: Copy>(src: &[T], w: usize, h: usize, w8: usize, h8: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(w8 * h8);
    for y in 0..h {
        let row = &src[y * w..y * w + w];
        out.extend_from_slice(row);
        out.resize(out.len() + (w8 - w), row[w - 1]);
    }
    for _ in h..h8 {
        out.extend_from_within((h - 1) * w8..h * w8);
    }
    out
}

/// Smallest `k` such that `(blk << k) >= target` (AV1 spec `tile_log2`).
fn tile_log2(blk: u32, target: u32) -> u32 {
    let mut k = 0;
    while (blk << k) < target {
        k += 1;
    }
    k
}

/// AV1 `increment_*_log2` bit sequence signaling `target` to a decoder that
/// starts at `min` and reads bits while its running value is `< max`: a `1` for
/// each step up, then a terminating `0` when `target < max` (at `max` the
/// decoder's loop ends on its own and reads no further bit).
fn increment_bits(min: u32, max: u32, target: u32) -> Vec<bool> {
    let mut v = Vec::new();
    let mut cur = min;
    while cur < max {
        if cur < target {
            v.push(true);
            cur += 1;
        } else {
            v.push(false);
            break;
        }
    }
    v
}

/// Full tiling decision: the chosen `(TileColsLog2, TileRowsLog2)` plus the
/// `increment_*_log2` bit sequences the frame header must emit to signal them.
pub(crate) struct Tiling {
    tcl: u32,
    trl: u32,
    cols_incr: Vec<bool>,
    rows_incr: Vec<bool>,
}

/// Pick a tiling for a frame of `sb_cols` x `sb_rows` superblocks.
fn plan_tiling(sb_cols: u32, sb_rows: u32, target_tiles: usize) -> Tiling {
    const MAX_TILE_WIDTH_SB: u32 = 4096 / 64; // 64
    const MAX_TILE_AREA_SB: u32 = (4096 * 2304) / (64 * 64); // 2304
    let min_log2_tile_cols = tile_log2(MAX_TILE_WIDTH_SB, sb_cols);
    let max_log2_tile_cols = tile_log2(1, sb_cols.min(64));
    let max_log2_tile_rows = tile_log2(1, sb_rows.min(64));
    let min_log2_tiles = min_log2_tile_cols.max(tile_log2(MAX_TILE_AREA_SB, sb_rows * sb_cols));

    // Start at the spec minimum.
    let mut tcl = min_log2_tile_cols.min(max_log2_tile_cols);
    let mut trl = min_log2_tiles.saturating_sub(tcl).min(max_log2_tile_rows);

    // Climb toward target_tiles, splitting whichever side currently has the
    // larger tiles so the grid stays balanced.
    let target = target_tiles.max(1) as u32;
    while (1u32 << (tcl + trl)) < target {
        let can_col = tcl < max_log2_tile_cols;
        let can_row = trl < max_log2_tile_rows;
        if !can_col && !can_row {
            break;
        }
        let col_span = sb_cols >> tcl; // SBs per tile column (approx)
        let row_span = sb_rows >> trl; // SBs per tile row (approx)
        if can_col && (!can_row || col_span >= row_span) {
            tcl += 1;
        } else {
            trl += 1;
        }
    }

    let cols_incr = increment_bits(min_log2_tile_cols, max_log2_tile_cols, tcl);
    // The decoder derives its row minimum from the (now decoded) TileColsLog2.
    let min_log2_tile_rows = min_log2_tiles.saturating_sub(tcl);
    let rows_incr = increment_bits(min_log2_tile_rows, max_log2_tile_rows, trl);
    Tiling {
        tcl,
        trl,
        cols_incr,
        rows_incr,
    }
}

/// Uniform-spacing tile start offsets (in SB units), matching the decoder's
/// `for (startSb = 0; startSb < sbs; startSb += sizeSb)` loop. The returned vec
/// has one entry per tile; the implied end of tile `i` is `starts[i+1]` (or
/// `sbs` for the last). The tile count may be **less** than `1 << log2`.
fn tile_starts_sb(sbs: u32, log2: u32) -> Vec<u32> {
    let size_sb = sbs.div_ceil(1 << log2);
    let mut starts = Vec::new();
    let mut s = 0;
    while s < sbs {
        starts.push(s);
        s += size_sb;
    }
    starts
}

fn crop_plane<T: Copy>(
    src: &[T],
    full_w: usize,
    x0: usize,
    y0: usize,
    tw: usize,
    th: usize,
) -> Vec<T> {
    let mut out = Vec::with_capacity(tw * th);
    for r in 0..th {
        let s = (y0 + r) * full_w + x0;
        out.extend_from_slice(&src[s..s + tw]);
    }
    out
}

fn stitch_plane(
    dst: &mut [u16],
    full_w: usize,
    x0: usize,
    y0: usize,
    tile: &[u16],
    tw: usize,
    th: usize,
) {
    for r in 0..th {
        let d = (y0 + r) * full_w + x0;
        dst[d..d + tw].copy_from_slice(&tile[r * tw..(r + 1) * tw]);
    }
}
