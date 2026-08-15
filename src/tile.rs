/*
 * // Copyright (c) Radzivon Bartoshyk 6/2026. All rights reserved.
 * //
 * // Redistribution and use in source and binary forms, with or without modification,
 * // are permitted provided that the following conditions are met:
 * //
 * // 1.  Redistributions of source code must retain the above copyright notice, this
 * // list of conditions and the following disclaimer.
 * //
 * // 2.  Redistributions in binary form must reproduce the above copyright notice,
 * // this list of conditions and the following disclaimer in the documentation
 * // and/or other materials provided with the distribution.
 * //
 * // 3.  Neither the name of the copyright holder nor the names of its
 * // contributors may be used to endorse or promote products derived from
 * // this software without specific prior written permission.
 * //
 * // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

use crate::Speed;
use crate::coder::Cdfs;
use crate::coefs_ctx::encode_coefs;
use crate::cost::coef_rate_bits;
use crate::intrapred::{
    CFL_ALPHA_CDF, CFL_PRED, CFL_SIGN_CDF, INTRA_MODE_CTX, cfl_ac_444, cfl_best_alpha,
    cfl_pred_pixel, intra_predict_nd_ad_i16, palette_pred,
};
use crate::msac_enc::Writer;
use crate::skip_tables::SKIP_CTX;
use crate::tables::*;
use crate::util::dirty_log2f;
use crate::wht::levels_from_resid;

/// DC prediction for a 4×4 block at pixel origin (ox, oy).
/// Reads from the source plane directly (lossless: recon == src).
fn dc_pred_4x4(plane: &[i16], stride: usize, ox: usize, oy: usize, base: i32) -> i16 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let s = plane[(oy - 1) * stride + ox..][..4]
                .iter()
                .map(|&v| v as i32)
                .sum::<i32>()
                + plane[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(4)
                    .map(|&v| v as i32)
                    .sum::<i32>();
            ((s + 4) >> 3) as i16
        }
        (true, false) => {
            let s = plane[(oy - 1) * stride + ox..][..4]
                .iter()
                .map(|&v| v as i32)
                .sum::<i32>();
            ((s + 2) >> 2) as i16
        }
        (false, true) => {
            let s = plane[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(4)
                .map(|&v| v as i32)
                .sum::<i32>();
            ((s + 2) >> 2) as i16
        }
        (false, false) => base as i16,
    }
}

/// dav1d `sm_weights` for n=4 (SMOOTH predictors).
static SM4: [i32; 4] = [255, 149, 85, 64];

/// Every AV1 luma/chroma intra predictor eligible for the square lossless
/// blocks emitted here, in CDF symbol order: DC, the eight directionals,
/// SMOOTH variants, and PAETH.
static LL_MODES: [usize; 13] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
// The decoder-verified BLOCK_4X4 subset. Directional edge extension and the
// two one-axis smooth variants are kept out of this special geometry.
static LL4_MODES: [usize; 3] = [0, 9, 12];

#[derive(Clone)]
struct LumaPalette {
    colors: Vec<i32>,
    map: Vec<u8>,
    packed_map: Vec<u8>,
    width: usize,
    height: usize,
}

/// Build an exact AV1 luma palette for a complete 8..64-pixel coding block.
/// Lossless palette prediction is only selected when every source sample is
/// represented exactly; the residual is consequently all zero.
fn exact_luma_palette(
    plane: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
) -> Option<LumaPalette> {
    if width < 8 || height < 8 || width > 64 || height > 64 {
        return None;
    }
    let max = (1i32 << bit_depth) - 1;
    let mut colors = Vec::with_capacity(8);
    for y in 0..height {
        for &sample in &plane[(py + y) * stride + px..][..width] {
            let sample = i32::from(sample);
            if !(0..=max).contains(&sample) {
                return None;
            }
            if !colors.contains(&sample) {
                if colors.len() == 8 {
                    return None;
                }
                colors.push(sample);
            }
        }
    }
    if colors.len() < 2 {
        return None;
    }
    colors.sort_unstable();

    let mut map = Vec::with_capacity(width * height);
    let mut packed_map = vec![0u8; width.div_ceil(2) * height];
    let packed_stride = width.div_ceil(2);
    for y in 0..height {
        for x in 0..width {
            let sample = i32::from(plane[(py + y) * stride + px + x]);
            let index = colors.binary_search(&sample).unwrap() as u8;
            map.push(index);
            let packed = &mut packed_map[y * packed_stride + x / 2];
            if x & 1 == 0 {
                *packed |= index;
            } else {
                *packed |= index << 4;
            }
        }
    }
    Some(LumaPalette {
        colors,
        map,
        packed_map,
        width,
        height,
    })
}

fn uniform_symbol_bits(n: usize, value: usize) -> f32 {
    debug_assert!(n > 0 && value < n);
    let bits = usize::BITS as usize - (n - 1).leading_zeros() as usize;
    let cutoff = (1usize << bits) - n;
    if value < cutoff {
        (bits - 1) as f32
    } else {
        bits as f32
    }
}

fn palette_color_bits(colors: &[i32], cache: &[i32], bit_depth: u8) -> f32 {
    let mut bits = 0.0f32;
    let mut in_cache = 0usize;
    for &cached in cache {
        if in_cache == colors.len() {
            break;
        }
        bits += 1.0;
        in_cache += usize::from(colors.binary_search(&cached).is_ok());
    }
    let out: Vec<u32> = colors
        .iter()
        .filter(|color| cache.binary_search(color).is_err())
        .map(|&color| color as u32)
        .collect();
    if out.is_empty() {
        return bits;
    }
    bits += bit_depth as f32;
    if out.len() == 1 {
        return bits;
    }
    let max_delta = out.array_windows::<2>().map(|v| v[1] - v[0]).max().unwrap();
    let min_bits = bit_depth - 3;
    let mut delta_bits = ceil_log2(max_delta).max(min_bits);
    bits += 2.0;
    let mut range = (1u32 << bit_depth) - out[0] - 1;
    for pair in out.array_windows::<2>() {
        bits += f32::from(delta_bits);
        range -= pair[1] - pair[0];
        delta_bits = delta_bits.min(ceil_log2(range));
    }
    bits
}

fn palette_estimated_bits(
    palette: &LumaPalette,
    bit_depth: u8,
    cache: &[i32],
    mode_ctx: usize,
) -> f32 {
    let pixels = palette.width * palette.height;
    let size = palette.colors.len();

    let bsize_ctx = palette_bsize_ctx(palette.width);
    let mode_bits = raw_symbol_cost(&[palette_y_mode_raw(bsize_ctx, mode_ctx)], 1)
        + raw_symbol_cost(palette_y_size_raw(bsize_ctx), size - 2);
    let color_bits = palette_color_bits(&palette.colors, cache, bit_depth);

    // The first index is uniform. Remaining indices use AV1's spatial palette
    // contexts and therefore can be much cheaper than fixed-width coding.
    let mut map_bits = uniform_symbol_bits(size, palette.map[0] as usize);
    for diagonal in 1..palette.width + palette.height - 1 {
        let first_x = diagonal.min(palette.width - 1);
        let last_x = diagonal.saturating_sub(palette.height - 1);
        for x in (last_x..=first_x).rev() {
            let y = diagonal - x;
            let (ctx, symbol) = palette_color_ctx(&palette.map, palette.width, y, x, size);
            map_bits += raw_symbol_cost(palette_y_color_raw(size, ctx), symbol);
        }
    }
    let zero_txb_bits = (pixels / 16) as f32;
    mode_bits + color_bits + map_bits + zero_txb_bits
}

fn palette_best_case_bits(palette: &LumaPalette, bit_depth: u8) -> f32 {
    (0..=2)
        .map(|mode_ctx| palette_estimated_bits(palette, bit_depth, &palette.colors, mode_ctx))
        .fold(f32::INFINITY, f32::min)
}

#[inline]
fn raw_symbol_cost(raw: &[u16], symbol: usize) -> f32 {
    let low = if symbol == 0 { 0 } else { raw[symbol - 1] };
    let high = raw.get(symbol).copied().unwrap_or(32768);
    -dirty_log2f((high - low).max(1) as f32 / 32768.0)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntrabcMatch {
    ref_x: usize,
    ref_y: usize,
    mv: (i16, i16),
}

#[derive(Clone, Copy, Default)]
struct IntrabcIndexEntry {
    hash: u32,
    pos: u32,
}

/// Compact frame-wide index for IntraBC's exact block matcher.  A 4x4 prefix
/// is sufficient as the lookup key for every legal IntraBC block size; every
/// hit is still verified over the complete block and all coded planes.  The
/// upper 16 hash bits select a bucket while the full hash rejects collisions.
/// This costs eight bytes per integer-pixel origin and, unlike a HashMap, has
/// no per-entry allocation or pointer overhead.
struct IntrabcIndex {
    width: usize,
    entries: Vec<IntrabcIndexEntry>,
    offsets: Vec<u32>,
}

impl IntrabcIndex {
    fn fingerprint(planes: &[&[i16]], stride: usize, x: usize, y: usize) -> u32 {
        // Include every coded plane in the key.  Luma-only fingerprints create
        // huge false-match lists in dark UI regions whose chroma still varies.
        // Complete equality is nevertheless verified after lookup.
        let mut hash = 0x811c_9dc5u32;
        for plane in planes {
            for (dx, dy) in [(0usize, 0usize), (3, 0), (0, 3), (3, 3)] {
                hash ^= plane[(y + dy) * stride + x + dx] as u16 as u32;
                hash = hash.wrapping_mul(0x0100_0193);
            }
            hash ^= 0x9e37_79b9;
            hash = hash.rotate_left(7);
        }
        hash
    }

    fn new(planes: &[&[i16]], width: usize, height: usize) -> Self {
        const BUCKETS: usize = 1 << 16;
        if width < 4
            || height < 4
            || width
                .checked_mul(height)
                .is_none_or(|n| n > u32::MAX as usize)
        {
            return Self {
                width,
                entries: Vec::new(),
                offsets: vec![0; BUCKETS + 1],
            };
        }
        let origins_w = width - 3;
        let origins_h = height - 3;
        let count = origins_w * origins_h;
        let mut offsets = vec![0u32; BUCKETS + 1];
        let mut hashes = Vec::with_capacity(count);
        for y in 0..origins_h {
            for x in 0..origins_w {
                let hash = Self::fingerprint(planes, width, x, y);
                hashes.push(hash);
                offsets[(hash >> 16) as usize + 1] += 1;
            }
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursor = offsets[..BUCKETS].to_vec();
        let mut entries = vec![IntrabcIndexEntry::default(); count];
        for (origin, hash) in hashes.into_iter().enumerate() {
            let (x, y) = (origin % origins_w, origin / origins_w);
            let bucket = (hash >> 16) as usize;
            let slot = cursor[bucket] as usize;
            entries[slot] = IntrabcIndexEntry {
                hash,
                pos: (y * width + x) as u32,
            };
            cursor[bucket] += 1;
        }
        Self {
            width,
            entries,
            offsets,
        }
    }

    fn candidates_in(
        &self,
        hash: u32,
        min_pos: usize,
        max_pos: usize,
    ) -> impl DoubleEndedIterator<Item = (usize, usize)> + '_ {
        let bucket = (hash >> 16) as usize;
        let start = self.offsets[bucket] as usize;
        let end = self.offsets[bucket + 1] as usize;
        let entries = &self.entries[start..end];
        // Entries retain raster order within each bucket.  Trim future frame
        // rows (and the undecoded right side of the current SB row) before
        // testing full hashes; flat screen content otherwise walked millions
        // of future equal-hash entries for every partition candidate.
        let lo = entries.partition_point(|entry| entry.pos < min_pos as u32);
        let hi = entries.partition_point(|entry| entry.pos <= max_pos as u32);
        entries[lo..hi]
            .iter()
            .filter(move |entry| entry.hash == hash)
            .map(|entry| {
                let pos = entry.pos as usize;
                (pos % self.width, pos / self.width)
            })
    }
}

/// Find an exact block match in the portion of the tile that AV1 has decoded.
/// The default DV is tried first, followed by indexed reverse-raster lookup at
/// every integer-pixel position. Reverse raster favours recently decoded data;
/// the index makes the work depend on hash hits rather than decoded-area size.
#[allow(clippy::too_many_arguments)]
fn find_exact_intrabc(
    planes: &[&[i16]],
    stride: usize,
    px: usize,
    py: usize,
    size: usize,
    width: usize,
    height: usize,
    index: &IntrabcIndex,
) -> Option<IntrabcMatch> {
    if px + size > width || py + size > height {
        return None;
    }
    let sbx = px / 64 * 64;
    let sby = py / 64 * 64;
    let exact = |ref_x: usize, ref_y: usize| {
        planes.iter().all(|plane| {
            (0..size).all(|y| {
                plane[(py + y) * stride + px..][..size]
                    == plane[(ref_y + y) * stride + ref_x..][..size]
            })
        })
    };
    let legal = |ref_x: usize, ref_y: usize| {
        ref_x + size <= width
            && ref_y + size <= height
            && ref_y + size <= sby + 64
            && (ref_y + size <= sby || ref_x + size <= sbx)
    };
    let make = |ref_x: usize, ref_y: usize| {
        let dy = (ref_y as isize - py as isize) * 8;
        let dx = (ref_x as isize - px as isize) * 8;
        i16::try_from(dy)
            .ok()
            .zip(i16::try_from(dx).ok())
            .map(|(dy, dx)| IntrabcMatch {
                ref_x,
                ref_y,
                mv: (dy, dx),
            })
    };

    let default = if py < 64 {
        px.checked_sub(320).map(|x| (x, py))
    } else if py.is_multiple_of(64) {
        Some((px, py - 64))
    } else {
        None
    };
    if let Some((x, y)) = default
        && legal(x, y)
        && exact(x, y)
    {
        return make(x, y);
    }
    let wanted = IntrabcIndex::fingerprint(planes, stride, px, py);
    // A highly ambiguous 4x4 prefix is not a useful general block-search key.
    // Bound expensive complete-block comparisons; unique and normally
    // repeated keys still reach arbitrary positions without a spatial horizon.
    const MAX_MATCH_CANDIDATES: usize = 16;
    let mut candidates_checked = 0usize;
    // First search decoded superblocks to the left in the current SB row.
    // Querying each possible y separately excludes the undecoded right side
    // in the index itself rather than linearly skipping it.
    if sbx >= size {
        for ref_y in (sby..=sby + 64 - size).rev() {
            let min_pos = ref_y * width;
            let max_pos = min_pos + sbx - size;
            for (ref_x, ref_y) in index.candidates_in(wanted, min_pos, max_pos).rev() {
                candidates_checked += 1;
                if candidates_checked > MAX_MATCH_CANDIDATES {
                    return None;
                }
                if legal(ref_x, ref_y)
                    && exact(ref_x, ref_y)
                    && let Some(found) = make(ref_x, ref_y)
                {
                    return Some(found);
                }
            }
        }
    }
    // Then search all complete preceding SB rows.  This is one contiguous
    // raster prefix, so no future-frame entries are visited.
    if sby >= size {
        let max_pos = (sby - size) * width + width - size;
        for (ref_x, ref_y) in index.candidates_in(wanted, 0, max_pos).rev() {
            candidates_checked += 1;
            if candidates_checked > MAX_MATCH_CANDIDATES {
                return None;
            }
            if legal(ref_x, ref_y)
                && exact(ref_x, ref_y)
                && let Some(found) = make(ref_x, ref_y)
            {
                return Some(found);
            }
        }
    }
    None
}

#[inline]
fn intrabc_default_mv(py: usize) -> (i16, i16) {
    if py < 64 {
        (0, -2560) // 320 pixels left, in 1/8-pixel MV units
    } else {
        (-512, 0) // 64 pixels up
    }
}

/// Conservative planning cost for the IntraBC flag, skip flag and integer DV
/// residual.  Coding uses the live spatial-stack predictor; planning does not
/// yet have that mutable state, so the normative fallback DV is the safe rate
/// proxy.  Unlike the old constant, arbitrary DVs now pay for their residual.
fn intrabc_estimated_bits(candidate: IntrabcMatch, py: usize) -> f32 {
    let pred = intrabc_default_mv(py);
    let residual_pixels = ((i32::from(candidate.mv.0) - i32::from(pred.0)).unsigned_abs()
        + (i32::from(candidate.mv.1) - i32::from(pred.1)).unsigned_abs())
        as f32
        / 8.0;
    8.0 + dirty_log2f(residual_pixels.max(1.0)) * 2.0
}

#[inline]
fn intrabc_rdo_margin(size: usize) -> f32 {
    // Small copies amortize the DV syntax poorly, and their partition-time
    // fallback predictor is more likely to differ from the live spatial stack.
    // Require a real win instead of replacing ordinary intra on a near tie.
    match size {
        8 => 256.0,
        16 => 128.0,
        32 => 64.0,
        _ => 32.0,
    }
}

/// Resolve a predictor from the spatial locations scanned by AV1's reference-
/// MV stack. We only use a spatial predictor when the mandatory nearest row or
/// column contributes it and every other possible contributor agrees.
/// Optional-only candidates are safe only when they equal the normative
/// fallback: depending on partition z-order the decoder may not see them.
/// Ambiguous stacks are rejected instead of risking a DV residual desync.
fn intrabc_mv_predictor(st: &LlState, px: usize, py: usize, size: usize) -> Option<(i16, i16)> {
    let (x4, y4, n4) = (px / 4, py / 4, size / 4);
    let (w4, h4) = (st.w / 4, st.h / 4);
    let fallback = intrabc_default_mv(py);
    let mut found = None;
    let mut nearest_found = false;
    let mut ambiguous = false;
    let mut visit = |x: usize, y: usize, nearest: bool| {
        if x >= w4 || y >= h4 {
            return;
        }
        if let Some(mv) = st.ibc_mv[y * w4 + x] {
            nearest_found |= nearest;
            if found.is_some_and(|old| old != mv) {
                ambiguous = true;
            } else {
                found = Some(mv);
            }
        }
    };

    if let Some(y) = y4.checked_sub(1) {
        for x in x4..(x4 + n4).min(w4) {
            visit(x, y, true);
        }
    }
    if let Some(x) = x4.checked_sub(1) {
        for y in y4..(y4 + n4).min(h4) {
            visit(x, y, true);
        }
    }

    if let (Some(x), Some(y)) = (x4.checked_sub(1), y4.checked_sub(1)) {
        visit(x, y, false);
    }
    if let Some(y) = y4.checked_sub(1) {
        visit(x4 + n4, y, false);
    }
    for offset in [3usize, 5] {
        if let Some(y) = y4.checked_sub(offset) {
            for x in (x4 + 1)..(x4 + n4).min(w4) {
                visit(x, y, false);
            }
        }
        if let Some(x) = x4.checked_sub(offset) {
            for y in (y4 + 1)..(y4 + n4).min(h4) {
                visit(x, y, false);
            }
        }
    }

    if ambiguous {
        None
    } else if nearest_found {
        found
    } else if found.is_none_or(|mv| mv == fallback) {
        Some(fallback)
    } else {
        None
    }
}

/// Build the 4x4 intra reference edges (top[4], left[4], corner) from the
/// source plane, matching dav1d's neighbor construction (recon == src in
/// lossless). Non-directional modes need no top-right / bottom-left extension.
#[inline]
fn edges_4x4(
    plane: &[i16],
    stride: usize,
    ox: usize,
    oy: usize,
    base: i32,
) -> ([i32; 4], [i32; 4], i32) {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let mut top = [0i32; 4];
    let mut left = [0i32; 4];
    if have_top {
        for (t, &v) in top.iter_mut().zip(plane[(oy - 1) * stride + ox..].iter()) {
            *t = v as i32;
        }
    } else {
        let fill = if have_left {
            plane[oy * stride + ox - 1] as i32
        } else {
            base - 1
        };
        top = [fill; 4];
    }
    if have_left {
        for (lf, &v) in left
            .iter_mut()
            .zip(plane[oy * stride + ox - 1..].iter().step_by(stride))
        {
            *lf = v as i32;
        }
    } else {
        let fill = if have_top {
            plane[(oy - 1) * stride + ox] as i32
        } else {
            base + 1
        };
        left = [fill; 4];
    }
    let corner = if have_left {
        if have_top {
            plane[(oy - 1) * stride + ox - 1] as i32
        } else {
            plane[oy * stride + ox - 1] as i32
        }
    } else if have_top {
        plane[(oy - 1) * stride + ox] as i32
    } else {
        base
    };
    (top, left, corner)
}

/// Predict a 4x4 block with a non-directional `mode`, writing 16 raster samples.
/// Mirrors dav1d's (and the lossy path's) bit-exact predictor formulas.
fn predict_4x4(
    mode: usize,
    plane: &[i16],
    stride: usize,
    ox: usize,
    oy: usize,
    out: &mut [i32; 16],
    base: i32,
) {
    if mode == 0 {
        let d = dc_pred_4x4(plane, stride, ox, oy, base) as i32;
        *out = [d; 16];
        return;
    }
    let (top, left, corner) = edges_4x4(plane, stride, ox, oy, base);
    match mode {
        1 => {
            // V_PRED
            for orow in out.as_chunks_mut::<4>().0.iter_mut() {
                orow.copy_from_slice(&top);
            }
        }
        2 => {
            // H_PRED
            for (orow, &lv) in out.as_chunks_mut::<4>().0.iter_mut().zip(left.iter()) {
                orow.iter_mut().for_each(|o| *o = lv);
            }
        }
        9 => {
            // SMOOTH
            let (right, bottom) = (top[3], left[3]);
            for ((orow, &smy), &lv) in out
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(SM4.iter())
                .zip(left.iter())
            {
                for (o, (&tv, &smx)) in orow.iter_mut().zip(top.iter().zip(SM4.iter())) {
                    let p = smy * tv + (256 - smy) * bottom + smx * lv + (256 - smx) * right;
                    *o = (p + 256) >> 9;
                }
            }
        }
        10 => {
            // SMOOTH_V
            let bottom = left[3];
            for (orow, &smy) in out.as_chunks_mut::<4>().0.iter_mut().zip(SM4.iter()) {
                for (o, &tv) in orow.iter_mut().zip(top.iter()) {
                    let p = smy * tv + (256 - smy) * bottom;
                    *o = (p + 128) >> 8;
                }
            }
        }
        11 => {
            // SMOOTH_H
            let right = top[3];
            for (orow, &lv) in out.as_chunks_mut::<4>().0.iter_mut().zip(left.iter()) {
                for (o, &smx) in orow.iter_mut().zip(SM4.iter()) {
                    let p = smx * lv + (256 - smx) * right;
                    *o = (p + 128) >> 8;
                }
            }
        }
        12 => {
            // PAETH
            for (orow, &lv) in out.as_chunks_mut::<4>().0.iter_mut().zip(left.iter()) {
                for (o, &tv) in orow.iter_mut().zip(top.iter()) {
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
        _ => unreachable!("predict_4x4 mode {}", mode),
    }
}

#[allow(clippy::too_many_arguments)]
fn predict_lossless_4x4(
    mode: usize,
    angle_delta: i32,
    plane: &[i16],
    stride: usize,
    ox: usize,
    oy: usize,
    block_x: usize,
    block_y: usize,
    block_size: usize,
    out: &mut [i32; 16],
    base: i32,
    bit_depth: u8,
) {
    if (1..=8).contains(&mode) {
        let tx = (ox - block_x) / 4;
        let ty = (oy - block_y) / 4;
        let n = block_size / 4;
        let (outer_tr, outer_bl) = lossless_leaf_edge_flags(block_x, block_y, block_size);
        let have_tr = tx + 1 < n || (ty == 0 && outer_tr);
        let have_bl = tx == 0 && (ty + 1 < n || outer_bl);
        intra_predict_nd_ad_i16(
            mode,
            angle_delta,
            plane,
            stride,
            ox,
            oy,
            4,
            4,
            have_tr,
            have_bl,
            stride,
            plane.len() / stride,
            false,
            out,
            bit_depth,
        );
    } else {
        predict_4x4(mode, plane, stride, ox, oy, out, base);
    }
}

/// Return AV1's top-right and bottom-left availability flags for a square leaf
/// in the 64x64 superblock partition tree.  These flags describe coding-order
/// availability, not merely whether the referenced pixels are inside the
/// frame: a geometrically present bottom-left block may not have been decoded
/// yet.  This is the square SPLIT/NONE subset of dav1d's `intra_edge_tree`.
fn lossless_leaf_edge_flags(block_x: usize, block_y: usize, block_size: usize) -> (bool, bool) {
    debug_assert!(matches!(block_size, 4 | 8 | 16 | 32 | 64));
    let mut top_has_right = true;
    let mut left_has_bottom = false;
    let mut size = 64usize;
    let mut x = block_x & 63;
    let mut y = block_y & 63;

    while size > block_size {
        let half = size / 2;
        let right = x >= half;
        let bottom = y >= half;
        let quadrant = usize::from(right) + 2 * usize::from(bottom);
        let parent_tr = top_has_right;
        let parent_bl = left_has_bottom;
        top_has_right = !(quadrant == 3 || (quadrant == 1 && !parent_tr));
        left_has_bottom = quadrant == 0 || (quadrant == 2 && parent_bl);
        if right {
            x -= half;
        }
        if bottom {
            y -= half;
        }
        size = half;
    }
    (top_has_right, left_has_bottom)
}

/// Partition context for a node, from the above/left partition-context arrays
/// (absolute 8px-unit indexing). Matches dav1d `get_partition_ctx`.
fn get_partition_ctx(a: &[u8], l: &[u8], bl: usize, x8: usize, y8: usize) -> usize {
    let sh = 4 - bl;
    ((a[x8] >> sh) & 1) as usize + ((((l[y8] >> sh) & 1) as usize) << 1)
}

/// Probability (0..32768) for the binary `is_split` decision at a frame edge.
/// `top` → `gather_top_partition_prob` (have_h only), else
/// `gather_left_partition_prob` (have_v only). Operates directly on the icdf
/// partition CDF.
fn gather_split_prob(cdf: &[u16], top: bool) -> u16 {
    let i = |s: usize| cdf[s] as i32;
    let out = if top {
        (i(1) - i(4)) + i(5) + (i(8) - i(7))
    } else {
        (i(0) - i(1)) + (i(2) - i(6)) + (i(7) - i(8))
    };
    out.clamp(1, 32767) as u16
}

/// Encode one plane of a square block of `n_tx`×`n_tx` TX_4×4 blocks at pixel
/// origin `(bx, by)` within a `stride`-wide frame. Above/left coefficient
/// context arrays are absolute 4px-unit indexed (frame-spanning).
#[allow(clippy::too_many_arguments)]
fn encode_plane_block(
    w: &mut Writer,
    cdfs: &mut Cdfs,
    plane: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    n_tx: usize,
    chroma: bool,
    mode: usize,
    angle_delta: i32,
    base: i32,
    bit_depth: u8,
    a: &mut [u8],
    l: &mut [u8],
    palette: Option<&LumaPalette>,
    cfl: Option<(&[i16], i32)>,
) {
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];
    let palette_predicted = palette.map(|palette| {
        debug_assert_eq!(palette.width, n_tx * 4);
        debug_assert_eq!(palette.height, n_tx * 4);
        let mut out = vec![0i32; palette.width * palette.height];
        palette_pred(
            &mut out,
            palette.width,
            &palette.colors,
            &palette.packed_map,
            palette.width,
            palette.height,
        );
        out
    });
    for ty in 0..n_tx {
        for tx in 0..n_tx {
            let ox = bx + tx * 4;
            let oy = by + ty * 4;
            let (ax, ly) = (ox / 4, oy / 4);

            let skip_ctx = if n_tx == 1 && !chroma {
                // TX size equals the BLOCK_4X4 prediction block.
                0
            } else if n_tx == 1 {
                // Chroma one-block path (`not_one_blk = 0`).
                7 + (a[ax] != 0x40) as usize + (l[ly] != 0x40) as usize
            } else if !chroma {
                let av = (a[ax] & 0x3F).min(4) as usize;
                let lv = (l[ly] & 0x3F).min(4) as usize;
                SKIP_CTX[av][lv] as usize
            } else {
                10 + (a[ax] != 0x40) as usize + (l[ly] != 0x40) as usize
            };
            let t = (a[ax] >> 6) as i32 + (l[ly] >> 6) as i32;
            let s = t - 2;
            let dc_sign_ctx = ((s != 0) as usize) + ((s > 0) as usize);

            if let Some(block_pred) = palette_predicted.as_ref() {
                for row in 0..4 {
                    let src = &block_pred[(ty * 4 + row) * n_tx * 4 + tx * 4..][..4];
                    pred[row * 4..row * 4 + 4].copy_from_slice(src);
                }
            } else if let Some((luma, alpha)) = cfl {
                // Lossless always uses TX_4X4. CfL is invoked for each chroma
                // transform block, so both the DC base and luma-AC mean are
                // local to this 4x4 (the alpha remains shared by the leaf).
                predict_lossless_4x4(
                    0,
                    0,
                    plane,
                    stride,
                    ox,
                    oy,
                    bx,
                    by,
                    n_tx * 4,
                    &mut pred,
                    base,
                    bit_depth,
                );
                let mut luma_rec = [0i32; 16];
                for row in 0..4 {
                    for x in 0..4 {
                        luma_rec[row * 4 + x] = i32::from(luma[(oy + row) * stride + ox + x]);
                    }
                }
                let mut ac = [0i32; 16];
                cfl_ac_444(&luma_rec, 4, 4, &mut ac);
                for i in 0..16 {
                    pred[i] = cfl_pred_pixel(pred[i], ac[i], alpha, bit_depth);
                }
            } else {
                predict_lossless_4x4(
                    mode,
                    angle_delta,
                    plane,
                    stride,
                    ox,
                    oy,
                    bx,
                    by,
                    n_tx * 4,
                    &mut pred,
                    base,
                    bit_depth,
                );
            }
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let srow = &plane[(oy + ry) * stride + ox..];
                for (r, (&srv, &pv)) in rrow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                    *r = srv as i32 - pv;
                }
            }
            levels_from_resid(&mut resid);
            let res_ctx = encode_coefs(w, cdfs, chroma, &resid, skip_ctx, dc_sign_ctx);
            a[ax] = res_ctx;
            l[ly] = res_ctx;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_plane_rect_block(
    wr: &mut Writer,
    cdfs: &mut Cdfs,
    plane: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    width: usize,
    height: usize,
    chroma: bool,
    mode: usize,
    base: i32,
    a: &mut [u8],
    l: &mut [u8],
) {
    let (nw, nh) = (width / 4, height / 4);
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];
    for ty in 0..nh {
        for tx in 0..nw {
            let (ox, oy) = (bx + tx * 4, by + ty * 4);
            let (ax, ly) = (ox / 4, oy / 4);
            let skip_ctx = if nw * nh == 1 && !chroma {
                0
            } else if nw * nh == 1 {
                7 + (a[ax] != 0x40) as usize + (l[ly] != 0x40) as usize
            } else if !chroma {
                let av = (a[ax] & 0x3f).min(4) as usize;
                let lv = (l[ly] & 0x3f).min(4) as usize;
                SKIP_CTX[av][lv] as usize
            } else {
                10 + (a[ax] != 0x40) as usize + (l[ly] != 0x40) as usize
            };
            let s = (a[ax] >> 6) as i32 + (l[ly] >> 6) as i32 - 2;
            let dc_sign_ctx = (s != 0) as usize + (s > 0) as usize;
            predict_4x4(mode, plane, stride, ox, oy, &mut pred, base);
            for row in 0..4 {
                for x in 0..4 {
                    let i = row * 4 + x;
                    resid[i] = i32::from(plane[(oy + row) * stride + ox + x]) - pred[i];
                }
            }
            levels_from_resid(&mut resid);
            let res_ctx = encode_coefs(wr, cdfs, chroma, &resid, skip_ctx, dc_sign_ctx);
            a[ax] = res_ctx;
            l[ly] = res_ctx;
        }
    }
}

/// Residual-rate proxy for one chroma plane under lossless CfL. The luma AC
/// normalization and DC base intentionally follow the decoder per TX_4X4.
#[allow(clippy::too_many_arguments)]
fn plane_cfl_bits(
    alpha: i32,
    luma: &[i16],
    chroma: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    n_tx: usize,
    base: i32,
    bit_depth: u8,
) -> f32 {
    let mut bits = 0.0f32;
    let mut dc = [0i32; 16];
    let mut luma_rec = [0i32; 16];
    let mut ac = [0i32; 16];
    let mut resid = [0i32; 16];
    for ty in 0..n_tx {
        for tx in 0..n_tx {
            let (ox, oy) = (bx + tx * 4, by + ty * 4);
            predict_lossless_4x4(
                0,
                0,
                chroma,
                stride,
                ox,
                oy,
                bx,
                by,
                n_tx * 4,
                &mut dc,
                base,
                bit_depth,
            );
            for row in 0..4 {
                for x in 0..4 {
                    luma_rec[row * 4 + x] = i32::from(luma[(oy + row) * stride + ox + x]);
                }
            }
            cfl_ac_444(&luma_rec, 4, 4, &mut ac);
            let mut any = false;
            for row in 0..4 {
                for x in 0..4 {
                    let i = row * 4 + x;
                    let pred = cfl_pred_pixel(dc[i], ac[i], alpha, bit_depth);
                    resid[i] = i32::from(chroma[(oy + row) * stride + ox + x]) - pred;
                    any |= resid[i] != 0;
                }
            }
            if !any {
                bits += 1.0;
                continue;
            }
            levels_from_resid(&mut resid);
            bits += 2.0;
            for &lv in &resid {
                bits += coef_rate_bits(lv.unsigned_abs());
            }
        }
    }
    bits
}

#[inline]
fn cfl_alpha_sign(alpha: i32) -> usize {
    if alpha == 0 {
        0
    } else if alpha < 0 {
        1
    } else {
        2
    }
}

fn cfl_alpha_syntax_bits(alpha: [i32; 2]) -> f32 {
    let su = cfl_alpha_sign(alpha[0]);
    let sv = cfl_alpha_sign(alpha[1]);
    let sign = su * 3 + sv;
    debug_assert_ne!(sign, 0);
    let mut bits = raw_symbol_cost(&CFL_SIGN_CDF, sign - 1);
    if su != 0 {
        let ctx = usize::from(su == 2) * 3 + sv;
        bits += raw_symbol_cost(&CFL_ALPHA_CDF[ctx], (alpha[0].abs() - 1) as usize);
    }
    if sv != 0 {
        let ctx = usize::from(sv == 2) * 3 + su;
        bits += raw_symbol_cost(&CFL_ALPHA_CDF[ctx], (alpha[1].abs() - 1) as usize);
    }
    bits
}

#[allow(clippy::too_many_arguments)]
fn cfl_alpha_candidates(
    luma: &[i16],
    chroma: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    base: i32,
    bit_depth: u8,
    exhaustive: bool,
) -> Vec<i32> {
    if exhaustive {
        return (-16..=16).collect();
    }
    let mut dc = [0i32; 16];
    predict_lossless_4x4(
        0, 0, chroma, stride, px, py, px, py, 4, &mut dc, base, bit_depth,
    );
    let mut luma_rec = [0i32; 16];
    let mut src = [0i32; 16];
    for row in 0..4 {
        for x in 0..4 {
            let i = row * 4 + x;
            luma_rec[i] = i32::from(luma[(py + row) * stride + px + x]);
            src[i] = i32::from(chroma[(py + row) * stride + px + x]);
        }
    }
    let mut ac = [0i32; 16];
    cfl_ac_444(&luma_rec, 4, 4, &mut ac);
    let seed = cfl_best_alpha(&ac, &src, dc[0], 16, bit_depth);
    let mut out = vec![0];
    for alpha in seed - 2..=seed + 2 {
        if (-16..=16).contains(&alpha) && !out.contains(&alpha) {
            out.push(alpha);
        }
    }
    out
}

/// Residual bits (coef-rate proxy) of coding one plane of an `n_tx`x`n_tx` leaf
/// at `(bx, by)` with `mode`.
#[allow(clippy::too_many_arguments)]
fn plane_leaf_bits(
    mode: usize,
    angle_delta: i32,
    plane: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    n_tx: usize,
    base: i32,
    bit_depth: u8,
) -> f32 {
    let mut bits = 0f32;
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];

    for ty in 0..n_tx {
        for tx in 0..n_tx {
            let (ox, oy) = (bx + tx * 4, by + ty * 4);
            predict_lossless_4x4(
                mode,
                angle_delta,
                plane,
                stride,
                ox,
                oy,
                bx,
                by,
                n_tx * 4,
                &mut pred,
                base,
                bit_depth,
            );
            let mut any = false;
            for (ry, (rrow, prow)) in resid
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .zip(pred.as_chunks::<4>().0.iter())
                .enumerate()
            {
                let srow = &plane[(oy + ry) * stride + ox..];
                for (r, (&srv, &pv)) in rrow.iter_mut().zip(srow.iter().zip(prow.iter())) {
                    *r = srv as i32 - pv;
                    any |= *r != 0;
                }
            }
            if !any {
                bits += 1.0; // all-zero flag
                continue;
            }
            levels_from_resid(&mut resid);
            bits += 2.0; // eob / skip overhead
            for &lv in resid.iter() {
                bits += coef_rate_bits(lv.unsigned_abs());
            }
        }
    }
    bits
}

static LL_RECT_MODES: [usize; 7] = [0, 1, 2, 9, 10, 11, 12];
static LL_RECT_FAST_MODES: [usize; 5] = [0, 1, 2, 9, 12];

/// Lossless TX_4X4 rate proxy for a rectangular intra block.  Directional
/// edge derivation depends on rectangular block geometry and is deliberately
/// left to square leaves for now; all decoder-safe non-directional predictors
/// are searched here.
#[allow(clippy::too_many_arguments)]
fn plane_rect_bits(
    mode: usize,
    plane: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    width: usize,
    height: usize,
    base: i32,
) -> f32 {
    let mut bits = 0.0f32;
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];
    for ty in 0..height / 4 {
        for tx in 0..width / 4 {
            let (ox, oy) = (bx + tx * 4, by + ty * 4);
            predict_4x4(mode, plane, stride, ox, oy, &mut pred, base);
            let mut any = false;
            for row in 0..4 {
                for x in 0..4 {
                    let i = row * 4 + x;
                    resid[i] = i32::from(plane[(oy + row) * stride + ox + x]) - pred[i];
                    any |= resid[i] != 0;
                }
            }
            if !any {
                bits += 1.0;
            } else {
                levels_from_resid(&mut resid);
                bits += 2.0;
                bits += resid
                    .iter()
                    .map(|v| coef_rate_bits(v.unsigned_abs()))
                    .sum::<f32>();
            }
        }
    }
    bits
}

#[derive(Clone)]
struct BlockDecision {
    dx: usize,
    dy: usize,
    width: usize,
    height: usize,
    y_mode: usize,
    y_delta: i32,
    uv_mode: usize,
    uv_delta: i32,
    uv_alpha: [i32; 2],
    palette: Option<LumaPalette>,
    intrabc: bool,
}

#[allow(clippy::too_many_arguments)]
fn best_rect_block(
    planes: [&[i16]; 3],
    stride: usize,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    base: i32,
    full_search: bool,
) -> (f32, BlockDecision) {
    let modes: &[usize] = if full_search {
        &LL_RECT_MODES
    } else {
        &LL_RECT_FAST_MODES
    };
    let mut y = (f32::INFINITY, 0usize);
    for &mode in modes {
        let bits = plane_rect_bits(mode, planes[0], stride, px, py, width, height, base);
        if bits < y.0 {
            y = (bits, mode);
        }
    }
    let mut uv = (f32::INFINITY, 0usize);
    for &mode in modes {
        let bits = plane_rect_bits(mode, planes[1], stride, px, py, width, height, base)
            + plane_rect_bits(mode, planes[2], stride, px, py, width, height, base);
        if bits < uv.0 {
            uv = (bits, mode);
        }
    }
    (
        y.0 + uv.0 + 7.0,
        BlockDecision {
            dx: 0,
            dy: 0,
            width,
            height,
            y_mode: y.1,
            y_delta: 0,
            uv_mode: uv.1,
            uv_delta: 0,
            uv_alpha: [0, 0],
            palette: None,
            intrabc: false,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn best_leaf(
    planes: [&[i16]; 3],
    stride: usize,
    px: usize,
    py: usize,
    n_tx: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (
    f32,
    usize,
    i32,
    usize,
    i32,
    [i32; 2],
    Option<LumaPalette>,
    bool,
) {
    let mut y_mode = 0usize;
    let mut y_delta = 0i32;
    let mut yb = f32::INFINITY;
    let mut y_directional = [(f32::INFINITY, 0usize); 8];
    let angle_allowed = n_tx >= 2;
    let modes = if n_tx == 1 {
        &LL4_MODES[..]
    } else {
        &LL_MODES[..]
    };
    for &m in modes {
        let mut b = plane_leaf_bits(m, 0, planes[0], stride, px, py, n_tx, base, bit_depth);
        if angle_allowed && (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            y_directional[m - 1] = (b, m);
        }
        if b < yb {
            yb = b;
            y_mode = m;
            y_delta = 0;
        }
    }
    if angle_allowed && angle_delta_rdo {
        y_directional.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, m) in &y_directional[..3] {
            for delta in [-3, -2, -1, 1, 2, 3] {
                let b = plane_leaf_bits(m, delta, planes[0], stride, px, py, n_tx, base, bit_depth)
                    + raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], (delta + 3) as usize);
                if b < yb {
                    yb = b;
                    y_mode = m;
                    y_delta = delta;
                }
            }
        }
    }
    let mut uv_mode = 0usize;
    let mut uv_delta = 0i32;
    let mut uv_alpha = [0i32; 2];
    let mut ub = f32::INFINITY;
    let mut uv_directional = [(f32::INFINITY, 0usize); 8];
    // Normative coded-lossless exception: CfL is legal only when the chroma
    // prediction block equals the forced TX_4X4 transform size.
    let cfl_allowed = n_tx == 1;
    for &m in modes {
        let mut b = plane_leaf_bits(m, 0, planes[1], stride, px, py, n_tx, base, bit_depth)
            + plane_leaf_bits(m, 0, planes[2], stride, px, py, n_tx, base, bit_depth);
        b += raw_symbol_cost(
            if cfl_allowed {
                &UV_MODE_CFL_CDF[y_mode]
            } else {
                &UV_MODE_NOCFL_CDF[y_mode]
            },
            m,
        );
        if angle_allowed && (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            uv_directional[m - 1] = (b, m);
        }
        if b < ub {
            ub = b;
            uv_mode = m;
            uv_delta = 0;
        }
    }
    if angle_allowed && angle_delta_rdo {
        uv_directional.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, m) in &uv_directional[..3] {
            for delta in [-3, -2, -1, 1, 2, 3] {
                let b = plane_leaf_bits(m, delta, planes[1], stride, px, py, n_tx, base, bit_depth)
                    + plane_leaf_bits(m, delta, planes[2], stride, px, py, n_tx, base, bit_depth)
                    + raw_symbol_cost(
                        if cfl_allowed {
                            &UV_MODE_CFL_CDF[y_mode]
                        } else {
                            &UV_MODE_NOCFL_CDF[y_mode]
                        },
                        m,
                    )
                    + raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], (delta + 3) as usize);
                if b < ub {
                    ub = b;
                    uv_mode = m;
                    uv_delta = delta;
                }
            }
        }
    }
    if cfl_allowed {
        // Compute the two planes independently for every normative alpha, then
        // combine their rates with the exact joint-sign/magnitude syntax cost.
        // This keeps U and V free to choose different signs and magnitudes.
        let alpha_candidates = [
            cfl_alpha_candidates(planes[0], planes[1], stride, px, py, base, bit_depth, false),
            cfl_alpha_candidates(planes[0], planes[2], stride, px, py, base, bit_depth, false),
        ];
        let mut alpha_bits = [[f32::INFINITY; 33]; 2];
        for (ci, plane) in planes[1..].iter().enumerate() {
            for &alpha in &alpha_candidates[ci] {
                alpha_bits[ci][(alpha + 16) as usize] = plane_cfl_bits(
                    alpha, planes[0], plane, stride, px, py, n_tx, base, bit_depth,
                );
            }
        }
        let cfl_mode_bits = raw_symbol_cost(&UV_MODE_CFL_CDF[y_mode], CFL_PRED);
        for &au in &alpha_candidates[0] {
            for &av in &alpha_candidates[1] {
                if au == 0 && av == 0 {
                    continue;
                }
                let alpha = [au, av];
                let b = alpha_bits[0][(au + 16) as usize]
                    + alpha_bits[1][(av + 16) as usize]
                    + cfl_mode_bits
                    + cfl_alpha_syntax_bits(alpha);
                if b < ub {
                    ub = b;
                    uv_mode = CFL_PRED;
                    uv_delta = 0;
                    uv_alpha = alpha;
                }
            }
        }
    }
    let palette = (n_tx >= 2 && px + n_tx * 4 <= visible_w && py + n_tx * 4 <= visible_h)
        .then(|| exact_luma_palette(planes[0], stride, px, py, n_tx * 4, n_tx * 4, bit_depth))
        .flatten()
        .filter(|palette| palette_best_case_bits(palette, bit_depth) < yb);
    if let Some(palette) = palette.as_ref() {
        yb = yb.min(palette_estimated_bits(palette, bit_depth, &[], 0));
    }
    let ovh = 7.0; // skip + y_mode + uv_mode (angle-delta rate is above)
    let regular_bits = yb + ub + ovh;
    let ibc_margin = intrabc_rdo_margin(n_tx * 4);
    let ibc = (n_tx >= 2 && regular_bits > 8.0 + ibc_margin)
        .then(|| {
            find_exact_intrabc(
                &planes,
                stride,
                px,
                py,
                n_tx * 4,
                stride,
                planes[0].len() / stride,
                intrabc_index,
            )
        })
        .flatten()
        .map(|candidate| intrabc_estimated_bits(candidate, py))
        .filter(|&bits| bits + ibc_margin < regular_bits);
    if let Some(bits) = ibc {
        (bits, 0, 0, 0, 0, [0, 0], None, true)
    } else {
        (
            regular_bits,
            y_mode,
            y_delta,
            uv_mode,
            uv_delta,
            uv_alpha,
            palette,
            false,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn best_partition_block(
    planes: [&[i16]; 3],
    stride: usize,
    parent_x: usize,
    parent_y: usize,
    dx: usize,
    dy: usize,
    width: usize,
    height: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (f32, BlockDecision) {
    let (px, py) = (parent_x + dx, parent_y + dy);
    if width == height {
        let (bits, ym, yd, uv, uvd, uva, palette, intrabc) = best_leaf(
            planes,
            stride,
            px,
            py,
            width / 4,
            base,
            bit_depth,
            visible_w,
            visible_h,
            angle_delta_rdo,
            intrabc_index,
        );
        (
            bits,
            BlockDecision {
                dx,
                dy,
                width,
                height,
                y_mode: ym,
                y_delta: yd,
                uv_mode: uv,
                uv_delta: uvd,
                uv_alpha: uva,
                palette,
                intrabc,
            },
        )
    } else {
        let (bits, mut block) =
            best_rect_block(planes, stride, px, py, width, height, base, angle_delta_rdo);
        block.dx = dx;
        block.dy = dy;
        (bits, block)
    }
}

/// Adaptive partition plan for a fully-in-frame square block.
enum Plan {
    Leaf {
        y_mode: usize,
        y_delta: i32,
        uv_mode: usize,
        uv_delta: i32,
        uv_alpha: [i32; 2],
        palette: Option<LumaPalette>,
        intrabc: bool,
    },
    Split(Box<[Plan; 4]>),
    Partition {
        symbol: usize,
        blocks: Vec<BlockDecision>,
    },
}

const PART_NONE_BITS: f32 = 1.0;
const PART_SPLIT_BITS: f32 = 1.5;
const PART_RECT_BITS: f32 = 3.5;

#[cfg(test)]
thread_local! {
    static FORCE_LL_PARTITION: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[inline]
#[allow(clippy::needless_return)] // cfg(test) early-return; cfg!(test) can't see the test-only thread-local
fn forced_ll_partition() -> usize {
    #[cfg(test)]
    {
        return FORCE_LL_PARTITION.with(std::cell::Cell::get);
    }
    #[cfg(not(test))]
    0
}

/// Decide none-vs-split by estimated bits; returns the plan and its cost. At
/// 8x8, PARTITION_SPLIT produces four normative 4x4 leaves (needed for coded-lossless CfL, which is disallowed at larger block sizes).
#[allow(clippy::too_many_arguments)]
fn plan_full(
    planes: [&[i16]; 3],
    stride: usize,
    px: usize,
    py: usize,
    sz8: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (f32, Plan) {
    let (bits_leaf, ym, yd, uv, uvd, uva, palette, intrabc) = best_leaf(
        planes,
        stride,
        px,
        py,
        sz8 * 2,
        base,
        bit_depth,
        visible_w,
        visible_h,
        angle_delta_rdo,
        intrabc_index,
    );
    let none = PART_NONE_BITS + bits_leaf;
    let eval_partition = |symbol: usize, geometry: &[(usize, usize, usize, usize)]| {
        let mut bits = PART_RECT_BITS;
        let mut blocks = Vec::with_capacity(geometry.len());
        for &(dx, dy, width, height) in geometry {
            let (block_bits, block) = best_partition_block(
                planes,
                stride,
                px,
                py,
                dx,
                dy,
                width,
                height,
                base,
                bit_depth,
                visible_w,
                visible_h,
                angle_delta_rdo,
                intrabc_index,
            );
            bits += block_bits;
            blocks.push(block);
        }
        (bits, Plan::Partition { symbol, blocks })
    };
    if sz8 == 1 {
        let mut split = PART_SPLIT_BITS;
        let mut kids: [Option<Plan>; 4] = [None, None, None, None];
        for (i, (cx, cy)) in [(px, py), (px + 4, py), (px, py + 4), (px + 4, py + 4)]
            .into_iter()
            .enumerate()
        {
            let (b, cym, cyd, cuv, cuvd, cuva, cpalette, cibc) = best_leaf(
                planes,
                stride,
                cx,
                cy,
                1,
                base,
                bit_depth,
                visible_w,
                visible_h,
                false,
                intrabc_index,
            );
            split += b;
            kids[i] = Some(Plan::Leaf {
                y_mode: cym,
                y_delta: cyd,
                uv_mode: cuv,
                uv_delta: cuvd,
                uv_alpha: cuva,
                palette: cpalette,
                intrabc: cibc,
            });
        }
        let mut best = (
            none,
            Plan::Leaf {
                y_mode: ym,
                y_delta: yd,
                uv_mode: uv,
                uv_delta: uvd,
                uv_alpha: uva,
                palette,
                intrabc,
            },
        );
        let split_plan = (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))));
        if split_plan.0 < best.0 {
            best = split_plan;
        }
        for candidate in [
            eval_partition(1, &[(0, 0, 8, 4), (0, 4, 8, 4)]),
            eval_partition(2, &[(0, 0, 4, 8), (4, 0, 4, 8)]),
        ] {
            if candidate.0 < best.0 {
                best = candidate;
            }
        }
        return best;
    }
    let hh = sz8 / 2;
    let mut split = PART_SPLIT_BITS;
    let mut kids: [Option<Plan>; 4] = [None, None, None, None];
    for (i, (cx, cy)) in [
        (px, py),
        (px + hh * 8, py),
        (px, py + hh * 8),
        (px + hh * 8, py + hh * 8),
    ]
    .into_iter()
    .enumerate()
    {
        let (b, p) = plan_full(
            planes,
            stride,
            cx,
            cy,
            hh,
            base,
            bit_depth,
            visible_w,
            visible_h,
            angle_delta_rdo,
            intrabc_index,
        );
        split += b;
        kids[i] = Some(p);
    }
    let mut best = (
        none,
        Plan::Leaf {
            y_mode: ym,
            y_delta: yd,
            uv_mode: uv,
            uv_delta: uvd,
            uv_alpha: uva,
            palette,
            intrabc,
        },
    );
    let split_plan = (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))));
    if split_plan.0 < best.0 {
        best = split_plan;
    }
    let size = sz8 * 8;
    let half = size / 2;
    let quarter = size / 4;
    #[allow(clippy::type_complexity)]
    let geometries: [(usize, Vec<(usize, usize, usize, usize)>); 8] = [
        (1, vec![(0, 0, size, half), (0, half, size, half)]),
        (2, vec![(0, 0, half, size), (half, 0, half, size)]),
        (
            4,
            vec![
                (0, 0, half, half),
                (half, 0, half, half),
                (0, half, size, half),
            ],
        ),
        (
            5,
            vec![
                (0, 0, size, half),
                (0, half, half, half),
                (half, half, half, half),
            ],
        ),
        (
            6,
            vec![
                (0, 0, half, half),
                (0, half, half, half),
                (half, 0, half, size),
            ],
        ),
        (
            7,
            vec![
                (0, 0, half, size),
                (half, 0, half, half),
                (half, half, half, half),
            ],
        ),
        (8, (0..4).map(|i| (0, i * quarter, size, quarter)).collect()),
        (9, (0..4).map(|i| (i * quarter, 0, quarter, size)).collect()),
    ];
    let rect_candidates = if angle_delta_rdo {
        if size >= 32 { 8 } else { 6 }
    } else {
        2
    };
    for (symbol, geometry) in geometries.iter().take(rect_candidates) {
        let candidate = eval_partition(*symbol, geometry);
        if forced_ll_partition() == *symbol {
            return candidate;
        }
        if candidate.0 < best.0 {
            best = candidate;
        }
    }
    best
}

/// Mutable frame-spanning state shared across the lossless partition recursion.
struct LlState {
    w: usize,
    h: usize,
    visible_w: usize,
    visible_h: usize,
    base: i32,
    bit_depth: u8,
    angle_delta_rdo: bool,
    intrabc_index: IntrabcIndex,
    cdfs: Box<Cdfs>,
    a_coef: [Vec<u8>; 3],
    l_coef: [Vec<u8>; 3],
    a_part: Vec<u8>,
    l_part: Vec<u8>,
    a_mode: Vec<u8>, // luma y_mode per 4px unit (for kf_y context)
    l_mode: Vec<u8>,
    a_skip: Vec<u8>, // block skip flags per 4px unit
    l_skip: Vec<u8>,
    ibc_mv: Vec<Option<(i16, i16)>>, // decoded IntraBC MVs per 4x4 unit
    a_palette: Vec<Vec<i32>>,        // palette cache per 4px unit
    l_palette: Vec<Vec<i32>>,
}

fn write_block_skip(
    wr: &mut Writer,
    st: &mut LlState,
    px: usize,
    py: usize,
    size: usize,
    skip: bool,
) {
    let (x4, y4, n4) = (px / 4, py / 4, size / 4);
    let ctx = usize::from(st.a_skip[x4]) + usize::from(st.l_skip[y4]);
    wr.symbol_adapt(usize::from(skip), &mut st.cdfs.skip[ctx]);
    st.a_skip[x4..x4 + n4].fill(skip as u8);
    st.l_skip[y4..y4 + n4].fill(skip as u8);
}

fn write_block_skip_rect(
    wr: &mut Writer,
    st: &mut LlState,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    skip: bool,
) {
    let (x4, y4, nw, nh) = (px / 4, py / 4, width / 4, height / 4);
    let ctx = usize::from(st.a_skip[x4]) + usize::from(st.l_skip[y4]);
    wr.symbol_adapt(usize::from(skip), &mut st.cdfs.skip[ctx]);
    st.a_skip[x4..x4 + nw].fill(skip as u8);
    st.l_skip[y4..y4 + nh].fill(skip as u8);
}

fn write_intrabc_mv_component(wr: &mut Writer, cdfs: &mut Cdfs, component: usize, diff: i32) {
    debug_assert_ne!(diff, 0);
    debug_assert_eq!(diff & 7, 0);
    wr.symbol_adapt(usize::from(diff < 0), &mut cdfs.mv_sign[component]);
    let up = diff.unsigned_abs() as usize / 8 - 1;
    let class = if up <= 1 {
        0
    } else {
        usize::BITS as usize - 1 - up.leading_zeros() as usize
    };
    debug_assert!(class <= 10);
    wr.symbol_adapt(class, &mut cdfs.mv_classes[component]);
    if class == 0 {
        wr.symbol_adapt(up, &mut cdfs.mv_class0[component]);
    } else {
        for n in 0..class {
            wr.symbol_adapt((up >> n) & 1, &mut cdfs.mv_class_n[component][n]);
        }
    }
}

fn write_intrabc_mv(wr: &mut Writer, cdfs: &mut Cdfs, mv: (i16, i16), pred: (i16, i16)) {
    let dy = i32::from(mv.0) - i32::from(pred.0);
    let dx = i32::from(mv.1) - i32::from(pred.1);
    let joint = usize::from(dx != 0) | (usize::from(dy != 0) << 1);
    wr.symbol_adapt(joint, &mut cdfs.mv_joint);
    if dy != 0 {
        write_intrabc_mv_component(wr, cdfs, 0, dy);
    }
    if dx != 0 {
        write_intrabc_mv_component(wr, cdfs, 1, dx);
    }
}

fn write_uniform(wr: &mut Writer, n: usize, value: usize) {
    debug_assert!(n > 0 && value < n);
    let bits = usize::BITS - (n - 1).leading_zeros();
    let cutoff = (1usize << bits) - n;
    if value < cutoff {
        wr.literal((bits - 1) as u8, value as u32);
    } else {
        let value = value - cutoff;
        wr.literal((bits - 1) as u8, (cutoff + (value >> 1)) as u32);
        wr.bit((value & 1) as u16);
    }
}

fn palette_bsize_ctx(size: usize) -> usize {
    debug_assert!(matches!(size, 8 | 16 | 32 | 64));
    2 * (size.trailing_zeros() as usize - 3)
}

fn palette_y_mode_raw(bsize_ctx: usize, mode_ctx: usize) -> u16 {
    static RAW: [[u16; 3]; 7] = [
        [31676, 3419, 1261],
        [31912, 2859, 980],
        [31823, 3400, 781],
        [32030, 3561, 904],
        [32309, 7337, 1462],
        [32265, 4015, 1521],
        [32450, 7946, 129],
    ];
    RAW[bsize_ctx][mode_ctx]
}

fn palette_y_size_raw(bsize_ctx: usize) -> &'static [u16; 6] {
    const RAW: [[u16; 6]; 7] = [
        [7952, 13000, 18149, 21478, 25527, 29241],
        [7139, 11421, 16195, 19544, 23666, 28073],
        [7788, 12741, 17325, 20500, 24315, 28530],
        [8271, 14064, 18246, 21564, 25071, 28533],
        [12725, 19180, 21863, 24839, 27535, 30120],
        [9711, 14888, 16923, 21052, 25661, 27875],
        [14940, 20797, 21678, 24186, 27033, 28999],
    ];
    &RAW[bsize_ctx]
}

fn palette_y_color_raw(size: usize, ctx: usize) -> &'static [u16] {
    match (size, ctx) {
        (2, 0) => &[28710],
        (2, 1) => &[16384],
        (2, 2) => &[10553],
        (2, 3) => &[27036],
        (2, 4) => &[31603],
        (3, 0) => &[27877, 30490],
        (3, 1) => &[11532, 25697],
        (3, 2) => &[6544, 30234],
        (3, 3) => &[23018, 28072],
        (3, 4) => &[31915, 32385],
        (4, 0) => &[25572, 28046, 30045],
        (4, 1) => &[9478, 21590, 27256],
        (4, 2) => &[7248, 26837, 29824],
        (4, 3) => &[19167, 24486, 28349],
        (4, 4) => &[31400, 31825, 32250],
        (5, 0) => &[24779, 26955, 28576, 30282],
        (5, 1) => &[8669, 20364, 24073, 28093],
        (5, 2) => &[4255, 27565, 29377, 31067],
        (5, 3) => &[19864, 23674, 26716, 29530],
        (5, 4) => &[31646, 31893, 32147, 32426],
        (6, 0) => &[23132, 25407, 26970, 28435, 30073],
        (6, 1) => &[7443, 17242, 20717, 24762, 27982],
        (6, 2) => &[6300, 24862, 26944, 28784, 30671],
        (6, 3) => &[18916, 22895, 25267, 27435, 29652],
        (6, 4) => &[31270, 31550, 31808, 32059, 32353],
        (7, 0) => &[23105, 25199, 26464, 27684, 28931, 30318],
        (7, 1) => &[6950, 15447, 18952, 22681, 25567, 28563],
        (7, 2) => &[7560, 23474, 25490, 27203, 28921, 30708],
        (7, 3) => &[18544, 22373, 24457, 26195, 28119, 30045],
        (7, 4) => &[31198, 31451, 31670, 31882, 32123, 32391],
        (8, 0) => &[21689, 23883, 25163, 26352, 27506, 28827, 30195],
        (8, 1) => &[6892, 15385, 17840, 21606, 24287, 26753, 29204],
        (8, 2) => &[5651, 23182, 25042, 26518, 27982, 29392, 30900],
        (8, 3) => &[19349, 22578, 24418, 25994, 27524, 29031, 30448],
        (8, 4) => &[31028, 31270, 31504, 31705, 31927, 32153, 32392],
        _ => unreachable!("palette size/context {size}/{ctx}"),
    }
}

fn palette_cache(above: &[i32], left: &[i32], allow_above: bool) -> Vec<i32> {
    let mut cache = Vec::with_capacity(16);
    if allow_above {
        cache.extend_from_slice(above);
    }
    cache.extend_from_slice(left);
    cache.sort_unstable();
    cache.dedup();
    cache
}

fn ceil_log2(value: u32) -> u8 {
    if value <= 1 {
        0
    } else {
        (32 - (value - 1).leading_zeros()) as u8
    }
}

fn write_palette_colors(wr: &mut Writer, colors: &[i32], cache: &[i32], bit_depth: u8) {
    let mut in_cache = 0usize;
    for &cached in cache {
        if in_cache == colors.len() {
            break;
        }
        let found = colors.binary_search(&cached).is_ok();
        wr.bit(found as u16);
        in_cache += usize::from(found);
    }
    let out: Vec<u32> = colors
        .iter()
        .filter(|color| cache.binary_search(color).is_err())
        .map(|&color| color as u32)
        .collect();
    if out.is_empty() {
        return;
    }
    wr.literal(bit_depth, out[0]);
    if out.len() == 1 {
        return;
    }
    let max_delta = out.array_windows::<2>().map(|v| v[1] - v[0]).max().unwrap();
    let min_bits = bit_depth - 3;
    let mut bits = ceil_log2(max_delta).max(min_bits);
    wr.literal(2, u32::from(bits - min_bits));
    let mut range = (1u32 << bit_depth) - out[0] - 1;
    for pair in out.array_windows::<2>() {
        let delta = pair[1] - pair[0];
        wr.literal(bits, delta - 1);
        range -= delta;
        bits = bits.min(ceil_log2(range));
    }
}

fn palette_color_ctx(map: &[u8], stride: usize, y: usize, x: usize, size: usize) -> (usize, usize) {
    let current = map[y * stride + x] as usize;
    let mut scores = [0u8; 8];
    if x > 0 {
        scores[map[y * stride + x - 1] as usize] += 2;
    }
    if y > 0 {
        scores[map[(y - 1) * stride + x] as usize] += 2;
    }
    if x > 0 && y > 0 {
        scores[map[(y - 1) * stride + x - 1] as usize] += 1;
    }
    let mut ranked: Vec<usize> = (0..size).filter(|&i| scores[i] != 0).collect();
    ranked.sort_by_key(|&i| (std::cmp::Reverse(scores[i]), i));
    let ctx = if x == 0 || y == 0 {
        0
    } else {
        const MULT: [usize; 3] = [1, 2, 2];
        let hash: usize = ranked
            .iter()
            .zip(MULT)
            .map(|(&color, mult)| scores[color] as usize * mult)
            .sum();
        9 - hash
    };
    let mut order = ranked;
    for color in 0..size {
        if !order.contains(&color) {
            order.push(color);
        }
    }
    let symbol = order.iter().position(|&color| color == current).unwrap();
    (ctx, symbol)
}

fn write_palette_map(wr: &mut Writer, cdfs: &mut Cdfs, palette: &LumaPalette) {
    let size = palette.colors.len();
    write_uniform(wr, size, palette.map[0] as usize);
    for diagonal in 1..palette.width + palette.height - 1 {
        let first_x = diagonal.min(palette.width - 1);
        let last_x = diagonal.saturating_sub(palette.height - 1);
        for x in (last_x..=first_x).rev() {
            let y = diagonal - x;
            let (ctx, symbol) = palette_color_ctx(&palette.map, palette.width, y, x, size);
            wr.symbol_adapt(symbol, &mut cdfs.palette_y_color[size - 2][ctx]);
        }
    }
}

fn angle_delta_bits(mode: usize, delta: i32) -> f32 {
    if (1..=8).contains(&mode) {
        raw_symbol_cost(&ANGLE_DELTA_CDF[mode - 1], (delta + 3) as usize)
    } else {
        0.0
    }
}

fn uv_mode_syntax_bits(
    y_mode: usize,
    uv_mode: usize,
    uv_delta: i32,
    uv_alpha: [i32; 2],
    cfl_allowed: bool,
    luma_palette: bool,
) -> f32 {
    let mut bits = raw_symbol_cost(
        if cfl_allowed {
            &UV_MODE_CFL_CDF[y_mode]
        } else {
            &UV_MODE_NOCFL_CDF[y_mode]
        },
        uv_mode,
    );
    if uv_mode == CFL_PRED {
        bits += cfl_alpha_syntax_bits(uv_alpha);
    } else {
        bits += angle_delta_bits(uv_mode, uv_delta);
    }
    if uv_mode == 0 {
        bits += raw_symbol_cost(&[if luma_palette { 21488 } else { 32461 }], 0);
    }
    bits
}

#[allow(clippy::too_many_arguments)]
fn palette_wins_live(
    st: &LlState,
    luma: &[i16],
    px: usize,
    py: usize,
    size: usize,
    y_mode: usize,
    y_delta: i32,
    uv: Option<(usize, i32, [i32; 2])>,
    palette: &LumaPalette,
) -> bool {
    let (x4, y4, n_tx) = (px / 4, py / 4, size / 4);
    let bsize_ctx = palette_bsize_ctx(size);
    let mode_ctx =
        usize::from(!st.a_palette[x4].is_empty()) + usize::from(!st.l_palette[y4].is_empty());
    let cache = palette_cache(&st.a_palette[x4], &st.l_palette[y4], !py.is_multiple_of(64));
    let kf_raw = &KF_Y_MODE_CDF[INTRA_MODE_CTX[st.a_mode[x4] as usize]]
        [INTRA_MODE_CTX[st.l_mode[y4] as usize]];

    let mut normal_bits = plane_leaf_bits(
        y_mode,
        y_delta,
        luma,
        st.w,
        px,
        py,
        n_tx,
        st.base,
        st.bit_depth,
    ) + raw_symbol_cost(kf_raw, y_mode)
        + angle_delta_bits(y_mode, y_delta);
    if y_mode == 0 {
        normal_bits += raw_symbol_cost(&[palette_y_mode_raw(bsize_ctx, mode_ctx)], 0);
    }

    let mut palette_bits = raw_symbol_cost(kf_raw, 0)
        + palette_estimated_bits(palette, st.bit_depth, &cache, mode_ctx);
    if let Some((uv_mode, uv_delta, uv_alpha)) = uv {
        let cfl_allowed = size == 4;
        normal_bits += uv_mode_syntax_bits(y_mode, uv_mode, uv_delta, uv_alpha, cfl_allowed, false);
        palette_bits += uv_mode_syntax_bits(0, uv_mode, uv_delta, uv_alpha, cfl_allowed, true);
    }
    palette_bits < normal_bits
}

/// Code a square lossless leaf of `size`×`size` px at `(px, py)`: block mode
/// info (skip=0, y_mode=DC, uv_mode) then `(size/4)²` TX_4×4 WHT per plane.
/// uv_mode is non-CfL `UV_DC` at 64×64 (CfL not allowed) and the CfL DC symbol
/// otherwise.
#[allow(clippy::too_many_arguments)]
fn code_leaf(
    wr: &mut Writer,
    planes: [&[i16]; 3],
    st: &mut LlState,
    px: usize,
    py: usize,
    size: usize,
    y_mode: usize,
    y_delta: i32,
    uv_mode: usize,
    uv_delta: i32,
    uv_alpha: [i32; 2],
    palette: Option<&LumaPalette>,
    intrabc: bool,
) {
    let n_tx = size / 4;
    let (x4, y4) = (px / 4, py / 4);
    let candidate = intrabc
        .then(|| find_exact_intrabc(&planes, st.w, px, py, size, st.w, st.h, &st.intrabc_index))
        .flatten();
    let predictor = candidate.and_then(|_| intrabc_mv_predictor(st, px, py, size));
    let intrabc = candidate.zip(predictor).is_some_and(|(candidate, pred)| {
        (i32::from(candidate.mv.0) - i32::from(pred.0)).unsigned_abs() <= 16_384
            && (i32::from(candidate.mv.1) - i32::from(pred.1)).unsigned_abs() <= 16_384
    });
    write_block_skip(wr, st, px, py, size, intrabc);
    // `allow_intrabc` is frame-wide, so every block carries this flag. The
    // default CDF strongly favours ordinary intra blocks.
    wr.symbol_adapt(intrabc as usize, &mut st.cdfs.intrabc);
    if intrabc {
        let candidate = candidate.unwrap();
        write_intrabc_mv(wr, &mut st.cdfs, candidate.mv, predictor.unwrap());

        st.a_mode[x4..x4 + n_tx].fill(0);
        st.l_mode[y4..y4 + n_tx].fill(0);
        for slot in &mut st.a_palette[x4..x4 + n_tx] {
            slot.clear();
        }
        for slot in &mut st.l_palette[y4..y4 + n_tx] {
            slot.clear();
        }
        let mv = candidate.mv;
        let stride4 = st.w / 4;
        for y in y4..y4 + n_tx {
            st.ibc_mv[y * stride4 + x4..y * stride4 + x4 + n_tx].fill(Some(mv));
        }
        for plane in 0..3 {
            st.a_coef[plane][x4..x4 + n_tx].fill(0x40);
            st.l_coef[plane][y4..y4 + n_tx].fill(0x40);
        }
        return;
    }

    let palette = palette.filter(|palette| {
        palette_wins_live(
            st,
            planes[0],
            px,
            py,
            size,
            y_mode,
            y_delta,
            Some((uv_mode, uv_delta, uv_alpha)),
            palette,
        )
    });
    let (y_mode, y_delta) = if palette.is_some() {
        (0, 0)
    } else {
        (y_mode, y_delta)
    };

    let y_ctx = INTRA_MODE_CTX[st.a_mode[x4] as usize] * 5 + INTRA_MODE_CTX[st.l_mode[y4] as usize];
    wr.symbol_adapt(y_mode, &mut st.cdfs.kf_y[y_ctx]);
    if size >= 8 && (1..=8).contains(&y_mode) {
        wr.symbol_adapt((y_delta + 3) as usize, &mut st.cdfs.angle_delta[y_mode - 1]);
    }
    let cfl_allowed = size == 4;
    let uv_cdf = if cfl_allowed { 13 + y_mode } else { y_mode };
    wr.symbol_adapt(uv_mode, &mut st.cdfs.uv_mode[uv_cdf]);
    if uv_mode == CFL_PRED {
        let su = cfl_alpha_sign(uv_alpha[0]);
        let sv = cfl_alpha_sign(uv_alpha[1]);
        let sign = su * 3 + sv;
        debug_assert_ne!(sign, 0);
        wr.symbol_adapt(sign - 1, &mut st.cdfs.cfl_sign);
        if su != 0 {
            let ctx = usize::from(su == 2) * 3 + sv;
            wr.symbol_adapt(
                (uv_alpha[0].abs() - 1) as usize,
                &mut st.cdfs.cfl_alpha[ctx],
            );
        }
        if sv != 0 {
            let ctx = usize::from(sv == 2) * 3 + su;
            wr.symbol_adapt(
                (uv_alpha[1].abs() - 1) as usize,
                &mut st.cdfs.cfl_alpha[ctx],
            );
        }
    } else if size >= 8 && (1..=8).contains(&uv_mode) {
        wr.symbol_adapt(
            (uv_delta + 3) as usize,
            &mut st.cdfs.angle_delta[uv_mode - 1],
        );
    }
    let palette_allowed = size >= 8;
    let bsize_ctx = palette_allowed.then(|| palette_bsize_ctx(size));
    let mode_ctx =
        usize::from(!st.a_palette[x4].is_empty()) + usize::from(!st.l_palette[y4].is_empty());
    if palette_allowed && y_mode == 0 {
        wr.symbol_adapt(
            usize::from(palette.is_some()),
            &mut st.cdfs.palette_y_mode[bsize_ctx.unwrap()][mode_ctx],
        );
        if let Some(palette) = palette {
            wr.symbol_adapt(
                palette.colors.len() - 2,
                &mut st.cdfs.palette_y_size[bsize_ctx.unwrap()],
            );
            let cache = palette_cache(&st.a_palette[x4], &st.l_palette[y4], !py.is_multiple_of(64));
            write_palette_colors(wr, &palette.colors, &cache, st.bit_depth);
        }
    }
    if palette_allowed && uv_mode == 0 {
        // No chroma palette is selected. The context is one when luma uses a
        // palette and zero otherwise.
        wr.symbol_adapt(
            0,
            &mut st.cdfs.palette_uv_mode[usize::from(palette.is_some())],
        );
    }
    if let Some(palette) = palette {
        write_palette_map(wr, &mut st.cdfs, palette);
    }
    for u in x4..x4 + n_tx {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y4..y4 + n_tx {
        st.l_mode[u] = y_mode as u8;
    }
    let stored_palette = palette.map_or_else(Vec::new, |p| p.colors.clone());
    for slot in &mut st.a_palette[x4..x4 + n_tx] {
        slot.clone_from(&stored_palette);
    }
    for slot in &mut st.l_palette[y4..y4 + n_tx] {
        slot.clone_from(&stored_palette);
    }
    let modes = [y_mode, uv_mode, uv_mode];
    let deltas = [y_delta, uv_delta, uv_delta];
    for plane in 0..3 {
        encode_plane_block(
            wr,
            &mut st.cdfs,
            planes[plane],
            st.w,
            px,
            py,
            n_tx,
            plane != 0,
            modes[plane],
            deltas[plane],
            st.base,
            st.bit_depth,
            &mut st.a_coef[plane],
            &mut st.l_coef[plane],
            if plane == 0 { palette } else { None },
            if plane == 0 || uv_mode != CFL_PRED {
                None
            } else {
                Some((planes[0], uv_alpha[plane - 1]))
            },
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn code_leaf_rect(
    wr: &mut Writer,
    planes: [&[i16]; 3],
    st: &mut LlState,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    y_mode: usize,
    uv_mode: usize,
) {
    debug_assert_ne!(width, height);
    debug_assert!(LL_RECT_MODES.contains(&y_mode));
    debug_assert!(LL_RECT_MODES.contains(&uv_mode));
    let (x4, y4, nw, nh) = (px / 4, py / 4, width / 4, height / 4);
    write_block_skip_rect(wr, st, px, py, width, height, false);
    wr.symbol_adapt(0, &mut st.cdfs.intrabc);

    let y_ctx = INTRA_MODE_CTX[st.a_mode[x4] as usize] * 5 + INTRA_MODE_CTX[st.l_mode[y4] as usize];
    wr.symbol_adapt(y_mode, &mut st.cdfs.kf_y[y_ctx]);
    let angle_allowed = width >= 8 && height >= 8;
    if angle_allowed && (1..=8).contains(&y_mode) {
        wr.symbol_adapt(3, &mut st.cdfs.angle_delta[y_mode - 1]);
    }
    wr.symbol_adapt(uv_mode, &mut st.cdfs.uv_mode[y_mode]);
    if angle_allowed && (1..=8).contains(&uv_mode) {
        wr.symbol_adapt(3, &mut st.cdfs.angle_delta[uv_mode - 1]);
    }

    let palette_allowed = width >= 8 && height >= 8;
    if palette_allowed && y_mode == 0 {
        let bctx = (width.trailing_zeros() as usize + height.trailing_zeros() as usize - 6).min(6);
        let mctx =
            usize::from(!st.a_palette[x4].is_empty()) + usize::from(!st.l_palette[y4].is_empty());
        wr.symbol_adapt(0, &mut st.cdfs.palette_y_mode[bctx][mctx]);
    }
    if palette_allowed && uv_mode == 0 {
        wr.symbol_adapt(0, &mut st.cdfs.palette_uv_mode[0]);
    }
    st.a_mode[x4..x4 + nw].fill(y_mode as u8);
    st.l_mode[y4..y4 + nh].fill(y_mode as u8);
    for slot in &mut st.a_palette[x4..x4 + nw] {
        slot.clear();
    }
    for slot in &mut st.l_palette[y4..y4 + nh] {
        slot.clear();
    }
    for (plane, &pp) in planes.iter().enumerate().take(3) {
        encode_plane_rect_block(
            wr,
            &mut st.cdfs,
            pp,
            st.w,
            px,
            py,
            width,
            height,
            plane != 0,
            if plane == 0 { y_mode } else { uv_mode },
            st.base,
            &mut st.a_coef[plane],
            &mut st.l_coef[plane],
        );
    }
}

/// Edge-aware partition recursion for lossless. A fully-in-frame 64×64
/// superblock is one `PARTITION_NONE` block (the validated path); any block
/// crossing the frame edge is split (4-way, or the constrained split-or-horz /
/// split-or-vert bool, or an implicit split) down to 8×8 leaves, all square.
fn partition_ctx(st: &LlState, bl: usize, x8: usize, y8: usize) -> usize {
    get_partition_ctx(&st.a_part, &st.l_part, bl, x8, y8)
}

fn write_partition_symbol(
    wr: &mut Writer,
    st: &mut LlState,
    bl: usize,
    x8: usize,
    y8: usize,
    symbol: usize,
) {
    let ctx = get_partition_ctx(&st.a_part, &st.l_part, bl, x8, y8);
    if bl == 4 {
        wr.symbol_adapt(symbol, &mut st.cdfs.part_bl8[ctx]);
    } else {
        wr.symbol_adapt(symbol, &mut st.cdfs.part_split[bl - 1][ctx]);
    }
}

fn part_byte(sz8: usize) -> u8 {
    match sz8 {
        1 => 0x1e,
        2 => 0x1c,
        4 => 0x18,
        _ => 0x10,
    }
}

fn part_byte_px(size: usize) -> u8 {
    match size {
        4 => 0x1f,
        8 => 0x1e,
        16 => 0x1c,
        32 => 0x18,
        _ => 0x10,
    }
}

fn mark_rect_partition(st: &mut LlState, x8: usize, y8: usize, sz8: usize, symbol: usize) {
    let size = sz8 * 8;
    let half = size / 2;
    let quarter = size / 4;
    let (above_size, left_size) = match symbol {
        1 | 4 => (size, half),
        2 | 6 => (half, size),
        5 => (half, half),
        7 => (half, half),
        8 => (size, quarter),
        9 => (quarter, size),
        _ => unreachable!("rectangular partition symbol {symbol}"),
    };
    st.a_part[x8..x8 + sz8].fill(part_byte_px(above_size));
    st.l_part[y8..y8 + sz8].fill(part_byte_px(left_size));
}

fn code_partition_block(
    wr: &mut Writer,
    planes: [&[i16]; 3],
    st: &mut LlState,
    parent_x: usize,
    parent_y: usize,
    block: &BlockDecision,
) {
    let (px, py) = (parent_x + block.dx, parent_y + block.dy);
    if block.width == block.height {
        code_leaf(
            wr,
            planes,
            st,
            px,
            py,
            block.width,
            block.y_mode,
            block.y_delta,
            block.uv_mode,
            block.uv_delta,
            block.uv_alpha,
            block.palette.as_ref(),
            block.intrabc,
        );
    } else {
        code_leaf_rect(
            wr,
            planes,
            st,
            px,
            py,
            block.width,
            block.height,
            block.y_mode,
            block.uv_mode,
        );
    }
}

/// Encode a fully-in-frame block following its adaptive `plan`.
#[allow(clippy::too_many_arguments)]
fn encode_plan(
    wr: &mut Writer,
    planes: [&[i16]; 3],
    st: &mut LlState,
    bl: usize,
    x8: usize,
    y8: usize,
    sz8: usize,
    plan: &Plan,
) {
    let (px, py) = (x8 * 8, y8 * 8);
    match plan {
        Plan::Leaf {
            y_mode,
            y_delta,
            uv_mode,
            uv_delta,
            uv_alpha,
            palette,
            intrabc,
        } => {
            write_partition_symbol(wr, st, bl, x8, y8, 0); // PARTITION_NONE
            code_leaf(
                wr,
                planes,
                st,
                px,
                py,
                sz8 * 8,
                *y_mode,
                *y_delta,
                *uv_mode,
                *uv_delta,
                *uv_alpha,
                palette.as_ref(),
                *intrabc,
            );
            let pb = part_byte(sz8);
            for u in x8..x8 + sz8 {
                st.a_part[u] = pb;
            }
            for u in y8..y8 + sz8 {
                st.l_part[u] = pb;
            }
        }
        Plan::Split(kids) => {
            write_partition_symbol(wr, st, bl, x8, y8, 3); // PARTITION_SPLIT
            if sz8 == 1 {
                for (kid, (cx, cy)) in
                    kids.iter()
                        .zip([(px, py), (px + 4, py), (px, py + 4), (px + 4, py + 4)])
                {
                    let Plan::Leaf {
                        y_mode,
                        y_delta,
                        uv_mode,
                        uv_delta,
                        uv_alpha,
                        palette,
                        intrabc,
                    } = kid
                    else {
                        unreachable!("4x4 partition child must be a leaf")
                    };
                    code_leaf(
                        wr,
                        planes,
                        st,
                        cx,
                        cy,
                        4,
                        *y_mode,
                        *y_delta,
                        *uv_mode,
                        *uv_delta,
                        *uv_alpha,
                        palette.as_ref(),
                        *intrabc,
                    );
                }
                st.a_part[x8] = 0x1f;
                st.l_part[y8] = 0x1f;
                return;
            }
            let hh = sz8 / 2;
            let corners = [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)];
            for (i, (cx, cy)) in corners.into_iter().enumerate() {
                encode_plan(wr, planes, st, bl + 1, cx, cy, hh, &kids[i]);
            }
        }
        Plan::Partition { symbol, blocks } => {
            write_partition_symbol(wr, st, bl, x8, y8, *symbol);
            for block in blocks {
                code_partition_block(wr, planes, st, px, py, block);
            }
            mark_rect_partition(st, x8, y8, sz8, *symbol);
        }
    }
}

/// Edge-region recursion (block crosses the frame boundary): force-split using
/// AV1's frame-edge partition logic down to fully-in-frame 8x8 leaves, each
/// mode-searched.
fn decode_sb_ll(
    wr: &mut Writer,
    planes: [&[i16]; 3],
    st: &mut LlState,
    bl: usize,
    x8: usize,
    y8: usize,
    sz8: usize,
) {
    let (px, py, size) = (x8 * 8, y8 * 8, sz8 * 8);
    let full = px + size <= st.w && py + size <= st.h;

    if full {
        // fully-in-frame: plan adaptively and encode
        let (_bits, plan) = plan_full(
            planes,
            st.w,
            px,
            py,
            sz8,
            st.base,
            st.bit_depth,
            st.visible_w,
            st.visible_h,
            st.angle_delta_rdo,
            &st.intrabc_index,
        );
        encode_plan(wr, planes, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        // 8x8 leaf (in-frame for multiple-of-8 dims): mode-search and code
        write_partition_symbol(wr, st, 4, x8, y8, 0); // PARTITION_NONE
        let (_b, ym, yd, uv, uvd, uva, palette, intrabc) = best_leaf(
            planes,
            st.w,
            px,
            py,
            2,
            st.base,
            st.bit_depth,
            st.visible_w,
            st.visible_h,
            st.angle_delta_rdo,
            &st.intrabc_index,
        );
        code_leaf(
            wr,
            planes,
            st,
            px,
            py,
            8,
            ym,
            yd,
            uv,
            uvd,
            uva,
            palette.as_ref(),
            intrabc,
        );
        st.a_part[x8] = 0x1e;
        st.l_part[y8] = 0x1e;
        return;
    }

    // split node crossing the frame edge
    let hh = sz8 / 2;
    let have_h = (x8 + hh) * 8 < st.w;
    let have_v = (y8 + hh) * 8 < st.h;
    if have_h && have_v {
        write_partition_symbol(wr, st, bl, x8, y8, 3); // PARTITION_SPLIT
    } else if have_h {
        let ctx = partition_ctx(st, bl, x8, y8);
        wr.bool(
            true,
            gather_split_prob(&st.cdfs.part_split[bl - 1][ctx], true),
        );
    } else if have_v {
        let ctx = partition_ctx(st, bl, x8, y8);
        wr.bool(
            true,
            gather_split_prob(&st.cdfs.part_split[bl - 1][ctx], false),
        );
    }
    for (cx, cy) in [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)] {
        if cx * 8 < st.w && cy * 8 < st.h {
            decode_sb_ll(wr, planes, st, bl + 1, cx, cy, hh);
        }
    }
}

/// Encode a lossless AV1 tile for an arbitrary frame whose width and height are
/// multiples of 8. The frame is tiled into 64×64 superblocks (raster order,
/// single tile). Fully-in-frame superblocks are one `PARTITION_NONE` 64×64
/// block of 256 `TX_4X4` `WHT` per plane (`DC_PRED`, recon == src). Superblocks
/// crossing the frame boundary are split down to 8×8 leaves using AV1's
/// frame-edge partition logic. `planes[0]`=G, `[1]`=B, `[2]`=R, each a `w*h`
/// raster.
#[allow(clippy::too_many_arguments)]
pub fn encode_tile_lossless(
    w: usize,
    h: usize,
    visible_w: usize,
    visible_h: usize,
    bit_depth: u8,
    planes: [&[i16]; 3],
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new().with_updating_cdf(updating_cdf);
    let mut st = LlState {
        w,
        h,
        visible_w,
        visible_h,
        base: 1i32 << (bit_depth - 1),
        bit_depth,
        angle_delta_rdo: speed.try_angle_deltas(),
        intrabc_index: IntrabcIndex::new(&planes, w, h),
        cdfs: Box::new(Cdfs::new_lossless(updating_cdf)),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 4],
        l_mode: vec![0; h / 4],
        a_skip: vec![0; w / 4],
        l_skip: vec![0; h / 4],
        ibc_mv: vec![None; (w / 4) * (h / 4)],
        a_palette: vec![Vec::new(); w / 4],
        l_palette: vec![Vec::new(); h / 4],
    };
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            decode_sb_ll(&mut wr, planes, &mut st, 1, sb_x / 8, sb_y / 8, 8);
        }
    }
    wr.finish()
}

// ─── Monochrome (1-plane) lossless ───────────────────────────────────────────
//
// A monochrome AV1 frame has `mono_chrome = 1`, so `NumPlanes == 1` and
// `HasChroma` is false. Relative to the 4:4:4 leaf, the decoder codes **no**
// `uv_mode` symbol and **no** chroma transform blocks. These mono functions are
// exact subsets of their 4:4:4 counterparts above — identical skip / kf_y mode /
// angle_delta / luma-coefficient coding (so the verified luma bitstream is
// reused byte-for-byte), with only the chroma elements removed. The `Plan`,
// `LlState`, partition-context and coefficient helpers are shared unchanged; the
// `uv_mode` slot in `Plan::Leaf` is carried as a `0` placeholder and ignored.

/// Best luma mode for a mono leaf, with total residual+overhead bits (no uv).
#[allow(clippy::too_many_arguments)]
fn best_leaf_mono(
    luma: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    n_tx: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (f32, usize, i32, Option<LumaPalette>, bool) {
    let mut y_mode = 0usize;
    let mut y_delta = 0i32;
    let mut yb = f32::INFINITY;
    let mut directional = [(f32::INFINITY, 0usize); 8];
    let angle_allowed = n_tx >= 2;
    let modes = if n_tx == 1 {
        &LL4_MODES[..]
    } else {
        &LL_MODES[..]
    };
    for &m in modes {
        let mut b = plane_leaf_bits(m, 0, luma, stride, px, py, n_tx, base, bit_depth);
        if angle_allowed && (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            directional[m - 1] = (b, m);
        }
        if b < yb {
            yb = b;
            y_mode = m;
            y_delta = 0;
        }
    }
    if angle_allowed && angle_delta_rdo {
        directional.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, m) in &directional[..3] {
            for delta in [-3, -2, -1, 1, 2, 3] {
                let b = plane_leaf_bits(m, delta, luma, stride, px, py, n_tx, base, bit_depth)
                    + raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], (delta + 3) as usize);
                if b < yb {
                    yb = b;
                    y_mode = m;
                    y_delta = delta;
                }
            }
        }
    }
    let palette = (n_tx >= 2 && px + n_tx * 4 <= visible_w && py + n_tx * 4 <= visible_h)
        .then(|| exact_luma_palette(luma, stride, px, py, n_tx * 4, n_tx * 4, bit_depth))
        .flatten()
        .filter(|palette| palette_best_case_bits(palette, bit_depth) < yb);
    if let Some(palette) = palette.as_ref() {
        yb = yb.min(palette_estimated_bits(palette, bit_depth, &[], 0));
    }
    // skip + y_mode (+ angle_delta); no uv_mode symbol in a mono frame.
    let ovh = 4.0;
    let regular_bits = yb + ovh;
    let ibc_margin = intrabc_rdo_margin(n_tx * 4);
    let ibc = (n_tx >= 2 && regular_bits > 8.0 + ibc_margin)
        .then(|| {
            find_exact_intrabc(
                &[luma],
                stride,
                px,
                py,
                n_tx * 4,
                stride,
                luma.len() / stride,
                intrabc_index,
            )
        })
        .flatten()
        .map(|candidate| intrabc_estimated_bits(candidate, py))
        .filter(|&bits| bits + ibc_margin < regular_bits);
    if let Some(bits) = ibc {
        (bits, 0, 0, None, true)
    } else {
        (regular_bits, y_mode, y_delta, palette, false)
    }
}

#[allow(clippy::too_many_arguments)]
fn best_partition_block_mono(
    luma: &[i16],
    stride: usize,
    parent_x: usize,
    parent_y: usize,
    dx: usize,
    dy: usize,
    width: usize,
    height: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (f32, BlockDecision) {
    let (px, py) = (parent_x + dx, parent_y + dy);
    if width == height {
        let (mut bits, mut ym, yd, palette, intrabc) = best_leaf_mono(
            luma,
            stride,
            px,
            py,
            width / 4,
            base,
            bit_depth,
            visible_w,
            visible_h,
            false,
            intrabc_index,
        );
        // A/B rectangular partitions give their square children different
        // top-right/bottom-left availability from the ordinary SPLIT tree.
        // Until those directional edge flags are carried in BlockDecision,
        // keep these children on predictors that need no extended diagonal
        // edge. Palette and IntraBC remain independently eligible.
        if palette.is_none() && !intrabc && (3..=8).contains(&ym) {
            let mut safe = (f32::INFINITY, 0usize);
            for &mode in &LL_RECT_FAST_MODES {
                let mut candidate =
                    plane_leaf_bits(mode, 0, luma, stride, px, py, width / 4, base, bit_depth);
                if (1..=2).contains(&mode) {
                    candidate += raw_symbol_cost(&ANGLE_DELTA_CDF[mode - 1], 3);
                }
                if candidate < safe.0 {
                    safe = (candidate, mode);
                }
            }
            bits = safe.0 + 4.0;
            ym = safe.1;
        }
        (
            bits,
            BlockDecision {
                dx,
                dy,
                width,
                height,
                y_mode: ym,
                y_delta: yd,
                uv_mode: 0,
                uv_delta: 0,
                uv_alpha: [0, 0],
                palette,
                intrabc,
            },
        )
    } else {
        let mut best = (f32::INFINITY, 0usize);
        let modes: &[usize] = if angle_delta_rdo {
            &LL_RECT_MODES
        } else {
            &LL_RECT_FAST_MODES
        };
        for &mode in modes {
            let bits = plane_rect_bits(mode, luma, stride, px, py, width, height, base);
            if bits < best.0 {
                best = (bits, mode);
            }
        }
        (
            best.0 + 4.0,
            BlockDecision {
                dx,
                dy,
                width,
                height,
                y_mode: best.1,
                y_delta: 0,
                uv_mode: 0,
                uv_delta: 0,
                uv_alpha: [0, 0],
                palette: None,
                intrabc: false,
            },
        )
    }
}

#[allow(clippy::type_complexity)]
fn rectangular_geometries(size: usize) -> Vec<(usize, Vec<(usize, usize, usize, usize)>)> {
    let half = size / 2;
    let mut out = vec![
        (1, vec![(0, 0, size, half), (0, half, size, half)]),
        (2, vec![(0, 0, half, size), (half, 0, half, size)]),
    ];
    if size == 8 {
        return out;
    }
    let quarter = size / 4;
    out.extend([
        (
            4,
            vec![
                (0, 0, half, half),
                (half, 0, half, half),
                (0, half, size, half),
            ],
        ),
        (
            5,
            vec![
                (0, 0, size, half),
                (0, half, half, half),
                (half, half, half, half),
            ],
        ),
        (
            6,
            vec![
                (0, 0, half, half),
                (0, half, half, half),
                (half, 0, half, size),
            ],
        ),
        (
            7,
            vec![
                (0, 0, half, size),
                (half, 0, half, half),
                (half, half, half, half),
            ],
        ),
    ]);
    if size >= 32 {
        out.extend([
            (8, (0..4).map(|i| (0, i * quarter, size, quarter)).collect()),
            (9, (0..4).map(|i| (i * quarter, 0, quarter, size)).collect()),
        ]);
    }
    out
}

/// Mono partition plan for a fully-in-frame square block (luma only).
#[allow(clippy::too_many_arguments)]
fn plan_full_mono(
    luma: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    sz8: usize,
    base: i32,
    bit_depth: u8,
    visible_w: usize,
    visible_h: usize,
    angle_delta_rdo: bool,
    intrabc_index: &IntrabcIndex,
) -> (f32, Plan) {
    let (bits_leaf, ym, yd, palette, intrabc) = best_leaf_mono(
        luma,
        stride,
        px,
        py,
        sz8 * 2,
        base,
        bit_depth,
        visible_w,
        visible_h,
        angle_delta_rdo,
        intrabc_index,
    );
    let none = PART_NONE_BITS + bits_leaf;
    let mut best = (
        none,
        Plan::Leaf {
            y_mode: ym,
            y_delta: yd,
            uv_mode: 0,
            uv_delta: 0,
            uv_alpha: [0, 0],
            palette,
            intrabc,
        },
    );
    let eval_partition = |symbol: usize, geometry: &[(usize, usize, usize, usize)]| {
        let mut bits = PART_RECT_BITS;
        let mut blocks = Vec::with_capacity(geometry.len());
        for &(dx, dy, width, height) in geometry {
            let (block_bits, block) = best_partition_block_mono(
                luma,
                stride,
                px,
                py,
                dx,
                dy,
                width,
                height,
                base,
                bit_depth,
                visible_w,
                visible_h,
                angle_delta_rdo,
                intrabc_index,
            );
            bits += block_bits;
            blocks.push(block);
        }
        (bits, Plan::Partition { symbol, blocks })
    };
    if sz8 == 1 {
        if angle_delta_rdo || forced_ll_partition() == 3 {
            let mut split = PART_SPLIT_BITS;
            let mut kids: [Option<Plan>; 4] = [None, None, None, None];
            for (i, (cx, cy)) in [(px, py), (px + 4, py), (px, py + 4), (px + 4, py + 4)]
                .into_iter()
                .enumerate()
            {
                let (b, cym, cyd, cpalette, cibc) = best_leaf_mono(
                    luma,
                    stride,
                    cx,
                    cy,
                    1,
                    base,
                    bit_depth,
                    visible_w,
                    visible_h,
                    false,
                    intrabc_index,
                );
                split += b;
                kids[i] = Some(Plan::Leaf {
                    y_mode: cym,
                    y_delta: cyd,
                    uv_mode: 0,
                    uv_delta: 0,
                    uv_alpha: [0, 0],
                    palette: cpalette,
                    intrabc: cibc,
                });
            }
            let split_plan = (split, Plan::Split(Box::new(kids.map(|kid| kid.unwrap()))));
            if forced_ll_partition() == 3 {
                return split_plan;
            }
            if split_plan.0 < best.0 {
                best = split_plan;
            }
        }
        for (symbol, geometry) in rectangular_geometries(8) {
            let candidate = eval_partition(symbol, &geometry);
            if forced_ll_partition() == symbol {
                return candidate;
            }
            if candidate.0 < best.0 {
                best = candidate;
            }
        }
        return best;
    }
    let hh = sz8 / 2;
    let mut split = PART_SPLIT_BITS;
    let mut kids: [Option<Plan>; 4] = [None, None, None, None];
    for (i, (cx, cy)) in [
        (px, py),
        (px + hh * 8, py),
        (px, py + hh * 8),
        (px + hh * 8, py + hh * 8),
    ]
    .into_iter()
    .enumerate()
    {
        let (b, p) = plan_full_mono(
            luma,
            stride,
            cx,
            cy,
            hh,
            base,
            bit_depth,
            visible_w,
            visible_h,
            angle_delta_rdo,
            intrabc_index,
        );
        split += b;
        kids[i] = Some(p);
    }
    let split_plan = (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))));
    if split_plan.0 < best.0 {
        best = split_plan;
    }
    let rect_candidates = if angle_delta_rdo { 8 } else { 2 };
    for (symbol, geometry) in rectangular_geometries(sz8 * 8)
        .into_iter()
        .take(rect_candidates)
    {
        let candidate = eval_partition(symbol, &geometry);
        if forced_ll_partition() == symbol {
            return candidate;
        }
        if candidate.0 < best.0 {
            best = candidate;
        }
    }
    best
}

/// Code a mono lossless leaf: block skip + kf_y mode (+ angle_delta) then the
/// luma `(size/4)²` `TX_4X4` WHT. No uv_mode symbol, no chroma blocks.
#[allow(clippy::too_many_arguments)]
fn code_leaf_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    px: usize,
    py: usize,
    size: usize,
    y_mode: usize,
    y_delta: i32,
    palette: Option<&LumaPalette>,
    intrabc: bool,
) {
    let n_tx = size / 4;
    let (x4, y4) = (px / 4, py / 4);
    let candidate = intrabc
        .then(|| find_exact_intrabc(&[luma], st.w, px, py, size, st.w, st.h, &st.intrabc_index))
        .flatten();
    let predictor = candidate.and_then(|_| intrabc_mv_predictor(st, px, py, size));
    let intrabc = candidate.zip(predictor).is_some_and(|(candidate, pred)| {
        (i32::from(candidate.mv.0) - i32::from(pred.0)).unsigned_abs() <= 16_384
            && (i32::from(candidate.mv.1) - i32::from(pred.1)).unsigned_abs() <= 16_384
    });
    write_block_skip(wr, st, px, py, size, intrabc);
    wr.symbol_adapt(intrabc as usize, &mut st.cdfs.intrabc);
    if intrabc {
        let candidate = candidate.unwrap();
        write_intrabc_mv(wr, &mut st.cdfs, candidate.mv, predictor.unwrap());
        st.a_mode[x4..x4 + n_tx].fill(0);
        st.l_mode[y4..y4 + n_tx].fill(0);
        for slot in &mut st.a_palette[x4..x4 + n_tx] {
            slot.clear();
        }
        for slot in &mut st.l_palette[y4..y4 + n_tx] {
            slot.clear();
        }
        let mv = candidate.mv;
        let stride4 = st.w / 4;
        for y in y4..y4 + n_tx {
            st.ibc_mv[y * stride4 + x4..y * stride4 + x4 + n_tx].fill(Some(mv));
        }
        st.a_coef[0][x4..x4 + n_tx].fill(0x40);
        st.l_coef[0][y4..y4 + n_tx].fill(0x40);
        return;
    }
    let palette = palette.filter(|palette| {
        palette_wins_live(st, luma, px, py, size, y_mode, y_delta, None, palette)
    });
    let (y_mode, y_delta) = if palette.is_some() {
        (0, 0)
    } else {
        (y_mode, y_delta)
    };
    let y_ctx = INTRA_MODE_CTX[st.a_mode[x4] as usize] * 5 + INTRA_MODE_CTX[st.l_mode[y4] as usize];
    wr.symbol_adapt(y_mode, &mut st.cdfs.kf_y[y_ctx]);
    if (1..=8).contains(&y_mode) {
        wr.symbol_adapt((y_delta + 3) as usize, &mut st.cdfs.angle_delta[y_mode - 1]);
    }
    if size >= 8 && y_mode == 0 {
        let bsize_ctx = palette_bsize_ctx(size);
        let mode_ctx =
            usize::from(!st.a_palette[x4].is_empty()) + usize::from(!st.l_palette[y4].is_empty());
        wr.symbol_adapt(
            usize::from(palette.is_some()),
            &mut st.cdfs.palette_y_mode[bsize_ctx][mode_ctx],
        );
        if let Some(palette) = palette {
            wr.symbol_adapt(
                palette.colors.len() - 2,
                &mut st.cdfs.palette_y_size[bsize_ctx],
            );
            let cache = palette_cache(&st.a_palette[x4], &st.l_palette[y4], !py.is_multiple_of(64));
            write_palette_colors(wr, &palette.colors, &cache, st.bit_depth);
            write_palette_map(wr, &mut st.cdfs, palette);
        }
    }
    // (mono: HasChroma == false ⇒ no uv_mode symbol, no chroma residual)
    for u in x4..x4 + n_tx {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y4..y4 + n_tx {
        st.l_mode[u] = y_mode as u8;
    }
    let stored_palette = palette.map_or_else(Vec::new, |p| p.colors.clone());
    for slot in &mut st.a_palette[x4..x4 + n_tx] {
        slot.clone_from(&stored_palette);
    }
    for slot in &mut st.l_palette[y4..y4 + n_tx] {
        slot.clone_from(&stored_palette);
    }
    encode_plane_block(
        wr,
        &mut st.cdfs,
        luma,
        st.w,
        px,
        py,
        n_tx,
        false, // luma
        y_mode,
        y_delta,
        st.base,
        st.bit_depth,
        &mut st.a_coef[0],
        &mut st.l_coef[0],
        palette,
        None,
    );
}

#[allow(clippy::too_many_arguments)]
fn code_leaf_rect_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    px: usize,
    py: usize,
    width: usize,
    height: usize,
    y_mode: usize,
) {
    let (x4, y4, nw, nh) = (px / 4, py / 4, width / 4, height / 4);
    write_block_skip_rect(wr, st, px, py, width, height, false);
    wr.symbol_adapt(0, &mut st.cdfs.intrabc);
    let y_ctx = INTRA_MODE_CTX[st.a_mode[x4] as usize] * 5 + INTRA_MODE_CTX[st.l_mode[y4] as usize];
    wr.symbol_adapt(y_mode, &mut st.cdfs.kf_y[y_ctx]);
    if width >= 8 && height >= 8 && (1..=8).contains(&y_mode) {
        wr.symbol_adapt(3, &mut st.cdfs.angle_delta[y_mode - 1]);
    }
    if width >= 8 && height >= 8 && y_mode == 0 {
        let bctx = (width.trailing_zeros() as usize + height.trailing_zeros() as usize - 6).min(6);
        let mctx =
            usize::from(!st.a_palette[x4].is_empty()) + usize::from(!st.l_palette[y4].is_empty());
        wr.symbol_adapt(0, &mut st.cdfs.palette_y_mode[bctx][mctx]);
    }
    st.a_mode[x4..x4 + nw].fill(y_mode as u8);
    st.l_mode[y4..y4 + nh].fill(y_mode as u8);
    for slot in &mut st.a_palette[x4..x4 + nw] {
        slot.clear();
    }
    for slot in &mut st.l_palette[y4..y4 + nh] {
        slot.clear();
    }
    encode_plane_rect_block(
        wr,
        &mut st.cdfs,
        luma,
        st.w,
        px,
        py,
        width,
        height,
        false,
        y_mode,
        st.base,
        &mut st.a_coef[0],
        &mut st.l_coef[0],
    );
}

fn code_partition_block_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    parent_x: usize,
    parent_y: usize,
    block: &BlockDecision,
) {
    let (px, py) = (parent_x + block.dx, parent_y + block.dy);
    if block.width == block.height {
        code_leaf_mono(
            wr,
            luma,
            st,
            px,
            py,
            block.width,
            block.y_mode,
            block.y_delta,
            block.palette.as_ref(),
            block.intrabc,
        );
    } else {
        code_leaf_rect_mono(
            wr,
            luma,
            st,
            px,
            py,
            block.width,
            block.height,
            block.y_mode,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn encode_plan_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    bl: usize,
    x8: usize,
    y8: usize,
    sz8: usize,
    plan: &Plan,
) {
    let (px, py) = (x8 * 8, y8 * 8);
    match plan {
        Plan::Leaf {
            y_mode,
            y_delta,
            palette,
            intrabc,
            ..
        } => {
            write_partition_symbol(wr, st, bl, x8, y8, 0); // PARTITION_NONE
            code_leaf_mono(
                wr,
                luma,
                st,
                px,
                py,
                sz8 * 8,
                *y_mode,
                *y_delta,
                palette.as_ref(),
                *intrabc,
            );
            let pb = part_byte(sz8);
            st.a_part[x8..x8 + sz8].fill(pb);
            st.l_part[y8..y8 + sz8].fill(pb);
        }
        Plan::Split(kids) => {
            write_partition_symbol(wr, st, bl, x8, y8, 3); // PARTITION_SPLIT
            if sz8 == 1 {
                for (kid, (cx, cy)) in
                    kids.iter()
                        .zip([(px, py), (px + 4, py), (px, py + 4), (px + 4, py + 4)])
                {
                    let Plan::Leaf {
                        y_mode,
                        y_delta,
                        palette,
                        intrabc,
                        ..
                    } = kid
                    else {
                        unreachable!("4x4 monochrome partition child must be a leaf")
                    };
                    code_leaf_mono(
                        wr,
                        luma,
                        st,
                        cx,
                        cy,
                        4,
                        *y_mode,
                        *y_delta,
                        palette.as_ref(),
                        *intrabc,
                    );
                }
                st.a_part[x8] = 0x1f;
                st.l_part[y8] = 0x1f;
                return;
            }
            let hh = sz8 / 2;
            let corners = [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)];
            for (i, (cx, cy)) in corners.into_iter().enumerate() {
                encode_plan_mono(wr, luma, st, bl + 1, cx, cy, hh, &kids[i]);
            }
        }
        Plan::Partition { symbol, blocks } => {
            write_partition_symbol(wr, st, bl, x8, y8, *symbol);
            for block in blocks {
                code_partition_block_mono(wr, luma, st, px, py, block);
            }
            mark_rect_partition(st, x8, y8, sz8, *symbol);
        }
    }
}

/// Mono counterpart of [`decode_sb_ll`] (frame-edge force-split, luma only).
fn decode_sb_ll_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    bl: usize,
    x8: usize,
    y8: usize,
    sz8: usize,
) {
    let (px, py, size) = (x8 * 8, y8 * 8, sz8 * 8);
    let full = px + size <= st.w && py + size <= st.h;

    if full {
        let (_bits, plan) = plan_full_mono(
            luma,
            st.w,
            px,
            py,
            sz8,
            st.base,
            st.bit_depth,
            st.visible_w,
            st.visible_h,
            st.angle_delta_rdo,
            &st.intrabc_index,
        );
        encode_plan_mono(wr, luma, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        write_partition_symbol(wr, st, 4, x8, y8, 0); // PARTITION_NONE
        let (_b, ym, yd, palette, intrabc) = best_leaf_mono(
            luma,
            st.w,
            px,
            py,
            2,
            st.base,
            st.bit_depth,
            st.visible_w,
            st.visible_h,
            st.angle_delta_rdo,
            &st.intrabc_index,
        );
        code_leaf_mono(wr, luma, st, px, py, 8, ym, yd, palette.as_ref(), intrabc);
        st.a_part[x8] = 0x1e;
        st.l_part[y8] = 0x1e;
        return;
    }

    let hh = sz8 / 2;
    let have_h = (x8 + hh) * 8 < st.w;
    let have_v = (y8 + hh) * 8 < st.h;
    if have_h && have_v {
        write_partition_symbol(wr, st, bl, x8, y8, 3); // PARTITION_SPLIT
    } else if have_h {
        let ctx = partition_ctx(st, bl, x8, y8);
        wr.bool(
            true,
            gather_split_prob(&st.cdfs.part_split[bl - 1][ctx], true),
        );
    } else if have_v {
        let ctx = partition_ctx(st, bl, x8, y8);
        wr.bool(
            true,
            gather_split_prob(&st.cdfs.part_split[bl - 1][ctx], false),
        );
    }
    for (cx, cy) in [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)] {
        if cx * 8 < st.w && cy * 8 < st.h {
            decode_sb_ll_mono(wr, luma, st, bl + 1, cx, cy, hh);
        }
    }
}

/// Encode a **monochrome** lossless AV1 tile (single luma plane), width/height
/// multiples of 8. Structure mirrors [`encode_tile_lossless`] but for a 1-plane
/// (`mono_chrome = 1`) frame: only the luma plane is coded.
#[allow(clippy::too_many_arguments)]
pub fn encode_tile_lossless_mono(
    w: usize,
    h: usize,
    visible_w: usize,
    visible_h: usize,
    bit_depth: u8,
    luma: &[i16],
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new().with_updating_cdf(updating_cdf);
    let mut st = LlState {
        w,
        h,
        visible_w,
        visible_h,
        base: 1i32 << (bit_depth - 1),
        bit_depth,
        angle_delta_rdo: speed.try_angle_deltas(),
        intrabc_index: IntrabcIndex::new(&[luma], w, h),
        cdfs: Box::new(Cdfs::new_lossless(updating_cdf)),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 4],
        l_mode: vec![0; h / 4],
        a_skip: vec![0; w / 4],
        l_skip: vec![0; h / 4],
        ibc_mv: vec![None; (w / 4) * (h / 4)],
        a_palette: vec![Vec::new(); w / 4],
        l_palette: vec![Vec::new(); h / 4],
    };
    for sb_y in (0..h).step_by(64) {
        for sb_x in (0..w).step_by(64) {
            decode_sb_ll_mono(&mut wr, luma, &mut st, 1, sb_x / 8, sb_y / 8, 8);
        }
    }
    wr.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BitDepth;
    use crate::encoder::{
        PlanarImage, encode_lossless_gray_obu, encode_lossless_obu, encode_lossy_gray_obu,
    };
    use std::path::{Path, PathBuf};
    use std::process::Command;

    fn dav1d() -> Option<PathBuf> {
        std::env::var_os("DAV1D").map(PathBuf::from).or_else(|| {
            let path = PathBuf::from("/opt/homebrew/bin/dav1d");
            path.is_file().then_some(path)
        })
    }

    fn decode_obu(decoder: &Path, obu: &[u8], tag: &str) -> Vec<u8> {
        let stem = format!("maroontree-av1-palette-{}-{tag}", std::process::id());
        let input = std::env::temp_dir().join(format!("{stem}.obu"));
        let output = std::env::temp_dir().join(format!("{stem}.yuv"));
        std::fs::write(&input, obu).unwrap();
        let result = Command::new(decoder)
            .args(["--demuxer", "section5", "--muxer", "yuv", "-q", "-o"])
            .arg(&output)
            .arg("-i")
            .arg(&input)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "dav1d rejected {tag}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let decoded = std::fs::read(&output).unwrap();
        let _ = std::fs::remove_file(input);
        let _ = std::fs::remove_file(output);
        decoded
    }

    #[test]
    fn lossless_directional_candidates_cover_all_modes_and_deltas() {
        let (w, h) = (64usize, 64usize);
        let luma: Vec<i16> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                ((x * 37 + y * 53 + x * y * 3 + 19) & 255) as i16
            })
            .collect();

        for mode in 1..=8 {
            for delta in -3..=3 {
                let bits = plane_leaf_bits(mode, delta, &luma, w, 16, 16, 2, 128, 8);
                assert!(bits.is_finite(), "mode={mode} delta={delta}");
            }
        }

        let index = IntrabcIndex::new(&[&luma], w, h);
        let (_, _, delta, _, _) = best_leaf_mono(&luma, w, 16, 16, 2, 128, 8, w, h, false, &index);
        assert_eq!(delta, 0, "Medium/Fast must not perform angle-delta RDO");
    }

    #[test]
    fn lossless_directional_edges_follow_partition_z_order() {
        assert_eq!(lossless_leaf_edge_flags(0, 0, 64), (true, false));
        assert_eq!(lossless_leaf_edge_flags(0, 0, 32), (true, true));
        assert_eq!(lossless_leaf_edge_flags(32, 0, 32), (true, false));
        assert_eq!(lossless_leaf_edge_flags(0, 32, 32), (true, false));
        assert_eq!(lossless_leaf_edge_flags(32, 32, 32), (false, false));

        // Coordinates are superblock-local for the edge tree.
        assert_eq!(lossless_leaf_edge_flags(64, 0, 32), (true, true));
    }

    #[test]
    fn lossless_cfl_444_is_selected_and_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (16usize, 8usize);
        // Every TX_4X4 has an exactly zero-mean checkerboard. U follows luma
        // with alpha +8 and V follows it with alpha -8, while all chroma DC
        // edges average to 128. This also exercises opposite joint signs.
        let y: Vec<u8> = (0..w * h)
            .map(|i| {
                if ((i % w) + (i / w)) & 1 == 0 {
                    96
                } else {
                    160
                }
            })
            .collect();
        let u = y.clone();
        let v: Vec<u8> = y.iter().map(|&sample| 0u8.wrapping_sub(sample)).collect();
        let yi: Vec<i16> = y.iter().map(|&sample| i16::from(sample)).collect();
        let ui: Vec<i16> = u.iter().map(|&sample| i16::from(sample)).collect();
        let vi: Vec<i16> = v.iter().map(|&sample| i16::from(sample)).collect();
        let planes = [&yi[..], &ui[..], &vi[..]];

        let index = IntrabcIndex::new(&planes, w, h);
        let (_, _, _, uv_mode, _, alpha, _, intrabc) =
            best_leaf(planes, w, 0, 0, 1, 128, 8, w, h, false, &index);
        assert_eq!(uv_mode, CFL_PRED);
        assert_eq!(alpha, [8, -8]);
        assert!(!intrabc);
        let (_, plan) = plan_full(planes, w, 0, 0, 1, 128, 8, w, h, false, &index);
        assert!(
            matches!(plan, Plan::Split(_)),
            "8x8 must split for lossless CfL"
        );

        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [y.clone(), u.clone(), v.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&image, None, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "lossless-cfl-444"),
            [y, u, v].concat()
        );

        // Cropped dimensions exercise padded storage while CfL itself must use
        // only the reconstructed samples of each visible/padded TX_4X4.
        let (cw, ch) = (12usize, 8usize);
        let cy: Vec<u8> = (0..cw * ch)
            .map(|i| {
                if ((i % cw) + (i / cw)) & 1 == 0 {
                    96
                } else {
                    160
                }
            })
            .collect();
        let cu = cy.clone();
        let cv: Vec<u8> = cy.iter().map(|&sample| 0u8.wrapping_sub(sample)).collect();
        let cropped = PlanarImage {
            width: cw,
            height: ch,
            bit_depth: BitDepth::Eight,
            planes: [cy.clone(), cu.clone(), cv.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&cropped, None, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "lossless-cfl-cropped"),
            [cy, cu, cv].concat()
        );

        // Same alpha pair at 10-bit verifies the signed rounding and clipping
        // path without reducing samples to eight-bit precision.
        let hy: Vec<u16> = (0..64)
            .map(|i| {
                if ((i % 8) + (i / 8)) & 1 == 0 {
                    384
                } else {
                    640
                }
            })
            .collect();
        let hu = hy.clone();
        let hv: Vec<u16> = hy.iter().map(|&sample| 1024 - sample).collect();
        let highbd = PlanarImage {
            width: 8,
            height: 8,
            bit_depth: BitDepth::Ten,
            planes: [hy.clone(), hu.clone(), hv.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&highbd, None, 1).unwrap();
        let decoded = decode_obu(&decoder, &obu, "lossless-cfl-10-bit");
        let expected: Vec<u8> = [hy, hu, hv]
            .concat()
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn exact_palette_wins_lossless_rdo_for_all_normative_sizes() {
        let (w, h) = (64usize, 64usize);
        for size in 2..=8usize {
            let luma: Vec<i16> = (0..w * h)
                .map(|i| {
                    let x = i % w;
                    let y = i / w;
                    (7 + (((x / 4) + (y / 4)) % size) * 29) as i16
                })
                .collect();
            let index = IntrabcIndex::new(&[&luma], w, h);
            let (_, mode, _, palette, _) =
                best_leaf_mono(&luma, w, 0, 0, 16, 128, 8, w, h, true, &index);
            assert_eq!(mode, 0);
            assert_eq!(palette.map(|p| p.colors.len()), Some(size), "size {size}");
        }
    }

    #[test]
    fn exact_palette_loses_when_spatial_prediction_is_cheaper() {
        let (w, h) = (64usize, 64usize);
        // This block has an exact two-entry palette, but DC predicts every
        // sample except one. Coding that single residual is far cheaper than
        // transmitting a palette index for all 4096 samples.
        let mut luma = vec![128i16; w * h];
        luma[w * h - 1] = 129;

        let index = IntrabcIndex::new(&[&luma], w, h);
        let (_, mode, _, palette, intrabc) =
            best_leaf_mono(&luma, w, 0, 0, 16, 128, 8, w, h, true, &index);
        assert_eq!(mode, 0);
        assert!(palette.is_none());
        assert!(!intrabc);
    }

    #[test]
    fn palette_rate_models_uniform_map_and_color_cache_exactly() {
        assert_eq!(uniform_symbol_bits(3, 0), 1.0);
        assert_eq!(uniform_symbol_bits(3, 1), 2.0);
        assert_eq!(uniform_symbol_bits(3, 2), 2.0);

        let colors = [10, 20];
        let uncached = palette_color_bits(&colors, &[], 8);
        assert_eq!(palette_color_bits(&colors, &colors, 8), 2.0);
        let misses: Vec<i32> = (30..46).collect();
        assert_eq!(palette_color_bits(&colors, &misses, 8), uncached + 16.0);
    }

    #[test]
    fn lossy_palette_predictor_codes_quantized_residual_and_decodes() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (8usize, 8usize);
        // Two noisy color clusters: an exact palette is impossible, but the
        // palette predictor is much closer than spatial DC and the remaining
        // error must travel through the ordinary lossy TX/quant path.
        let pixels: Vec<u8> = (0..w * h)
            .map(|i| {
                let base = if ((i % w) + (i / w)) & 1 == 0 {
                    28
                } else {
                    220
                };
                (base + (i % 17) as i32 - 8) as u8
            })
            .collect();
        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        crate::coder::LOSSY_PALETTE_EMITTED.store(0, std::sync::atomic::Ordering::Relaxed);
        crate::coder::LOSSY_PALETTE_RESIDUAL_EMITTED.store(0, std::sync::atomic::Ordering::Relaxed);
        let obu = encode_lossy_gray_obu(
            &image,
            BitDepth::Eight,
            64,
            true,
            1,
            crate::Speed::Slow,
            false,
            crate::coder::VarianceBoost::off(),
            false,
            false,
            true,
            true,
            true,
        )
        .unwrap();
        let decoded = decode_obu(&decoder, &obu, "lossy-residual");
        assert_eq!(decoded.len(), w * h);
        let max_error = decoded
            .iter()
            .zip(&pixels)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(max_error < 64, "palette residual max error was {max_error}");
        assert!(crate::coder::LOSSY_PALETTE_EMITTED.load(std::sync::atomic::Ordering::Relaxed) > 0);
        assert!(
            crate::coder::LOSSY_PALETTE_RESIDUAL_EMITTED.load(std::sync::atomic::Ordering::Relaxed)
                > 0
        );

        for n in 2..=8usize {
            // Coherent 4px vertical stripes of n colors. (A per-pixel
            // pseudo-random map was used here before the palette rate model
            // became entropy-accurate: such a map is maximally anti-correlated
            // and the context coder correctly prices it above the transform
            // path, so palette rightly LOSES on it now.)
            let exact: Vec<u8> = (0..w * h)
                .map(|i| 7 + ((((i % w) / 4 + (i / w) / 8) % n) * 31) as u8)
                .collect();
            let exact_image = PlanarImage::from_luma(w, h, BitDepth::Eight, &exact).unwrap();
            let exact_obu = encode_lossy_gray_obu(
                &exact_image,
                BitDepth::Eight,
                64,
                true,
                1,
                crate::Speed::Slow,
                false,
                crate::coder::VarianceBoost::off(),
                false,
                false,
                true,
                true,
                true,
            )
            .unwrap();
            assert_eq!(decode_obu(&decoder, &exact_obu, "lossy-exact"), exact);
        }

        // Cross a 64-pixel superblock-row boundary with adjacent palette
        // blocks. The above palette must not enter the color cache at y=64.
        let (mw, mh) = (64usize, 128usize);
        let many: Vec<u8> = (0..mw * mh)
            .map(|i| {
                let x = i % mw;
                let y = i / mw;
                let block = (x / 8) + (y / 8) * 8;
                let n = 2 + block % 7;
                7 + ((((x * 37 + y * 11 + block * 3) % n) * 31) as u8)
            })
            .collect();
        let many_image = PlanarImage::from_luma(mw, mh, BitDepth::Eight, &many).unwrap();
        let many_obu = encode_lossy_gray_obu(
            &many_image,
            BitDepth::Eight,
            64,
            true,
            1,
            crate::Speed::Slow,
            false,
            crate::coder::VarianceBoost::off(),
            false,
            false,
            true,
            true,
            true,
        )
        .unwrap();
        let many_decoded = decode_obu(&decoder, &many_obu, "lossy-many");
        let many_max_error = many_decoded
            .iter()
            .zip(&many)
            .map(|(&a, &b)| a.abs_diff(b))
            .max()
            .unwrap();
        assert!(
            many_max_error < 64,
            "multi-palette boundary max error was {many_max_error}"
        );
    }

    #[test]
    fn palette_streams_are_bit_exact_with_dav1d() {
        let Some(decoder) = dav1d() else {
            return;
        };
        // Two superblock rows exercise the normative color-cache reset at y=64.
        let (w, h) = (128usize, 128usize);
        for size in 2..=8usize {
            let pixels: Vec<u8> = (0..w * h)
                .map(|i| {
                    let x = i % w;
                    let y = i / w;
                    7 + (((x / 4) + (y / 4)) % size) as u8 * 29
                })
                .collect();
            let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
            let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
            assert_eq!(decode_obu(&decoder, &obu, &format!("{size}-color")), pixels);
        }

        let (w, h) = (70usize, 59usize);
        let cropped: Vec<u8> = (0..w * h)
            .map(|i| 19 + (((i % w) / 3 + (i / w) / 5) & 1) as u8 * 173)
            .collect();
        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &cropped).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        assert_eq!(decode_obu(&decoder, &obu, "cropped"), cropped);

        let (w, h) = (64usize, 64usize);
        let highbd: Vec<u16> = (0..w * h)
            .map(|i| 11 + ((((i % w) / 4 + (i / w) / 4) % 8) as u16 * 137))
            .collect();
        let image = PlanarImage::from_luma(w, h, BitDepth::Ten, &highbd).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        let raw = decode_obu(&decoder, &obu, "10-bit");
        let decoded: Vec<u16> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        assert_eq!(decoded, highbd);

        let y: Vec<u8> = (0..w * h)
            .map(|i| 23 + ((((i % w) / 4 + (i / w) / 4) & 1) as u8 * 181))
            .collect();
        let u = vec![79u8; w * h];
        let v = vec![167u8; w * h];
        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [y.clone(), u.clone(), v.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&image, None, 1).unwrap();
        assert_eq!(decode_obu(&decoder, &obu, "444"), [y, u, v].concat());
    }

    #[test]
    fn lossless_intrabc_default_dv_is_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (128usize, 128usize);
        let mut first_row = vec![0u8; w * 64];
        let mut state = 0x1bc0_6400u32;
        for sample in &mut first_row {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        let pixels = [first_row.clone(), first_row].concat();
        let luma_i16: Vec<i16> = pixels.iter().map(|&v| i16::from(v)).collect();
        let index = IntrabcIndex::new(&[&luma_i16], w, h);
        let (_, plan) = plan_full_mono(&luma_i16, w, 0, 64, 8, 128, 8, w, h, true, &index);
        assert!(matches!(plan, Plan::Leaf { intrabc: true, .. }));

        let mono = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let mono_obu = encode_lossless_gray_obu(&mono, true, 1).unwrap();
        assert_eq!(decode_obu(&decoder, &mono_obu, "intrabc-mono"), pixels);

        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [pixels.clone(), pixels.clone(), pixels.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&image, None, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "intrabc-444"),
            [pixels.clone(), pixels.clone(), pixels].concat()
        );
    }

    #[test]
    fn lossless_intrabc_neighbor_mv_transition_is_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        // The first SB row can use the default horizontal DV at x=320. At the
        // next SB row the fallback changes to the vertical DV, but the block
        // above remains a spatial MV candidate. A zero residual must therefore
        // not be coded against the vertical fallback at that position.
        let (w, h) = (384usize, 128usize);
        let mut first = vec![0u8; w * 64];
        let mut state = 0x1bc0_0320u32;
        for sample in &mut first {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        for y in 0..64 {
            let row = y * w;
            first.copy_within(row..row + 64, row + 320);
        }
        let pixels = [first.clone(), first].concat();
        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "intrabc-neighbor-mv-transition"),
            pixels
        );
    }

    #[test]
    fn lossless_intrabc_nondefault_dv_and_stack_prediction_are_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (256usize, 64usize);
        let mut pixels = vec![0u8; w * h];
        for y in 0..64 {
            for x in 0..64 {
                pixels[y * w + x] = ((x * 37 + y * 19 + (x ^ y) * 7) & 255) as u8;
            }
        }
        for y in 0..64 {
            for x in 64..128 {
                pixels[y * w + x] = ((x * 11 + y * 43 + 91) & 255) as u8;
            }
        }
        // The first copy uses the deliberately unaligned integer DV -125px and
        // therefore a +195px residual from the first-row fallback (-320px).
        // The second uses -128px, exercising a nonzero residual from the -125px
        // spatial-stack predictor installed by the first copy.
        for y in 0..64 {
            pixels.copy_within(y * w + 3..y * w + 67, y * w + 128);
            pixels.copy_within(y * w + 64..y * w + 128, y * w + 192);
        }
        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        let decoded = decode_obu(&decoder, &obu, "intrabc-nondefault-dv");
        if let Some(i) = decoded.iter().zip(&pixels).position(|(a, b)| a != b) {
            panic!(
                "first mismatch at ({}, {}): decoded={} source={}",
                i % w,
                i / w,
                decoded[i],
                pixels[i]
            );
        }
    }

    #[test]
    fn lossless_intrabc_indexes_small_block_beyond_old_probe_horizon() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (256usize, 128usize);
        let mut pixels = vec![0u8; w * h];
        let mut state = 0x1bc0_0008u32;
        for sample in &mut pixels {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u8;
        }
        // An unaligned 8x8 source in the preceding SB row.  From (192, 64)
        // this lies far beyond the old reverse-raster 4096-probe window and
        // the old matcher rejected 8x8 blocks before searching at all.
        for row in 0..8 {
            pixels.copy_within(row * w + 3..row * w + 11, (64 + row) * w + 192);
        }
        let luma: Vec<i16> = pixels.iter().map(|&v| i16::from(v)).collect();
        let index = IntrabcIndex::new(&[&luma], w, h);
        let found = find_exact_intrabc(&[&luma], w, 192, 64, 8, w, h, &index).unwrap();
        assert_eq!((found.ref_x, found.ref_y), (3, 0));
        let (_, _, _, _, selected) =
            best_leaf_mono(&luma, w, 192, 64, 2, 128, 8, w, h, false, &index);
        assert!(selected, "the indexed 8x8 match must enter lossless RDO");

        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "intrabc-indexed-small-far"),
            pixels
        );
    }

    #[test]
    fn lossless_mono_4x4_split_is_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (8usize, 8usize);
        let pixels: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                ((x * 31 + y * 47 + (x / 4) * 73 + (y / 4) * 109) & 255) as u8
            })
            .collect();
        let luma: Vec<i16> = pixels.iter().map(|&v| i16::from(v)).collect();
        let index = IntrabcIndex::new(&[&luma], w, h);
        FORCE_LL_PARTITION.with(|forced| forced.set(3));
        let (_, plan) = plan_full_mono(&luma, w, 0, 0, 1, 128, 8, w, h, true, &index);
        assert!(matches!(plan, Plan::Split(_)));

        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        FORCE_LL_PARTITION.with(|forced| forced.set(0));
        assert_eq!(
            decode_obu(&decoder, &obu, "lossless-mono-4x4-split"),
            pixels
        );
    }

    #[test]
    fn lossless_rectangular_partition_families_are_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (64usize, 64usize);
        let mut planes = [vec![0u8; w * h], vec![0u8; w * h], vec![0u8; w * h]];
        for (plane_index, plane) in planes.iter_mut().enumerate() {
            for y in 0..h {
                for x in 0..w {
                    plane[y * w + x] = ((x * (17 + plane_index * 6)
                        + y * (29 + plane_index * 4)
                        + (x ^ y) * 3
                        + plane_index * 71)
                        & 255) as u8;
                }
            }
        }
        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                planes[0].clone(),
                planes[1].clone(),
                planes[2].clone(),
                Vec::new(),
            ],
        };
        let expected = [planes[0].clone(), planes[1].clone(), planes[2].clone()].concat();
        for symbol in 1..=7 {
            FORCE_LL_PARTITION.with(|forced| forced.set(symbol));
            let obu = encode_lossless_obu(&image, None, 1).unwrap();
            FORCE_LL_PARTITION.with(|forced| forced.set(0));
            assert_eq!(
                decode_obu(&decoder, &obu, &format!("lossless-partition-{symbol}")),
                expected,
                "partition symbol {symbol}"
            );
        }
        let mono = PlanarImage::from_luma(w, h, BitDepth::Eight, &planes[0]).unwrap();
        for symbol in 1..=9 {
            FORCE_LL_PARTITION.with(|forced| forced.set(symbol));
            let obu = encode_lossless_gray_obu(&mono, true, 1).unwrap();
            FORCE_LL_PARTITION.with(|forced| forced.set(0));
            assert_eq!(
                decode_obu(&decoder, &obu, &format!("lossless-mono-partition-{symbol}")),
                planes[0],
                "monochrome partition symbol {symbol}"
            );
        }
    }

    #[test]
    fn lossless_mono_rectangular_partition_contexts_are_bit_exact_across_superblocks() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (128usize, 64usize);
        let pixels: Vec<u8> = (0..w * h)
            .map(|i| {
                let (x, y) = (i % w, i / w);
                ((x * 17 + y * 29 + (x ^ y) * 3) & 255) as u8
            })
            .collect();
        let mono = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        for symbol in 1..=9 {
            FORCE_LL_PARTITION.with(|forced| forced.set(symbol));
            let obu = encode_lossless_gray_obu(&mono, true, 1).unwrap();
            FORCE_LL_PARTITION.with(|forced| forced.set(0));
            assert_eq!(
                decode_obu(
                    &decoder,
                    &obu,
                    &format!("lossless-mono-partition-context-{symbol}")
                ),
                pixels,
                "monochrome partition context after symbol {symbol}"
            );
        }
    }

    #[test]
    fn lossless_rectangular_minimum_geometries_are_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (16usize, 16usize);
        let planes: [Vec<u8>; 3] = std::array::from_fn(|plane| {
            (0..w * h)
                .map(|i| ((i * (37 + plane * 10) + (i / w) * 23 + plane * 61) & 255) as u8)
                .collect()
        });
        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [
                planes[0].clone(),
                planes[1].clone(),
                planes[2].clone(),
                Vec::new(),
            ],
        };
        let expected = [planes[0].clone(), planes[1].clone(), planes[2].clone()].concat();
        let mono = PlanarImage::from_luma(w, h, BitDepth::Eight, &planes[0]).unwrap();
        for symbol in 1..=9 {
            FORCE_LL_PARTITION.with(|forced| forced.set(symbol));
            let obu = encode_lossless_obu(&image, None, 1).unwrap();
            FORCE_LL_PARTITION.with(|forced| forced.set(0));
            assert_eq!(
                decode_obu(&decoder, &obu, &format!("lossless-min-partition-{symbol}")),
                expected
            );
            FORCE_LL_PARTITION.with(|forced| forced.set(symbol));
            let obu = encode_lossless_gray_obu(&mono, true, 1).unwrap();
            FORCE_LL_PARTITION.with(|forced| forced.set(0));
            assert_eq!(
                decode_obu(
                    &decoder,
                    &obu,
                    &format!("lossless-min-mono-partition-{symbol}")
                ),
                planes[0]
            );
        }
    }

    #[test]
    fn lossless_rdo_selects_rectangular_partition_for_mixed_orientation() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (64usize, 64usize);
        let mut p = vec![0i16; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = if y < 32 {
                    ((x * 4) & 255) as i16
                } else {
                    ((y * 4) & 255) as i16
                };
            }
        }
        let planes = [&p[..], &p[..], &p[..]];
        let index = IntrabcIndex::new(&planes, w, h);
        let (_, plan) = plan_full(planes, w, 0, 0, 8, 128, 8, w, h, false, &index);
        match plan {
            Plan::Partition { symbol: 1, .. } => {}
            Plan::Partition { symbol, .. } => panic!("selected partition symbol {symbol}"),
            Plan::Leaf { .. } => panic!("selected PARTITION_NONE"),
            Plan::Split(_) => panic!("selected PARTITION_SPLIT"),
        }
        let bytes: Vec<u8> = p.iter().map(|&v| v as u8).collect();
        let image = PlanarImage {
            width: w,
            height: h,
            bit_depth: BitDepth::Eight,
            planes: [bytes.clone(), bytes.clone(), bytes.clone(), Vec::new()],
        };
        let obu = encode_lossless_obu(&image, None, 1).unwrap();
        assert_eq!(
            decode_obu(&decoder, &obu, "lossless-rdo-rect-directional"),
            [bytes.clone(), bytes.clone(), bytes].concat()
        );
    }

    #[test]
    fn lossless_d203_zero_residual_is_bit_exact() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (16usize, 8usize);
        let mut luma = vec![0i16; w * h];
        for y in 0..h {
            for x in 0..8 {
                luma[y * w + x] = ((x * 37 + y * 43 + 91) & 255) as i16;
            }
        }
        for ty in 0..2 {
            for tx in 0..2 {
                let mut pred = [0i32; 16];
                predict_lossless_4x4(
                    7,
                    0,
                    &luma,
                    w,
                    8 + tx * 4,
                    ty * 4,
                    8,
                    0,
                    8,
                    &mut pred,
                    128,
                    8,
                );
                for y in 0..4 {
                    for x in 0..4 {
                        luma[(ty * 4 + y) * w + 8 + tx * 4 + x] = pred[y * 4 + x] as i16;
                    }
                }
            }
        }
        let index = IntrabcIndex::new(&[&luma], w, h);
        let (_, mode, delta, palette, intrabc) =
            best_leaf_mono(&luma, w, 8, 0, 2, 128, 8, w, h, true, &index);
        assert_eq!(
            (mode, delta, palette.is_some(), intrabc),
            (7, 0, false, false)
        );
        let pixels: Vec<u8> = luma.iter().map(|&v| v as u8).collect();
        let image = PlanarImage::from_luma(w, h, BitDepth::Eight, &pixels).unwrap();
        let obu = encode_lossless_gray_obu(&image, true, 1).unwrap();
        assert_eq!(decode_obu(&decoder, &obu, "d203-zero"), pixels);
    }

    #[test]
    fn lossy_intrabc_64_decodes_and_copies_reconstruction() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (128usize, 128usize);
        let mut first = vec![0u16; w * 64];
        let mut state = 0x1055_1bc0u32;
        for sample in &mut first {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as u16;
        }
        let plane = [first.clone(), first].concat();
        let src = [plane.clone(), plane.clone(), plane];
        crate::coder::LOSSY_INTRABC_EMITTED.store(0, std::sync::atomic::Ordering::Relaxed);
        let pool = crate::par::Pool::new(1);
        let obu = crate::dispatch::encode_lossy_444(
            80,
            8,
            w,
            h,
            &src[0],
            &src[1],
            &src[2],
            None,
            &pool,
            crate::Speed::Slow,
            false,
            crate::coder::VarianceBoost::off(),
            false,
            false,
            true,
            true,
            true,
        );
        let decoded = decode_obu(&decoder, &obu, "lossy-intrabc-64");
        assert_eq!(decoded.len(), 3 * w * h);
        for decoded_plane in decoded.chunks_exact(w * h) {
            assert_eq!(&decoded_plane[..w * 64], &decoded_plane[w * 64..]);
        }
        assert!(crate::coder::LOSSY_INTRABC_EMITTED.load(std::sync::atomic::Ordering::Relaxed) > 0);
    }

    #[test]
    fn lossy_intrabc_nondefault_dv_decodes_and_uses_reconstruction() {
        let Some(decoder) = dav1d() else {
            return;
        };
        let (w, h) = (256usize, 64usize);
        let mut plane = vec![0u16; w * h];
        for y in 0..h {
            for x in 0..128 {
                plane[y * w + x] = ((x * 29 + y * 47 + (x ^ y) * 3) & 255) as u16;
            }
            plane.copy_within(y * w + 3..y * w + 67, y * w + 128);
            plane.copy_within(y * w + 64..y * w + 128, y * w + 192);
        }
        let src = [plane.clone(), plane.clone(), plane];
        crate::coder::LOSSY_INTRABC_EMITTED.store(0, std::sync::atomic::Ordering::Relaxed);
        let pool = crate::par::Pool::new(1);
        let obu = crate::dispatch::encode_lossy_444(
            80,
            8,
            w,
            h,
            &src[0],
            &src[1],
            &src[2],
            None,
            &pool,
            crate::Speed::Slow,
            false,
            crate::coder::VarianceBoost::off(),
            false,
            false,
            true,
            true,
            true,
        );
        let decoded = decode_obu(&decoder, &obu, "lossy-intrabc-nondefault-dv");
        for decoded_plane in decoded.chunks_exact(w * h) {
            for y in 0..h {
                assert_eq!(
                    &decoded_plane[y * w + 3..y * w + 67],
                    &decoded_plane[y * w + 128..y * w + 192]
                );
                assert_eq!(
                    &decoded_plane[y * w + 64..y * w + 128],
                    &decoded_plane[y * w + 192..y * w + 256]
                );
            }
        }
        assert!(
            crate::coder::LOSSY_INTRABC_EMITTED.load(std::sync::atomic::Ordering::Relaxed) >= 2
        );
    }

    #[test]
    fn mono_lossless_deterministic_nonempty() {
        let (w, h) = (72usize, 64usize); // includes a frame-edge split column
        let mut y = vec![0i16; w * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = (((i * 5 + j * 3) % 47) as i16) - 8;
            }
        }
        let a = encode_tile_lossless_mono(w, h, w, h, 8, &y, Speed::Slow, true);
        let b = encode_tile_lossless_mono(w, h, w, h, 8, &y, Speed::Slow, true);
        assert!(!a.is_empty(), "mono lossless output must be non-empty");
        assert_eq!(a, b, "mono lossless must be deterministic");
        // A mono leaf omits the uv_mode symbol + 2 chroma planes, so for the same
        // luma it must be strictly smaller than the 4:4:4 tile carrying that luma
        // in all three planes.
        let c444 = encode_tile_lossless(w, h, w, h, 8, [&y, &y, &y], Speed::Slow, true);
        assert!(
            a.len() < c444.len(),
            "mono ({}) must be smaller than 4:4:4 ({})",
            a.len(),
            c444.len()
        );
    }

    /// 10- and 12-bit mono lossless must encode without panicking and stay
    /// deterministic (the `base` pivot is `1<<(bd-1)`).
    #[test]
    fn mono_lossless_highbd_runs() {
        let (w, h) = (64usize, 64usize);
        for &bd in &[10u8, 12u8] {
            let maxv = (1i32 << bd) - 1;
            let mut y = vec![0i16; w * h];
            for j in 0..h {
                for i in 0..w {
                    y[j * w + i] = (((i * 17 + j * 11) as i32) % (maxv + 1)) as i16;
                }
            }
            let p = encode_tile_lossless_mono(w, h, w, h, bd, &y, Speed::Slow, true);
            assert!(!p.is_empty());
            assert_eq!(
                p,
                encode_tile_lossless_mono(w, h, w, h, bd, &y, Speed::Slow, true)
            );
        }
    }
}
