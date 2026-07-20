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
use crate::cdf_tables as C;
use crate::coefs::encode_coefs;
use crate::cost::coef_rate_bits;
use crate::intrapred::{INTRA_MODE_CTX, intra_predict_nd_ad_i16, palette_pred};
use crate::msac_enc::Writer;
use crate::skip_tables::SKIP_CTX;
use crate::tables::*;
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

fn palette_estimated_bits(palette: &LumaPalette, bit_depth: u8) -> f32 {
    let pixels = palette.width * palette.height;
    let size = palette.colors.len();

    // Palette mode and size use fixed frame CDFs in the lossless path. The
    // neighbour palette context is unavailable while planning, so use the
    // neutral (no cached neighbour) mode context.
    let bsize_ctx = palette_bsize_ctx(palette.width);
    let mode_bits = raw_symbol_cost(&[palette_y_mode_raw(bsize_ctx, 0)], 1)
        + raw_symbol_cost(palette_y_size_raw(bsize_ctx), size - 2);

    // Price the no-cache color-delta syntax exactly. A live cache can only
    // make this candidate cheaper, so this remains conservative.
    let colors: Vec<u32> = palette.colors.iter().map(|&color| color as u32).collect();
    let mut color_bits = bit_depth as f32;
    if colors.len() > 1 {
        let max_delta = colors.windows(2).map(|v| v[1] - v[0]).max().unwrap();
        let min_bits = bit_depth - 3;
        let mut bits = ceil_log2(max_delta).max(min_bits);
        color_bits += 2.0;
        let mut range = (1u32 << bit_depth) - colors[0] - 1;
        for pair in colors.windows(2) {
            color_bits += f32::from(bits);
            let delta = pair[1] - pair[0];
            range -= delta;
            bits = bits.min(ceil_log2(range));
        }
    }

    // The first index is uniform. Remaining indices use AV1's spatial palette
    // contexts and therefore can be much cheaper than log2(size) on runs.
    let mut map_bits = (size as f32).log2();
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

#[inline]
fn raw_symbol_cost(raw: &[u16], symbol: usize) -> f32 {
    let low = if symbol == 0 { 0 } else { raw[symbol - 1] };
    let high = raw.get(symbol).copied().unwrap_or(32768);
    -((high - low).max(1) as f32 / 32768.0).log2()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct IntrabcMatch {
    ref_x: usize,
    ref_y: usize,
    mv: (i16, i16),
}

/// Find an exact block match in the portion of the tile that AV1 has decoded.
/// The default DV is tried first, followed by a reverse-raster hash search over
/// every integer-pixel position. Reverse raster favours short DVs and
/// the 4096-probe cap bounds photographic-image work without restricting the
/// displacement represented in the bitstream.
fn find_exact_intrabc(
    planes: &[&[i16]],
    stride: usize,
    px: usize,
    py: usize,
    size: usize,
    width: usize,
    height: usize,
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
        (i16::try_from(dy).ok())
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
    // Exhaustive integer-position probing is reserved for the large blocks
    // where IntraBC amortizes its DV syntax. Small blocks retain the cheap
    // default-DV probe above and avoid multiplying partition-search work.
    if size < 32 {
        return None;
    }

    // Four widely-spaced samples are the hash gate; a hit is always verified
    // across every sample and plane above.
    let p = planes[0];
    let fingerprint = |x: usize, y: usize| {
        let q = size - 1;
        [
            p[y * stride + x],
            p[y * stride + x + q],
            p[(y + q) * stride + x],
            p[(y + q) * stride + x + q],
        ]
    };
    let wanted = fingerprint(px, py);
    let mut probes = 0usize;
    for ref_y in (0..=py.min(height - size)).rev() {
        for ref_x in (0..=width - size).rev() {
            if !legal(ref_x, ref_y) {
                continue;
            }
            probes += 1;
            if fingerprint(ref_x, ref_y) == wanted
                && exact(ref_x, ref_y)
                && let Some(found) = make(ref_x, ref_y)
            {
                return Some(found);
            }
            if probes == 4096 {
                return None;
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

/// Resolve a predictor without porting decoder candidate weights: all IntraBC
/// candidates in the decoder's enclosing spatial search area must agree. With
/// no spatial candidate, use the normative fallback DV. Mixed stacks are
/// rejected conservatively instead of risking a predictor mismatch.
fn intrabc_mv_predictor(st: &LlState, px: usize, py: usize, size: usize) -> Option<(i16, i16)> {
    let (x4, y4, n4) = (px / 4, py / 4, size / 4);
    let x_start = x4.saturating_sub(5);
    let x_end = (x4 + n4).min(st.w / 4 - 1);
    let y_start = y4.saturating_sub(5);
    let y_end = (y4 + n4).min(st.h / 4 - 1);

    let mut found = None;
    for y in y_start..y4 {
        for x in x4.saturating_sub(1)..=x_end {
            if let Some(mv) = st.ibc_mv[y * (st.w / 4) + x] {
                if found.is_some_and(|old| old != mv) {
                    return None;
                }
                found = Some(mv);
            }
        }
    }
    for x in x_start..x4 {
        for y in y4.saturating_sub(1)..=y_end {
            if let Some(mv) = st.ibc_mv[y * (st.w / 4) + x] {
                if found.is_some_and(|old| old != mv) {
                    return None;
                }
                found = Some(mv);
            }
        }
    }
    Some(found.unwrap_or_else(|| intrabc_default_mv(py)))
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
    debug_assert!(matches!(block_size, 8 | 16 | 32 | 64));
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
/// partition CDF (cdf_tables already store dav1d's inverse-cdf entries).
fn gather_split_prob(cdf: &[u16; 10], top: bool) -> u16 {
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

            let skip_ctx = if !chroma {
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
            let res_ctx = encode_coefs(w, chroma, &resid, skip_ctx, dc_sign_ctx);
            a[ax] = res_ctx;
            l[ly] = res_ctx;
        }
    }
}

/// Residual bits (coef-rate proxy) of coding one plane of an `n_tx`x`n_tx` leaf
/// at `(bx, by)` with `mode`.
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

#[allow(clippy::too_many_arguments)]
/// Best luma + best uv mode for a leaf, with total residual+overhead bits.
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
) -> (f32, usize, i32, usize, i32, Option<LumaPalette>, bool) {
    let mut y_mode = 0usize;
    let mut y_delta = 0i32;
    let mut yb = f32::INFINITY;
    let mut y_directional = [(f32::INFINITY, 0usize); 8];
    for &m in LL_MODES.iter() {
        let mut b = plane_leaf_bits(m, 0, planes[0], stride, px, py, n_tx, base, bit_depth);
        if (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            y_directional[m - 1] = (b, m);
        }
        if b < yb {
            yb = b;
            y_mode = m;
            y_delta = 0;
        }
    }
    if angle_delta_rdo {
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
    let mut ub = f32::INFINITY;
    let mut uv_directional = [(f32::INFINITY, 0usize); 8];
    for &m in LL_MODES.iter() {
        let mut b = plane_leaf_bits(m, 0, planes[1], stride, px, py, n_tx, base, bit_depth)
            + plane_leaf_bits(m, 0, planes[2], stride, px, py, n_tx, base, bit_depth);
        if (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            uv_directional[m - 1] = (b, m);
        }
        if b < ub {
            ub = b;
            uv_mode = m;
            uv_delta = 0;
        }
    }
    if angle_delta_rdo {
        uv_directional.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, m) in &uv_directional[..3] {
            for delta in [-3, -2, -1, 1, 2, 3] {
                let b = plane_leaf_bits(m, delta, planes[1], stride, px, py, n_tx, base, bit_depth)
                    + plane_leaf_bits(m, delta, planes[2], stride, px, py, n_tx, base, bit_depth)
                    + raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], (delta + 3) as usize);
                if b < ub {
                    ub = b;
                    uv_mode = m;
                    uv_delta = delta;
                }
            }
        }
    }
    let palette = (px + n_tx * 4 <= visible_w && py + n_tx * 4 <= visible_h)
        .then(|| exact_luma_palette(planes[0], stride, px, py, n_tx * 4, n_tx * 4, bit_depth))
        .flatten()
        .filter(|palette| palette_estimated_bits(palette, bit_depth) < yb);
    if let Some(palette) = palette.as_ref() {
        yb = palette_estimated_bits(palette, bit_depth);
        y_mode = 0;
        y_delta = 0;
    }
    let ovh = 7.0; // skip + y_mode + uv_mode (angle-delta rate is above)
    let regular_bits = yb + ub + ovh;
    // skip=1 + use_intrabc=1 + zero MV-joint. The exact static CDF cost is
    // close to seven bits; eight is deliberately conservative so ordinary
    // intra/palette still wins when it is equally compact.
    const INTRABC_BITS: f32 = 8.0;
    let ibc = find_exact_intrabc(
        &planes,
        stride,
        px,
        py,
        n_tx * 4,
        stride,
        planes[0].len() / stride,
    )
    .is_some()
        && INTRABC_BITS < regular_bits;
    if ibc {
        (INTRABC_BITS, 0, 0, 0, 0, None, true)
    } else {
        (
            regular_bits,
            y_mode,
            y_delta,
            uv_mode,
            uv_delta,
            palette,
            false,
        )
    }
}

/// Adaptive partition plan for a fully-in-frame square block.
enum Plan {
    Leaf {
        y_mode: usize,
        y_delta: i32,
        uv_mode: usize,
        uv_delta: i32,
        palette: Option<LumaPalette>,
        intrabc: bool,
    },
    Split(Box<[Plan; 4]>),
}

const PART_NONE_BITS: f32 = 1.0;
const PART_SPLIT_BITS: f32 = 1.5;

/// Decide none-vs-split by estimated bits; returns the plan and its cost. Min
/// leaf is 8x8 (sz8 == 1).
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
) -> (f32, Plan) {
    let (bits_leaf, ym, yd, uv, uvd, palette, intrabc) = best_leaf(
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
    );
    let none = PART_NONE_BITS + bits_leaf;
    if sz8 == 1 {
        return (
            none,
            Plan::Leaf {
                y_mode: ym,
                y_delta: yd,
                uv_mode: uv,
                uv_delta: uvd,
                palette,
                intrabc,
            },
        );
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
        );
        split += b;
        kids[i] = Some(p);
    }
    if none <= split {
        (
            none,
            Plan::Leaf {
                y_mode: ym,
                y_delta: yd,
                uv_mode: uv,
                uv_delta: uvd,
                palette,
                intrabc,
            },
        )
    } else {
        (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))))
    }
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
    a_coef: [Vec<u8>; 3],
    l_coef: [Vec<u8>; 3],
    a_part: Vec<u8>,
    l_part: Vec<u8>,
    a_mode: Vec<u8>, // luma y_mode per 8px unit (for kf_y context)
    l_mode: Vec<u8>,
    a_skip: Vec<u8>, // block skip flags per 4px unit
    l_skip: Vec<u8>,
    ibc_mv: Vec<Option<(i16, i16)>>, // decoded IntraBC MVs per 4x4 unit
    a_palette: Vec<Vec<i32>>,
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
    const RAW: [u16; 3] = [31671, 16515, 4576];
    let (x4, y4, n4) = (px / 4, py / 4, size / 4);
    let ctx = usize::from(st.a_skip[x4]) + usize::from(st.l_skip[y4]);
    write_raw_symbol(wr, usize::from(skip), &[RAW[ctx]]);
    st.a_skip[x4..x4 + n4].fill(skip as u8);
    st.l_skip[y4..y4 + n4].fill(skip as u8);
}

fn write_raw_symbol(wr: &mut Writer, symbol: usize, raw: &[u16]) {
    let mut cdf = Vec::with_capacity(raw.len() + 1);
    cdf.extend(raw.iter().map(|&v| 32768 - v));
    cdf.push(0);
    wr.symbol(symbol as u32, &cdf);
}

fn write_intrabc_mv_component(wr: &mut Writer, diff: i32) {
    debug_assert_ne!(diff, 0);
    debug_assert_eq!(diff & 7, 0);
    const CLASSES: [u16; 10] = [
        28672, 30976, 31858, 32320, 32551, 32656, 32740, 32757, 32762, 32767,
    ];
    const CLASS_N: [u16; 10] = [
        17408, 17920, 18944, 20480, 22528, 24576, 28672, 29952, 29952, 30720,
    ];
    write_raw_symbol(wr, usize::from(diff < 0), &[16384]);
    let up = diff.unsigned_abs() as usize / 8 - 1;
    let class = if up <= 1 {
        0
    } else {
        usize::BITS as usize - 1 - up.leading_zeros() as usize
    };
    debug_assert!(class <= 10);
    write_raw_symbol(wr, class, &CLASSES);
    if class == 0 {
        write_raw_symbol(wr, up, &[27648]);
    } else {
        for (n, &cdf) in CLASS_N.iter().enumerate().take(class) {
            write_raw_symbol(wr, (up >> n) & 1, &[cdf]);
        }
    }
}

fn write_intrabc_mv(wr: &mut Writer, mv: (i16, i16), pred: (i16, i16)) {
    let dy = i32::from(mv.0) - i32::from(pred.0);
    let dx = i32::from(mv.1) - i32::from(pred.1);
    let joint = usize::from(dx != 0) | (usize::from(dy != 0) << 1);
    write_raw_symbol(wr, joint, &[4096, 11264, 19328]);
    if dy != 0 {
        write_intrabc_mv_component(wr, dy);
    }
    if dx != 0 {
        write_intrabc_mv_component(wr, dx);
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
    let max_delta = out.windows(2).map(|v| v[1] - v[0]).max().unwrap();
    let min_bits = bit_depth - 3;
    let mut bits = ceil_log2(max_delta).max(min_bits);
    wr.literal(2, u32::from(bits - min_bits));
    let mut range = (1u32 << bit_depth) - out[0] - 1;
    for pair in out.windows(2) {
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

fn write_palette_map(wr: &mut Writer, palette: &LumaPalette) {
    let size = palette.colors.len();
    write_uniform(wr, size, palette.map[0] as usize);
    for diagonal in 1..palette.width + palette.height - 1 {
        let first_x = diagonal.min(palette.width - 1);
        let last_x = diagonal.saturating_sub(palette.height - 1);
        for x in (last_x..=first_x).rev() {
            let y = diagonal - x;
            let (ctx, symbol) = palette_color_ctx(&palette.map, palette.width, y, x, size);
            write_raw_symbol(wr, symbol, palette_y_color_raw(size, ctx));
        }
    }
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
    palette: Option<&LumaPalette>,
    intrabc: bool,
) {
    let n_tx = size / 4;
    let (x8, y8) = (px / 8, py / 8);
    let candidate = intrabc
        .then(|| find_exact_intrabc(&planes, st.w, px, py, size, st.w, st.h))
        .flatten();
    let predictor = candidate.and_then(|_| intrabc_mv_predictor(st, px, py, size));
    let intrabc = candidate.zip(predictor).is_some_and(|(candidate, pred)| {
        (i32::from(candidate.mv.0) - i32::from(pred.0)).unsigned_abs() <= 16_384
            && (i32::from(candidate.mv.1) - i32::from(pred.1)).unsigned_abs() <= 16_384
    });
    write_block_skip(wr, st, px, py, size, intrabc);
    // `allow_intrabc` is frame-wide, so every block carries this flag. The
    // default CDF strongly favours ordinary intra blocks.
    write_raw_symbol(wr, intrabc as usize, &[30531]);
    if intrabc {
        let candidate = candidate.unwrap();
        write_intrabc_mv(wr, candidate.mv, predictor.unwrap());

        let u8sz = size / 8;
        st.a_mode[x8..x8 + u8sz].fill(0);
        st.l_mode[y8..y8 + u8sz].fill(0);
        for slot in &mut st.a_palette[x8..x8 + u8sz] {
            slot.clear();
        }
        for slot in &mut st.l_palette[y8..y8 + u8sz] {
            slot.clear();
        }
        let (x4, y4) = (px / 4, py / 4);
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

    let kfy = icdf13(
        &KF_Y_MODE_CDF[INTRA_MODE_CTX[st.a_mode[x8] as usize]]
            [INTRA_MODE_CTX[st.l_mode[y8] as usize]],
    );
    wr.symbol(y_mode as u32, &kfy);
    if (1..=8).contains(&y_mode) {
        wr.symbol((y_delta + 3) as u32, &icdf7(&ANGLE_DELTA_CDF[y_mode - 1]));
    }
    let uvc = icdf13(&UV_MODE_NOCFL_CDF[y_mode]);
    wr.symbol(uv_mode as u32, &uvc);
    if (1..=8).contains(&uv_mode) {
        wr.symbol((uv_delta + 3) as u32, &icdf7(&ANGLE_DELTA_CDF[uv_mode - 1]));
    }
    let bsize_ctx = palette_bsize_ctx(size);
    let mode_ctx =
        usize::from(!st.a_palette[x8].is_empty()) + usize::from(!st.l_palette[y8].is_empty());
    if y_mode == 0 {
        write_raw_symbol(
            wr,
            usize::from(palette.is_some()),
            &[palette_y_mode_raw(bsize_ctx, mode_ctx)],
        );
        if let Some(palette) = palette {
            write_raw_symbol(wr, palette.colors.len() - 2, palette_y_size_raw(bsize_ctx));
            let cache = palette_cache(&st.a_palette[x8], &st.l_palette[y8], !py.is_multiple_of(64));
            write_palette_colors(wr, &palette.colors, &cache, st.bit_depth);
        }
    }
    if uv_mode == 0 {
        // No chroma palette is selected. The context is one when luma uses a
        // palette and zero otherwise.
        let raw = if palette.is_some() { 21488 } else { 32461 };
        write_raw_symbol(wr, 0, &[raw]);
    }
    if let Some(palette) = palette {
        write_palette_map(wr, palette);
    }
    let u8sz = size / 8;
    for u in x8..x8 + u8sz {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y8..y8 + u8sz {
        st.l_mode[u] = y_mode as u8;
    }
    let stored_palette = palette.map_or_else(Vec::new, |p| p.colors.clone());
    for slot in &mut st.a_palette[x8..x8 + u8sz] {
        slot.clone_from(&stored_palette);
    }
    for slot in &mut st.l_palette[y8..y8 + u8sz] {
        slot.clone_from(&stored_palette);
    }
    let modes = [y_mode, uv_mode, uv_mode];
    let deltas = [y_delta, uv_delta, uv_delta];
    for plane in 0..3 {
        encode_plane_block(
            wr,
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
        );
    }
}

/// Convert a 12-entry dav1d raw CDF to the writer's inverse-cdf form (`32768-p`,
/// trailing 0).
#[inline]
fn icdf13(raw: &[u16; 12]) -> [u16; 13] {
    let mut o = [0u16; 13];
    for (ov, &rv) in o.iter_mut().zip(raw.iter()) {
        *ov = 32768 - rv;
    }
    o
}

/// Convert a 6-entry dav1d raw CDF (angle_delta, 7 symbols) to inverse-cdf form.
#[inline]
fn icdf7(raw: &[u16; 6]) -> [u16; 7] {
    let mut o = [0u16; 7];
    for (ov, &rv) in o.iter_mut().zip(raw.iter()) {
        *ov = 32768 - rv;
    }
    o
}

/// Edge-aware partition recursion for lossless. A fully-in-frame 64×64
/// superblock is one `PARTITION_NONE` block (the validated path); any block
/// crossing the frame edge is split (4-way, or the constrained split-or-horz /
/// split-or-vert bool, or an implicit split) down to 8×8 leaves, all square.
/// Partition CDF for a node at level `bl` and the given context.
fn part_cdf(st: &LlState, bl: usize, x8: usize, y8: usize) -> &[u16] {
    let ctx = get_partition_ctx(&st.a_part, &st.l_part, bl, x8, y8);
    match bl {
        1 => &C::PART_SPLIT_64[ctx],
        2 => &C::PART_SPLIT_32[ctx],
        3 => &C::PART_SPLIT_16[ctx],
        _ => &C::PART_8[ctx],
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
            palette,
            intrabc,
        } => {
            wr.symbol(0, part_cdf(st, bl, x8, y8)); // PARTITION_NONE
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
            wr.symbol(3, part_cdf(st, bl, x8, y8)); // PARTITION_SPLIT
            let hh = sz8 / 2;
            let corners = [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)];
            for (i, (cx, cy)) in corners.into_iter().enumerate() {
                encode_plan(wr, planes, st, bl + 1, cx, cy, hh, &kids[i]);
            }
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
        );
        encode_plan(wr, planes, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        // 8x8 leaf (in-frame for multiple-of-8 dims): mode-search and code
        let ctx = get_partition_ctx(&st.a_part, &st.l_part, 4, x8, y8);
        wr.symbol(0, &C::PART_8[ctx]); // PARTITION_NONE
        let (_b, ym, yd, uv, uvd, palette, intrabc) = best_leaf(
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
    let cdf = part_cdf(st, bl, x8, y8);
    if have_h && have_v {
        wr.symbol(3, cdf); // PARTITION_SPLIT
    } else if have_h {
        wr.bool(true, gather_split_prob(cdf.try_into().unwrap(), true));
    } else if have_v {
        wr.bool(true, gather_split_prob(cdf.try_into().unwrap(), false));
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
pub fn encode_tile_lossless(
    w: usize,
    h: usize,
    visible_w: usize,
    visible_h: usize,
    bit_depth: u8,
    planes: [&[i16]; 3],
    speed: Speed,
) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new();
    let mut st = LlState {
        w,
        h,
        visible_w,
        visible_h,
        base: 1i32 << (bit_depth - 1),
        bit_depth,
        angle_delta_rdo: speed.try_angle_deltas(),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 8],
        l_mode: vec![0; h / 8],
        a_skip: vec![0; w / 4],
        l_skip: vec![0; h / 4],
        ibc_mv: vec![None; (w / 4) * (h / 4)],
        a_palette: vec![Vec::new(); w / 8],
        l_palette: vec![Vec::new(); h / 8],
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
) -> (f32, usize, i32, Option<LumaPalette>, bool) {
    let mut y_mode = 0usize;
    let mut y_delta = 0i32;
    let mut yb = f32::INFINITY;
    let mut directional = [(f32::INFINITY, 0usize); 8];
    for &m in LL_MODES.iter() {
        let mut b = plane_leaf_bits(m, 0, luma, stride, px, py, n_tx, base, bit_depth);
        if (1..=8).contains(&m) {
            b += raw_symbol_cost(&ANGLE_DELTA_CDF[m - 1], 3);
            directional[m - 1] = (b, m);
        }
        if b < yb {
            yb = b;
            y_mode = m;
            y_delta = 0;
        }
    }
    if angle_delta_rdo {
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
    let palette = (px + n_tx * 4 <= visible_w && py + n_tx * 4 <= visible_h)
        .then(|| exact_luma_palette(luma, stride, px, py, n_tx * 4, n_tx * 4, bit_depth))
        .flatten()
        .filter(|palette| palette_estimated_bits(palette, bit_depth) < yb);
    if let Some(palette) = palette.as_ref() {
        yb = palette_estimated_bits(palette, bit_depth);
        y_mode = 0;
        y_delta = 0;
    }
    // skip + y_mode (+ angle_delta); no uv_mode symbol in a mono frame.
    let ovh = 4.0;
    let regular_bits = yb + ovh;
    const INTRABC_BITS: f32 = 8.0;
    let ibc = find_exact_intrabc(
        &[luma],
        stride,
        px,
        py,
        n_tx * 4,
        stride,
        luma.len() / stride,
    )
    .is_some()
        && INTRABC_BITS < regular_bits;
    if ibc {
        (INTRABC_BITS, 0, 0, None, true)
    } else {
        (regular_bits, y_mode, y_delta, palette, false)
    }
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
    );
    let none = PART_NONE_BITS + bits_leaf;
    if sz8 == 1 {
        return (
            none,
            Plan::Leaf {
                y_mode: ym,
                y_delta: yd,
                uv_mode: 0,
                uv_delta: 0,
                palette,
                intrabc,
            },
        );
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
        );
        split += b;
        kids[i] = Some(p);
    }
    if none <= split {
        (
            none,
            Plan::Leaf {
                y_mode: ym,
                y_delta: yd,
                uv_mode: 0,
                uv_delta: 0,
                palette,
                intrabc,
            },
        )
    } else {
        (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))))
    }
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
    let (x8, y8) = (px / 8, py / 8);
    let candidate = intrabc
        .then(|| find_exact_intrabc(&[luma], st.w, px, py, size, st.w, st.h))
        .flatten();
    let predictor = candidate.and_then(|_| intrabc_mv_predictor(st, px, py, size));
    let intrabc = candidate.zip(predictor).is_some_and(|(candidate, pred)| {
        (i32::from(candidate.mv.0) - i32::from(pred.0)).unsigned_abs() <= 16_384
            && (i32::from(candidate.mv.1) - i32::from(pred.1)).unsigned_abs() <= 16_384
    });
    write_block_skip(wr, st, px, py, size, intrabc);
    write_raw_symbol(wr, intrabc as usize, &[30531]);
    if intrabc {
        let candidate = candidate.unwrap();
        write_intrabc_mv(wr, candidate.mv, predictor.unwrap());
        let u8sz = size / 8;
        st.a_mode[x8..x8 + u8sz].fill(0);
        st.l_mode[y8..y8 + u8sz].fill(0);
        for slot in &mut st.a_palette[x8..x8 + u8sz] {
            slot.clear();
        }
        for slot in &mut st.l_palette[y8..y8 + u8sz] {
            slot.clear();
        }
        let (x4, y4) = (px / 4, py / 4);
        let mv = candidate.mv;
        let stride4 = st.w / 4;
        for y in y4..y4 + n_tx {
            st.ibc_mv[y * stride4 + x4..y * stride4 + x4 + n_tx].fill(Some(mv));
        }
        st.a_coef[0][x4..x4 + n_tx].fill(0x40);
        st.l_coef[0][y4..y4 + n_tx].fill(0x40);
        return;
    }
    let kfy = icdf13(
        &KF_Y_MODE_CDF[INTRA_MODE_CTX[st.a_mode[x8] as usize]]
            [INTRA_MODE_CTX[st.l_mode[y8] as usize]],
    );
    wr.symbol(y_mode as u32, &kfy);
    if (1..=8).contains(&y_mode) {
        wr.symbol((y_delta + 3) as u32, &icdf7(&ANGLE_DELTA_CDF[y_mode - 1]));
    }
    let bsize_ctx = palette_bsize_ctx(size);
    let mode_ctx =
        usize::from(!st.a_palette[x8].is_empty()) + usize::from(!st.l_palette[y8].is_empty());
    if y_mode == 0 {
        write_raw_symbol(
            wr,
            usize::from(palette.is_some()),
            &[palette_y_mode_raw(bsize_ctx, mode_ctx)],
        );
        if let Some(palette) = palette {
            write_raw_symbol(wr, palette.colors.len() - 2, palette_y_size_raw(bsize_ctx));
            let cache = palette_cache(&st.a_palette[x8], &st.l_palette[y8], !py.is_multiple_of(64));
            write_palette_colors(wr, &palette.colors, &cache, st.bit_depth);
            write_palette_map(wr, palette);
        }
    }
    // (mono: HasChroma == false ⇒ no uv_mode symbol, no chroma residual)
    let u8sz = size / 8;
    for u in x8..x8 + u8sz {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y8..y8 + u8sz {
        st.l_mode[u] = y_mode as u8;
    }
    let stored_palette = palette.map_or_else(Vec::new, |p| p.colors.clone());
    for slot in &mut st.a_palette[x8..x8 + u8sz] {
        slot.clone_from(&stored_palette);
    }
    for slot in &mut st.l_palette[y8..y8 + u8sz] {
        slot.clone_from(&stored_palette);
    }
    encode_plane_block(
        wr,
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
    );
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
            wr.symbol(0, part_cdf(st, bl, x8, y8)); // PARTITION_NONE
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
            wr.symbol(3, part_cdf(st, bl, x8, y8)); // PARTITION_SPLIT
            let hh = sz8 / 2;
            let corners = [(x8, y8), (x8 + hh, y8), (x8, y8 + hh), (x8 + hh, y8 + hh)];
            for (i, (cx, cy)) in corners.into_iter().enumerate() {
                encode_plan_mono(wr, luma, st, bl + 1, cx, cy, hh, &kids[i]);
            }
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
        );
        encode_plan_mono(wr, luma, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        let ctx = get_partition_ctx(&st.a_part, &st.l_part, 4, x8, y8);
        wr.symbol(0, &C::PART_8[ctx]); // PARTITION_NONE
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
        );
        code_leaf_mono(wr, luma, st, px, py, 8, ym, yd, palette.as_ref(), intrabc);
        st.a_part[x8] = 0x1e;
        st.l_part[y8] = 0x1e;
        return;
    }

    let hh = sz8 / 2;
    let have_h = (x8 + hh) * 8 < st.w;
    let have_v = (y8 + hh) * 8 < st.h;
    let cdf = part_cdf(st, bl, x8, y8);
    if have_h && have_v {
        wr.symbol(3, cdf); // PARTITION_SPLIT
    } else if have_h {
        wr.bool(true, gather_split_prob(cdf.try_into().unwrap(), true));
    } else if have_v {
        wr.bool(true, gather_split_prob(cdf.try_into().unwrap(), false));
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
pub fn encode_tile_lossless_mono(
    w: usize,
    h: usize,
    visible_w: usize,
    visible_h: usize,
    bit_depth: u8,
    luma: &[i16],
    speed: Speed,
) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new();
    let mut st = LlState {
        w,
        h,
        visible_w,
        visible_h,
        base: 1i32 << (bit_depth - 1),
        bit_depth,
        angle_delta_rdo: speed.try_angle_deltas(),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 8],
        l_mode: vec![0; h / 8],
        a_skip: vec![0; w / 4],
        l_skip: vec![0; h / 4],
        ibc_mv: vec![None; (w / 4) * (h / 4)],
        a_palette: vec![Vec::new(); w / 8],
        l_palette: vec![Vec::new(); h / 8],
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

        let (_, _, delta, _, _) = best_leaf_mono(&luma, w, 16, 16, 2, 128, 8, w, h, false);
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
            let (_, mode, _, palette, _) = best_leaf_mono(&luma, w, 0, 0, 16, 128, 8, w, h, true);
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

        let (_, mode, _, palette, intrabc) = best_leaf_mono(&luma, w, 0, 0, 16, 128, 8, w, h, true);
        assert_eq!(mode, 0);
        assert!(palette.is_none());
        assert!(!intrabc);
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
            let exact: Vec<u8> = (0..w * h)
                .map(|i| 7 + (((i * 37 + (i / w) * 11 + 3) % n) * 31) as u8)
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
        let (_, plan) = plan_full_mono(&luma_i16, w, 0, 64, 8, 128, 8, w, h, true);
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
        let (_, mode, delta, palette, intrabc) =
            best_leaf_mono(&luma, w, 8, 0, 2, 128, 8, w, h, true);
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
        let mut first = vec![0i32; w * 64];
        let mut state = 0x1055_1bc0u32;
        for sample in &mut first {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *sample = (state >> 24) as i32;
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
        let mut plane = vec![0i32; w * h];
        for y in 0..h {
            for x in 0..128 {
                plane[y * w + x] = ((x * 29 + y * 47 + (x ^ y) * 3) & 255) as i32;
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
        let a = encode_tile_lossless_mono(w, h, w, h, 8, &y, Speed::Slow);
        let b = encode_tile_lossless_mono(w, h, w, h, 8, &y, Speed::Slow);
        assert!(!a.is_empty(), "mono lossless output must be non-empty");
        assert_eq!(a, b, "mono lossless must be deterministic");
        // A mono leaf omits the uv_mode symbol + 2 chroma planes, so for the same
        // luma it must be strictly smaller than the 4:4:4 tile carrying that luma
        // in all three planes.
        let c444 = encode_tile_lossless(w, h, w, h, 8, [&y, &y, &y], Speed::Slow);
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
            let p = encode_tile_lossless_mono(w, h, w, h, bd, &y, Speed::Slow);
            assert!(!p.is_empty());
            assert_eq!(
                p,
                encode_tile_lossless_mono(w, h, w, h, bd, &y, Speed::Slow)
            );
        }
    }
}
