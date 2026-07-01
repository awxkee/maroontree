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

use crate::av1_coefs::encode_coefs;
use crate::av1_tables::SKIP_CTX;
use crate::cdf_tables as C;
use crate::cost::coef_rate_bits;
use crate::intrapred::INTRA_MODE_CTX;
use crate::msac_enc::Writer;
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

/// Non-directional luma/chroma modes evaluated by the lossless mode search, in
/// CDF symbol order. DC(0), V(1), H(2), SMOOTH(9), SMOOTH_V(10), SMOOTH_H(11),
/// PAETH(12). Directional modes are omitted (their small 4x4 reach and the
/// edge-array machinery add little for lossless photographic content).
static LL_MODES: [usize; 7] = [0, 1, 2, 9, 10, 11, 12];

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
    base: i32,
    a: &mut [u8],
    l: &mut [u8],
) {
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];
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

            predict_4x4(mode, plane, stride, ox, oy, &mut pred, base);
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
    plane: &[i16],
    stride: usize,
    bx: usize,
    by: usize,
    n_tx: usize,
    base: i32,
) -> f64 {
    let mut bits = 0f64;
    let mut pred = [0i32; 16];
    let mut resid = [0i32; 16];

    for ty in 0..n_tx {
        for tx in 0..n_tx {
            let (ox, oy) = (bx + tx * 4, by + ty * 4);
            predict_4x4(mode, plane, stride, ox, oy, &mut pred, base);
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

/// Best luma + best uv mode for a leaf, with total residual+overhead bits.
fn best_leaf(
    planes: [&[i16]; 3],
    stride: usize,
    px: usize,
    py: usize,
    n_tx: usize,
    base: i32,
) -> (f64, usize, usize) {
    let mut y_mode = 0usize;
    let mut yb = f64::INFINITY;
    for &m in LL_MODES.iter() {
        let b = plane_leaf_bits(m, planes[0], stride, px, py, n_tx, base);
        if b < yb {
            yb = b;
            y_mode = m;
        }
    }
    let mut uv_mode = 0usize;
    let mut ub = f64::INFINITY;
    for &m in LL_MODES.iter() {
        let b = plane_leaf_bits(m, planes[1], stride, px, py, n_tx, base)
            + plane_leaf_bits(m, planes[2], stride, px, py, n_tx, base);
        if b < ub {
            ub = b;
            uv_mode = m;
        }
    }
    let ang = |m: usize| if (1..=8).contains(&m) { 1.5 } else { 0.0 };
    let ovh = 7.0 + ang(y_mode) + ang(uv_mode); // skip + y_mode + uv_mode (+ angle_deltas)
    (yb + ub + ovh, y_mode, uv_mode)
}

/// Adaptive partition plan for a fully-in-frame square block.
enum Plan {
    Leaf(usize, usize), // (y_mode, uv_mode)
    Split(Box<[Plan; 4]>),
}

const PART_NONE_BITS: f64 = 1.0;
const PART_SPLIT_BITS: f64 = 1.5;

/// Decide none-vs-split by estimated bits; returns the plan and its cost. Min
/// leaf is 8x8 (sz8 == 1).
fn plan_full(
    planes: [&[i16]; 3],
    stride: usize,
    px: usize,
    py: usize,
    sz8: usize,
    base: i32,
) -> (f64, Plan) {
    let (bits_leaf, ym, uv) = best_leaf(planes, stride, px, py, sz8 * 2, base);
    let none = PART_NONE_BITS + bits_leaf;
    if sz8 == 1 {
        return (none, Plan::Leaf(ym, uv));
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
        let (b, p) = plan_full(planes, stride, cx, cy, hh, base);
        split += b;
        kids[i] = Some(p);
    }
    if none <= split {
        (none, Plan::Leaf(ym, uv))
    } else {
        (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))))
    }
}

/// Mutable frame-spanning state shared across the lossless partition recursion.
struct LlState {
    w: usize,
    h: usize,
    base: i32,
    a_coef: [Vec<u8>; 3],
    l_coef: [Vec<u8>; 3],
    a_part: Vec<u8>,
    l_part: Vec<u8>,
    a_mode: Vec<u8>, // luma y_mode per 8px unit (for kf_y context)
    l_mode: Vec<u8>,
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
    uv_mode: usize,
) {
    let n_tx = size / 4;
    wr.symbol(0, &C::BLK_SKIP); // skip = 0
    let (x8, y8) = (px / 8, py / 8);
    let kfy = icdf13(
        &KF_Y_MODE_CDF[INTRA_MODE_CTX[st.a_mode[x8] as usize]]
            [INTRA_MODE_CTX[st.l_mode[y8] as usize]],
    );
    wr.symbol(y_mode as u32, &kfy);
    if (1..=8).contains(&y_mode) {
        wr.symbol(3, &icdf7(&ANGLE_DELTA_CDF[y_mode - 1])); // angle_delta = 0
    }
    let uvc = icdf13(&UV_MODE_NOCFL_CDF[y_mode]);
    wr.symbol(uv_mode as u32, &uvc);
    if (1..=8).contains(&uv_mode) {
        wr.symbol(3, &icdf7(&ANGLE_DELTA_CDF[uv_mode - 1]));
    }
    let u8sz = size / 8;
    for u in x8..x8 + u8sz {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y8..y8 + u8sz {
        st.l_mode[u] = y_mode as u8;
    }
    let modes = [y_mode, uv_mode, uv_mode];
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
            st.base,
            &mut st.a_coef[plane],
            &mut st.l_coef[plane],
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
        Plan::Leaf(ym, uv) => {
            wr.symbol(0, part_cdf(st, bl, x8, y8)); // PARTITION_NONE
            code_leaf(wr, planes, st, px, py, sz8 * 8, *ym, *uv);
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
        let (_bits, plan) = plan_full(planes, st.w, px, py, sz8, st.base);
        encode_plan(wr, planes, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        // 8x8 leaf (in-frame for multiple-of-8 dims): mode-search and code
        let ctx = get_partition_ctx(&st.a_part, &st.l_part, 4, x8, y8);
        wr.symbol(0, &C::PART_8[ctx]); // PARTITION_NONE
        let (_b, ym, uv) = best_leaf(planes, st.w, px, py, 2, st.base);
        code_leaf(wr, planes, st, px, py, 8, ym, uv);
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
pub fn encode_tile_lossless(w: usize, h: usize, bit_depth: u8, planes: [&[i16]; 3]) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new();
    let mut st = LlState {
        w,
        h,
        base: 1i32 << (bit_depth - 1),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 8],
        l_mode: vec![0; h / 8],
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
fn best_leaf_mono(
    luma: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    n_tx: usize,
    base: i32,
) -> (f64, usize) {
    let mut y_mode = 0usize;
    let mut yb = f64::INFINITY;
    for &m in LL_MODES.iter() {
        let b = plane_leaf_bits(m, luma, stride, px, py, n_tx, base);
        if b < yb {
            yb = b;
            y_mode = m;
        }
    }
    let ang = |m: usize| if (1..=8).contains(&m) { 1.5 } else { 0.0 };
    // skip + y_mode (+ angle_delta); no uv_mode symbol in a mono frame.
    let ovh = 4.0 + ang(y_mode);
    (yb + ovh, y_mode)
}

/// Mono partition plan for a fully-in-frame square block (luma only).
fn plan_full_mono(
    luma: &[i16],
    stride: usize,
    px: usize,
    py: usize,
    sz8: usize,
    base: i32,
) -> (f64, Plan) {
    let (bits_leaf, ym) = best_leaf_mono(luma, stride, px, py, sz8 * 2, base);
    let none = PART_NONE_BITS + bits_leaf;
    if sz8 == 1 {
        return (none, Plan::Leaf(ym, 0));
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
        let (b, p) = plan_full_mono(luma, stride, cx, cy, hh, base);
        split += b;
        kids[i] = Some(p);
    }
    if none <= split {
        (none, Plan::Leaf(ym, 0))
    } else {
        (split, Plan::Split(Box::new(kids.map(|k| k.unwrap()))))
    }
}

/// Code a mono lossless leaf: block skip + kf_y mode (+ angle_delta) then the
/// luma `(size/4)²` `TX_4X4` WHT. No uv_mode symbol, no chroma blocks.
fn code_leaf_mono(
    wr: &mut Writer,
    luma: &[i16],
    st: &mut LlState,
    px: usize,
    py: usize,
    size: usize,
    y_mode: usize,
) {
    let n_tx = size / 4;
    wr.symbol(0, &C::BLK_SKIP); // skip = 0
    let (x8, y8) = (px / 8, py / 8);
    let kfy = icdf13(
        &KF_Y_MODE_CDF[INTRA_MODE_CTX[st.a_mode[x8] as usize]]
            [INTRA_MODE_CTX[st.l_mode[y8] as usize]],
    );
    wr.symbol(y_mode as u32, &kfy);
    if (1..=8).contains(&y_mode) {
        wr.symbol(3, &icdf7(&ANGLE_DELTA_CDF[y_mode - 1])); // angle_delta = 0
    }
    // (mono: HasChroma == false ⇒ no uv_mode symbol, no chroma residual)
    let u8sz = size / 8;
    for u in x8..x8 + u8sz {
        st.a_mode[u] = y_mode as u8;
    }
    for u in y8..y8 + u8sz {
        st.l_mode[u] = y_mode as u8;
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
        st.base,
        &mut st.a_coef[0],
        &mut st.l_coef[0],
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
        Plan::Leaf(ym, _uv) => {
            wr.symbol(0, part_cdf(st, bl, x8, y8)); // PARTITION_NONE
            code_leaf_mono(wr, luma, st, px, py, sz8 * 8, *ym);
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
        let (_bits, plan) = plan_full_mono(luma, st.w, px, py, sz8, st.base);
        encode_plan_mono(wr, luma, st, bl, x8, y8, sz8, &plan);
        return;
    }
    if sz8 == 1 {
        let ctx = get_partition_ctx(&st.a_part, &st.l_part, 4, x8, y8);
        wr.symbol(0, &C::PART_8[ctx]); // PARTITION_NONE
        let (_b, ym) = best_leaf_mono(luma, st.w, px, py, 2, st.base);
        code_leaf_mono(wr, luma, st, px, py, 8, ym);
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
pub fn encode_tile_lossless_mono(w: usize, h: usize, bit_depth: u8, luma: &[i16]) -> Vec<u8> {
    assert!(
        w.is_multiple_of(8) && h.is_multiple_of(8),
        "width/height must be multiples of 8"
    );
    let mut wr = Writer::new();
    let mut st = LlState {
        w,
        h,
        base: 1i32 << (bit_depth - 1),
        a_coef: [vec![0x40; w / 4], vec![0x40; w / 4], vec![0x40; w / 4]],
        l_coef: [vec![0x40; h / 4], vec![0x40; h / 4], vec![0x40; h / 4]],
        a_part: vec![0; w / 8],
        l_part: vec![0; h / 8],
        a_mode: vec![0; w / 8],
        l_mode: vec![0; h / 8],
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

    /// Arbitrary-size (non-multiple-of-64) lossless regression guard. 72×64 has
    /// one full superblock plus an 8px-wide edge column split to 8×8 leaves via
    /// the frame-edge partition logic. Bytes verified bit-exact in dav1d 1.4.1
    /// and ffmpeg.
    #[test]
    fn lossless_72x64_edge_stable() {
        let (w, h) = (72usize, 64usize);
        let (mut g, mut b, mut r) = (vec![0i16; w * h], vec![0i16; w * h], vec![0i16; w * h]);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                g[i] = ((x + y) % 64) as i16;
                b[i] = (x % 64) as i16;
                r[i] = (y % 64) as i16;
            }
        }
        let p = encode_tile_lossless(w, h, 8, [&g, &b, &r]);
        assert_eq!(p.len(), 1671);
        assert_eq!(p.iter().map(|&x| x as u64).sum::<u64>(), 224364);
        assert_eq!(&p[..6], &[221, 107, 90, 215, 91, 24]);
    }

    /// Multi-superblock lossless regression guard (128x64, two superblocks).
    /// Bytes verified to decode exactly in dav1d 1.4.1 and ffmpeg.
    #[test]
    fn lossless_128x64_stable() {
        let (w, h) = (128usize, 64usize);
        let (mut g, mut b, mut r) = (vec![0i16; w * h], vec![0i16; w * h], vec![0i16; w * h]);
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                g[i] = ((x + y) % 64) as i16;
                b[i] = (x % 64) as i16;
                r[i] = (y % 64) as i16;
            }
        }
        let p = encode_tile_lossless(w, h, 8, [&g, &b, &r]);
        assert_eq!(p.len(), 2904);
        assert_eq!(p.iter().map(|&x| x as u64).sum::<u64>(), 384244);
        assert_eq!(&p[..6], &[0xdd, 0x6b, 0x5a, 0xd7, 0x5b, 0x18]);
    }

    /// Mono lossless determinism + non-emptiness. The mono leaf is an exact
    /// subset of the verified 4:4:4 leaf (no uv_mode symbol, no chroma blocks),
    /// so its luma coding is byte-for-byte the 4:4:4 luma coding. Golden bytes
    /// for these assertions should be captured once the stream is confirmed to
    /// round-trip in dav1d/avifdec (a mono AVIF wrapper with `mono_chrome = 1`).
    #[test]
    fn mono_lossless_deterministic_nonempty() {
        let (w, h) = (72usize, 64usize); // includes a frame-edge split column
        let mut y = vec![0i16; w * h];
        for j in 0..h {
            for i in 0..w {
                y[j * w + i] = (((i * 5 + j * 3) % 47) as i16) - 8;
            }
        }
        let a = encode_tile_lossless_mono(w, h, 8, &y);
        let b = encode_tile_lossless_mono(w, h, 8, &y);
        assert!(!a.is_empty(), "mono lossless output must be non-empty");
        assert_eq!(a, b, "mono lossless must be deterministic");
        // A mono leaf omits the uv_mode symbol + 2 chroma planes, so for the same
        // luma it must be strictly smaller than the 4:4:4 tile carrying that luma
        // in all three planes.
        let c444 = encode_tile_lossless(w, h, 8, [&y, &y, &y]);
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
            let p = encode_tile_lossless_mono(w, h, bd, &y);
            assert!(!p.is_empty());
            assert_eq!(p, encode_tile_lossless_mono(w, h, bd, &y));
        }
    }
}
