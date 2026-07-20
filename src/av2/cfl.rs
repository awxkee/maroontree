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

use crate::av2::helpers::{
    cfl_sse_i32, coeff_rate_f32, dc_pred, dc_pred_cfl_subsampled, pixel_sse_rounded,
};
use crate::av2::mhccp;
use crate::av2::proj::Basis;
use crate::av2::{itx422, tables};
use crate::util::{FastRound, dirty_log2f};

pub(crate) const CFL_ADD_BITS_ALPHA: i32 = 5;
pub(crate) const CFL_ALPHABET_SIZE: u8 = 8; // magnitude indices 0..=7 -> |alpha| 1..=8

/// Branchless round-half-away-from-zero by `2^n`, bit-identical to
/// `if v < 0 { -round_pow2(-v, n) } else { round_pow2(v, n) }`.
#[inline(always)]
fn round_pow2_signed(v: i32, n: u32) -> i32 {
    let s = v >> 31;
    let av = (v ^ s) - s;
    let q = (av + (1i32 << (n - 1))) >> n;
    (q ^ s) - s
}

/// Minimum variance (in Q0 luma², i.e. plain 8-bit luma units squared) that the
/// MHCCP derivation reference region must have for the cross-component filter to
/// be numerically well-conditioned.
const MHCCP_MIN_REF_VAR: f64 = 64.0;

/// Population variance of the luma samples in the MHCCP derivation region — the
/// L-shaped `above`/`left` border of `l`, matching the sample set
/// [`mhccp::derive_params`] accumulates (interior `1..ref-1`, excluding the
/// block body `i >= left && j >= above`).
fn mhccp_ref_luma_variance(l: &mhccp::MhccpRefBuf, above: usize, left: usize) -> f64 {
    let stride = mhccp::MHCCP_REF_STRIDE;
    let mut sum = 0.0f64;
    let mut sum2 = 0.0f64;
    let mut n = 0.0f64;
    for j in 1..l.ref_height.saturating_sub(1) {
        for i in 1..l.ref_width.saturating_sub(1) {
            if i >= left && j >= above {
                continue;
            }
            // Q3 luma -> Q0, matching the `>> 3` the derivation applies.
            let v = (l.data[j * stride + i] >> 3) as f64;
            sum += v;
            sum2 += v * v;
            n += 1.0;
        }
    }
    if n < 1.0 {
        return f64::INFINITY;
    }
    (sum2 - sum * sum / n) / n
}

/// avm `get_scaled_luma_q0`: round(alpha_q3 * ac_q3, 6 + CFL_ADD_BITS_ALPHA).
#[inline(always)]
pub(crate) fn scaled_luma_q0(alpha_q3: i32, ac_q3: i32) -> i32 {
    round_pow2_signed(alpha_q3 * ac_q3, 6 + CFL_ADD_BITS_ALPHA as u32)
}

#[inline(always)]
fn scaled_luma_q0_i32(alpha_q3: i32, ac_q3: i32) -> i32 {
    let p = alpha_q3 * ac_q3;
    let s = p >> 31;
    let av = (p ^ s) - s;
    let q = (av + (1 << 10)) >> 11; // 6 + CFL_ADD_BITS_ALPHA = 11
    (q ^ s) - s
}

/// Resolve a (sign, magnitude-index) pair to the predictor `alpha_q3`.
/// sign: 0 = zero, 1 = negative, 2 = positive. Returns avm's `cfl_idx_to_alpha * 32`.
#[inline]
pub(crate) fn idx_to_alpha_q3(sign: u8, mag: u8) -> i32 {
    let a = match sign {
        1 => -(mag as i32 + 1),
        2 => mag as i32 + 1,
        _ => 0,
    };
    a * (1 << CFL_ADD_BITS_ALPHA)
}

/// Subsample a reconstructed luma block (stride `lstride`) to chroma resolution
/// `cw x ch`, scaled to Q3. `ssx`/`ssy` select 4:2:0 (both), 4:2:2 (ssx only) or
/// 4:4:4 (neither). Box filter (cfl_ds_filter_index = 0).
pub(crate) fn subsample_luma_q3(
    luma: &[i32],
    lstride: usize,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
) -> Vec<i32> {
    let mut out = vec![0i32; cw * ch];
    match (ssx, ssy) {
        (true, true) => {
            for (y, orow) in out.chunks_exact_mut(cw).enumerate() {
                let top = &luma[y * 2 * lstride..y * 2 * lstride + 2 * cw];
                let bot = &luma[(y * 2 + 1) * lstride..(y * 2 + 1) * lstride + 2 * cw];
                for (o, (t, b)) in orow.iter_mut().zip(
                    top.as_chunks::<2>()
                        .0
                        .iter()
                        .zip(bot.as_chunks::<2>().0.iter()),
                ) {
                    *o = (t[0] + t[1] + b[0] + b[1]) << 1;
                }
            }
        }
        (true, false) => {
            for (y, orow) in out.chunks_exact_mut(cw).enumerate() {
                let row = &luma[y * lstride..y * lstride + 2 * cw];
                for (o, p) in orow.iter_mut().zip(row.as_chunks::<2>().0.iter()) {
                    *o = (p[0] + p[1]) << 2;
                }
            }
        }
        (false, false) => {
            for (y, orow) in out.chunks_exact_mut(cw).enumerate() {
                let row = &luma[y * lstride..y * lstride + cw];
                for (o, &v) in orow.iter_mut().zip(row) {
                    *o = v << 3;
                }
            }
        }
        _ => unreachable!("ssy without ssx is not a supported chroma layout"),
    }
    out
}

/// Luma-plane location, chroma dimensions and causal availability for a Q3
/// subsampled reference ring.
#[derive(Clone, Copy)]
pub(crate) struct LumaRingSpec {
    pub(crate) stride: usize,
    pub(crate) luma_y: usize,
    pub(crate) luma_x: usize,
    pub(crate) chroma_width: usize,
    pub(crate) chroma_height: usize,
    pub(crate) subsample_x: bool,
    pub(crate) subsample_y: bool,
    pub(crate) have_top: bool,
    pub(crate) have_left: bool,
}

#[allow(dead_code)]
pub(crate) fn subsample_luma_ring_q3(recy: &[f32], spec: &LumaRingSpec) -> Vec<i32> {
    let LumaRingSpec {
        stride: pw,
        luma_y: ly,
        luma_x: lx,
        chroma_width: cw,
        chroma_height: ch,
        subsample_x: ssx,
        subsample_y: ssy,
        have_top,
        have_left,
    } = *spec;
    let sx = ssx as usize;
    let sy = ssy as usize;
    let stride = cw + 1;
    let mut ring = vec![0i32; (cw + 1) * (ch + 1)];
    let px = |y: usize, x: usize| -> i32 { recy[y * pw + x].fast_round() as i32 };
    // For each chroma coord (cx in -1..cw, cy in -1..ch), gather the luma
    // support and box-filter to Q3, replicating at plane edges.
    for cy in -1i32..ch as i32 {
        for cx in -1i32..cw as i32 {
            // luma top-left for this chroma sample
            let lyy = ly as i32 + (cy << sy);
            let lxx = lx as i32 + (cx << sx);
            let get = |dy: i32, dx: i32| -> i32 {
                let yy = (lyy + dy).max(0) as usize;
                let xx = (lxx + dx).max(0) as usize;
                px(yy, xx)
            };
            let q3 = match (ssx, ssy) {
                (true, true) => (get(0, 0) + get(0, 1) + get(1, 0) + get(1, 1)) << 1,
                (true, false) => (get(0, 0) + get(0, 1)) << 2,
                (false, false) => get(0, 0) << 3,
                _ => unreachable!(),
            };
            let idx = ((cy + 1) as usize) * stride + (cx + 1) as usize;
            ring[idx] = q3;
        }
    }
    // Edge replication when neighbors are unavailable.
    if !have_top {
        for x in 0..=cw {
            ring[x] = ring[stride + x];
        }
    }
    if !have_left {
        for y in 0..=ch {
            ring[y * stride] = ring[y * stride + 1];
        }
    }
    ring
}

/// the average. `w*h` must be a power of two (always true for valid block sizes).
pub(crate) fn compute_ac(recon_q3: &[i32], w: usize, h: usize) -> (Vec<i32>, i32) {
    let npel = w * h;
    let log2 = npel.trailing_zeros();
    let sum: i32 = recon_q3.iter().copied().sum::<i32>() + ((npel as i32) >> 1);
    let avg = sum >> log2;
    (recon_q3.iter().map(|&v| v - avg).collect(), avg)
}

/// CfL prediction: `chroma_dc + scaled_luma(alpha_q3, ac)`, clipped to bit depth.
pub(crate) fn cfl_predict(dc: i32, ac: &[i32], alpha_q3: i32, bitdepth: i32) -> Vec<i32> {
    let maxv = (1i32 << bitdepth) - 1;
    ac.iter()
        .map(|&a| (dc + scaled_luma_q0_i32(alpha_q3, a)).clamp(0, maxv))
        .collect()
}

/// Pick the per-plane alpha (sign, magnitude index) minimizing SSE of the CfL
/// prediction vs. the chroma source. Returns `(sign, mag, sse)`; `sign == 0` (alpha
/// zero, i.e. flat DC) is the baseline. Encoder-side only — any choice is valid as
/// long as it is signaled; the decoder reproduces the prediction deterministically.
pub(crate) fn pick_alpha_plane(src: &[i32], dc: i32, ac: &[i32], bitdepth: i32) -> (u8, u8, f32) {
    let maxv = (1i32 << bitdepth) - 1;
    let sse_of = |alpha_q3: i32| cfl_sse_i32(src, ac, alpha_q3, dc, maxv);
    // alpha == 0 baseline (prediction is flat clipped DC).
    let base_sse = sse_of(0);
    let mut best = (0u8, 0u8, base_sse);
    for sign in [1u8, 2u8] {
        for mag in 0u8..CFL_ALPHABET_SIZE {
            let sse = sse_of(idx_to_alpha_q3(sign, mag));
            if sse < best.2 {
                best = (sign, mag, sse);
            }
        }
    }
    best
}

// is_cfl bool, 3 neighbor contexts: cfl_cdf default = AVM_CDF2{20441,11610,4643}.
pub(crate) static CFL_IS_CDF: [u16; 3] = [12327, 21158, 28125];
// cfl_index / cfl_type (2-symbol; we always emit 0 = CFL_EXPLICIT): AVM_CDF2(12507).
pub(crate) const CFL_INDEX_CDF: u16 = 20261;
// cfl_mhccp_switch bool: is this block MHCCP (CFL_MULTI_PARAM)? AVM_CDF2(15499).
// Stored as ICDF (32768 - 15499). Symbol 1 = use MHCCP.
pub(crate) const CFL_MHCCP_SWITCH_CDF: u16 = 17269;
// filter_dir (mh_dir) 3-symbol CDF, one row per MHCCP context group
// (size_group_lookup). AVM_CDF3 cumulative probs converted to ICDF (32768 - p).
pub(crate) static FILTER_DIR_CDF: [[u16; 3]; 4] = [
    [21845, 10923, 0],
    [23973, 17663, 0],
    [22335, 16794, 0],
    [15683, 11079, 0],
];
// AVM PARA tuples for the filter_dir adaptation rate, per context group.
// AVM `filter_dir_cdf` adaptation PARA per context group, already transformed
// by AVM_PARA3(a,b,c) = (a+2, b+3, c+4) from the raw args
// (0,0,0),(0,-1,-1),(-1,-1,-2),(-1,-1,-2). These are the exact stored rate
// offsets the maroontree CDF adapter consumes.
pub(crate) static FILTER_DIR_PARA: [(u8, u8, u8); 4] = [(2, 3, 4), (2, 2, 3), (1, 2, 2), (1, 2, 2)];
// AVM PARA tuple for cfl_mhccp_switch.
pub(crate) const CFL_MHCCP_SWITCH_PARA: (i8, i8, i8) = (-1, -1, 0);
// cfl_sign joint-sign, 8 symbols: AVM_CDF7(2421,4332,11256,12766,21386,28725,32087)
// expanded to stored icdf (32768 - arg) with the trailing implicit-zero entry.
pub(crate) static CFL_SIGN_ICDF: [u16; 8] = [30347, 28436, 21512, 20002, 11382, 4043, 681, 0];
// cfl_alpha magnitude, 6 contexts x 8 symbols.
pub(crate) static CFL_ALPHA_ICDF: [[u16; 8]; 6] = [
    [11089, 7463, 2122, 1256, 231, 122, 72, 0],
    [24506, 16466, 8686, 3346, 1370, 482, 243, 0],
    [15533, 6602, 2390, 1463, 395, 219, 100, 0],
    [15150, 7036, 4903, 2430, 1643, 1246, 530, 0],
    [15226, 9702, 4861, 4040, 2066, 1603, 1333, 0],
    [15093, 7966, 2300, 1985, 927, 504, 346, 0],
];

/// avm `CFL_SIGN_U(js)` / `CFL_SIGN_V(js)` (enums.h): split a joint sign back into the
/// per-plane signs (0 = zero, 1 = neg, 2 = pos).
#[inline]
pub(crate) fn cfl_sign_u(js: u8) -> u8 {
    (((js as u32 + 1) * 11) >> 5) as u8
}

#[inline]
pub(crate) fn cfl_sign_v(js: u8) -> u8 {
    (js + 1) - 3 * cfl_sign_u(js)
}

/// The resolved CfL choice for a chroma block: joint sign, per-plane magnitude indices,
/// alpha-cdf contexts, and the two prediction blocks (clipped, ready as reconstruction
/// bases). `sign_*`/`mag_*` are encoder bookkeeping; the bitstream carries `js` + mags.
#[derive(Clone)]
pub(crate) struct CflChoice {
    pub(crate) js: u8,
    pub(crate) sign_u: u8,
    pub(crate) sign_v: u8,
    pub(crate) mag_u: u8,
    pub(crate) mag_v: u8,
    pub(crate) ctx_u: usize,
    pub(crate) ctx_v: usize,
    pub(crate) pred_u: Vec<i32>,
    pub(crate) pred_v: Vec<i32>,
    /// When `Some`, this block uses MHCCP (avm `CFL_MULTI_PARAM`) instead of the
    /// classic alpha CfL. Carries the chosen filter direction (`mh_dir` 0/1/2)
    /// and the size-group context for `filter_dir` signaling. `pred_u`/`pred_v`
    /// then hold the MHCCP predictor and the alpha fields are unused.
    pub(crate) mhccp: Option<MhccpDecision>,
}

#[derive(Clone, Copy)]
pub(crate) struct MhccpDecision {
    pub(crate) mh_dir: u8,
    pub(crate) size_group: u8,
}

pub(crate) fn cfl_candidate(
    src_u: &[i32],
    src_v: &[i32],
    ac: &[i32],
    dc_u: i32,
    dc_v: i32,
    bitdepth: i32,
) -> Option<CflChoice> {
    let (sign_u, mag_u, _) = pick_alpha_plane(src_u, dc_u, ac, bitdepth);
    let (sign_v, mag_v, _) = pick_alpha_plane(src_v, dc_v, ac, bitdepth);
    if sign_u == 0 && sign_v == 0 {
        return None;
    }
    let js = 3 * sign_u + sign_v - 1;
    let ctx_u = if sign_u != 0 { (js - 2) as usize } else { 0 };
    let ctx_v = if sign_v != 0 {
        (sign_v * 3 + sign_u - 3) as usize
    } else {
        0
    };
    let pred_u = cfl_predict(dc_u, ac, idx_to_alpha_q3(sign_u, mag_u), bitdepth);
    let pred_v = cfl_predict(dc_v, ac, idx_to_alpha_q3(sign_v, mag_v), bitdepth);
    Some(CflChoice {
        js,
        sign_u,
        sign_v,
        mag_u,
        mag_v,
        ctx_u,
        ctx_v,
        pred_u,
        pred_v,
        mhccp: None,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_avg_l(
    recy: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
    bd: i32,
) -> i32 {
    let have_top = sb_y > 0;
    let have_left = sb_x > 0;
    let ss_h = if cw > 32 { 2 } else { 1 };
    let ss_v = if ch > 32 { 2 } else { 1 };
    let l = |y: usize, x: usize| -> i32 { recy[y * pw + x].fast_round() as i32 };
    let mut sum: i32 = 0;
    let mut count: i32 = 0;
    if have_top {
        let mut kk = 0;
        while kk < cw {
            let s = match (ssx, ssy) {
                (false, false) => l(sb_y - 1, sb_x + kk) << 3,
                // 4:2:0, is_top_sb_boundary=1: single row sb_y-1 counted twice, <<1.
                (true, true) => {
                    let a = l(sb_y - 1, sb_x + 2 * kk);
                    let b = l(sb_y - 1, sb_x + 2 * kk + 1);
                    (a + b + a + b) << 1
                }
                // 4:2:2: horizontal luma pair of row sb_y-1, <<2.
                (true, false) => (l(sb_y - 1, sb_x + 2 * kk) + l(sb_y - 1, sb_x + 2 * kk + 1)) << 2,
                _ => unreachable!(),
            };
            sum += s;
            count += 1;
            kk += ss_h;
        }
    }
    if have_left {
        let mut jj = 0;
        while jj < ch {
            let s = match (ssx, ssy) {
                (false, false) => l(sb_y + jj, sb_x - 1) << 3,
                // 4:2:0: 2x2 luma box at cols sb_x-2,sb_x-1 over rows 2jj,2jj+1, <<1.
                (true, true) => {
                    let a = l(sb_y + 2 * jj, sb_x - 2);
                    let b = l(sb_y + 2 * jj, sb_x - 1);
                    let c = l(sb_y + 2 * jj + 1, sb_x - 2);
                    let d = l(sb_y + 2 * jj + 1, sb_x - 1);
                    (a + b + c + d) << 1
                }
                // 4:2:2: horizontal luma pair at cols sb_x-2,sb_x-1 of row sb_y+jj, <<2.
                (true, false) => (l(sb_y + jj, sb_x - 2) + l(sb_y + jj, sb_x - 1)) << 2,
                _ => unreachable!(),
            };
            sum += s;
            count += 1;
            jj += ss_v;
        }
    }
    if count > 0 {
        let val = (sum + count / 2) / count;
        let max_v = (1 << (bd + 3)) - 1;
        val.min(max_v)
    } else {
        8 << (bd - 1)
    }
}

/// Coded tile/frame extents used by MHCCP reference construction. The plane
/// allocations are usually padded to whole superblocks, but samples beyond these
/// extents are not causal neighbors from the decoder's point of view.
#[derive(Clone, Copy)]
pub(crate) struct MhccpBounds {
    pub(crate) luma_width: usize,
    pub(crate) luma_height: usize,
    pub(crate) chroma_width: usize,
    pub(crate) chroma_height: usize,
}

impl MhccpBounds {
    /// Build bounds from the tile-local display dimensions. AVM's block grid is
    /// 8-pixel aligned; chroma extents follow the format subsampling.
    #[inline]
    pub(crate) fn from_luma(width: usize, height: usize, ssx: bool, ssy: bool) -> Self {
        let luma_width = (width + 7) & !7;
        let luma_height = (height + 7) & !7;
        Self {
            luma_width,
            luma_height,
            chroma_width: luma_width >> (ssx as usize),
            chroma_height: luma_height >> (ssy as usize),
        }
    }
}

/// Reconstructed-neighbor context needed to evaluate MHCCP for a chroma block.
/// `recy` is the reconstructed luma plane (Q0, stride `pw`), `recu`/`recv` the
/// reconstructed chroma planes (Q0, stride `pcw`). `(ly, lx)` is the block's
/// luma top-left, `(cy, cx)` its chroma top-left. `ssx`/`ssy` are the format
/// subsampling. `have_top`/`have_left` indicate causal availability;
/// `is_top_sb_boundary` marks blocks at the top edge of a superblock (one line
/// of padding above). `size_group` is `size_group_lookup[bsize]`.
pub(crate) struct MhccpCtx<'a> {
    pub(crate) recy: &'a [f32],
    pub(crate) pw: usize,
    pub(crate) recu: &'a [f32],
    pub(crate) recv: &'a [f32],
    pub(crate) pcw: usize,
    pub(crate) bounds: MhccpBounds,
    pub(crate) ly: usize,
    pub(crate) lx: usize,
    pub(crate) cy: usize,
    pub(crate) cx: usize,
    pub(crate) ssx: bool,
    pub(crate) ssy: bool,
    pub(crate) have_top: bool,
    pub(crate) have_left: bool,
    pub(crate) is_top_sb_boundary: bool,
    pub(crate) size_group: u8,
    /// SB-relative (16x16) luma-mi coded mask: `coded[r*16 + c] != 0` when that
    /// luma mi has already been reconstructed. Mirrors the decoder's
    /// `xd->is_mi_coded`, which gates MHCCP's top-right / bottom-left reference
    /// extension (`has_top_right` / `has_bottom_left`). The interior split emits
    /// leaves in VERT-then-HORZ order, so a leaf's top-right neighbor is often
    /// coded LATER and its bottom-left neighbor EARLIER — the opposite of raster
    /// order. An all-zero (or empty) mask means "no interior neighbor coded yet".
    pub(crate) coded: &'a [u8],
}

/// Build the AVM-layout luma (Q3) reference buffer for a chroma block, matching
/// `mhccp_implicit_fetch_neighbor_luma` with the default box downsample filter
/// (`cfl_ds_filter_index = 0`). `width`/`height` are the chroma-block (tx) dims;
/// `above_lines`/`left_lines` the causal-line counts in chroma samples.
fn build_luma_ref(
    ctx: &MhccpCtx,
    width: usize,
    height: usize,
    above: usize,
    left: usize,
    tr_ext: usize,
    bl_ext: usize,
) -> mhccp::MhccpRefBuf {
    let ref_width = left + width + tr_ext;
    let ref_height = above + height + bl_ext;
    let mut buf = mhccp::MhccpRefBuf::new(
        ref_width,
        ref_height,
        above,
        left,
        width,
        height,
        ctx.is_top_sb_boundary,
    );
    let sx = ctx.ssx as usize;
    let sy = ctx.ssy as usize;
    // Luma coordinate of the reference buffer's top-left (buffer (0,0) maps to
    // chroma (cx-left, cy-above) -> luma ((cx-left)<<sx, (cy-above)<<sy)).
    let lx0 = ctx.lx as i32 - ((left as i32) << sx);
    let ly0 = ctx.ly as i32 - ((above as i32) << sy);
    let stride = ctx.pw as i32;
    let plane_height = ctx.recy.len() / ctx.pw;
    let bound_width = ctx.bounds.luma_width.min(ctx.pw).max(1) as i32;
    let bound_height = ctx.bounds.luma_height.min(plane_height).max(1) as i32;
    let lp = |y: i32, x: i32| -> i32 {
        let yy = y.clamp(0, bound_height - 1);
        let xx = x.clamp(0, bound_width - 1);
        ctx.recy[(yy * stride + xx) as usize].fast_round() as i32
    };
    // AVM `av2_mhccp_implicit_fetch_neighbor_luma_420`: near the superblock top
    // boundary, only one line above is available, so the padding rows in the above
    // region offset which luma lines are actually sampled. In luma domain the
    // reference has `above << sy` lines; the offset fires while the luma row
    // `h = j<<sy` is inside that above region.
    let above_luma = (above as i32) << sy;
    let line1_luma = ((mhccp::LINE_NUM as i32) + 1) << (sy as i32);
    for j in 0..ref_height {
        for i in 0..ref_width {
            // luma top-left sample for this chroma ref position
            let ly = ly0 + ((j as i32) << sy);
            let lx = lx0 + ((i as i32) << sx);
            let h = (j as i32) << sy; // luma row within the reference region
            let (cent_off, bot_off) =
                if above_luma == line1_luma && ctx.is_top_sb_boundary && h < above_luma {
                    (line1_luma - (h + 1), line1_luma - (h + 2))
                } else {
                    (0, 0)
                };
            let q3 = match (ctx.ssx, ctx.ssy) {
                (true, true) => {
                    (lp(ly + cent_off, lx)
                        + lp(ly + cent_off, lx + 1)
                        + lp(ly + 1 + bot_off, lx)
                        + lp(ly + 1 + bot_off, lx + 1))
                        << 1
                }
                (true, false) => (lp(ly + cent_off, lx) + lp(ly + cent_off, lx + 1)) << 2,
                (false, false) => lp(ly + cent_off, lx) << 3,
                _ => unreachable!(),
            };
            buf.set(i, j, q3);
        }
    }
    buf
}

/// Plane bounds, block geometry and causal extensions for one MHCCP chroma
/// reference buffer.
#[derive(Clone, Copy)]
struct ChromaRefSpec {
    stride: usize,
    bound_width: usize,
    bound_height: usize,
    y: usize,
    x: usize,
    width: usize,
    height: usize,
    above: usize,
    left: usize,
    top_right_extension: usize,
    bottom_left_extension: usize,
    is_top_sb: bool,
}

fn build_chroma_ref(rec: &[f32], spec: &ChromaRefSpec) -> mhccp::MhccpRefBuf {
    let ChromaRefSpec {
        stride: pcw,
        bound_width,
        bound_height,
        y: cy,
        x: cx,
        width,
        height,
        above,
        left,
        top_right_extension: tr_ext,
        bottom_left_extension: bl_ext,
        is_top_sb,
    } = *spec;
    let ref_width = left + width + tr_ext;
    let ref_height = above + height + bl_ext;
    let mut buf =
        mhccp::MhccpRefBuf::new(ref_width, ref_height, above, left, width, height, is_top_sb);
    let stride = pcw as i32;
    let plane_height = rec.len() / pcw;
    let bound_width = bound_width.min(pcw).max(1) as i32;
    let bound_height = bound_height.min(plane_height).max(1) as i32;
    let cp = |y: i32, x: i32| -> i32 {
        let yy = y.clamp(0, bound_height - 1);
        let xx = x.clamp(0, bound_width - 1);
        rec[(yy * stride + xx) as usize].round() as i32
    };
    // buffer (0,0) maps to chroma (cx-left, cy-above)
    let x0 = cx as i32 - left as i32;
    let y0 = cy as i32 - above as i32;
    let line_num1 = mhccp::LINE_NUM as i32 + 1;
    for j in 0..ref_height as i32 {
        for i in 0..ref_width as i32 {
            let mut ref_h = 0i32;
            let mut ref_w = 0i32;
            if above as i32 == line_num1 {
                if is_top_sb && j < above as i32 {
                    ref_h = line_num1 - (j + 1);
                } else if j == 0 {
                    ref_h = 1;
                }
            }
            if left as i32 == line_num1 && i == 0 {
                ref_w = 1;
            }
            let v = cp(y0 + j + ref_h, x0 + i + ref_w);
            buf.set(i as usize, j as usize, v);
        }
    }
    buf
}

/// Transform, quantizer and distortion-rate state shared by chroma mode trials.
#[derive(Clone, Copy)]
pub(crate) struct ChromaRdSpec<'a> {
    pub(crate) basis: &'a Basis,
    pub(crate) qstep: i32,
    pub(crate) lambda: f32,
    pub(crate) bit_depth: i32,
}

/// Source/reconstruction geometry and DC state for a classic CfL decision.
#[derive(Clone, Copy)]
pub(crate) struct CflDecisionInput<'a> {
    pub(crate) reconstructed_luma: &'a [f32],
    pub(crate) luma_stride: usize,
    pub(crate) luma_y: usize,
    pub(crate) luma_x: usize,
    pub(crate) source_u: &'a [f32],
    pub(crate) source_v: &'a [f32],
    pub(crate) dc_u: f32,
    pub(crate) dc_v: f32,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) subsample_x: bool,
    pub(crate) subsample_y: bool,
    pub(crate) luma_average_q3: i32,
}

/// Source and predictor pairs scored under one chroma RD configuration.
#[derive(Clone, Copy)]
pub(crate) struct PredictorScoreInput<'a> {
    pub(crate) source_u: &'a [f32],
    pub(crate) source_v: &'a [f32],
    pub(crate) predictor_u: &'a [i32],
    pub(crate) predictor_v: &'a [i32],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) mode_bits: f32,
}

/// `cw*ch` chroma source blocks (`src_u`/`src_v`), the DC intra predictions (`dc_u_f`/
/// `dc_v_f`) and the neighbor luma DC (`avg_l` from `cfl_avg_l`). Builds the AC from the
/// block luma (subsampled per `ssx`/`ssy`) minus `avg_l`, forms the best-alpha candidate,
/// then does real rate-distortion: transform/quantise/reconstruct both DC and CfL per plane
/// and pick CfL only when it lowers J = SSE + lambda*bits. `chroma` is the format's basis.
pub(crate) fn cfl_decide(input: &CflDecisionInput<'_>, rd: &ChromaRdSpec<'_>) -> Option<CflChoice> {
    let CflDecisionInput {
        reconstructed_luma: recy,
        luma_stride: pw,
        luma_y: sb_y,
        luma_x: sb_x,
        source_u: src_u,
        source_v: src_v,
        dc_u: dc_u_f,
        dc_v: dc_v_f,
        width: cw,
        height: ch,
        subsample_x: ssx,
        subsample_y: ssy,
        luma_average_q3: avg_l,
    } = *input;
    let ChromaRdSpec {
        basis: chroma,
        qstep,
        lambda,
        bit_depth: bd,
    } = *rd;
    let n = cw * ch;
    let lw = cw << (ssx as usize);
    let lh = ch << (ssy as usize);
    let mut luma = vec![0i32; lw * lh];
    for r in 0..lh {
        let b = (sb_y + r) * pw + sb_x;
        let luma_s = &mut luma[r * lw..];
        let recy_s = &recy[b..];
        for (l, &r) in luma_s[..lw].iter_mut().zip(recy_s.iter()) {
            *l = r.fast_round() as i32;
        }
    }
    let luma_q3 = subsample_luma_q3(&luma, lw, cw, ch, ssx, ssy);
    let ac: Vec<i32> = luma_q3.iter().map(|&v| v - avg_l).collect();
    let su: Vec<i32> = src_u.iter().map(|&s| s.fast_round() as i32).collect();
    let sv: Vec<i32> = src_v.iter().map(|&s| s.fast_round() as i32).collect();
    let dc_u = dc_u_f.fast_round() as i32;
    let dc_v = dc_v_f.fast_round() as i32;
    let cand = cfl_candidate(&su, &sv, &ac, dc_u, dc_v, bd)?;

    let scan = &tables::SCAN;
    let res_dc_u: Vec<f32> = src_u.iter().map(|&s| s - dc_u_f).collect();
    let res_dc_v: Vec<f32> = src_v.iter().map(|&s| s - dc_v_f).collect();
    let lev_dc_u = chroma.project(&res_dc_u, 0.0);
    let lev_dc_v = chroma.project(&res_dc_v, 0.0);
    let rec_dc_u = itx422::reconstruct_chroma(dc_u_f, &lev_dc_u, qstep, scan, cw, ch, bd);
    let rec_dc_v = itx422::reconstruct_chroma(dc_v_f, &lev_dc_v, qstep, scan, cw, ch, bd);
    let j_dc = (pixel_sse_rounded(src_u, &rec_dc_u) + pixel_sse_rounded(src_v, &rec_dc_v))
        + lambda * (coeff_rate_f32(&lev_dc_u) + coeff_rate_f32(&lev_dc_v));
    let res_cf_u: Vec<f32> = src_u[..n]
        .iter()
        .zip(cand.pred_u[..n].iter())
        .map(|(&s, &c)| s - c as f32)
        .collect();
    let res_cf_v: Vec<f32> = src_v[..n]
        .iter()
        .zip(cand.pred_v[..n].iter())
        .map(|(&s, &v)| s - v as f32)
        .collect();
    let lev_cf_u = chroma.project(&res_cf_u, 0.0);
    let lev_cf_v = chroma.project(&res_cf_v, 0.0);
    let rec_cf_u = itx422::reconstruct_chroma_cfl(&cand.pred_u, &lev_cf_u, qstep, scan, cw, ch, bd);
    let rec_cf_v = itx422::reconstruct_chroma_cfl(&cand.pred_v, &lev_cf_v, qstep, scan, cw, ch, bd);
    let alpha_bits = 2.0
        + 3.0
        + if cand.sign_u != 0 { 3.0 } else { 0.0 }
        + if cand.sign_v != 0 { 3.0 } else { 0.0 };
    let j_cfl = (pixel_sse_rounded(src_u, &rec_cf_u) + pixel_sse_rounded(src_v, &rec_cf_v)) as f32
        + lambda * ((coeff_rate_f32(&lev_cf_u) + coeff_rate_f32(&lev_cf_v)) as f32 + alpha_bits);
    if j_cfl < j_dc { Some(cand) } else { None }
}

/// Score a chroma predictor pair with the same J metric used by `cfl_decide`
/// (SSE + lambda*coeff_bits, excluding mode bits which the caller adds). Lets the
/// caller obtain the incumbent CfL/DC J to compare against MHCCP on equal terms.
pub(crate) fn score_predictor(input: &PredictorScoreInput<'_>, rd: &ChromaRdSpec<'_>) -> f32 {
    let PredictorScoreInput {
        source_u: src_u,
        source_v: src_v,
        predictor_u: pred_u,
        predictor_v: pred_v,
        width: cw,
        height: ch,
        mode_bits,
    } = *input;
    let ChromaRdSpec {
        basis: chroma,
        qstep,
        lambda,
        bit_depth: bd,
    } = *rd;
    let n = cw * ch;
    let scan = &tables::SCAN;
    let res_u: Vec<f32> = src_u[..n]
        .iter()
        .zip(pred_u.iter())
        .map(|(&s, &c)| s - c as f32)
        .collect();
    let res_v: Vec<f32> = src_v[..n]
        .iter()
        .zip(pred_v.iter())
        .map(|(&s, &c)| s - c as f32)
        .collect();
    let lev_u = chroma.project(&res_u, 0.0);
    let lev_v = chroma.project(&res_v, 0.0);
    let rec_u = itx422::reconstruct_chroma_cfl(pred_u, &lev_u, qstep, scan, cw, ch, bd);
    let rec_v = itx422::reconstruct_chroma_cfl(pred_v, &lev_v, qstep, scan, cw, ch, bd);
    (pixel_sse_rounded(src_u, &rec_u) + pixel_sse_rounded(src_v, &rec_v))
        + lambda * ((coeff_rate_f32(&lev_u) + coeff_rate_f32(&lev_v)) + mode_bits)
}

/// Map a block width/height in 4-sample units to the AVM `BLOCK_SIZE` enum
/// index (subset covering square + rectangular intra sizes used here). Returns
/// `None` for sizes with no enum entry.
pub(crate) fn bsize_from_wh4(bw4: usize, bh4: usize) -> Option<usize> {
    let w = bw4 * 4;
    let h = bh4 * 4;
    let idx = match (w, h) {
        (4, 4) => 0,
        (4, 8) => 1,
        (8, 4) => 2,
        (8, 8) => 3,
        (8, 16) => 4,
        (16, 8) => 5,
        (16, 16) => 6,
        (16, 32) => 7,
        (32, 16) => 8,
        (32, 32) => 9,
        (32, 64) => 10,
        (64, 32) => 11,
        (64, 64) => 12,
        // Extended (1:4 / 4:1) sizes — AVM enum order (BLOCK_4X16 = 19, ...).
        (4, 16) => 19,
        (16, 4) => 20,
        (8, 32) => 21,
        (32, 8) => 22,
        (16, 64) => 23,
        (64, 16) => 24,
        (4, 32) => 25,
        (32, 4) => 26,
        (8, 64) => 27,
        (64, 8) => 28,
        _ => return None,
    };
    Some(idx)
}

/// MHCCP `filter_dir` context (size group) for a block given its 4-sample dims.
pub(crate) fn mhccp_size_group_wh4(bw4: usize, bh4: usize) -> u8 {
    bsize_from_wh4(bw4, bh4).map(mhccp_size_group).unwrap_or(3)
}

/// avm `is_mhccp_allowed` (non-lossless) expressed from the luma block size in
/// 4-sample units and the chroma subsampling. Returns whether the decoder will
/// read the `cfl_mhccp_switch` on a CfL block of this size — the encoder must
/// emit the switch (=0 if not doing MHCCP) exactly when this is true.
pub(crate) fn is_mhccp_allowed(bw4: usize, bh4: usize, ssx: bool, ssy: bool) -> bool {
    let luma_w = bw4 * 4;
    let luma_h = bh4 * 4;
    // chroma plane block size
    let cw = luma_w >> (ssx as usize);
    let ch = luma_h >> (ssy as usize);
    // not 4x4 chroma
    if cw == 4 && ch == 4 {
        return false;
    }
    // chroma within max UV tx (32x32)
    if cw > 32 || ch > 32 {
        return false;
    }
    // luma partition <= 64x64 (CFL_BUF_LINE/2)
    luma_w <= 64 && luma_h <= 64
}

/// avm `size_group_lookup[bsize]` for the MHCCP `filter_dir` context. `bsize` is
/// the luma block-size enum index.
#[inline]
pub(crate) fn mhccp_size_group(bsize: usize) -> u8 {
    mhccp::SIZE_GROUP_LOOKUP.get(bsize).copied().unwrap_or(3)
}

/// Size group for a BLOCK_32X32 luma partition (enum index 9 -> group 3).
#[inline]
pub(crate) fn mhccp_size_group_32() -> u8 {
    mhccp_size_group(9)
}

/// MHCCP rate-distortion decision. Evaluates the three filter directions
/// (`mh_dir` 0=center / 1=top / 2=left), each solving the 3-parameter model from
/// the causal neighbor ring, reconstructing both chroma planes, and scoring
/// J = SSE + lambda*(coeff_bits + mode_bits). Returns a `CflChoice` carrying the
/// MHCCP predictor when the best direction beats `baseline_j` (the J of the
/// incumbent — DC or classic CfL). Mirrors the `CFL_MULTI_PARAM` arm of avm's
/// chroma intra RD loop.
/// Chroma source block, transform state and incumbent cost for one MHCCP trial.
#[derive(Clone, Copy)]
pub(crate) struct MhccpDecisionInput<'a> {
    pub(crate) source_u: &'a [f32],
    pub(crate) source_v: &'a [f32],
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) rd: ChromaRdSpec<'a>,
    pub(crate) scan: &'a [u16],
    pub(crate) baseline_j: f32,
}

pub(crate) fn mhccp_decide(ctx: &MhccpCtx, input: &MhccpDecisionInput<'_>) -> Option<CflChoice> {
    let MhccpDecisionInput {
        source_u: src_u,
        source_v: src_v,
        width: cw,
        height: ch,
        rd,
        scan,
        baseline_j,
    } = *input;
    let ChromaRdSpec {
        basis: chroma,
        qstep,
        lambda,
        bit_depth: bd,
    } = rd;
    let n = cw * ch;

    // avm above_lines/left_lines: (LINE_NUM+1) chroma lines when the neighbor is
    // available (interior blocks). At the top-of-superblock boundary avm uses the
    // same count but with a one-line padding offset handled inside the builders.
    let line_num1 = mhccp::LINE_NUM + 1;
    let above = if ctx.have_top { line_num1 } else { 0 };
    let left = if ctx.have_left { line_num1 } else { 0 };
    if above == 0 && left == 0 {
        return None; // no causal support -> MHCCP degenerates
    }
    // Native edge partitions may code a transform whose nominal rectangle extends
    // beyond the tile's coded plane. The current MHCCP model does not reproduce the
    // decoder's clipped partial-block syntax bit-exactly, so keep this optional mode
    // off for those leaves. The DC fallback below uses explicit edge replication.
    if ctx.cx.saturating_add(cw) > ctx.bounds.chroma_width
        || ctx.cy.saturating_add(ch) > ctx.bounds.chroma_height
    {
        return None;
    }

    let sx = ctx.ssx as usize;
    let sy = ctx.ssy as usize;
    // Coding-order availability of the top-right / bottom-left reference blocks,
    // mirroring the decoder's `has_top_right` / `has_bottom_left` general case
    // (checks `is_mi_coded` for the neighbor luma mi within the SB). Indices are
    // SB-relative luma mi; leaf luma mi size = chroma size upsampled.
    const SB_MI: usize = 16;
    let sb_r = ((ctx.ly & 63) >> 2) as i32; // SB-relative luma mi row of the leaf
    let sb_c = ((ctx.lx & 63) >> 2) as i32;
    let w_mi = ((cw << sx) >> 2) as i32; // leaf luma width  in mi
    let h_mi = ((ch << sy) >> 2) as i32; // leaf luma height in mi
    let coded_at = |r: i32, c: i32| -> bool {
        r >= 0
            && c >= 0
            && (r as usize) < SB_MI
            && (c as usize) < SB_MI
            && ctx
                .coded
                .get(r as usize * SB_MI + c as usize)
                .copied()
                .unwrap_or(0)
                != 0
    };
    // has_top_right: neighbor luma mi at (sb_r-1, sb_c+w_mi).
    let has_tr = ctx.have_top && {
        let tr_row = sb_r - 1;
        let tr_col = sb_c + w_mi;
        if tr_row < 0 {
            true // top-right lies in the (already coded) SB row above
        } else if tr_col >= SB_MI as i32 {
            false // right of this SB — not yet coded
        } else {
            coded_at(tr_row, tr_col)
        }
    };
    // has_bottom_left: neighbor luma mi at (sb_r+h_mi, sb_c-1).
    let has_bl = ctx.have_left && {
        let bl_row = sb_r + h_mi;
        let bl_col = sb_c - 1;
        if bl_col < 0 {
            // Bottom-left lies in the SB column to the left. Available only if the
            // block does not already reach the SB's bottom edge.
            ((sb_r + h_mi) as usize) < SB_MI
        } else if bl_row >= SB_MI as i32 {
            false // below this SB — not yet coded
        } else {
            coded_at(bl_row, bl_col)
        }
    };
    // AVM caps the reference region at 128 luma samples (64 chroma for 4:2:0)
    // before subsampling; also clamp to the coded tile extent.
    let tr_ext = if has_tr && cw > 4 {
        let cap_c = 128usize >> sx;
        let cap_room = cap_c.saturating_sub(left + cw);
        let tile_room = ctx
            .bounds
            .chroma_width
            .saturating_sub(ctx.cx.saturating_add(cw));
        cap_room.min(tile_room).min(cw)
    } else {
        0
    };
    let bl_ext = if has_bl && ch > 4 {
        let cap_c = 128usize >> sy;
        let cap_room = cap_c.saturating_sub(above + ch);
        let tile_room = ctx
            .bounds
            .chroma_height
            .saturating_sub(ctx.cy.saturating_add(ch));
        cap_room.min(tile_room).min(ch)
    } else {
        0
    };
    let luma_ref = build_luma_ref(ctx, cw, ch, above, left, tr_ext, bl_ext);
    let cref_u = build_chroma_ref(
        ctx.recu,
        &ChromaRefSpec {
            stride: ctx.pcw,
            bound_width: ctx.bounds.chroma_width,
            bound_height: ctx.bounds.chroma_height,
            y: ctx.cy,
            x: ctx.cx,
            width: cw,
            height: ch,
            above,
            left,
            top_right_extension: tr_ext,
            bottom_left_extension: bl_ext,
            is_top_sb: ctx.is_top_sb_boundary,
        },
    );
    let cref_v = build_chroma_ref(
        ctx.recv,
        &ChromaRefSpec {
            stride: ctx.pcw,
            bound_width: ctx.bounds.chroma_width,
            bound_height: ctx.bounds.chroma_height,
            y: ctx.cy,
            x: ctx.cx,
            width: cw,
            height: ch,
            above,
            left,
            top_right_extension: tr_ext,
            bottom_left_extension: bl_ext,
            is_top_sb: ctx.is_top_sb_boundary,
        },
    );

    // filter_dir signaling cost (3-ary) + mhccp switch (1 bit), approximated in
    // bits from the default CDFs; the classic alpha/index bits saved are folded
    // into `baseline_j`'s incumbent already, so here we only add MHCCP's own.
    let switch_bits = 1.0_f32;

    // Numerical-stability guard: skip MHCCP entirely on low-variance reference
    // regions, where the cross-component least-squares is near-singular and the
    // encoder's f32-derived filter diverges from the decoder's integer-derived
    // one (see [`MHCCP_MIN_REF_VAR`]). Measured on the luma reference the
    // decoder also sees, so the encoder's skip keeps the decoder from ever
    // deriving the exploded filter.
    if mhccp_ref_luma_variance(&luma_ref, above, left) < MHCCP_MIN_REF_VAR {
        return None;
    }

    let mut best: Option<(f32, CflChoice)> = None;
    for dir in 0u8..mhccp::MHCCP_MODE_NUM as u8 {
        let pu = mhccp::derive_params(&luma_ref, &cref_u, dir, bd);
        let pv = mhccp::derive_params(&luma_ref, &cref_v, dir, bd);
        let mut pred_u = vec![0i32; n];
        let mut pred_v = vec![0i32; n];
        mhccp::predict_block(
            &luma_ref,
            &pu,
            dir,
            ctx.have_top,
            ctx.have_left,
            bd,
            &mut pred_u,
        );
        mhccp::predict_block(
            &luma_ref,
            &pv,
            dir,
            ctx.have_top,
            ctx.have_left,
            bd,
            &mut pred_v,
        );
        let res_u: Vec<f32> = src_u[..n]
            .iter()
            .zip(pred_u.iter())
            .map(|(&s, &c)| s - c as f32)
            .collect();
        let res_v: Vec<f32> = src_v[..n]
            .iter()
            .zip(pred_v.iter())
            .map(|(&s, &c)| s - c as f32)
            .collect();
        let lev_u = chroma.project_scan(&res_u, 0.0, scan);
        let lev_v = chroma.project_scan(&res_v, 0.0, scan);
        let rec_u = itx422::reconstruct_chroma_cfl(&pred_u, &lev_u, qstep, scan, cw, ch, bd);
        let rec_v = itx422::reconstruct_chroma_cfl(&pred_v, &lev_v, qstep, scan, cw, ch, bd);
        // filter_dir bits from the size-group default CDF.
        let dir_bits = filter_dir_bits(ctx.size_group as usize, dir);
        let j = (pixel_sse_rounded(src_u, &rec_u) + pixel_sse_rounded(src_v, &rec_v)) as f32
            + lambda
                * ((coeff_rate_f32(&lev_u) + coeff_rate_f32(&lev_v) + switch_bits + dir_bits)
                    as f32);
        if best.as_ref().map(|(bj, _)| j < *bj).unwrap_or(true) {
            best = Some((
                j,
                CflChoice {
                    js: 0,
                    sign_u: 0,
                    sign_v: 0,
                    mag_u: 0,
                    mag_v: 0,
                    ctx_u: 0,
                    ctx_v: 0,
                    pred_u,
                    pred_v,
                    mhccp: Some(MhccpDecision {
                        mh_dir: dir,
                        size_group: ctx.size_group,
                    }),
                },
            ));
        }
    }

    match best {
        Some((j, choice)) => {
            // AVM selects MHCCP purely by RD cost (`av2_txfm_uvrd` + mode-signal
            // rate, compared via `RDCOST`); it wins only when its full rate-distortion
            // cost `j` beats the incumbent (DC / classic-alpha CfL). `j` already folds
            // in the reconstructed SSE plus lambda-weighted coefficient, switch and
            // filter_dir bits, so the decision is a direct `j < baseline_j`. An earlier
            // MSE-margin heuristic over-selected MHCCP (spending chroma bits for no
            // SSIMULACRA2 benefit); the pure-RD rule matches AVM and is never RD-negative
            // across correlated / semi-correlated / decorrelated calibration content.
            if j < baseline_j { Some(choice) } else { None }
        }
        _ => None,
    }
}

/// Reconstruction, source, geometry and RD state for one per-leaf MHCCP trial.
#[derive(Clone, Copy)]
pub(crate) struct MhccpLeafInput<'a> {
    pub(crate) reconstructed_luma: &'a [f32],
    pub(crate) luma_stride: usize,
    pub(crate) reconstructed_u: &'a [f32],
    pub(crate) reconstructed_v: &'a [f32],
    pub(crate) chroma_stride: usize,
    pub(crate) bounds: MhccpBounds,
    pub(crate) luma_y: usize,
    pub(crate) luma_x: usize,
    pub(crate) chroma_y: usize,
    pub(crate) chroma_x: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) subsample_x: bool,
    pub(crate) subsample_y: bool,
    pub(crate) have_top: bool,
    pub(crate) have_left: bool,
    pub(crate) source_u: &'a [f32],
    pub(crate) source_v: &'a [f32],
    pub(crate) dc_u: f32,
    pub(crate) dc_v: f32,
    pub(crate) rd: ChromaRdSpec<'a>,
    pub(crate) scan: &'a [u16],
    /// SB-relative 16x16 luma-mi coded mask (see [`MhccpCtx::coded`]).
    pub(crate) coded_mask: &'a [u8],
}

/// Per-leaf MHCCP evaluation shared by all chroma leaves and formats.
pub(crate) fn mhccp_eval_leaf(input: &MhccpLeafInput<'_>) -> Option<CflChoice> {
    let MhccpLeafInput {
        reconstructed_luma: recy,
        luma_stride: pw,
        reconstructed_u: recu,
        reconstructed_v: recv,
        chroma_stride: pcw,
        bounds,
        luma_y: ly,
        luma_x: lx,
        chroma_y: cy,
        chroma_x: cx,
        width: cw,
        height: ch,
        subsample_x: ssx,
        subsample_y: ssy,
        have_top,
        have_left,
        source_u: src_u,
        source_v: src_v,
        dc_u,
        dc_v,
        rd,
        scan,
        coded_mask,
    } = *input;
    let ChromaRdSpec {
        basis: chroma,
        qstep,
        lambda,
        bit_depth: bd,
    } = rd;
    let n = cw * ch;
    // DC-incumbent J = the FULL RD cost of the actual DC-prediction path: code the
    // DC-predicted chroma residual (project -> quantise -> reconstruct) and measure
    // reconstructed SSE + lambda*coeff-rate, exactly like the MHCCP candidate's `j`.
    // The previous baseline used only the flat DC-prediction SSE (no residual coding,
    // no rate), which is far larger than the real DC path -> MHCCP always "won" the
    // comparison and was selected even though DC+residuals reconstructs the chroma
    // 8+ dB better (the 4:4:4 low/mid-quality "green cast" collapse).
    let bres_u: Vec<f32> = src_u[..n].iter().map(|&s| s - dc_u).collect();
    let bres_v: Vec<f32> = src_v[..n].iter().map(|&s| s - dc_v).collect();
    let blev_u = chroma.project_scan(&bres_u, 0.0, scan);
    let blev_v = chroma.project_scan(&bres_v, 0.0, scan);
    // Match the actual non-MHCCP DC chroma path (code_444_chroma_leaf else-branch),
    // which uses the scalar-pred `reconstruct_chroma` (its own dc tx-scale/index) —
    // NOT the array-pred `_cfl` variant, which reconstructs differently and made the
    // baseline underestimate the DC path, letting MHCCP win when DC was better.
    let brec_u = itx422::reconstruct_chroma(dc_u, &blev_u, qstep, scan, cw, ch, bd);
    let brec_v = itx422::reconstruct_chroma(dc_v, &blev_v, qstep, scan, cw, ch, bd);
    let sse = (pixel_sse_rounded(&src_u[..n], &brec_u) + pixel_sse_rounded(&src_v[..n], &brec_v))
        + lambda * (coeff_rate_f32(&blev_u) + coeff_rate_f32(&blev_v));
    let bw4 = (cw << (ssx as usize)) / 4;
    let bh4 = (ch << (ssy as usize)) / 4;
    let ctx = MhccpCtx {
        recy,
        pw,
        recu,
        recv,
        pcw,
        bounds,
        ly,
        lx,
        cy,
        cx,
        ssx,
        ssy,
        have_top,
        have_left,
        // The current encoder uses 64x64 superblocks. AVM repeats the single
        // available line above only for blocks whose top edge is an actual SB
        // boundary; treating every partition leaf as such derives different
        // MHCCP parameters from the decoder for leaves lower in the SB.
        is_top_sb_boundary: (ly & 63) == 0,
        size_group: mhccp_size_group_wh4(bw4, bh4),
        coded: coded_mask,
    };
    mhccp_decide(
        &ctx,
        &MhccpDecisionInput {
            source_u: src_u,
            source_v: src_v,
            width: cw,
            height: ch,
            rd,
            scan,
            baseline_j: sse,
        },
    )
}

/// Approximate bits to code `mh_dir` under the default `filter_dir` CDF for the
/// given size-group context. Uses -log2(p_symbol) from the cumulative table.
fn filter_dir_bits(size_group: usize, dir: u8) -> f32 {
    // Cumulative CDF (AVM domain) per group: [c0, c1]; p(0)=c0, p(1)=c1-c0,
    // p(2)=32768-c1. (Stored elsewhere as ICDF; reconstruct AVM cdf here.)
    static CUM: [[u32; 2]; 4] = [
        [10923, 21845],
        [8795, 15105],
        [10433, 15974],
        [17085, 21689],
    ];
    let g = size_group.min(3);
    let (c0, c1) = (CUM[g][0], CUM[g][1]);
    let p = match dir {
        0 => c0,
        1 => c1 - c0,
        _ => 32768 - c1,
    } as f32
        / 32768.0;
    -dirty_log2f(p.max(1e-6))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_prediction<const N: usize>(
    pcw: usize,
    up: &[f32],
    vp: &[f32],
    cy: usize,
    cx: usize,
    ch: &&CflChoice,
    ru: &mut [f32],
    rv: &mut [f32],
) {
    for (r, (ru, rv)) in ru
        .as_chunks_mut::<N>()
        .0
        .iter_mut()
        .zip(rv.as_chunks_mut::<N>().0.iter_mut())
        .enumerate()
    {
        let b = (cy + r) * pcw + cx;
        let up_s = &up[b..b + N];
        let vp_s = &vp[b..b + N];
        let pred_u = &ch.pred_u[r * N..r * N + N];
        let pred_v = &ch.pred_v[r * N..r * N + N];
        for (((((ru, rv), &up), &vp), &pred_u), &pred_v) in ru
            .iter_mut()
            .zip(rv.iter_mut())
            .zip(up_s.iter())
            .zip(vp_s.iter())
            .zip(pred_u.iter())
            .zip(pred_v)
        {
            *ru = up - pred_u as f32;
            *rv = vp - pred_v as f32;
        }
    }
}

pub(crate) fn cfl_partition_prediction<const N: usize>(
    pcw: usize,
    up: &[f32],
    vp: &[f32],
    cy: usize,
    cx: usize,
    suf: &mut [f32],
    svf: &mut [f32],
) {
    for (r, (suf, svf)) in suf
        .as_chunks_mut::<N>()
        .0
        .iter_mut()
        .zip(svf.as_chunks_mut::<N>().0.iter_mut())
        .enumerate()
    {
        let b = (cy + r) * pcw + cx;
        let up_s = &up[b..b + N];
        let vp_s = &vp[b..b + N];
        for (((suf, svf), &up), &vp) in suf
            .iter_mut()
            .zip(svf.iter_mut())
            .zip(up_s.iter())
            .zip(vp_s.iter())
        {
            *suf = up;
            *svf = vp;
        }
    }
}

fn cfl_avg_l_444(
    recy: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    w: usize,
    h: usize,
    bd: i32,
) -> i32 {
    let have_top = sb_y > 0;
    let have_left = sb_x > 0;
    let ss_hor = if w > 32 { 2 } else { 1 };
    let ss_ver = if h > 32 { 2 } else { 1 };
    let mut sum: i32 = 0;
    let mut count: i32 = 0;
    if have_top {
        let base = (sb_y - 1) * pw + sb_x;
        let mut i = 0;
        while i < w {
            sum += (recy[base + i].fast_round() as i32) << 3;
            count += 1;
            i += ss_hor;
        }
    }
    if have_left {
        let mut i = 0;
        while i < h {
            sum += (recy[(sb_y + i) * pw + sb_x - 1].fast_round() as i32) << 3;
            count += 1;
            i += ss_ver;
        }
    }
    if count > 0 {
        let val = (sum + count / 2) / count;
        let max_v = (1 << (bd + 3)) - 1;
        val.min(max_v)
    } else {
        8 << (bd - 1)
    }
}

/// Plane references and block origin for a whole 64x64 4:4:4 CfL decision.
#[derive(Clone, Copy)]
pub(crate) struct Cfl64Input<'a> {
    pub(crate) reconstructed_luma: &'a [f32],
    pub(crate) source_u: &'a [f32],
    pub(crate) source_v: &'a [f32],
    pub(crate) reconstructed_u: &'a [f32],
    pub(crate) reconstructed_v: &'a [f32],
    pub(crate) stride: usize,
    pub(crate) y: usize,
    pub(crate) x: usize,
    pub(crate) neutral: f32,
}

pub(crate) fn cfl_decide_64(input: &Cfl64Input<'_>, rd: &ChromaRdSpec<'_>) -> Option<CflChoice> {
    let Cfl64Input {
        reconstructed_luma: recy,
        source_u: up,
        source_v: vp,
        reconstructed_u: recu,
        reconstructed_v: recv,
        stride: pw,
        y: sb_y,
        x: sb_x,
        neutral,
    } = *input;
    let ChromaRdSpec {
        basis: chroma,
        qstep,
        lambda,
        bit_depth: bd,
    } = *rd;
    let scan = &tables::SCAN;
    let mut luma = vec![0i32; 64 * 64];
    let mut su = vec![0i32; 64 * 64];
    let mut sv = vec![0i32; 64 * 64];
    let mut suf = vec![0f32; 64 * 64];
    let mut svf = vec![0f32; 64 * 64];
    for (r, ((((luma, su), sv), suf), svf)) in luma
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .zip(su.as_chunks_mut::<64>().0.iter_mut())
        .zip(sv.as_chunks_mut::<64>().0.iter_mut())
        .zip(suf.as_chunks_mut::<64>().0.iter_mut())
        .zip(svf.as_chunks_mut::<64>().0.iter_mut())
        .enumerate()
    {
        let b = (sb_y + r) * pw + sb_x;
        let recy = &recy[b..b + 64];
        let up = &up[b..b + 64];
        let vp = &vp[b..b + 64];
        for (((((((luma, su), sv), suf), svf), recy), &up), &vp) in luma
            .iter_mut()
            .zip(su.iter_mut())
            .zip(sv.iter_mut())
            .zip(suf.iter_mut())
            .zip(svf.iter_mut())
            .zip(recy.iter())
            .zip(up.iter())
            .zip(vp.iter())
        {
            *luma = recy.fast_round() as i32;
            *su = up.fast_round() as i32;
            *sv = vp.fast_round() as i32;
            *suf = up;
            *svf = vp;
        }
    }
    let luma_q3 = subsample_luma_q3(&luma, 64, 64, 64, false, false);
    let avg_l = cfl_avg_l_444(recy, pw, sb_y, sb_x, 64, 64, bd);
    let ac: Vec<i32> = luma_q3.iter().map(|&v| v - avg_l).collect();
    // TX_64X64 is one CfL prediction block. AVM limits the transform's coded
    // coefficient support to 32x32, but it does not split the intra predictor into
    // four independently predicted TX_32X32 quadrants. The decoder first builds one
    // 64x64 DC predictor and then adds the whole-block CfL AC signal to it.
    // Flat-DC alternative (the non-CfL DC intra mode) uses the standard
    // full-reference DC predictor, matching the caller's DC reconstruction path.
    let predu_f = dc_pred(recu, pw, sb_y, sb_x, 64, neutral);
    let predv_f = dc_pred(recv, pw, sb_y, sb_x, 64, neutral);
    // The CfL prediction base, however, must use AVM's subsampled-reference DC
    // (`highbd_dc_predictor_subsampled`), applied for `uv_mode == CFL && tx > 32`.
    let predu_cfl = dc_pred_cfl_subsampled(recu, pw, sb_y, sb_x, 64, neutral, bd);
    let predv_cfl = dc_pred_cfl_subsampled(recv, pw, sb_y, sb_x, 64, neutral, bd);
    let cand = cfl_candidate(
        &su,
        &sv,
        &ac,
        predu_cfl.fast_round() as i32,
        predv_cfl.fast_round() as i32,
        bd,
    )?;

    // DC candidate (flat prediction) for both planes.
    let res_dc_u: Vec<f32> = suf.iter().map(|&s| s - predu_f).collect();
    let res_dc_v: Vec<f32> = svf.iter().map(|&s| s - predv_f).collect();
    let lev_dc_u = chroma.project(&res_dc_u, 0.0);
    let lev_dc_v = chroma.project(&res_dc_v, 0.0);
    let rec_dc_u = itx422::reconstruct_chroma(predu_f, &lev_dc_u, qstep, scan, 64, 64, bd);
    let rec_dc_v = itx422::reconstruct_chroma(predv_f, &lev_dc_v, qstep, scan, 64, 64, bd);
    let j_dc = (pixel_sse_rounded(&suf, &rec_dc_u) + pixel_sse_rounded(&svf, &rec_dc_v)) as f32
        + lambda * (coeff_rate_f32(&lev_dc_u) + coeff_rate_f32(&lev_dc_v)) as f32;
    // CfL candidate for both planes.
    let res_cf_u: Vec<f32> = suf
        .iter()
        .zip(cand.pred_u.iter())
        .map(|(&s, &u)| s - u as f32)
        .collect();
    let res_cf_v: Vec<f32> = svf
        .iter()
        .zip(cand.pred_v.iter())
        .map(|(&s, &v)| s - v as f32)
        .collect();
    let lev_cf_u = chroma.project(&res_cf_u, 0.0);
    let lev_cf_v = chroma.project(&res_cf_v, 0.0);
    let rec_cf_u = itx422::reconstruct_chroma_cfl(&cand.pred_u, &lev_cf_u, qstep, scan, 64, 64, bd);
    let rec_cf_v = itx422::reconstruct_chroma_cfl(&cand.pred_v, &lev_cf_v, qstep, scan, 64, 64, bd);
    // Side info: is_cfl(~1) + cfl_index(~1) + joint sign(~3) + each nonzero mag(~3).
    let alpha_bits = 2.0
        + 3.0
        + if cand.sign_u != 0 { 3.0 } else { 0.0 }
        + if cand.sign_v != 0 { 3.0 } else { 0.0 };
    let j_cfl = (pixel_sse_rounded(&suf, &rec_cf_u) + pixel_sse_rounded(&svf, &rec_cf_v)) as f32
        + lambda * ((coeff_rate_f32(&lev_cf_u) + coeff_rate_f32(&lev_cf_v)) as f32 + alpha_bits);
    if j_cfl < j_dc { Some(cand) } else { None }
}
