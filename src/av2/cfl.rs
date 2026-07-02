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
use crate::av2::helpers::dc_pred;
use crate::av2::mhccp;
use crate::av2::proj::Basis;
use crate::av2::{itx422, tables};
use crate::util::FastRound;

pub(crate) const CFL_ADD_BITS_ALPHA: i32 = 5;
pub(crate) const CFL_ALPHABET_SIZE: u8 = 8; // magnitude indices 0..=7 -> |alpha| 1..=8

/// Branchless round-half-away-from-zero by `2^n`, bit-identical to
/// `if v < 0 { -round_pow2(-v, n) } else { round_pow2(v, n) }`.
#[inline(always)]
fn round_pow2_signed(v: i64, n: u32) -> i64 {
    let s = v >> 63;
    let av = (v ^ s) - s;
    let q = (av + (1i64 << (n - 1))) >> n;
    (q ^ s) - s
}

/// avm `get_scaled_luma_q0`: round(alpha_q3 * ac_q3, 6 + CFL_ADD_BITS_ALPHA).
#[inline(always)]
pub(crate) fn scaled_luma_q0(alpha_q3: i32, ac_q3: i32) -> i32 {
    round_pow2_signed(
        (alpha_q3 as i64) * (ac_q3 as i64),
        6 + CFL_ADD_BITS_ALPHA as u32,
    ) as i32
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

#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn subsample_luma_ring_q3(
    recy: &[f32],
    pw: usize,
    ly: usize,
    lx: usize,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
    have_top: bool,
    have_left: bool,
) -> Vec<i32> {
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
    // Edge replication when neighbours are unavailable.
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
    let sum: i64 = recon_q3.iter().map(|&v| v as i64).sum::<i64>() + ((npel as i64) >> 1);
    let avg = (sum >> log2) as i32;
    (recon_q3.iter().map(|&v| v - avg).collect(), avg)
}

/// CfL prediction: `chroma_dc + scaled_luma(alpha_q3, ac)`, clipped to bit depth.
pub(crate) fn cfl_predict(dc: i32, ac: &[i32], alpha_q3: i32, bitdepth: i32) -> Vec<i32> {
    let maxv = (1i32 << bitdepth) - 1;
    ac.iter()
        .map(|&a| (dc + scaled_luma_q0_i32(alpha_q3, a)).clamp(0, maxv))
        .collect()
}

/// Pick the per-plane alpha (sign, magnitude index) minimising SSE of the CfL
/// prediction vs. the chroma source. Returns `(sign, mag, sse)`; `sign == 0` (alpha
/// zero, i.e. flat DC) is the baseline. Encoder-side only — any choice is valid as
/// long as it is signalled; the decoder reproduces the prediction deterministically.
pub(crate) fn pick_alpha_plane(src: &[i32], dc: i32, ac: &[i32], bitdepth: i32) -> (u8, u8, u64) {
    let maxv = (1i32 << bitdepth) - 1;
    let dc_c = dc.clamp(0, maxv);
    // SSE of the CfL prediction for a fixed alpha. i32-only inner math (see
    // `scaled_luma_q0_i32`). For <=8-bit the squared-error sum is bounded by
    // `64*64*255^2 < 2^31`, so an i32 accumulator is safe and the loop autovectorises;
    // higher bit depths fall back to an i64 accumulator. Both are bit-identical.
    let sse_of = |alpha_q3: i32| -> u64 {
        if bitdepth <= 8 {
            let mut sse: i32 = 0;
            for (&s, &a) in src.iter().zip(ac.iter()) {
                let p = (dc + scaled_luma_q0_i32(alpha_q3, a)).clamp(0, maxv);
                let d = s - p;
                sse += d * d;
            }
            sse as u64
        } else {
            let mut sse: i64 = 0;
            for (&s, &a) in src.iter().zip(ac.iter()) {
                let p = (dc + scaled_luma_q0_i32(alpha_q3, a)).clamp(0, maxv);
                let d = (s - p) as i64;
                sse += d * d;
            }
            sse as u64
        }
    };
    // alpha == 0 baseline (prediction is flat dc)
    let base_sse = if bitdepth <= 8 {
        let mut s: i32 = 0;
        for &v in src {
            let d = v - dc_c;
            s += d * d;
        }
        s as u64
    } else {
        let mut s: i64 = 0;
        for &v in src {
            let d = (v - dc_c) as i64;
            s += d * d;
        }
        s as u64
    };
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

/// Build the best-alpha CfL candidate for a chroma block: per-plane alpha (sign,mag) by
/// SSE, resolved into joint sign + contexts + the two prediction blocks. Returns `None`
/// when both planes prefer flat DC (no CfL). The DC-vs-CfL rate-distortion decision is the
/// caller's job (it has the transform basis to measure real reconstructed cost).
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
    let l = |y: usize, x: usize| -> i64 { recy[y * pw + x].fast_round() as i64 };
    let mut sum: i64 = 0;
    let mut count: i64 = 0;
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
        let val = ((sum + count / 2) / count) as i32;
        let max_v = (1 << (bd + 3)) - 1;
        val.min(max_v)
    } else {
        8 << (bd - 1)
    }
}

/// Reconstructed-neighbour context needed to evaluate MHCCP for a chroma block.
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
) -> mhccp::MhccpRefBuf {
    let ref_width = left + width;
    let ref_height = above + height;
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
    let pw = ctx.pw as i32;
    let ph = (ctx.recy.len() / ctx.pw) as i32;
    let lp = |y: i32, x: i32| -> i32 {
        let yy = y.clamp(0, ph - 1);
        let xx = x.clamp(0, pw - 1);
        ctx.recy[(yy * pw + xx) as usize].fast_round() as i32
    };
    for j in 0..ref_height {
        for i in 0..ref_width {
            // luma top-left sample for this chroma ref position
            let ly = ly0 + ((j as i32) << sy);
            let lx = lx0 + ((i as i32) << sx);
            let q3 = match (ctx.ssx, ctx.ssy) {
                (true, true) => {
                    (lp(ly, lx) + lp(ly, lx + 1) + lp(ly + 1, lx) + lp(ly + 1, lx + 1)) << 1
                }
                (true, false) => (lp(ly, lx) + lp(ly, lx + 1)) << 2,
                (false, false) => lp(ly, lx) << 3,
                _ => unreachable!(),
            };
            buf.set(i, j, q3);
        }
    }
    buf
}

#[allow(clippy::too_many_arguments)]
fn build_chroma_ref(
    rec: &[f32],
    pcw: usize,
    cy: usize,
    cx: usize,
    width: usize,
    height: usize,
    above: usize,
    left: usize,
    is_top_sb: bool,
) -> mhccp::MhccpRefBuf {
    let ref_width = left + width;
    let ref_height = above + height;
    let mut buf =
        mhccp::MhccpRefBuf::new(ref_width, ref_height, above, left, width, height, is_top_sb);
    let pw = pcw as i32;
    let ph = (rec.len() / pcw) as i32;
    let cp = |y: i32, x: i32| -> i32 {
        let yy = y.clamp(0, ph - 1);
        let xx = x.clamp(0, pw - 1);
        rec[(yy * pw + xx) as usize].round() as i32
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

/// `cw*ch` chroma source blocks (`src_u`/`src_v`), the DC intra predictions (`dc_u_f`/
/// `dc_v_f`) and the neighbor luma DC (`avg_l` from `cfl_avg_l`). Builds the AC from the
/// block luma (subsampled per `ssx`/`ssy`) minus `avg_l`, forms the best-alpha candidate,
/// then does real rate-distortion: transform/quantise/reconstruct both DC and CfL per plane
/// and pick CfL only when it lowers J = SSE + lambda*bits. `chroma` is the format's basis.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_decide(
    recy: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    src_u: &[f32],
    src_v: &[f32],
    dc_u_f: f32,
    dc_v_f: f32,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
    avg_l: i32,
    bd: i32,
    chroma: &Basis,
    qstep: i32,
    lambda: f64,
) -> Option<CflChoice> {
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
    let coeff_bits = |lev: &[f32]| -> f64 {
        lev.iter()
            .filter(|&&x| x != 0.0)
            .map(|&x| 2.0 + 2.0 * ((x.abs() as f64) + 1.0).log2())
            .sum()
    };
    let sse = |s: &[f32], r: &[f32]| -> f64 {
        s.iter()
            .zip(r)
            .map(|(&a, &b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum()
    };
    let res_dc_u: Vec<f32> = src_u.iter().map(|&s| s - dc_u_f).collect();
    let res_dc_v: Vec<f32> = src_v.iter().map(|&s| s - dc_v_f).collect();
    let lev_dc_u = chroma.project(&res_dc_u, 0.0);
    let lev_dc_v = chroma.project(&res_dc_v, 0.0);
    let rec_dc_u = itx422::reconstruct_chroma(dc_u_f, &lev_dc_u, qstep, scan, cw, ch, bd);
    let rec_dc_v = itx422::reconstruct_chroma(dc_v_f, &lev_dc_v, qstep, scan, cw, ch, bd);
    let j_dc = sse(src_u, &rec_dc_u)
        + sse(src_v, &rec_dc_v)
        + lambda * (coeff_bits(&lev_dc_u) + coeff_bits(&lev_dc_v));
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
    let j_cfl = sse(src_u, &rec_cf_u)
        + sse(src_v, &rec_cf_v)
        + lambda * (coeff_bits(&lev_cf_u) + coeff_bits(&lev_cf_v) + alpha_bits);
    if j_cfl < j_dc { Some(cand) } else { None }
}

/// Score a chroma predictor pair with the same J metric used by `cfl_decide`
/// (SSE + lambda*coeff_bits, excluding mode bits which the caller adds). Lets the
/// caller obtain the incumbent CfL/DC J to compare against MHCCP on equal terms.
#[allow(clippy::too_many_arguments)]
pub(crate) fn score_predictor(
    src_u: &[f32],
    src_v: &[f32],
    pred_u: &[i32],
    pred_v: &[i32],
    cw: usize,
    ch: usize,
    bd: i32,
    chroma: &Basis,
    qstep: i32,
    lambda: f64,
    mode_bits: f64,
) -> f64 {
    let n = cw * ch;
    let scan = &tables::SCAN;
    let coeff_bits = |lev: &[f32]| -> f64 {
        lev.iter()
            .filter(|&&x| x != 0.0)
            .map(|&x| 2.0 + 2.0 * ((x.abs() as f64) + 1.0).log2())
            .sum()
    };
    let sse = |s: &[f32], r: &[f32]| -> f64 {
        s.iter()
            .zip(r)
            .map(|(&a, &b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum()
    };
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
    sse(src_u, &rec_u)
        + sse(src_v, &rec_v)
        + lambda * (coeff_bits(&lev_u) + coeff_bits(&lev_v) + mode_bits)
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
/// the causal neighbour ring, reconstructing both chroma planes, and scoring
/// J = SSE + lambda*(coeff_bits + mode_bits). Returns a `CflChoice` carrying the
/// MHCCP predictor when the best direction beats `baseline_j` (the J of the
/// incumbent — DC or classic CfL). Mirrors the `CFL_MULTI_PARAM` arm of avm's
/// chroma intra RD loop.
///
/// `mode_bits_extra` is the approximate signaling delta of choosing MHCCP over
/// the incumbent: the `cfl_mhccp_switch` symbol plus the 3-ary `filter_dir`
/// symbol, minus the alpha/index bits the incumbent would have spent. It is
/// supplied by a small model calibrated against SSIMULACRA2 (see `MHCCP_*` in
/// mod.rs) so the tool is only taken when it actually helps perceptually.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mhccp_decide(
    ctx: &MhccpCtx,
    src_u: &[f32],
    src_v: &[f32],
    cw: usize,
    ch: usize,
    bd: i32,
    chroma: &Basis,
    qstep: i32,
    lambda: f64,
    scan: &[u16],
    baseline_j: f64,
) -> Option<CflChoice> {
    let n = cw * ch;
    let coeff_bits = |lev: &[f32]| -> f64 {
        lev.iter()
            .filter(|&&x| x != 0.0)
            .map(|&x| 2.0 + 2.0 * ((x.abs() as f64) + 1.0).log2())
            .sum()
    };
    let sse = |s: &[f32], r: &[f32]| -> f64 {
        s.iter()
            .zip(r)
            .map(|(&a, &b)| {
                let d = (a - b) as f64;
                d * d
            })
            .sum()
    };

    // avm above_lines/left_lines: (LINE_NUM+1) chroma lines when the neighbour is
    // available (interior blocks). At the top-of-superblock boundary avm uses the
    // same count but with a one-line padding offset handled inside the builders.
    let line_num1 = mhccp::LINE_NUM + 1;
    let above = if ctx.have_top { line_num1 } else { 0 };
    let left = if ctx.have_left { line_num1 } else { 0 };
    if above == 0 && left == 0 {
        return None; // no causal support -> MHCCP degenerates
    }

    let luma_ref = build_luma_ref(ctx, cw, ch, above, left);
    let cref_u = build_chroma_ref(
        ctx.recu,
        ctx.pcw,
        ctx.cy,
        ctx.cx,
        cw,
        ch,
        above,
        left,
        ctx.is_top_sb_boundary,
    );
    let cref_v = build_chroma_ref(
        ctx.recv,
        ctx.pcw,
        ctx.cy,
        ctx.cx,
        cw,
        ch,
        above,
        left,
        ctx.is_top_sb_boundary,
    );

    // filter_dir signaling cost (3-ary) + mhccp switch (1 bit), approximated in
    // bits from the default CDFs; the classic alpha/index bits saved are folded
    // into `baseline_j`'s incumbent already, so here we only add MHCCP's own.
    let switch_bits = 1.0_f64;

    let mut best: Option<(f64, CflChoice)> = None;
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
        let j = sse(src_u, &rec_u)
            + sse(src_v, &rec_v)
            + lambda * (coeff_bits(&lev_u) + coeff_bits(&lev_v) + switch_bits + dir_bits);
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
            // SSIMULACRA2-calibrated acceptance margin. MSE-optimal MHCCP over-
            // selects at high quality where it gives no perceptual benefit, so we
            // require the J-improvement to exceed a quality-dependent threshold
            // T(qstep) = MHCCP_MARGIN_A + MHCCP_MARGIN_B / qstep, normalized by the
            // block energy scale (qstep^2 * N). Derived from SSIMULACRA2 agreement
            // analysis over correlated-chroma content (see calibration notes):
            // strict at low qstep (high quality), permissive at high qstep.
            let n_f = (cw * ch) as f64;
            let q = qstep.max(1) as f64;
            let t = MHCCP_MARGIN_A + MHCCP_MARGIN_B / q;
            let norm_gain = (baseline_j - j) / (q * q * n_f);
            if norm_gain > t { Some(choice) } else { None }
        }
        _ => None,
    }
}

/// Per-leaf MHCCP evaluation shared by all chroma leaves and formats
#[allow(clippy::too_many_arguments)]
pub(crate) fn mhccp_eval_leaf(
    recy: &[f32],
    pw: usize,
    recu: &[f32],
    recv: &[f32],
    pcw: usize,
    ly: usize,
    lx: usize,
    cy: usize,
    cx: usize,
    cw: usize,
    ch: usize,
    ssx: bool,
    ssy: bool,
    have_top: bool,
    have_left: bool,
    src_u: &[f32],
    src_v: &[f32],
    dc_u: f32,
    dc_v: f32,
    chroma: &Basis,
    qstep: i32,
    lambda: f64,
    scan: &[u16],
    bd: i32,
) -> Option<CflChoice> {
    let n = cw * ch;
    // DC-incumbent J = SSE of the DC-predicted residual across both planes.
    let mut sse = 0f64;
    for i in 0..n {
        let du = src_u[i] - dc_u;
        let dv = src_v[i] - dc_v;
        sse += (du * du + dv * dv) as f64;
    }
    let bw4 = (cw << (ssx as usize)) / 4;
    let bh4 = (ch << (ssy as usize)) / 4;
    let ctx = MhccpCtx {
        recy,
        pw,
        recu,
        recv,
        pcw,
        ly,
        lx,
        cy,
        cx,
        ssx,
        ssy,
        have_top,
        have_left,
        is_top_sb_boundary: true,
        size_group: mhccp_size_group_wh4(bw4, bh4),
    };
    mhccp_decide(
        &ctx, src_u, src_v, cw, ch, bd, chroma, qstep, lambda, scan, sse,
    )
}
/// `T(qstep) = MHCCP_MARGIN_A + MHCCP_MARGIN_B / qstep`; MHCCP is accepted only
/// when the normalized J improvement `(J_incumbent - J_mhccp)/(qstep^2 * N)`
/// exceeds `T`. Fit against SSIMULACRA2 on correlated-chroma blocks.
pub(crate) const MHCCP_MARGIN_A: f64 = -0.0039;
pub(crate) const MHCCP_MARGIN_B: f64 = 0.097;

/// Approximate bits to code `mh_dir` under the default `filter_dir` CDF for the
/// given size-group context. Uses -log2(p_symbol) from the cumulative table.
fn filter_dir_bits(size_group: usize, dir: u8) -> f64 {
    // Cumulative CDF (AVM domain) per group: [c0, c1]; p(0)=c0, p(1)=c1-c0,
    // p(2)=32768-c1. (Stored elsewhere as ICDF; reconstruct AVM cdf here.)
    const CUM: [[u32; 2]; 4] = [
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
    } as f64
        / 32768.0;
    -(p.max(1e-6)).log2()
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
    let mut sum: i64 = 0;
    let mut count: i64 = 0;
    if have_top {
        let base = (sb_y - 1) * pw + sb_x;
        let mut i = 0;
        while i < w {
            sum += (recy[base + i].fast_round() as i64) << 3;
            count += 1;
            i += ss_hor;
        }
    }
    if have_left {
        let mut i = 0;
        while i < h {
            sum += (recy[(sb_y + i) * pw + sb_x - 1].fast_round() as i64) << 3;
            count += 1;
            i += ss_ver;
        }
    }
    if count > 0 {
        let val = ((sum + count / 2) / count) as i32;
        let max_v = (1 << (bd + 3)) - 1;
        val.min(max_v)
    } else {
        8 << (bd - 1)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn cfl_decide_64(
    recy: &[f32],
    up: &[f32],
    vp: &[f32],
    recu: &[f32],
    recv: &[f32],
    pw: usize,
    sb_y: usize,
    sb_x: usize,
    bd: i32,
    neutral: f32,
    chroma: &Basis,
    qstep: i32,
    lambda: f64,
) -> Option<CflChoice> {
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
    let predu_f = dc_pred(recu, pw, sb_y, sb_x, 64, neutral);
    let predv_f = dc_pred(recv, pw, sb_y, sb_x, 64, neutral);
    let dc_u = predu_f.fast_round() as i32;
    let dc_v = predv_f.fast_round() as i32;
    let cand = cfl_candidate(&su, &sv, &ac, dc_u, dc_v, bd)?;

    let coeff_bits = |lev: &[f32]| -> f64 {
        let mut b = 0.0;
        for &l in lev {
            if l != 0.0 {
                b += 2.0 + 2.0 * ((l.abs() as f64) + 1.0).log2();
            }
        }
        b
    };
    let sse = |src: &[f32], rec: &[f32]| -> f64 {
        src.iter()
            .zip(rec)
            .map(|(&s, &r)| {
                let d = (s - r) as f64;
                d * d
            })
            .sum()
    };
    // DC candidate (flat prediction) for both planes.
    let res_dc_u: Vec<f32> = suf.iter().map(|&s| s - predu_f).collect();
    let res_dc_v: Vec<f32> = svf.iter().map(|&s| s - predv_f).collect();
    let lev_dc_u = chroma.project(&res_dc_u, 0.0);
    let lev_dc_v = chroma.project(&res_dc_v, 0.0);
    let rec_dc_u = itx422::reconstruct_chroma(predu_f, &lev_dc_u, qstep, scan, 64, 64, bd);
    let rec_dc_v = itx422::reconstruct_chroma(predv_f, &lev_dc_v, qstep, scan, 64, 64, bd);
    let j_dc = sse(&suf, &rec_dc_u)
        + sse(&svf, &rec_dc_v)
        + lambda * (coeff_bits(&lev_dc_u) + coeff_bits(&lev_dc_v));
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
    let j_cfl = sse(&suf, &rec_cf_u)
        + sse(&svf, &rec_cf_v)
        + lambda * (coeff_bits(&lev_cf_u) + coeff_bits(&lev_cf_v) + alpha_bits);
    if j_cfl < j_dc { Some(cand) } else { None }
}
