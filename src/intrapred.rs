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

pub(crate) const DC_PRED: usize = 0;

/// directional modes (V/H and the diagonals, 1..=8) are intentionally omitted
/// from this set: they would also require the `angle_delta` symbol on >= 8x8
/// blocks. These four need no extra symbols beyond `y_mode` itself.
pub(crate) const SMOOTH_PRED: usize = 9;
pub(crate) const SMOOTH_V_PRED: usize = 10;
pub(crate) const SMOOTH_H_PRED: usize = 11;
pub(crate) const PAETH_PRED: usize = 12;
/// Directional modes added in this increment (axis-aligned only, at
/// `angle_delta = 0`): pure vertical / horizontal copy. They sit in the
/// directional range 1..=8 (`VERT_LEFT_PRED`), so they emit an `angle_delta`
/// symbol — but at delta 0 (angle 90/180) the decoder maps them straight to the
/// plain copy predictors, with no top-right/bottom-left edge extension.
pub(crate) const V_PRED: usize = 1;
pub(crate) const H_PRED: usize = 2;
/// Z2 diagonal modes (down-right directions). At `angle_delta = 0` their angles
/// are 135/113/157, all in (90,180) -> the dav1d `ipred_z2` path, which reads
/// only top + left + corner (no top-right/bottom-left edge extension), so they
/// reuse the same reference arrays as the other modes. The Z1 (D45/D67) and Z3
/// (D203) diagonals — which DO need the extension samples — are still deferred.
pub(crate) const D135_PRED: usize = 4;
pub(crate) const D113_PRED: usize = 5;
pub(crate) const D157_PRED: usize = 6;
pub(crate) const VERT_LEFT_PRED: usize = 8;
/// Z1 diagonals (up-right): `D45` (45 deg) and `D67` (= `VERT_LEFT_PRED`, 67 deg).
/// They read the top row extended to the right (top-right samples). Z3 diagonal
/// (down-left): `D203` (203 deg), reading the left column extended downward
/// (bottom-left samples). These need the neighbor-availability derivation
/// (dav1d's intra-edge tree) and the extended reference arrays.
pub(crate) const D45_PRED: usize = 3;
pub(crate) const D203_PRED: usize = 7;
/// Chroma-from-luma. Signalled as a `uv_mode` symbol; its tx-type is **not** in
/// `txtp_from_uvmode`, so it defaults to `DCT_DCT` — i.e. CfL needs no ADST.
pub(crate) const CFL_PRED: usize = 13;

/// `default_cfl_sign_cdf` (libaom/dav1d): joint sign of the U/V alphas, 8 symbols.
pub(crate) static CFL_SIGN_CDF: [u16; 7] = [1418, 2123, 13340, 18405, 26972, 28343, 32294];
/// `default_cfl_alpha_cdf[6]`: per-plane alpha magnitude (1..=16 -> symbols 0..=15),
/// indexed by a context derived from the joint sign.
pub(crate) static CFL_ALPHA_CDF: [[u16; 15]; 6] = [
    [
        7637, 20719, 31401, 32481, 32657, 32688, 32692, 32696, 32700, 32704, 32708, 32712, 32716,
        32720, 32724,
    ],
    [
        14365, 23603, 28135, 31168, 32167, 32395, 32487, 32573, 32620, 32647, 32668, 32672, 32676,
        32680, 32684,
    ],
    [
        11532, 22380, 28445, 31360, 32349, 32523, 32584, 32649, 32673, 32677, 32681, 32685, 32689,
        32693, 32697,
    ],
    [
        26990, 31402, 32282, 32571, 32692, 32696, 32700, 32704, 32708, 32712, 32716, 32720, 32724,
        32728, 32732,
    ],
    [
        17248, 26058, 28904, 30608, 31305, 31877, 32126, 32321, 32394, 32464, 32516, 32560, 32576,
        32593, 32622,
    ],
    [
        14738, 21678, 25779, 27901, 29024, 30302, 30980, 31843, 32144, 32413, 32520, 32594, 32622,
        32656, 32660,
    ],
];

/// `dav1d_dr_intra_derivative[44]` — angle -> projection step (1/64 px). Indexed
/// `[(angle-90)>>1]` for the vertical step and `[(180-angle)>>1]` for the
/// horizontal step in the Z2 predictor.
pub(crate) static DR_INTRA_DERIVATIVE: [i32; 44] = [
    0, 1023, 0, 547, 372, 0, 0, 273, 215, 0, 178, 151, 0, 132, 116, 0, 102, 0, 90, 80, 0, 71, 64,
    0, 57, 51, 0, 45, 0, 40, 35, 0, 31, 27, 0, 23, 19, 0, 15, 0, 11, 0, 7, 3,
];

/// `dav1d_intra_mode_context` — maps an intra mode to its keyframe y-mode CDF
/// context (0..=4), used for both the above and left neighbors.
pub(crate) static INTRA_MODE_CTX: [usize; 13] = [0, 1, 2, 3, 4, 4, 4, 4, 3, 0, 1, 2, 0];

/// `dav1d_sm_weights` slice for a given block dimension (SMOOTH predictors).
pub(crate) fn sm_weights(n: usize) -> &'static [i32] {
    match n {
        4 => &[255, 149, 85, 64],
        8 => &[255, 197, 146, 105, 73, 50, 37, 32],
        16 => &[
            255, 225, 196, 170, 145, 123, 102, 84, 68, 54, 43, 33, 26, 20, 17, 16,
        ],
        32 => &[
            255, 240, 225, 210, 196, 182, 169, 157, 145, 133, 122, 111, 101, 92, 83, 74, 66, 59,
            52, 45, 39, 34, 29, 25, 21, 17, 14, 12, 10, 9, 8, 8,
        ],
        _ => unreachable!("sm_weights size {}", n),
    }
}

/// Build the AV1 intra reference edges from the reconstructed plane and predict
/// `mode` into `out` (row-major `bw*bh`). Bit-exact with dav1d's non-directional
/// predictors (`ipred_{paeth,smooth,smooth_v,smooth_h}_c`) and the default-fill
/// rules of `dav1d_prepare_intra_edges` (single-tile raster order: above/left
/// availability = not at the frame's top/left edge). `recon`/`stride` is the
/// reconstructed plane; `(ox, oy)` the block's pixel origin. DC is handled by
/// the dedicated `dc_pred_*` helpers, not here.
/// CfL luma-AC for 4:4:4: the reconstructed luma block scaled by 8 with its mean
/// removed, exactly as dav1d's `cfl_ac` with `ss_hor = ss_ver = 0`.
pub(crate) fn cfl_ac_444(luma_rec: &[i32], w: usize, h: usize, ac: &mut [i32]) {
    let n = w * h;
    for (ac, luma) in ac[..n].iter_mut().zip(luma_rec[..n].iter()) {
        *ac = *luma << 3;
    }
    let log2sz = w.trailing_zeros() + h.trailing_zeros();
    let mut sum: i64 = (1i64 << log2sz) >> 1;
    for ac in ac[..n].iter() {
        sum += *ac as i64;
    }
    let mean = (sum >> log2sz) as i32;
    for ac in ac[..n].iter_mut() {
        *ac -= mean;
    }
}

/// CfL luma-AC for subsampled chroma (dav1d `cfl_ac` with `ss_hor`/`ss_ver`).
/// `luma_rec` is the reconstructed luma block, stride `lstride`, covering the
/// chroma block of `(cw, ch)` samples. Each chroma position sums the covered
/// luma samples and is scaled by `1 << (1 + !ss_ver + !ss_hor)` (so 4:4:4 ->
/// `<< 3`, matching `cfl_ac_444`), then the block mean is removed.
pub(crate) fn cfl_ac_sub(
    luma_rec: &[i32],
    lstride: usize,
    cw: usize,
    ch: usize,
    ss_hor: bool,
    ss_ver: bool,
    ac: &mut [i32],
) {
    let shift = 1 + (!ss_ver as u32) + (!ss_hor as u32);
    let sx = ss_hor as usize;
    let sy = ss_ver as usize;
    for y in 0..ch {
        let ac = &mut ac[y * cw..y * cw + cw];
        for (x, dst) in ac[..cw].iter_mut().enumerate() {
            let ly = y << sy;
            let lx = x << sx;
            let mut s = luma_rec[ly * lstride + lx];
            if ss_hor {
                s += luma_rec[ly * lstride + lx + 1];
            }
            if ss_ver {
                s += luma_rec[(ly + 1) * lstride + lx];
                if ss_hor {
                    s += luma_rec[(ly + 1) * lstride + lx + 1];
                }
            }
            *dst = s << shift;
        }
    }
    let n = cw * ch;
    let log2sz = cw.trailing_zeros() + ch.trailing_zeros();
    let mut sum: i64 = (1i64 << log2sz) >> 1;
    for v in ac[..n].iter() {
        sum += *v as i64;
    }
    let mean = (sum >> log2sz) as i32;
    for v in ac[..n].iter_mut() {
        *v -= mean;
    }
}

/// CfL prediction combine (dav1d `cfl_pred`): `dc + sign(diff)*((|diff|+32)>>6)`.
#[inline]
pub(crate) fn cfl_pred_pixel(dc: i32, ac: i32, alpha: i32, bd: u8) -> i32 {
    let diff = alpha * ac;
    let mag = (diff.abs() + 32) >> 6;
    let s = if diff < 0 { -mag } else { mag };
    (dc + s).clamp(0, (1 << bd) - 1)
}

/// Energy-minimising CfL alpha for one plane, in dav1d alpha units (the predictor
/// applies `alpha/64` after the <<3 AC scaling). Returns the best of the analytic
/// optimum and its +/-1 neighbors by pre-quantisation residual energy, clamped to
/// the signalled range [-16, 16] (0 means "CfL useless for this plane").
pub(crate) fn cfl_best_alpha(ac: &[i32], src: &[i32], dc: i32, n: usize, bd: u8) -> i32 {
    let mut num: i64 = 0;
    let mut den: i64 = 0;
    for i in 0..n {
        num += (src[i] - dc) as i64 * ac[i] as i64;
        den += ac[i] as i64 * ac[i] as i64;
    }
    if den == 0 {
        return 0;
    }
    let a0 = ((64 * num + (den >> 1) * num.signum()) / den).clamp(-16, 16) as i32;
    let mut best_a = 0i32;
    let mut best_e = i64::MAX;
    for cand in [a0 - 1, a0, a0 + 1] {
        if !(-16..=16).contains(&cand) {
            continue;
        }
        let mut e: i64 = 0;
        for i in 0..n {
            let d = (src[i] - cfl_pred_pixel(dc, ac[i], cand, bd)) as i64;
            e += d * d;
        }
        if e < best_e {
            best_e = e;
            best_a = cand;
        }
    }
    best_a
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_predict_nd(
    mode: usize,
    recon: &[i32],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    out: &mut [i32],
    bd: u8,
) {
    intra_predict_nd_ad(
        mode, 0, recon, stride, ox, oy, bw, bh, have_tr, have_bl, fw, fh, out, bd,
    )
}

/// As [`intra_predict_nd`] but with an explicit AV1 `angle_delta` in
/// `-3..=3` (steps of 3°). The delta is applied only to the six pure diagonal
/// modes (D45/D67/D135/D113/D157/D203), whose ±9° range stays within a single
/// z1/z2/z3 prediction path so the existing dispatch and reference setup are
/// reused unchanged; V/H/DC/SMOOTH*/PAETH ignore it. The `DR_INTRA_DERIVATIVE`
/// table has valid entries at every `base + delta*3` angle, so this is bit-exact
/// with dav1d (edge filter / upsampling remain off).
#[allow(clippy::too_many_arguments)]
pub(crate) fn intra_predict_nd_ad(
    mode: usize,
    angle_delta: i32,
    recon: &[i32],
    stride: usize,
    ox: usize,
    oy: usize,
    bw: usize,
    bh: usize,
    have_tr: bool,
    have_bl: bool,
    fw: usize,
    fh: usize,
    out: &mut [i32],
    bd: u8,
) {
    let have_top = oy > 0;
    let have_left = ox > 0;
    let base = 1i32 << (bd - 1);
    // Sized for the directional reach at the largest supported block (32): top
    // row + top-right extension (Z1) and left column + bottom-left extension
    // (Z3), each up to 2x the block dim, so 2*32 = 64.
    let mut top = [0i32; 64];
    let mut left = [0i32; 64];
    if have_top {
        for i in 0..bw {
            top[i] = recon[(oy - 1) * stride + ox + i];
        }
    } else {
        let fill = if have_left {
            recon[oy * stride + ox - 1]
        } else {
            base - 1
        };
        top[..bw].fill(fill);
    }
    if have_left {
        for j in 0..bh {
            left[j] = recon[(oy + j) * stride + ox - 1];
        }
    } else {
        let fill = if have_top {
            recon[(oy - 1) * stride + ox]
        } else {
            base + 1
        };
        left[..bh].fill(fill);
    }
    let corner = if have_left {
        if have_top {
            recon[(oy - 1) * stride + ox - 1]
        } else {
            recon[oy * stride + ox - 1]
        }
    } else if have_top {
        recon[(oy - 1) * stride + ox]
    } else {
        base
    };
    // Top-right extension (top[bw..2bw]) for Z1, and bottom-left extension
    // (left[bh..2bh]) for Z3, following dav1d_prepare_intra_edges: copy the
    // available samples (clamped to the frame edge) then replicate, or — when
    // the neighbor is unavailable — replicate the last edge sample.
    if have_tr {
        let px_have = bw.min(fw - (ox + bw));
        for i in 0..px_have {
            top[bw + i] = recon[(oy - 1) * stride + ox + bw + i];
        }
        let fill = top[bw + px_have - 1];
        for i in px_have..bw {
            top[bw + i] = fill;
        }
    } else {
        let fill = top[bw - 1];
        for i in 0..bw {
            top[bw + i] = fill;
        }
    }
    if have_bl {
        let px_have = bh.min(fh - (oy + bh));
        for i in 0..px_have {
            left[bh + i] = recon[(oy + bh + i) * stride + ox - 1];
        }
        let fill = left[bh + px_have - 1];
        for i in px_have..bh {
            left[bh + i] = fill;
        }
    } else {
        let fill = left[bh - 1];
        for i in 0..bh {
            left[bh + i] = fill;
        }
    }

    match mode {
        V_PRED => {
            for orow in out.chunks_exact_mut(bw) {
                orow.copy_from_slice(&top[..bw]);
            }
        }
        H_PRED => {
            for (orow, &lv) in out.chunks_exact_mut(bw).zip(left.iter()) {
                orow.iter_mut().for_each(|o| *o = lv);
            }
        }
        D45_PRED | VERT_LEFT_PRED => {
            // dav1d ipred_z1 (edge filter/upsampling off): project from the top
            // row (extended with top-right samples). D45 -> 45 deg, D67 -> 67 deg.
            let angle: i32 = (if mode == D45_PRED { 45 } else { 67 }) + angle_delta * 3;
            let dx = DR_INTRA_DERIVATIVE[(angle >> 1) as usize];
            let max_base_x = (bw + bw.min(bh) - 1) as i32;
            for y in 0..bh {
                let xpos = dx * (y as i32 + 1);
                let frac = xpos & 0x3E;
                let mut bx = xpos >> 6;
                #[allow(clippy::explicit_counter_loop)]
                for x in 0..bw {
                    if bx < max_base_x {
                        let v = top[bx as usize] * (64 - frac) + top[(bx + 1) as usize] * frac;
                        out[y * bw + x] = (v + 32) >> 6;
                    } else {
                        let fill = top[max_base_x as usize];
                        let row = y * bw;
                        out[row + x..row + bw].fill(fill);
                        break;
                    }
                    bx += 1;
                }
            }
        }
        D203_PRED => {
            // dav1d ipred_z3 (edge filter/upsampling off): project from the left
            // column (extended with bottom-left samples). D203 -> 203 deg.
            let angle: i32 = 203 + angle_delta * 3;
            let dy = DR_INTRA_DERIVATIVE[((270 - angle) >> 1) as usize];
            let max_base_y = (bh + bw.min(bh) - 1) as i32;
            for x in 0..bw {
                let ypos = dy * (x as i32 + 1);
                let frac = ypos & 0x3E;
                let mut by = ypos >> 6;
                #[allow(clippy::explicit_counter_loop)]
                for y in 0..bh {
                    if by < max_base_y {
                        let v = left[by as usize] * (64 - frac) + left[(by + 1) as usize] * frac;
                        out[y * bw + x] = (v + 32) >> 6;
                    } else {
                        let fill = left[max_base_y as usize];
                        for yy in y..bh {
                            out[yy * bw + x] = fill;
                        }
                        break;
                    }
                    by += 1;
                }
            }
        }
        D135_PRED | D113_PRED | D157_PRED => {
            // dav1d ipred_z2 with edge filter/upsampling disabled: pure angular
            // projection from the top row, left column and corner.
            let angle: i32 = (match mode {
                D135_PRED => 135,
                D113_PRED => 113,
                _ => 157,
            }) + angle_delta * 3;
            let dy = DR_INTRA_DERIVATIVE[((angle - 90) >> 1) as usize];
            let dx = DR_INTRA_DERIVATIVE[((180 - angle) >> 1) as usize];
            // topleft[idx]: idx 0 = corner, idx>=1 = top[idx-1], idx<0 = left[-idx-1]
            let tl = |idx: i32| -> i32 {
                if idx >= 0 {
                    if idx == 0 {
                        corner
                    } else {
                        top[((idx - 1) as usize).min(bw - 1)]
                    }
                } else {
                    left[((-idx - 1) as usize).min(bh - 1)]
                }
            };
            for y in 0..bh as i32 {
                let xpos = (1 << 6) - dx * (y + 1);
                let mut base_x = xpos >> 6;
                let frac_x = xpos & 0x3E;
                let mut ypos = (y << 6) - dy;
                #[allow(clippy::explicit_counter_loop)]
                for x in 0..bw as i32 {
                    let v = if base_x >= 0 {
                        tl(base_x) * (64 - frac_x) + tl(base_x + 1) * frac_x
                    } else {
                        let base_y = ypos >> 6;
                        let frac_y = ypos & 0x3E;
                        tl(-1 - base_y) * (64 - frac_y) + tl(-2 - base_y) * frac_y
                    };
                    out[(y * bw as i32 + x) as usize] = (v + 32) >> 6;
                    base_x += 1;
                    ypos -= dy;
                }
            }
        }
        PAETH_PRED => paeth_pred(bw, bh, &top, &left, corner, out),
        SMOOTH_PRED => smooth_pred(bw, bh, &top, &left, out),
        SMOOTH_V_PRED => smooth_v_pred(bw, bh, &top, &left, out),
        SMOOTH_H_PRED => smooth_h_pred(bw, bh, &top, &left, out),
        _ => unreachable!("intra_predict_nd called with mode {}", mode),
    }
}

pub(crate) fn paeth_pred(
    bw: usize,
    _bh: usize,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [i32],
) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bw.is_multiple_of(4) {
            unsafe { neon::paeth(bw, _bh, top, left, corner, out) };
            return;
        }
    }
    for (y, orow) in out.chunks_exact_mut(bw).enumerate() {
        let lv = left[y];
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

/// AV1 SMOOTH predictor (4-tap vertical+horizontal weighted blend), bit-exact to
/// dav1d `ipred_smooth_c`. Dispatches to a NEON+MAC kernel on aarch64, scalar
/// elsewhere. `top`/`left` hold the prepared edges; output is row-major `bw*bh`.
pub(crate) fn smooth_pred(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bw.is_multiple_of(4) {
            unsafe { neon::smooth(bw, bh, top, left, out) };
            return;
        }
    }
    let (wv, wh) = (sm_weights(bh), sm_weights(bw));
    let (right, bottom) = (top[bw - 1], left[bh - 1]);
    for ((orow, &wvy), &lv) in out.chunks_exact_mut(bw).zip(wv.iter()).zip(left.iter()) {
        for (o, (&tv, &whx)) in orow.iter_mut().zip(top.iter().zip(wh.iter())) {
            let pred = wvy * tv + (256 - wvy) * bottom + whx * lv + (256 - whx) * right;
            *o = (pred + 256) >> 9;
        }
    }
}

/// AV1 SMOOTH_V predictor (vertical half), bit-exact to dav1d `ipred_smooth_v_c`.
pub(crate) fn smooth_v_pred(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bw.is_multiple_of(4) {
            unsafe { neon::smooth_v(bw, bh, top, left, out) };
            return;
        }
    }
    let wv = sm_weights(bh);
    let bottom = left[bh - 1];
    for (orow, &wvy) in out.chunks_exact_mut(bw).zip(wv.iter()) {
        for (o, &tv) in orow.iter_mut().zip(top.iter()) {
            *o = (wvy * tv + (256 - wvy) * bottom + 128) >> 8;
        }
    }
}

/// AV1 SMOOTH_H predictor (horizontal half), bit-exact to dav1d `ipred_smooth_h_c`.
pub(crate) fn smooth_h_pred(bw: usize, _bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if bw.is_multiple_of(4) {
            unsafe { neon::smooth_h(bw, _bh, top, left, out) };
            return;
        }
    }
    let wh = sm_weights(bw);
    let right = top[bw - 1];
    for (orow, &lv) in out.chunks_exact_mut(bw).zip(left.iter()) {
        for (o, &whx) in orow.iter_mut().zip(wh.iter()) {
            *o = (whx * lv + (256 - whx) * right + 128) >> 8;
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod neon {
    use super::sm_weights;
    use core::arch::aarch64::*;

    #[inline]
    #[target_feature(enable = "neon")]
    fn mla_n(acc: int32x4_t, v: int32x4_t, k: i32) -> int32x4_t {
        vmlaq_s32(acc, v, vdupq_n_s32(k))
    }

    #[target_feature(enable = "neon")]
    pub(super) fn smooth(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
        let (wv, wh) = (sm_weights(bh), sm_weights(bw));
        let (right, bottom) = (top[bw - 1], left[bh - 1]);
        let c256 = vdupq_n_s32(256);
        let rnd = vdupq_n_s32(256);
        for y in 0..bh {
            let (wvy, lv) = (wv[y], left[y]);
            let base = vdupq_n_s32((256 - wvy) * bottom);
            let row = &mut out[y * bw..y * bw + bw];
            let mut x = 0;
            while x < bw {
                unsafe {
                    let tv = vld1q_s32(top[x..].as_ptr());
                    let whx = vld1q_s32(wh[x..].as_ptr());
                    let w2 = vsubq_s32(c256, whx);
                    let mut acc = mla_n(base, tv, wvy); // base + top*wvy
                    acc = mla_n(acc, whx, lv); // + wh*left[y]
                    acc = mla_n(acc, w2, right); // + (256-wh)*right
                    vst1q_s32(row[x..].as_mut_ptr(), vshrq_n_s32::<9>(vaddq_s32(acc, rnd)));
                }
                x += 4;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn smooth_v(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
        let wv = sm_weights(bh);
        let bottom = left[bh - 1];
        let rnd = vdupq_n_s32(128);
        for y in 0..bh {
            let wvy = wv[y];
            let base = vdupq_n_s32((256 - wvy) * bottom);
            let row = &mut out[y * bw..y * bw + bw];
            let mut x = 0;
            while x < bw {
                unsafe {
                    let tv = vld1q_s32(top[x..].as_ptr());
                    let acc = mla_n(base, tv, wvy);
                    vst1q_s32(row[x..].as_mut_ptr(), vshrq_n_s32(vaddq_s32(acc, rnd), 8));
                }
                x += 4;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn smooth_h(bw: usize, bh: usize, top: &[i32], left: &[i32], out: &mut [i32]) {
        let wh = sm_weights(bw);
        let right = top[bw - 1];
        let c256 = vdupq_n_s32(256);
        let rnd = vdupq_n_s32(128);
        for (y, &lv) in left[..bh].iter().enumerate() {
            let row = &mut out[y * bw..y * bw + bw];
            let mut x = 0;
            while x < bw {
                let whx = unsafe { vld1q_s32(wh[x..].as_ptr()) };
                let w2 = vsubq_s32(c256, whx);
                let mut acc = mla_n(vdupq_n_s32(0), w2, right); // (256-wh)*right
                acc = mla_n(acc, whx, lv); // + wh*left[y]
                unsafe {
                    vst1q_s32(row[x..].as_mut_ptr(), vshrq_n_s32(vaddq_s32(acc, rnd), 8));
                }
                x += 4;
            }
        }
    }

    #[target_feature(enable = "neon")]
    pub(super) fn paeth(
        bw: usize,
        bh: usize,
        top: &[i32],
        left: &[i32],
        corner: i32,
        out: &mut [i32],
    ) {
        let cn = vdupq_n_s32(corner);
        for (y, &lv) in left[..bh].iter().enumerate() {
            let lvv = vdupq_n_s32(lv);
            let lmc = vdupq_n_s32(lv - corner);
            let row = &mut out[y * bw..y * bw + bw];
            let mut x = 0;
            while x < bw {
                let tv = unsafe { vld1q_s32(top[x..].as_ptr()) };
                let b = vaddq_s32(tv, lmc); // lv + tv - corner
                let ld = vabdq_s32(lvv, b);
                let td = vabdq_s32(tv, b);
                let cd = vabdq_s32(cn, b);
                let m_tv = vcleq_s32(td, cd); // td <= cd
                let m_lv = vandq_u32(vcleq_s32(ld, td), vcleq_s32(ld, cd)); // ld<=td && ld<=cd
                let mut res = vbslq_s32(m_tv, tv, cn); // td<=cd ? tv : corner
                res = vbslq_s32(m_lv, lvv, res); // ld<=...? lv : res
                unsafe {
                    vst1q_s32(row[x..].as_mut_ptr(), res);
                }
                x += 4;
            }
        }
    }
}

pub(crate) static ND_LUMA_MODES: [usize; 13] = [
    DC_PRED,
    V_PRED,
    H_PRED,
    D45_PRED,
    D135_PRED,
    D113_PRED,
    D157_PRED,
    D203_PRED,
    VERT_LEFT_PRED,
    SMOOTH_PRED,
    SMOOTH_V_PRED,
    SMOOTH_H_PRED,
    PAETH_PRED,
];

/// Candidate luma modes evaluated by the intra mode search.
pub(crate) fn nd_modes() -> &'static [usize] {
    &ND_LUMA_MODES
}

/// Reduced candidate set for the fast RDO path (`speed >= 1`): keep DC, the
/// planar-like SMOOTH, and PAETH, and drop the SMOOTH_V/SMOOTH_H variants
/// (their wins over SMOOTH are rare and small). Mirrors libaom's intra-mode
/// pruning at higher `--cpu-used`.
pub(crate) fn fast_nd_modes() -> &'static [usize] {
    const FAST: [usize; 3] = [DC_PRED, SMOOTH_PRED, PAETH_PRED];
    &FAST
}

/// 8x8 DC_PRED from a reconstructed plane (stride 64). `(ox, oy)` pixel origin.
pub(crate) fn dc_pred_8x8(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(8)
                    .sum::<i32>();
            (s + 8) >> 4
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            (s + 4) >> 3
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            (s + 4) >> 3
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 4x4 chroma block (dav1d `dc_gen`, 8-bit). w==h==4 is a
/// power of two so no reciprocal multiply is needed.
pub(crate) fn dc_pred_4x4(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 4i32; // (4+4)>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(4)
                    .sum::<i32>();
            s >> 3 // ctz(8)
        }
        (true, false) => {
            let mut s = 2i32;
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s >> 2
        }
        (false, true) => {
            let mut s = 2i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(4)
                .sum::<i32>();
            s >> 2
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 4-wide x 8-tall chroma block (dav1d `dc_gen`, 8-bit).
/// w+h = 12 is not a power of two, so the both-edges case uses the reciprocal
/// multiply (ctz(12)=2 shift, then *0x5556>>16 since 8 is not > 2*4).
/// DC predictor for an 8-wide x 4-tall block (transpose of 4x8): 8 above + 4 left.
pub(crate) fn dc_pred_8x4(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 6i32; // (8+4)>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(4)
                .sum::<i32>();
            s >>= 2; // ctz(12)
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 4i32; // 8>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s >> 3
        }
        (false, true) => {
            let mut s = 2i32; // 4>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(4)
                .sum::<i32>();
            s >> 2
        }
        (false, false) => 1 << (bd - 1),
    }
}

pub(crate) fn dc_pred_4x8(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 6i32; // (4+8)>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >>= 2; // ctz(4+8)
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 2i32; // 4>>1
            s += recon[(oy - 1) * stride + ox..][..4].iter().sum::<i32>();
            s >> 2 // ctz(4)
        }
        (false, true) => {
            let mut s = 4i32; // 8>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >> 3 // ctz(8)
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC predictor for an 8-wide x 16-tall chroma block (4:2:2 `RTX_8X16`). Mirrors
/// dav1d/AV1 DC_PRED: average of the 8 above + 16 left reconstructed neighbors
/// (w+h = 24 = 8*3, so `>>3` then the `*0x5556>>16` divide-by-3); single-edge
/// and no-edge cases fall back to the available average or 128.
pub(crate) fn dc_pred_8x16(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 12i32; // (8+16)>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >>= 3; // ctz(8+16) = ctz(24) = 3
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 4i32; // 8>>1
            s += recon[(oy - 1) * stride + ox..][..8].iter().sum::<i32>();
            s >> 3 // ctz(8)
        }
        (false, true) => {
            let mut s = 8i32; // 16>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >> 4 // ctz(16)
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 16x8 (wide) block: above 16 samples, left 8 samples.
pub(crate) fn dc_pred_16x8(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 12i32; // (16+8)>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >>= 3; // ctz(16+8) = ctz(24) = 3
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 8i32; // 16>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s >> 4 // ctz(16)
        }
        (false, true) => {
            let mut s = 4i32; // 8>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(8)
                .sum::<i32>();
            s >> 3 // ctz(8)
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 16x16 block (mirror of `dc_pred_8x8`).
pub(crate) fn dc_pred_16x16(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(16)
                    .sum::<i32>();
            (s + 16) >> 5
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            (s + 8) >> 4
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            (s + 8) >> 4
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC prediction for a 32x32 block (mirror of `dc_pred_16x16`).
pub(crate) fn dc_pred_32x32(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>()
                + recon[oy * stride + ox - 1..]
                    .iter()
                    .step_by(stride)
                    .take(32)
                    .sum::<i32>();
            (s + 32) >> 6
        }
        (true, false) => {
            let mut s = 0i32;
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>();
            (s + 16) >> 5
        }
        (false, true) => {
            let mut s = 0i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            (s + 16) >> 5
        }
        (false, false) => 1 << (bd - 1),
    }
}

/// DC predictor for a 16-wide x 32-tall chroma block (4:2:2 `RTX_16X32`).
/// Mirrors `dc_pred_8x16`: sum 16 above + 32 left = 48 = 16*3 samples.
/// DC predictor for 32-wide x 16-tall (transpose of 16x32): 32 above + 16 left.
pub(crate) fn dc_pred_32x16(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 24i32; // (32+16)>>1
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >>= 4; // ctz(48)
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 16i32;
            s += recon[(oy - 1) * stride + ox..][..32].iter().sum::<i32>();
            s >> 5
        }
        (false, true) => {
            let mut s = 8i32;
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(16)
                .sum::<i32>();
            s >> 4
        }
        (false, false) => 1 << (bd - 1),
    }
}

pub(crate) fn dc_pred_16x32(recon: &[i32], stride: usize, ox: usize, oy: usize, bd: i32) -> i32 {
    let above = oy > 0;
    let left = ox > 0;
    match (above, left) {
        (true, true) => {
            let mut s = 24i32; // (16+32)>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            s >>= 4; // ctz(48) = 4
            (((s as u32) * 0x5556) >> 16) as i32
        }
        (true, false) => {
            let mut s = 8i32; // 16>>1
            s += recon[(oy - 1) * stride + ox..][..16].iter().sum::<i32>();
            s >> 4
        }
        (false, true) => {
            let mut s = 16i32; // 32>>1
            s += recon[oy * stride + ox - 1..]
                .iter()
                .step_by(stride)
                .take(32)
                .sum::<i32>();
            s >> 5
        }
        (false, false) => 1 << (bd - 1),
    }
}
