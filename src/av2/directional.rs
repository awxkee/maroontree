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
#[rustfmt::skip]
static DR: [i32; 90] = [
    0,4096,2048,1365,1024,819,682,585,512,455,409,409,409,372,341,292,273,256,
    227,215,204,186,178,170,157,151,146,136,132,128,117,110,107,99,97,97,93,87,
    83,81,77,74,73,69,66,64,62,59,56,55,53,50,49,47,44,42,42,41,38,37,35,32,
    31,30,28,27,26,24,23,22,20,19,18,16,15,14,12,11,10,10,10,9,8,7,6,5,4,3,2,1,
];

// 4-tap luma directional sub-pixel filter (DR_INTERP_FILTER), 32 phases.
#[rustfmt::skip]
static F4: [[i32;4];32] = [
    [0,128,0,0],[-2,127,4,-1],[-3,125,8,-2],[-5,123,13,-3],[-6,121,17,-4],[-7,118,22,-5],
    [-9,116,27,-6],[-9,112,32,-7],[-10,109,37,-8],[-11,106,41,-8],[-11,102,46,-9],[-12,98,52,-10],
    [-12,94,56,-10],[-12,90,61,-11],[-12,85,66,-11],[-12,81,71,-12],[-12,76,76,-12],[-12,71,81,-12],
    [-11,66,85,-12],[-11,61,90,-12],[-10,56,94,-12],[-10,52,98,-12],[-9,46,102,-11],[-8,41,106,-11],
    [-8,37,109,-10],[-7,32,112,-9],[-6,27,116,-9],[-5,22,118,-7],[-4,17,121,-6],[-3,13,123,-5],
    [-2,8,125,-3],[-1,4,127,-2],
];

#[inline]
fn clip_px(v: i32, max_pixel: i32) -> i32 {
    v.clamp(0, max_pixel)
}
#[inline]
fn iclip(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Intra-edge reference filter strength (AVM `intra_edge_filter_strength`,
/// reconintra.c:958). `smooth` = filter type 1: the above or left block uses a
/// SMOOTH / SMOOTH_V / SMOOTH_H mode (`get_filt_type`).
fn filter_strength(wh: i32, angle: i32, smooth: bool) -> i32 {
    let d = angle.abs();
    if smooth {
        return if wh <= 8 {
            if d >= 64 { 2 } else { (d >= 40) as i32 }
        } else if wh <= 16 {
            if d >= 48 { 2 } else { (d >= 20) as i32 }
        } else if wh <= 24 {
            if d >= 4 { 3 } else { 0 }
        } else if d >= 1 {
            3
        } else {
            0
        };
    }
    if wh <= 8 {
        if d >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if d >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if d >= 32 {
            return 3;
        }
        if d >= 16 {
            return 2;
        }
        if d >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if d >= 32 {
            return 3;
        }
        if d >= 4 {
            return 2;
        }
        if d >= 1 {
            return 1;
        }
    } else if d >= 1 {
        return 3;
    }
    0
}

/// 5-tap intra-edge filter (`filter_edge` in the decoder).
#[allow(clippy::too_many_arguments)]
fn filter_edge(
    out: &mut [i32],
    sz: usize,
    lim_from: i32,
    lim_to: i32,
    inp: &[i32],
    from: i32,
    to: i32,
    strength: usize,
) {
    static K: [[i32; 5]; 3] = [[0, 4, 8, 4, 0], [0, 5, 6, 5, 0], [2, 4, 4, 4, 2]];
    // pad[k] = inp[clip(k - 2)]: the clipped reads become a plain 5-tap window.
    let mut pad = [0i32; 129 + 4];
    let pad = &mut pad[..sz + 4];
    for (k, p) in pad.iter_mut().enumerate() {
        *p = inp[iclip(k as i32 - 2, from, to - 1) as usize];
    }
    let k = &K[strength - 1];
    let lo = (lim_from.max(0) as usize).min(sz);
    let hi = (lim_to.max(0) as usize).min(sz).max(lo);
    out[..lo].copy_from_slice(&pad[2..lo + 2]);
    let (p0, p1, p2, p3, p4) = (
        &pad[lo..hi],
        &pad[lo + 1..hi + 1],
        &pad[lo + 2..hi + 2],
        &pad[lo + 3..hi + 3],
        &pad[lo + 4..hi + 4],
    );
    for (i, o) in out[lo..hi].iter_mut().enumerate() {
        *o = (k[0] * p0[i] + k[1] * p1[i] + k[2] * p2[i] + k[3] * p3[i] + k[4] * p4[i] + 8) >> 4;
    }
    out[hi..sz].copy_from_slice(&pad[hi + 2..sz + 2]);
}

/// Build the decoder-style edge buffer (`tl`,`o`) from encoder refs.
/// `tl[o] = corner`, `tl[o+1+i] = above[i]`, `tl[o-1-i] = left[i]`.
fn edge_buffer(bs: usize, above: &[i32], left: &[i32], corner: i32) -> ([i32; 260], usize) {
    let o = 2 * bs;
    assert!(bs <= 64);
    let mut tl = [0i32; 4 * 64 + 4];
    tl[o] = corner;
    for i in 0..2 * bs {
        tl[o + 1 + i] = above[i];
        tl[o - 1 - i] = left[i];
    }
    (tl, o)
}

/// z1: above-only directional prediction (angles 0..90), port of `ipred_z1`.
#[allow(clippy::too_many_arguments)]
fn z1(
    dst: &mut [i32],
    bs: usize,
    tl: &[i32],
    o: usize,
    angle: i32,
    is_luma: bool,
    edge: bool,
    smooth: bool,
    max_width: i32,
    max_pixel: i32,
) {
    let (w, h) = (bs, bs);
    let dx = DR[angle as usize];
    let max_base_x = (w + h) as i32 - 1;
    let mut filt = [0i32; 141];
    let top_off = 2usize;
    let sz = 1 + w + h;
    let str = if edge {
        filter_strength((w + h) as i32, 90 - angle, smooth)
    } else {
        0
    };
    if str > 0 {
        filter_edge(
            &mut filt[1..],
            sz,
            1,
            sz as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[1..1 + sz].copy_from_slice(&tl[o..o + sz]);
    }
    filt[0] = filt[1];
    filt[sz + 1] = filt[sz];
    filt[sz + 2] = filt[sz + 1];

    let fill = filt[top_off + max_base_x as usize];
    let mut ypos = dx;
    for row in dst[..w * h].chunks_exact_mut(w) {
        let base = ypos >> 6;
        // Samples whose base passes max_base_x repeat the last edge sample.
        let n = (max_base_x - base + 1).clamp(0, w as i32) as usize;
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let (head, tail) = row.split_at_mut(n);
        if n > 0 {
            let s = (top_off as i32 + base) as usize;
            if is_luma {
                interp4(head, &filt[s - 1..s + n + 2], &F4[shift], max_pixel);
            } else {
                interp2(head, &filt[s..s + n + 1], shift as i32, max_pixel);
            }
        }
        tail.fill(fill);
        ypos += dx;
    }
}

/// `out[i]` = the 4-tap luma filter over `edge[i..i + 4]`.
#[inline(always)]
fn interp4(out: &mut [i32], edge: &[i32], f: &[i32; 4], max_pixel: i32) {
    let n = out.len();
    let (e0, e1, e2, e3) = (
        &edge[..n],
        &edge[1..n + 1],
        &edge[2..n + 2],
        &edge[3..n + 3],
    );
    for i in 0..n {
        let v = f[0] * e0[i] + f[1] * e1[i] + f[2] * e2[i] + f[3] * e3[i];
        out[i] = clip_px((v + 64) >> 7, max_pixel);
    }
}

/// `out[i]` = the 2-tap chroma filter over `edge[i..i + 2]`.
#[inline(always)]
fn interp2(out: &mut [i32], edge: &[i32], shift: i32, max_pixel: i32) {
    let n = out.len();
    let (e0, e1) = (&edge[..n], &edge[1..n + 1]);
    for i in 0..n {
        let v = (32 - shift) * e0[i] + shift * e1[i];
        out[i] = clip_px((v + 16) >> 5, max_pixel);
    }
}

/// z3: left-only directional prediction (angles 180..270), port of `ipred_z3`.
#[allow(clippy::too_many_arguments)]
fn z3(
    dst: &mut [i32],
    bs: usize,
    tl: &[i32],
    o: usize,
    angle: i32,
    is_luma: bool,
    edge: bool,
    smooth: bool,
    max_height: i32,
    max_pixel: i32,
) {
    let (w, h) = (bs, bs);
    let dy = DR[(270 - angle) as usize];
    let max_base_y = (w + h) as i32 - 1;
    let mut filt = [0i32; 141];
    let left_off = 1 + w + h;
    let sz = 1 + w + h;
    let str = if edge {
        filter_strength((w + h) as i32, angle - 180, smooth)
    } else {
        0
    };
    if str > 0 {
        // The decoder filters the left column together with the above-left corner:
        // av2_filter_intra_edge_high(left_col - ab_le, n_left + ab_le + txwpx, ...) with
        // ab_le = 1 for angle > 180 (need_above_left). In decoder order the samples are
        // [corner, left[0], left[1], ...]; the filter leaves index 0 (corner) unchanged
        // and filters the rest. Our `tl` buffer stores the left column in reverse with the
        // corner at the deep end (tl[o]); filtering that reversed order with the symmetric
        // kernel matches the decoder for the interior, but the corner must sit at the far
        // (unfiltered) end so it plays the same boundary role. sz = 1 + w + h already
        // includes the corner, so filter the whole span.
        filter_edge(
            &mut filt[2..],
            sz,
            h as i32 - max_height,
            sz as i32 - 1,
            &tl[o + 1 - sz..],
            0,
            sz as i32,
            str as usize,
        );
    } else {
        filt[2..2 + sz].copy_from_slice(&tl[o + 1 - sz..o + 1]);
    }
    filt[0] = filt[2];
    filt[1] = filt[2];
    filt[sz + 2] = filt[sz + 1];

    // Each column walks the left edge downwards (decreasing `filt` index), so it is
    // filtered in increasing index order with mirrored taps into `col`, reversed.
    let fill = filt[left_off - max_base_y as usize];
    let mut col = [0i32; 64];
    let mut ypos = dy;
    for x in 0..w {
        let base = ypos >> 6;
        let n = (max_base_y - base + 1).clamp(0, h as i32) as usize;
        let shift = ((ypos & 0x3F) >> 1) as usize;
        if n > 0 {
            let e = left_off - base as usize;
            if is_luma {
                let f = &F4[shift];
                interp4(
                    &mut col[..n],
                    &filt[e - n - 1..e + 2],
                    &[f[3], f[2], f[1], f[0]],
                    max_pixel,
                );
            } else {
                interp2(
                    &mut col[..n],
                    &filt[e - n..e + 1],
                    32 - shift as i32,
                    max_pixel,
                );
            }
        }
        for (y, &v) in col[..n].iter().rev().enumerate() {
            dst[y * w + x] = v;
        }
        for y in n..h {
            dst[y * w + x] = fill;
        }
        ypos += dy;
    }
}

/// Edge-filter and geometry state used by the z2 directional predictor.
#[derive(Clone, Copy)]
struct Z2Spec {
    block_size: usize,
    is_luma: bool,
    filter_corner: bool,
    filter_top: bool,
    filter_left: bool,
    smooth: bool,
    max_width: i32,
    max_height: i32,
    max_pixel: i32,
}

/// z2: both-edge directional prediction (angles 90..180), port of `ipred_z2`.
fn z2(dst: &mut [i32], tl: &[i32], o: usize, angle: i32, spec: &Z2Spec) {
    let Z2Spec {
        block_size: bs,
        is_luma,
        filter_corner,
        filter_top,
        filter_left,
        smooth,
        max_width,
        max_height,
        max_pixel,
    } = *spec;
    let (w, h) = (bs, bs);
    let dy = DR[(angle - 90) as usize];
    let dx = DR[(180 - angle) as usize];

    // Corner filter (decoder `filter_intra_edge_corner_high`): for z2, when the intra
    // edge filter is enabled and w+h >= 24, the above-left sample is smoothed with a
    // 5-6-5 kernel over [left[0], corner, above[0]] BEFORE the per-edge filters run and
    // independent of their strengths. The smoothed corner feeds both edges.
    let sz_t = 1 + w;
    let sz_l = 1 + h;
    let corner_s = if filter_corner && w + h >= 24 {
        (tl[o - 1] * 5 + tl[o] * 6 + tl[o + 1] * 5 + 8) >> 4
    } else {
        tl[o]
    };
    let mut top_in = [0i32; 72];
    top_in[..sz_t].copy_from_slice(&tl[o..o + sz_t]);
    top_in[0] = corner_s;
    let mut left_in = [0i32; 72];
    left_in[..sz_l].copy_from_slice(&tl[o - h..o + 1]);
    left_in[sz_l - 1] = corner_s;

    // top edge buffer
    let mut filt = [0i32; 72];
    let top_off = 0usize;
    let str_t = if filter_top {
        filter_strength((w + h) as i32, angle - 90, smooth)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - w as i32,
            &top_in,
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&top_in[..sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // left edge buffer
    let mut filt2 = [0i32; 72];
    let left_off = h + 2;
    let str_l = if filter_left {
        filter_strength((w + h) as i32, 180 - angle, smooth)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &left_in,
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&left_in[..sz_l]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    // Row y takes its first `split[y]` samples from the left edge (xpos < -64)
    // and the rest from the top edge; `split` is non-decreasing in y.
    let mut split = [0usize; 64];
    for (y, sp) in split[..h].iter_mut().enumerate() {
        let t = (y as i32 + 1) * dx - 64;
        *sp = if t <= 0 {
            0
        } else {
            (((t + 63) >> 6) as usize).min(w)
        };
    }
    // left-reference part, by column: the phase is fixed down a column and the
    // base advances by one per row (decreasing `filt2` index), as in z3.
    let mut col = [0i32; 64];
    let mut y0 = 0usize;
    for x in 0..w {
        while y0 < h && split[y0] <= x {
            y0 += 1;
        }
        if y0 == h {
            break;
        }
        let n = h - y0;
        let c = -((x as i32 + 1) * dy);
        let shift = ((c & 0x3F) >> 1) as usize;
        // filt2 index of the last row's base sample.
        let last = left_off as i32 - (h as i32 - 1) - (c >> 6);
        if is_luma {
            let f = &F4[shift];
            let s = (last - 4) as usize;
            interp4(
                &mut col[..n],
                &filt2[s..s + n + 3],
                &[f[3], f[2], f[1], f[0]],
                max_pixel,
            );
        } else {
            let s = (last - 3) as usize;
            interp2(
                &mut col[..n],
                &filt2[s..s + n + 1],
                32 - shift as i32,
                max_pixel,
            );
        }
        for (k, &v) in col[..n].iter().enumerate() {
            dst[(h - 1 - k) * w + x] = v;
        }
    }
    // top-reference part: xpos steps by whole samples, so the phase is fixed
    // and the base advances by one per pixel.
    for y in 0..h {
        let x = split[y];
        if x < w {
            let xpos = -((y as i32 + 1) * dx) + 64 * x as i32;
            let m = w - x;
            let shift = ((xpos & 0x3F) >> 1) as usize;
            // base_x may be -1; the +1 / +2 tap offsets keep the index >= 0.
            let ti = top_off as i32 + (xpos >> 6);
            let row = &mut dst[y * w + x..(y + 1) * w];
            if is_luma {
                let s = (ti + 1) as usize;
                interp4(row, &filt[s..s + m + 3], &F4[shift], max_pixel);
            } else {
                let s = (ti + 2) as usize;
                interp2(row, &filt[s..s + m + 1], shift as i32, max_pixel);
            }
        }
    }
}

/// Nominal directional luma intra modes (angle_delta = 0).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum Dir {
    V,
    H,
    D45,
    D135,
    D113,
    D157,
    D203,
    D67,
}

impl Dir {
    pub(crate) fn angle(self) -> i32 {
        match self {
            Dir::V => 90,
            Dir::H => 180,
            Dir::D45 => 45,
            Dir::D135 => 135,
            Dir::D113 => 113,
            Dir::D157 => 157,
            Dir::D203 => 203,
            Dir::D67 => 67,
        }
    }
}

/// Geometry, angle delta and edge availability for one directional prediction.
#[derive(Clone, Copy)]
pub(crate) struct DirectionalSpec {
    pub(crate) angle_delta: i32,
    pub(crate) block_size: usize,
    pub(crate) is_luma: bool,
    pub(crate) have_top: bool,
    pub(crate) have_left: bool,
    pub(crate) edge_filter: bool,
    pub(crate) max_width: i32,
    pub(crate) max_height: i32,
    /// Largest sample value, `(1 << bit_depth) - 1`; AVM clips every directional
    /// output to it (an 8-bit clip mispredicts every 10/12-bit block).
    pub(crate) max_pixel: i32,
    /// Edge-filter type 1: the above or left block is SMOOTH/SMOOTH_V/SMOOTH_H.
    /// Only changes the strength when the TX width + height is <= 32.
    pub(crate) smooth_neighbour: bool,
}

/// Directional predictor for a square block, conformant to the decoder.
pub(crate) fn directional(
    dir: Dir,
    above: &[i32],
    left: &[i32],
    corner: i32,
    spec: &DirectionalSpec,
) -> Vec<f32> {
    let bs = spec.block_size;
    let mut out = vec![0f32; bs * bs];
    directional_into(dir, above, left, corner, spec, &mut out);
    out
}

thread_local! {
    /// Integer prediction scratch of [`directional_into`] (up to 64x64).
    static DIR_SCRATCH: std::cell::RefCell<Box<[i32; 4096]>> =
        std::cell::RefCell::new(Box::new([0; 4096]));
}

/// [`directional`] into `dst[..bs*bs]`.
pub(crate) fn directional_into(
    dir: Dir,
    above: &[i32],
    left: &[i32],
    corner: i32,
    spec: &DirectionalSpec,
    dst: &mut [f32],
) {
    let DirectionalSpec {
        angle_delta,
        block_size: bs,
        is_luma,
        have_top,
        have_left,
        edge_filter,
        max_width,
        max_height,
        max_pixel,
        smooth_neighbour,
    } = *spec;
    DIR_SCRATCH.with(|cell| {
        let out = &mut cell.borrow_mut()[..bs * bs];
        out.fill(0);
        // AVM filters the above edge when have_top and the left edge when have_left,
        // independently (reconintra.c:1301-1311). z1 uses above, z3 uses left, z2 both.
        let ang = dir.angle() + angle_delta * 3;
        let edge_top = edge_filter && have_top;
        let edge_left = edge_filter && have_left;
        let vert_copy = ang == 90 || (ang < 90 && !have_top);
        let hor_copy = ang == 180 || (ang > 180 && !have_left);
        if vert_copy {
            for row in out.chunks_mut(bs) {
                row.copy_from_slice(&above[..bs]);
            }
        } else if hor_copy {
            for (r, row) in out.chunks_mut(bs).enumerate() {
                row.fill(left[r]);
            }
        } else {
            let (tl, o) = edge_buffer(bs, above, left, corner);
            if ang < 90 {
                z1(
                    out,
                    bs,
                    &tl,
                    o,
                    ang,
                    is_luma,
                    edge_top,
                    smooth_neighbour,
                    max_width,
                    max_pixel,
                );
            } else if ang < 180 {
                // z2 filters each edge independently when its neighbor exists; the corner
                // is prepared whenever the edge filter is enabled (missing physical samples
                // were already synthesized by build_refs), matching AVM.
                z2(
                    out,
                    &tl,
                    o,
                    ang,
                    &Z2Spec {
                        block_size: bs,
                        is_luma,
                        filter_corner: edge_filter,
                        filter_top: edge_top,
                        filter_left: edge_left,
                        smooth: smooth_neighbour,
                        max_width,
                        max_height,
                        max_pixel,
                    },
                );
            } else {
                z3(
                    out,
                    bs,
                    &tl,
                    o,
                    ang,
                    is_luma,
                    edge_left,
                    smooth_neighbour,
                    max_height,
                    max_pixel,
                );
            }
        }
        for (d, &v) in dst[..bs * bs].iter_mut().zip(out.iter()) {
            *d = v as f32;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn refs(bs: usize, seed: u64) -> (Vec<i32>, Vec<i32>, i32) {
        let mut s = seed;
        let mut nx = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) % 256) as i32
        };
        (
            (0..2 * bs).map(|_| nx()).collect(),
            (0..2 * bs).map(|_| nx()).collect(),
            nx(),
        )
    }

    #[test]
    fn v_h_are_copies() {
        let (a, l, c) = refs(8, 1);
        let v = directional(
            Dir::V,
            &a,
            &l,
            c,
            &DirectionalSpec {
                angle_delta: 0,
                block_size: 8,
                is_luma: true,
                have_top: true,
                have_left: true,
                edge_filter: true,
                max_width: 8,
                max_height: 8,
                max_pixel: 255,
                smooth_neighbour: false,
            },
        );
        let h = directional(
            Dir::H,
            &a,
            &l,
            c,
            &DirectionalSpec {
                angle_delta: 0,
                block_size: 8,
                is_luma: true,
                have_top: true,
                have_left: true,
                edge_filter: true,
                max_width: 8,
                max_height: 8,
                max_pixel: 255,
                smooth_neighbour: false,
            },
        );
        for r in 0..8 {
            for x in 0..8 {
                assert_eq!(v[r * 8 + x], a[x] as f32);
                assert_eq!(h[r * 8 + x], l[r] as f32);
            }
        }
    }

    #[test]
    fn d45_no_edge_filter_is_exact_diagonal() {
        let (a, l, c) = refs(8, 3);
        let p = directional(
            Dir::D45,
            &a,
            &l,
            c,
            &DirectionalSpec {
                angle_delta: 0,
                block_size: 8,
                is_luma: true,
                have_top: true,
                have_left: false,
                edge_filter: false,
                max_width: 8,
                max_height: 8,
                max_pixel: 255,
                smooth_neighbour: false,
            },
        );
        for r in 0..8 {
            for x in 0..8 {
                assert_eq!(p[r * 8 + x], a[x + r + 1] as f32);
            }
        }
    }

    #[test]
    fn high_bit_depth_prediction_is_not_clipped_to_8_bit() {
        // 10-bit neighbours far above 255: every directional mode must keep them.
        let a = vec![900i32; 16];
        let l = vec![700i32; 16];
        for dir in [Dir::D45, Dir::D135, Dir::D203, Dir::D67] {
            let p = directional(
                dir,
                &a,
                &l,
                800,
                &DirectionalSpec {
                    angle_delta: 0,
                    block_size: 8,
                    is_luma: false,
                    have_top: true,
                    have_left: true,
                    edge_filter: true,
                    max_width: 16,
                    max_height: 16,
                    max_pixel: 1023,
                    smooth_neighbour: false,
                },
            );
            assert!(
                p.iter().all(|&v| (700.0..=900.0).contains(&v)),
                "{dir:?}: {p:?}"
            );
        }
    }
}
