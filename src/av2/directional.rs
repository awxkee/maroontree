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
fn clip8(v: i32) -> i32 {
    v.clamp(0, 255)
}
#[inline]
fn iclip(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Intra-edge reference filter strength (non-smooth branch; is_sm = false).
fn filter_strength(wh: i32, angle: i32) -> i32 {
    if wh <= 8 {
        if angle >= 56 {
            return 1;
        }
    } else if wh <= 16 {
        if angle >= 40 {
            return 1;
        }
    } else if wh <= 24 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 16 {
            return 2;
        }
        if angle >= 8 {
            return 1;
        }
    } else if wh <= 32 {
        if angle >= 32 {
            return 3;
        }
        if angle >= 4 {
            return 2;
        }
        return 1;
    } else {
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
    let mut i = 0i32;
    while i < (sz as i32).min(lim_from) {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
    while i < lim_to.min(sz as i32) {
        let mut s = 0;
        for j in 0..5 {
            s += inp[iclip(i - 2 + j, from, to - 1) as usize] * K[strength - 1][j as usize];
        }
        out[i as usize] = (s + 8) >> 4;
        i += 1;
    }
    while i < sz as i32 {
        out[i as usize] = inp[iclip(i, from, to - 1) as usize];
        i += 1;
    }
}

/// Build the decoder-style edge buffer (`tl`,`o`) from encoder refs.
/// `tl[o] = corner`, `tl[o+1+i] = above[i]`, `tl[o-1-i] = left[i]`.
fn edge_buffer(bs: usize, above: &[i32], left: &[i32], corner: i32) -> (Vec<i32>, usize) {
    let o = 2 * bs;
    let mut tl = vec![0i32; 4 * bs + 4];
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
    max_width: i32,
) {
    let (w, h) = (bs, bs);
    let dx = DR[angle as usize];
    let max_base_x = (w + h) as i32 - 1;
    let mut filt = vec![0i32; 141];
    let top_off = 2usize;
    let sz = 1 + w + h;
    let str = if edge {
        filter_strength((w + h) as i32, 90 - angle)
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

    let mut ypos = dx;
    for y in 0..h {
        let mut base = ypos >> 6;
        let fill = filt[top_off + max_base_x as usize];
        if base > max_base_x {
            for yy in y..h {
                for x in 0..w {
                    dst[yy * w + x] = fill;
                }
            }
            break;
        }
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &F4[shift];
        for x in 0..w {
            if base > max_base_x {
                for xx in x..w {
                    dst[y * w + xx] = fill;
                }
                break;
            }
            let bi = (top_off as i32 + base) as usize;
            dst[y * w + x] = if is_luma {
                let v = f[0] * filt[bi - 1]
                    + f[1] * filt[bi]
                    + f[2] * filt[bi + 1]
                    + f[3] * filt[bi + 2];
                clip8((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * filt[bi] + shift as i32 * filt[bi + 1];
                clip8((v + 16) >> 5)
            };
            base += 1;
        }
        ypos += dx;
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
    max_height: i32,
) {
    let (w, h) = (bs, bs);
    let dy = DR[(270 - angle) as usize];
    let max_base_y = (w + h) as i32 - 1;
    let mut filt = vec![0i32; 141];
    let left_off = 1 + w + h;
    let sz = 1 + w + h;
    let str = if edge {
        filter_strength((w + h) as i32, angle - 180)
    } else {
        0
    };
    if str > 0 {
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

    let mut ypos = dy;
    for x in 0..w {
        let shift = ((ypos & 0x3F) >> 1) as usize;
        let f = &F4[shift];
        let mut base = ypos >> 6;
        let fill = filt[left_off - max_base_y as usize];
        for y in 0..h {
            if base <= max_base_y {
                let bi = (left_off as i32 - base) as usize;
                dst[y * w + x] = if is_luma {
                    let v = f[0] * filt[bi + 1]
                        + f[1] * filt[bi]
                        + f[2] * filt[bi - 1]
                        + f[3] * filt[bi - 2];
                    clip8((v + 64) >> 7)
                } else {
                    let v = (32 - shift as i32) * filt[bi] + shift as i32 * filt[bi - 1];
                    clip8((v + 16) >> 5)
                };
                base += 1;
            } else {
                for yy in y..h {
                    dst[yy * w + x] = fill;
                }
                break;
            }
        }
        ypos += dy;
    }
}

/// z2: both-edge directional prediction (angles 90..180), port of `ipred_z2`.
#[allow(clippy::too_many_arguments)]
fn z2(
    dst: &mut [i32],
    bs: usize,
    tl: &[i32],
    o: usize,
    angle: i32,
    is_luma: bool,
    edge: bool,
    max_width: i32,
    max_height: i32,
) {
    let (w, h) = (bs, bs);
    let dy = DR[(angle - 90) as usize];
    let dx = DR[(180 - angle) as usize];

    // top edge buffer
    let mut filt = vec![0i32; 72];
    let top_off = 0usize;
    let sz_t = 1 + w;
    let str_t = if edge {
        filter_strength((w + h) as i32, angle - 90)
    } else {
        0
    };
    if str_t > 0 {
        filter_edge(
            &mut filt[1..],
            sz_t,
            1,
            sz_t as i32 + max_width - w as i32,
            &tl[o..],
            0,
            sz_t as i32,
            str_t as usize,
        );
    } else {
        filt[1..1 + sz_t].copy_from_slice(&tl[o..o + sz_t]);
    }
    filt[0] = filt[1];
    filt[sz_t + 1] = filt[sz_t];

    // left edge buffer
    let mut filt2 = vec![0i32; 72];
    let left_off = h + 2;
    let sz_l = 1 + h;
    let str_l = if edge {
        filter_strength((w + h) as i32, 180 - angle)
    } else {
        0
    };
    if str_l > 0 {
        filter_edge(
            &mut filt2[1..],
            sz_l,
            h as i32 - max_height,
            sz_l as i32 - 1,
            &tl[o - h..],
            0,
            sz_l as i32,
            str_l as usize,
        );
    } else {
        filt2[1..1 + sz_l].copy_from_slice(&tl[o - h..o + 1]);
    }
    filt2[1 + sz_l] = filt2[sz_l];
    filt2[0] = filt2[1];

    for y in 0..h {
        let mut xpos = -((y as i32 + 1) * dx);
        let mut x = 0usize;
        // left-reference run
        while x < w && xpos < -64 {
            let ypos_l = ((y as i32) << 6) - ((x as i32 + 1) * dy);
            let base_y = ypos_l >> 6;
            let shift = ((ypos_l & 0x3F) >> 1) as usize;
            let bi = left_off as i32 - base_y; // keep signed
            let g = |k: i32| filt2[(bi + k) as usize];
            dst[y * w + x] = if is_luma {
                let f = &F4[shift];
                let v = f[0] * g(-1) + f[1] * g(-2) + f[2] * g(-3) + f[3] * g(-4);
                clip8((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * g(-2) + shift as i32 * g(-3);
                clip8((v + 16) >> 5)
            };
            x += 1;
            xpos += 64;
        }
        // top-reference run
        while x < w {
            let base_x = xpos >> 6;
            let shift = ((xpos & 0x3F) >> 1) as usize;
            let ti = top_off as i32 + base_x; // keep signed: base_x may be -1
            let g = |k: i32| filt[(ti + k) as usize];
            dst[y * w + x] = if is_luma {
                let f = &F4[shift];
                let v = f[0] * g(1) + f[1] * g(2) + f[2] * g(3) + f[3] * g(4);
                clip8((v + 64) >> 7)
            } else {
                let v = (32 - shift as i32) * g(2) + shift as i32 * g(3);
                clip8((v + 16) >> 5)
            };
            x += 1;
            xpos += 64;
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

/// Directional predictor for a square `bs`x`bs` block, conformant to the
/// decoder. `is_luma` selects 4-tap vs 2-tap interpolation; `have_top`/
/// `have_left`/`edge_filter` gate the normative intra-edge reference filter
/// (it applies only when both neighbors exist, mrl==0 and the seq flag is on).
/// `max_width`/`max_height` are the available real-sample bounds for the edge
/// filter (pass `bs` for a fully-available interior block).
#[allow(clippy::too_many_arguments)]
pub(crate) fn directional(
    dir: Dir,
    angle_delta: i32,
    bs: usize,
    above: &[i32],
    left: &[i32],
    corner: i32,
    is_luma: bool,
    have_top: bool,
    have_left: bool,
    edge_filter: bool,
    max_width: i32,
    max_height: i32,
) -> Vec<f32> {
    let mut out = vec![0i32; bs * bs];
    // The decoder gates the edge filter on have_top && have_left && mrl0 &&
    // seq_edge_filter && (tw4+th4>=6); for bs>=4 the size test always holds.
    let edge = edge_filter && have_top && have_left;
    // Actual angle = nominal + 3*delta. Square blocks only (32x32 TUs), so no
    // wide-angle remap. Exactly 90/180 are pure copies (VertPred/HorPred); any
    // other angle (incl. V/H with a non-zero delta) takes the z-path.
    let ang = dir.angle() + angle_delta * 3;
    // Decoder fallback (prepare_intra_edges): exactly 90/180 are pure copies,
    // and a z1 angle (<90) with no top edge falls back to VertPred while a z3
    // angle (>180) with no left edge falls back to HorPred. z2 (90<ang<180)
    // always runs (it fills the missing side). Without this remap the encoder
    // runs z1/z3 on filled refs where the decoder copies, diverging at every
    // block that touches a tile top/left edge.
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
            z1(&mut out, bs, &tl, o, ang, is_luma, edge, max_width);
        } else if ang < 180 {
            z2(
                &mut out, bs, &tl, o, ang, is_luma, edge, max_width, max_height,
            );
        } else {
            z3(&mut out, bs, &tl, o, ang, is_luma, edge, max_height);
        }
    }
    out.iter().map(|&v| v as f32).collect()
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
        let v = directional(Dir::V, 0, 8, &a, &l, c, true, true, true, true, 8, 8);
        let h = directional(Dir::H, 0, 8, &a, &l, c, true, true, true, true, 8, 8);
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
        let p = directional(Dir::D45, 0, 8, &a, &l, c, true, true, false, false, 8, 8);
        for r in 0..8 {
            for x in 0..8 {
                assert_eq!(p[r * 8 + x], a[x + r + 1] as f32);
            }
        }
    }
}
