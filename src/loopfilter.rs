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

/// Per-level limit LUT (dav1d `dav1d_calc_eih`, sharpness 0).
/// Returns (E = blimit, I = limit, H = thresh) for a filter level.
#[inline]
fn limits(level: i32) -> (i32, i32, i32) {
    let i = level.max(1); // sharpness 0 => limit = max(level, 1)
    let e = 2 * (level + 2) + i;
    let h = level >> 4;
    (e, i, h)
}

#[inline]
fn iclip(v: i32, lo: i32, hi: i32) -> i32 {
    v.clamp(lo, hi)
}

/// Core sample filter, a direct port of dav1d's `loop_filter`. Filters the four
/// samples along the edge starting at `dst[base]`, stepping `stride_a` between
/// the four lines and `stride_b` across the edge (perpendicular). `wd` selects
/// the filter width: 4, 6 (chroma wide), 8, or 16.
#[allow(clippy::too_many_arguments)]
fn loop_filter(
    dst: &mut [i32],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    let bitdepth_min_8 = bd as i32 - 8;
    let f_flat = 1 << bitdepth_min_8;
    let e = e << bitdepth_min_8;
    let i_lim = i_lim << bitdepth_min_8;
    let h_thresh = h_thresh << bitdepth_min_8;
    let px_max = (1 << bd) - 1;
    let clip_lo = -128 * (1 << bitdepth_min_8);
    let clip_hi = 128 * (1 << bitdepth_min_8) - 1;

    let at = |p: usize, k: isize| -> usize { (p as isize + k) as usize };

    for line in 0..4 {
        let o = (base as isize + line * stride_a) as usize;
        let g = |k: isize| -> i32 { dst[at(o, k * stride_b)] };

        let p1 = g(-2);
        let p0 = g(-1);
        let q0 = g(0);
        let q1 = g(1);

        let mut fm = (p1 - p0).abs() <= i_lim
            && (q1 - q0).abs() <= i_lim
            && (p0 - q0).abs() * 2 + ((p1 - q1).abs() >> 1) <= e;

        let (mut p2, mut q2) = (0, 0);
        let (mut p3, mut q3) = (0, 0);
        if wd > 4 {
            p2 = g(-3);
            q2 = g(2);
            fm &= (p2 - p1).abs() <= i_lim && (q2 - q1).abs() <= i_lim;
            if wd > 6 {
                p3 = g(-4);
                q3 = g(3);
                fm &= (p3 - p2).abs() <= i_lim && (q3 - q2).abs() <= i_lim;
            }
        }
        if !fm {
            continue;
        }

        let (mut p6, mut p5, mut p4) = (0, 0, 0);
        let (mut q4, mut q5, mut q6) = (0, 0, 0);
        let mut flat8out = false;
        if wd >= 16 {
            p6 = g(-7);
            p5 = g(-6);
            p4 = g(-5);
            q4 = g(4);
            q5 = g(5);
            q6 = g(6);
            flat8out = (p6 - p0).abs() <= f_flat
                && (p5 - p0).abs() <= f_flat
                && (p4 - p0).abs() <= f_flat
                && (q4 - q0).abs() <= f_flat
                && (q5 - q0).abs() <= f_flat
                && (q6 - q0).abs() <= f_flat;
        }
        let mut flat8in = false;
        if wd >= 6 {
            flat8in = (p2 - p0).abs() <= f_flat
                && (p1 - p0).abs() <= f_flat
                && (q1 - q0).abs() <= f_flat
                && (q2 - q0).abs() <= f_flat;
        }
        if wd >= 8 {
            flat8in &= (p3 - p0).abs() <= f_flat && (q3 - q0).abs() <= f_flat;
        }

        let put = |dst: &mut [i32], k: isize, v: i32| {
            dst[at(o, k * stride_b)] = v;
        };

        if wd >= 16 && flat8out && flat8in {
            put(
                dst,
                -6,
                (p6 * 7 + p5 * 2 + p4 * 2 + p3 + p2 + p1 + p0 + q0 + 8) >> 4,
            );
            put(
                dst,
                -5,
                (p6 * 5 + p5 * 2 + p4 * 2 + p3 * 2 + p2 + p1 + p0 + q0 + q1 + 8) >> 4,
            );
            put(
                dst,
                -4,
                (p6 * 4 + p5 + p4 * 2 + p3 * 2 + p2 * 2 + p1 + p0 + q0 + q1 + q2 + 8) >> 4,
            );
            put(
                dst,
                -3,
                (p6 * 3 + p5 + p4 + p3 * 2 + p2 * 2 + p1 * 2 + p0 + q0 + q1 + q2 + q3 + 8) >> 4,
            );
            put(
                dst,
                -2,
                (p6 * 2 + p5 + p4 + p3 + p2 * 2 + p1 * 2 + p0 * 2 + q0 + q1 + q2 + q3 + q4 + 8)
                    >> 4,
            );
            put(
                dst,
                -1,
                (p6 + p5 + p4 + p3 + p2 + p1 * 2 + p0 * 2 + q0 * 2 + q1 + q2 + q3 + q4 + q5 + 8)
                    >> 4,
            );
            put(
                dst,
                0,
                (p5 + p4 + p3 + p2 + p1 + p0 * 2 + q0 * 2 + q1 * 2 + q2 + q3 + q4 + q5 + q6 + 8)
                    >> 4,
            );
            put(
                dst,
                1,
                (p4 + p3 + p2 + p1 + p0 + q0 * 2 + q1 * 2 + q2 * 2 + q3 + q4 + q5 + q6 + q6 + 8)
                    >> 4,
            );
            put(
                dst,
                2,
                (p3 + p2 + p1 + p0 + q0 + q1 * 2 + q2 * 2 + q3 * 2 + q4 + q5 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            put(
                dst,
                3,
                (p2 + p1 + p0 + q0 + q1 + q2 * 2 + q3 * 2 + q4 * 2 + q5 + q6 + q6 + q6 + q6 + 8)
                    >> 4,
            );
            put(
                dst,
                4,
                (p1 + p0 + q0 + q1 + q2 + q3 * 2 + q4 * 2 + q5 * 2 + q6 * 5 + 8) >> 4,
            );
            put(
                dst,
                5,
                (p0 + q0 + q1 + q2 + q3 + q4 * 2 + q5 * 2 + q6 * 7 + 8) >> 4,
            );
        } else if wd >= 8 && flat8in {
            put(dst, -3, (p3 * 3 + 2 * p2 + p1 + p0 + q0 + 4) >> 3);
            put(dst, -2, (p3 * 2 + p2 + 2 * p1 + p0 + q0 + q1 + 4) >> 3);
            put(dst, -1, (p3 + p2 + p1 + 2 * p0 + q0 + q1 + q2 + 4) >> 3);
            put(dst, 0, (p2 + p1 + p0 + 2 * q0 + q1 + q2 + q3 + 4) >> 3);
            put(dst, 1, (p1 + p0 + q0 + 2 * q1 + q2 + q3 + q3 + 4) >> 3);
            put(dst, 2, (p0 + q0 + q1 + 2 * q2 + q3 + q3 + q3 + 4) >> 3);
        } else if wd == 6 && flat8in {
            put(dst, -2, (p2 * 3 + 2 * p1 + 2 * p0 + q0 + 4) >> 3);
            put(dst, -1, (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) >> 3);
            put(dst, 0, (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) >> 3);
            put(dst, 1, (p0 + 2 * q0 + 2 * q1 + 2 * q2 + q2 + 4) >> 3);
        } else {
            let hev = (p1 - p0).abs() > h_thresh || (q1 - q0).abs() > h_thresh;
            if hev {
                let mut fv = iclip(p1 - q1, clip_lo, clip_hi);
                fv = iclip(3 * (q0 - p0) + fv, clip_lo, clip_hi);
                let f1 = (fv + 4).min(clip_hi) >> 3;
                let f2 = (fv + 3).min(clip_hi) >> 3;
                put(dst, -1, iclip(p0 + f2, 0, px_max));
                put(dst, 0, iclip(q0 - f1, 0, px_max));
            } else {
                let fv = iclip(3 * (q0 - p0), clip_lo, clip_hi);
                let f1 = (fv + 4).min(clip_hi) >> 3;
                let f2 = (fv + 3).min(clip_hi) >> 3;
                put(dst, -1, iclip(p0 + f2, 0, px_max));
                put(dst, 0, iclip(q0 - f1, 0, px_max));
                let f = (f1 + 1) >> 1;
                put(dst, -2, iclip(p1 + f, 0, px_max));
                put(dst, 1, iclip(q1 - f, 0, px_max));
            }
        }
    }
}

/// Luma filter-width class for a perpendicular tx dimension in 4-sample units:
/// 1 (8px)->wd8(idx1), 2/4/8 (>=16px)->wd16(idx2 capped). dav1d twl4c=min(2,lw).
#[inline]
fn luma_cls(dim4: u8) -> u8 {
    // dim4: 1->4px,2->8px,4->16px,8->32px ; lw=log2(dim4); twl4c=min(2,lw)
    match dim4 {
        1 => 0, // 4px  -> wd4
        2 => 1, // 8px  -> wd8
        _ => 2, // >=16px -> wd16
    }
}

#[inline]
fn chroma_cls(dim4: u8) -> u8 {
    // chroma supports wd4 (cls0) and wd6 (cls>=1)
    if dim4 >= 2 { 1 } else { 0 }
}

/// Filter one plane in place. `bw4`/`bh4` give the block width/height (in 4-sample
/// units) covering each 4x4 unit of THIS plane. `vedge4`/`hedge4`, when present,
/// explicitly mark block starts; this is required for asymmetric partitions
/// whose origins cannot be inferred from global size alignment. Empty edge maps
/// retain the alignment-derived path used by chroma. Frame edges (col 0 / row 0)
/// are not filtered. `is_luma` selects width tables (luma 4/8/16 vs chroma 4/6).
/// Processing order matches dav1d: per 64-px superblock row, all vertical edges
/// then all horizontal edges.
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_plane(
    px: &mut [i32],
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
    bw4: &[u8],
    bh4: &[u8],
    vedge4: &[bool],
    hedge4: &[bool],
    nc4: usize, // number of 4x4 cols in this plane's grid (== ceil(w/4))
    level: i32,
    is_luma: bool,
    sb_rows4: usize, // superblock height in 4-units for this plane (16 luma, 8 for 420 chroma...)
    bd: u8,
) {
    if level <= 0 {
        return;
    }
    let (e, i_lim, h_thresh) = limits(level);
    // Edge coverage is clipped to the VISIBLE frame (dav1d `f->w4`/`f->h4`):
    // the mult-8 coded padding is reconstructed but never deblocked, so edges
    // at or inside the padding stay unfiltered on both sides.
    let w4 = vis_w.div_ceil(4);
    let h4 = vis_h.div_ceil(4);
    let cls = |d: u8| if is_luma { luma_cls(d) } else { chroma_cls(d) };

    // Process per superblock row (top to bottom): vertical edges, then horizontal.
    let mut sb_top = 0usize;
    while sb_top < h4 {
        let sb_bot = (sb_top + sb_rows4).min(h4);

        // --- vertical edges (filter horizontally across them) ---
        for r4 in sb_top..sb_bot {
            for c4 in 1..w4 {
                let idx = r4 * nc4 + c4;
                let cur = bw4[idx];
                let left = bw4[idx - 1];
                // a vertical edge exists at c4 iff the current block starts here
                if if vedge4.is_empty() {
                    c4 % (cur as usize) != 0
                } else {
                    !vedge4[idx]
                } {
                    continue;
                }
                let wcls = cls(cur).min(cls(left));
                let wd = if is_luma {
                    4 << wcls
                } else {
                    4 + 2 * wcls as i32
                };
                // top of this 4x4 row segment
                let y0 = r4 * 4;
                let x0 = c4 * 4;
                if y0 >= h || x0 >= w {
                    continue;
                }
                let base = y0 * w + x0;
                loop_filter(px, base, e, i_lim, h_thresh, w as isize, 1, wd, bd);
            }
        }

        // --- horizontal edges (filter vertically across them) ---
        for r4 in sb_top..sb_bot {
            if r4 == 0 {
                continue; // frame top edge not filtered
            }
            for c4 in 0..w4 {
                let idx = r4 * nc4 + c4;
                let cur = bh4[idx];
                let top = bh4[(r4 - 1) * nc4 + c4];
                if if hedge4.is_empty() {
                    r4 % (cur as usize) != 0
                } else {
                    !hedge4[idx]
                } {
                    continue;
                }
                let wcls = cls(cur).min(cls(top));
                let wd = if is_luma {
                    4 << wcls
                } else {
                    4 + 2 * wcls as i32
                };
                let y0 = r4 * 4;
                let x0 = c4 * 4;
                if y0 >= h || x0 >= w {
                    continue;
                }
                let base = y0 * w + x0;
                loop_filter(px, base, e, i_lim, h_thresh, 1, w as isize, wd, bd);
            }
        }

        sb_top = sb_bot;
    }
}

#[cfg(test)]
mod tests {
    use super::filter_plane;

    #[test]
    fn explicit_vertical_edge_handles_asymmetric_block_origin() {
        let (w, h) = (32usize, 4usize);
        let mut px = vec![0i32; w * h];
        for row in px.chunks_mut(w) {
            row[..12].fill(100);
            row[12..].fill(110);
        }
        let bw4 = vec![2u8; w / 4];
        let bh4 = vec![1u8; w / 4];
        let mut vedge4 = vec![false; w / 4];
        vedge4[3] = true; // x=12 is not globally aligned to an 8px-wide block.

        let before = px.clone();
        filter_plane(
            &mut px,
            w,
            h,
            w,
            h,
            &bw4,
            &bh4,
            &vedge4,
            &[],
            w / 4,
            32,
            true,
            16,
            8,
        );

        assert_ne!(px, before, "the explicitly recorded edge must be filtered");
    }
}
