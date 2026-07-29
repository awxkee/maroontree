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

pub(crate) type LoopFilterFn = fn(&mut [u16], usize, i32, i32, i32, isize, isize, i32, u8);

pub(crate) static WIDE16_WEIGHTS: [[i32; 14]; 12] = [
    [7, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [5, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [4, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [3, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0, 0],
    [2, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2],
    [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 3],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 4],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 5],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 7],
];

pub(crate) static WIDE8_WEIGHTS: [[i32; 8]; 6] = [
    [3, 2, 1, 1, 1, 0, 0, 0],
    [2, 1, 2, 1, 1, 1, 0, 0],
    [1, 1, 1, 2, 1, 1, 1, 0],
    [0, 1, 1, 1, 2, 1, 1, 1],
    [0, 0, 1, 1, 1, 2, 1, 2],
    [0, 0, 0, 1, 1, 1, 2, 3],
];

pub(crate) static WIDE6_WEIGHTS: [[i32; 6]; 4] = [
    [3, 2, 2, 1, 0, 0],
    [1, 2, 2, 2, 1, 0],
    [0, 1, 2, 2, 2, 1],
    [0, 0, 1, 2, 2, 3],
];

/// Core sample filter, a direct port of dav1d's `loop_filter`. Filters the four
/// samples along the edge starting at `dst[base]`, stepping `stride_a` between
/// the four lines and `stride_b` across the edge (perpendicular). `wd` selects
/// the filter width: 4, 6 (chroma wide), 8, or 16.
#[allow(clippy::too_many_arguments)]
pub(crate) fn loop_filter_scalar(
    dst: &mut [u16],
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
        let g = |k: isize| -> i32 { dst[at(o, k * stride_b)] as i32 };

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

        let put = |dst: &mut [u16], k: isize, v: i32| {
            dst[at(o, k * stride_b)] = v as u16;
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

#[derive(Clone, Copy)]
pub(crate) struct LoopFilterDispatch {
    edge: LoopFilterFn,
    batch: Option<LoopFilterFn>,
    batch_lanes: usize,
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn loop_filter_neon_wrap(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    unsafe {
        crate::neon::loop_filter_neon(dst, base, e, i_lim, h_thresh, stride_a, stride_b, wd, bd)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "aarch64", feature = "neon"))]
fn loop_filter_neon_batch_wrap(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    unsafe {
        crate::neon::loop_filter_batch_neon(
            dst, base, e, i_lim, h_thresh, stride_a, stride_b, wd, bd,
        )
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn loop_filter_avx2_wrap(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    unsafe {
        crate::avx::loop_filter_avx2(dst, base, e, i_lim, h_thresh, stride_a, stride_b, wd, bd)
    }
}

#[allow(clippy::too_many_arguments)]
#[cfg(all(target_arch = "x86_64", feature = "avx"))]
fn loop_filter_avx2_batch_wrap(
    dst: &mut [u16],
    base: usize,
    e: i32,
    i_lim: i32,
    h_thresh: i32,
    stride_a: isize,
    stride_b: isize,
    wd: i32,
    bd: u8,
) {
    unsafe {
        crate::avx::loop_filter_batch_avx2(
            dst, base, e, i_lim, h_thresh, stride_a, stride_b, wd, bd,
        )
    }
}

impl LoopFilterDispatch {
    pub(crate) const fn scalar() -> Self {
        Self {
            edge: loop_filter_scalar,
            batch: None,
            batch_lanes: 4,
        }
    }

    pub(crate) fn selected() -> Self {
        #[allow(unused_mut)]
        let mut dispatch = Self::scalar();
        #[cfg(all(target_arch = "aarch64", feature = "neon"))]
        {
            dispatch.edge = loop_filter_neon_wrap;
            dispatch.batch = Some(loop_filter_neon_batch_wrap);
            dispatch.batch_lanes = 8;
        }
        #[cfg(all(target_arch = "x86_64", feature = "avx"))]
        if std::is_x86_feature_detected!("avx2") {
            dispatch.edge = loop_filter_avx2_wrap;
            dispatch.batch = Some(loop_filter_avx2_batch_wrap);
            dispatch.batch_lanes = 16;
        }
        dispatch
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

#[allow(clippy::too_many_arguments)]
#[cfg(test)]
pub(crate) fn filter_plane(
    dispatch: &LoopFilterDispatch,
    px: &mut [u16],
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
    filter_plane_impl(
        dispatch, px, w, h, vis_w, vis_h, bw4, bh4, vedge4, hedge4, nc4, level, is_luma, sb_rows4,
        bd, None,
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_plane_parallel(
    dispatch: &LoopFilterDispatch,
    px: &mut [u16],
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
    bw4: &[u8],
    bh4: &[u8],
    vedge4: &[bool],
    hedge4: &[bool],
    nc4: usize,
    level: i32,
    is_luma: bool,
    sb_rows4: usize,
    bd: u8,
    pool: &crate::par::Pool,
) {
    filter_plane_impl(
        dispatch,
        px,
        w,
        h,
        vis_w,
        vis_h,
        bw4,
        bh4,
        vedge4,
        hedge4,
        nc4,
        level,
        is_luma,
        sb_rows4,
        bd,
        Some(pool),
    );
}

#[allow(clippy::too_many_arguments)]
fn filter_plane_impl(
    dispatch: &LoopFilterDispatch,
    px: &mut [u16],
    w: usize,
    h: usize,
    vis_w: usize,
    vis_h: usize,
    bw4: &[u8],
    bh4: &[u8],
    vedge4: &[bool],
    hedge4: &[bool],
    nc4: usize,
    level: i32,
    is_luma: bool,
    sb_rows4: usize,
    bd: u8,
    pool: Option<&crate::par::Pool>,
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
        let stripe = &mut px[sb_top * 4 * w..sb_bot * 4 * w];
        let batch_lanes = dispatch.batch_lanes;
        let rows: Vec<_> = stripe
            .chunks_mut(batch_lanes * w)
            .enumerate()
            .map(|(row, band)| (sb_top + row * (batch_lanes / 4), band))
            .collect();
        let vertical = |(r4, band): (usize, &mut [u16])| {
            let segments = band.len() / (4 * w);
            for c4 in 1..w4 {
                let edge_width = |row4: usize| {
                    let idx = row4 * nc4 + c4;
                    let cur = bw4[idx];
                    let left = bw4[idx - 1];
                    let present = if vedge4.is_empty() {
                        c4 % cur as usize == 0
                    } else {
                        vedge4[idx]
                    };
                    present.then(|| {
                        let wcls = cls(cur).min(cls(left));
                        if is_luma {
                            4 << wcls
                        } else {
                            4 + 2 * wcls as i32
                        }
                    })
                };
                let x0 = c4 * 4;
                if x0 >= w {
                    continue;
                }
                let first = edge_width(r4);
                if let (Some(batch), Some(wd)) = (dispatch.batch, first)
                    && segments * 4 == batch_lanes
                    && (1..segments).all(|segment| edge_width(r4 + segment) == first)
                {
                    batch(band, x0, e, i_lim, h_thresh, w as isize, 1, wd, bd);
                    continue;
                }
                for segment in 0..segments {
                    let row4 = r4 + segment;
                    if row4 * 4 >= h {
                        continue;
                    }
                    if let Some(wd) = edge_width(row4) {
                        (dispatch.edge)(
                            band,
                            segment * 4 * w + x0,
                            e,
                            i_lim,
                            h_thresh,
                            w as isize,
                            1,
                            wd,
                            bd,
                        );
                    }
                }
            }
        };
        if let Some(pool) = pool
            && (sb_bot - sb_top).saturating_mul(w4.saturating_sub(1)) >= 1024
        {
            pool.for_each(pool.width(), rows, vertical);
        } else {
            rows.into_iter().for_each(vertical);
        }

        // --- horizontal edges (filter vertically across them) ---
        for r4 in sb_top..sb_bot {
            if r4 == 0 {
                continue; // frame top edge not filtered
            }
            let mut c4 = 0usize;
            while c4 < w4 {
                let idx = r4 * nc4 + c4;
                let cur = bh4[idx];
                let top = bh4[(r4 - 1) * nc4 + c4];
                if if hedge4.is_empty() {
                    r4 % (cur as usize) != 0
                } else {
                    !hedge4[idx]
                } {
                    c4 += 1;
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
                    c4 += 1;
                    continue;
                }
                let base = y0 * w + x0;
                let segments = dispatch.batch_lanes / 4;
                if let Some(batch) = dispatch.batch
                    && c4 + segments <= w4
                    && (1..segments).all(|segment| {
                        let idx = r4 * nc4 + c4 + segment;
                        let next = bh4[idx];
                        let next_top = bh4[(r4 - 1) * nc4 + c4 + segment];
                        let present = if hedge4.is_empty() {
                            r4 % next as usize == 0
                        } else {
                            hedge4[idx]
                        };
                        let next_cls = cls(next).min(cls(next_top));
                        let next_wd = if is_luma {
                            4 << next_cls
                        } else {
                            4 + 2 * next_cls as i32
                        };
                        present && next_wd == wd
                    })
                {
                    batch(px, base, e, i_lim, h_thresh, 1, w as isize, wd, bd);
                    c4 += segments;
                    continue;
                }
                (dispatch.edge)(px, base, e, i_lim, h_thresh, 1, w as isize, wd, bd);
                c4 += 1;
            }
        }

        sb_top = sb_bot;
    }
}

#[cfg(test)]
mod tests {
    use super::{
        LoopFilterDispatch, LoopFilterFn, filter_plane, filter_plane_parallel, limits,
        loop_filter_scalar,
    };

    fn set_crossing_line(
        pixels: &mut [u16],
        base: usize,
        stride_a: isize,
        stride_b: isize,
        line: usize,
        samples: [i32; 14],
    ) {
        for (index, value) in samples.into_iter().enumerate() {
            let pos = base as isize + line as isize * stride_a + (index as isize - 7) * stride_b;
            pixels[pos as usize] = value as u16;
        }
    }

    fn mixed_edge_case(bd: u8, vertical: bool) -> (Vec<u16>, usize, isize, isize) {
        const W: usize = 48;
        const H: usize = 48;
        let scale = 1i32 << (bd - 8);
        let mid = 1i32 << (bd - 1);
        let mut pixels = vec![mid as u16; W * H];
        let base = 20 * W + 20;
        let (stride_a, stride_b) = if vertical {
            (W as isize, 1)
        } else {
            (1, W as isize)
        };

        // Flat wide-filter lane.
        set_crossing_line(
            &mut pixels,
            base,
            stride_a,
            stride_b,
            0,
            [
                mid,
                mid + scale,
                mid - scale,
                mid,
                mid + scale,
                mid - scale,
                mid,
                mid + scale,
                mid,
                mid + 2 * scale,
                mid + scale,
                mid,
                mid + 2 * scale,
                mid + scale,
            ],
        );
        // Rejected lane.
        set_crossing_line(
            &mut pixels,
            base,
            stride_a,
            stride_b,
            1,
            std::array::from_fn(|i| if i < 7 { 0 } else { (1 << bd) - 1 }),
        );
        // Filter-mask true, flat false, HEV false.
        set_crossing_line(
            &mut pixels,
            base,
            stride_a,
            stride_b,
            2,
            std::array::from_fn(|i| mid + (i as i32 - 7) * 2 * scale),
        );
        // Filter-mask true, flat false, HEV true.
        let mut hev = [mid; 14];
        hev[5] = mid - 8 * scale;
        hev[7] = mid + 2 * scale;
        hev[8] = mid + 10 * scale;
        set_crossing_line(&mut pixels, base, stride_a, stride_b, 3, hev);

        (pixels, base, stride_a, stride_b)
    }

    fn assert_loop_filter_matches_scalar(name: &str, simd: LoopFilterFn) {
        for bd in [8u8, 10, 12] {
            for vertical in [false, true] {
                for wd in [4, 6, 8, 16] {
                    for level in [1, 16, 32, 63] {
                        let (input, base, stride_a, stride_b) = mixed_edge_case(bd, vertical);
                        let (e, i_lim, h_thresh) = limits(level);
                        let mut scalar = input.clone();
                        let mut vector = input;
                        loop_filter_scalar(
                            &mut scalar,
                            base,
                            e,
                            i_lim,
                            h_thresh,
                            stride_a,
                            stride_b,
                            wd,
                            bd,
                        );
                        simd(
                            &mut vector,
                            base,
                            e,
                            i_lim,
                            h_thresh,
                            stride_a,
                            stride_b,
                            wd,
                            bd,
                        );
                        assert_eq!(
                            scalar, vector,
                            "{name}: bd={bd}, vertical={vertical}, wd={wd}, level={level}"
                        );
                    }
                }
            }
        }
    }

    fn assert_loop_filter_batch_matches_scalar(name: &str, simd: LoopFilterFn, lanes: usize) {
        for bd in [8u8, 10, 12] {
            for vertical in [false, true] {
                for wd in [4, 6, 8, 16] {
                    for level in [1, 16, 32, 63] {
                        let (mut input, base, stride_a, stride_b) = mixed_edge_case(bd, vertical);
                        for line in 4..lanes {
                            for offset in -7isize..=6 {
                                let src = (base as isize
                                    + (line % 4) as isize * stride_a
                                    + offset * stride_b)
                                    as usize;
                                let dst =
                                    (base as isize + line as isize * stride_a + offset * stride_b)
                                        as usize;
                                input[dst] = input[src];
                            }
                        }
                        let (e, i_lim, h_thresh) = limits(level);
                        let mut scalar = input.clone();
                        let mut vector = input;
                        for group in 0..lanes / 4 {
                            loop_filter_scalar(
                                &mut scalar,
                                (base as isize + (group * 4) as isize * stride_a) as usize,
                                e,
                                i_lim,
                                h_thresh,
                                stride_a,
                                stride_b,
                                wd,
                                bd,
                            );
                        }
                        simd(
                            &mut vector,
                            base,
                            e,
                            i_lim,
                            h_thresh,
                            stride_a,
                            stride_b,
                            wd,
                            bd,
                        );
                        assert_eq!(
                            scalar, vector,
                            "{name}: bd={bd}, vertical={vertical}, wd={wd}, level={level}"
                        );
                    }
                }
            }
        }

        // Exercise the unsigned wide-16 accumulator at the top of the 12-bit
        // range; its mathematical maximum including rounding bias is 65528.
        for vertical in [false, true] {
            const W: usize = 48;
            const H: usize = 48;
            let mut input = vec![4095u16; W * H];
            let base = 20 * W + 20;
            let (stride_a, stride_b) = if vertical {
                (W as isize, 1)
            } else {
                (1, W as isize)
            };
            for line in 0..lanes {
                set_crossing_line(
                    &mut input,
                    base,
                    stride_a,
                    stride_b,
                    line,
                    std::array::from_fn(|i| 4095 - ((i + line) & 1) as i32),
                );
            }
            let (e, i_lim, h_thresh) = limits(63);
            let mut scalar = input.clone();
            let mut vector = input;
            for group in 0..lanes / 4 {
                loop_filter_scalar(
                    &mut scalar,
                    (base as isize + (group * 4) as isize * stride_a) as usize,
                    e,
                    i_lim,
                    h_thresh,
                    stride_a,
                    stride_b,
                    16,
                    12,
                );
            }
            simd(
                &mut vector,
                base,
                e,
                i_lim,
                h_thresh,
                stride_a,
                stride_b,
                16,
                12,
            );
            assert_eq!(scalar, vector, "{name}: 12-bit wide16 maximum");
        }
    }

    #[test]
    fn selected_loop_filter_matches_scalar() {
        assert_loop_filter_matches_scalar("selected", LoopFilterDispatch::selected().edge);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_loop_filter_matches_scalar() {
        assert_loop_filter_matches_scalar("neon", super::loop_filter_neon_wrap);
    }

    #[cfg(all(target_arch = "aarch64", feature = "neon"))]
    #[test]
    fn neon_batch_loop_filter_matches_scalar() {
        assert_loop_filter_batch_matches_scalar(
            "neon batch",
            super::loop_filter_neon_batch_wrap,
            8,
        );
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_loop_filter_matches_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            assert_loop_filter_matches_scalar("avx2", super::loop_filter_avx2_wrap);
        }
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx"))]
    #[test]
    fn avx2_batch_loop_filter_matches_scalar() {
        if std::is_x86_feature_detected!("avx2") {
            assert_loop_filter_batch_matches_scalar(
                "avx2 batch",
                super::loop_filter_avx2_batch_wrap,
                16,
            );
        }
    }

    #[test]
    fn explicit_vertical_edge_handles_asymmetric_block_origin() {
        let (w, h) = (32usize, 4usize);
        let mut px = vec![0u16; w * h];
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
            &LoopFilterDispatch::scalar(),
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

    #[test]
    fn parallel_vertical_bands_match_serial_plane_filtering() {
        const W: usize = 264;
        const H: usize = 88;
        const VIS_W: usize = 259;
        const VIS_H: usize = 83;
        let nc4 = W / 4;
        let nr4 = H / 4;
        let dispatch = LoopFilterDispatch::selected();
        let pool = crate::par::Pool::new(4);

        for (bd, dim4, is_luma) in [(8u8, 1u8, true), (10, 4, true), (12, 2, false)] {
            let max = (1u32 << bd) - 1;
            let input: Vec<u16> = (0..W * H)
                .scan(0x243f_6a88u32, |state, _| {
                    *state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    Some((*state & max) as u16)
                })
                .collect();
            let dims = vec![dim4; nc4 * nr4];
            let mut serial = input.clone();
            let mut parallel = input;
            filter_plane(
                &dispatch,
                &mut serial,
                W,
                H,
                VIS_W,
                VIS_H,
                &dims,
                &dims,
                &[],
                &[],
                nc4,
                32,
                is_luma,
                16,
                bd,
            );
            filter_plane_parallel(
                &dispatch,
                &mut parallel,
                W,
                H,
                VIS_W,
                VIS_H,
                &dims,
                &dims,
                &[],
                &[],
                nc4,
                32,
                is_luma,
                16,
                bd,
                &pool,
            );
            assert_eq!(parallel, serial, "bd={bd}, dim4={dim4}, is_luma={is_luma}");
        }
    }

    #[test]
    fn batch_coalescing_with_mixed_geometry_matches_scalar() {
        const W: usize = 264;
        const H: usize = 88;
        let nc4 = W / 4;
        let nr4 = H / 4;
        let input: Vec<u16> = (0..W * H)
            .map(|i| (((i % W) / 8 + (i / W) / 8) & 31) as u16 * 4 + 64)
            .collect();
        let mut bw4 = vec![1u8; nc4 * nr4];
        let mut bh4 = vec![1u8; nc4 * nr4];
        let mut vedge4 = vec![false; nc4 * nr4];
        let mut hedge4 = vec![false; nc4 * nr4];
        for r4 in 0..nr4 {
            for c4 in 0..nc4 {
                let idx = r4 * nc4 + c4;
                bw4[idx] = [1, 2, 4, 8][(r4 + 2 * c4) & 3];
                bh4[idx] = [1, 2, 4, 8][(2 * r4 + c4) & 3];
                vedge4[idx] = c4 >= 2 && c4 + 2 < nc4 && (3 * r4 + c4) % 5 <= 2;
                hedge4[idx] = r4 >= 2 && r4 + 2 < nr4 && (r4 + 3 * c4) % 5 <= 2;
            }
        }

        let mut scalar = input.clone();
        let mut selected = input;
        filter_plane(
            &LoopFilterDispatch::scalar(),
            &mut scalar,
            W,
            H,
            W,
            H,
            &bw4,
            &bh4,
            &vedge4,
            &hedge4,
            nc4,
            32,
            true,
            16,
            8,
        );
        filter_plane(
            &LoopFilterDispatch::selected(),
            &mut selected,
            W,
            H,
            W,
            H,
            &bw4,
            &bh4,
            &vedge4,
            &hedge4,
            nc4,
            32,
            true,
            16,
            8,
        );
        assert_eq!(selected, scalar);
    }
}
