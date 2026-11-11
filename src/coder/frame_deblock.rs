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

// Frame-level deblocking: edge geometry, filtering and the level search.

#[allow(clippy::too_many_arguments)]
fn cdblk_variant() -> u32 {
    1
}

/// Frame geometry the deblocking filter needs (luma transform-edge maps and
/// prediction-block maps the chroma edges derive from).
struct DeblockGeom<'a> {
    w8: usize,
    h8: usize,
    cw8: usize,
    ch8: usize,
    disp_w: usize,
    disp_h: usize,
    blk4: &'a [u8],    // luma block width map (vertical edges)
    blk4h: &'a [u8],   // luma block height map (horizontal edges)
    blk4v: &'a [bool], // luma block starts at this 4x4 column
    blk4t: &'a [bool], // luma block starts at this 4x4 row
    pblk4: &'a [u8],   // luma PREDICTION-block width map (chroma edges derive from this)
    pblk4h: &'a [u8],  // luma PREDICTION-block height map
    pblk4v: &'a [bool],
    pblk4t: &'a [bool],
    nc4: usize, // luma 4-col count == w8/4
    sub_x: usize,
    sub_y: usize,
    mono: bool,
    sharp: i32,
    bd: u8,
}

/// Chroma deblock geometry (4-unit block width/height maps + block-start
/// flags in the chroma grid) derived from the luma PREDICTION-block map.
struct ChromaDeblockGeom {
    bw4: Vec<u8>,
    bh4: Vec<u8>,
    bv4: Vec<bool>,
    bt4: Vec<bool>,
    nc4: usize,
    vis_w: usize,
    vis_h: usize,
    sb_rows4: usize,
}

impl ChromaDeblockGeom {
    fn vedge(&self) -> &[bool] {
        if cdblk_variant() & 2 != 0 {
            &self.bv4
        } else {
            &[]
        }
    }
    fn hedge(&self) -> &[bool] {
        if cdblk_variant() & 2 != 0 {
            &self.bt4
        } else {
            &[]
        }
    }
}

fn chroma_deblock_geom(g: &DeblockGeom<'_>) -> ChromaDeblockGeom {
    let ss_hor = g.sub_x;
    let ss_ver = g.sub_y;
    let cnc4 = g.cw8 / 4;
    let cnr4 = g.ch8 / 4;
    // Chroma transform geometry comes from the PREDICTION-block map, not the
    // luma transform map: chroma carries no tx_depth here, so one chroma
    // transform spans the whole chroma block even where luma split its own
    // transform. Deriving from `blk4` (transform-granular) invented interior
    // chroma edges the decoder never filters. The block-start flags are carried
    // across too — the alignment fallback in `filter_plane` cannot place edges
    // for blocks whose origin is not a multiple of their size (asymmetric and
    // rectangular partitions).
    let mut cbw4 = vec![0u8; cnc4 * cnr4];
    let mut cbh4 = vec![0u8; cnc4 * cnr4];
    let mut cbv4 = vec![false; cnc4 * cnr4];
    let mut cbt4 = vec![false; cnc4 * cnr4];
    for cr in 0..cnr4 {
        for cc in 0..cnc4 {
            let lr = cr << ss_ver;
            let lc = cc << ss_hor;
            let li = lr * g.nc4 + lc;
            let ci = cr * cnc4 + cc;
            // Variant matrix: bit0 = use PREDICTION geometry for the
            // size map (else the luma TRANSFORM map, as before); bit1 = pass
            // explicit block-start flags (else the alignment fallback).
            let v = cdblk_variant();
            let (sw, sh) = if v & 1 != 0 {
                (g.pblk4[li], g.pblk4h[li])
            } else {
                (g.blk4[li], g.blk4h[li])
            };
            // AV1 caps a chroma transform at 32x32 (`av1_get_max_uv_txsize`).
            // In 4:2:2 a 64x64 luma block yields a 32x64 chroma block, which is
            // coded as stacked 32x32 transforms — so the deblock geometry must
            // clamp to 8 4-units or it misses the interior chroma edge.
            const MAX_UV_TX4: u8 = 8; // 32 px
            cbw4[ci] = (sw >> ss_hor).max(1).min(MAX_UV_TX4);
            cbh4[ci] = (sh >> ss_ver).max(1).min(MAX_UV_TX4);
            if v & 2 != 0 {
                cbv4[ci] = if v & 1 != 0 {
                    g.pblk4v[li]
                } else {
                    g.blk4v[li]
                };
                cbt4[ci] = if v & 1 != 0 {
                    g.pblk4t[li]
                } else {
                    g.blk4t[li]
                };
            }
        }
    }
    ChromaDeblockGeom {
        bw4: cbw4,
        bh4: cbh4,
        bv4: cbv4,
        bt4: cbt4,
        nc4: cnc4,
        vis_w: g.disp_w.div_ceil(1 << ss_hor),
        vis_h: g.disp_h.div_ceil(1 << ss_ver),
        sb_rows4: 16 >> ss_ver,
    }
}

fn deblock_luma_plane(
    loopfilter: &crate::loopfilter::LoopFilterDispatch,
    pool: &Pool,
    px: &mut [u16],
    g: &DeblockGeom<'_>,
    level: i32,
) {
    if level <= 0 {
        return;
    }
    crate::loopfilter::filter_plane_parallel(
        loopfilter, px, g.w8, g.h8, g.disp_w, g.disp_h, g.blk4, g.blk4h, g.blk4v, g.blk4t, g.nc4,
        level, g.sharp, true, 16, // 64px superblock -> 16 4-unit rows
        g.bd, pool,
    );
}

fn deblock_chroma_plane(
    loopfilter: &crate::loopfilter::LoopFilterDispatch,
    pool: &Pool,
    px: &mut [u16],
    g: &DeblockGeom<'_>,
    cg: &ChromaDeblockGeom,
    level: i32,
) {
    if level <= 0 {
        return;
    }
    crate::loopfilter::filter_plane_parallel(
        loopfilter,
        px,
        g.cw8,
        g.ch8,
        cg.vis_w,
        cg.vis_h,
        &cg.bw4,
        &cg.bh4,
        cg.vedge(),
        cg.hedge(),
        cg.nc4,
        level,
        g.sharp,
        false,
        cg.sb_rows4,
        g.bd,
        pool,
    );
}

fn frame_deblock(
    loopfilter: &crate::loopfilter::LoopFilterDispatch,
    pool: &Pool,
    recon: &mut [Vec<u16>; 3],
    g: &DeblockGeom<'_>,
    level_y: i32,
    level_u: i32,
    level_v: i32,
) {
    deblock_luma_plane(loopfilter, pool, &mut recon[0], g, level_y);
    // AV1 only signals chroma levels when luma is filtered (see the frame
    // header): with luma at 0 the chroma levels are implicitly 0.
    if g.mono || level_y <= 0 || (level_u <= 0 && level_v <= 0) {
        return;
    }
    let cg = chroma_deblock_geom(g);
    deblock_chroma_plane(loopfilter, pool, &mut recon[1], g, &cg, level_u);
    deblock_chroma_plane(loopfilter, pool, &mut recon[2], g, &cg, level_v);
}

/// Luma deblock-search distortion over the visible window: plain SSE, or the
/// libaom perceptual `cdef_dist_8x8` per full 8x8 block (partial edge blocks
/// fall back to SSE).
#[allow(clippy::too_many_arguments)]
fn plane_dist_vis(
    dst: &[u16],
    src: &[u16],
    stride: usize,
    vis_w: usize,
    vis_h: usize,
    bd: u8,
    metric: u32,
    rd: &crate::rd_sse::RdDispatch,
) -> i64 {
    if metric == 2 {
        // Unification V7: normalized wSSE over the visible window, zero-padded
        // to whole 4x4 tiles at the right / bottom edge.
        let (pw, ph) = (vis_w.next_multiple_of(4), vis_h.next_multiple_of(4));
        let mut err = vec![0i32; pw * ph];
        for y in 0..vis_h {
            for x in 0..vis_w {
                err[y * pw + x] = i32::from(dst[y * stride + x]) - i32::from(src[y * stride + x]);
            }
        }
        return rd.wsse_err(&err, pw, ph);
    }
    if metric != 1 {
        return plane_sse_vis(dst, src, stride, vis_w, vis_h);
    }
    let rows = dst.len() / stride;
    let mut s = 0i64;
    for y in (0..vis_h).step_by(8) {
        for x in (0..vis_w).step_by(8) {
            if x + 8 <= vis_w && y + 8 <= vis_h {
                s += crate::cdef::cdef_dist_8x8(src, dst, stride, rows, x, y, (bd - 8) as u32);
            } else {
                for yy in y..(y + 8).min(vis_h) {
                    for xx in x..(x + 8).min(vis_w) {
                        let d = dst[yy * stride + xx] as i64 - src[yy * stride + xx] as i64;
                        s += d * d;
                    }
                }
            }
        }
    }
    s
}

/// Sum of squared errors over the visible window of a plane.
fn plane_sse_vis(a: &[u16], b: &[u16], stride: usize, vis_w: usize, vis_h: usize) -> i64 {
    let mut s = 0i64;
    for y in 0..vis_h {
        let ra = &a[y * stride..y * stride + vis_w];
        let rb = &b[y * stride..y * stride + vis_w];
        s += ra
            .iter()
            .zip(rb)
            .map(|(&p, &q)| {
                let d = p as i32 - q as i32;
                (d * d) as i64
            })
            .sum::<i64>();
    }
    s
}

/// libaom `search_filter_level`: bisection from `start` with a bias against
/// raising the level, over an error oracle `try_level`.
fn lf_bisect(mut try_level: impl FnMut(i32) -> i64, start: i32, bias_scale: f32) -> i32 {
    const MAX_LEVEL: i32 = 63;
    let mut ss_err = [-1i64; (MAX_LEVEL + 1) as usize];
    let mut mid = start.clamp(0, MAX_LEVEL);
    let mut step = if mid < 16 { 4 } else { mid / 4 };
    let mut best_err = try_level(mid);
    ss_err[mid as usize] = best_err;
    let mut best = mid;
    let mut dir = 0i32;
    while step > 0 {
        let hi = (mid + step).min(MAX_LEVEL);
        let lo = (mid - step).max(0);
        // aom: bias = (best_err >> (15 - mid/8)) * step, halved unless ONLY_4X4.
        let bias = (((best_err >> (15 - mid / 8)) * step as i64) >> 1) as f32 * bias_scale;
        let bias = bias as i64;
        if dir <= 0 && lo != mid {
            if ss_err[lo as usize] < 0 {
                ss_err[lo as usize] = try_level(lo);
            }
            if ss_err[lo as usize] < best_err + bias {
                if ss_err[lo as usize] < best_err {
                    best_err = ss_err[lo as usize];
                }
                best = lo;
            }
        }
        if dir >= 0 && hi != mid {
            if ss_err[hi as usize] < 0 {
                ss_err[hi as usize] = try_level(hi);
            }
            if ss_err[hi as usize] < best_err - bias {
                best_err = ss_err[hi as usize];
                best = hi;
            }
        }
        if best == mid {
            step /= 2;
            dir = 0;
        } else {
            dir = if best < mid { -1 } else { 1 };
            mid = best;
        }
    }
    best
}

/// Frame-level deblock level search (libaom `LPF_PICK_FROM_FULL_IMAGE_NON_DUAL`,
/// used by its allintra speeds <= 5): luma (one level for both directions),
/// then U and V, each by SSE bisection on the stitched reconstruction.
#[allow(clippy::too_many_arguments)]
fn frame_lf_search(
    loopfilter: &crate::loopfilter::LoopFilterDispatch,
    pool: &Pool,
    rd: &crate::rd_sse::RdDispatch,
    recon: &[Vec<u16>; 3],
    src: &[Vec<u16>; 3],
    g: &DeblockGeom<'_>,
    law_y: i32,
    law_uv: i32,
) -> (i32, i32, i32) {
    let t = crate::tuning::get();
    let (start_y, start_uv) = if t.lf_search_start0 {
        (0, 0)
    } else {
        (law_y, law_uv)
    };
    let rd = frame_decision_rd(rd, g.sub_x, g.sub_y, g.mono);
    let mut scratch = recon[0].clone();
    let lvl_y = lf_bisect(
        |lvl| {
            scratch.copy_from_slice(&recon[0]);
            deblock_luma_plane(loopfilter, pool, &mut scratch, g, lvl);
            plane_dist_vis(
                &scratch,
                &src[0],
                g.w8,
                g.disp_w,
                g.disp_h,
                g.bd,
                t.lf_search_metric,
                &rd,
            )
        },
        start_y,
        t.lf_search_bias,
    );
    if g.mono || lvl_y == 0 {
        return (lvl_y, 0, 0);
    }
    let cg = chroma_deblock_geom(g);
    let mut lvls = [0i32; 2];
    for (pl, out) in lvls.iter_mut().enumerate() {
        let plane = &recon[pl + 1];
        let mut scratch = plane.clone();
        *out = lf_bisect(
            |lvl| {
                scratch.copy_from_slice(plane);
                deblock_chroma_plane(loopfilter, pool, &mut scratch, g, &cg, lvl);
                if t.lf_search_metric == 2 {
                    plane_dist_vis(
                        &scratch,
                        &src[pl + 1],
                        g.cw8,
                        cg.vis_w,
                        cg.vis_h,
                        g.bd,
                        2,
                        &rd,
                    )
                } else {
                    plane_sse_vis(&scratch, &src[pl + 1], g.cw8, cg.vis_w, cg.vis_h)
                }
            },
            start_uv,
            t.lf_search_bias,
        );
    }

    (lvl_y, lvls[0], lvls[1])
}
