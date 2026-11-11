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

// Lossless frame drivers (colour and monochrome): tile encode, tile-group and OBU assembly.

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossless_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    threads: usize,
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_tilegroup(
        bd,
        w8,
        h8,
        visible_w,
        visible_h,
        src,
        &pool,
        speed,
        updating_cdf,
    );
    assemble_lossless_frame_obus(&plan, &tilegroup, updating_cdf)
}

#[allow(clippy::too_many_arguments)]
fn encode_one_lossless_tile(
    bd: u8,
    full_w: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    r: &(usize, usize, usize, usize),
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(&src[0], full_w, x0, y0, tw, th);
    let p1 = crop_plane(&src[1], full_w, x0, y0, tw, th);
    let p2 = crop_plane(&src[2], full_w, x0, y0, tw, th);
    let tile_visible_w = visible_w.saturating_sub(x0).min(tw);
    let tile_visible_h = visible_h.saturating_sub(y0).min(th);
    crate::tile::encode_tile_lossless(
        tw,
        th,
        tile_visible_w,
        tile_visible_h,
        bd,
        [&p0, &p1, &p2],
        speed,
        updating_cdf,
    )
}

/// Tile count requested for a lossless frame. Lossless has no SB wavefront —
/// independent tiles are its only parallelism — but the grid must be a
/// function of the FRAME alone: the old one-tile-per-worker target made the
/// bitstream depend on `-t`. At least 16 SBs per tile, at most 16 tiles (what
/// `-t 12` already produced); tiling costs ~0.2-1% bytes over a single tile.
fn lossless_tile_target(sb_cols: u32, sb_rows: u32) -> usize {
    ((sb_cols as usize * sb_rows as usize) / 16).clamp(1, 16)
}

#[allow(clippy::too_many_arguments)]
fn encode_lossless_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    src: &[Vec<i16>; 3],
    pool: &Pool,
    speed: Speed,
    updating_cdf: bool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let plan = plan_tiling(sb_cols, sb_rows, lossless_tile_target(sb_cols, sb_rows));
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    // Tile pixel rectangles in raster order (top-to-bottom, left-to-right).
    let mut rects: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            rects.push((x0, y0, x1 - x0, y1 - y0));
        }
    }

    let n = rects.len();
    let nthreads = pool.width().clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile(
            bd,
            w8,
            visible_w,
            visible_h,
            src,
            &rects[i],
            speed,
            updating_cdf,
        )
    });

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a lossless tile group with the matching frame header: a single tile uses
/// a combined `OBU_FRAME` (type 6); multiple tiles use a separate
/// `OBU_FRAME_HEADER` (type 3) + `OBU_TILE_GROUP` (type 4), the layout strict
/// parsers (ffmpeg's cbs_av1) accept.
fn assemble_lossless_frame_obus(plan: &Tiling, tilegroup: &[u8], updating_cdf: bool) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            updating_cdf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            updating_cdf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Crop the single luma plane to a tile rect and encode it as a mono lossless
/// tile. Pure function of its inputs (safe on any thread).
#[allow(clippy::too_many_arguments)]
fn encode_one_lossless_tile_mono(
    bd: u8,
    full_w: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    r: &(usize, usize, usize, usize),
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let (x0, y0, tw, th) = *r;
    let p0 = crop_plane(luma, full_w, x0, y0, tw, th);
    let tile_visible_w = visible_w.saturating_sub(x0).min(tw);
    let tile_visible_h = visible_h.saturating_sub(y0).min(th);
    crate::tile::encode_tile_lossless_mono(
        tw,
        th,
        tile_visible_w,
        tile_visible_h,
        bd,
        &p0,
        speed,
        updating_cdf,
    )
}

/// Monochrome counterpart of [`encode_lossless_tilegroup`]: a single full-res
/// luma plane (`w8*h8`, padded to a multiple of 8) tiled identically to the
/// 4:4:4 path. Byte-identical output for a fixed tiling regardless of thread
/// count.
#[allow(clippy::too_many_arguments)]
fn encode_lossless_mono_tilegroup(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    pool: &Pool,
    speed: Speed,
    updating_cdf: bool,
) -> (Vec<u8>, Tiling) {
    let sb_cols = w8.div_ceil(64) as u32;
    let sb_rows = h8.div_ceil(64) as u32;
    let plan = plan_tiling(sb_cols, sb_rows, lossless_tile_target(sb_cols, sb_rows));
    let col_starts = tile_starts_sb(sb_cols, plan.tcl);
    let row_starts = tile_starts_sb(sb_rows, plan.trl);

    let mut rects: Vec<(usize, usize, usize, usize)> =
        Vec::with_capacity(col_starts.len() * row_starts.len());
    for (ti, &rsb) in row_starts.iter().enumerate() {
        let y0 = rsb as usize * 64;
        let y1 = (row_starts.get(ti + 1).map_or(sb_rows, |&n| n) as usize * 64).min(h8);
        for (tj, &csb) in col_starts.iter().enumerate() {
            let x0 = csb as usize * 64;
            let x1 = (col_starts.get(tj + 1).map_or(sb_cols, |&n| n) as usize * 64).min(w8);
            rects.push((x0, y0, x1 - x0, y1 - y0));
        }
    }

    let n = rects.len();
    let nthreads = pool.width().clamp(1, n.max(1));
    let payloads: Vec<Vec<u8>> = pool.map_indexed(nthreads, n, |i| {
        encode_one_lossless_tile_mono(
            bd,
            w8,
            visible_w,
            visible_h,
            luma,
            &rects[i],
            speed,
            updating_cdf,
        )
    });

    (assemble_tilegroup(payloads), plan)
}

/// Wrap a mono lossless tile group with a `mono_chrome = 1` lossless frame
/// header (single tile ⇒ combined `OBU_FRAME`; multi-tile ⇒ `OBU_FRAME_HEADER` +
/// `OBU_TILE_GROUP`).
fn assemble_lossless_mono_frame_obus(
    plan: &Tiling,
    tilegroup: &[u8],
    updating_cdf: bool,
) -> Vec<u8> {
    if plan.tcl + plan.trl > 0 {
        let fh = crate::obu::frame_header_lossless_mono_multitile_th(
            &plan.cols_incr,
            &plan.rows_incr,
            plan.tcl,
            plan.trl,
            updating_cdf,
        );
        wrap_obu_frame_split(&fh, tilegroup)
    } else {
        let fh = crate::obu::frame_header_lossless_mono_multitile(
            &plan.cols_incr,
            &plan.rows_incr,
            0,
            0,
            updating_cdf,
        );
        wrap_obu_frame(&fh, tilegroup)
    }
}

/// Encode a monochrome lossless frame's OBU portion from a padded `w8*h8` luma
/// plane. Caller prepends temporal delimiter, sequence header, metadata.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_lossless_mono_frame_obus(
    bd: u8,
    w8: usize,
    h8: usize,
    visible_w: usize,
    visible_h: usize,
    luma: &[i16],
    threads: usize,
    speed: Speed,
    updating_cdf: bool,
) -> Vec<u8> {
    let pool = Pool::new(threads);
    let (tilegroup, plan) = encode_lossless_mono_tilegroup(
        bd,
        w8,
        h8,
        visible_w,
        visible_h,
        luma,
        &pool,
        speed,
        updating_cdf,
    );
    assemble_lossless_mono_frame_obus(&plan, &tilegroup, updating_cdf)
}
