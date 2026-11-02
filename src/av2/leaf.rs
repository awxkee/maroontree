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

use super::*;
use crate::Speed;

#[allow(clippy::too_many_arguments)]
fn fill_scaled_residual(
    dst: &mut [f32],
    src: &[f32],
    src_stride: usize,
    src_y: usize,
    src_x: usize,
    pred: &[f32],
    width: usize,
    height: usize,
    scale: f32,
) {
    let dst_rows = dst.chunks_exact_mut(width);
    let src_rows = rect_rows(src, src_stride, src_y, src_x, width, height);
    let pred_rows = pred.chunks_exact(width);
    for ((dst_row, src_row), pred_row) in dst_rows.zip(src_rows).zip(pred_rows) {
        for ((dst, &src), &pred) in dst_row.iter_mut().zip(src_row).zip(pred_row) {
            *dst = (src - pred) * scale;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatch_intra_pred(
    max_pixel: i32,
    m: usize,
    adelta: i32,
    ab: &[i32],
    lf: &[i32],
    corner: i32,
    have_top: bool,
    have_left: bool,
    max_w: i32,
    max_h: i32,
) -> Vec<f32> {
    let mut out = vec![0f32; 1024];
    dispatch_intra_pred_into(
        max_pixel, m, adelta, ab, lf, corner, have_top, have_left, max_w, max_h, &mut out,
    );
    out
}

/// [`dispatch_intra_pred`] into `out[..1024]`.
#[allow(clippy::too_many_arguments)]
fn dispatch_intra_pred_into(
    max_pixel: i32,
    m: usize,
    adelta: i32,
    ab: &[i32],
    lf: &[i32],
    corner: i32,
    have_top: bool,
    have_left: bool,
    max_w: i32,
    max_h: i32,
    out: &mut [f32],
) {
    use directional::Dir::*;
    let dir = match m {
        1 => return intrapred::smooth_into(32, ab, lf, out),
        2 => return intrapred::smooth_v_into(32, ab, lf, out),
        3 => return intrapred::smooth_h_into(32, ab, lf, out),
        4 => return intrapred::paeth_into(32, ab, lf, corner, out),
        5 => V,
        6 => H,
        7 => D45,
        8 => D135,
        9 => D113,
        10 => D157,
        11 => D203,
        _ => D67,
    };
    directional::directional_into(
        dir,
        ab,
        lf,
        corner,
        &directional::DirectionalSpec {
            angle_delta: adelta,
            block_size: 32,
            is_luma: true,
            have_top,
            have_left,
            edge_filter: true,
            max_width: max_w,
            max_height: max_h,
            max_pixel,
            smooth_neighbour: false,
        },
        out,
    )
}

/// `midx_above` / `midx_left` entry of a leaf coded SMOOTH / SMOOTH_V / SMOOTH_H:
/// no mode-set midx (reads as non-directional), but it selects intra edge-filter
/// type 1 for a directional neighbour (avm `get_filt_type`).
pub(crate) const MIDX_SMOOTH: i32 = -2;

/// Neighbour-array class of a non-directional internal luma mode (0..=4).
pub(crate) fn midx_class(mode: usize) -> i32 {
    if (1..=3).contains(&mode) {
        MIDX_SMOOTH
    } else {
        -1
    }
}

/// A square luma block coded as a single TU (8x8 / 16x16 leaves).
#[derive(Clone, Copy)]
pub(super) struct SquareLumaBlock {
    pub(super) stride: usize,
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) size: usize,
}

/// A luma intra mode for [`predict_luma_square`]: internal mode index (1 = SMOOTH,
/// 2 = SMOOTH_V, 3 = SMOOTH_H, 4 = PAETH, 5..=12 directional as in
/// `dispatch_intra_pred`), angle delta (directional only) and whether the above or
/// left block is a SMOOTH mode (edge-filter type 1).
#[derive(Clone, Copy)]
pub(super) struct SquareLumaMode {
    pub(super) mode: usize,
    pub(super) angle_delta: i32,
    pub(super) smooth_neighbour: bool,
}

impl SquareLumaMode {
    pub(super) fn nondir(mode: usize) -> Self {
        Self {
            mode,
            angle_delta: 0,
            smooth_neighbour: false,
        }
    }
}

/// Map an internal directional mode index (5..=12) to its direction.
fn dir_of(m: usize) -> directional::Dir {
    use directional::Dir::*;
    match m {
        5 => V,
        6 => H,
        7 => D45,
        8 => D135,
        9 => D113,
        10 => D157,
        11 => D203,
        _ => D67,
    }
}

/// Directional SATD screen for the coded leaves and the probe: rank the 8 nominal
/// angles, then try the angle deltas only around the best few, and return the
/// SATD-best `keep` (mode, angle_delta) pairs, best first. `satd` predicts one
/// candidate and scores it against the source.
pub(crate) fn screen_directional(
    keep: usize,
    deltas: bool,
    refine: usize,
    mut satd: impl FnMut(usize, i32) -> u64,
) -> Vec<(usize, i32)> {
    let mut ranked: Vec<(u64, (usize, i32))> = Vec::with_capacity(8 + 6 * refine);
    ranked.extend((5..=12usize).map(|m| (satd(m, 0), (m, 0))));
    ranked.sort_by_key(|&(s, _)| s);
    if deltas {
        let roots: Vec<usize> = ranked.iter().take(refine).map(|&(_, (m, _))| m).collect();
        for m in roots {
            for d in [-1, 1, -2, 2, -3, 3] {
                ranked.push((satd(m, d), (m, d)));
            }
        }
        ranked.sort_by_key(|&(s, _)| s);
    }
    ranked.into_iter().take(keep).map(|(_, md)| md).collect()
}

thread_local! {
    /// The directional picks of the last 8x8 `probe_square_luma` call (its caller
    /// files them into the partition decision).
    static PROBE_DIR8_LAST: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Pack up to three (mode 5..=12, delta -3..=3) picks: count in bits 0-1, then 6
/// bits per pick. 0 = nothing screened.
fn dir8_pack(picks: &[(usize, i32)]) -> u32 {
    let mut v = picks.len().min(3) as u32;
    for (i, &(m, d)) in picks.iter().take(3).enumerate() {
        v |= ((((m - 5) as u32) << 3) | (d + 3) as u32) << (2 + 6 * i);
    }
    v | 1 << 31
}

fn dir8_unpack(v: u32) -> Option<Vec<(usize, i32)>> {
    (v != 0).then(|| {
        (0..(v & 3) as usize)
            .map(|i| {
                let f = (v >> (2 + 6 * i)) & 63;
                (5 + (f >> 3) as usize, (f & 7) as i32 - 3)
            })
            .collect()
    })
}

/// Top-right / bottom-left reference pixels available to a square block: limited
/// by the tile extent and by this superblock's coded-mi mask (AVM coding order).
pub(super) fn square_edge_px(
    b: &SquareLumaBlock,
    coded: &[u8; 256],
    need_tr: bool,
    need_bl: bool,
) -> (usize, usize) {
    let n = b.size;
    let (mi_row, mi_col, mi_n) = ((b.y >> 2) as i64, (b.x >> 2) as i64, (n >> 2) as i64);
    // Tile extent in mi: ALIGN(dim, 8) >> 2, as the decoder's mi_col_end/mi_row_end.
    let mi_col_end = (((b.frame_width + 7) & !7) >> 2) as i64;
    let mi_row_end = (((b.frame_height + 7) & !7) >> 2) as i64;
    let have_top = b.y > 0;
    let have_left = b.x > 0;
    let xr = (mi_col_end - mi_col - mi_n) << 2;
    let yd = (mi_row_end - mi_row - mi_n) << 2;
    let right_available = mi_col + mi_n < mi_col_end;
    let bottom_available = yd > 0 && mi_row + mi_n < mi_row_end;
    let (r_sb, c_sb) = (mi_row & 15, mi_col & 15);
    let coded_at = |r: i64, c: i64| coded[(r * 16 + c) as usize] != 0;
    let tr_px = if !(need_tr && have_top && right_available) || xr.min(n as i64) <= 0 {
        0
    } else {
        let common = xr.min(n as i64);
        let (tr_row, tr_col) = (r_sb - 1, c_sb + mi_n);
        if tr_row < 0 {
            common
        } else if tr_col >= 16 || !coded_at(tr_row, tr_col) {
            0
        } else {
            let run = (0..mi_n)
                .take_while(|&i| tr_col + i < 16 && coded_at(tr_row, tr_col + i))
                .count();
            ((run as i64) << 2).min(common)
        }
    } as usize;
    let bl_px = if !(need_bl && have_left && bottom_available) || yd.min(n as i64) <= 0 {
        0
    } else {
        let common = yd.min(n as i64);
        let (bl_row, bl_col) = (r_sb + mi_n, c_sb - 1);
        if bl_col < 0 {
            (64 - ((r_sb << 2) + n as i64)).min(common).max(0)
        } else if bl_row >= 16 || !coded_at(bl_row, bl_col) {
            0
        } else {
            let run = (0..mi_n)
                .take_while(|&i| bl_row + i < 16 && coded_at(bl_row + i, bl_col))
                .count();
            ((run as i64) << 2).min(common)
        }
    } as usize;
    (tr_px, bl_px)
}

/// Intra prediction (every mode but DC) for a square block whose TU equals the
/// block. Top-right / bottom-left availability follows avm `has_top_right` /
/// `has_bottom_left` (row_off = col_off = 0) against `coded`, the superblock's
/// coded-mi mask (16x16, row-major). DC is not handled here; callers keep their
/// existing DC predictor.
pub(super) fn predict_luma_square(
    recy: &[f32],
    b: &SquareLumaBlock,
    coded: &[u8; 256],
    lm: SquareLumaMode,
    neutral: f32,
    bit_depth: i32,
) -> Vec<f32> {
    let mut out = vec![0f32; b.size * b.size];
    predict_luma_square_into(recy, b, coded, lm, neutral, bit_depth, &mut out);
    out
}

/// [`predict_luma_square`] into `out[..size*size]`.
pub(super) fn predict_luma_square_into(
    recy: &[f32],
    b: &SquareLumaBlock,
    coded: &[u8; 256],
    lm: SquareLumaMode,
    neutral: f32,
    bit_depth: i32,
    out: &mut [f32],
) {
    let mut refs = SquareRefCache::default();
    predict_luma_square_cached(recy, b, coded, lm, neutral, bit_depth, &mut refs, out);
}

/// The reference edges of one square block, per (top-right, bottom-left) need,
/// built on first use: a block's candidate modes share them while its
/// neighbourhood is unchanged.
#[derive(Default)]
pub(super) struct SquareRefCache {
    refs: [Option<SquareRefs>; 4],
}

type SquareRefs = (
    [i32; intrapred::REF_CAP],
    [i32; intrapred::REF_CAP],
    i32,
    usize,
    usize,
);

/// [`predict_luma_square_into`] reusing `refs` (only valid for one block and an
/// unchanged `recy` / `coded`).
#[allow(clippy::too_many_arguments)]
pub(super) fn predict_luma_square_cached(
    recy: &[f32],
    b: &SquareLumaBlock,
    coded: &[u8; 256],
    lm: SquareLumaMode,
    neutral: f32,
    bit_depth: i32,
    refs: &mut SquareRefCache,
    out: &mut [f32],
) {
    let mode = lm.mode;
    debug_assert!((1..=12).contains(&mode));
    let p_angle = if mode >= 5 {
        dir_of(mode).angle() + 3 * lm.angle_delta
    } else {
        0
    };
    let n = b.size;
    let mi_col_end = (((b.frame_width + 7) & !7) >> 2) as i64;
    let mi_row_end = (((b.frame_height + 7) & !7) >> 2) as i64;
    let have_top = b.y > 0;
    let have_left = b.x > 0;
    // SMOOTH needs both, SMOOTH_V bottom-left, SMOOTH_H top-right, PAETH neither;
    // directional by angle (avm: p_angle < 90 / > 180, IBP off).
    let need_tr = matches!(mode, 1 | 3) || (mode >= 5 && p_angle < 90);
    let need_bl = matches!(mode, 1 | 2) || (mode >= 5 && p_angle > 180);
    let (ab, lf, corner, tr_px, bl_px) = refs.refs[need_tr as usize * 2 + need_bl as usize]
        .get_or_insert_with(|| {
            let (tr_px, bl_px) = square_edge_px(b, coded, need_tr, need_bl);
            let avail_above = ((mi_col_end << 2) as usize).saturating_sub(b.x).clamp(1, n);
            let avail_left = ((mi_row_end << 2) as usize).saturating_sub(b.y).clamp(1, n);
            let (ab, lf, corner) = intrapred::build_refs(
                recy,
                &intrapred::IntraRefSpec {
                    stride: b.stride,
                    y: b.y,
                    x: b.x,
                    block_size: n,
                    have_above: have_top,
                    have_left,
                    top_right: tr_px,
                    bottom_left: bl_px,
                    neutral,
                    available_above: avail_above,
                    available_left: avail_left,
                },
            );
            (ab, lf, corner, tr_px, bl_px)
        });
    let (ab, lf): (&[i32], &[i32]) = (&ab[..], &lf[..]);
    let (corner, tr_px, bl_px) = (*corner, *tr_px, *bl_px);
    match mode {
        1 => intrapred::smooth_into(n, ab, lf, out),
        2 => intrapred::smooth_v_into(n, ab, lf, out),
        3 => intrapred::smooth_h_into(n, ab, lf, out),
        4 => intrapred::paeth_into(n, ab, lf, corner, out),
        _ => directional::directional_into(
            dir_of(mode),
            ab,
            lf,
            corner,
            &directional::DirectionalSpec {
                angle_delta: lm.angle_delta,
                block_size: n,
                is_luma: true,
                have_top,
                have_left,
                edge_filter: true,
                max_width: (n + tr_px) as i32,
                max_height: (n + bl_px) as i32,
                max_pixel: (1 << bit_depth) - 1,
                smooth_neighbour: lm.smooth_neighbour,
            },
            out,
        ),
    }
}

/// Source luma plane and its row stride.
#[derive(Clone, Copy)]
pub(super) struct LumaSource<'a> {
    pub(super) plane: &'a [f32],
    pub(super) stride: usize,
}

/// Frame-bounded location of one superblock-sized luma coding region.
#[derive(Clone, Copy)]
pub(super) struct LumaFrameBlock {
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
    pub(super) y: usize,
    pub(super) x: usize,
}

/// MI-grid-bounded location shared by the split luma-leaf encoders.
#[derive(Clone, Copy)]
pub(super) struct LumaGridBlock {
    pub(super) mi_cols: i64,
    pub(super) mi_rows: i64,
    pub(super) y: usize,
    pub(super) x: usize,
}

/// Transform, quantization and speed state shared by luma mode searches.
#[derive(Clone, Copy)]
pub(super) struct LumaQuantSpec<'a> {
    pub(super) basis: &'a Basis,
    pub(super) qstep: i32,
    pub(super) scan: &'a [u16],
    pub(super) neutral: f32,
    pub(super) quant_context: usize,
    pub(super) rdoq_lambda: f32,
    /// Price luma mode/partition bits in pixel SSE (stills). Video keeps the
    /// historical QTX-scaled lambda until its inter paths are re-tuned.
    pub(super) pixel_lambda: bool,
    pub(super) speed: Speed,
    pub(super) bit_depth: i32,
    /// Candidate pruning enabled for this format: [`PRUNE_PROBE`] (the 8/16
    /// probe codes only the SATD-best candidates), [`PRUNE_LARGE`] (likewise the
    /// 32x32 leaf and the whole-64 search).
    pub(super) prune: u8,
}

pub(super) const PRUNE_PROBE: u8 = 1;
pub(super) const PRUNE_LARGE: u8 = 2;

/// Search options specific to a full 64x64 luma superblock.
#[derive(Clone, Copy)]
pub(super) struct LumaSbSearch {
    pub(super) residual_scale: f32,
    pub(super) allow_directional: bool,
}

/// Geometry and mode state for a frame-bounded 32x32 luma prediction.
#[derive(Clone, Copy)]
pub(super) struct LumaPredictSpec {
    pub(super) stride: usize,
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
    pub(super) tx_index: usize,
    pub(super) y: usize,
    pub(super) x: usize,
    pub(super) mode: usize,
    pub(super) angle_delta: i32,
    pub(super) neutral: f32,
}

/// Geometry and mode state for one TX_32X32 within a split luma leaf.
#[derive(Clone, Copy)]
pub(super) struct LumaLeafPredictSpec {
    pub(super) stride: usize,
    pub(super) mi_cols: i64,
    pub(super) mi_rows: i64,
    pub(super) block_y: usize,
    pub(super) block_x: usize,
    pub(super) tx_y: usize,
    pub(super) tx_x: usize,
    pub(super) tx_index: usize,
    pub(super) mode: usize,
    pub(super) neutral: f32,
    pub(super) split_leaf: bool,
}

/// Build the prediction block for luma candidate `m` (0=DC, 1=SMOOTH, 4=PAETH)
/// at TX index `i` (raster within the 64x64 SB) and pixel origin `(y0,x0)`.
pub(super) fn predict_luma(recy: &[f32], spec: &LumaPredictSpec) -> Vec<f32> {
    let mut out = vec![0f32; 1024];
    predict_luma_into(recy, spec, &mut out);
    out
}

/// [`predict_luma`] into `out[..1024]`.
pub(super) fn predict_luma_into(recy: &[f32], spec: &LumaPredictSpec, out: &mut [f32]) {
    let &LumaPredictSpec {
        stride: pw,
        frame_width: width,
        frame_height: height,
        tx_index: i,
        y: y0,
        x: x0,
        mode: m,
        angle_delta: adelta,
        neutral,
    } = spec;
    if m == 0 {
        out[..1024].fill(crate::av2::helpers::dc_pred_bounded(
            recy, pw, y0, x0, 32, neutral, width, height,
        ));
        return;
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    // Match the decoder's mi grid: mi_cols/mi_rows = ALIGN(dim, 8) >> MI_SIZE_LOG2
    // (av2 aligns the frame to 8 luma px for CDEF, not to the 64-px superblock),
    // so reference-sample availability at partial edge SBs matches tile.mi_col_end.
    let mi_col_end = (((width + 7) & !7) >> 2) as i64;
    let mi_row_end = (((height + 7) & !7) >> 2) as i64;
    let sb_y0 = (y0 / 64) * 64;
    let sb_x0 = (x0 / 64) * 64;
    let mi_row = (sb_y0 >> 2) as i64;
    let mi_col = (sb_x0 >> 2) as i64;
    let (row_off, col_off) = ((y0 - sb_y0) as i64 / 4, (x0 - sb_x0) as i64 / 4);
    let (lx, ly) = ((x0 - sb_x0) as i64, (y0 - sb_y0) as i64); // TX offset px within SB
    let xr = ((mi_col_end - mi_col - 16) << 2) + 32 - lx;
    let yd = ((mi_row_end - mi_row - 16) << 2) + 32 - ly;
    let right_available = (mi_col + col_off + 8) < mi_col_end;
    let bottom_available = (yd > 0) && ((mi_row + row_off + 8) < mi_row_end);
    // top-right: needed by TX 0/1/2 (TX 3 has col_off+txw==block width -> none)
    let tr_ok = matches!(i, 0..=2) && have_above && right_available && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    // bottom-left: only TX 0 (others sit at/under the block's bottom-left edge)
    let bl_ok = i == 0 && have_left && bottom_available && yd > 0;
    let bl_px = if bl_ok {
        (yd.min(32)).max(0) as usize
    } else {
        0
    };
    let avail_above = (((mi_col_end << 2) as usize).saturating_sub(x0))
        .min(32)
        .max(1);
    let avail_left = (((mi_row_end << 2) as usize).saturating_sub(y0))
        .min(32)
        .max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred_into(
        2 * neutral as i32 - 1,
        m,
        adelta,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32 + bl_px as i32,
        out,
    )
}

pub(crate) fn part_lambda(qstep: i32, c: f32) -> f32 {
    c * (qstep as f32) * (qstep as f32)
}

#[inline]
fn luma_mode_lambda(rdoq_lambda: f32, qstep: i32, pixel_lambda: bool) -> f32 {
    let level_lambda = rdoq_lambda.max(proj::DEFAULT_RDOQ_LAMBDA);
    if pixel_lambda {
        crate::av2::rd::lambda_from_level(level_lambda, qstep) * crate::av2::rd::knob("LUMA")
    } else {
        level_lambda * (qstep as f32) * (qstep as f32)
    }
}

/// The palette candidate of a `w` x `h` luma block for the leaf coders that
/// price with `quant`'s mode lambda: None when palettes are off for this SB, the
/// block crosses the frame edge or is flat. Deterministic in the source, so a
/// leaf that picked [`palette::PAL_MODE`] re-derives the same palette to emit it.
pub(crate) fn luma_palette(
    source: &LumaSource<'_>,
    frame: (usize, usize),
    y: usize,
    x: usize,
    w: usize,
    h: usize,
    quant: &LumaQuantSpec<'_>,
) -> Option<palette::Palette> {
    if !palette::enabled() || y + h > frame.1 || x + w > frame.0 {
        return None;
    }
    let mut src = [0f32; 64 * 64];
    for (d, &v) in src
        .iter_mut()
        .zip(rect_rows(source.plane, source.stride, y, x, w, h).flatten())
    {
        *d = v;
    }
    palette::search(
        &src[..w * h],
        w,
        h,
        quant.bit_depth as u32,
        luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda),
    )
}

fn project_luma_rdoq(
    luma: &Basis,
    resid: &[f32],
    scan: &[u16],
    qc: usize,
    cost: &mut f32,
    lambda: f32,
) -> Vec<f32> {
    if lambda > 0.0 {
        let (l, c) = luma.project_scan_rdoq(resid, scan, |prm, l| {
            coder::rdoq_luma(prm, l, qc, scan, 1024, lambda)
        });
        *cost += c;
        l
    } else {
        let l = luma.project(resid, 0.0);
        *cost += coeff_rate_f32(&l);
        l
    }
}

pub(super) fn encode_luma_sb(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaFrameBlock,
    quant: &LumaQuantSpec<'_>,
    search: &LumaSbSearch,
) -> ([Vec<Coeff>; 4], usize, i8, f32) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaFrameBlock {
        frame_width: width,
        frame_height: height,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        pixel_lambda,
        speed,
        bit_depth: bd,
        ..
    } = quant;
    let &LumaSbSearch {
        residual_scale: resid_scale,
        allow_directional: allow_dir,
    } = search;
    static POS: [(usize, usize); 4] = [(0, 0), (0, 32), (32, 0), (32, 32)];
    let mut best_cost = f32::INFINITY;
    let mut best_mode = 0usize;
    let mut best_delta = 0i32;
    let mut best_tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut best_region = [0f32; 64 * 64];
    // Fast reduces the intra candidate set; non-Full tiers rank candidates with a
    // cheap coeff cost (RDOQ disabled) and re-RDOQ the winner only.
    let base_modes: &[usize] = if speed.reduced_modes() {
        if allow_dir {
            &[0usize, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12]
        } else {
            &[0usize, 1, 2]
        }
    } else if allow_dir {
        &[0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    // Directional modes trial all angle deltas (Δ = -3..=3, ×3°) only on the
    // Slow tier; faster tiers use the nominal angle (Δ=0) to avoid 7× the
    // directional RD candidates. Non-directional modes are always Δ=0.
    let dir_deltas: &[i32] = if speed.try_angle_deltas() {
        &[0, -1, 1, -2, 2, -3, 3]
    } else {
        &[0]
    };

    let mut cands: Vec<(usize, i32)> = Vec::new();
    for &m in base_modes {
        if m >= 5 {
            for &d in dir_deltas {
                cands.push((m, d));
            }
        } else {
            cands.push((m, 0));
        }
    }
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Residual scratch reused across every (mode, angle) candidate of this SB,
    // instead of allocating a fresh 1024-element buffer per candidate.
    let resid_buf = std::cell::RefCell::new(vec![0f32; 1024]);
    // Encode one (mode, angle_delta) into `recy`, returning its TU coeffs, coded-bit
    // cost, and reconstruction SSE (for a proper rate-distortion mode decision).
    // Screen content: the palette (DC_PRED + map) is one more full-RD candidate.
    let pal = luma_palette(source, (width, height), sb_y, sb_x, 64, 64, quant)
        .map(|p| (p.pred(), p.bits));
    let encode_mode =
        |recy: &mut [f32], m: usize, adelta: i32, lambda: f32| -> ([Vec<Coeff>; 4], f32, f32) {
            let mut resid = resid_buf.borrow_mut();
            let mut cost = 0f32;
            let mut sse = 0f32;
            let mut tus: [Vec<Coeff>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
            for (i, &(ty, tx)) in POS.iter().enumerate() {
                let (y0, x0) = (sb_y + ty, sb_x + tx);
                let pblk = if m == palette::PAL_MODE {
                    let pp = &pal.as_ref().expect("palette candidate").0;
                    rect_rows(pp, 64, ty, tx, 32, 32)
                        .flatten()
                        .copied()
                        .collect()
                } else {
                    predict_luma(
                        recy,
                        &LumaPredictSpec {
                            stride: pw,
                            frame_width: width,
                            frame_height: height,
                            tx_index: i,
                            y: y0,
                            x: x0,
                            mode: m,
                            angle_delta: adelta,
                            neutral,
                        },
                    )
                };
                fill_scaled_residual(&mut resid, yp, pw, y0, x0, &pblk, 32, 32, resid_scale);
                let lev = if lambda > 0.0 {
                    // Trellis RDOQ: pick coefficient levels by real rate-distortion
                    // (rate = true coded bits), then RD-trim the EOB.
                    let (l, c) =
                        luma.project_scan_rdoq(&resid[..], &crate::av2::tables::SCAN, |prm, l| {
                            coder::rdoq_luma(prm, l, qc, scan, 1024, lambda)
                        });
                    cost += c;
                    l
                } else {
                    let l = luma.project(&resid[..], 0.0);
                    cost += coeff_rate_f32(&l) as f32;
                    l
                };
                let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
                // real reconstruction SSE vs source for this TU
                sse += rect_sse_f32(
                    &PlaneRect {
                        plane: yp,
                        stride: pw,
                        y: y0,
                        x: x0,
                    },
                    &PlaneRect {
                        plane: &rb,
                        stride: 32,
                        y: 0,
                        x: 0,
                    },
                    32,
                    32,
                );
                put_block(recy, pw, y0, x0, 32, &rb);
                tus[i] = levels_to_coeffs(&lev);
            }
            // Mode-signaling cost (once per 64x64 block). DC is cheapest; SMOOTH/PAETH
            // and directional cost a few extra bits (directional a bit more for the
            // set/idx symbols), so only win when they earn them.
            if m == palette::PAL_MODE {
                cost += pal.as_ref().map_or(0.0, |p| p.1);
            } else if m != 0 {
                cost += if m >= 5 { 9.0 } else { 6.0 };
            }
            (tus, cost, sse)
        };
    // Stage 1 — cheap SATD prune. Score each (mode, angle) by the SATD of its
    // prediction vs source (predict-only: no forward transform, RDOQ or
    // reconstruct), keep only the top-K, and run the expensive SSE/RD pass on
    // those. SATD tracks post-transform coding cost well, so the true winner is
    // essentially always in the top few. The prediction is written into `recy` so
    // subsequent 32x32 sub-blocks predict from the prior ones (as the real coder
    // does); TU[0] always predicts from the SB's finalized neighbors, and the
    // full Stage-2 pass overwrites `recy` correctly, so this scratch use is safe.
    let satd_of_mode = |recy: &mut [f32], m: usize, adelta: i32| -> u64 {
        let mut satd = 0u64;
        let mut pblk = [0f32; 1024];
        for (i, &(ty, tx)) in POS.iter().enumerate() {
            let (y0, x0) = (sb_y + ty, sb_x + tx);
            predict_luma_into(
                recy,
                &LumaPredictSpec {
                    stride: pw,
                    frame_width: width,
                    frame_height: height,
                    tx_index: i,
                    y: y0,
                    x: x0,
                    mode: m,
                    angle_delta: adelta,
                    neutral,
                },
                &mut pblk,
            );
            satd += crate::av2::metrics::satd_f32(&yp[y0 * pw + x0..], pw, &pblk, 32, 32, 32);
            put_block(recy, pw, y0, x0, 32, &pblk);
        }
        satd
    };
    let keep = crate::av2::rd::knob_or(
        "W64K",
        if quant.prune & PRUNE_LARGE != 0 || !speed.try_angle_deltas() {
            4.0
        } else {
            8.0
        },
    ) as usize;
    // W64R = R: hierarchical screen — nominal angles first, then the angle
    // deltas around the R SATD-best directional modes only.
    let refine = crate::av2::rd::knob_or(
        "W64R",
        if quant.prune & PRUNE_PROBE != 0 {
            2.0
        } else {
            0.0
        },
    ) as usize;
    let cands: Vec<(usize, i32)> = if cands.len() > keep {
        let mut ranked: Vec<(u64, (usize, i32))> = if refine > 0 {
            let mut r: Vec<(u64, (usize, i32))> = cands
                .iter()
                .filter(|&&(_, d)| d == 0)
                .map(|&(m, d)| (satd_of_mode(recy, m, d), (m, d)))
                .collect();
            let mut roots: Vec<(u64, usize)> = r
                .iter()
                .filter(|&&(_, (m, _))| m >= 5)
                .map(|&(sc, (m, _))| (sc, m))
                .collect();
            roots.sort_by_key(|&(sc, _)| sc);
            for (_, m) in roots.into_iter().take(refine) {
                for &(cm, d) in cands.iter().filter(|&&(cm, d)| cm == m && d != 0) {
                    r.push((satd_of_mode(recy, cm, d), (cm, d)));
                }
            }
            r
        } else {
            cands
                .iter()
                .map(|&(m, d)| (satd_of_mode(recy, m, d), (m, d)))
                .collect()
        };
        ranked.sort_by_key(|&(s, _)| s);
        ranked.truncate(keep);
        ranked.into_iter().map(|(_, md)| md).collect()
    } else {
        cands
    };
    // Mode-RD weight in pixel-SSE per bit: rdoq_lambda is level^2/bit, qstep^2 is
    // pixel^2/level^2, so their product converts coded bits to the SSE domain. This makes
    // the mode search a proper rate-distortion decision instead of cheapest-to-code.
    let mut cands = cands;
    if pal.is_some() {
        cands.push((palette::PAL_MODE, 0));
    }
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep, pixel_lambda);
    let mut best_bits = 0f32;
    for &(m, d) in &cands {
        let (tus, cost, sse) = encode_mode(recy, m, d, search_lambda);
        let j = crate::av2::rd::rd_cost(sse, cost, mode_lambda);
        if j < best_cost {
            best_cost = j;
            best_bits = cost;
            best_mode = m;
            best_delta = d;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 64))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        // Winner already coded at the final RDOQ setting (or RDOQ is off):
        // restore the saved winning reconstruction.
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 64, 64).zip(best_region.as_chunks::<64>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        // Winner-only RDOQ: re-encode the chosen mode with real RDOQ.
        let (tus, cost, _) = encode_mode(recy, best_mode, best_delta, rdoq_lambda);
        best_tus = tus;
        best_bits = cost;
    }
    (best_tus, best_mode, best_delta as i8, best_bits)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn predict_luma_leaf32(
    recy: &[f32],
    pw: usize,
    mi_cols: i64,
    _mi_rows: i64,
    sb_y: usize,
    sb_x: usize,
    ti: usize,
    m: usize,
    neutral: f32,
) -> Vec<f32> {
    let (y0, x0) = (sb_y, sb_x + ti * 32);
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    let mi_col = (sb_x >> 2) as i64;
    let lx = (ti * 32) as i64;
    let col_off = lx / 4;
    // top-right reference width (px), clamped to 32. Same geometry as predict_luma
    // but with the native column bound.
    let xr = ((mi_cols - mi_col - 16) << 2) + 32 - lx;
    let right_available = (mi_col + col_off + 8) < mi_cols;
    let tr_ok = have_above && right_available && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    let avail_above = ((mi_cols << 2) as usize).saturating_sub(x0).min(32).max(1);
    let avail_left = ((_mi_rows << 2) as usize).saturating_sub(y0).min(32).max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: 0,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred(
        2 * neutral as i32 - 1,
        m,
        0,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32,
    )
}

pub(super) fn encode_luma_leaf32(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
    dir: Option<LeafDirSearch<'_>>,
) -> ([Vec<Coeff>; 2], usize, i8) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols,
        mi_rows,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        pixel_lambda,
        speed,
        bit_depth: bd,
        ..
    } = quant;
    let mut best_cost = f32::INFINITY;
    let mut best_mode = (0usize, 0i32);
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = [0f32; 64 * 32];
    let nm = crate::av2::rd::knob_or("R64M", 5.0) as usize;
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2][..nm.min(3)]
    } else {
        &[0usize, 1, 2, 3, 4][..nm.min(5)]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Each TX_32X32 of the leaf predicts as a square 32x32 (directional modes
    // too: no wide-angle remap for square TUs), from the coded-mi mask. Every
    // non-DC mode goes through it: AVM's SMOOTH reads left[bh] / above[bw], one
    // sample past the TU, so the true bottom-left / top-right availability matters
    // (the legacy TU predictor assumes none, right only at the frame's bottom edge).
    let predict_tu = |recy: &[f32], ti: usize, (m, d): (usize, i32)| -> Vec<f32> {
        match dir {
            Some(ds) if m >= 1 => predict_luma_square(
                recy,
                &SquareLumaBlock {
                    stride: pw,
                    frame_width: ds.frame_width,
                    frame_height: ds.frame_height,
                    y: sb_y,
                    x: sb_x + ti * 32,
                    size: 32,
                },
                ds.coded,
                SquareLumaMode {
                    mode: m,
                    angle_delta: d,
                    smooth_neighbour: false,
                },
                neutral,
                bd,
            ),
            _ => predict_luma_leaf32(recy, pw, mi_cols, mi_rows, sb_y, sb_x, ti, m, neutral),
        }
    };
    let encode_mode =
        |recy: &mut [f32], (m, d): (usize, i32), lambda: f32| -> ([Vec<Coeff>; 2], f32, f32) {
            let mut resid = [0f32; 1024];
            let mut cost = 0f32;
            let mut sse = 0f32;
            let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
            for (ti, tu) in tus.iter_mut().enumerate() {
                let (y0, x0) = (sb_y, sb_x + ti * 32);
                let pblk = predict_tu(recy, ti, (m, d));
                fill_scaled_residual(
                    &mut resid,
                    yp,
                    pw,
                    y0,
                    x0,
                    &pblk,
                    32,
                    32,
                    luma.qstep as f32 / qstep as f32,
                );
                let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
                let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
                sse += rect_sse_f32(
                    &PlaneRect {
                        plane: yp,
                        stride: pw,
                        y: y0,
                        x: x0,
                    },
                    &PlaneRect {
                        plane: &rb,
                        stride: 32,
                        y: 0,
                        x: 0,
                    },
                    32,
                    32,
                );
                put_block(recy, pw, y0, x0, 32, &rb);
                *tu = levels_to_coeffs(&lev);
            }
            if m >= 5 {
                cost += crate::av2::rd::knob_or("MBDIR", 9.0);
            } else if m != 0 {
                cost += 6.0;
            }
            (tus, cost, sse)
        };
    let mut cands: Vec<(usize, i32)> = cands.iter().map(|&m| (m, 0)).collect();
    if let Some(ds) = dir.filter(|_| speed.try_directional()) {
        cands.extend(screen_directional(
            crate::av2::rd::knob_or("DIR32", 3.0) as usize,
            speed.try_angle_deltas(),
            ds.refine,
            |m, d| {
                (0..2)
                    .map(|ti| {
                        let p = predict_tu(recy, ti, (m, d));
                        crate::av2::metrics::satd_f32(
                            &yp[sb_y * pw + sb_x + ti * 32..],
                            pw,
                            &p,
                            32,
                            32,
                            32,
                        )
                    })
                    .sum()
            },
        ));
    }
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep, pixel_lambda);
    for &m in &cands {
        let (tus, rate, sse) = encode_mode(recy, m, search_lambda);
        let j = crate::av2::rd::rd_cost(sse, rate, mode_lambda);
        if j < best_cost {
            best_cost = j;
            best_mode = m;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<64>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 64, 32))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 64, 32).zip(best_region.as_chunks::<64>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        let (tus, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode.0, best_mode.1 as i8)
}

pub(crate) fn predict_luma_leaf_tu(recy: &[f32], spec: &LumaLeafPredictSpec) -> Vec<f32> {
    let &LumaLeafPredictSpec {
        stride: pw,
        mi_cols: mc,
        mi_rows: mr,
        block_y: sb_y,
        block_x: sb_x,
        tx_y: ty,
        tx_x: tx,
        tx_index: i,
        mode: m,
        neutral,
        split_leaf,
    } = spec;
    let (y0, x0) = (sb_y + ty, sb_x + tx);
    if m == 0 {
        return vec![dc_pred(recy, pw, y0, x0, 32, neutral); 1024];
    }
    let have_above = y0 > 0;
    let have_left = x0 > 0;
    let sb_x0 = (x0 / 64) * 64;
    let sb_y0 = (y0 / 64) * 64;
    let is_right = (x0 - sb_x0) >= 32;
    let is_bottom = (y0 - sb_y0) >= 32;
    let mi_col = (sb_x >> 2) as i64;
    let mi_row = (sb_y >> 2) as i64;
    let (lx, ly) = (tx as i64, ty as i64);
    let (col_off, row_off) = (lx / 4, ly / 4);
    let xr = ((mc - mi_col - 16) << 2) + 32 - lx;
    let yd = ((mr - mi_row - 16) << 2) + 32 - ly;
    // top-right unavailable for the bottom-right leaf; bottom-left only for the top-left leaf.
    // avm has_top_right: top-row leaves (tr_mask_row<0) read above-right from the
    // coded SB row above, so TL and TR both qualify; only bottom-row leaves don't.
    #[allow(clippy::overly_complex_bool_expr)]
    let tr_avail = !(is_right && is_bottom) && !is_bottom && (mi_col + col_off + 8) < mc;
    #[allow(clippy::overly_complex_bool_expr)]
    let bl_avail = !is_right && !is_bottom && (mi_row + row_off + 8) < mr;
    // SMOOTH/SMOOTH_H/D45/D67 need above-right; SMOOTH/SMOOTH_V/D203 need bottom-left.
    let need_tr = matches!(m, 1 | 3 | 7) || m >= 12;
    let need_bl = matches!(m, 1 | 2 | 11);
    let tr_ok = need_tr && matches!(i, 0..=2) && have_above && tr_avail && xr > 0;
    let tr_px = if tr_ok { xr.min(32).max(0) as usize } else { 0 };
    let bl_ok = need_bl
        && i == 0
        && have_left
        && (bl_avail || (split_leaf && is_right && !is_bottom))
        && yd > 0;
    let bl_px = if bl_ok { yd.min(32).max(0) as usize } else { 0 };
    let avail_above = (((mc << 2) as usize).saturating_sub(x0)).min(32).max(1);
    let avail_left = (((mr << 2) as usize).saturating_sub(y0)).min(32).max(1);
    let (ab, lf, corner) = intrapred::build_refs(
        recy,
        &intrapred::IntraRefSpec {
            stride: pw,
            y: y0,
            x: x0,
            block_size: 32,
            have_above,
            have_left,
            top_right: tr_px,
            bottom_left: bl_px,
            neutral,
            available_above: avail_above,
            available_left: avail_left,
        },
    );
    dispatch_intra_pred(
        2 * neutral as i32 - 1,
        m,
        0,
        &ab,
        &lf,
        corner,
        have_above,
        have_left,
        32 + tr_px as i32,
        32 + bl_px as i32,
    )
}

/// Project + trial-code a right-edge 32x64 luma leaf as two stacked TX_32X32
/// (top i=0, bottom i=2). Mirrors `encode_luma_leaf32` but vertical.
pub(super) fn encode_luma_leaf_v32x64(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
    dir: Option<LeafDirSearch<'_>>,
) -> ([Vec<Coeff>; 2], usize, i8) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols: mc,
        mi_rows: mr,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        pixel_lambda,
        speed,
        bit_depth: bd,
        ..
    } = quant;
    let tu_i = [(0usize, 0usize), (32usize, 2usize)]; // (ty, raster-i)
    let mut best_cost = f32::INFINITY;
    let mut best_mode = (0usize, 0i32);
    let mut best_tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
    let mut best_region = [0f32; 32 * 64];
    let nm = crate::av2::rd::knob_or("R64M", 5.0) as usize;
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2][..nm.min(3)]
    } else {
        &[0usize, 1, 2, 3, 4][..nm.min(5)]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Each TX_32X32 predicts as a square 32x32 (see encode_luma_leaf32).
    let predict_tu = |recy: &[f32], ty: usize, i: usize, (m, d): (usize, i32)| -> Vec<f32> {
        match dir {
            Some(ds) if m >= 1 => predict_luma_square(
                recy,
                &SquareLumaBlock {
                    stride: pw,
                    frame_width: ds.frame_width,
                    frame_height: ds.frame_height,
                    y: sb_y + ty,
                    x: sb_x,
                    size: 32,
                },
                ds.coded,
                SquareLumaMode {
                    mode: m,
                    angle_delta: d,
                    smooth_neighbour: false,
                },
                neutral,
                bd,
            ),
            _ => predict_luma_leaf_tu(
                recy,
                &LumaLeafPredictSpec {
                    stride: pw,
                    mi_cols: mc,
                    mi_rows: mr,
                    block_y: sb_y,
                    block_x: sb_x,
                    tx_y: ty,
                    tx_x: 0,
                    tx_index: i,
                    mode: m,
                    neutral,
                    split_leaf: false,
                },
            ),
        }
    };
    let encode_mode =
        |recy: &mut [f32], (m, d): (usize, i32), lambda: f32| -> ([Vec<Coeff>; 2], f32, f32) {
            let mut resid = [0f32; 1024];
            let mut cost = 0f32;
            let mut sse = 0f32;
            let mut tus: [Vec<Coeff>; 2] = [Vec::new(), Vec::new()];
            for (k, &(ty, i)) in tu_i.iter().enumerate() {
                let (y0, x0) = (sb_y + ty, sb_x);
                let pblk = predict_tu(recy, ty, i, (m, d));
                fill_scaled_residual(
                    &mut resid,
                    yp,
                    pw,
                    y0,
                    x0,
                    &pblk,
                    32,
                    32,
                    luma.qstep as f32 / qstep as f32,
                );
                let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
                let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
                sse += rect_sse_f32(
                    &PlaneRect {
                        plane: yp,
                        stride: pw,
                        y: y0,
                        x: x0,
                    },
                    &PlaneRect {
                        plane: &rb,
                        stride: 32,
                        y: 0,
                        x: 0,
                    },
                    32,
                    32,
                );
                put_block(recy, pw, y0, x0, 32, &rb);
                tus[k] = levels_to_coeffs(&lev);
            }
            if m >= 5 {
                cost += crate::av2::rd::knob_or("MBDIR", 9.0);
            } else if m != 0 {
                cost += 6.0;
            }
            (tus, cost, sse)
        };
    let mut cands: Vec<(usize, i32)> = cands.iter().map(|&m| (m, 0)).collect();
    if let Some(ds) = dir.filter(|_| speed.try_directional()) {
        cands.extend(screen_directional(
            crate::av2::rd::knob_or("DIR32", 3.0) as usize,
            speed.try_angle_deltas(),
            ds.refine,
            |m, d| {
                tu_i.iter()
                    .map(|&(ty, i)| {
                        let p = predict_tu(recy, ty, i, (m, d));
                        crate::av2::metrics::satd_f32(
                            &yp[(sb_y + ty) * pw + sb_x..],
                            pw,
                            &p,
                            32,
                            32,
                            32,
                        )
                    })
                    .sum()
            },
        ));
    }
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep, pixel_lambda);
    for &m in &cands {
        let (tus, rate, sse) = encode_mode(recy, m, search_lambda);
        let j = crate::av2::rd::rd_cost(sse, rate, mode_lambda);
        if j < best_cost {
            best_cost = j;
            best_mode = m;
            best_tus = tus;
            for (dst_row, src_row) in best_region
                .as_chunks_mut::<32>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, sb_y, sb_x, 32, 64))
            {
                dst_row.copy_from_slice(src_row);
            }
        }
    }
    if speed.per_candidate_rdoq() || rdoq_lambda <= 0.0 {
        for (dst_row, src_row) in
            rect_rows_mut(recy, pw, sb_y, sb_x, 32, 64).zip(best_region.as_chunks::<32>().0.iter())
        {
            dst_row.copy_from_slice(src_row);
        }
    } else {
        let (tus, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tus = tus;
    }
    (best_tus, best_mode.0, best_mode.1 as i8)
}

/// Directional candidates for a coded 32x32 leaf: the superblock's coded-mi mask
/// (top-right / bottom-left availability) and the frame (tile) extent.
#[derive(Clone, Copy)]
pub(super) struct LeafDirSearch<'a> {
    /// Nominal angles whose angle deltas the SATD screen also tries.
    pub(super) refine: usize,
    pub(super) coded: &'a [u8; 256],
    pub(super) frame_width: usize,
    pub(super) frame_height: usize,
}

/// Project + trial-code a corner 32x32 luma leaf as a single TX_32X32 (i=0).
/// With `dir`, the SATD-best directional (mode, angle delta) pairs also get the
/// full RD; returns (coeffs, internal mode, angle delta).
pub(super) fn encode_luma_leaf_s32x32(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaGridBlock,
    quant: &LumaQuantSpec<'_>,
    dir: Option<LeafDirSearch<'_>>,
    forced: Option<(usize, i32)>,
) -> (Vec<Coeff>, usize, i8) {
    let &LumaSource {
        plane: yp,
        stride: pw,
    } = source;
    let &LumaGridBlock {
        mi_cols: mc,
        mi_rows: mr,
        y: sb_y,
        x: sb_x,
    } = block;
    let &LumaQuantSpec {
        basis: luma,
        qstep,
        scan,
        neutral,
        quant_context: qc,
        rdoq_lambda,
        pixel_lambda,
        speed,
        bit_depth: bd,
        ..
    } = quant;
    let cands: &[usize] = if speed.reduced_modes() {
        &[0usize, 1, 2]
    } else {
        &[0usize, 1, 2, 3, 4]
    };
    let search_lambda = if speed.per_candidate_rdoq() {
        rdoq_lambda
    } else {
        0.0
    };
    // Single TU: no intra-leaf feedback, so a mode is fully described by its
    // (recon, coeffs, cost) and the winner can simply be re-projected with RDOQ.
    // Every candidate predicts from the same neighbourhood: share its edges.
    let refs = std::cell::RefCell::new(SquareRefCache::default());
    let predict_dir = |recy: &[f32], m: usize, d: i32, out: &mut [f32]| {
        let ds = dir.expect("directional candidate without a dir search");
        predict_luma_square_cached(
            recy,
            &SquareLumaBlock {
                stride: pw,
                frame_width: ds.frame_width,
                frame_height: ds.frame_height,
                y: sb_y,
                x: sb_x,
                size: 32,
            },
            ds.coded,
            SquareLumaMode {
                mode: m,
                angle_delta: d,
                // TX_32X32: both edge-filter types give the same strengths.
                smooth_neighbour: false,
            },
            neutral,
            bd,
            &mut refs.borrow_mut(),
            out,
        )
    };
    let pal = if forced.is_none_or(|(m, _)| m == palette::PAL_MODE) {
        luma_palette(
            source,
            (mc as usize * 4, mr as usize * 4),
            sb_y,
            sb_x,
            32,
            32,
            quant,
        )
        .map(|p| (p.pred(), p.bits))
    } else {
        None
    };
    let encode_mode =
        |recy: &[f32], (m, d): (usize, i32), lambda: f32| -> ([f32; 1024], Vec<Coeff>, f32, f32) {
            // Every non-DC mode predicts from the coded-mi mask when it is known:
            // SMOOTH reads one sample past the TU (left[bh] / above[bw]), so the true
            // bottom-left / top-right availability matters outside the quad tree.
            let pblk = if m == palette::PAL_MODE {
                pal.as_ref().expect("palette candidate").0.clone()
            } else if m >= 5 || (m >= 1 && dir.is_some()) {
                let mut p = vec![0f32; 1024];
                predict_dir(recy, m, d, &mut p);
                p
            } else {
                predict_luma_leaf_tu(
                    recy,
                    &LumaLeafPredictSpec {
                        stride: pw,
                        mi_cols: mc,
                        mi_rows: mr,
                        block_y: sb_y,
                        block_x: sb_x,
                        tx_y: 0,
                        tx_x: 0,
                        tx_index: 0,
                        mode: m,
                        neutral,
                        split_leaf: true,
                    },
                )
            };
            let mut resid = [0f32; 1024];
            fill_scaled_residual(
                &mut resid,
                yp,
                pw,
                sb_y,
                sb_x,
                &pblk,
                32,
                32,
                luma.qstep as f32 / qstep as f32,
            );
            let mut cost = 0f32;
            let lev = project_luma_rdoq(luma, &resid, scan, qc, &mut cost, lambda);
            if m == palette::PAL_MODE {
                cost += pal.as_ref().map_or(0.0, |p| p.1);
            } else if m >= 5 {
                cost += crate::av2::rd::knob_or("MBDIR", 9.0);
            } else if m != 0 {
                cost += 6.0;
            }
            let rb = reconstruct_luma(&pblk, &lev, qstep, scan, bd);
            let sse = rect_sse_f32(
                &PlaneRect {
                    plane: yp,
                    stride: pw,
                    y: sb_y,
                    x: sb_x,
                },
                &PlaneRect {
                    plane: &rb,
                    stride: 32,
                    y: 0,
                    x: 0,
                },
                32,
                32,
            );
            (rb, levels_to_coeffs(&lev), cost, sse)
        };
    // SATD prune (same as encode_luma_sb): this is a single 32x32 TU with no
    // intra-leaf feedback, so every mode's prediction is independent — SATD-rank
    // them (predict-only) and full-encode just the top-K.
    // Unified prune (knob PRK32 = K): every non-DC candidate, directional ones
    // included, is SATD-ranked together and only the best K are coded.
    let prk32 = crate::av2::rd::knob_or(
        "PRK32",
        if quant.prune & PRUNE_LARGE != 0 {
            2.0
        } else {
            0.0
        },
    ) as usize;
    let keep = if prk32 > 0 {
        usize::MAX
    } else if speed.reduced_modes() {
        2
    } else {
        3
    };
    let mut satd_nondir: Vec<(u64, usize)> = Vec::new();
    let cands: Vec<usize> = if forced.is_some() {
        Vec::new()
    } else if prk32 > 0 || cands.len() > keep {
        let mut ranked: Vec<(u64, usize)> = cands
            .iter()
            .map(|&m| {
                let pblk = predict_luma_leaf_tu(
                    recy,
                    &LumaLeafPredictSpec {
                        stride: pw,
                        mi_cols: mc,
                        mi_rows: mr,
                        block_y: sb_y,
                        block_x: sb_x,
                        tx_y: 0,
                        tx_x: 0,
                        tx_index: 0,
                        mode: m,
                        neutral,
                        split_leaf: true,
                    },
                );
                (
                    crate::av2::metrics::satd_f32(&yp[sb_y * pw + sb_x..], pw, &pblk, 32, 32, 32),
                    m,
                )
            })
            .collect();
        ranked.sort_by_key(|&(s, _)| s);
        ranked.truncate(keep);
        satd_nondir = ranked.clone();
        ranked.into_iter().map(|(_, m)| m).collect()
    } else {
        cands.to_vec()
    };
    let mut cands: Vec<(usize, i32)> = cands.into_iter().map(|m| (m, 0)).collect();
    let dir_keep = crate::av2::rd::knob_or("DIR32", 3.0) as usize;
    if let Some(md) = forced {
        // Replay of a captured winner: code just that candidate.
        cands.push(md);
    } else if dir.is_some() && dir_keep > 0 && speed.try_directional() {
        cands.extend(screen_directional(
            dir_keep,
            speed.try_angle_deltas(),
            dir.map_or(8, |d| d.refine),
            |m, d| {
                let mut p = [0f32; 1024];
                predict_dir(recy, m, d, &mut p);
                crate::av2::metrics::satd_f32(&yp[sb_y * pw + sb_x..], pw, &p, 32, 32, 32)
            },
        ));
    }
    if forced.is_none() && prk32 > 0 && cands.len() > prk32 {
        // DC stays; the rest compete on SATD.
        let mut ranked: Vec<(u64, (usize, i32))> = cands
            .iter()
            .filter(|&&(m, _)| m != 0)
            .map(|&(m, d)| {
                let s = match satd_nondir
                    .iter()
                    .find(|&&(_, nm)| nm == m && d == 0 && m < 5)
                {
                    Some(&(s, _)) => s,
                    None => {
                        let mut p = [0f32; 1024];
                        predict_dir(recy, m, d, &mut p);
                        crate::av2::metrics::satd_f32(&yp[sb_y * pw + sb_x..], pw, &p, 32, 32, 32)
                    }
                };
                (s, (m, d))
            })
            .collect();
        ranked.sort_by_key(|&(s, _)| s);
        let has_dc = cands.iter().any(|&(m, _)| m == 0);
        cands = has_dc
            .then_some((0, 0))
            .into_iter()
            .chain(ranked.into_iter().take(prk32).map(|(_, md)| md))
            .collect();
    }
    if forced.is_none() && pal.is_some() {
        cands.push((palette::PAL_MODE, 0));
    }
    let mut best_cost = f32::INFINITY;
    let mut best_mode = (0usize, 0i32);
    let mut best_tu: Vec<Coeff> = Vec::new();
    let mut best_region = [0f32; 32 * 32];
    let mode_lambda = luma_mode_lambda(rdoq_lambda, qstep, pixel_lambda);
    for &md in &cands {
        let (rb, tu, rate, sse) = encode_mode(recy, md, search_lambda);
        let j = crate::av2::rd::rd_cost(sse, rate, mode_lambda);
        if j < best_cost {
            best_cost = j;
            best_mode = md;
            best_tu = tu;
            best_region.copy_from_slice(&rb);
        }
    }
    if !speed.per_candidate_rdoq() && rdoq_lambda > 0.0 {
        // Winner-only RDOQ: re-project the chosen mode with real RDOQ.
        let (rb, tu, _, _) = encode_mode(recy, best_mode, rdoq_lambda);
        best_tu = tu;
        best_region.copy_from_slice(&rb);
    }
    put_block(recy, pw, sb_y, sb_x, 32, &best_region);
    (best_tu, best_mode.0, best_mode.1 as i8)
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct LumaPartitionDecision {
    pub(super) split64: bool,
    pub(super) split32: [bool; 4],
    pub(super) split16: [bool; 16],
    /// Rect leaves of an unsplit node: 0 = NONE, 1 = HORZ pair, 2 = VERT pair.
    /// `rect64` applies when `!split64` (64x32 / 32x64 leaves), `rect32[k]` to an
    /// unsplit 32x32 quadrant (32x16 / 16x32), `rect16[i]` to an unsplit 16x16
    /// (16x8 / 8x16). The partition probe never sets them; real-RD trials do.
    pub(super) rect64: u8,
    pub(super) rect32: [u8; 4],
    pub(super) rect16: [u8; 16],
    /// A free binary partition tree (overrides the fields above when set).
    pub(crate) gtree: Option<GTree>,
    /// The probe's packed directional picks per 8x8 (raster within the SB), reused
    /// by the 8x8 leaf coder; 0 where the probe did not screen.
    pub(crate) dir8: [[u32; 32]; 2],
}

/// A 64x64 luma partition tree in pre-order: 0 = leaf, 1 = HORZ, 2 = VERT (each
/// split node is followed by its two subtrees).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GTree {
    pub(crate) len: u8,
    pub(crate) codes: [u8; 128],
}

impl GTree {
    pub(crate) fn from_codes(codes: &[u8]) -> Self {
        let mut c = [0u8; 128];
        c[..codes.len()].copy_from_slice(codes);
        GTree {
            len: codes.len() as u8,
            codes: c,
        }
    }

    pub(crate) fn codes(&self) -> &[u8] {
        &self.codes[..self.len as usize]
    }
}

impl LumaPartitionDecision {
    /// The probe's directional picks for the 8x8 at (`dy`, `dx`) within the SB.
    pub(crate) fn dir8_at(&self, dy: usize, dx: usize) -> Option<Vec<(usize, i32)>> {
        let i = (dy / 8) * 8 + dx / 8;
        dir8_unpack(self.dir8[i / 32][i % 32])
    }

    /// Whether the SB codes anything but one whole-64 leaf.
    pub(crate) fn is_tree(&self) -> bool {
        self.split64 || self.rect64 != 0 || self.gtree.is_some_and(|g| g.len > 1)
    }
}

#[derive(Clone, Copy)]
pub(super) struct LumaPartitionSearch<'a> {
    pub(super) quant: LumaQuantSpec<'a>,
    pub(super) sb: LumaSbSearch,
    pub(super) basis16: &'a Basis,
    pub(super) basis8: &'a Basis,
    /// Subsampled leaf coders currently support 16x16 only for boundary residue,
    /// not as a regular interior chroma-reference leaf.
    pub(super) allow_16x16: bool,
    /// 8x8 luma leaves require chroma-reference sharing in subsampled formats.
    /// Keep them out of the search until that syntax is emitted by the leaf coder.
    pub(super) allow_8x8: bool,
    /// The coded 16x16 / 8x8 leaves search the non-directional luma modes (4:2:0),
    /// so the probe's candidates may too; DC-only leaf coders must keep it off.
    pub(super) leaf_modes: bool,
    /// The coded 8x8 leaves also search those modes (4:2:0 only so far); else the
    /// probe's 8x8 candidates stay DC like the leaves they stand for.
    pub(super) leaf_modes8: bool,
    /// Weight on the 8x8 candidates' chroma SSE in the 16x16 split decision.
    pub(super) split8_chroma_weight: f32,
}

/// 4:2:0 chroma state for the partition probe: each candidate leaf size is also
/// charged its chroma cost (DC prediction + projected residual, U+V SSE and
/// heuristic coefficient bits), so the luma-driven tree does not pick shapes whose
/// chroma is expensive. Chroma recon is written as the probe goes (neighbours for
/// later blocks) and restored afterwards.
pub(super) struct ChromaProbe<'a> {
    pub(super) src_u: &'a [f32],
    pub(super) src_v: &'a [f32],
    pub(super) rec_u: &'a mut [f32],
    pub(super) rec_v: &'a mut [f32],
    pub(super) stride: usize,
    /// Bases for 32x32, 16x16, 8x8 and 4x4 chroma TUs (the ones the leaves code with).
    pub(super) bases: [&'a Basis; 4],
    /// 4:4:4: chroma is full resolution. The probe still addresses it in 4:2:0
    /// units (luma / 2); positions and sizes are doubled here, and the whole-64
    /// block is coded as four 32x32 TUs.
    pub(super) full: bool,
}

impl ChromaProbe<'_> {
    /// Code an `n`x`n` chroma block (both planes) at chroma (cy, cx) into the recon and
    /// return its (SSE, bits).
    fn code(&mut self, cy: usize, cx: usize, n: usize, quant: &LumaQuantSpec<'_>) -> (f32, f32) {
        let _qm = crate::av2::qm::ChromaScope::new();
        if !self.full {
            return self.code_tu(cy, cx, n, quant);
        }
        let (cy, cx, n) = (2 * cy, 2 * cx, 2 * n);
        if n <= 32 {
            return self.code_tu(cy, cx, n, quant);
        }
        let (mut d, mut r) = (0.0, 0.0);
        for (dy, dx) in [(0, 0), (0, 32), (32, 0), (32, 32)] {
            let (dd, rr) = self.code_tu(cy + dy, cx + dx, 32, quant);
            d += dd;
            r += rr;
        }
        (d, r)
    }

    /// Code one `n`x`n` chroma TU at plane position (cy, cx).
    fn code_tu(&mut self, cy: usize, cx: usize, n: usize, quant: &LumaQuantSpec<'_>) -> (f32, f32) {
        let (basis, scan): (&Basis, &[u16]) = match n {
            32 => (self.bases[0], &tables::SCAN),
            16 => (self.bases[1], &SCAN16),
            8 => (self.bases[2], &SCAN8X8),
            _ => (self.bases[3], &crate::av2::coder::SCAN4X4_LOSSY_PACKED),
        };
        let scale = basis.qstep as f32 / quant.qstep as f32;
        let weight = crate::av2::rd::knob("CW");
        let (mut sse, mut bits) = (0.0, 0.0);
        for (src, rec) in [
            (self.src_u, &mut *self.rec_u),
            (self.src_v, &mut *self.rec_v),
        ] {
            let pred = dc_pred_rect(
                rec,
                self.stride,
                cy,
                cx,
                n,
                n,
                quant.neutral,
                quant.bit_depth,
            );
            let resid = aq::scale_resid(
                &get_residual_rect(src, self.stride, cy, cx, n, n, pred),
                scale,
            );
            let lev = basis.project_scan(&resid, 0.0, scan);
            let out =
                itx422::reconstruct_chroma(pred, &lev, quant.qstep, scan, n, n, quant.bit_depth);
            put_block_rect(rec, self.stride, cy, cx, n, n, &out);
            sse += pixel_sse_rounded_block(
                src,
                self.stride,
                cy,
                cx,
                &rec[cy * self.stride + cx..],
                self.stride,
                n,
                n,
            );
            bits += coeff_rate_f32(&lev);
        }
        (sse * weight, bits * weight)
    }

    /// Plane position and size of a probe chroma block given in 4:2:0 units.
    fn plane_rect(&self, cy: usize, cx: usize, n: usize) -> (usize, usize, usize) {
        if self.full {
            (2 * cy, 2 * cx, 2 * n)
        } else {
            (cy, cx, n)
        }
    }

    fn save(&self, cy: usize, cx: usize, n: usize) -> [Vec<f32>; 2] {
        let (cy, cx, n) = self.plane_rect(cy, cx, n);
        let grab = |rec: &[f32]| {
            rect_rows(rec, self.stride, cy, cx, n, n)
                .flatten()
                .copied()
                .collect()
        };
        [grab(self.rec_u), grab(self.rec_v)]
    }

    fn restore(&mut self, cy: usize, cx: usize, n: usize, saved: &[Vec<f32>; 2]) {
        let (cy, cx, n) = self.plane_rect(cy, cx, n);
        put_block_rect(self.rec_u, self.stride, cy, cx, n, n, &saved[0]);
        put_block_rect(self.rec_v, self.stride, cy, cx, n, n, &saved[1]);
    }
}

/// Probe-side best non-directional luma mode for an `n`x`n` DCT leaf: returns the
/// reconstruction (written into `recy`), its SSE and its bits (coefficients plus a
/// flat mode cost). DC keeps the scalar predictor the coded leaves use. Within-SB
/// top-right / bottom-left are treated as unavailable (an estimate only).
#[allow(clippy::too_many_arguments)]
fn probe_square_luma(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    frame: (usize, usize),
    y: usize,
    x: usize,
    n: usize,
    basis: &Basis,
    quant: &LumaQuantSpec<'_>,
    lambda: f32,
    rate_scale: f32,
    exact_modes: bool,
    leaf_modes: bool,
) -> (f32, f32) {
    let pw = source.stride;
    // Exact luma-mode bits from the live CDFs (y_ctx 0: the coded path has no
    // directional neighbours), else the flat 0 (DC) / 6 bits.
    let flat = crate::av2::rd::knob_or(if n == 16 { "MB16" } else { "MB8" }, 6.0);
    let mode_bits = |m: usize| -> f32 {
        if exact_modes && crate::av2::rd::knob_or("EXACTMODE", 0.0) > 0.0 {
            crate::av2::entropy::RangeEncoder::y_mode_nondir_bits(0, m)
        } else if m == 0 {
            0.0
        } else {
            flat
        }
    };
    let scan: &[u16] = if n == 16 { &SCAN16 } else { &SCAN8X8 };
    let scale = basis.qstep as f32 / quant.qstep as f32;
    debug_assert!(n <= 16);
    let mut src_buf = [0f32; 256];
    for (d, &v) in src_buf
        .iter_mut()
        .zip(rect_rows(source.plane, pw, y, x, n, n).flatten())
    {
        *d = v;
    }
    let src = &src_buf[..n * n];
    let dc = dc_pred_rect(recy, pw, y, x, n, n, quant.neutral, quant.bit_depth);
    // One candidate: levels + reconstruction into reused buffers.
    let code = |pred: &[i32], lev: &mut Vec<f32>, rec: &mut Vec<f32>| {
        let mut resid = [0f32; 256];
        for ((r, &s), &p) in resid.iter_mut().zip(src).zip(pred) {
            *r = (s - p as f32) * scale;
        }
        basis.project_scan_into(&resid[..n * n], 0.0, scan, lev);
        rec.resize(n * n, 0.0);
        itx422::reconstruct_chroma_pred_tx_into(
            pred,
            lev,
            quant.qstep,
            scan,
            n,
            n,
            quant.bit_depth,
            0,
            rec,
        );
    };
    let (mut cand_lev, mut cand_rec) = (Vec::new(), Vec::new());
    let sse = |rec: &[f32]| -> f32 { crate::av2::helpers::dist_rounded(src, n, rec, n, n, n) };
    // DC keeps the reconstruction the probe always used (luma16 / scalar-pred paths).
    let lev = basis.project_scan(
        &get_residual_rect(source.plane, pw, y, x, n, n, dc)
            .iter()
            .map(|&v| v * scale)
            .collect::<Vec<f32>>(),
        0.0,
        scan,
    );
    let rec: Vec<f32> = if n == 16 {
        itx422::reconstruct_luma16(&[dc; 256], &lev, quant.qstep, &SCAN16, quant.bit_depth).to_vec()
    } else {
        itx422::reconstruct_chroma(dc, &lev, quant.qstep, scan, n, n, quant.bit_depth)
    };
    let d0 = sse(&rec);
    let mut best = (rec, d0, 0.0f32);
    let mut best_rate = coeff_rate_f32(&lev) * rate_scale + mode_bits(0);
    let mut best_cost = crate::av2::rd::rd_cost(best.1, best_rate, lambda);
    if leaf_modes && crate::av2::rd::knob("PMODES") > 0.0 {
        let block = SquareLumaBlock {
            stride: pw,
            frame_width: frame.0,
            frame_height: frame.1,
            y,
            x,
            size: n,
        };
        let no_coded = [0u8; 256];
        let mut pbuf = [0f32; 256];
        let mut refs = SquareRefCache::default();
        // SATD of every screened (mode, delta), and the non-directional predictions
        // made while ranking, so neither is computed twice.
        let mut dir_scores: Vec<((usize, i32), u64)> = Vec::with_capacity(24);
        let mut nondir_pred = [[0f32; 256]; 4];
        let mut have_nondir = [false; 4];
        let mut cands: Vec<SquareLumaMode> = (1..=4usize).map(SquareLumaMode::nondir).collect();
        // 16x16 and 8x8 leaves also code directional modes: the SATD-best few join
        // the RD.
        let dir_keep =
            crate::av2::rd::knob_or(if n == 8 { "PDIRK8" } else { "PDIR" }, 3.0) as usize;
        let dir_size = n == 16 || (n == 8 && crate::av2::rd::knob_or("PDIR8", 1.0) > 0.0);
        if dir_size && dir_keep > 0 && quant.speed.try_directional() {
            let dir_lm = |mode, angle_delta| SquareLumaMode {
                mode,
                angle_delta,
                smooth_neighbour: false,
            };
            let refine =
                crate::av2::rd::knob_or(if n == 8 { "PDREF8" } else { "PDREF" }, 2.0) as usize;
            let picked =
                screen_directional(dir_keep, quant.speed.try_angle_deltas(), refine, |m, d| {
                    let p = &mut pbuf[..n * n];
                    predict_luma_square_cached(
                        recy,
                        &block,
                        &no_coded,
                        dir_lm(m, d),
                        quant.neutral,
                        quant.bit_depth,
                        &mut refs,
                        p,
                    );
                    let sc = crate::av2::metrics::satd_f32(src, n, p, n, n, n);
                    dir_scores.push(((m, d), sc));
                    sc
                });
            if n == 8 {
                PROBE_DIR8_LAST.with(|c| c.set(dir8_pack(&picked)));
            }
            cands.extend(picked.into_iter().map(|(m, d)| dir_lm(m, d)));
        }
        // Unified SATD prune: rank every candidate by its prediction SATD (+ mode
        // bits) and fully code only the best `keep`, or those within `ratio` of
        // the best.
        let keep = crate::av2::rd::knob_or(
            "PRK",
            if quant.prune & PRUNE_PROBE != 0 {
                2.0
            } else {
                0.0
            },
        ) as usize;
        let ratio = crate::av2::rd::knob_or("PRT", 0.0);
        if keep > 0 || ratio > 0.0 {
            let wbits = crate::av2::rd::knob_or("PRKL", 0.0) * lambda.sqrt();
            let mut ranked: Vec<(f32, SquareLumaMode)> = cands
                .iter()
                .map(|&lm| {
                    let bits = if lm.mode >= 5 {
                        9.0
                    } else {
                        mode_bits(lm.mode)
                    };
                    let screened = dir_scores
                        .iter()
                        .find(|&&(md, _)| lm.mode >= 5 && md == (lm.mode, lm.angle_delta));
                    let satd = match screened {
                        Some(&(_, sc)) => sc,
                        None => {
                            let pf = if lm.mode <= 4 {
                                have_nondir[lm.mode - 1] = true;
                                &mut nondir_pred[lm.mode - 1][..n * n]
                            } else {
                                &mut pbuf[..n * n]
                            };
                            predict_luma_square_cached(
                                recy,
                                &block,
                                &no_coded,
                                lm,
                                quant.neutral,
                                quant.bit_depth,
                                &mut refs,
                                pf,
                            );
                            crate::av2::metrics::satd_f32(src, n, pf, n, n, n)
                        }
                    } as f32;
                    (satd + wbits * bits, lm)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.total_cmp(&b.0));
            let best_s = ranked.first().map_or(0.0, |r| r.0);
            cands = ranked
                .into_iter()
                .enumerate()
                .filter(|&(i, (sc, _))| {
                    (keep == 0 || i < keep) && (ratio <= 0.0 || sc <= best_s * ratio)
                })
                .map(|(_, (_, lm))| lm)
                .collect();
        }
        for lm in cands {
            let m = lm.mode;
            let pf: &[f32] = if (1..=4).contains(&m) && have_nondir[m - 1] {
                &nondir_pred[m - 1][..n * n]
            } else {
                let pf = &mut pbuf[..n * n];
                predict_luma_square_cached(
                    recy,
                    &block,
                    &no_coded,
                    lm,
                    quant.neutral,
                    quant.bit_depth,
                    &mut refs,
                    pf,
                );
                pf
            };
            let mut p = [0i32; 256];
            for (d, &v) in p.iter_mut().zip(pf.iter()) {
                *d = v as i32;
            }
            code(&p[..n * n], &mut cand_lev, &mut cand_rec);
            let d = sse(&cand_rec);
            let bits = if m >= 5 {
                crate::av2::rd::knob_or(if n == 8 { "MBDIR8" } else { "MBDIR" }, 9.0)
            } else {
                mode_bits(m)
            };
            let r = coeff_rate_f32(&cand_lev) * rate_scale + bits;
            let c = crate::av2::rd::rd_cost(d, r, lambda);
            if c < best_cost {
                best_cost = c;
                std::mem::swap(&mut best.0, &mut cand_rec);
                best.1 = d;
                best_rate = r;
            }
        }
    }
    // Screen content: the palette candidate (DCT residual, palette syntax bits).
    if let Some(pal) = luma_palette(source, frame, y, x, n, n, quant) {
        let p: Vec<i32> = pal.pred().iter().map(|&v| v as i32).collect();
        code(&p, &mut cand_lev, &mut cand_rec);
        let d = sse(&cand_rec);
        let r = coeff_rate_f32(&cand_lev) * rate_scale + pal.bits;
        let c = crate::av2::rd::rd_cost(d, r, lambda);
        if c < best_cost {
            best = (cand_rec, d, 0.0);
            best_rate = r;
        }
    }
    put_block_rect(recy, pw, y, x, n, n, &best.0);
    let d = pixel_sse_rounded_block(source.plane, pw, y, x, &recy[y * pw + x..], pw, n, n);
    let _ = best.1;
    (d, best_rate)
}

pub(super) fn choose_luma_64x64_partition(
    recy: &mut [f32],
    source: &LumaSource<'_>,
    block: &LumaFrameBlock,
    grid: &LumaGridBlock,
    search: &LumaPartitionSearch<'_>,
    mut chroma: Option<&mut ChromaProbe<'_>>,
    pricer: Option<&crate::av2::partition::PartitionPricer>,
) -> LumaPartitionDecision {
    let quant = &search.quant;
    let pw = source.stride;
    // Flat split-signalling cost. With chroma priced the probe needs none (4:2:0:
    // 5 -> 0 bits is -0.43% SS2 luma-only, -1.26% with chroma); luma-only callers
    // keep 5 (0 bits costs 4:0:0 +0.83%).
    let split_bits = crate::av2::rd::knob_or("SPLITB", if chroma.is_some() { 0.0 } else { 5.0 });
    // Exact partition signalling (when a pricer is given): each decision is charged
    // the marginal bits of its subtree given the decisions made so far (`t32`/`t16`);
    // otherwise the flat `split_bits` per split.
    let pricer = pricer.filter(|_| crate::av2::rd::knob_or("EXACTPART", 0.0) > 0.0);
    let (mut t32, mut t16) = ([false; 4], [false; 16]);
    let mut dir8 = [[0u32; 32]; 2];
    let (r32_scale, r16_scale) = (crate::av2::rd::knob("R32"), crate::av2::rd::knob("R16"));
    let mut saved = [0f32; 64 * 64];
    for (dst, src) in saved
        .as_chunks_mut::<64>()
        .0
        .iter_mut()
        .zip(rect_rows(recy, pw, block.y, block.x, 64, 64))
    {
        dst.copy_from_slice(src);
    }
    let restore = |recy: &mut [f32]| {
        for (dst, src) in
            rect_rows_mut(recy, pw, block.y, block.x, 64, 64).zip(saved.as_chunks::<64>().0.iter())
        {
            dst.copy_from_slice(src);
        }
    };

    let (_, _, _, whole_rate) = encode_luma_sb(recy, source, block, quant, &search.sb);
    let whole_sse = pixel_sse_rounded_block(
        source.plane,
        pw,
        block.y,
        block.x,
        &recy[block.y * pw + block.x..],
        pw,
        64,
        64,
    );
    restore(recy);
    let (cy0, cx0) = (block.y / 2, block.x / 2);
    let chroma_saved = chroma.as_deref().map(|c| c.save(cy0, cx0, 32));
    let (mut whole_sse, mut whole_rate) = (whole_sse, whole_rate);
    if let Some(c) = chroma.as_deref_mut() {
        let (d, r) = c.code(cy0, cx0, 32, quant);
        whole_sse += d;
        whole_rate += r;
        c.restore(cy0, cx0, 32, chroma_saved.as_ref().unwrap());
    }

    let mut split_rate = 0.0f32;
    let mut split_chroma_sse = 0.0f32;
    let mut split32 = [false; 4];
    let mut split16 = [false; 16];
    // Coded-mi mask of the probe's own decisions (whole quadrants, in coding order)
    // for the 32x32 leaves' top-right / bottom-left availability.
    let mut probe_coded = [0u8; 256];
    for (child, (dy, dx)) in [(0usize, 0usize), (0, 32), (32, 0), (32, 32)]
        .into_iter()
        .enumerate()
    {
        let mut child_saved = [0f32; 32 * 32];
        for (dst, src) in child_saved
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(rect_rows(recy, pw, block.y + dy, block.x + dx, 32, 32))
        {
            dst.copy_from_slice(src);
        }
        let restore_child = |recy: &mut [f32]| {
            for (dst, src) in rect_rows_mut(recy, pw, block.y + dy, block.x + dx, 32, 32)
                .zip(child_saved.as_chunks::<32>().0.iter())
            {
                dst.copy_from_slice(src);
            }
        };
        let leaf = LumaGridBlock {
            mi_cols: grid.mi_cols,
            mi_rows: grid.mi_rows,
            y: block.y + dy,
            x: block.x + dx,
        };
        // The coded 32x32 leaf searches directional modes when the leaves are
        // equipped; the probe follows with its own coded-quadrant mask.
        let dir32 = (search.leaf_modes && crate::av2::rd::knob_or("PDIR32", 1.0) > 0.0).then_some(
            LeafDirSearch {
                refine: crate::av2::rd::knob_or("PDREF", 2.0) as usize,
                coded: &probe_coded,
                frame_width: block.frame_width,
                frame_height: block.frame_height,
            },
        );
        let (tu, mode, _) = encode_luma_leaf_s32x32(recy, source, &leaf, quant, dir32, None);
        let rate32 = coeff_tu_rate_proxy_f32(&tu);
        let rate32 = rate32 * r32_scale
            + if mode == palette::PAL_MODE {
                let frame = (block.frame_width, block.frame_height);
                luma_palette(source, frame, leaf.y, leaf.x, 32, 32, quant).map_or(0.0, |p| p.bits)
            } else if mode >= 5 {
                crate::av2::rd::knob_or("MBDIR", 9.0)
            } else if mode != 0 {
                6.0
            } else {
                0.0
            };
        // Cache the 32x32 leaf reconstruction so the not-split branch can restore it
        // instead of re-encoding the identical leaf (rate32 already is its cost).
        let mut leaf32_recon = [0f32; 32 * 32];
        for (dst, src) in leaf32_recon
            .as_chunks_mut::<32>()
            .0
            .iter_mut()
            .zip(rect_rows(recy, pw, block.y + dy, block.x + dx, 32, 32))
        {
            dst.copy_from_slice(src);
        }
        let sse32 = pixel_sse_rounded_block(
            source.plane,
            pw,
            block.y + dy,
            block.x + dx,
            &recy[(block.y + dy) * pw + block.x + dx..],
            pw,
            32,
            32,
        );
        restore_child(recy);
        let (ccy, ccx) = ((block.y + dy) / 2, (block.x + dx) / 2);
        let (mut sse32, mut rate32) = (sse32, rate32);
        let mut chroma32_sse = 0.0f32;
        let mut chroma32_recon = None;
        if let Some(c) = chroma.as_deref_mut() {
            let before = c.save(ccy, ccx, 16);
            let (d, r) = c.code(ccy, ccx, 16, quant);
            chroma32_sse = d;
            sse32 += d;
            rate32 += r;
            chroma32_recon = Some(c.save(ccy, ccx, 16));
            c.restore(ccy, ccx, 16, &before);
        }

        let mut rate16 = 0.0;
        let mut chroma16_sse = 0.0f32;
        for (sy, sx) in [(0usize, 0usize), (0, 16), (16, 0), (16, 16)] {
            let (y, x) = (block.y + dy + sy, block.x + dx + sx);
            let mut saved16 = [0f32; 256];
            for (dst, src) in saved16
                .as_chunks_mut::<16>()
                .0
                .iter_mut()
                .zip(rect_rows(recy, pw, y, x, 16, 16))
            {
                dst.copy_from_slice(src);
            }
            let restore16 = |recy: &mut [f32]| {
                for (dst, src) in
                    rect_rows_mut(recy, pw, y, x, 16, 16).zip(saved16.as_chunks::<16>().0.iter())
                {
                    dst.copy_from_slice(src);
                }
            };
            let lambda16 = luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda);
            let (sse16c, rate16c) = probe_square_luma(
                recy,
                source,
                (block.frame_width, block.frame_height),
                y,
                x,
                16,
                search.basis16,
                quant,
                lambda16,
                r16_scale,
                pricer.is_some(),
                search.leaf_modes,
            );
            let rec: Vec<f32> = rect_rows(recy, pw, y, x, 16, 16)
                .flatten()
                .copied()
                .collect();
            let mut candidate16_rate = rate16c;
            let mut candidate16_sse = sse16c;
            restore16(recy);
            let (c16y, c16x) = (y / 2, x / 2);
            let mut chroma16 = (0.0f32, None, None);
            if let Some(c) = chroma.as_deref_mut() {
                let before = c.save(c16y, c16x, 8);
                let (d, r) = c.code(c16y, c16x, 8, quant);
                candidate16_rate += r;
                chroma16 = (d, Some(c.save(c16y, c16x, 8)), Some(before));
                c.restore(c16y, c16x, 8, chroma16.2.as_ref().unwrap());
            }
            candidate16_sse += chroma16.0;

            let index16 = ((dy + sy) / 16) * 4 + (dx + sx) / 16;
            let mut rate8 = 0.0;
            let mut chroma8_sse = 0.0f32;
            // Early out: a 16x16 candidate already cheaper than `P8J` bits of lambda
            // leaves its 8x8 split nothing to win.
            let p8j = crate::av2::rd::knob_or("P8J", 0.0);
            let cheap16 = p8j > 0.0 && sse16c + lambda16 * rate16c < p8j * lambda16;
            if (search.allow_8x8 || crate::av2::rd::knob_or("ALLOW8", 0.0) > 0.0) && !cheap16 {
                let lambda8 = luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda);
                for (ey, ex) in [(0usize, 0usize), (0, 8), (8, 0), (8, 8)] {
                    PROBE_DIR8_LAST.with(|c| c.set(0));
                    let (_, r8) = probe_square_luma(
                        recy,
                        source,
                        (block.frame_width, block.frame_height),
                        y + ey,
                        x + ex,
                        8,
                        search.basis8,
                        quant,
                        lambda8,
                        1.0,
                        pricer.is_some(),
                        search.leaf_modes8,
                    );
                    let i = ((y + ey - block.y) / 8) * 8 + (x + ex - block.x) / 8;
                    dir8[i / 32][i % 32] = PROBE_DIR8_LAST.with(|c| c.get());
                    rate8 += r8;
                }
                let mut sse8 = pixel_sse_rounded_block(
                    source.plane,
                    pw,
                    y,
                    x,
                    &recy[y * pw + x..],
                    pw,
                    16,
                    16,
                );
                if let Some(c) = chroma.as_deref_mut() {
                    for (ey, ex) in [(0usize, 0usize), (0, 4), (4, 0), (4, 4)] {
                        let (d, r) = c.code(c16y + ey, c16x + ex, 4, quant);
                        chroma8_sse += d;
                        rate8 += r;
                    }
                }
                sse8 += chroma8_sse * search.split8_chroma_weight;
                let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda);
                let d16 = match pricer {
                    Some(p) => {
                        let mut s32 = t32;
                        s32[child] = true;
                        let mut s16 = t16;
                        s16[index16] = true;
                        let with = p.split_bits(s32, s16);
                        s16[index16] = false;
                        with - p.split_bits(s32, s16)
                    }
                    None => split_bits,
                };
                split16[index16] = crate::av2::rd::rd_cost(sse8, rate8 + d16, lambda)
                    < crate::av2::rd::rd_cost(candidate16_sse, candidate16_rate, lambda);
                t16[index16] = split16[index16];
            }
            if split16[index16] {
                rate16 += rate8 + if pricer.is_some() { 0.0 } else { split_bits };
                chroma16_sse += chroma8_sse;
            } else {
                restore16(recy);
                put_block_rect(recy, pw, y, x, 16, 16, &rec);
                rate16 += candidate16_rate;
                if let (Some(c), Some(rec16)) = (chroma.as_deref_mut(), chroma16.1.as_ref()) {
                    c.restore(c16y, c16x, 8, rec16);
                }
                chroma16_sse += chroma16.0;
            }
        }
        let sse16 = pixel_sse_rounded_block(
            source.plane,
            pw,
            block.y + dy,
            block.x + dx,
            &recy[(block.y + dy) * pw + block.x + dx..],
            pw,
            32,
            32,
        ) + chroma16_sse;
        let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda);
        let d32 = match pricer {
            Some(p) => {
                let mut s32 = t32;
                s32[child] = true;
                let with = p.split_bits(s32, t16);
                s32[child] = false;
                with - p.split_bits(s32, t16)
            }
            None => split_bits,
        };
        split32[child] = search.allow_16x16
            && crate::av2::rd::rd_cost(sse16, rate16 + d32, lambda)
                < crate::av2::rd::rd_cost(sse32, rate32, lambda);
        t32[child] = split32[child];
        if !split32[child] {
            for (k, t) in t16.iter_mut().enumerate() {
                if (k / 4) / 2 == child / 2 && (k % 4) / 2 == child % 2 {
                    *t = false;
                }
            }
        }
        if split32[child] {
            // With a pricer the tree's signalling is added once at the 64 level below.
            split_rate += rate16 + if pricer.is_some() { 0.0 } else { split_bits };
            split_chroma_sse += chroma16_sse;
        } else {
            // Not split: restore the cached 32x32 leaf recon (no re-encode).
            put_block(recy, pw, block.y + dy, block.x + dx, 32, &leaf32_recon);
            split_rate += rate32;
            if let (Some(c), Some(rec32)) = (chroma.as_deref_mut(), chroma32_recon.as_ref()) {
                c.restore(ccy, ccx, 16, rec32);
            }
            split_chroma_sse += chroma32_sse;
        }
        for r in dy / 4..dy / 4 + 8 {
            probe_coded[r * 16 + dx / 4..r * 16 + dx / 4 + 8].fill(1);
        }
    }
    let split_sse = split_chroma_sse
        + pixel_sse_rounded_block(
            source.plane,
            pw,
            block.y,
            block.x,
            &recy[block.y * pw + block.x..],
            pw,
            64,
            64,
        );
    restore(recy);
    if let (Some(c), Some(saved)) = (chroma, chroma_saved.as_ref()) {
        c.restore(cy0, cx0, 32, saved);
    }

    // With a pricer the children carried only their own marginal bits; add the 64
    // split-vs-whole difference (the whole side's own do_split symbol included).
    split_rate += match pricer {
        Some(p) => p.split_bits(t32, t16) - p.whole_bits(),
        None => split_bits,
    };
    let lambda = luma_mode_lambda(quant.rdoq_lambda, quant.qstep, quant.pixel_lambda);
    LumaPartitionDecision {
        split64: crate::av2::rd::rd_cost(split_sse, split_rate, lambda)
            < crate::av2::rd::rd_cost(whole_sse, whole_rate, lambda),
        split32,
        split16,
        dir8,
        ..Default::default()
    }
}
