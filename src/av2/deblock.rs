/*
 * Copyright (c) Radzivon Bartoshyk 7/2026. All rights reserved.
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

// ---- constants (avm_dsp/loopfilter.h) ----
const DF_SHIFT: i32 = 8; // rounding shift for the tap sum is 3 + DF_SHIFT = 11
const ROUND_BITS: i32 = 3 + DF_SHIFT; // 11
const DF_6_THRESH: i32 = 4;
const DF_8_THRESH: i32 = 3;
const FILT_8_THRESH_SHIFT: i32 = 3;
const DF_Q_THRESH_SHIFT: i32 = 4;
const MAX_DBL_FLT_LEN: i32 = 8;
const DF_DELTA_SCALE: i32 = 8;

/// Per-tap taper weights, indexed by `width - 1`.
static W_MULT: [i32; 8] = [85, 51, 37, 28, 23, 20, 17, 15];
/// Clamp scale for `delta_m2`, indexed by `width - 1`.
static Q_THRESH_MULTS: [i32; 8] = [32, 25, 19, 19, 18, 18, 17, 17];
/// Long-tap transition multipliers, indexed `[dist - 4]` (only 0,2,4 used).
static Q_FIRST: [i32; 5] = [45, 43, 40, 35, 32];

static MAX_SIDE_TABLE: i32 = 296;
/// `side_thresholds[296]` (av2_loopfilter.c). Indices 0..=56 are all -16.
static SIDE_THRESHOLDS: [i16; 296] = [
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16, -16,
    -14, -13, -11, -10, -9, -7, -6, -4, -3, -2, 0, 0, 2, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 19,
    20, 22, 23, 24, 26, 27, 29, 30, 32, 33, 34, 36, 37, 39, 40, 42, 43, 44, 46, 47, 49, 50, 51, 53,
    54, 56, 57, 59, 60, 61, 63, 64, 66, 67, 69, 70, 71, 73, 74, 76, 77, 78, 80, 81, 83, 84, 86, 87,
    88, 90, 91, 93, 94, 96, 101, 111, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230,
    240, 249, 259, 269, 279, 289, 299, 309, 319, 329, 339, 349, 359, 368, 378, 388, 398, 408, 418,
    428, 438, 448, 458, 468, 478, 488, 497, 507, 517, 527, 537, 547, 557, 567, 577, 587, 597, 607,
    616, 626, 636, 646, 656, 666, 676, 686, 696, 706, 716, 726, 736, 745, 755, 765, 775, 785, 795,
    805, 815, 825, 835, 845, 855, 864, 874, 884, 894, 904, 914, 924, 934, 944, 954, 964, 974, 984,
    993, 1003, 1013, 1023, 1033, 1043, 1053, 1063, 1073, 1083, 1093, 1103, 1112, 1122, 1132, 1142,
    1152, 1162, 1172, 1182, 1192, 1202, 1212, 1222, 1232, 1241, 1251, 1261, 1271, 1281, 1291, 1301,
    1311, 1321, 1331, 1341, 1351, 1360, 1370, 1380, 1390, 1400, 1410, 1420, 1430, 1440, 1450, 1460,
    1470, 1480, 1489, 1499, 1509, 1519, 1529, 1539, 1549, 1559, 1569, 1579, 1589, 1599, 1608, 1618,
    1628, 1638, 1648, 1658, 1668, 1678,
];

/// `av2_ac_quant_QTX(qindex, 0, 0, bit_depth)` — the transform-domain AC quant
/// step. The QTX lookup values are bit-depth independent; only the qindex clamp
/// (`MAXQ`) widens with bit depth.
#[inline]
fn ac_quant_qtx(qindex: i32, bit_depth: u32) -> i32 {
    if qindex == 0 {
        return 64; // ac_qlookup_QTX[0]
    }
    let maxq = match bit_depth {
        10 => 303,
        12 => 351,
        _ => 255,
    };
    let q = qindex.clamp(1, maxq) as u32;
    crate::av2::quant::qstep(q) as i32
}

/// `df_quant_from_qindex`: `ROUND_POWER_OF_TWO(ac_quant, QUANT_TABLE_BITS) >> 6`.
#[inline]
fn df_quant_from_qindex(qindex: i32, bit_depth: u32) -> i32 {
    let ac = ac_quant_qtx(qindex, bit_depth);
    // QUANT_TABLE_BITS = 3 -> ROUND_POWER_OF_TWO(ac, 3) = (ac + 4) >> 3
    ((ac + 4) >> 3) >> 6
}

/// `df_side_from_qindex`.
#[inline]
fn df_side_from_qindex(qindex: i32, bit_depth: u32) -> i32 {
    let bd = bit_depth as i32;
    let q_ind = (qindex - 24 * (bd - 8)).clamp(0, MAX_SIDE_TABLE - 1);
    let side = SIDE_THRESHOLDS[q_ind as usize] as i32;
    ((side + (1 << (12 - bd))).max(0)) >> (13 - bd)
}

/// Rounded arithmetic shift matching C `ROUND_POWER_OF_TWO(v, 11)` on a signed
/// intermediate: `(v + 1024) >> 11` with arithmetic (sign-preserving) shift.
#[inline]
fn round_pow2_11(v: i32) -> i32 {
    (v + (1 << (ROUND_BITS - 1))) >> ROUND_BITS
}

/// Per-plane per-direction filter thresholds, combining the two blocks that meet
/// at an edge (avm `set_lpf_parameters`). Returns `(q_threshold, side_threshold)`.
#[inline]
fn combine_thresholds(
    curr_qindex: i32,
    prev_qindex: i32,
    delta_q: i32,
    delta_side: i32,
    bit_depth: u32,
) -> (i32, i32) {
    let curr_q = df_quant_from_qindex(curr_qindex + delta_q * DF_DELTA_SCALE, bit_depth);
    let pv_q = df_quant_from_qindex(prev_qindex + delta_q * DF_DELTA_SCALE, bit_depth);
    let curr_s = df_side_from_qindex(curr_qindex + delta_side * DF_DELTA_SCALE, bit_depth);
    let pv_s = df_side_from_qindex(prev_qindex + delta_side * DF_DELTA_SCALE, bit_depth);
    let q_threshold = if curr_q != 0 && pv_q != 0 {
        (curr_q + pv_q + 1) >> 1
    } else if curr_q != 0 {
        curr_q
    } else {
        pv_q
    };
    let side_threshold = if curr_s != 0 && pv_s != 0 {
        (curr_s + pv_s + 1) >> 1
    } else if curr_s != 0 {
        curr_s
    } else {
        pv_s
    };
    (q_threshold, side_threshold)
}

/// Filter-length ladder (`set_lpf_parameters`). `min_ts` is the minimum square TX
/// size index in the filter direction (0=4,1=8,2=16,3=32,4=64) across the two
/// sides. `edge_asym` is `horz_superblock_edge || vert_tile_edge`. Returns
/// `(filter_length_neg, filter_length_pos)`.
#[inline]
fn filter_lengths(min_ts: i32, is_chroma: bool, edge_asym: bool) -> (i32, i32) {
    match min_ts {
        0 => (4, 4),
        1 => {
            if is_chroma {
                if edge_asym { (6, 8) } else { (8, 8) }
            } else {
                (8, 8)
            }
        }
        2 => {
            if is_chroma {
                if edge_asym { (6, 10) } else { (10, 10) }
            } else {
                (14, 14)
            }
        }
        _ => {
            if is_chroma {
                if edge_asym { (6, 10) } else { (10, 10) }
            } else if edge_asym {
                (14, 18)
            } else {
                (18, 18)
            }
        }
    }
}

/// One line's samples across an edge, read into p[0..=8]/q[0..=8]. `base` is the
/// byte offset of q0; `perp` is the signed stride from q0 toward q1 (p goes the
/// other way).
struct Line {
    p: [i32; 9],
    q: [i32; 9],
}

#[inline]
fn read_line(buf: &[f32], base: isize, perp: isize) -> Line {
    // The reference recon buffer is edge-extended; ours is tight (exactly the plane),
    // so clamp indices into range (edge replication). Only the samples the filter length
    // actually consumes matter, and near a frame edge the border-clamp keeps that short,
    // so the clamped deep reads never affect the decision.
    let len = buf.len() as isize;
    let get = |off: isize| -> i32 { buf[off.clamp(0, len - 1) as usize] as i32 };
    let mut p = [0i32; 9];
    let mut q = [0i32; 9];
    for k in 0..9 {
        q[k] = get(base + perp * k as isize);
        p[k] = get(base - perp * (k as isize + 1));
    }
    Line { p, q }
}

#[inline]
fn sd(a: i32, b: i32, c: i32) -> i32 {
    (a - 2 * b + c).abs()
}

/// The per-4-line filter-width decision (`filt_choice_highbd`). `l0`/`l3` are the
/// first and last lines of the segment. Returns the number of positive-side
/// samples to filter.
#[allow(clippy::too_many_arguments)]
fn filt_choice(
    l0: &Line,
    l3: &Line,
    max_filt_neg: i32,
    max_filt_pos: i32,
    q_thresh: i32,
    side_thresh: i32,
) -> i32 {
    if q_thresh == 0 || side_thresh == 0 {
        return 0;
    }
    let max_samples_neg = if max_filt_neg == 0 {
        0
    } else {
        max_filt_neg / 2 - 1
    };
    let max_samples_pos = max_filt_pos / 2 - 1;
    if max_samples_pos < 1 || max_samples_pos < max_samples_neg {
        return 0;
    }
    let avg = |a: i32, b: i32| (a + b + 1) >> 1;
    let (p0, q0) = (&l0.p, &l0.q);
    let (p3, q3) = (&l3.p, &l3.q);

    // 1-sample test.
    let dm2 = avg(sd(p0[2], p0[1], p0[0]), sd(p3[2], p3[1], p3[0]));
    let d1 = avg(sd(q0[0], q0[1], q0[2]), sd(q3[0], q3[1], q3[2]));
    if dm2 > side_thresh || d1 > side_thresh {
        return 0;
    }
    if max_samples_pos == 1 {
        return 1;
    }

    let dm1 = avg(sd(p0[1], p0[0], q0[0]), sd(p3[1], p3[0], q3[0]));
    let d0 = avg(sd(p0[0], q0[0], q0[1]), sd(p3[0], q3[0], q3[1]));

    // 2-sample test.
    let side2 = side_thresh >> 2;
    let mut mask = dm2 > side2 || d1 > side2;
    if dm1 + d0 > q_thresh * DF_6_THRESH {
        mask = true;
    }
    if mask {
        return 1;
    }

    // 3-sample test.
    let side3 = side_thresh >> FILT_8_THRESH_SHIFT;
    let mut mask = dm2 > side3 || d1 > side3;
    if dm1 + d0 > q_thresh * DF_8_THRESH {
        mask = true;
    }
    let end_dir_thresh = (side_thresh * 3) >> 4;
    let ramp = |a0: i32, b0: i32, c0: i32, a3: i32, b3: i32, c3: i32, m: i32| -> i32 {
        avg(
            ((a0 - b0) - m * (a0 - c0)).abs(),
            ((a3 - b3) - m * (a3 - c3)).abs(),
        )
    };
    if max_samples_neg > 2 && ramp(p0[0], p0[3], p0[1], p3[0], p3[3], p3[1], 3) > end_dir_thresh {
        mask = true;
    }
    if ramp(q0[0], q0[3], q0[1], q3[0], q3[3], q3[1], 3) > end_dir_thresh {
        mask = true;
    }
    if mask {
        return 2;
    }
    if max_samples_pos == 3 {
        return 3;
    }

    // >=4-sample tests.
    let transition = (dm1 + d0) << DF_Q_THRESH_SHIFT;
    let mut dist = 4;
    while dist < MAX_DBL_FLT_LEN + 1 {
        let mut mask = false;
        let q_thresh4 = q_thresh * Q_FIRST[(dist - 4) as usize];
        if transition > q_thresh4 {
            mask = true;
        }
        let end_dir_thresh = (side_thresh * dist) >> 4;
        let rampdist = if dist == 8 { 7 } else { dist } as usize;
        if max_samples_neg >= rampdist as i32
            && ramp(
                p0[0],
                p0[rampdist],
                p0[1],
                p3[0],
                p3[rampdist],
                p3[1],
                rampdist as i32,
            ) > end_dir_thresh
        {
            mask = true;
        }
        if ramp(
            q0[0],
            q0[rampdist],
            q0[1],
            q3[0],
            q3[rampdist],
            q3[1],
            rampdist as i32,
        ) > end_dir_thresh
        {
            mask = true;
        }
        if mask {
            return if dist == 4 { dist - 1 } else { dist - 2 };
        }
        if max_samples_pos <= dist {
            return (dist >> 1) << 1;
        }
        dist += 2;
    }
    MAX_DBL_FLT_LEN
}

/// The tap kernel (`filt_generic_asym_highbd`) applied to one line. Modifies the
/// samples in `buf` in place.
#[allow(clippy::too_many_arguments)]
fn apply_taps(
    buf: &mut [f32],
    base: isize,
    perp: isize,
    width_neg: i32,
    width_pos: i32,
    q_threshold: i32,
    max_pixel: i32,
) {
    if width_neg < 1 || width_pos < 1 {
        return;
    }
    let len = buf.len() as isize;
    let rd = |off: isize| -> i32 { buf[off.clamp(0, len - 1) as usize] as i32 };
    let width = width_neg.max(width_pos);
    let q0 = rd(base);
    let p0 = rd(base - perp);
    let q1 = rd(base + perp);
    let p1 = rd(base - 2 * perp);
    let mut delta_m2 = (3 * (q0 - p0) - (q1 - p1)) * 4;
    let q_clamp = q_threshold * Q_THRESH_MULTS[(width - 1) as usize];
    delta_m2 = delta_m2.clamp(-q_clamp, q_clamp);

    let dneg = delta_m2 * W_MULT[(width_neg - 1) as usize];
    for i in 0..width_neg {
        // Skip out-of-frame writes: those are the decoder's edge-extended border,
        // which never appears in the output. In-frame samples are always written.
        let off = base - perp * (i as isize + 1);
        if off >= 0 && off < len {
            let v = buf[off as usize] as i32 + round_pow2_11(dneg * (width_neg - i));
            buf[off as usize] = v.clamp(0, max_pixel) as f32;
        }
    }
    let dpos = delta_m2 * W_MULT[(width_pos - 1) as usize];
    for i in 0..width_pos {
        let off = base + perp * i as isize;
        if off >= 0 && off < len {
            let v = buf[off as usize] as i32 - round_pow2_11(dpos * (width_pos - i));
            buf[off as usize] = v.clamp(0, max_pixel) as f32;
        }
    }
}

/// Filter one 4-line edge segment. `seg_base` is the q0 offset of line 0; `perp`
/// steps across the edge (toward q1); `line` steps between the 4 lines.
#[allow(clippy::too_many_arguments)]
fn filter_segment(
    buf: &mut [f32],
    seg_base: isize,
    perp: isize,
    line: isize,
    filt_neg: i32,
    filt_pos: i32,
    q_threshold: i32,
    side_threshold: i32,
    max_pixel: i32,
) {
    let l0 = read_line(buf, seg_base, perp);
    let l3 = read_line(buf, seg_base + line * 3, perp);
    let filter = filt_choice(&l0, &l3, filt_neg, filt_pos, q_threshold, side_threshold);
    if filter < 1 {
        return;
    }
    let max_samples_neg = if filt_neg == 0 { 0 } else { filt_neg / 2 - 1 };
    let width_pos = filter;
    let width_neg = filter.min(max_samples_neg);
    for k in 0..4 {
        apply_taps(
            buf,
            seg_base + line * k,
            perp,
            width_neg,
            width_pos,
            q_threshold,
            max_pixel,
        );
    }
}

/// Per-MI transform layout for one plane, driving the edge decisions. All arrays
/// are `mi_cols * mi_rows`, row-major over the plane's own MI grid (4 plane px
/// per MI unit). `sq_w`/`sq_h` are the square TX-size index (0=4..4=64) derived
/// from the covering TX's width/height. `vedge`/`hedge` mark TU boundaries.
pub(crate) struct TxLayout<'a> {
    pub(crate) mi_cols: usize,
    pub(crate) mi_rows: usize,
    pub(crate) vedge: &'a [bool],
    pub(crate) hedge: &'a [bool],
    pub(crate) sq_w: &'a [u8],
    pub(crate) sq_h: &'a [u8],
    /// Per-MI AC qindex (`final_qindex_ac`) for this plane.
    pub(crate) qindex: &'a [u16],
}

/// One plane's deblock inputs.
pub(crate) struct PlaneCfg<'a> {
    pub(crate) buf: &'a mut [f32],
    pub(crate) stride: usize,
    /// Coded plane width/height in pixels (real, 8px-aligned MI extent basis).
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) is_chroma: bool,
    pub(crate) bit_depth: u32,
    pub(crate) delta_q: i32,
    pub(crate) delta_side: i32,
    /// Superblock width/height in this plane's MI units. Luma: 16×16. 4:2:0 chroma:
    /// 8×8. 4:2:2 chroma: 8 wide × 16 high (half-width, full-height). `sb_mi_w`
    /// drives the SB-column schedule; `sb_mi_h` drives the horizontal-SB-edge asymmetry.
    pub(crate) sb_mi_w: usize,
    pub(crate) sb_mi_h: usize,
}

/// Deblock one plane in place, reproducing avm's combined per-superblock order
/// (vertical edges of an SB, then the previous SB's horizontal edges), so the
/// horizontal pass reads vertically-filtered pixels.
pub(crate) fn deblock_plane(
    cfg: &mut PlaneCfg<'_>,
    tx: &TxLayout<'_>,
    apply_vert: bool,
    apply_horz: bool,
) {
    let max_pixel = (1i32 << cfg.bit_depth) - 1;
    let stride = cfg.stride as isize;
    let mi_cols = tx.mi_cols;
    let mi_rows = tx.mi_rows;
    // Frame-border clamp cap (in square-TX index): luma within 16px -> TX_16X16(2),
    // chroma within 8px -> TX_8X8(1).
    let (border_px, border_cap) = if cfg.is_chroma {
        (8usize, 1i32)
    } else {
        (16usize, 2i32)
    };

    let idx = |r: usize, c: usize| r * mi_cols + c;

    // Vertical edges at superblock `sb_c` (all MI columns in that SB).
    let do_vert_sb = |buf: &mut [f32], sb_c: usize| {
        let c_start = sb_c * cfg.sb_mi_w;
        let c_end = (c_start + cfg.sb_mi_w).min(mi_cols);
        for r in 0..mi_rows {
            for c in c_start..c_end {
                if c == 0 || !tx.vedge[idx(r, c)] {
                    continue;
                }
                let x = c * 4;
                // Frame-border clamp: within border_px of the right frame edge.
                let mut min_ts = tx.sq_w[idx(r, c)].min(tx.sq_w[idx(r, c - 1)]) as i32;
                if x + border_px > cfg.width {
                    min_ts = min_ts.min(border_cap);
                }
                let (fneg, fpos) = filter_lengths(min_ts, cfg.is_chroma, false);
                let (qt, st) = combine_thresholds(
                    tx.qindex[idx(r, c)] as i32,
                    tx.qindex[idx(r, c - 1)] as i32,
                    cfg.delta_q,
                    cfg.delta_side,
                    cfg.bit_depth,
                );
                let seg_base = (r * 4) as isize * stride + x as isize;
                filter_segment(buf, seg_base, 1, stride, fneg, fpos, qt, st, max_pixel);
            }
        }
    };

    // Horizontal edges at superblock `sb_c`.
    let do_horz_sb = |buf: &mut [f32], sb_c: usize| {
        let c_start = sb_c * cfg.sb_mi_w;
        let c_end = (c_start + cfg.sb_mi_w).min(mi_cols);
        for c in c_start..c_end {
            for r in 0..mi_rows {
                if r == 0 || !tx.hedge[idx(r, c)] {
                    continue;
                }
                let y = r * 4;
                let mut min_ts = tx.sq_h[idx(r, c)].min(tx.sq_h[idx(r - 1, c)]) as i32;
                if y + border_px > cfg.height {
                    min_ts = min_ts.min(border_cap);
                }
                // horz_superblock_edge: horizontal edge aligned to a 64px SB row.
                let edge_asym = r % cfg.sb_mi_h == 0;
                let (fneg, fpos) = filter_lengths(min_ts, cfg.is_chroma, edge_asym);
                let (qt, st) = combine_thresholds(
                    tx.qindex[idx(r, c)] as i32,
                    tx.qindex[idx(r - 1, c)] as i32,
                    cfg.delta_q,
                    cfg.delta_side,
                    cfg.bit_depth,
                );
                let seg_base = y as isize * stride + (c * 4) as isize;
                filter_segment(buf, seg_base, stride, 1, fneg, fpos, qt, st, max_pixel);
            }
        }
    };

    let sb_cols = mi_cols.div_ceil(cfg.sb_mi_w);
    // Combined per-SB schedule: within an SB-row, do each SB's vertical edges, then
    // (one SB behind) the previous SB's horizontal edges, then the row's last SB.
    for sb_c in 0..sb_cols {
        if apply_vert {
            do_vert_sb(cfg.buf, sb_c);
        }
        if apply_horz && sb_c >= 1 {
            do_horz_sb(cfg.buf, sb_c - 1);
        }
    }
    if apply_horz && sb_cols >= 1 {
        do_horz_sb(cfg.buf, sb_cols - 1);
    }
}

/// Build a per-MI TX-boundary grid for a plane. `leaves` are the luma prediction
/// blocks `(mi_row, mi_col, bw_mi, bh_mi)` in LUMA MI; `ssx`/`ssy` subsample them
/// into this plane's grid (0 for luma / 4:4:4; 1 for a halved chroma dimension).
/// Each block tiles into TUs of `min(dim, 32px)` in this plane's MI. Returns
/// `(vedge, hedge, sq_w, sq_h)` sized `mi_cols * mi_rows`.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn build_tx_grid(
    leaves: &[(usize, usize, usize, usize)],
    skip_leaves: &[(usize, usize, usize, usize)],
    mi_cols: usize,
    mi_rows: usize,
    ssx: usize,
    ssy: usize,
    cap_w: usize,
    cap_h: usize,
) -> (Vec<bool>, Vec<bool>, Vec<u8>, Vec<u8>) {
    let n = mi_cols * mi_rows;
    let mut vedge = vec![false; n];
    let mut hedge = vec![false; n];
    let mut sq_w = vec![0u8; n];
    let mut sq_h = vec![0u8; n];
    for &(lmr, lmc, bw, bh) in leaves {
        let (blr, blc) = (lmr >> ssy, lmc >> ssx);
        let (bbw, bbh) = ((bw >> ssx).max(1), (bh >> ssy).max(1));
        let (txw, txh) = (bbw.min(cap_w), bbh.min(cap_h));
        let (sw, sh) = (txw.trailing_zeros() as u8, txh.trailing_zeros() as u8);
        let mut ty = 0;
        while ty < bbh {
            let mut tx = 0;
            while tx < bbw {
                for dr in 0..txh {
                    for dc in 0..txw {
                        let (r, c) = (blr + ty + dr, blc + tx + dc);
                        if r < mi_rows && c < mi_cols {
                            let i = r * mi_cols + c;
                            vedge[i] = dc == 0;
                            hedge[i] = dr == 0;
                            sq_w[i] = sw;
                            sq_h[i] = sh;
                        }
                    }
                }
                tx += txw;
            }
            ty += txh;
        }
    }
    // A skipped inter block has no residual transform partition. The decoder
    // derives its deblock geometry from the prediction block's maximum transform,
    // so retain only the prediction-unit boundaries inside this region.
    for &(lmr, lmc, bw, bh) in skip_leaves {
        let (blr, blc) = (lmr >> ssy, lmc >> ssx);
        let (bbw, bbh) = ((bw >> ssx).max(1), (bh >> ssy).max(1));
        let (sw, sh) = (bbw.trailing_zeros() as u8, bbh.trailing_zeros() as u8);
        for dr in 0..bbh {
            for dc in 0..bbw {
                let (r, c) = (blr + dr, blc + dc);
                if r < mi_rows && c < mi_cols {
                    let i = r * mi_cols + c;
                    vedge[i] = dc == 0;
                    hedge[i] = dr == 0;
                    sq_w[i] = sw;
                    sq_h[i] = sh;
                }
            }
        }
    }
    (vedge, hedge, sq_w, sq_h)
}

/// Per-MI qindex for a plane, from the per-luma-SB accumulator. A plane MI maps to
/// the covering 64px luma superblock via the subsampling shift.
fn build_qindex_grid(
    sb_qidx: &[u16],
    sb_cols: usize,
    mi_cols: usize,
    mi_rows: usize,
    ssx: usize,
    ssy: usize,
    base_q: u16,
) -> Vec<u16> {
    let mut q = vec![base_q; mi_cols * mi_rows];
    for r in 0..mi_rows {
        for c in 0..mi_cols {
            let sb = (((r << ssy) / 16) * sb_cols + ((c << ssx) / 16)).min(sb_qidx.len() - 1);
            q[r * mi_cols + c] = sb_qidx[sb];
        }
    }
    q
}

/// Whole-frame deblock inputs. `leaves` are the luma prediction blocks; the whole-SB
/// fast path passes one 64×64 leaf per superblock, the partition path passes its
/// recorded leaves. Chroma grids are derived by subsampling the same leaves.
pub(crate) struct FrameDeblock<'a> {
    pub(crate) recy: &'a mut [f32],
    pub(crate) recu: &'a mut [f32],
    pub(crate) recv: &'a mut [f32],
    pub(crate) luma_stride: usize,
    pub(crate) chroma_stride: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) ssx: usize,
    pub(crate) ssy: usize,
    pub(crate) has_chroma: bool,
    pub(crate) chroma_deblock: bool,
    pub(crate) bit_depth: u32,
    pub(crate) delta_y: i32,
    pub(crate) delta_uv: i32,
    pub(crate) base_q: u16,
    pub(crate) sb_cols: usize,
    pub(crate) sb_qidx: &'a [u16],
    pub(crate) leaves: &'a [(usize, usize, usize, usize)],
    /// Chroma TU leaves (LUMA-MI units; subsampled by ssx/ssy in `build_tx_grid`).
    /// Equals `leaves` everywhere except whole-64 4:4:4 SBs, where the chroma is a
    /// single TX_64X64 (one 16×16-MI leaf) rather than the luma's 4× TX_32X32.
    pub(crate) chroma_leaves: &'a [(usize, usize, usize, usize)],
    /// Inter prediction leaves that signal skip_txfm, in luma MI units.
    pub(crate) skip_leaves: &'a [(usize, usize, usize, usize)],
    /// Max luma TX size in MI. (8,8) when `leaves` are prediction blocks tiled into
    /// 32×32 TUs (4:2:0/4:2:2/4:0:0); (16,16) when `leaves` are already TX-granular
    /// rectangles that must not be re-tiled (4:4:4, whose whole-SB mixes SPLIT 32×32
    /// and VERT4 16×64 tilings that no single tiling cap can express).
    pub(crate) luma_tx_cap: (usize, usize),
    /// Max chroma TX size in chroma MI: 4:2:0=(8,8), 4:2:2=(8,16), 4:4:4=(16,16).
    pub(crate) chroma_tx_cap: (usize, usize),
}

/// Deblock a whole frame (luma, then chroma) for any 4:x:x layout.
pub(crate) fn deblock_frame(f: FrameDeblock<'_>) {
    let mi_cols = ((f.width + 7) & !7) / 4;
    let mi_rows = ((f.height + 7) & !7) / 4;
    let (vedge, hedge, sq_w, sq_h) = build_tx_grid(
        f.leaves,
        f.skip_leaves,
        mi_cols,
        mi_rows,
        0,
        0,
        f.luma_tx_cap.0,
        f.luma_tx_cap.1,
    );
    let qidx = build_qindex_grid(f.sb_qidx, f.sb_cols, mi_cols, mi_rows, 0, 0, f.base_q);
    let tx = TxLayout {
        mi_cols,
        mi_rows,
        vedge: &vedge,
        hedge: &hedge,
        sq_w: &sq_w,
        sq_h: &sq_h,
        qindex: &qidx,
    };
    let mut lcfg = PlaneCfg {
        buf: f.recy,
        stride: f.luma_stride,
        width: f.width,
        height: f.height,
        is_chroma: false,
        bit_depth: f.bit_depth,
        delta_q: f.delta_y,
        delta_side: f.delta_y,
        sb_mi_w: 16,
        sb_mi_h: 16,
    };
    deblock_plane(&mut lcfg, &tx, true, true);

    if f.has_chroma && f.chroma_deblock {
        let cw = (f.width + ((1 << f.ssx) - 1)) >> f.ssx;
        let ch = (f.height + ((1 << f.ssy) - 1)) >> f.ssy;
        let cmi_cols = ((cw + 7) & !7) / 4;
        let cmi_rows = ((ch + 7) & !7) / 4;
        let (cve, che, csw, csh) = build_tx_grid(
            f.chroma_leaves,
            f.skip_leaves,
            cmi_cols,
            cmi_rows,
            f.ssx,
            f.ssy,
            f.chroma_tx_cap.0,
            f.chroma_tx_cap.1,
        );
        let cq = build_qindex_grid(
            f.sb_qidx, f.sb_cols, cmi_cols, cmi_rows, f.ssx, f.ssy, f.base_q,
        );
        let ctx = TxLayout {
            mi_cols: cmi_cols,
            mi_rows: cmi_rows,
            vedge: &cve,
            hedge: &che,
            sq_w: &csw,
            sq_h: &csh,
            qindex: &cq,
        };
        let (sbw, sbh) = (16 >> f.ssx, 16 >> f.ssy);
        for cbuf in [f.recu, f.recv] {
            let mut ccfg = PlaneCfg {
                buf: cbuf,
                stride: f.chroma_stride,
                width: cw,
                height: ch,
                is_chroma: true,
                bit_depth: f.bit_depth,
                delta_q: f.delta_uv,
                delta_side: f.delta_uv,
                sb_mi_w: sbw,
                sb_mi_h: sbh,
            };
            deblock_plane(&mut ccfg, &ctx, true, true);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn df_quant_matches_config_formula() {
        // Encoder config() computes df_quant = ((qstep+4)>>3)>>6 at 8-bit.
        for q in 1..=255u32 {
            let expect = ((crate::av2::quant::qstep(q) as i32 + 4) >> 3) >> 6;
            assert_eq!(df_quant_from_qindex(q as i32, 8), expect, "q={q}");
        }
    }

    #[test]
    fn df_side_boundary_values() {
        // Indices 0..=56 of side_thresholds are -16; at 8-bit,
        // side = max(-16 + 16, 0) >> 5 = 0. So low qindex -> side 0 (filter off).
        assert_eq!(df_side_from_qindex(0, 8), 0);
        assert_eq!(df_side_from_qindex(56, 8), 0);
        // Top of the table (index 295 = 1678): max(1678 + 16, 0) >> 5 = 52.
        assert_eq!(df_side_from_qindex(295, 8), (1678 + 16) >> 5);
        // Clamp saturates above the table.
        assert_eq!(df_side_from_qindex(10_000, 8), (1678 + 16) >> 5);
    }

    #[test]
    fn round_pow2_11_signed() {
        assert_eq!(round_pow2_11(0), 0);
        assert_eq!(round_pow2_11(1024), 1);
        assert_eq!(round_pow2_11(-1024), 0); // (-1024 + 1024) >> 11 = 0
        assert_eq!(round_pow2_11(-2048), -1); // (-2048 + 1024) >> 11 = -1024>>11 = -1
    }

    #[test]
    fn flat_edge_unchanged_when_thresholds_zero() {
        // side/q threshold 0 -> no filtering.
        let mut buf = vec![100.0f32; 16 * 16];
        let l0 = read_line(&buf, 8 * 16 + 8, 1);
        let l3 = read_line(&buf, 8 * 16 + 8, 1);
        assert_eq!(filt_choice(&l0, &l3, 8, 8, 0, 10), 0);
        assert_eq!(filt_choice(&l0, &l3, 8, 8, 10, 0), 0);
        // A genuine step with zero side threshold still does nothing.
        for i in 0..16 {
            buf[i * 16 + 8] = 150.0;
        }
        let l0 = read_line(&buf, 8 * 16 + 8, 1);
        assert_eq!(filt_choice(&l0, &l0, 8, 8, 10, 0), 0);
    }

    #[test]
    fn skipped_64_block_has_no_internal_32_tx_edges() {
        let leaf = [(0, 0, 16, 16)];

        let (generic_v, generic_h, _, _) = build_tx_grid(&leaf, &[], 16, 16, 0, 0, 8, 8);
        assert!(generic_v[8]);
        assert!(generic_h[8 * 16]);

        let (skip_v, skip_h, skip_w, skip_hsize) = build_tx_grid(&leaf, &leaf, 16, 16, 0, 0, 8, 8);
        assert!(skip_v[0]);
        assert!(skip_h[0]);
        assert!(!skip_v[8]);
        assert!(!skip_h[8 * 16]);
        assert!(skip_w.iter().all(|&size| size == 4));
        assert!(skip_hsize.iter().all(|&size| size == 4));
    }
}
