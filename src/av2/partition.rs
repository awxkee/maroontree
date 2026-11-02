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

// BLOCK_SIZES_ALL enum indices 0..28.
// 0:4X4 1:4X8 2:8X4 3:8X8 4:8X16 5:16X8 6:16X16 7:16X32 8:32X16 9:32X32 10:32X64
// 11:64X32 12:64X64 13:64X128 14:128X64 15:128X128 16:128X256 17:256X128 18:256X256
// 19:4X16 20:16X4 21:8X32 22:32X8 23:16X64 24:64X16 25:4X32 26:32X4 27:8X64 28:64X8
pub const INV: u8 = 0xFF;

static MI_W: [usize; 29] = [
    1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 16, 16, 16, 32, 32, 32, 64, 64, 1, 4, 2, 8, 4, 16, 1, 8, 2, 16,
];
static MI_H: [usize; 29] = [
    1, 2, 1, 2, 4, 2, 4, 8, 4, 8, 16, 8, 16, 32, 16, 32, 64, 32, 64, 4, 1, 8, 2, 16, 4, 8, 1, 16, 2,
];
static MI_WL: [usize; 29] = [
    0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 0, 2, 1, 3, 2, 4, 0, 3, 1, 4,
];
static MI_HL: [usize; 29] = [
    0, 1, 0, 1, 2, 1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 6, 5, 6, 2, 0, 3, 1, 4, 2, 3, 0, 4, 1,
];

// SPLIT_CTX_MODE group (do_split cdf), indices 0..24 (partition points are < 19,
// but the rect ext leaves 19..24 are non-partition-points so never indexed here).
static BSIZE_MAP: [usize; 25] = [
    0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];
// RECT_TYPE_CTX_MODE group (rect_type cdf).
static BSIZE_RECT_MAP: [usize; 25] = [
    0, 0, 0, 0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 13, 14, 13, 14,
];

// partition_context_lookup[bsize].{above,left} written at NONE leaves.
static PCTX_ABOVE: [u8; 29] = [
    63, 63, 62, 62, 62, 60, 60, 60, 56, 56, 56, 48, 48, 48, 32, 32, 32, 0, 0, 63, 60, 62, 56, 60,
    48, 63, 56, 62, 48,
];
static PCTX_LEFT: [u8; 29] = [
    63, 62, 63, 62, 60, 62, 60, 56, 60, 56, 48, 56, 48, 32, 48, 32, 0, 32, 0, 60, 63, 56, 62, 48,
    60, 56, 63, 48, 62,
];

// get_partition_subsize for PARTITION_HORZ / PARTITION_VERT (INV = BLOCK_INVALID).
static SUBSIZE_HORZ: [u8; 29] = [
    INV, 0, INV, 2, 3, 20, 5, 6, 22, 8, 9, 24, 11, 12, INV, 14, 15, INV, 17, 1, INV, 4, 26, 7, 28,
    19, INV, 21, INV,
];
static SUBSIZE_VERT: [u8; 29] = [
    INV, INV, 0, 1, 19, 3, 4, 21, 6, 7, 23, 9, 10, INV, 12, 13, INV, 15, 16, INV, 2, 25, 5, 27, 8,
    INV, 20, INV, 22,
];

// do_split cdf0 (my encode_bool form = 32768 - AVM_CDF2), plane 0, 64 contexts.
pub(crate) static DO_SPLIT_CDF0: [u32; 64] = [
    4684, 9013, 9134, 13400, 7807, 17827, 16614, 26863, 10834, 22328, 20784, 29294, 12276, 25805,
    24669, 31239, 8651, 24897, 9164, 24339, 5412, 10327, 23871, 25957, 15176, 27120, 27429, 31686,
    6625, 21389, 12626, 25367, 6533, 9094, 20327, 22286, 12105, 28576, 27494, 32055, 4513, 5398,
    9241, 11778, 6041, 11581, 7444, 14930, 6632, 16177, 12930, 22163, 9854, 20159, 21427, 28212,
    8550, 19709, 17390, 26910, 11124, 25001, 24459, 31081,
];
// do_square_split cdf0 (my encode_bool form = 32768 - AVM_CDF2), plane 0 (LUMA_PART),
// 8 square-split contexts. AVM default_do_square_split_cdf[0][ctx] values are
// {18000,10521,11395,4419,12996,8185,10979,5010}. Context is `left*2 + above`
// (from the split partition context bits), with no bsize offset for < 256x256.
pub(crate) static DO_SQUARE_SPLIT_CDF0: [u32; 8] =
    [14768, 22247, 21373, 28349, 19772, 24583, 21789, 27758];
// rect_type cdf0 (HORZ vs VERT), plane 0, 64 contexts.
pub(crate) static RECT_TYPE_CDF0: [u32; 64] = [
    18124, 22595, 14239, 16697, 12505, 19955, 6156, 9491, 22174, 25768, 12766, 19879, 18914, 22018,
    14388, 15263, 18338, 21214, 12690, 13671, 17490, 22631, 10847, 18147, 13438, 16847, 6550, 8450,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16702, 23543, 9919, 17951, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 14225, 19558, 8401, 14351, 8067, 16384, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
];

// do_split / rect_type cdf0 for plane 1 (the SDP chroma tree), 64 contexts.
pub(crate) static DO_SPLIT_CDF1: [u32; 64] = [
    16384, 16384, 16384, 16384, 7214, 13876, 14238, 21962, 10264, 20628, 20802, 27784, 8308, 24070,
    23113, 30205, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 6541, 16384, 15099, 16384, 6589, 14879,
    16384, 16384, 9268, 19653, 17515, 26310, 10202, 21271, 22723, 29018,
];
pub(crate) static RECT_TYPE_CDF1: [u32; 64] = [
    17114, 19268, 13591, 18029, 13999, 19268, 9185, 11841, 17723, 22240, 10294, 18518, 16207,
    21341, 10894, 16424, 11202, 19411, 5413, 8651, 21867, 26988, 13712, 23627, 12332, 17075, 6232,
    9101, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384,
    16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 16384, 13527,
    19730, 3865, 7966, 23671, 30019, 17567, 28319, 16384, 16384, 16384, 16384,
];

const HORZ: u8 = 0; // RECT_PART_TYPE / PARTITION ordering: HORZ first
const VERT: u8 = 1;

#[inline]
fn is_partition_point(b: usize) -> bool {
    b != 0 && b < 25
}

/// rect_type_implied_by_bsize for luma (None = RECT_INVALID).
fn rect_type_implied(b: usize) -> Option<u8> {
    match b {
        1 | 13 | 16 | 19 => Some(HORZ), // 4X8, 64X128, 128X256, 4X16
        2 | 14 | 17 | 20 => Some(VERT), // 8X4, 128X64, 256X128, 16X4
        _ => None,
    }
}

/// is_partition_implied_at_boundary -> Some(HORZ|VERT) or None (INVALID).
fn implied_boundary(
    mi_row: usize,
    mi_col: usize,
    b: usize,
    mi_rows: usize,
    mi_cols: usize,
) -> Option<u8> {
    let (w, h) = (MI_W[b], MI_H[b]);
    let has_rows = mi_row + h / 2 < mi_rows;
    let has_cols = mi_col + w / 2 < mi_cols;
    if has_rows && has_cols {
        return None;
    }
    if w == h {
        // square: always implied
        return Some(if has_rows && !has_cols { VERT } else { HORZ });
    }
    if h > w {
        // tall
        if !has_rows {
            return Some(HORZ);
        }
        // !has_cols
        let sub_has_cols = mi_col + w / 4 < mi_cols;
        if w >= 4 && !sub_has_cols {
            Some(HORZ)
        } else {
            None
        }
    } else {
        // wide
        if !has_cols {
            return Some(VERT);
        }
        // !has_rows
        let sub_has_rows = mi_row + h / 4 < mi_rows;
        if h >= 4 && !sub_has_rows {
            Some(VERT)
        } else {
            None
        }
    }
}

/// raw partition context (left*2 + above) from the partition-context arrays.
fn raw_ctx(above_pctx: &[u8], left_pctx: &[u8], mi_row: usize, mi_col: usize, b: usize) -> usize {
    let (bsl_w, bsl_h) = (MI_WL[b], MI_HL[b]);
    let above = ((above_pctx[mi_col] >> bsl_w.saturating_sub(1)) & 1) as usize;
    let left = ((left_pctx[mi_row & 15] >> bsl_h.saturating_sub(1)) & 1) as usize;
    left * 2 + above
}

fn update_pctx(
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
    mi_row: usize,
    mi_col: usize,
    b: usize,
) {
    let (w, h) = (MI_W[b], MI_H[b]);
    let max_len = above_pctx.len();
    above_pctx[mi_col..(mi_col + w).min(max_len)].fill(PCTX_ABOVE[b]);
    let base = mi_row & 15;
    left_pctx[base..(base + h).min(16)].fill(PCTX_LEFT[b]);
}

/// A partition op in pre-order. `RectType` is a signaled rect_type bool; `Leaf`
/// is a PARTITION_NONE block to code (with its do_split cdf, None if non-pp).
#[derive(Clone, Debug)]
pub(crate) enum Op {
    RectType {
        cdf: u32,
        val: u32,
        ctx: usize,
    },
    /// Emit `do_split` bool (1 = split) with the given static cdf, then, when
    /// `square_cdf` is `Some`, emit `do_square_split = 1` selecting PARTITION_SPLIT.
    Split {
        do_split_cdf: u32,
        square_cdf: u32,
    },
    Leaf {
        mi_row: usize,
        mi_col: usize,
        bw_mi: usize,
        bh_mi: usize,
        part_cdf: Option<u32>,
    },
}

#[allow(clippy::too_many_arguments)]
fn walk(
    mi_row: usize,
    mi_col: usize,
    b: usize,
    mi_rows: usize,
    mi_cols: usize,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
    out: &mut Vec<Op>,
) {
    if mi_row >= mi_rows || mi_col >= mi_cols {
        return; // fully out of frame: skipped
    }
    let (w, h) = (MI_W[b], MI_H[b]);

    // Non-partition-point blocks (4x4 / narrow ext) read no partition bit -> NONE leaf.
    if !is_partition_point(b) {
        out.push(Op::Leaf {
            mi_row,
            mi_col,
            bw_mi: w,
            bh_mi: h,
            part_cdf: None,
        });
        update_pctx(above_pctx, left_pctx, mi_row, mi_col, b);
        return;
    }

    let has_rows = mi_row + h / 2 < mi_rows;
    let has_cols = mi_col + w / 2 < mi_cols;
    let none_allowed = has_rows && has_cols;
    let rimp = rect_type_implied(b);
    let horz_valid = SUBSIZE_HORZ[b] != INV && rimp != Some(VERT);
    let vert_valid = SUBSIZE_VERT[b] != INV && rimp != Some(HORZ);

    // 1) implied-at-boundary (no bits) if that partition is allowed.
    let imp = implied_boundary(mi_row, mi_col, b, mi_rows, mi_cols);
    let part: u8;
    if let Some(p) = imp {
        let allowed = match p {
            HORZ => horz_valid,
            _ => vert_valid,
        };
        if allowed {
            part = p;
            recurse(
                part, mi_row, mi_col, b, mi_rows, mi_cols, above_pctx, left_pctx, out,
            );
            return;
        }
        part = decide_signaled(
            &SignaledNode {
                mi_row,
                mi_col,
                block: b,
                has_rows,
                none_allowed,
                implied_rect: rimp,
                horz_valid,
                vert_valid,
                above_context: above_pctx,
                left_context: left_pctx,
            },
            out,
        );
    } else {
        // 2) only-allowed (no bits) when exactly one partition is allowed.
        let n_allowed = (none_allowed as u8) + (horz_valid as u8) + (vert_valid as u8);
        if n_allowed == 1 {
            part = if none_allowed {
                255 /*NONE*/
            } else if horz_valid {
                HORZ
            } else {
                VERT
            };
        } else {
            part = decide_signaled(
                &SignaledNode {
                    mi_row,
                    mi_col,
                    block: b,
                    has_rows,
                    none_allowed,
                    implied_rect: rimp,
                    horz_valid,
                    vert_valid,
                    above_context: above_pctx,
                    left_context: left_pctx,
                },
                out,
            );
        }
    }

    if part == 255 {
        // PARTITION_NONE leaf
        out.push(Op::Leaf {
            mi_row,
            mi_col,
            bw_mi: w,
            bh_mi: h,
            part_cdf: None,
        });
        // part_cdf filled below (need ctx before update)
        let ctx = raw_ctx(above_pctx, left_pctx, mi_row, mi_col, b) + BSIZE_MAP[b] * 4;
        if let Some(Op::Leaf { part_cdf, .. }) = out.last_mut() {
            *part_cdf = Some(DO_SPLIT_CDF0[ctx]);
        }
        update_pctx(above_pctx, left_pctx, mi_row, mi_col, b);
    } else {
        recurse(
            part, mi_row, mi_col, b, mi_rows, mi_cols, above_pctx, left_pctx, out,
        );
    }
}

/// Inputs for resolving one signaled partition node.
struct SignaledNode<'a> {
    mi_row: usize,
    mi_col: usize,
    block: usize,
    has_rows: bool,
    none_allowed: bool,
    implied_rect: Option<u8>,
    horz_valid: bool,
    vert_valid: bool,
    above_context: &'a [u8],
    left_context: &'a [u8],
}

/// Resolve a signaled node. Emits a do_split=0 (NONE) — represented by returning
/// 255 so the caller pushes the leaf — or, when NONE isn't allowed, picks a rect
/// direction (emitting a rect_type bit only when both HORZ and VERT are valid).
fn decide_signaled(node: &SignaledNode<'_>, out: &mut Vec<Op>) -> u8 {
    let SignaledNode {
        mi_row,
        mi_col,
        block: b,
        has_rows,
        none_allowed,
        implied_rect: rimp,
        horz_valid,
        vert_valid,
        above_context: above_pctx,
        left_context: left_pctx,
    } = *node;
    if none_allowed {
        return 255; // PARTITION_NONE (do_split=0, cdf added by caller)
    }
    // NONE not allowed -> do_split implied (no bit). do_square_split never (ext off,
    // not 128/256). Determine rect_type:
    if let Some(r) = rimp {
        return r;
    }
    match (horz_valid, vert_valid) {
        (true, false) => HORZ,
        (false, true) => VERT,
        (true, true) => {
            // signal rect_type: prefer reducing the out-of-frame dimension.
            let r = if !has_rows { HORZ } else { VERT };
            let ctx = raw_ctx(above_pctx, left_pctx, mi_row, mi_col, b) + BSIZE_RECT_MAP[b] * 4;
            out.push(Op::RectType {
                cdf: RECT_TYPE_CDF0[ctx],
                val: r as u32,
                ctx,
            });
            r
        }
        (false, false) => HORZ, // unreachable for our block set
    }
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    part: u8,
    mi_row: usize,
    mi_col: usize,
    b: usize,
    mi_rows: usize,
    mi_cols: usize,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
    out: &mut Vec<Op>,
) {
    let (w, h) = (MI_W[b], MI_H[b]);
    if part == HORZ {
        let sub = SUBSIZE_HORZ[b] as usize;
        let hbs_h = h / 2;
        walk(
            mi_row, mi_col, sub, mi_rows, mi_cols, above_pctx, left_pctx, out,
        );
        walk(
            mi_row + hbs_h,
            mi_col,
            sub,
            mi_rows,
            mi_cols,
            above_pctx,
            left_pctx,
            out,
        );
    } else {
        let sub = SUBSIZE_VERT[b] as usize;
        let hbs_w = w / 2;
        walk(
            mi_row, mi_col, sub, mi_rows, mi_cols, above_pctx, left_pctx, out,
        );
        walk(
            mi_row,
            mi_col + hbs_w,
            sub,
            mi_rows,
            mi_cols,
            above_pctx,
            left_pctx,
            out,
        );
    }
}

/// Build the pre-order op list for one superblock (top-left at sb_row/sb_col in mi).
/// `above_pctx` is frame-wide (persists down columns); `left_pctx` is len-16 and is
/// zeroed by the caller at the start of each SB row.
pub(crate) fn sb_partition_ops(
    sb_row: usize,
    sb_col: usize,
    mi_rows: usize,
    mi_cols: usize,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) -> Vec<Op> {
    let mut out = Vec::new();
    walk(
        sb_row * 16,
        sb_col * 16,
        12, /*64X64*/
        mi_rows,
        mi_cols,
        above_pctx,
        left_pctx,
        &mut out,
    );
    out
}

/// Update the partition contexts for a whole 64x64 SB coded as PARTITION_NONE, exactly
/// as the decoder does (`update_pctx` for BLOCK_64X64). The whole-64 fast path must call
/// this so that neighbors of a chroma-motivated split SB (which set pctx=56 on their
/// 32x32 leaves) observe the correct do_split / rect_type contexts. Without it, a NONE SB
/// following a split leaves stale context and the decoder mis-parses the next partition.
pub(crate) fn sb_none_pctx(
    sb_row: usize,
    sb_col: usize,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) {
    update_pctx(above_pctx, left_pctx, sb_row * 16, sb_col * 16, 12);
}

/// The `do_split` cdf a whole-64 PARTITION_NONE SB must use, given the current
/// partition contexts. In an all-whole-64 frame every SB has raw_ctx 0 and this
/// returns `DO_SPLIT_CDF0[12]` (== 12276), preserving the legacy fast path; once a
/// neighboring SB has been square-split (setting pctx=56 on its 32x32 leaves) the
/// context becomes non-zero and the correct cdf is selected, matching the decoder.
pub(crate) fn sb_none_do_split_cdf(
    sb_row: usize,
    sb_col: usize,
    above_pctx: &[u8],
    left_pctx: &[u8],
) -> u32 {
    let ctx = raw_ctx(above_pctx, left_pctx, sb_row * 16, sb_col * 16, 12) + BSIZE_MAP[12] * 4;
    DO_SPLIT_CDF0[ctx]
}

/// Ops for a full interior 64x64 SB that we choose to split down to 32x32 leaves.
/// A 64x64 is NOT square-split-eligible in this configuration (square split only
/// applies to 128/256 SBs), so we reach 32x32 via two levels of rectangular
/// partition: 64x64 --VERT--> two 32x64, each --HORZ--> two 32x32. Extended
/// partitions are disabled in the sequence header, so `do_ext_partition` is implied
/// (no bit); each rect node emits only `do_split=1` + `rect_type`. Each 32x32 leaf
/// then carries its own `do_split=0`. Contexts are updated exactly as the decoder does.
/// Valid only for a full interior SB (all children in-frame) so no boundary-implied
/// partitions apply.
pub(crate) fn sb_square_split_ops(
    sb_row: usize,
    sb_col: usize,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) -> Vec<Op> {
    let mut out = Vec::new();
    let mi_row = sb_row * 16;
    let mi_col = sb_col * 16;
    const B64: usize = 12; // BLOCK_64X64
    const B32X64: usize = 10; // BLOCK_32X64
    const B32: usize = 9; // BLOCK_32X32
    // 64x64 -> VERT: do_split=1 (implied by NONE-not-forced? no: NONE is allowed for a
    // full interior block, so do_split is signaled), rect_type=VERT. do_ext implied off.
    out.push(Op::Split {
        do_split_cdf: DO_SPLIT_CDF0
            [raw_ctx(above_pctx, left_pctx, mi_row, mi_col, B64) + BSIZE_MAP[B64] * 4],
        square_cdf: 0, // sentinel: no square-split symbol for 64x64 (not eligible)
    });
    let rt_ctx = raw_ctx(above_pctx, left_pctx, mi_row, mi_col, B64) + BSIZE_RECT_MAP[B64] * 4;
    out.push(Op::RectType {
        cdf: RECT_TYPE_CDF0[rt_ctx],
        val: VERT as u32,
        ctx: rt_ctx,
    });
    // Two 32x64 children (left, right). Each does HORZ -> two 32x32.
    for dc in [0usize, 8usize] {
        let cr = mi_row;
        let cc = mi_col + dc;
        // 32x64 -> HORZ: do_split=1 + rect_type=HORZ.
        out.push(Op::Split {
            do_split_cdf: DO_SPLIT_CDF0
                [raw_ctx(above_pctx, left_pctx, cr, cc, B32X64) + BSIZE_MAP[B32X64] * 4],
            square_cdf: 0,
        });
        let rt2 = raw_ctx(above_pctx, left_pctx, cr, cc, B32X64) + BSIZE_RECT_MAP[B32X64] * 4;
        out.push(Op::RectType {
            cdf: RECT_TYPE_CDF0[rt2],
            val: HORZ as u32,
            ctx: rt2,
        });
        // Two 32x32 leaves (top, bottom).
        for dr in [0usize, 8usize] {
            let lr = cr + dr;
            let lc = cc;
            let ctx = raw_ctx(above_pctx, left_pctx, lr, lc, B32) + BSIZE_MAP[B32] * 4;
            out.push(Op::Leaf {
                mi_row: lr,
                mi_col: lc,
                bw_mi: 8,
                bh_mi: 8,
                part_cdf: Some(DO_SPLIT_CDF0[ctx]),
            });
            update_pctx(above_pctx, left_pctx, lr, lc, B32);
        }
    }
    out
}

/// Content-adaptive interior tree: always enters the 64→32 prediction split,
/// then optionally splits each 32×32 child into four independently predicted
/// 16×16 leaves. AV2 represents both square steps through VERT+HORZ rectangular
/// nodes in this sequence configuration.
pub(crate) fn sb_rd_split_ops(
    sb_row: usize,
    sb_col: usize,
    split32: [bool; 4],
    split16: [bool; 16],
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) -> Vec<Op> {
    let mut out = Vec::new();
    let mi_row = sb_row * 16;
    let mi_col = sb_col * 16;
    const B64: usize = 12;
    const B32X64: usize = 10;
    const B32: usize = 9;
    const B16X32: usize = 7;
    const B16: usize = 6;
    const B8X16: usize = 4;
    const B8: usize = 3;

    push_rect_split(&mut out, above_pctx, left_pctx, mi_row, mi_col, B64, VERT);
    for dc in [0usize, 8usize] {
        push_rect_split(
            &mut out,
            above_pctx,
            left_pctx,
            mi_row,
            mi_col + dc,
            B32X64,
            HORZ,
        );
        for dr in [0usize, 8usize] {
            let (r, c) = (mi_row + dr, mi_col + dc);
            let child = (dr / 8) * 2 + dc / 8;
            if !split32[child] {
                let ctx = raw_ctx(above_pctx, left_pctx, r, c, B32) + BSIZE_MAP[B32] * 4;
                out.push(Op::Leaf {
                    mi_row: r,
                    mi_col: c,
                    bw_mi: 8,
                    bh_mi: 8,
                    part_cdf: Some(DO_SPLIT_CDF0[ctx]),
                });
                update_pctx(above_pctx, left_pctx, r, c, B32);
                continue;
            }
            push_rect_split(&mut out, above_pctx, left_pctx, r, c, B32, VERT);
            for dc16 in [0usize, 4usize] {
                push_rect_split(&mut out, above_pctx, left_pctx, r, c + dc16, B16X32, HORZ);
                for dr16 in [0usize, 4usize] {
                    let (lr, lc) = (r + dr16, c + dc16);
                    let child16 = ((lr - mi_row) / 4) * 4 + (lc - mi_col) / 4;
                    if split16[child16] {
                        push_rect_split(&mut out, above_pctx, left_pctx, lr, lc, B16, VERT);
                        for dc8 in [0usize, 2usize] {
                            push_rect_split(
                                &mut out,
                                above_pctx,
                                left_pctx,
                                lr,
                                lc + dc8,
                                B8X16,
                                HORZ,
                            );
                            for dr8 in [0usize, 2usize] {
                                let (r8, c8) = (lr + dr8, lc + dc8);
                                let ctx =
                                    raw_ctx(above_pctx, left_pctx, r8, c8, B8) + BSIZE_MAP[B8] * 4;
                                out.push(Op::Leaf {
                                    mi_row: r8,
                                    mi_col: c8,
                                    bw_mi: 2,
                                    bh_mi: 2,
                                    part_cdf: Some(DO_SPLIT_CDF0[ctx]),
                                });
                                update_pctx(above_pctx, left_pctx, r8, c8, B8);
                            }
                        }
                        continue;
                    }
                    let ctx = raw_ctx(above_pctx, left_pctx, lr, lc, B16) + BSIZE_MAP[B16] * 4;
                    out.push(Op::Leaf {
                        mi_row: lr,
                        mi_col: lc,
                        bw_mi: 4,
                        bh_mi: 4,
                        part_cdf: Some(DO_SPLIT_CDF0[ctx]),
                    });
                    update_pctx(above_pctx, left_pctx, lr, lc, B16);
                }
            }
        }
    }
    out
}

/// Ops for a full interior SB coded as `d`: the probe's square tree
/// (`split64` / `split32` / `split16`, each square step a VERT + HORZ rect pair)
/// with rect leaf pairs where a node stays unsplit (`rect64` / `rect32` /
/// `rect16`: 64x32|32x64, 32x16|16x32, 16x8|8x16).
pub(crate) fn sb_tree_ops(
    sb_row: usize,
    sb_col: usize,
    d: &crate::av2::leaf::LumaPartitionDecision,
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) -> Vec<Op> {
    const B64: usize = 12;
    const B32X64: usize = 10;
    const B32: usize = 9;
    const B16X32: usize = 7;
    const B16: usize = 6;
    const B8X16: usize = 4;
    const B8: usize = 3;
    let mut out = Vec::new();
    let (mi_row, mi_col) = (sb_row * 16, sb_col * 16);
    let leaf = |out: &mut Vec<Op>, ap: &mut [u8], lp: &mut [u8], r: usize, c: usize, b: usize| {
        let ctx = raw_ctx(ap, lp, r, c, b) + BSIZE_MAP[b] * 4;
        out.push(Op::Leaf {
            mi_row: r,
            mi_col: c,
            bw_mi: MI_W[b],
            bh_mi: MI_H[b],
            part_cdf: Some(DO_SPLIT_CDF0[ctx]),
        });
        update_pctx(ap, lp, r, c, b);
    };
    // An unsplit node of size `b`: one leaf, or a HORZ / VERT pair of rect leaves.
    let node = |out: &mut Vec<Op>,
                ap: &mut [u8],
                lp: &mut [u8],
                r: usize,
                c: usize,
                b: usize,
                rect: u8| {
        match rect {
            1 => {
                push_rect_split(out, ap, lp, r, c, b, HORZ);
                let sub = SUBSIZE_HORZ[b] as usize;
                leaf(out, ap, lp, r, c, sub);
                leaf(out, ap, lp, r + MI_H[b] / 2, c, sub);
            }
            2 => {
                push_rect_split(out, ap, lp, r, c, b, VERT);
                let sub = SUBSIZE_VERT[b] as usize;
                leaf(out, ap, lp, r, c, sub);
                leaf(out, ap, lp, r, c + MI_W[b] / 2, sub);
            }
            _ => leaf(out, ap, lp, r, c, b),
        }
    };
    if !d.split64 {
        node(
            &mut out, above_pctx, left_pctx, mi_row, mi_col, B64, d.rect64,
        );
        return out;
    }
    push_rect_split(&mut out, above_pctx, left_pctx, mi_row, mi_col, B64, VERT);
    for dc in [0usize, 8usize] {
        push_rect_split(
            &mut out,
            above_pctx,
            left_pctx,
            mi_row,
            mi_col + dc,
            B32X64,
            HORZ,
        );
        for dr in [0usize, 8usize] {
            let (r, c) = (mi_row + dr, mi_col + dc);
            let child = (dr / 8) * 2 + dc / 8;
            if !d.split32[child] {
                node(&mut out, above_pctx, left_pctx, r, c, B32, d.rect32[child]);
                continue;
            }
            push_rect_split(&mut out, above_pctx, left_pctx, r, c, B32, VERT);
            for dc16 in [0usize, 4usize] {
                push_rect_split(&mut out, above_pctx, left_pctx, r, c + dc16, B16X32, HORZ);
                for dr16 in [0usize, 4usize] {
                    let (lr, lc) = (r + dr16, c + dc16);
                    let child16 = ((lr - mi_row) / 4) * 4 + (lc - mi_col) / 4;
                    if !d.split16[child16] {
                        node(
                            &mut out,
                            above_pctx,
                            left_pctx,
                            lr,
                            lc,
                            B16,
                            d.rect16[child16],
                        );
                        continue;
                    }
                    push_rect_split(&mut out, above_pctx, left_pctx, lr, lc, B16, VERT);
                    for dc8 in [0usize, 2usize] {
                        push_rect_split(&mut out, above_pctx, left_pctx, lr, lc + dc8, B8X16, HORZ);
                        for dr8 in [0usize, 2usize] {
                            leaf(&mut out, above_pctx, left_pctx, lr + dr8, lc + dc8, B8);
                        }
                    }
                }
            }
        }
    }
    out
}

/// Ops for a full interior SB coded as the free binary tree `codes` (see
/// [`crate::av2::leaf::GTree`]).
pub(crate) fn sb_gtree_ops(
    sb_row: usize,
    sb_col: usize,
    codes: &[u8],
    above_pctx: &mut [u8],
    left_pctx: &mut [u8],
) -> Vec<Op> {
    #[allow(clippy::too_many_arguments)]
    fn rec(
        codes: &[u8],
        at: &mut usize,
        r: usize,
        c: usize,
        b: usize,
        ap: &mut [u8],
        lp: &mut [u8],
        out: &mut Vec<Op>,
    ) {
        let code = codes[*at];
        *at += 1;
        match code {
            0 => {
                let ctx = raw_ctx(ap, lp, r, c, b) + BSIZE_MAP[b] * 4;
                out.push(Op::Leaf {
                    mi_row: r,
                    mi_col: c,
                    bw_mi: MI_W[b],
                    bh_mi: MI_H[b],
                    part_cdf: Some(DO_SPLIT_CDF0[ctx]),
                });
                update_pctx(ap, lp, r, c, b);
            }
            1 => {
                push_rect_split(out, ap, lp, r, c, b, HORZ);
                let sub = SUBSIZE_HORZ[b] as usize;
                rec(codes, at, r, c, sub, ap, lp, out);
                rec(codes, at, r + MI_H[b] / 2, c, sub, ap, lp, out);
            }
            _ => {
                push_rect_split(out, ap, lp, r, c, b, VERT);
                let sub = SUBSIZE_VERT[b] as usize;
                rec(codes, at, r, c, sub, ap, lp, out);
                rec(codes, at, r, c + MI_W[b] / 2, sub, ap, lp, out);
            }
        }
    }
    let mut out = Vec::new();
    let mut at = 0;
    rec(
        codes,
        &mut at,
        sb_row * 16,
        sb_col * 16,
        12,
        above_pctx,
        left_pctx,
        &mut out,
    );
    debug_assert_eq!(at, codes.len());
    out
}

fn push_rect_split(
    out: &mut Vec<Op>,
    above_pctx: &[u8],
    left_pctx: &[u8],
    row: usize,
    col: usize,
    bsize: usize,
    rect_type: u8,
) {
    out.push(Op::Split {
        do_split_cdf: DO_SPLIT_CDF0
            [raw_ctx(above_pctx, left_pctx, row, col, bsize) + BSIZE_MAP[bsize] * 4],
        square_cdf: 0,
    });
    let ctx = raw_ctx(above_pctx, left_pctx, row, col, bsize) + BSIZE_RECT_MAP[bsize] * 4;
    out.push(Op::RectType {
        cdf: RECT_TYPE_CDF0[ctx],
        val: rect_type as u32,
        ctx,
    });
}

/// A chroma-tree (SDP, plane 1) partition op: the split / direction bools carry
/// their context into the plane-1 CDF halves.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum ChromaOp {
    Split {
        ctx: usize,
        bit: u32,
    },
    Dir {
        ctx: usize,
        bit: u32,
    },
    Leaf {
        mi_row: usize,
        mi_col: usize,
        bw_mi: usize,
        bh_mi: usize,
    },
}

/// Partition of a node, 255 = NONE, else HORZ / VERT.
pub(crate) const PART_NONE: u8 = 255;

/// The luma tree's partition at a node, recovered from its leaf rectangles
/// `(mi_row, mi_col, bw_mi, bh_mi)` and the luma op generators' conventions:
/// boundary-implied direction first, NONE when a leaf covers the node exactly,
/// else square and wide nodes split VERT and tall ones HORZ.
pub(crate) fn luma_part_from_leaves(
    leaves: &[(usize, usize, usize, usize)],
    mi_row: usize,
    mi_col: usize,
    b: usize,
    mi_rows: usize,
    mi_cols: usize,
) -> u8 {
    let (w, h) = (MI_W[b], MI_H[b]);
    if leaves.contains(&(mi_row, mi_col, w, h)) {
        return PART_NONE;
    }
    if let Some(p) = implied_boundary(mi_row, mi_col, b, mi_rows, mi_cols) {
        return p;
    }
    let has_rows = mi_row + h / 2 < mi_rows;
    let has_cols = mi_col + w / 2 < mi_cols;
    if !(has_rows && has_cols) {
        // `decide_signaled` with NONE disallowed.
        let rimp = rect_type_implied(b);
        if let Some(r) = rimp {
            return r;
        }
        let horz_valid = SUBSIZE_HORZ[b] != INV;
        let vert_valid = SUBSIZE_VERT[b] != INV;
        return match (horz_valid, vert_valid) {
            (true, false) => HORZ,
            (false, true) => VERT,
            _ if !has_rows => HORZ,
            _ => VERT,
        };
    }
    // Rect leaf pairs: a node whose leaves never cross its horizontal midline
    // (but do cross the vertical one) was split HORZ, and vice versa. With
    // neither crossed (square quadrants) the tree builders' convention applies:
    // squares split VERT first, tall blocks HORZ.
    let (mr, mc) = (mi_row + h / 2, mi_col + w / 2);
    let inside = |&&(r, c, lw, lh): &&(usize, usize, usize, usize)| {
        r >= mi_row && r < mi_row + h && c >= mi_col && c < mi_col + w && (lw, lh) != (0, 0)
    };
    let crosses_h = leaves
        .iter()
        .filter(inside)
        .any(|&(r, _, _, lh)| r < mr && r + lh > mr);
    let crosses_v = leaves
        .iter()
        .filter(inside)
        .any(|&(_, c, lw, _)| c < mc && c + lw > mc);
    match (crosses_h, crosses_v) {
        (false, true) => HORZ,
        (true, false) => VERT,
        _ if h > w => HORZ,
        _ => VERT,
    }
}

/// Some luma leaf in `leaves` is rectangular.
pub(crate) fn luma_has_rects(leaves: &[(usize, usize, usize, usize)]) -> bool {
    leaves
        .iter()
        .any(|&(_, _, w, h)| w != h && (w, h) != (0, 0))
}

/// The `b`-sized luma node at (`mi_row`, `mi_col`) lies inside one luma leaf.
pub(crate) fn luma_node_covered(
    leaves: &[(usize, usize, usize, usize)],
    mi_row: usize,
    mi_col: usize,
    b: usize,
) -> bool {
    let (w, h) = (MI_W[b], MI_H[b]);
    leaves.iter().any(|&(r, c, lw, lh)| {
        r <= mi_row && c <= mi_col && r + lh >= mi_row + h && c + lw >= mi_col + w
    })
}

/// Luma-space `w`x`h` (mi) sub-block has a valid chroma block size under the
/// `(ssx, ssy)` subsampling.
fn chroma_valid(w: usize, h: usize, ss: (usize, usize)) -> bool {
    let (cw, ch) = ((4 * w) >> ss.0, (4 * h) >> ss.1);
    cw >= 4 && ch >= 4 && cw.max(ch) <= 8 * cw.min(ch)
}

/// Pre-order chroma-tree ops for one 64x64 SB (4:2:0, ext partitions off), a port
/// of the decoder's plane-1 `decode_partition`: forced NONE at luma-coord 8x8,
/// boundary-implied splits, the root inferred from the luma root, else a split
/// bool and (unless the size implies it) a direction bool from the plane-1 CDFs.
/// `want` gives the encoder's choice at signalled nodes (PART_NONE / HORZ / VERT).
/// Returns the ops and `sdp_cfl_disallowed` (chroma root direction family differs
/// from the luma root's, which turns CfL / MHCCP off for the whole SB).
#[allow(clippy::too_many_arguments)]
pub(crate) fn sdp_chroma_ops(
    sb_row: usize,
    sb_col: usize,
    mi_rows: usize,
    mi_cols: usize,
    ss: (usize, usize),
    above_c: &mut [u8],
    left_c: &mut [u8],
    luma_part: &dyn Fn(usize, usize, usize) -> u8,
    want: &dyn Fn(usize, usize, usize) -> u8,
) -> (Vec<ChromaOp>, bool) {
    let (r0, c0) = (sb_row * 16, sb_col * 16);
    // Luma root and whether every on-frame root child split the other way.
    let lroot = luma_part(r0, c0, 12);
    let children_other = lroot != PART_NONE && {
        let sub = if lroot == HORZ {
            SUBSIZE_HORZ[12]
        } else {
            SUBSIZE_VERT[12]
        } as usize;
        let other = if lroot == HORZ { VERT } else { HORZ };
        let kids: [(usize, usize); 2] = if lroot == HORZ {
            [(r0, c0), (r0 + 8, c0)]
        } else {
            [(r0, c0), (r0, c0 + 8)]
        };
        kids.iter()
            .filter(|&&(r, c)| r < mi_rows && c < mi_cols)
            .all(|&(r, c)| luma_part(r, c, sub) == other)
    };
    let mut out = Vec::new();
    let mut root_dir: Option<u8> = None;
    struct Walk<'a> {
        mi_rows: usize,
        mi_cols: usize,
        above_c: &'a mut [u8],
        left_c: &'a mut [u8],
        want: &'a dyn Fn(usize, usize, usize) -> u8,
        out: &'a mut Vec<ChromaOp>,
        root_inferred: Option<u8>,
        ss: (usize, usize),
    }
    fn walk_c(
        st: &mut Walk<'_>,
        r: usize,
        c: usize,
        b: usize,
        root_dir: &mut Option<u8>,
        is_root: bool,
    ) {
        if r >= st.mi_rows || c >= st.mi_cols {
            return;
        }
        let (w, h) = (MI_W[b], MI_H[b]);
        let (cw4, ch4) = ((w >> st.ss.0).max(1), (h >> st.ss.1).max(1));
        let has_rows = r + h / 2 < st.mi_rows;
        let has_cols = c + w / 2 < st.mi_cols;
        let mut part: Option<u8> = None;
        if cw4.max(ch4) == 1 || b == 3 || (SUBSIZE_HORZ[b] == INV && SUBSIZE_VERT[b] == INV) {
            part = Some(PART_NONE);
        } else if !has_rows || !has_cols {
            if w == h {
                part = Some(if !has_rows { HORZ } else { VERT });
            } else if w > h {
                if !has_cols || st.mi_rows <= r + h / 4 {
                    part = Some(VERT);
                }
            } else if !has_rows || st.mi_cols <= c + w / 4 {
                part = Some(HORZ);
            }
        }
        if part.is_none() && is_root && st.root_inferred.is_some() {
            part = st.root_inferred;
        }
        let part = match part {
            Some(p) => p,
            None => {
                let ctx1 = raw_ctx(st.above_c, st.left_c, r, c, b);
                let want = (st.want)(r, c, b);
                let split = if !has_rows || !has_cols {
                    true
                } else {
                    let s = want != PART_NONE;
                    st.out.push(ChromaOp::Split {
                        ctx: ctx1 + BSIZE_MAP[b] * 4,
                        bit: s as u32,
                    });
                    s
                };
                if !split {
                    PART_NONE
                } else if cw4.min(ch4) == 1 {
                    if cw4 > ch4 { VERT } else { HORZ }
                } else if b == 4 || b == 21 {
                    HORZ
                } else if b == 5 || b == 22 {
                    VERT
                } else {
                    let v_ok = SUBSIZE_VERT[b] != INV && chroma_valid(w / 2, h, st.ss);
                    let h_ok = SUBSIZE_HORZ[b] != INV && chroma_valid(w, h / 2, st.ss);
                    if v_ok && h_ok {
                        let d = if want == HORZ { HORZ } else { VERT };
                        st.out.push(ChromaOp::Dir {
                            ctx: ctx1 + BSIZE_RECT_MAP[b] * 4,
                            bit: d as u32,
                        });
                        d
                    } else if v_ok {
                        VERT
                    } else {
                        HORZ
                    }
                }
            }
        };
        if is_root {
            *root_dir = Some(part);
        }
        if part == PART_NONE {
            st.out.push(ChromaOp::Leaf {
                mi_row: r,
                mi_col: c,
                bw_mi: w,
                bh_mi: h,
            });
            update_pctx(st.above_c, st.left_c, r, c, b);
        } else if part == HORZ {
            let sub = SUBSIZE_HORZ[b] as usize;
            walk_c(st, r, c, sub, root_dir, false);
            walk_c(st, r + h / 2, c, sub, root_dir, false);
        } else {
            let sub = SUBSIZE_VERT[b] as usize;
            walk_c(st, r, c, sub, root_dir, false);
            walk_c(st, r, c + w / 2, sub, root_dir, false);
        }
    }
    let root_inferred = if lroot == PART_NONE {
        Some(PART_NONE)
    } else if children_other {
        Some(lroot)
    } else {
        None
    };
    let mut st = Walk {
        mi_rows,
        mi_cols,
        above_c,
        left_c,
        want,
        out: &mut out,
        root_inferred,
        ss,
    };
    walk_c(&mut st, r0, c0, 12, &mut root_dir, true);
    let cfl_disallowed = match root_dir {
        Some(d) if d != PART_NONE => lroot == PART_NONE || d != lroot,
        _ => false,
    };
    (out, cfl_disallowed)
}
