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

use crate::av2::cdfs_qctx::*;
use crate::av2::cdfx_4tx::*;
use crate::av2::entropy::RangeEncoder;
use crate::av2::lossless::SCAN_4X4;
use crate::av2::tables::*;
use crate::av2::tables_tx32::*;

/// A coefficient as fed to the coders.
pub(crate) type Coeff = (usize, i32);

/// Append an escape entry to a CDF so the escape symbol has a coding interval.
/// floor(log2(x)) for x >= 1.
fn floor_log2(x: u32) -> u32 {
    31 - x.leading_zeros()
}

/// Luma low/high-frequency token context from already-coded neighbour levels.
/// Returns `(base_context, hi_range_context)`.
fn luma_coeff_context(levels: &[i32], rc: i32, xy: i32) -> (usize, usize) {
    let low_freq = xy < 4;
    let mut limit: i32 = if low_freq { 5 } else { 3 };
    let neighbour = |dy: i32, dx: i32| -> i32 { levels[(rc + dy * STRIDE + dx) as usize] };
    let mut low_mag = 0i32;
    let mut hi_mag = 0i32;
    for (dy, dx) in [(0, 1), (1, 0), (1, 1)] {
        let v = neighbour(dy, dx);
        low_mag += v.min(limit);
        hi_mag += v.min(5);
    }
    low_mag += neighbour(0, 2).min(limit) + neighbour(2, 0).min(limit);

    let offset;
    if low_freq {
        offset = if xy == 0 {
            0
        } else if xy < 2 {
            9
        } else {
            16
        };
        limit = if xy == 0 {
            8
        } else if xy < 2 {
            6
        } else {
            4
        };
    } else {
        offset = if xy < 6 {
            0
        } else if xy < 8 {
            5
        } else {
            10
        };
        limit = 4;
    }
    let hi_range_ctx = (if low_freq && xy > 0 { 7 } else { 0 }) + ((hi_mag + 1) >> 1).min(6);
    let base_ctx = offset + ((low_mag + 1) >> 1).min(limit);
    (base_ctx as usize, hi_range_ctx as usize)
}

/// Chroma (2D DCT) token context. `plane_offset` is 0 for U and 4 for V.
/// Returns `(base_context, hi_range_context)`.
fn chroma_coeff_context(levels: &[i32], rc: i32, xy: i32, plane_offset: usize) -> (usize, usize) {
    let add_limit: i32 = if xy < 1 { 5 } else { 3 };
    let neighbour = |dy: i32, dx: i32| -> i32 { levels[(rc + dy * STRIDE + dx) as usize] };
    let (right, below, below_right) = (neighbour(0, 1), neighbour(1, 0), neighbour(1, 1));
    let low_mag = right.min(add_limit) + below.min(add_limit) + below_right.min(add_limit);
    let hi_mag = right.min(5) + below.min(5) + below_right.min(5);
    let base_ctx = plane_offset + (((low_mag + 1) >> 1).min(3)) as usize;
    let hi_range_ctx = (((hi_mag + 1) >> 1).min(3)) as usize;
    (base_ctx, hi_range_ctx)
}

// ----- high-range residual (avm adaptive Truncated-Rice) -----------------------
//
// avm codes the high-range part of a coefficient level (the amount above
// max_base_range) with an adaptive Truncated-Rice code, NOT exp-Golomb. The Rice
// parameter m is derived from a running average of previously coded HR values via
// `adaptive_table`, and the residual `hr = mag - max_base_range` and the running
// average update `(avg + hr) >> 1` are threaded across the block in scan order.
// (av2/common/hr_coding.c, av2/encoder/encodetxb.c)

fn get_adaptive_param(ctx: i32) -> u32 {
    const TABLE: [i32; 5] = [4, 8, 16, 32, 64];
    let mut m = 0usize;
    while m < TABLE.len() && ctx >= TABLE[m] {
        m += 1;
    }
    m as u32 + 1
}

/// avm `write_exp_golomb`: order-k Exp-Golomb of `level` (bypass bits, MSB first).
fn write_exp_golomb(enc: &mut RangeEncoder, level: u32, k: u32) {
    let x = level + (1 << k);
    let length = floor_log2(x) + 1; // get_msb(x) + 1
    for _ in 0..(length - 1 - k) {
        enc.encode_bypass(0, 1);
    }
    for b in (0..length).rev() {
        enc.encode_bypass((x >> b) & 1, 1);
    }
}

/// avm `write_truncated_rice(level, m, k = m+1, cmax = min(m+4,6))`.
fn write_truncated_rice(enc: &mut RangeEncoder, level: u32, m: u32, k: u32, cmax: u32) {
    let q = level >> m;
    if q >= cmax {
        for _ in 0..cmax {
            enc.encode_bypass(0, 1);
        }
        write_exp_golomb(enc, level - (cmax << m), k);
    } else {
        for _ in 0..q {
            enc.encode_bypass(0, 1);
        }
        enc.encode_bypass(1, 1);
        let mask = (1u32 << m) - 1;
        for b in (0..m).rev() {
            enc.encode_bypass(((level & mask) >> b) & 1, 1);
        }
    }
}

/// Code one high-range residual `hr` with the adaptive Rice parameter from the
/// running average, and return the updated average. Mirrors avm `write_high_range`
/// (tcq disabled): `m = get_adaptive_param(avg)`, then truncated-Rice.
fn encode_high_range(enc: &mut RangeEncoder, hr: u32, running_avg: i32) -> i32 {
    let m = get_adaptive_param(running_avg);
    write_truncated_rice(enc, hr, m, m + 1, (m + 4).min(6));
    (running_avg + hr as i32) >> 1
}

/// Luma base-range symbol (high or low frequency).
fn encode_luma_base_range(
    enc: &mut RangeEncoder,
    level: u32,
    hi_range_ctx: usize,
    high_freq: bool,
) {
    let limit = if high_freq { 3u32 } else { 5u32 };
    let over = level - limit;
    let cdf = if high_freq {
        &BR_TOK_HF_QC[enc.qc][hi_range_ctx]
    } else {
        &BR_TOK_QC[enc.qc][hi_range_ctx]
    };
    if over <= 2 {
        enc.encode_symbol(cdf, over as usize, 3);
    } else {
        enc.encode_symbol_esc(cdf, 3, 3);
    }
}

/// Chroma high-frequency base-range symbol (limit 3; golomb handles the tail).
fn encode_chroma_base_range(enc: &mut RangeEncoder, magnitude: u32, hi_range_ctx: usize) {
    let over = magnitude - 3;
    let cdf = &CHROMA_BR_TOK_HF_QC[enc.qc][hi_range_ctx];
    if over <= 2 {
        enc.encode_symbol(cdf, over as usize, 3);
    } else {
        enc.encode_symbol_esc(cdf, 3, 3);
    }
}

// ----- shared end-of-block coding ----------------------------------------------

/// Encode the end-of-block position using the given bin/hi-bit CDFs.
fn encode_eob(enc: &mut RangeEncoder, eob: usize, eob_bin: &[u16], eob_hi_bit: u16) {
    if eob <= 1 {
        enc.encode_symbol(eob_bin, eob, 7);
        return;
    }
    let mut bin = 2usize;
    while (2usize << (bin - 1)) <= eob {
        bin += 1;
    }
    if bin <= 6 {
        enc.encode_symbol(eob_bin, bin, 7);
    } else {
        enc.encode_symbol_esc(eob_bin, 7, 7);
        enc.encode_bypass((bin - 7) as u32, 2);
    }
    let extra_bits = bin - 2;
    let hi = (eob >> extra_bits) & 1;
    enc.encode_bool(eob_hi_bit as u32, hi as u32);
    if extra_bits > 0 {
        let low = eob & ((1 << extra_bits) - 1);
        for k in (0..extra_bits).rev() {
            enc.encode_bypass(((low >> k) & 1) as u32, 1);
        }
    }
}

fn level_at(coeffs: &[Coeff], scan_pos: usize) -> i32 {
    coeffs
        .iter()
        .find(|&&(s, _)| s == scan_pos)
        .map(|&(_, l)| l)
        .unwrap_or(0)
}

// ----- luma block ---------------------------------------------------------------

/// Encode the intra mode information that precedes a luma block's coefficients.
fn encode_intra_modes(
    enc: &mut RangeEncoder,
    mode_idx: usize,
    has_chroma: bool,
    lossless: bool,
    partition_cdf: Option<u32>,
) {
    // do_split bool (=0, PARTITION_NONE) with the leaf's per-bsize/context cdf.
    // None for non-partition-point leaves (4x4 / narrow ext blocks), which read
    // no partition bit at all.
    if let Some(cdf) = partition_cdf {
        enc.encode_bool(cdf, 0);
    }
    if lossless {
        // Lossless intra reads use_dpcm_y (dpcm_cdf, AVM_CDF2(16384)) before the luma
        // mode. 0 = no DPCM, then the normal intra-mode path follows.
        enc.encode_bool(16384, 0);
    }
    enc.encode_symbol(&[3905, 1746, 1044], 0, 3); // intra_y mode set 0
    enc.encode_symbol(&[17593, 12693, 11040, 8670, 6363, 5113, 3908], mode_idx, 7);
    if has_chroma {
        if lossless {
            // Lossless intra also reads use_dpcm_uv (dpcm_uv_cdf, AVM_CDF2(16384))
            // before the chroma mode. 0 = no DPCM.
            enc.encode_bool(16384, 0);
        }
        // intra_uv_mode = 0 (DC chroma); CfL disabled so no extra bit.
        enc.encode_symbol(&[23405, 11811, 9903, 8015, 6357, 4785, 2340], 0, 7);
    }
}

/// Stored per-coefficient data for the sign/golomb pass: `(rc, x, y, mag, high_freq)`.
type LumaStored = (i32, i32, i32, i32, bool); // (rc, x, y, signed_level, high_freq)

/// Sign + golomb residual pass for luma (DC sign is adaptive, AC is bypass).
fn encode_luma_signs(
    enc: &mut RangeEncoder,
    _coeffs: &[Coeff],
    stored: &[LumaStored],
    dc_sign_ctx: usize,
) {
    let mut running_avg = 0i32;
    for &(_rc, x, y, level, high_freq) in stored {
        if level == 0 {
            continue;
        }
        let mag = level.unsigned_abs();
        let sign = if level < 0 { 1u32 } else { 0u32 };
        if x == 0 && y == 0 {
            enc.encode_bool(DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32, sign);
        } else {
            enc.encode_bypass(sign, 1);
        }
        let max_base_range = if high_freq { 6 } else { 8 };
        if mag >= max_base_range {
            running_avg = encode_high_range(enc, mag - max_base_range, running_avg);
        }
    }
}

/// Stored per-coefficient data for chroma: `(rc, mag, is_dc)`.
type ChromaStored = (i32, u32, bool);

/// Reverse-scan token pass for chroma; fills the neighbour-level grid.
/// DC is low-frequency; all AC is high-frequency.
fn encode_chroma_tokens(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    plane_offset: usize,
) -> Vec<ChromaStored> {
    let mut levels = vec![0i32; (STRIDE as usize) * 40 + 64];
    let mut stored: Vec<ChromaStored> = vec![];
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = SCAN[scan_pos] as i32;
        let x = rc >> 5;
        let y = rc & 31;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let is_dc = scan_pos == 0;
        if is_eob && is_dc {
            // DC-only block: low-frequency eob token.
            if mag <= 4 {
                enc.encode_symbol(&CHROMA_EOB_TOK_LF_QC[enc.qc][0], (mag - 1) as usize, 4);
            } else {
                enc.encode_symbol_esc(&CHROMA_EOB_TOK_LF_QC[enc.qc][0], 4, 4);
            }
        } else if is_eob {
            // AC eob token (high frequency); base-range context is 0.
            let eob_ctx = 1 + (eob > (2 << 6)) as usize + (eob > (4 << 6)) as usize;
            if mag <= 2 {
                enc.encode_symbol(
                    &CHROMA_EOB_TOK_HF_QC[enc.qc][eob_ctx],
                    (mag - 1) as usize,
                    2,
                );
            } else {
                enc.encode_symbol_esc(&CHROMA_EOB_TOK_HF_QC[enc.qc][eob_ctx], 2, 2);
                encode_chroma_base_range(enc, mag, 0);
            }
        } else if is_dc {
            // DC base token (low frequency).
            let (base_ctx, _) = chroma_coeff_context(&levels, rc, 0, plane_offset);
            if mag <= 4 {
                enc.encode_symbol(&CHROMA_BASE_TOK_LF_QC[enc.qc][base_ctx], mag as usize, 5);
            } else {
                enc.encode_symbol_esc(&CHROMA_BASE_TOK_LF_QC[enc.qc][base_ctx], 5, 5);
            }
        } else {
            // AC base token (high frequency) with base-range escape.
            let (base_ctx, hi_range_ctx) = chroma_coeff_context(&levels, rc, x + y, plane_offset);
            if mag <= 2 {
                enc.encode_symbol(&CHROMA_BASE_TOK_HF_QC[enc.qc][base_ctx], mag as usize, 3);
            } else {
                enc.encode_symbol_esc(&CHROMA_BASE_TOK_HF_QC[enc.qc][base_ctx], 3, 3);
                encode_chroma_base_range(enc, mag, hi_range_ctx);
            }
        }
        levels[rc as usize] = (mag as i32).min(5);
        stored.push((rc, mag, is_dc));
    }
    stored
}

/// Sign + golomb residual pass for chroma (all signs are bypass).
fn encode_chroma_signs(enc: &mut RangeEncoder, coeffs: &[Coeff], stored: &[ChromaStored]) {
    let mut running_avg = 0i32;
    for &(rc, mag, is_dc) in stored {
        if mag == 0 {
            continue;
        }
        let scan_pos = SCAN.iter().position(|&s| s as i32 == rc).unwrap();
        let level = level_at(coeffs, scan_pos);
        enc.encode_bypass(if level < 0 { 1 } else { 0 }, 1);
        let max_base_range = if is_dc { 5u32 } else { 6u32 };
        if mag >= max_base_range {
            running_avg = encode_high_range(enc, mag - max_base_range, running_avg);
        }
    }
}

/// Encode one chroma plane block. `skip_cdf` is the layout/neighbour-dependent
/// all-zero CDF and `is_u_plane` selects the U (offset 0) or V (offset 4) context.
pub(crate) fn encode_chroma_block(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    is_u_plane: bool,
) {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.encode_bool(skip_cdf, 1);
        return;
    }
    enc.encode_bool(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(
        enc,
        eob,
        &CHROMA_EOB_BIN_QC[enc.qc],
        CHROMA_EOB_HI_BIT_QC[enc.qc],
    );
    let plane_offset = if is_u_plane { 0 } else { 4 };
    let stored = encode_chroma_tokens(enc, &nonzero, eob, plane_offset);
    encode_chroma_signs(enc, &nonzero, &stored);
}

// ----- TX_32X32 luma split path -------------------------------------------------

/// One 32x32 luma token (ctx=3 tables). Mirrors `encode_luma_token`.
fn encode_luma32_token(
    enc: &mut RangeEncoder,
    level: u32,
    is_eob: bool,
    base_ctx: usize,
    hi_range_ctx: usize,
    high_freq: bool,
) -> i32 {
    let limit = if high_freq { 3 } else { 5 };
    if !high_freq {
        if is_eob {
            if level <= 4 {
                enc.encode_symbol(
                    &LUMA32_EOB_TOK_LF_QC[enc.qc][base_ctx],
                    (level - 1) as usize,
                    4,
                );
            } else {
                enc.encode_symbol_esc(&LUMA32_EOB_TOK_LF_QC[enc.qc][base_ctx], 4, 4);
                encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
            }
        } else if level <= 4 {
            enc.encode_symbol(&LUMA32_BASE_TOK_LF_QC[enc.qc][base_ctx], level as usize, 5);
        } else {
            enc.encode_symbol_esc(&LUMA32_BASE_TOK_LF_QC[enc.qc][base_ctx], 5, 5);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if is_eob {
        if level <= 2 {
            enc.encode_symbol(
                &LUMA32_EOB_TOK_HF_QC[enc.qc][base_ctx],
                (level - 1) as usize,
                2,
            );
        } else {
            enc.encode_symbol_esc(&LUMA32_EOB_TOK_HF_QC[enc.qc][base_ctx], 2, 2);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if level <= 2 {
        enc.encode_symbol(&LUMA32_BASE_TOK_HF_QC[enc.qc][base_ctx], level as usize, 3);
    } else {
        enc.encode_symbol_esc(&LUMA32_BASE_TOK_HF_QC[enc.qc][base_ctx], 3, 3);
        encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
    }
    if (level as i32) < limit {
        level as i32
    } else {
        limit + (level as i32 - limit).min(3)
    }
}

fn encode_luma32_tokens(enc: &mut RangeEncoder, coeffs: &[Coeff], eob: usize) -> Vec<LumaStored> {
    let mut levels = vec![0i32; (STRIDE as usize) * 40 + 64];
    let mut stored: Vec<LumaStored> = vec![];
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = SCAN[scan_pos] as i32;
        let x = rc >> 5;
        let y = rc & 31;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let high_freq = scan_pos >= LUMA_HI_TO_LOW;
        let (base_ctx, hi_range_ctx) = if is_eob {
            if eob == 0 {
                (0usize, 0usize)
            } else {
                (
                    1 + (eob > (2 << 6)) as usize + (eob > (4 << 6)) as usize,
                    if high_freq { 0 } else { 7 },
                )
            }
        } else {
            luma_coeff_context(&levels, rc, x + y)
        };
        let stored_level = encode_luma32_token(enc, mag, is_eob, base_ctx, hi_range_ctx, high_freq);
        levels[rc as usize] = stored_level;
        stored.push((rc, x, y, level, high_freq));
    }
    stored
}

/// Encode one 32x32 luma transform unit. `skip_cdf` = CHROMA_SKIP_TX32[sctx];
/// `dc_sign_ctx` indexes DC_SIGN_QC[enc.qc]. Returns the cumulative level (capped 63) for context.
pub(crate) fn encode_luma_tu32(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
) -> u32 {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.encode_bool(skip_cdf, 1);
        return 0;
    }
    enc.encode_bool(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, &EOB_BIN_QC[enc.qc], EOB_HI_BIT_QC[enc.qc]);
    let stored = encode_luma32_tokens(enc, &nonzero, eob);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

/// Encode a 64x64 intra luma block split into four 32x32 TUs (raster order).
/// `skip_cdfs`/`dc_sign_ctxs` are the per-TU contexts. Returns the four cul_levels.
pub(crate) fn encode_luma_block_split(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    has_chroma: bool,
) -> [u32; 4] {
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(12276));
    enc.encode_bool(TX_SPLIT_64 as u32, 1); // tx_split = 1
    enc.encode_symbol(&TX_PART_2D_64, 0, 6); // tx_part symbol 0 = SPLIT
    let mut cul = [0u32; 4];
    for i in 0..4 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    cul
}

// Padded levels grid: bwl=2, stride = (1<<2)+TX_PAD_HOR(4) = 8. get_padded_idx.
#[inline]
fn pidx(rc: usize) -> usize {
    rc + (rc >> 2) * 4
}

// get_lower_levels_ctx_eob(bwl=2, height=4, scan_idx): height<<bwl = 16 -> /8=2, /4=4.
#[inline]
fn ctx_eob4(c: usize) -> usize {
    if c == 0 {
        0
    } else if c <= 2 {
        1
    } else if c <= 4 {
        2
    } else {
        3
    }
}

// get_lower_levels_ctx_lf_2d (luma) — also handles the DC (coeff_idx==0) branch.
fn ctx_lf_2d(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag = levels[b + 1].min(5) as i32
        + levels[b + 8].min(5) as i32
        + levels[b + 9].min(5) as i32
        + levels[b + 2].min(5) as i32
        + levels[b + 16].min(5) as i32;
    let ctx = (mag + 1) >> 1;
    let row = (rc >> 2) as i32;
    let col = (rc & 3) as i32;
    if rc == 0 {
        return ctx.min(8) as usize;
    }
    if row + col < 2 {
        return (ctx.min(6) + 9) as usize;
    }
    (ctx.min(4) + 16) as usize
}

// get_lower_levels_ctx_2d (luma, high-frequency region).
fn ctx_2d(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag = levels[b + 1].min(3) as i32
        + levels[b + 8].min(3) as i32
        + levels[b + 9].min(3) as i32
        + levels[b + 2].min(3) as i32
        + levels[b + 16].min(3) as i32;
    let ctx = ((mag + 1) >> 1).min(4);
    let row = (rc >> 2) as i32;
    let col = (rc & 3) as i32;
    if row + col < 6 {
        ctx as usize
    } else if row + col < 8 {
        (ctx + 5) as usize
    } else {
        (ctx + 10) as usize
    }
}

// get_br_lf_ctx (2D): MAX_VAL_BR_CTX=5. DC (rc==0) -> mag (no +7); else mag+7.
fn br_lf_ctx(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    let m = ((mag + 1) >> 1).min(6);
    if rc == 0 {
        m as usize
    } else {
        (m + 7) as usize
    }
}

// get_br_ctx_2d (high-frequency): mag (0..6), no offset.
fn br_hf_ctx(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    (((mag + 1) >> 1).min(6)) as usize
}

// br symbol (BR_CDF_SIZE=4 outcomes -> nsyms=3), then golomb tail in the sign pass.
fn encode_br4(enc: &mut RangeEncoder, level: u32, ctx: usize, lf: bool) {
    let limit = if lf { 5u32 } else { 3u32 };
    let over = level - limit;
    let cdf = if lf { &BR_LF_Q0[ctx] } else { &BR_Q0[ctx] };
    if over <= 2 {
        enc.encode_symbol(cdf, over as usize, 3);
    } else {
        enc.encode_symbol_esc(cdf, 3, 3);
    }
}

// One coefficient's base(+br) symbols. Returns the stored (capped) level for context.
fn encode_luma4_token(
    enc: &mut RangeEncoder,
    level: u32,
    is_eob: bool,
    base_ctx: usize,
    hi_ctx: usize,
    lf: bool,
) -> i32 {
    let limit = if lf { 5 } else { 3 };
    if lf {
        if is_eob {
            if level <= 4 {
                enc.encode_symbol(&BASE_LF_EOB_TX4_Q0[base_ctx], (level - 1) as usize, 4);
            } else {
                enc.encode_symbol_esc(&BASE_LF_EOB_TX4_Q0[base_ctx], 4, 4);
                encode_br4(enc, level, hi_ctx, true);
            }
        } else if level <= 4 {
            enc.encode_symbol(&BASE_LF_TX4_Q0[base_ctx][0], level as usize, 5);
        } else {
            enc.encode_symbol_esc(&BASE_LF_TX4_Q0[base_ctx][0], 5, 5);
            encode_br4(enc, level, hi_ctx, true);
        }
    } else if is_eob {
        if level <= 2 {
            enc.encode_symbol(&BASE_EOB_TX4_Q0[base_ctx], (level - 1) as usize, 2);
        } else {
            enc.encode_symbol_esc(&BASE_EOB_TX4_Q0[base_ctx], 2, 2);
            encode_br4(enc, level, hi_ctx, false);
        }
    } else if level <= 2 {
        enc.encode_symbol(&BASE_TX4_Q0[base_ctx][0], level as usize, 3);
    } else {
        enc.encode_symbol_esc(&BASE_TX4_Q0[base_ctx][0], 3, 3);
        encode_br4(enc, level, hi_ctx, false);
    }
    if (level as i32) < limit {
        level as i32
    } else {
        limit + (level as i32 - limit).min(3)
    }
}

// EOB position: eob_pt via eob_flag_cdf16 (5 outcomes -> nsyms 4), then extra bits.
fn encode_eob_4x4(enc: &mut RangeEncoder, eob_count: usize, plctx: usize) {
    let (pt, start, obits): (usize, usize, usize) = match eob_count {
        1 => (1, 1, 0),
        2 => (2, 2, 0),
        3 | 4 => (3, 3, 1),
        5..=8 => (4, 5, 2),
        _ => (5, 9, 3),
    };
    let cdf = &EOB16_Q0[plctx];
    if pt - 1 <= 3 {
        enc.encode_symbol(cdf, pt - 1, 4);
    } else {
        enc.encode_symbol_esc(cdf, 4, 4);
    }
    if obits > 0 {
        let extra = eob_count - start;
        let msb = (extra >> (obits - 1)) & 1;
        enc.encode_bool(EOB_HI_BIT_QC[enc.qc] as u32, msb as u32);
        for k in (0..obits - 1).rev() {
            enc.encode_bypass(((extra >> k) & 1) as u32, 1);
        }
    }
}

// Pass A: base/br for c = eob-1 .. 0 (eob coeff first, DC last), filling the level grid.
fn encode_luma4_tokens(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob_count: usize,
) -> ([LumaStored; 16], usize) {
    let mut levels = [0u8; 64];
    let mut full = [0i32; 16];
    for &(p, l) in coeffs {
        full[p] = l;
    }
    let mut stored = [(0i32, 0i32, 0i32, 0i32, false); 16];
    let mut ns = 0usize;
    let last = eob_count - 1;
    for c in (0..=last).rev() {
        let level = full[c];
        let rc = SCAN_4X4[c] as usize;
        let row = rc >> 2;
        let col = rc & 3;
        let mag = level.unsigned_abs();
        let is_eob = c == last;
        let lf = (row + col) < 4;
        let (base_ctx, hi_ctx) = if is_eob {
            let hctx = if lf { if rc == 0 { 0 } else { 7 } } else { 0 };
            (ctx_eob4(c), hctx)
        } else if lf {
            (ctx_lf_2d(&levels, rc), br_lf_ctx(&levels, rc))
        } else {
            (ctx_2d(&levels, rc), br_hf_ctx(&levels, rc))
        };
        let sl = encode_luma4_token(enc, mag, is_eob, base_ctx, hi_ctx, lf);
        levels[pidx(rc)] = sl as u8;
        stored[ns] = (rc as i32, col as i32, row as i32, level, !lf);
        ns += 1;
    }
    (stored, ns)
}

// Pass B: sign (DC via dc_sign cdf, else bypass) then golomb tail, in reverse scan order.
fn encode_luma4_signs(
    enc: &mut RangeEncoder,
    _coeffs: &[Coeff],
    stored: &[LumaStored],
    dc_sign_ctx: usize,
) {
    let mut running_avg = 0i32;
    for &(_rc, x, y, level, high_freq) in stored {
        if level == 0 {
            continue;
        }
        let mag = level.unsigned_abs();
        let sign = if level < 0 { 1u32 } else { 0u32 };
        if x == 0 && y == 0 {
            enc.encode_bool(DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32, sign);
        } else {
            enc.encode_bypass(sign, 1);
        }
        let max_base_range = if high_freq { 6 } else { 8 };
        if mag >= max_base_range {
            running_avg = encode_high_range(enc, mag - max_base_range, running_avg);
        }
    }
}

/// Encode one 4x4 lossless luma transform unit (skip → eob_pt → base/br → sign/golomb).
pub(crate) fn encode_luma_tu4(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
) -> u32 {
    if coeffs.is_empty() {
        enc.encode_bool(skip_cdf, 1);
        return 0;
    }
    enc.encode_bool(skip_cdf, 0);
    let eob_count = coeffs.iter().map(|&(s, _)| s).max().unwrap() + 1; // avm eob = max scan idx + 1
    encode_eob_4x4(enc, eob_count, 0); // luma plane ctx 0
    let (stored, ns) = encode_luma4_tokens(enc, coeffs, eob_count);
    encode_luma4_signs(enc, coeffs, &stored[..ns], dc_sign_ctx);
    coeffs
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

// ----- TX_4X4 lossless CHROMA path -----------------------------------------------
// avm chroma: LF_2D_LIM_UV=1 so only the DC is lf (base_lf_uv, no br); every other
// position is hf (base_uv + br_uv). Signs are all bypass (no dc_sign cdf). Golomb
// threshold is 5 for lf (LF_NUM_BASE_LEVELS+1), 6 for hf. plane ctx for eob = 2.
// Context fns use 3 neighbours; U uses ctx 0..3, V adds +4.

fn ctx_lf_2d_chroma(levels: &[u8], rc: usize, voff: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    (((mag + 1) >> 1).min(3)) as usize + voff
}
fn ctx_2d_chroma(levels: &[u8], rc: usize, voff: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(3) as i32 + levels[b + 8].min(3) as i32 + levels[b + 9].min(3) as i32;
    (((mag + 1) >> 1).min(3)) as usize + voff
}
fn br_ctx_2d_chroma(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    (((mag + 1) >> 1).min(3)) as usize
}
fn encode_br_uv(enc: &mut RangeEncoder, level: u32, ctx: usize) {
    let over = level - 3;
    let cdf = &BR_UV_Q0[ctx];
    if over <= 2 {
        enc.encode_symbol(cdf, over as usize, 3);
    } else {
        enc.encode_symbol_esc(cdf, 3, 3);
    }
}
fn encode_chroma4_token(
    enc: &mut RangeEncoder,
    level: u32,
    is_eob: bool,
    base_ctx: usize,
    hi_ctx: usize,
    lf: bool,
) -> i32 {
    if lf {
        if is_eob {
            if level <= 4 {
                enc.encode_symbol(&BASE_LF_EOB_UV_Q0[base_ctx], (level - 1) as usize, 4);
            } else {
                enc.encode_symbol_esc(&BASE_LF_EOB_UV_Q0[base_ctx], 4, 4);
            }
        } else if level <= 4 {
            enc.encode_symbol(&BASE_LF_UV_Q0[base_ctx], level as usize, 5);
        } else {
            enc.encode_symbol_esc(&BASE_LF_UV_Q0[base_ctx], 5, 5);
        }
        if level <= 4 { level as i32 } else { 5 } // chroma lf: no br, capped at 5
    } else {
        if is_eob {
            if level <= 2 {
                enc.encode_symbol(&BASE_EOB_UV_Q0[base_ctx], (level - 1) as usize, 2);
            } else {
                enc.encode_symbol_esc(&BASE_EOB_UV_Q0[base_ctx], 2, 2);
                encode_br_uv(enc, level, hi_ctx);
            }
        } else if level <= 2 {
            enc.encode_symbol(&BASE_UV_Q0[base_ctx], level as usize, 3);
        } else {
            enc.encode_symbol_esc(&BASE_UV_Q0[base_ctx], 3, 3);
            encode_br_uv(enc, level, hi_ctx);
        }
        if (level as i32) <= 2 {
            level as i32
        } else {
            3 + (level as i32 - 3).min(3)
        }
    }
}

/// Encode one 4x4 lossless chroma transform unit. `plane_v` selects U(false)/V(true)
/// (context offset only). Returns the cumulative level (capped 63) for skip context.
pub(crate) fn encode_chroma_tu4(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    plane_v: bool,
) -> u32 {
    if coeffs.is_empty() {
        enc.encode_bool(skip_cdf, 1);
        return 0;
    }
    enc.encode_bool(skip_cdf, 0);
    let eob_count = coeffs.iter().map(|&(s, _)| s).max().unwrap() + 1;
    encode_eob_4x4(enc, eob_count, 2); // chroma plane ctx = 2
    let voff = if plane_v { 4 } else { 0 };
    let mut levels = [0u8; 64];
    let mut full = [0i32; 16];
    for &(p, l) in coeffs {
        full[p] = l;
    }
    let mut stored = [(0i32, 0i32, 0i32, 0i32, false); 16];
    let mut ns = 0usize;
    let last = eob_count - 1;
    for c in (0..=last).rev() {
        let level = full[c];
        let rc = SCAN_4X4[c] as usize;
        let row = rc >> 2;
        let col = rc & 3;
        let mag = level.unsigned_abs();
        let is_eob = c == last;
        let lf = (row + col) < 1; // chroma: only the DC is low-frequency
        let (base_ctx, hi_ctx) = if is_eob {
            (ctx_eob4(c), 0)
        } else if lf {
            (ctx_lf_2d_chroma(&levels, rc, voff), 0)
        } else {
            (
                ctx_2d_chroma(&levels, rc, voff),
                br_ctx_2d_chroma(&levels, rc),
            )
        };
        let sl = encode_chroma4_token(enc, mag, is_eob, base_ctx, hi_ctx, lf);
        levels[pidx(rc)] = sl as u8;
        stored[ns] = (rc as i32, col as i32, row as i32, level, !lf);
        ns += 1;
    }
    // signs (always bypass for chroma) + golomb (lf threshold 5, hf 6)
    let mut running_avg = 0i32;
    for &(_, _, _, level, high_freq) in &stored[..ns] {
        if level == 0 {
            continue;
        }
        let mag = level.unsigned_abs();
        enc.encode_bypass(if level < 0 { 1 } else { 0 }, 1);
        let max_base_range = if high_freq { 6 } else { 5 };
        if mag >= max_base_range {
            running_avg = encode_high_range(enc, mag - max_base_range, running_avg);
        }
    }
    coeffs
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

/// Assemble one 64x64 lossless luma superblock: partition no-split + intra mode (lossless
/// forces TX_4X4, so no tx-partition bits), then the 256 4x4 TUs in raster order.
pub(crate) fn encode_lossless_luma_sb(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>],
    skip_cdfs: &[u32],
    dc_sign_ctxs: &[usize],
    mode_idx: usize,
    has_chroma: bool,
    partition_cdf: Option<u32>,
) {
    encode_intra_modes(enc, mode_idx, has_chroma, true, partition_cdf);
    for (i, tu) in tus.iter().enumerate() {
        encode_luma_tu4(enc, tu, skip_cdfs[i], dc_sign_ctxs[i]);
    }
}
