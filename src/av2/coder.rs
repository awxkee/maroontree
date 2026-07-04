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

/// Luma low/high-frequency token context from already-coded neighbor levels.
/// Returns `(base_context, hi_range_context)`.
const PLVL_STRIDE: i32 = 36;
const PLVL_BUF: usize = (PLVL_STRIDE as usize) * 40;
#[inline]
fn plvl(rc: i32) -> i32 {
    (rc >> 5) * PLVL_STRIDE + (rc & 31)
}
fn plvl_w(rc: i32, bwl: i32) -> i32 {
    (rc >> bwl) * PLVL_STRIDE + (rc & ((1 << bwl) - 1))
}

fn luma_coeff_context(levels: &[i32], rc: i32, xy: i32) -> (usize, usize) {
    luma_coeff_context_w(levels, rc, xy, 5)
}
fn luma_coeff_context_w(levels: &[i32], rc: i32, xy: i32, bwl: i32) -> (usize, usize) {
    let low_freq = xy < 4;
    let mut limit: i32 = if low_freq { 5 } else { 3 };
    let p = plvl_w(rc, bwl);
    let neighbor = |dy: i32, dx: i32| -> i32 { levels[(p + dy * PLVL_STRIDE + dx) as usize] };
    let mut low_mag = 0i32;
    let mut hi_mag = 0i32;
    for (dy, dx) in [(0, 1), (1, 0), (1, 1)] {
        let v = neighbor(dy, dx);
        low_mag += v.min(limit);
        hi_mag += v.min(5);
    }
    low_mag += neighbor(0, 2).min(limit) + neighbor(2, 0).min(limit);

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
    chroma_coeff_context_w(levels, rc, xy, plane_offset, 5)
}
fn chroma_coeff_context_w(
    levels: &[i32],
    rc: i32,
    xy: i32,
    plane_offset: usize,
    bwl: i32,
) -> (usize, usize) {
    let add_limit: i32 = if xy < 1 { 5 } else { 3 };
    let p = plvl_w(rc, bwl);
    let neighbor = |dy: i32, dx: i32| -> i32 { levels[(p + dy * PLVL_STRIDE + dx) as usize] };
    let (right, below, below_right) = (neighbor(0, 1), neighbor(1, 0), neighbor(1, 1));
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
    if high_freq {
        if over <= 2 {
            enc.sym_br_hf(hi_range_ctx, over as usize, 3);
        } else {
            enc.sym_br_hf(hi_range_ctx, 3, 3);
        }
    } else {
        if over <= 2 {
            enc.sym_br(hi_range_ctx, over as usize, 3);
        } else {
            enc.sym_br(hi_range_ctx, 3, 3);
        }
    }
}

/// Encode the end-of-block position using the given bin/hi-bit CDFs.
/// Identifies which EOB CDF table to use for adaptive dispatch in encode_eob.
#[derive(Copy, Clone)]
pub(crate) enum EobCdf {
    EobBin,
    Eob64Luma,
    Eob128Luma,
    Eob256,
    Eob512,
    ChrEobBin,
    ChrEob256,
    ChrEob512,
    #[allow(unused)]
    Eob16Q0(usize), // ctx
    ChrEob128,
    ChrEob32,
    ChrEob64,
}

/// Dispatch an EOB symbol to the right adaptive CDF method.
#[inline]
fn eob_sym(enc: &mut RangeEncoder, tbl: EobCdf, s: usize, nsyms: usize) {
    match tbl {
        EobCdf::EobBin => enc.sym_eob_bin(s, nsyms),
        EobCdf::Eob64Luma => enc.sym_eob64_luma(s, nsyms),
        EobCdf::Eob128Luma => enc.sym_eob128_luma(s, nsyms),
        EobCdf::Eob256 => enc.sym_eob256(s, nsyms),
        EobCdf::Eob512 => enc.sym_eob512(s, nsyms),
        EobCdf::ChrEobBin => enc.sym_chr_eob_bin(s, nsyms),
        EobCdf::ChrEob256 => enc.sym_chr_eob256(s, nsyms),
        EobCdf::ChrEob512 => enc.sym_chr_eob512(s, nsyms),
        EobCdf::Eob16Q0(ctx) => enc.sym_eob16_q0(ctx, s, nsyms),
        EobCdf::ChrEob128 => enc.sym_chr_eob128(s, nsyms),
        EobCdf::ChrEob32 => enc.sym_chr_eob32(s, nsyms),
        EobCdf::ChrEob64 => enc.sym_chr_eob64(s, nsyms),
    }
}
#[inline]
fn eob_sym_esc(enc: &mut RangeEncoder, tbl: EobCdf, s: usize, nsyms: usize) {
    // The escape extension adds a trailing zero; update_cdf is called on
    // the base cdf (not the escape sentinel), matching AVM behaviour.
    eob_sym(enc, tbl, s, nsyms);
}

fn encode_eob(
    enc: &mut RangeEncoder,
    eob: usize,
    eob_cdf: EobCdf,
    eob_hi_bit: u16,
    esc_bits: u32,
    pt_nsyms: usize,
) {
    if eob <= 1 {
        eob_sym(enc, eob_cdf, eob, pt_nsyms);
        return;
    }
    let mut bin = 2usize;
    while (2usize << (bin - 1)) <= eob {
        bin += 1;
    }
    if bin < pt_nsyms {
        eob_sym(enc, eob_cdf, bin, pt_nsyms);
    } else if esc_bits == 0 {
        // No-escape eob classes: the top eob_pt symbol is coded directly (decode_eob
        // cases 2/3) with no escape literal. The esc helper extends the stored cdf so
        // the top symbol has a valid upper boundary. `pt_nsyms` is avm's eob symbol
        // count minus one (off-by-one MIN_PROB convention) and is tx-size dependent:
        // TX_8X8 (eob_multi_size 2, avm nsym 7) -> 6; TX_16X16+ (avm nsym 8) -> 7.
        eob_sym_esc(enc, eob_cdf, bin, pt_nsyms);
    } else {
        eob_sym_esc(enc, eob_cdf, pt_nsyms, pt_nsyms);
        enc.encode_bypass((bin - pt_nsyms) as u32, esc_bits);
    }
    let extra_bits = bin - 2;
    let hi = (eob >> extra_bits) & 1;
    enc.bool_eob_extra(eob_hi_bit as u32, hi as u32);
    if extra_bits > 0 {
        let low = eob & ((1 << extra_bits) - 1);
        for k in (0..extra_bits).rev() {
            enc.encode_bypass(((low >> k) & 1) as u32, 1);
        }
    }
}

// ----- trellis RDOQ (encoder-only; bit-exact by construction) ------------------
//
// The decoder reconstructs from whatever levels we ship, so we're free to pick
// levels by rate-distortion instead of round-to-nearest. `luma_level_bits`
// mirrors the real token coder (`encode_luma32_token` + base-range), so the rate
// term is the actual coded cost, contextualised by `luma_coeff_context`. We then
// (A) RD-optimise each coefficient's magnitude and (B) RD-trim the EOB.

/// Precomputed `log2(32768/d)` for d in 1..=32768. The coefficient-cost CDFs are fixed
/// (allow_update=0), so every `tok_cost` is a constant of the interval width `d = hi-lo`;
/// caching `log2` turns the innermost RDOQ rate term from a transcendental into a lookup
/// (values are bit-identical to the direct computation, so encoder output is unchanged).
static TOK_LOG2_LUT: std::sync::OnceLock<Vec<f64>> = std::sync::OnceLock::new();
#[inline]
fn tok_log2_lut() -> &'static [f64] {
    TOK_LOG2_LUT.get_or_init(|| {
        let mut v = vec![0.0f64; 32769];
        for (d, slot) in v.iter_mut().enumerate().skip(1) {
            *slot = (32768.0 / d as f64).log2();
        }
        v[0] = v[1];
        v
    })
}

fn tok_cost(icdf: &[u16], s: usize) -> f64 {
    let hi = if s == 0 { 32768i32 } else { icdf[s - 1] as i32 };
    let lo = if s < icdf.len() { icdf[s] as i32 } else { 0 };
    tok_log2_lut()[(hi - lo).max(1) as usize]
}

#[inline]
fn rice_tail_bits(hr: u32) -> f64 {
    2.0 * ((hr + 1) as f64).log2() + 2.0
}

fn base_range_bits(level: u32, hi_range_ctx: usize, high_freq: bool, qc: usize) -> f64 {
    let limit = if high_freq { 3u32 } else { 5u32 };
    let over = level - limit;
    let cdf: &[u16] = if high_freq {
        &BR_TOK_HF_QC[qc][hi_range_ctx]
    } else {
        &BR_TOK_QC[qc][hi_range_ctx]
    };
    if over <= 2 {
        tok_cost(cdf, over as usize)
    } else {
        tok_cost(cdf, 3) + rice_tail_bits(level - (limit + 3))
    }
}

/// Estimated bits to code a luma coefficient of magnitude `level` at the given
/// context, matching `encode_luma32_token`. ~1-bit sign for nonzero levels.
fn luma_level_bits(
    level: u32,
    is_eob: bool,
    base_ctx: usize,
    hi_range_ctx: usize,
    high_freq: bool,
    qc: usize,
) -> f64 {
    let mut bits = if !high_freq {
        if is_eob {
            if level <= 4 {
                tok_cost(&LUMA32_EOB_TOK_LF_QC[qc][base_ctx], (level - 1) as usize)
            } else {
                tok_cost(&LUMA32_EOB_TOK_LF_QC[qc][base_ctx], 4)
                    + base_range_bits(level, hi_range_ctx, false, qc)
            }
        } else if level <= 4 {
            tok_cost(&LUMA32_BASE_TOK_LF_QC[qc][base_ctx], level as usize)
        } else {
            tok_cost(&LUMA32_BASE_TOK_LF_QC[qc][base_ctx], 5)
                + base_range_bits(level, hi_range_ctx, false, qc)
        }
    } else if is_eob {
        if level <= 2 {
            tok_cost(&LUMA32_EOB_TOK_HF_QC[qc][base_ctx], (level - 1) as usize)
        } else {
            tok_cost(&LUMA32_EOB_TOK_HF_QC[qc][base_ctx], 2)
                + base_range_bits(level, hi_range_ctx, true, qc)
        }
    } else if level <= 2 {
        tok_cost(&LUMA32_BASE_TOK_HF_QC[qc][base_ctx], level as usize)
    } else {
        tok_cost(&LUMA32_BASE_TOK_HF_QC[qc][base_ctx], 3)
            + base_range_bits(level, hi_range_ctx, true, qc)
    };
    if level > 0 {
        bits += 1.0;
    }
    bits
}

/// RD-optimise the quantised luma coefficients of one TX_32X32 in place.
/// `prm[k]`=|unquantised projection| at scan pos k, `lev[k]`=round-to-nearest
/// level. `lambda` is the RD multiplier (level^2 per bit). Returns estimated coded
/// bits. `lambda <= 0` => no-op (returns round-to-nearest rate).
pub(crate) fn rdoq_luma(
    prm: &[f32],
    lev: &mut [f32],
    qc: usize,
    scan: &[u16],
    area: usize,
    lambda: f64,
) -> f64 {
    let n = lev.len();
    let mut eob = 0usize;
    for (k, &lev) in lev[..n].iter().enumerate() {
        if lev != 0.0 {
            eob = k;
        }
    }
    if lev[eob] == 0.0 {
        return 0.0;
    }
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = vec![0i32; PLVL_BUF];

    let ctx_at = |levels: &[i32], k: usize, is_eob: bool| -> (usize, usize) {
        if is_eob {
            let high_freq = k >= LUMA_HI_TO_LOW;
            (
                1 + (k > th1) as usize + (k > th2) as usize,
                if high_freq { 0 } else { 7 },
            )
        } else {
            let rc = scan[k] as i32;
            luma_coeff_context(levels, rc, (rc >> 5) + (rc & 31))
        }
    };
    let store = |levels: &mut [i32], k: usize, mag: i32| {
        let rc = scan[k] as i32;
        let high_freq = k >= LUMA_HI_TO_LOW;
        let limit = if high_freq { 3 } else { 5 };
        levels[plvl(rc) as usize] = if mag < limit {
            mag
        } else {
            limit + (mag - limit).min(3)
        };
    };

    // Phase A: per-coefficient magnitude RD (EOB fixed).
    let mut total_bits = 0.0f64;
    for k in (0..=eob).rev() {
        let is_eob = k == eob;
        let high_freq = k >= LUMA_HI_TO_LOW;
        let a = prm[k] as f64;
        let q = lev[k].abs() as u32;
        let (bc, hc) = ctx_at(&levels, k, is_eob);
        let lo = if is_eob { 1u32 } else { 0u32 };
        let hi = q.max(lo);
        let mut best_l = hi;
        let mut best_cost = f64::INFINITY;
        for l in lo..=hi {
            let d = (a - l as f64) * (a - l as f64);
            let r = luma_level_bits(l, is_eob, bc, hc, high_freq, qc);
            let cost = d + lambda * r;
            if cost < best_cost {
                best_cost = cost;
                best_l = l;
            }
        }
        lev[k] = best_l as f32 * lev[k].signum();
        store(&mut levels, k, best_l as i32);
        total_bits += luma_level_bits(best_l, is_eob, bc, hc, high_freq, qc);
    }

    // Phase B: EOB RD-trim.
    loop {
        let mut last = None;
        for (k, &lev) in lev[..=eob].iter().enumerate() {
            if lev != 0.0 {
                last = Some(k);
            }
        }
        let Some(p) = last else { break };
        if p == 0 {
            break;
        }
        let high_freq = p >= LUMA_HI_TO_LOW;
        let (bc, hc) = ctx_at(&levels, p, true);
        let a = prm[p] as f64;
        let drop_bits = luma_level_bits(lev[p].abs() as u32, true, bc, hc, high_freq, qc);
        if lambda * drop_bits > a * a {
            lev[p] = 0.0;
            let rc = scan[p] as i32;
            levels[plvl(rc) as usize] = 0;
            total_bits -= drop_bits;
        } else {
            break;
        }
    }
    total_bits
}

/// Estimated bits to code a chroma coefficient of magnitude `level` at the given
/// context, matching `encode_chroma_tokens_scan`. The chroma frequency split is
/// simply DC (`is_dc`, scan position 0) vs. everything-else (high-frequency); the
/// base-range tail always uses the chroma HF BR table. ~1-bit sign for nonzero
/// levels.
fn chroma_level_bits(
    level: u32,
    is_eob: bool,
    is_dc: bool,
    base_ctx: usize,
    hi_range_ctx: usize,
    qc: usize,
) -> f64 {
    let mut bits = if is_dc {
        if is_eob {
            // sym_chr_eob_lf (4 syms): mag-1 for mag<=4, else saturate at 4 + tail.
            if level <= 4 {
                tok_cost(&CHROMA_EOB_TOK_LF_QC[qc][0], (level - 1) as usize)
            } else {
                tok_cost(&CHROMA_EOB_TOK_LF_QC[qc][0], 4)
                    + chroma_base_range_bits(level, hi_range_ctx, true, qc)
            }
        } else if level <= 4 {
            tok_cost(&CHROMA_BASE_TOK_LF_QC[qc][base_ctx], level as usize)
        } else {
            tok_cost(&CHROMA_BASE_TOK_LF_QC[qc][base_ctx], 5)
                + chroma_base_range_bits(level, hi_range_ctx, true, qc)
        }
    } else if is_eob {
        // sym_chr_eob_hf (2 syms): mag-1 for mag<=2, else saturate at 2 + tail.
        // The HF EOB context (1 + (eob>th1) + (eob>th2)) is folded into base_ctx
        // by the caller.
        if level <= 2 {
            tok_cost(&CHROMA_EOB_TOK_HF_QC[qc][base_ctx], (level - 1) as usize)
        } else {
            tok_cost(&CHROMA_EOB_TOK_HF_QC[qc][base_ctx], 2)
                + chroma_base_range_bits(level, hi_range_ctx, false, qc)
        }
    } else if level <= 2 {
        tok_cost(&CHROMA_BASE_TOK_HF_QC[qc][base_ctx], level as usize)
    } else {
        tok_cost(&CHROMA_BASE_TOK_HF_QC[qc][base_ctx], 3)
            + chroma_base_range_bits(level, hi_range_ctx, false, qc)
    };
    if level > 0 {
        bits += 1.0;
    }
    bits
}

/// Chroma high-range residual bits: the chroma BR symbol (limit 3, HF BR table)
/// plus the adaptive-Rice/Golomb tail for magnitudes beyond `max_base_range`
/// (5 for DC, 6 for HF — the threshold the sign pass uses for `encode_high_range`).
fn chroma_base_range_bits(level: u32, hi_range_ctx: usize, is_dc: bool, qc: usize) -> f64 {
    let over = level - 3; // BR symbol residual (max_base_range for the BR symbol is 3)
    let bits = if over <= 2 {
        tok_cost(&CHROMA_BR_TOK_HF_QC[qc][hi_range_ctx], over as usize)
    } else {
        tok_cost(&CHROMA_BR_TOK_HF_QC[qc][hi_range_ctx], 3)
    };
    // Adaptive-Rice tail beyond the sign-pass threshold.
    let max_base_range = if is_dc { 5u32 } else { 6u32 };
    if level >= max_base_range {
        bits + rice_tail_bits(level - max_base_range)
    } else {
        bits
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn rdoq_chroma(
    prm: &[f32],
    lev: &mut [f32],
    qc: usize,
    scan: &[u16],
    area: usize,
    plane_offset: usize,
    lambda: f64,
) -> f64 {
    let n = lev.len();
    let mut eob = 0usize;
    for (k, &l) in lev[..n].iter().enumerate() {
        if l != 0.0 {
            eob = k;
        }
    }
    if lev[eob] == 0.0 {
        return 0.0;
    }
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = vec![0i32; PLVL_BUF];

    // Context for a coefficient at scan position `k`. For EOB the chroma model
    // uses a fixed DC context (0) or the HF eob-position bucket; for non-EOB it
    // uses the neighbor-level chroma context (with the U/V plane offset folded in
    // by `chroma_coeff_context`).
    let ctx_at = |levels: &[i32], k: usize, is_eob: bool, is_dc: bool| -> (usize, usize) {
        if is_eob {
            if is_dc {
                (0, 0)
            } else {
                (1 + (k > th1) as usize + (k > th2) as usize, 0)
            }
        } else {
            let rc = scan[k] as i32;
            chroma_coeff_context(levels, rc, (rc >> 5) + (rc & 31), plane_offset)
        }
    };
    let store = |levels: &mut [i32], k: usize, mag: i32| {
        let rc = scan[k] as i32;
        levels[plvl(rc) as usize] = mag.min(5);
    };

    // Phase A: per-coefficient magnitude RD (EOB fixed).
    let mut total_bits = 0.0f64;
    for k in (0..=eob).rev() {
        let is_eob = k == eob;
        let is_dc = k == 0;
        let a = prm[k] as f64;
        let q = lev[k].abs() as u32;
        let (bc, hc) = ctx_at(&levels, k, is_eob, is_dc);
        let lo = if is_eob { 1u32 } else { 0u32 };
        let hi = q.max(lo);
        let mut best_l = hi;
        let mut best_cost = f64::INFINITY;
        for l in lo..=hi {
            let d = (a - l as f64) * (a - l as f64);
            let r = chroma_level_bits(l, is_eob, is_dc, bc, hc, qc);
            let cost = d + lambda * r;
            if cost < best_cost {
                best_cost = cost;
                best_l = l;
            }
        }
        lev[k] = best_l as f32 * lev[k].signum();
        store(&mut levels, k, best_l as i32);
        total_bits += chroma_level_bits(best_l, is_eob, is_dc, bc, hc, qc);
    }

    // Phase B: EOB RD-trim — drop trailing nonzero coefficients while the bit
    // saving outweighs the distortion of zeroing them.
    loop {
        let mut last = None;
        for (k, &l) in lev[..=eob].iter().enumerate() {
            if l != 0.0 {
                last = Some(k);
            }
        }
        let Some(p) = last else { break };
        if p == 0 {
            break;
        }
        let is_dc = p == 0;
        let (bc, hc) = ctx_at(&levels, p, true, is_dc);
        let a = prm[p] as f64;
        let drop_bits = chroma_level_bits(lev[p].abs() as u32, true, is_dc, bc, hc, qc);
        if lambda * drop_bits > a * a {
            lev[p] = 0.0;
            let rc = scan[p] as i32;
            levels[plvl(rc) as usize] = 0;
            total_bits -= drop_bits;
        } else {
            break;
        }
    }
    total_bits
}

fn level_at(coeffs: &[Coeff], scan_pos: usize) -> i32 {
    coeffs
        .iter()
        .find(|&&(s, _)| s == scan_pos)
        .map(|&(_, l)| l)
        .unwrap_or(0)
}

#[rustfmt::skip]
static REORDERED_DIR_Y_MODE: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];
#[rustfmt::skip]
static DEFAULT_MODE_LIST_Y: [u8; 56] = [
    17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54,
    16, 18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6, 7, 13, 21,
    27, 28, 34, 35, 41, 49, 55,
];

/// Emit one superblock's delta-Q (called right after the SB's partition bit,
/// matching the decoder's read order: partition -> delta_q -> mode). `signaled`
/// is the qindex delta already divided by 2^res_log2; |signaled| must be <= 6.
/// Abs magnitude is coded with the adaptive (static) delta_q CDF; a non-zero
/// magnitude is followed by a sign bypass bit.
pub(crate) fn emit_delta_q(enc: &mut RangeEncoder, signaled: i32) {
    let a = signaled.unsigned_abs() as usize;
    debug_assert!(a <= 6, "delta_q magnitude {a} would hit the escape symbol");
    enc.sym_delta_q(a, 7);
    if a != 0 {
        enc.encode_bypass((signaled < 0) as u32, 1);
    }
}

/// Consume a pending per-SB delta-Q: if the caller armed `delta_q_pending` and
/// the frame enables delta-Q, emit the SB's symbol and disarm. Call immediately
/// after the leaf's partition bit, before the luma mode.
/// Consume a pending per-SB CCSO flag. Mirrors AVM `write_ccso` for the U plane:
/// emitted after the partition bit and before delta-Q, once per SB. Phase 1 always
/// filters (blk_idc = 1). The context is derived the way `av2_get_ccso_context`
/// does for the SB's top-left block: `above`/`above-right` neighbors are excluded
/// at the SB top boundary, so only the left/bottom-left neighbors (both in the SB
/// to the left, hence same-SB) contribute. With every SB filtered this collapses to
/// ctx = 0 in the first column (no left neighbor) and ctx = 2 elsewhere.
pub(crate) fn maybe_emit_ccso(enc: &mut RangeEncoder) {
    if enc.ccso_pending && (enc.ccso_u_enable || enc.ccso_v_enable) {
        let (r, c) = enc.ccso_sb_rc;
        let cols = enc.ccso_cols;
        // Per-SB decision. When a decision grid is present (Phase 3 RD pass), read
        // the chosen flag for this SB; otherwise filter every SB (Phase 2 all-on).
        // The bitstream context depends only on the LEFT SB's decision: at the SB
        // top boundary the above/above-right neighbors are excluded, leaving the
        // left/bottom-left neighbors (both in the left SB). So col 0 => ctx 0;
        // col > 0 => left_on ? 2 : 0. (U and V carry independent grids.)
        let grid_u = &enc.ccso_grid;
        let idx = r * cols + c;
        let u_on = if enc.ccso_u_enable {
            if grid_u.is_empty() {
                1
            } else {
                grid_u[idx] as usize
            }
        } else {
            0
        };
        let v_on = if enc.ccso_v_enable {
            if enc.ccso_grid_v.is_empty() {
                1
            } else {
                enc.ccso_grid_v[idx] as usize
            }
        } else {
            0
        };
        if enc.ccso_u_enable {
            let left = if c == 0 {
                0
            } else if grid_u.is_empty() {
                1
            } else {
                grid_u[idx - 1] as usize
            };
            let ctx = if c == 0 {
                0
            } else if left != 0 {
                2
            } else {
                0
            };
            enc.sym_ccso(1, ctx, u_on);
        }
        if enc.ccso_v_enable {
            let left = if c == 0 {
                0
            } else if enc.ccso_grid_v.is_empty() {
                1
            } else {
                enc.ccso_grid_v[idx - 1] as usize
            };
            let ctx = if c == 0 {
                0
            } else if left != 0 {
                2
            } else {
                0
            };
            enc.sym_ccso(2, ctx, v_on);
        }
        enc.ccso_pending = false;
    }
}

pub(crate) fn maybe_emit_delta_q(enc: &mut RangeEncoder) {
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
}

const NO_MIDX: u8 = 0xff;

fn internal_dir_to_ymode(m: usize) -> u8 {
    match m {
        5 => 1,
        6 => 2,
        7 => 3,
        8 => 4,
        9 => 5,
        10 => 6,
        11 => 7,
        _ => 8,
    }
}
fn nominal_midx(y_mode: u8) -> u8 {
    let p = REORDERED_DIR_Y_MODE
        .iter()
        .position(|&m| m == y_mode)
        .unwrap();
    (p * 7 + 3) as u8
}

/// Build the neighbor-adaptive directional mode list (decode.rs:4736-4790).
/// `lmidx`/`amidx` = left/above block midx (NO_MIDX if absent/non-directional).
/// Built in full (prefix-stable), so the position of a target midx is its dir_idx.
fn build_dir_list_y(bw4: usize, bh4: usize, lmidx: u8, amidx: u8) -> Vec<u8> {
    if bw4 * bh4 <= 2 {
        return DEFAULT_MODE_LIST_Y.to_vec();
    }
    let mut list = [0u8; 56];
    let mut mask = 0u64;
    let mut ptr = 0usize;
    if lmidx != NO_MIDX {
        list[ptr] = lmidx;
        mask |= 1 << lmidx;
        ptr += 1;
    }
    if amidx != NO_MIDX && (ptr == 0 || amidx != list[0]) {
        list[ptr] = amidx;
        mask |= 1 << amidx;
        ptr += 1;
    }
    let n_dirs = ptr;
    if n_dirs == 0 {
        return DEFAULT_MODE_LIST_Y.to_vec();
    }
    if bw4 * bh4 > 4 {
        for i in 1..5i32 {
            for n in 0..n_dirs {
                let c = list[n] as i32;
                for d in [-i, i] {
                    let dm = ((c + d + 56) % 56) as u8;
                    if mask & (1 << dm) == 0 {
                        list[ptr] = dm;
                        mask |= 1 << dm;
                        ptr += 1;
                    }
                }
            }
        }
    }
    for &fm in DEFAULT_MODE_LIST_Y.iter() {
        if mask & (1 << fm) == 0 {
            list[ptr] = fm;
            ptr += 1;
        }
    }
    list[..ptr].to_vec()
}

/// Emit the luma y_mode (and DC chroma) for a 64x64 PARTITION_NONE block,
/// supporting the directional modes. Returns the block's `midx` (NO_MIDX for
/// non-directional) so the caller can store it for neighbor context.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_intra_modes_dir(
    enc: &mut RangeEncoder,
    mode_idx: usize,
    angle_delta: i8,
    has_chroma: bool,
    partition_cdf: Option<u32>,
    bw4: usize,
    bh4: usize,
    lmidx: u8,
    amidx: u8,
) -> u8 {
    if let Some(cdf) = partition_cdf {
        enc.bool_do_split(cdf, 0);
    }
    // MHCCP: the decoder reads the cfl_mhccp_switch on every CfL block where its
    // own is_mhccp_allowed() is true, so the encoder must mirror that exactly
    // (emitting switch=0 when we are not doing MHCCP). The switch/mh_dir CDF
    // context uses the actual coded block size via bw4/bh4. The whole-64 CfL fast
    // path only produced a valid MHCCP predictor for genuine 64x64 luma blocks;
    // for any other allowed size we still emit the (switch=0) symbol.
    if enc.mhccp {
        let allowed = crate::av2::cfl::is_mhccp_allowed(bw4, bh4, enc.mhccp_ssx, enc.mhccp_ssy);
        enc.mhccp_allowed = allowed;
        if allowed {
            enc.mhccp_size_group = crate::av2::cfl::mhccp_size_group_wh4(bw4, bh4);
            if !(bw4 == 16 && bh4 == 16) {
                enc.mhccp_use = false; // predictor only valid for 64x64 fast path
            }
        } else {
            enc.mhccp_use = false;
        }
    }
    maybe_emit_ccso(enc);
    maybe_emit_delta_q(enc);
    #[allow(clippy::needless_late_init)]
    let midx;
    if mode_idx < 5 {
        // non-directional: set 0, idx0[ctx], symbol = mode_idx (0..4). The idx0
        // context counts directional neighbors and must match the decoder even
        // for non-directional blocks adjacent to directional ones.
        let y_ctx = (lmidx != NO_MIDX) as usize + (amidx != NO_MIDX) as usize;
        enc.sym_y_set(0);
        enc.sym_y_idx0(y_ctx, mode_idx, 7);
        midx = NO_MIDX;
    } else {
        let y_mode = internal_dir_to_ymode(mode_idx);
        // target midx encodes both mode and angle delta: nominal (pos*7+3) + delta.
        let target = (nominal_midx(y_mode) as i32 + angle_delta as i32) as u8;
        let list = build_dir_list_y(bw4, bh4, lmidx, amidx);
        let dir_idx = list
            .iter()
            .position(|&m| m == target)
            .expect("target midx in list");
        let y_mode_idx = dir_idx + 5;
        let y_set = (y_mode_idx + 3) / 16; // 0 for 5..=12, else 1/2/3
        let y_ctx = (lmidx != NO_MIDX) as usize + (amidx != NO_MIDX) as usize;
        // y_set: escape (last) symbol of the 4-symbol alphabet is value 3.
        if y_set == 3 {
            enc.sym_y_set(3);
        } else {
            enc.sym_y_set(y_set);
        }
        if y_set == 0 {
            if y_mode_idx < 7 {
                enc.sym_y_idx0(y_ctx, y_mode_idx, 7);
            } else {
                // idx0 == 7 is the escape (last) symbol of the 8-symbol alphabet.
                enc.sym_y_idx0(y_ctx, 7, 7);
                let i1 = y_mode_idx - 7;
                if i1 == 5 {
                    enc.sym_y_idx1(y_ctx, 5, 5);
                } else {
                    enc.sym_y_idx1(y_ctx, i1, 5);
                }
            }
        } else {
            let bits = (y_mode_idx - (y_set * 16 - 3)) as u32;
            enc.encode_bypass(bits, 4);
        }
        midx = target;
    }
    if has_chroma {
        // Same read condition as encode_intra_modes: is_cfl is read whenever
        // cfl OR (mhccp && block-allowed); switch only coded when cfl is on.
        if enc.cfl || (enc.mhccp && enc.mhccp_allowed) {
            let isc = crate::av2::cfl::CFL_IS_CDF[enc.cfl_ctx];
            if enc.cfl_use {
                enc.bool_cfl_is(enc.cfl_ctx, isc as u32, 1);
                if enc.mhccp && enc.mhccp_allowed {
                    if enc.cfl {
                        enc.bool_cfl_mhccp(
                            crate::av2::cfl::CFL_MHCCP_SWITCH_CDF as u32,
                            enc.mhccp_use as u32,
                        );
                    }
                    if !enc.mhccp_use {
                        enc.bool_cfl_index(crate::av2::cfl::CFL_INDEX_CDF as u32, 0);
                    }
                } else {
                    enc.bool_cfl_index(crate::av2::cfl::CFL_INDEX_CDF as u32, 0);
                }
                if enc.mhccp && enc.mhccp_allowed && enc.mhccp_use {
                    // MHCCP (CFL_MULTI_PARAM): emit filter direction, no alpha.
                    enc.sym_mh_dir(enc.mhccp_size_group as usize, enc.mhccp_dir as usize);
                    return midx;
                }
                enc.sym_cfl_sign(enc.cfl_js as usize);
                let su = crate::av2::cfl::cfl_sign_u(enc.cfl_js);
                let sv = crate::av2::cfl::cfl_sign_v(enc.cfl_js);
                if su != 0 {
                    enc.sym_cfl_alpha(enc.cfl_ctx_u, enc.cfl_mag_u as usize);
                }
                if sv != 0 {
                    enc.sym_cfl_alpha(enc.cfl_ctx_v, enc.cfl_mag_v as usize);
                }
                return midx;
            }
            enc.bool_cfl_is(enc.cfl_ctx, isc as u32, 0);
        }
        // DC chroma. uv_mode_ctx = (luma is directional); with ctx=1 the DC index
        // is shifted by one ("slot 0" encodes same-as-luma).
        let uv_ctx = (midx != NO_MIDX) as usize;
        // Reordered chroma list: ctx==1 (directional luma) is
        // [luma_mode, DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, ...] so non-CfL chroma
        // modes sit at index (uv_ctx + internal_mode): DC=uv_ctx, SMOOTH=uv_ctx+1, ...
        let uv_idx = uv_ctx + enc.uv_mode;
        emit_uv_mode_idx(enc, uv_ctx, uv_idx);
    }
    midx
}

fn emit_uv_mode_idx(enc: &mut RangeEncoder, uv_ctx: usize, uv_idx: usize) {
    if uv_idx < 7 {
        enc.sym_uv_mode(uv_ctx, uv_idx, 7);
    } else {
        enc.sym_uv_mode(uv_ctx, 7, 7);
        enc.encode_bypass((uv_idx - 7) as u32, 3);
    }
}

fn encode_intra_modes(
    enc: &mut RangeEncoder,
    mode_idx: usize,
    has_chroma: bool,
    lossless: bool,
    partition_cdf: Option<u32>,
    _cfl_allowed: bool,
) {
    // do_split bool (=0, PARTITION_NONE) with the leaf's per-bsize/context cdf.
    // None for non-partition-point leaves (4x4 / narrow ext blocks), which read
    // no partition bit at all.
    // The decoder reads the cfl_mhccp_switch on every CfL block where its own
    // is_mhccp_allowed() is true, regardless of which encoder leaf emitted the
    // block, so the switch MUST be emitted (=0 if not doing MHCCP) whenever the
    // block size is MHCCP-allowed. Block dims come from enc.cur_bw4/cur_bh4.
    // If the caller already selected MHCCP for this block (enc.mhccp_use == true,
    // set just before this call), preserve its dir/size_group; otherwise this is
    // a non-MHCCP leaf and we emit switch=0.
    if enc.mhccp {
        enc.mhccp_allowed = crate::av2::cfl::is_mhccp_allowed(
            enc.cur_bw4,
            enc.cur_bh4,
            enc.mhccp_ssx,
            enc.mhccp_ssy,
        );
        if enc.mhccp_allowed && !enc.mhccp_use {
            enc.mhccp_size_group = crate::av2::cfl::mhccp_size_group_wh4(enc.cur_bw4, enc.cur_bh4);
        }
        if !enc.mhccp_allowed {
            enc.mhccp_use = false;
        }
    } else {
        enc.mhccp_allowed = false;
        enc.mhccp_use = false;
    }
    if let Some(cdf) = partition_cdf {
        enc.bool_do_split(cdf, 0);
    }
    maybe_emit_ccso(enc);
    maybe_emit_delta_q(enc);
    if lossless {
        // Lossless intra reads use_dpcm_y (dpcm_cdf, AVM_CDF2(16384)) before the luma
        // mode. 0 = no DPCM, then the normal intra-mode path follows.
        enc.encode_bool(16384, 0);
    }
    enc.sym_y_set(0); // intra_y mode set 0
    enc.sym_y_idx0(0, mode_idx, 7); // simple path: set 0, idx0 ctx 0
    if has_chroma {
        if lossless {
            // Lossless intra also reads use_dpcm_uv (dpcm_uv_cdf, AVM_CDF2(16384))
            // before the chroma mode. 0 = no DPCM.
            enc.encode_bool(16384, 0);
        }
        // CfL-allowed chroma blocks read a leading is_cfl bool (avm_read_symbol(cfl_cdf
        // [ctx], 2)) before the uv-mode symbol. encode_bool is the verified inverse of
        // avm_read_symbol(.,2). When CfL is chosen (enc.cfl_use): emit is_cfl=1, then
        // cfl_index=0 (CFL_EXPLICIT), then the joint sign + per-plane magnitudes, and
        // skip the uv-mode symbol entirely (decoder sets uv_mode = UV_CFL_PRED & returns).
        // Otherwise emit is_cfl=0 with the neighbor context and fall through to uv-mode.
        // Decoder reads is_cfl iff is_cfl_allowed(enable_cfl_intra) || is_mhccp_allowed;
        // with cfl off but mhccp on, the switch is inferred =1 (not coded).
        if enc.cfl || (enc.mhccp && enc.mhccp_allowed) {
            let isc = crate::av2::cfl::CFL_IS_CDF[enc.cfl_ctx];
            if enc.cfl_use {
                enc.bool_cfl_is(enc.cfl_ctx, isc as u32, 1);
                if enc.mhccp && enc.mhccp_allowed {
                    if enc.cfl {
                        enc.bool_cfl_mhccp(
                            crate::av2::cfl::CFL_MHCCP_SWITCH_CDF as u32,
                            enc.mhccp_use as u32,
                        );
                    }
                    if !enc.mhccp_use {
                        enc.bool_cfl_index(crate::av2::cfl::CFL_INDEX_CDF as u32, 0);
                    }
                } else {
                    enc.bool_cfl_index(crate::av2::cfl::CFL_INDEX_CDF as u32, 0);
                }
                if enc.mhccp && enc.mhccp_allowed && enc.mhccp_use {
                    enc.sym_mh_dir(enc.mhccp_size_group as usize, enc.mhccp_dir as usize);
                    return;
                }
                enc.sym_cfl_sign(enc.cfl_js as usize);
                let su = crate::av2::cfl::cfl_sign_u(enc.cfl_js);
                let sv = crate::av2::cfl::cfl_sign_v(enc.cfl_js);
                if su != 0 {
                    enc.sym_cfl_alpha(enc.cfl_ctx_u, enc.cfl_mag_u as usize);
                }
                if sv != 0 {
                    enc.sym_cfl_alpha(enc.cfl_ctx_v, enc.cfl_mag_v as usize);
                }
                return;
            }
            enc.bool_cfl_is(enc.cfl_ctx, isc as u32, 0);
        }
        // Chroma uv-mode. Co-located luma here is non-directional, so the
        // reordered chroma list is [DC, SMOOTH, SMOOTH_V, SMOOTH_H, PAETH, ...]
        // and ctx = 0. The internal mode numbering (0=DC,1=SMOOTH,2=SMOOTH_V,
        // 3=SMOOTH_H,4=PAETH) maps directly onto that list index.
        let uv_idx = enc.uv_mode;
        emit_uv_mode_idx(enc, 0, uv_idx);
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
            enc.bool_dc_sign(DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32, sign);
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
type ChromaStored = (i32, bool);

/// Reverse-scan token pass for chroma; fills the neighbor-level grid.
/// DC is low-frequency; all AC is high-frequency.
fn encode_chroma_tokens(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    plane_offset: usize,
) -> Vec<ChromaStored> {
    encode_chroma_tokens_scan(enc, coeffs, eob, plane_offset, &SCAN, 1024)
}

/// Scan/area-parameterised chroma coeff-token coder. Chroma LF region is the DC only
/// (LF_2D_LIM_UV = 1), size-independent; only the AC EOB-token thresholds (area/8,
/// area/4) vary with the coeff-region size.
fn encode_chroma_tokens_scan(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    plane_offset: usize,
    scan: &[u16],
    area: usize,
) -> Vec<ChromaStored> {
    encode_chroma_tokens_scan_w(enc, coeffs, eob, plane_offset, scan, area, 5)
}
fn encode_chroma_tokens_scan_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    plane_offset: usize,
    scan: &[u16],
    area: usize,
    bwl: i32,
) -> Vec<ChromaStored> {
    // height = area / width = area >> bwl
    let height = area >> bwl;
    let t1 = (height << bwl) / 8;
    let t2 = (height << bwl) / 4;
    let mut levels = vec![0u8; PLVL_BUF];
    let mut stored: Vec<ChromaStored> = vec![];
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = scan[scan_pos] as usize;
        let row = rc >> bwl;
        let col = rc & mask;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let lf = (row + col) < 1;
        let (base_ctx, hi_ctx) = if is_eob {
            // get_lower_levels_ctx_eob(bwl, height, scan_idx)
            let c = scan_pos;
            let e = if c == 0 {
                0
            } else if c <= t1 {
                1
            } else if c <= t2 {
                2
            } else {
                3
            };
            (e, 0)
        } else if lf {
            (ctx_lf_2d_chroma_w(&levels, rc, plane_offset, bwl), 0)
        } else {
            (
                ctx_2d_chroma_w(&levels, rc, plane_offset, bwl),
                br_ctx_2d_chroma_w(&levels, rc, bwl),
            )
        };
        if std::env::var("CBE").is_ok() && bwl == 4 && plane_offset == 0 && !is_eob && !lf {
            eprintln!("CBE pos={} ctx={} mag={}", rc, base_ctx, mag);
        }
        let sl = encode_chroma4_token(enc, mag, is_eob, base_ctx, hi_ctx, lf);
        levels[pidx_w(rc, bwl)] = sl as u8;
        stored.push((level, !lf));
    }
    stored
}

/// Rectangular chroma block coder for the 16-tap family (TX_16X64/TX_64X16 chroma,
/// 16×32 / 32×16 coeff region). `scan` + `area` parameterise the region; `eob_bin`
/// selects the 512-region chroma EOB cdf (CHROMA_EOB512_QC).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_chroma_block_rect(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    is_u_plane: bool,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
) {
    encode_chroma_block_rect_w(
        enc, coeffs, skip_cdf, is_u_plane, scan, eob_cdf, eob_hi, area, 5,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_chroma_block_rect_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    is_u_plane: bool,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
    bwl: i32,
) {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    let skip_tbl: u8 = if is_u_plane { 1 } else { 2 };
    if nonzero.is_empty() {
        enc.bool_skip_tbl(skip_cdf, 1, skip_tbl);
        return;
    }
    enc.bool_skip_tbl(skip_cdf, 0, skip_tbl);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(
        enc,
        eob,
        eob_cdf,
        eob_hi,
        if area <= 128 {
            0
        } else if area == 256 {
            1
        } else {
            2
        },
        if area == 32 {
            5
        } else if area == 64 {
            6
        } else {
            7
        },
    );
    let plane_offset = if is_u_plane { 0 } else { 4 };
    let stored = encode_chroma_tokens_scan_w(enc, &nonzero, eob, plane_offset, scan, area, bwl);
    encode_chroma_signs(enc, &stored);
}

/// Sign + golomb residual pass for chroma (all signs are bypass).
fn encode_chroma_signs(enc: &mut RangeEncoder, stored: &[ChromaStored]) {
    let mut running_avg = 0i32;
    for &(level, high_freq) in stored {
        if level == 0 {
            continue;
        }
        let mag = level.unsigned_abs();
        enc.encode_bypass(if level < 0 { 1 } else { 0 }, 1);
        let max_base_range = if high_freq { 6u32 } else { 5u32 };
        if std::env::var("SGE").is_ok() {
            eprintln!(
                "SGE mag={} hf={} maxbr={} hr={}",
                mag,
                high_freq,
                max_base_range,
                mag.saturating_sub(max_base_range)
            );
        }
        if mag >= max_base_range {
            running_avg = encode_high_range(enc, mag - max_base_range, running_avg);
        }
    }
}

/// Encode one chroma plane block. `skip_cdf` is the layout/neighbor-dependent
/// all-zero CDF and `is_u_plane` selects the U (offset 0) or V (offset 4) context.
pub(crate) fn encode_chroma_block(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    is_u_plane: bool,
) {
    encode_chroma_block_ex(enc, coeffs, skip_cdf, is_u_plane, true)
}

/// `skip_cdf` carries the skip context index. `u_tx64` selects the TX64 vs TX32 U slot.
pub(crate) fn encode_chroma_block_ex(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    is_u_plane: bool,
    u_tx64: bool,
) {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    let emit = |enc: &mut RangeEncoder, bit: u32| {
        let ctx = skip_cdf as usize;
        if !is_u_plane {
            enc.bool_v_skip(ctx, bit);
        } else if u_tx64 {
            enc.bool_u_skip64(ctx, bit);
        } else {
            enc.bool_u_skip32(ctx, bit);
        }
    };
    if nonzero.is_empty() {
        emit(enc, 1);
        return;
    }
    emit(enc, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();

    encode_eob(
        enc,
        eob,
        EobCdf::ChrEobBin,
        CHROMA_EOB_HI_BIT_QC[enc.qc],
        2,
        7,
    );
    let plane_offset = if is_u_plane { 0 } else { 4 };
    let stored = encode_chroma_tokens(enc, &nonzero, eob, plane_offset);
    encode_chroma_signs(enc, &stored);
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
                enc.sym_luma32_eob_lf(base_ctx, (level - 1) as usize, 4);
            } else {
                enc.sym_luma32_eob_lf(base_ctx, 4, 4);
                encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
            }
        } else if level <= 4 {
            enc.sym_luma32_lf(base_ctx, level as usize);
        } else {
            enc.sym_luma32_lf(base_ctx, 5);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if is_eob {
        if level <= 2 {
            enc.sym_luma32_eob_hf(base_ctx, (level - 1) as usize, 2);
        } else {
            enc.sym_luma32_eob_hf(base_ctx, 2, 2);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if level <= 2 {
        enc.sym_luma32_hf(base_ctx, level as usize);
    } else {
        enc.sym_luma32_hf(base_ctx, 3);
        encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
    }
    if (level as i32) < limit {
        level as i32
    } else {
        limit + (level as i32 - limit).min(3)
    }
}

fn encode_luma32_tokens(enc: &mut RangeEncoder, coeffs: &[Coeff], eob: usize) -> Vec<LumaStored> {
    encode_luma_tokens_scan(enc, coeffs, eob, &SCAN, 1024)
}

// ---- TX_16X16 (entropy class 2) full-AC luma token coding (clones of the 32x32
// versions using the LUMA16 base/eob token cdfs; base-range cdf is shared). ----
fn encode_luma16_token(
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
                enc.sym_luma16_eob_lf(base_ctx, (level - 1) as usize, 4);
            } else {
                enc.sym_luma16_eob_lf(base_ctx, 4, 4);
                encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
            }
        } else if level <= 4 {
            enc.sym_luma16_lf(base_ctx, level as usize);
        } else {
            enc.sym_luma16_lf(base_ctx, 5);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if is_eob {
        if level <= 2 {
            enc.sym_luma16_eob_hf(base_ctx, (level - 1) as usize, 2);
        } else {
            enc.sym_luma16_eob_hf(base_ctx, 2, 2);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if level <= 2 {
        enc.sym_luma16_hf(base_ctx, level as usize);
    } else {
        enc.sym_luma16_hf(base_ctx, 3);
        encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
    }
    if (level as i32) < limit {
        level as i32
    } else {
        limit + (level as i32 - limit).min(3)
    }
}
// Width-aware (bwl = log2 block width) variant for non-32-wide rect leaves.
fn encode_luma16_tokens_scan_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    scan: &[u16],
    area: usize,
    bwl: i32,
) -> Vec<LumaStored> {
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = vec![0i32; PLVL_BUF];
    let mut stored: Vec<LumaStored> = vec![];
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = scan[scan_pos] as i32;
        let x = rc >> bwl;
        let y = rc & mask;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let high_freq = if bwl >= 5 {
            scan_pos >= LUMA_HI_TO_LOW
        } else {
            (x + y) >= 4
        };
        let (base_ctx, hi_range_ctx) = if is_eob {
            if eob == 0 {
                (0usize, 0usize)
            } else {
                (
                    1 + (eob > th1) as usize + (eob > th2) as usize,
                    if high_freq || eob == 0 { 0 } else { 7 },
                )
            }
        } else {
            luma_coeff_context_w(&levels, rc, x + y, bwl)
        };
        let stored_level = encode_luma16_token(enc, mag, is_eob, base_ctx, hi_range_ctx, high_freq);
        levels[plvl_w(rc, bwl) as usize] = stored_level;
        stored.push((rc, x, y, level, high_freq));
    }
    stored
}

/// Full-AC intra luma leaf for a native TX_16X16 block (EXT_NEW_TX_SET).
/// Bit order mirrors the validated rect path: intra modes, tx do_partition=NONE,
/// skip, EOB (class 256), then — when eob>0 — the EXT_NEW_TX_SET tx_type symbol
/// (DCT_DCT = index 0, mode-independent since bit 0 is always the lowest set bit
/// of av2_md_trfm_used_flag), then the class-2 tokens and signs.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_leaf_16x16_full(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
    do_part_cdf: u32,
    tx_type_idx: usize,
) -> u32 {
    enc.cur_bw4 = 4;
    enc.cur_bh4 = 4;
    // intra_ext_tx_cdf[eset=1][TX_16X16] = AVM_CDF7(13759,26108,27688,29793,30265,
    // 31576); icdf = 32768 - cumulative. reduced_tx_set=0, FSC/IST off (headers).
    // `tx_type_idx` indexes the EXT_NEW_TX_SET (av2_md_idx2type[size_class=2][mode]):
    // 0 = DCT_DCT, 1 = ADST_ADST (mode-independent for the DC/SMOOTH/PAETH classes
    // this encoder emits). CDFs are non-adaptive (frame disable_cdf_update), so the
    // fixed icdf stays in sync regardless of which index is coded.
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(do_part_cdf, 0); // tx do_partition = NONE -> single TX_16X16
    let nonzero: Vec<Coeff> = tu.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, EobCdf::Eob256, EOB_HI_BIT_QC[enc.qc], 1, 7);
    if eob >= 1 {
        enc.sym_intra_ext_tx16(tx_type_idx, 6); // tx_type index
    }
    let stored = encode_luma16_tokens_scan_w(enc, &nonzero, eob, &SCAN16, 256, 4);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

/// `txtp_ext(min=1)` intra ext-tx cdf for TX_8X8 (decoder mode ctx, offset 2368+8).
/// The 16×16 leaf uses `txtp_ext(2)` (INTRA_EXT_TX16); the 8×8 corner uses min=1.
pub(crate) static TXTP_EXT8: [u16; 6] = [17858, 7511, 5804, 3445, 2531, 1233];

// ---- ctx-1 (TX_8X8) luma token coder + 8x8 corner leaf (both-axis residue-2) ----
fn encode_luma8_token(
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
                enc.sym_luma8_eob_lf(base_ctx, (level - 1) as usize, 4);
            } else {
                enc.sym_luma8_eob_lf(base_ctx, 4, 4);
                encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
            }
        } else if level <= 4 {
            enc.sym_luma8_lf(base_ctx, level as usize);
        } else {
            enc.sym_luma8_lf(base_ctx, 5);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if is_eob {
        if level <= 2 {
            enc.sym_luma8_eob_hf(base_ctx, (level - 1) as usize, 2);
        } else {
            enc.sym_luma8_eob_hf(base_ctx, 2, 2);
            encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
        }
    } else if level <= 2 {
        enc.sym_luma8_hf(base_ctx, level as usize);
    } else {
        enc.sym_luma8_hf(base_ctx, 3);
        encode_luma_base_range(enc, level, hi_range_ctx, high_freq);
    }
    if (level as i32) < limit {
        level as i32
    } else {
        limit + (level as i32 - limit).min(3)
    }
}

fn encode_luma8_tokens_scan_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    scan: &[u16],
    area: usize,
    bwl: i32,
) -> Vec<LumaStored> {
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = vec![0i32; PLVL_BUF];
    let mut stored: Vec<LumaStored> = vec![];
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = scan[scan_pos] as i32;
        let x = rc >> bwl;
        let y = rc & mask;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let high_freq = if bwl >= 5 {
            scan_pos >= LUMA_HI_TO_LOW
        } else {
            (x + y) >= 4
        };
        let (base_ctx, hi_range_ctx) = if is_eob {
            if eob == 0 {
                (0usize, 0usize)
            } else {
                (
                    1 + (eob > th1) as usize + (eob > th2) as usize,
                    if high_freq || eob == 0 { 0 } else { 7 },
                )
            }
        } else {
            luma_coeff_context_w(&levels, rc, x + y, bwl)
        };
        let stored_level = encode_luma8_token(enc, mag, is_eob, base_ctx, hi_range_ctx, high_freq);
        levels[plvl_w(rc, bwl) as usize] = stored_level;
        stored.push((rc, x, y, level, high_freq));
    }
    stored
}

/// Bottom-right 8×8 corner leaf (residue-2 in both dims), TX_8X8, tx-size class ctx-1.
/// `do_part_cdf` / `tx_type_cdf` are passed so the exact decoder cdfs can be confirmed
/// empirically (trace the decoder on a real 8×8 corner). `emit_tx_type` toggles the
/// intra_ext_tx symbol for the validation pass.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_leaf_8x8(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
    do_part_cdf: u32,
    tx_type_cdf: Option<(&'static [u16], usize, usize)>, // (cdf, idx, nsym)
) -> u32 {
    enc.cur_bw4 = 2;
    enc.cur_bh4 = 2;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(do_part_cdf, 0); // tx do_partition = NONE -> single TX_8X8
    let nonzero: Vec<Coeff> = tu.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, EobCdf::Eob64Luma, EOB_HI_BIT_QC[enc.qc], 0, 6);
    if eob >= 1
        && let Some((cdf, idx, nsym)) = tx_type_cdf
    {
        enc.encode_symbol(cdf, idx, nsym);
    }
    let stored = encode_luma8_tokens_scan_w(enc, &nonzero, eob, &SCAN8X8, 64, 3);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_leaf_rect128(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    bw4: usize,
    bh4: usize,
    part_cdf: u32,
    do_part_cdf: u32,
    scan: &'static [u16],
    tx_type_cdf: Option<(&'static [u16], usize, usize)>,
) -> u32 {
    enc.cur_bw4 = bw4;
    enc.cur_bh4 = bh4;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(do_part_cdf, 0); // tx do_partition = NONE -> single rect TX
    let nonzero: Vec<Coeff> = tu.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, EobCdf::Eob128Luma, EOB_HI_BIT_QC[enc.qc], 0, 7);
    if eob >= 1
        && let Some((cdf, idx, nsym)) = tx_type_cdf
    {
        enc.encode_symbol(cdf, idx, nsym);
    }
    // bwl = log2(tx width) = log2(bw4*4)
    let bwl = (bw4 * 4).trailing_zeros() as i32;
    let stored = encode_luma16_tokens_scan_w(enc, &nonzero, eob, scan, 128, bwl);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(127)
}

/// Generalised luma coeff-token coder. `scan` is the coefficient scan in slimav
/// column-major convention (rc = a*32 + c); `area` = coeff-region width*height,
/// which sets the EOB-token base-context thresholds (avm get_lower_levels_ctx_eob:
/// area/8, area/4). Everything else (PLVL_STRIDE, LF split at scan pos 10, neighbor
/// template, TX_32X32-class cdfs) is size-independent in this convention.
fn encode_luma_tokens_scan(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    scan: &[u16],
    area: usize,
) -> Vec<LumaStored> {
    encode_luma_tokens_scan_w(enc, coeffs, eob, scan, area, 5)
}
fn encode_luma_tokens_scan_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    eob: usize,
    scan: &[u16],
    area: usize,
    bwl: i32,
) -> Vec<LumaStored> {
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = vec![0i32; PLVL_BUF];
    let mut stored: Vec<LumaStored> = vec![];
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = level_at(coeffs, scan_pos);
        let rc = scan[scan_pos] as i32;
        let x = rc >> bwl;
        let y = rc & mask;
        let mag = level.unsigned_abs();
        let is_eob = scan_pos == eob;
        let high_freq = if bwl >= 5 {
            scan_pos >= LUMA_HI_TO_LOW
        } else {
            (x + y) >= 4
        };
        let (base_ctx, hi_range_ctx) = if is_eob {
            if eob == 0 {
                (0usize, 0usize)
            } else {
                (
                    1 + (eob > th1) as usize + (eob > th2) as usize,
                    if high_freq || eob == 0 { 0 } else { 7 },
                )
            }
        } else {
            luma_coeff_context_w(&levels, rc, x + y, bwl)
        };
        let stored_level = encode_luma32_token(enc, mag, is_eob, base_ctx, hi_range_ctx, high_freq);
        levels[plvl_w(rc, bwl) as usize] = stored_level;
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
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, EobCdf::EobBin, EOB_HI_BIT_QC[enc.qc], 2, 7);
    let stored = encode_luma32_tokens(enc, &nonzero, eob);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

/// Encode one rectangular luma transform unit (single TX) for the 16-tap family:
/// TX_16X64 (scan SCAN16X32, area 512), TX_64X16 (SCAN32X16, area 512), TX_16X16
/// (SCAN16, area 256). Coeff base/br cdfs are the shared TX_32X32 class; only the
/// scan, eob cdf, and EOB-token thresholds (area) differ. `eob_bin`/`eob_hi` select
/// the EOB position-token cdf (EOB512_QC for 16X64/64X16).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_tu_rect(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
) -> u32 {
    encode_luma_tu_rect_w(
        enc,
        coeffs,
        skip_cdf,
        dc_sign_ctx,
        scan,
        eob_cdf,
        eob_hi,
        area,
        5,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_tu_rect_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
    bwl: i32,
) -> u32 {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(
        enc,
        eob,
        eob_cdf,
        eob_hi,
        if area <= 128 {
            0
        } else if area == 256 {
            1
        } else {
            2
        },
        if area == 64 { 6 } else { 7 },
    );
    if eob >= 1 {
        enc.sym_tx_short_side(1, 0);
    }
    let stored = encode_luma_tokens_scan_w(enc, &nonzero, eob, scan, area, bwl);
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
    part_cdf: u32,
) -> [u32; 4] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 16;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_SPLIT_64 as u32, 1); // tx_split = 1
    enc.sym_tx_part_64(0, 6); // tx_part symbol 0 = SPLIT
    let mut cul = [0u32; 4];
    for i in 0..4 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    cul
}

/// Directional-aware 64x64 PARTITION_NONE emit. Identical to
/// `encode_luma_block_split` but routes the y_mode through the directional
/// coder (so internal modes 5..=12 emit conformantly) and returns the block's
/// `midx` for neighbor context. `lmidx`/`amidx` are the left/above SB midx
/// (NO_MIDX if absent or non-directional); bw4=bh4=16 for a 64x64 block.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_block_split_dir(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    angle_delta: i8,
    has_chroma: bool,
    part_cdf: u32,
    lmidx: u8,
    amidx: u8,
) -> ([u32; 4], u8) {
    let midx = encode_intra_modes_dir(
        enc,
        mode_idx,
        angle_delta,
        has_chroma,
        Some(part_cdf),
        16,
        16,
        lmidx,
        amidx,
    );
    enc.bool_txfm_part(TX_SPLIT_64 as u32, 1);
    enc.sym_tx_part_64(0, 6);
    let mut cul = [0u32; 4];
    for i in 0..4 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    (cul, midx)
}

/// Encode a 64x64 intra luma block as tx-partition VERT4 → four side-by-side
/// TX_16X64 strips (left→right). The block stays PARTITION_NONE (one prediction);
/// only the luma transform is partitioned. do_partition=1 then 4-way type symbol 4
/// (TX_PARTITION_VERT4 = partition value 5). Each strip is a TX_16X64 (scan
/// SCAN16X32, eob class 512) with its own short-side tx_type (DCT) — same coding as
/// the validated right-edge 16×64 leaf.
pub(crate) fn encode_luma_block_vert4(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 4] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 16;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_SPLIT_64 as u32, 1); // do_partition = 1
    enc.sym_tx_part_64(4, 6); // type symbol 4 = VERT4
    let mut cul = [0u32; 4];
    for i in 0..4 {
        cul[i] = encode_luma_tu_rect_w(
            enc,
            &tus[i],
            skip_cdfs[i],
            dc_sign_ctxs[i],
            &SCAN16X32,
            EobCdf::Eob512,
            EOB_HI_BIT_QC[enc.qc],
            512,
            4,
        );
    }
    cul
}

/// HORZ4 tx-partition: four stacked TX_64X16 strips (top->bottom). Type symbol 3
/// (TX_PARTITION_HORZ4 = partition value 4). Mirror of VERT4 with the wide TU
/// (scan SCAN32X16, eob class 512, short-side tx_type DCT).
pub(crate) fn encode_luma_block_horz4(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 4] {
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_SPLIT_64 as u32, 1); // do_partition = 1
    enc.sym_tx_part_64(3, 6); // type symbol 3 = HORZ4
    let mut cul = [0u32; 4];
    for i in 0..4 {
        cul[i] = encode_luma_tu_rect(
            enc,
            &tus[i],
            skip_cdfs[i],
            dc_sign_ctxs[i],
            &SCAN32X16,
            EobCdf::Eob512,
            EOB_HI_BIT_QC[enc.qc],
            512,
        );
    }
    cul
}

/// do_partition cdf for an intra 64X32 luma block (txfm_do_partition_cdf[0][0][6],
/// avm AVM_CDF2(15952) → 32768-15952). Both horz/vert splits are allowed for
/// BLOCK_64X32 + TX_64X32, so a 4-way type symbol follows.
const TX_DO_PART_64X32: u32 = 16816;

/// Encode an intra 64x32 luma leaf as TX_PARTITION_VERT → two TX_32X32 (left,
/// right). `part_cdf` is the leaf's do_split (PARTITION_NONE) cdf from the
/// partition walk. `skip_cdfs`/`dc_sign_ctxs` index the two sub-TUs in order.
pub(crate) fn encode_luma_leaf_64x32(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 2],
    skip_cdfs: &[u32; 2],
    dc_sign_ctxs: &[usize; 2],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 2] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 8;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_DO_PART_64X32, 1); // do_partition = 1
    enc.sym_tx_part_64x32(2, 6); // type = VERT-1 = 2
    let mut cul = [0u32; 2];
    for i in 0..2 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    cul
}

/// do_partition cdf for an intra 32X32 luma block (txfm_do_partition_cdf[0][0][5],
/// avm AVM_CDF2(15391) → 32768-15391). The 32X32 leaf codes do_partition=0 (NONE)
/// → a single TX_32X32; no 4-way symbol follows.
const TX_DO_PART_32X32: u32 = 17377;

/// Encode an intra 32x64 luma leaf (right-edge) as TX_PARTITION_HORZ → two stacked
/// TX_32X32 (top, bottom). `part_cdf` is the leaf's do_split cdf from the walk.
pub(crate) fn encode_luma_leaf_32x64(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 2],
    skip_cdfs: &[u32; 2],
    dc_sign_ctxs: &[usize; 2],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 2] {
    enc.cur_bw4 = 8;
    enc.cur_bh4 = 16;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_DO_PART_64X32, 1); // do_partition = 1 (group 6 cdf == 16816)
    enc.sym_tx_part_32x64(1, 6); // type = HORZ-1 = 1
    let mut cul = [0u32; 2];
    for i in 0..2 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    cul
}

/// Encode a bottom-right 16×16 intra luma corner leaf (residue 4 in both dims).
/// BLOCK_16X16 is tx-part group 3 (cdf 11074); NONE → single TX_16X16 (entropy class
/// 2, eob class 256). Luma is coded DC-only: keeping eob count == 1 makes the decoder
/// skip the (otherwise complex) EXT_NEW_TX_SET tx_type read entirely (dc_skip). A
/// `dc_level` of 0 emits a skip. `tx_type` is luma-only, so chroma 16×16 still codes
/// full AC separately. The DC is the LF eob coeff at raster pos 0, so its base-range
/// context is 0 (get_br_ctx_lf_eob).
/// Encode a DC-only intra luma leaf in entropy class 2: the 16×16 corner (TX_16X16,
/// do_part group 3 → cdf 11074) and the residue-2 edges 8×32 / 32×8 (TX_8X32 / TX_32X8,
/// do_part group 7 → cdf 18032). Keeping eob count == 1 makes the decoder skip the
/// tx_type read (dc_skip), avoiding the EXT_NEW_TX_SET / LONG_SIDE_32 sets these sizes
/// would otherwise use. All three are entropy class 2 (LUMA16 LF eob cdf) with eob
/// class 256. `dc_level` 0 → skip. `tx_type` is luma-only so chroma still codes full
/// AC. The DC is the LF eob coeff at raster pos 0, base-range context 0.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_leaf_dc_class2(
    enc: &mut RangeEncoder,
    dc_level: i32,
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
    do_part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 32;
    enc.cur_bh4 = 32;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(do_part_cdf, 0); // tx do_partition = NONE → single transform
    if dc_level == 0 {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    // eob count 1 (position 0): decoder's dc_skip path skips tx_type + sec_tx_type.
    encode_eob(enc, 0, EobCdf::Eob256, EOB_HI_BIT_QC[enc.qc], 1, 7);
    let mag = dc_level.unsigned_abs();
    if mag <= 4 {
        enc.sym_luma16_eob_lf(0, (mag - 1) as usize, 4);
    } else {
        enc.sym_luma16_eob_lf(0, 4, 4);
        encode_luma_base_range(enc, mag, 0, false);
    }
    enc.bool_dc_sign(
        DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32,
        (dc_level < 0) as u32,
    );
    if mag >= 8 {
        encode_high_range(enc, mag - 8, 0);
    }
    mag.min(63)
}
/// do_partition (both splits allowed) → emit NONE (group-8 cdf 18958 = 32768 -
/// AVM_CDF2(13810)) for a single TX_16X64. Coeff region 16×32, scan SCAN16X32, eob
/// class 512. block==tx dimensionally (16×64) so the caller passes a ctx-0 skip cdf.
pub(crate) fn encode_luma_leaf_16x64(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 4;
    enc.cur_bh4 = 16;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(18958, 0); // tx do_partition = NONE → single TX_16X64
    encode_luma_tu_rect_w(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN16X32,
        EobCdf::Eob512,
        EOB_HI_BIT_QC[enc.qc],
        512,
        4,
    )
}

/// Encode a bottom-edge 64×16 intra luma leaf (residue 4). BLOCK_64X16 is also
/// tx-part group 8 (cdf 18958); NONE → single TX_64X16, coeff region 32×16, scan
/// SCAN32X16, eob class 512.
pub(crate) fn encode_luma_leaf_64x16(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 4;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(18958, 0); // tx do_partition = NONE → single TX_64X16
    encode_luma_tu_rect(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN32X16,
        EobCdf::Eob512,
        EOB_HI_BIT_QC[enc.qc],
        512,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_tu_rect_long32(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
    short_cdf: &[u16; 3],
    ctx2: bool,
) -> u32 {
    encode_luma_tu_rect_long32_w(
        enc,
        coeffs,
        skip_cdf,
        dc_sign_ctx,
        scan,
        eob_cdf,
        eob_hi,
        area,
        short_cdf,
        ctx2,
        5,
    )
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_tu_rect_long32_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    scan: &[u16],
    eob_cdf: EobCdf,
    eob_hi: u16,
    area: usize,
    short_cdf: &[u16; 3],
    ctx2: bool,
    bwl: i32,
) -> u32 {
    let nonzero: Vec<Coeff> = coeffs.iter().cloned().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(
        enc,
        eob,
        eob_cdf,
        eob_hi,
        if area <= 128 {
            0
        } else if area == 256 {
            1
        } else {
            2
        },
        if area == 64 { 6 } else { 7 },
    );
    if eob >= 1 {
        // txtp_long32_dct(0) default = 32732; symbol 1 = DCT on the long side.
        enc.bool_txfm_part(32732, 1);
        // txtp_intra_short_1d(min) short-side index 0 → DCT short side. The working-copy
        // context follows the passed short_cdf: [6068,..] = index 1 (ctx 0, 8x32/32x8),
        // [5853,..] = index 2 (ctx 1, 16x32/32x16). avmdec adapts these separately.
        let ss_ctx = if short_cdf[0] == 6068 { 0 } else { 1 };
        enc.sym_tx_short_side(ss_ctx, 0);
    }
    // 8-family rect leaves (TX_8X32 / TX_32X8) are tx-size class ctx=2; their eob/base
    // coefficient-token cdfs are the ctx-2 (LUMA16) tables, NOT the ctx-3 (LUMA32) ones
    // used by the 16-family (TX_16X32 / TX_32X16). The decoder selects these via
    // `452 + t_dim.ctx*160` (HF) / `1440 + t_dim.ctx*528` (LF) and eob_base_y_tok_*.
    let stored = if ctx2 {
        encode_luma16_tokens_scan_w(enc, &nonzero, eob, scan, area, bwl)
    } else {
        encode_luma_tokens_scan_w(enc, &nonzero, eob, scan, area, bwl)
    };
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .filter(|&a| a > 0)
        .count() as u32
}

/// Right×bottom corner 16×32 intra luma leaf (residue-4 width × residue-{6,8} height).
/// BLOCK_16X32 is tx-part group 4 (tx_split cdf 19451); NONE → single TX_16X32, coeff
/// region 16×32, scan SCAN16X32, eob class 512.
pub(crate) fn encode_luma_leaf_16x32(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 4;
    enc.cur_bh4 = 8;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(19451, 0); // tx_split (szctx 4) = NONE → single TX_16X32
    encode_luma_tu_rect_long32_w(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN16X32,
        EobCdf::Eob512,
        EOB_HI_BIT_QC[enc.qc],
        512,
        &[5853, 357, 20], // txtp_intra_short_1d(min=2)
        false,
        4,
    )
}

/// Corner 32×16 intra luma leaf (residue-{6,8} width × residue-4 height). Same group-4
/// tx_split cdf 19451; NONE → single TX_32X16, coeff region 32×16, scan SCAN32X16.
pub(crate) fn encode_luma_leaf_32x16(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 8;
    enc.cur_bh4 = 4;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(19451, 0); // tx_split (szctx 4) = NONE → single TX_32X16
    encode_luma_tu_rect_long32(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN32X16,
        EobCdf::Eob512,
        EOB_HI_BIT_QC[enc.qc],
        512,
        &[5853, 357, 20], // txtp_intra_short_1d(min=2)
        false,
    )
}

/// Bottom edge 8×32 intra luma leaf (residue-2 width). BLOCK_8X32 is tx-part group 8
/// (tx_split cdf 18958); NONE → single TX_8X32 (max side 32, min side 8 → long-side-32
/// ext-tx with txtp_long32_dct=1 + short_1d(min=1) idx 0 → DCT_DCT). Coeff region 8×32,
/// scan SCAN8X32, eob class 256.
pub(crate) fn encode_luma_leaf_8x32(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 2;
    enc.cur_bh4 = 8;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(18958, 0); // tx_split (szctx 8) = NONE → single TX_8X32
    encode_luma_tu_rect_long32_w(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN8X32,
        EobCdf::Eob256,
        EOB_HI_BIT_QC[enc.qc],
        256,
        &[6068, 608, 20], // txtp_intra_short_1d(min=1)
        true,
        3, // bwl = log2(8)
    )
}

/// Right edge 32×8 intra luma leaf (residue-2 height). Group 8 (cdf 18958); NONE →
/// single TX_32X8, coeff region 32×8, scan SCAN32X8, eob class 256.
pub(crate) fn encode_luma_leaf_32x8(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 8;
    enc.cur_bh4 = 2;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(18958, 0); // tx_split (szctx 8) = NONE → single TX_32X8
    encode_luma_tu_rect_long32(
        enc,
        tu,
        skip_cdf,
        dc_sign_ctx,
        &SCAN32X8,
        EobCdf::Eob256,
        EOB_HI_BIT_QC[enc.qc],
        256,
        &[6068, 608, 20], // txtp_intra_short_1d(min=1)
        true,
    )
}

/// Encode an intra 32x32 luma leaf (corner) as a single TX_32X32 (do_partition=0).
pub(crate) fn encode_luma_leaf_32x32(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> u32 {
    enc.cur_bw4 = 8;
    enc.cur_bh4 = 8;
    encode_intra_modes(enc, mode_idx, has_chroma, false, Some(part_cdf), false);
    enc.bool_txfm_part(TX_DO_PART_32X32, 0); // do_partition = 0 → single TX_32X32
    encode_luma_tu32(enc, tu, skip_cdf, dc_sign_ctx)
}

// Padded levels grid: bwl=2, stride = (1<<2)+TX_PAD_HOR(4) = 8. get_padded_idx.
#[inline]
fn pidx(rc: usize) -> usize {
    rc + (rc >> 2) * 4
}
fn pidx_w(rc: usize, bwl: i32) -> usize {
    let row = rc >> bwl;
    let col = rc & ((1 << bwl) - 1);
    row * ((1 << bwl) + 4) + col
}
fn ctx_lf_2d_chroma_w(levels: &[u8], rc: usize, voff: usize, bwl: i32) -> usize {
    let b = pidx_w(rc, bwl);
    let s = (1 << bwl) + 4;
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + s].min(5) as i32 + levels[b + s + 1].min(5) as i32;
    ((mag + 1) >> 1).min(3) as usize + voff
}
fn ctx_2d_chroma_w(levels: &[u8], rc: usize, voff: usize, bwl: i32) -> usize {
    let b = pidx_w(rc, bwl);
    let s = (1 << bwl) + 4;
    let mag =
        levels[b + 1].min(3) as i32 + levels[b + s].min(3) as i32 + levels[b + s + 1].min(3) as i32;
    ((mag + 1) >> 1).min(3) as usize + voff
}
fn br_ctx_2d_chroma_w(levels: &[u8], rc: usize, bwl: i32) -> usize {
    let b = pidx_w(rc, bwl);
    let s = (1 << bwl) + 4;
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + s].min(5) as i32 + levels[b + s + 1].min(5) as i32;
    ((mag + 1) >> 1).min(3) as usize
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
    if lf {
        if over <= 2 {
            enc.sym_br_lf_q0(ctx, over as usize, 3);
        } else {
            enc.sym_br_lf_q0(ctx, 3, 3);
        }
    } else {
        if over <= 2 {
            enc.sym_br_q0(ctx, over as usize, 3);
        } else {
            enc.sym_br_q0(ctx, 3, 3);
        }
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
                enc.sym_base_lf_eob_tx4(base_ctx, (level - 1) as usize, 4);
            } else {
                enc.sym_base_lf_eob_tx4(base_ctx, 4, 4);
                encode_br4(enc, level, hi_ctx, true);
            }
        } else if level <= 4 {
            enc.sym_base_lf_tx4(base_ctx, 0, level as usize);
        } else {
            enc.sym_base_lf_tx4(base_ctx, 0, 5);
            encode_br4(enc, level, hi_ctx, true);
        }
    } else if is_eob {
        if level <= 2 {
            enc.sym_base_eob_tx4(base_ctx, (level - 1) as usize, 2);
        } else {
            enc.sym_base_eob_tx4(base_ctx, 2, 2);
            encode_br4(enc, level, hi_ctx, false);
        }
    } else if level <= 2 {
        enc.sym_base_tx4(base_ctx, 0, level as usize);
    } else {
        enc.sym_base_tx4(base_ctx, 0, 3);
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
    if pt - 1 <= 3 {
        enc.sym_eob16_q0(plctx, pt - 1, 4);
    } else {
        enc.sym_eob16_q0(plctx, 4, 4);
    }
    if obits > 0 {
        let extra = eob_count - start;
        let msb = (extra >> (obits - 1)) & 1;
        enc.bool_eob_extra(EOB_HI_BIT_QC[enc.qc] as u32, msb as u32);
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
            enc.bool_dc_sign(DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32, sign);
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
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
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
// Context fns use 3 neighbors; U uses ctx 0..3, V adds +4.

fn ctx_lf_2d_chroma(levels: &[u8], rc: usize, voff: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    ((mag + 1) >> 1).min(3) as usize + voff
}
fn ctx_2d_chroma(levels: &[u8], rc: usize, voff: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(3) as i32 + levels[b + 8].min(3) as i32 + levels[b + 9].min(3) as i32;
    ((mag + 1) >> 1).min(3) as usize + voff
}
fn br_ctx_2d_chroma(levels: &[u8], rc: usize) -> usize {
    let b = pidx(rc);
    let mag =
        levels[b + 1].min(5) as i32 + levels[b + 8].min(5) as i32 + levels[b + 9].min(5) as i32;
    ((mag + 1) >> 1).min(3) as usize
}
fn encode_br_uv(enc: &mut RangeEncoder, level: u32, ctx: usize) {
    let over = level - 3;
    if over <= 2 {
        enc.sym_br_uv(ctx, over as usize, 3);
    } else {
        enc.sym_br_uv(ctx, 3, 3);
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
                enc.sym_base_lf_eob_uv(base_ctx, (level - 1) as usize, 4);
            } else {
                enc.sym_base_lf_eob_uv(base_ctx, 4, 4);
            }
        } else if level <= 4 {
            enc.sym_base_lf_uv(base_ctx, level as usize);
        } else {
            enc.sym_base_lf_uv(base_ctx, 5);
        }
        if level <= 4 { level as i32 } else { 5 } // chroma lf: no br, capped at 5
    } else {
        if is_eob {
            if level <= 2 {
                enc.sym_base_eob_uv(base_ctx, (level - 1) as usize, 2);
            } else {
                enc.sym_base_eob_uv(base_ctx, 2, 2);
                encode_br_uv(enc, level, hi_ctx);
            }
        } else if level <= 2 {
            enc.sym_base_uv(base_ctx, level as usize);
        } else {
            enc.sym_base_uv(base_ctx, 3);
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
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
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

/// Lossy DCT_DCT 4×4 scan (decoder `SCANS[TX_4X4]`, up-right diagonal). The lossless
/// path uses the transposed scan; lossy DCT needs this one.
pub(crate) static SCAN4X4_LOSSY: [u16; 16] = [0, 1, 4, 2, 5, 8, 3, 6, 9, 12, 7, 10, 13, 11, 14, 15];

/// Same scan order as `SCAN4X4_LOSSY`, but in the encoder's packed rc = (col<<5)|row
/// convention used by `project_scan` / `reconstruct_chroma` (which index `coeff[row*n+col]`
/// via `(rc&31)*n + (rc>>5)`). The token coder uses the plain-raster `SCAN4X4_LOSSY`.
pub(crate) static SCAN4X4_LOSSY_PACKED: [u16; 16] =
    [0, 32, 1, 64, 33, 2, 96, 65, 34, 3, 97, 66, 35, 98, 67, 99];

/// Scan-parameterized 4×4 chroma TU coder (clone of `encode_chroma_tu4` taking the scan
/// explicitly, for the lossy DCT corner chroma).
pub(crate) fn encode_chroma_tu4_scan(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    plane_v: bool,
    scan: &[u16],
    skip_ctx: usize,
) -> u32 {
    if coeffs.is_empty() {
        enc.bool_txb_skip_tx4_ctx(skip_cdf, 1, plane_v, skip_ctx);
        return 0;
    }
    enc.bool_txb_skip_tx4_ctx(skip_cdf, 0, plane_v, skip_ctx);
    let eob_count = coeffs.iter().map(|&(s, _)| s).max().unwrap() + 1;
    encode_eob_4x4(enc, eob_count, 2);
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
        let rc = scan[c] as usize;
        let row = rc >> 2;
        let col = rc & 3;
        let mag = level.unsigned_abs();
        let is_eob = c == last;
        let lf = (row + col) < 1;
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
    encode_intra_modes(enc, mode_idx, has_chroma, true, partition_cdf, false);
    for (i, tu) in tus.iter().enumerate() {
        encode_luma_tu4(enc, tu, skip_cdfs[i], dc_sign_ctxs[i]);
    }
}
