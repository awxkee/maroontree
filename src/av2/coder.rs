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
    Eob256Inter,
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
        EobCdf::Eob256Inter => enc.sym_eob256_inter(s, nsyms),
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
static TOK_LOG2_LUT: std::sync::OnceLock<Vec<f32>> = std::sync::OnceLock::new();
#[inline]
fn tok_log2_lut() -> &'static [f32] {
    TOK_LOG2_LUT.get_or_init(|| {
        let mut v = vec![0.0f32; 32769];
        for (d, slot) in v.iter_mut().enumerate().skip(1) {
            *slot = (32768.0 / d as f32).log2();
        }
        v[0] = v[1];
        v
    })
}

fn tok_cost(icdf: &[u16], s: usize) -> f32 {
    let hi = if s == 0 { 32768i32 } else { icdf[s - 1] as i32 };
    let lo = if s < icdf.len() { icdf[s] as i32 } else { 0 };
    tok_log2_lut()[(hi - lo).max(1) as usize]
}

#[inline]
fn rice_tail_bits(hr: u32) -> f32 {
    const LUT_LEN: usize = 1 << 16;
    static LUT: std::sync::OnceLock<Box<[f32; LUT_LEN]>> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| {
        Box::new(std::array::from_fn(|hr| {
            2.0 * ((hr + 1) as f32).log2() + 2.0
        }))
    });
    lut.get(hr as usize)
        .copied()
        .unwrap_or_else(|| 2.0 * ((hr + 1) as f32).log2() + 2.0)
}

struct LumaRateTable {
    base_lf: [[f32; 6]; 33],
    eob_lf: [[f32; 5]; 4],
    base_hf: [[f32; 4]; 20],
    eob_hf: [[f32; 3]; 4],
    br_lf: [[f32; 4]; 14],
    br_hf: [[f32; 4]; 7],
}

impl LumaRateTable {
    fn new(qc: usize) -> Self {
        Self {
            base_lf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&LUMA32_BASE_TOK_LF_QC[qc][ctx], symbol))
            }),
            eob_lf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&LUMA32_EOB_TOK_LF_QC[qc][ctx], symbol))
            }),
            base_hf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&LUMA32_BASE_TOK_HF_QC[qc][ctx], symbol))
            }),
            eob_hf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&LUMA32_EOB_TOK_HF_QC[qc][ctx], symbol))
            }),
            br_lf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&BR_TOK_QC[qc][ctx], symbol))
            }),
            br_hf: std::array::from_fn(|ctx| {
                std::array::from_fn(|symbol| tok_cost(&BR_TOK_HF_QC[qc][ctx], symbol))
            }),
        }
    }
}

#[inline]
fn luma_rate_table(qc: usize) -> &'static LumaRateTable {
    static TABLES: std::sync::OnceLock<[LumaRateTable; 4]> = std::sync::OnceLock::new();
    &TABLES.get_or_init(|| std::array::from_fn(LumaRateTable::new))[qc]
}

#[inline]
fn base_range_bits(level: u32, hi_range_ctx: usize, high_freq: bool, rates: &LumaRateTable) -> f32 {
    let limit = if high_freq { 3u32 } else { 5u32 };
    let over = level - limit;
    let costs = if high_freq {
        &rates.br_hf[hi_range_ctx]
    } else {
        &rates.br_lf[hi_range_ctx]
    };
    if over <= 2 {
        costs[over as usize]
    } else {
        costs[3] + rice_tail_bits(level - (limit + 3))
    }
}

/// Estimated bits to code a luma coefficient of magnitude `level` at the given
/// context, matching `encode_luma32_token`. ~1-bit sign for nonzero levels.
#[inline]
fn luma_level_bits(
    level: u32,
    is_eob: bool,
    base_ctx: usize,
    hi_range_ctx: usize,
    high_freq: bool,
    rates: &LumaRateTable,
) -> f32 {
    let mut bits = if !high_freq {
        if is_eob {
            if level <= 4 {
                rates.eob_lf[base_ctx][(level - 1) as usize]
            } else {
                rates.eob_lf[base_ctx][4] + base_range_bits(level, hi_range_ctx, false, rates)
            }
        } else if level <= 4 {
            rates.base_lf[base_ctx][level as usize]
        } else {
            rates.base_lf[base_ctx][5] + base_range_bits(level, hi_range_ctx, false, rates)
        }
    } else if is_eob {
        if level <= 2 {
            rates.eob_hf[base_ctx][(level - 1) as usize]
        } else {
            rates.eob_hf[base_ctx][2] + base_range_bits(level, hi_range_ctx, true, rates)
        }
    } else if level <= 2 {
        rates.base_hf[base_ctx][level as usize]
    } else {
        rates.base_hf[base_ctx][3] + base_range_bits(level, hi_range_ctx, true, rates)
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
    lambda: f32,
) -> f32 {
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
    let rates = luma_rate_table(qc);
    let (th1, th2) = (area / 8, area / 4);
    let mut levels = [0i32; PLVL_BUF];

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
    let mut total_bits = 0.0f32;
    for k in (0..=eob).rev() {
        let is_eob = k == eob;
        let high_freq = k >= LUMA_HI_TO_LOW;
        let a = prm[k];
        let q = lev[k].abs() as u32;
        let (bc, hc) = ctx_at(&levels, k, is_eob);
        let lo = if is_eob { 1u32 } else { 0u32 };
        let hi = q.max(lo);
        let mut best_l = hi;
        let mut best_cost = f32::INFINITY;
        for l in lo..=hi {
            let d = (a - l as f32) * (a - l as f32);
            let r = luma_level_bits(l, is_eob, bc, hc, high_freq, rates);
            let cost = d + lambda * r;
            if cost < best_cost {
                best_cost = cost;
                best_l = l;
            }
        }
        lev[k] = best_l as f32 * lev[k].signum();
        store(&mut levels, k, best_l as i32);
        total_bits += luma_level_bits(best_l, is_eob, bc, hc, high_freq, rates);
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
        let a = prm[p];
        let drop_bits = luma_level_bits(lev[p].abs() as u32, true, bc, hc, high_freq, rates);
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
) -> f32 {
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
fn chroma_base_range_bits(level: u32, hi_range_ctx: usize, is_dc: bool, qc: usize) -> f32 {
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
    lambda: f32,
) -> f32 {
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
    let mut levels = [0i32; PLVL_BUF];

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
    let mut total_bits = 0.0f32;
    for k in (0..=eob).rev() {
        let is_eob = k == eob;
        let is_dc = k == 0;
        let a = prm[k];
        let q = lev[k].abs() as u32;
        let (bc, hc) = ctx_at(&levels, k, is_eob, is_dc);
        let lo = if is_eob { 1u32 } else { 0u32 };
        let hi = q.max(lo);
        let mut best_l = hi;
        let mut best_cost = f32::INFINITY;
        for l in lo..=hi {
            let d = (a - l as f32) * (a - l as f32);
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
        let a = prm[p];
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

struct ReverseCoeffCursor<'a> {
    coeffs: &'a [Coeff],
    next: usize,
}

impl<'a> ReverseCoeffCursor<'a> {
    #[inline]
    fn new(coeffs: &'a [Coeff]) -> Self {
        debug_assert!(coeffs.windows(2).all(|w| w[0].0 < w[1].0));
        Self {
            coeffs,
            next: coeffs.len(),
        }
    }

    #[inline]
    fn level_at(&mut self, scan_pos: usize) -> i32 {
        if self.next == 0 {
            return 0;
        }
        let &(pos, level) = &self.coeffs[self.next - 1];
        debug_assert!(pos <= scan_pos);
        if pos == scan_pos {
            self.next -= 1;
            level
        } else {
            0
        }
    }
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
/// Emit a GLOBALMV inter block with skip_txfm=0 and a SPLIT tx partition (four
/// TX_32X32). `tu_coeffs` are quantized residual coeffs per TU (empty = all-zero).
#[allow(unused_variables)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_inter_residual_block(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    tu_coeffs: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    u_coeffs: &[Coeff],
    v_coeffs: &[Coeff],
    u_skip_ctx: usize,
    v_skip_ctx: usize,
) {
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 0); // not skipped -> residual follows
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    // skip_txfm==0 forces delta_q coding (AVM read_delta_qindex gate) even when the
    // block equals sb_size; emit the SB's signaled delta (0 if AQ off/first-in-SB).
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, 1); // GLOBALMV
    // tx partition: SPLIT (do_partition=1, type=0) -> four TX_32X32.

    enc.emit_tx_do_partition(1);
    enc.emit_tx_part_type(0);

    enc.inter_txb = true;
    // The caller resolves these against the live above/left entropy contexts,
    // including the preceding block at a superblock boundary.
    for i in 0..4 {
        encode_luma_tu32(enc, &tu_coeffs[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    // Chroma U/V (420: one TX_32X32 each), with contexts resolved from
    // the live neighboring coefficient state by the frame controller.
    enc.inter_txb = true;
    encode_chroma_block_ex(enc, u_coeffs, u_skip_ctx as u32, true, false);
    enc.inter_txb = false;
    encode_chroma_block_ex(enc, v_coeffs, v_skip_ctx as u32, false, false);
}

/// Inter-mode signalling and luma residual state shared by the single- and
/// multi-TU chroma NEWMV emitters.
#[derive(Clone, Copy)]
pub(crate) struct InterResidualSpec<'a> {
    pub(crate) part_cdf: u32,
    pub(crate) skip_ctx: usize,
    pub(crate) mode_ctx: usize,
    pub(crate) drl_ctx: usize,
    pub(crate) mode: usize,
    pub(crate) scaled_row: i32,
    pub(crate) scaled_col: i32,
    pub(crate) luma_tus: &'a [Vec<Coeff>; 4],
    pub(crate) luma_skip_cdfs: &'a [u32; 4],
    pub(crate) luma_dc_sign_ctxs: &'a [usize; 4],
}

/// Chroma TU lists and skip contexts for formats with more than one chroma TU.
#[derive(Clone, Copy)]
pub(crate) struct InterChromaTus<'a> {
    pub(crate) u_tus: &'a [Vec<Coeff>],
    pub(crate) v_tus: &'a [Vec<Coeff>],
    pub(crate) u_skip_cdfs: &'a [u32],
    pub(crate) v_skip_cdfs: &'a [u32],
    pub(crate) u_tx64: bool,
}

/// Emit a NEWMV inter block WITH residual: mode=NEWMV, DRL idx=0, MVD, then
/// skip_txfm=0 + 4 luma TX32 + chroma residual (recon = MC pred + inv-DCT).
pub(crate) fn emit_inter_newmv_residual_block(
    enc: &mut RangeEncoder,
    spec: &InterResidualSpec<'_>,
    u_coeffs: &[Coeff],
    v_coeffs: &[Coeff],
    u_skip_ctx: usize,
    v_skip_ctx: usize,
) -> [usize; 4] {
    let InterResidualSpec {
        part_cdf,
        skip_ctx,
        mode_ctx,
        drl_ctx,
        mode,
        scaled_row,
        scaled_col,
        luma_tus: tu_coeffs,
        luma_skip_cdfs: tu_skip_cdfs,
        luma_dc_sign_ctxs: dc_sign_ctxs,
    } = *spec;
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 0); // residual follows
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, mode); // 0=NEARMV, 2=NEWMV
    enc.emit_drl(drl_ctx, 0);
    if mode == 2 {
        crate::av2::video::mvd::encode_mvd_qtr(enc, scaled_row, scaled_col);
        if scaled_row != 0 {
            enc.encode_bypass((scaled_row < 0) as u32, 1);
        }
        if scaled_col != 0 {
            enc.encode_bypass((scaled_col < 0) as u32, 1);
        }
    }
    enc.emit_tx_do_partition(1);
    enc.emit_tx_part_type(0);
    enc.inter_txb = true;
    let mut cul = [0usize; 4];
    for i in 0..4 {
        cul[i] =
            (encode_luma_tu32(enc, &tu_coeffs[i], tu_skip_cdfs[i], dc_sign_ctxs[i]) as usize) & 7;
    }
    enc.inter_txb = true;
    encode_chroma_block_ex(enc, u_coeffs, u_skip_ctx as u32, true, false);
    enc.inter_txb = false;
    encode_chroma_block_ex(enc, v_coeffs, v_skip_ctx as u32, false, false);
    cul
}

/// Emit one 32x32 inter leaf with a single TX32 luma residual and one TX16
/// residual per 4:2:0 chroma plane. This is the dense subblock counterpart to
/// [`emit_inter_newmv_residual_block`]; callers provide already-resolved live
/// skip/DC contexts so candidate analysis stays separate from deterministic emit.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_inter_residual_leaf_32(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    drl_ctx: usize,
    mode: usize,
    scaled_row: i32,
    scaled_col: i32,
    luma: &[Coeff],
    luma_skip_cdf: u32,
    luma_dc_sign_ctx: usize,
    u: &[Coeff],
    v: &[Coeff],
    u_skip_cdf: u32,
    v_skip_ctx: u32,
) -> u32 {
    enc.cur_bw4 = 8;
    enc.cur_bh4 = 8;
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 0);
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, mode);
    enc.emit_drl(drl_ctx, 0);
    if mode == 2 {
        crate::av2::video::mvd::encode_mvd_qtr(enc, scaled_row, scaled_col);
        if scaled_row != 0 {
            enc.encode_bypass((scaled_row < 0) as u32, 1);
        }
        if scaled_col != 0 {
            enc.encode_bypass((scaled_col < 0) as u32, 1);
        }
    }

    // txfm_do_partition_cdf[0][is_inter=1][group(32x32)=5]:
    // AVM_CDF2(22159), stored here as the inverse CDF probability.
    enc.bool_txfm_part(TX_DO_PART_INTER_32X32, 0);
    enc.inter_txb = true;
    let cul = encode_luma_tu32(enc, luma, luma_skip_cdf, luma_dc_sign_ctx);
    encode_chroma_block_rect_w(
        enc,
        u,
        u_skip_cdf,
        true,
        &SCAN16,
        EobCdf::ChrEob256,
        CHROMA_EOB_HI_BIT_QC[enc.qc],
        256,
        4,
    );
    enc.inter_txb = false;
    encode_chroma_block_rect_w(
        enc,
        v,
        v_skip_ctx,
        false,
        &SCAN16,
        EobCdf::ChrEob256,
        CHROMA_EOB_HI_BIT_QC[enc.qc],
        256,
        4,
    );
    cul
}

/// Emit a 16x16 inter leaf with one TX16 DCT luma residual and one TX8 residual
/// on each 4:2:0 chroma plane.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_inter_residual_leaf_16(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    drl_ctx: usize,
    mode: usize,
    scaled_row: i32,
    scaled_col: i32,
    luma: &[Coeff],
    luma_skip_cdf: u32,
    luma_dc_sign_ctx: usize,
    u: &[Coeff],
    v: &[Coeff],
    u_skip_ctx: usize,
    v_skip_ctx: usize,
) -> u32 {
    enc.cur_bw4 = 4;
    enc.cur_bh4 = 4;
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 0);
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, mode);
    enc.emit_drl(drl_ctx, 0);
    if mode == 2 {
        crate::av2::video::mvd::encode_mvd_qtr(enc, scaled_row, scaled_col);
        if scaled_row != 0 {
            enc.encode_bypass((scaled_row < 0) as u32, 1);
        }
        if scaled_col != 0 {
            enc.encode_bypass((scaled_col < 0) as u32, 1);
        }
    }

    enc.bool_txfm_part(TX_DO_PART_INTER_16X16, 0);
    enc.inter_txb = true;
    let cul = encode_luma_tu16_inter(enc, luma, luma_skip_cdf, luma_dc_sign_ctx);
    encode_inter_chroma8(enc, u, u_skip_ctx, true);
    encode_inter_chroma8(enc, v, v_skip_ctx, false);
    enc.inter_txb = false;
    cul
}

fn encode_inter_chroma8(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_ctx: usize,
    is_u_plane: bool,
) {
    let nonzero: Vec<Coeff> = coeffs.iter().copied().filter(|&(_, l)| l != 0).collect();
    let emit_skip = |enc: &mut RangeEncoder, bit| {
        if is_u_plane {
            enc.bool_u_skip8_inter(skip_ctx, bit);
        } else {
            enc.bool_v_skip(skip_ctx, bit);
        }
    };
    if nonzero.is_empty() {
        emit_skip(enc, 1);
        return;
    }
    emit_skip(enc, 0);
    let eob = nonzero.iter().map(|&(scan, _)| scan).max().unwrap();
    encode_eob(
        enc,
        eob,
        EobCdf::ChrEob64,
        CHROMA_EOB_HI_BIT_QC[enc.qc],
        0,
        6,
    );
    let plane_offset = if is_u_plane { 0 } else { 4 };
    let stored = encode_chroma_tokens_scan_w(enc, &nonzero, eob, plane_offset, &SCAN8X8, 64, 3);
    encode_chroma_signs(enc, &stored);
}

/// Emit a NEWMV inter block with an arbitrary number of chroma TUs (for 4:4:4 and
/// 4:2:2, whose chroma planes tile into several 32x32 TUs instead of one).
pub(crate) fn emit_inter_newmv_residual_block_multi(
    enc: &mut RangeEncoder,
    spec: &InterResidualSpec<'_>,
    chroma: &InterChromaTus<'_>,
) -> [usize; 4] {
    let InterResidualSpec {
        part_cdf,
        skip_ctx,
        mode_ctx,
        drl_ctx,
        mode,
        scaled_row,
        scaled_col,
        luma_tus: tu_coeffs,
        luma_skip_cdfs: tu_skip_cdfs,
        luma_dc_sign_ctxs: dc_sign_ctxs,
    } = *spec;
    let InterChromaTus {
        u_tus,
        v_tus,
        u_skip_cdfs: u_skips,
        v_skip_cdfs: v_skips,
        u_tx64,
    } = *chroma;
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 0); // residual follows
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, mode); // 0=NEARMV, 2=NEWMV
    enc.emit_drl(drl_ctx, 0);
    if mode == 2 {
        crate::av2::video::mvd::encode_mvd_qtr(enc, scaled_row, scaled_col);
        if scaled_row != 0 {
            enc.encode_bypass((scaled_row < 0) as u32, 1);
        }
        if scaled_col != 0 {
            enc.encode_bypass((scaled_col < 0) as u32, 1);
        }
    }
    enc.emit_tx_do_partition(1);
    enc.emit_tx_part_type(0);
    enc.inter_txb = true;
    let mut cul = [0usize; 4];
    for i in 0..4 {
        cul[i] =
            (encode_luma_tu32(enc, &tu_coeffs[i], tu_skip_cdfs[i], dc_sign_ctxs[i]) as usize) & 7;
    }
    // Chroma: all U TUs then all V TUs, plane-major (matches the decoder's per-plane
    // TU walk with CCTX disabled).
    enc.inter_txb = true;
    for (tu, &sk) in u_tus.iter().zip(u_skips.iter()) {
        encode_chroma_block_ex(enc, tu, sk, true, u_tx64);
    }
    enc.inter_txb = false;
    for (tu, &sk) in v_tus.iter().zip(v_skips.iter()) {
        encode_chroma_block_ex(enc, tu, sk, false, false);
    }
    cul
}

/// Whole-SB inter SKIP block with an explicit mode (0=NEARMV no MVD, 2=NEWMV).
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_inter_mode_block(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    drl_ctx: usize,
    mode: usize,
    scaled_row: i32,
    scaled_col: i32,
) {
    emit_inter_mode_leaf(
        enc, part_cdf, skip_ctx, mode_ctx, drl_ctx, mode, scaled_row, scaled_col, true,
    );
}

/// Emit a NEARMV/NEWMV skip leaf. `full_superblock` controls the normative
/// delta-Q exception in the same way as `emit_inter_skip_leaf`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_inter_mode_leaf(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    drl_ctx: usize,
    mode: usize,
    scaled_row: i32,
    scaled_col: i32,
    full_superblock: bool,
) {
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 1);
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if full_superblock {
        enc.delta_q_pending = false;
    } else if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, mode);
    // DRL idx=0 (max_drl_bits=1): one drl bit = 0.
    enc.emit_drl(drl_ctx, 0);
    // motion_mode SIMPLE_TRANSLATION (0 bits when motion modes disabled).
    // MVD shell + signs (NEWMV only).
    if mode != 2 {
        return;
    }
    crate::av2::video::mvd::encode_mvd_qtr(enc, scaled_row, scaled_col);
    if scaled_row != 0 {
        enc.encode_bypass((scaled_row < 0) as u32, 1);
    }
    if scaled_col != 0 {
        enc.encode_bypass((scaled_col < 0) as u32, 1);
    }
}

/// av2_get_ref_pred_context at rank granularity. The line-buffer neighbors
/// resolve to the left and above blocks (AVM scans bottom-left → above-right →
/// left → above and keeps the first two, and both pairs land in the same
/// neighbor block at our coding granularities); when only one side is
/// available it fills both slots, so it counts twice. `None` = side
/// unavailable, `Some(None)` = intra neighbor, `Some(Some(rank))` = inter
/// neighbor predicting from `rank`.
pub(crate) fn single_ref_bit_ctx(
    above: Option<Option<usize>>,
    left: Option<Option<usize>>,
) -> usize {
    let mut counts = [0u32; 2];
    let weight = if above.is_some() && left.is_some() {
        1
    } else {
        2
    };
    for rank in [above, left].into_iter().flatten().flatten() {
        counts[rank.min(1)] += weight;
    }
    match counts[0].cmp(&counts[1]) {
        std::cmp::Ordering::Less => 0,
        std::cmp::Ordering::Equal => 1,
        std::cmp::Ordering::Greater => 2,
    }
}

/// Emit a GLOBALMV zero-motion skip inter block (static region, copies LAST).
/// Order per AVM: do_split(0), intra_inter=1, skip_txfm=1, gdf/cdef/ccso/delta_q,
/// ref(0b), inter_single_mode=GLOBALMV. No residual.
pub(crate) fn emit_inter_skip_block(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
) {
    emit_inter_skip_leaf(enc, part_cdf, skip_ctx, mode_ctx, true);
}

/// Emit a zero-motion GLOBALMV skip leaf. A whole-superblock skip suppresses
/// delta-Q by specification; a partition leaf still carries the SB's pending
/// delta-Q at the first coded leaf.
pub(crate) fn emit_inter_skip_leaf(
    enc: &mut RangeEncoder,
    part_cdf: u32,
    skip_ctx: usize,
    mode_ctx: usize,
    full_superblock: bool,
) {
    enc.bool_do_split(part_cdf, 0);
    enc.emit_intra_inter_val(1);
    enc.emit_skip_txfm(skip_ctx, 1);
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    if full_superblock {
        // AVM read_delta_qindex skips delta_q only for a full-SB skip_txfm block.
        enc.delta_q_pending = false;
    } else if enc.delta_q_present && enc.delta_q_pending {
        emit_delta_q(enc, enc.delta_q_signaled);
        enc.delta_q_pending = false;
    }
    // tip 0b, reference_mode SINGLE 0b; read_single_ref codes one bit when the
    // frame lists two references (none for n_refs=1).
    enc.emit_single_ref_rank();
    enc.emit_inter_single_mode(mode_ctx, 1); // GLOBALMV
    // motion_mode SIMPLE_TRANSLATION 0b; interp_filter non-switchable 0b.
}

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

/// Emit the per-CDEF-unit strength index for this superblock, once, at its first
/// coded block (mirrors AVM `read_cdef`). Only active for per-block CDEF
/// (`cdef_nb >= 2`); with `cdef_on_skip_txfm_frame_enable = 1` the read fires at
/// the SB's first block regardless of skip, so the deferral matches CCSO exactly.
/// nb == 2, so a single `is_index0` symbol fully selects off (index 0) vs the one
/// active strength (index 1). Context: at the SB top boundary the above CDEF unit
/// is a different SB and thus unavailable, leaving only the left unit — so
/// col 0 => ctx 0; else left-is-index0 ? 2 : 0 (`av2_get_cdef_context`).
pub(crate) fn maybe_emit_cdef(enc: &mut RangeEncoder) {
    if enc.cdef_pending && enc.cdef_nb >= 2 {
        let (r, c) = enc.cdef_sb_rc;
        let cols = enc.cdef_cols;
        let idx = r * cols + c;
        let this_idx0 = enc.cdef_grid.get(idx).copied().unwrap_or(0) == 0;
        let ctx = if c == 0 {
            0
        } else {
            let left_idx0 = enc.cdef_grid.get(idx - 1).copied().unwrap_or(0) == 0;
            if left_idx0 { 2 } else { 0 }
        };
        enc.sym_cdef(ctx, this_idx0 as usize);
        enc.cdef_pending = false;
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
    // Mirrors AVM get_y_intra_mode_set (reconintra.c). The decoder works in the
    // joint-midx space (directional value + NON_DIRECTIONAL_MODES_COUNT); we build the
    // directional-only list in target space (0..55) since the +5 offset is constant and
    // the caller searches for `target`. Neighbor order is [bottom_left, above_right] =
    // [left, above]; NO_MIDX marks a non-directional (or unavailable) neighbor.
    const NDMC: i32 = 5; // NON_DIRECTIONAL_MODES_COUNT
    let small = bw4 * bh4 <= 2; // BLOCK < 8x8
    if small {
        return DEFAULT_MODE_LIST_Y.to_vec();
    }
    // neighbor joint modes in target space; NO_MIDX -> treat as non-directional.
    let mut nb = [lmidx as i32, amidx as i32]; // [left(bottom_left), above(above_right)]
    let is_left_dir = lmidx != NO_MIDX;
    let is_above_dir = amidx != NO_MIDX;
    let mut cnt = is_left_dir as i32 + is_above_dir as i32;
    if cnt == 2 && nb[0] == nb[1] {
        cnt = 1;
    }
    // If only the above neighbor is directional, copy it into slot 0.
    if cnt == 1 && !is_left_dir {
        nb[0] = nb[1];
    }
    if cnt == 0 {
        return DEFAULT_MODE_LIST_Y.to_vec();
    }
    let mut list = [0u8; 56];
    let mut mask = 0u64;
    let mut ptr = 0usize;
    // The directional neighbor modes first.
    for &m in nb[..cnt as usize].iter() {
        let m = m as u8;
        if mask & (1 << m) == 0 {
            list[ptr] = m;
            mask |= 1 << m;
            ptr += 1;
        }
    }
    // Derived neighbor offsets (large block only; area > 64 samples). The decoder does
    // this in JOINT space (target + NDMC), and the mod-56 wrap is not shift-invariant, so
    // we must compute in joint space and convert back to target. i outer (0..4), neighbor
    // inner, left-derived then right-derived, matching the decoder exactly.
    let is_large = bw4 * bh4 * 16 > 64; // (bw4*4)*(bh4*4) samples > 64
    if is_large {
        for i in 0..4i32 {
            for &nb in nb[..cnt as usize].iter() {
                let cj = nb + NDMC; // neighbor in joint space
                let left = (((cj - i + (56 - NDMC - 1)) % 56 + NDMC) - NDMC) as u8;
                let right = (((cj + i - (NDMC - 1)) % 56 + NDMC) - NDMC) as u8;
                for dm in [left, right] {
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
    enc.emit_intra_inter(); // inter tile: intra_inter=0 before mode-info
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
    maybe_emit_cdef(enc);
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
        let (bl, ba) = (lmidx, amidx);
        let list = build_dir_list_y(bw4, bh4, bl, ba);
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
        // Chroma uv-mode index into the decoder's reordered list. With directional
        // co-located luma (uv_ctx = 1) the list is [luma_mode, DC, SMOOTH, SMOOTH_V,
        // SMOOTH_H, PAETH, default_mode_list_uv minus the co-located mode], so:
        //   uv == luma direction      -> index 0 (decoder also copies the luma
        //                                angle_delta; callers must only pick this
        //                                when angle_delta == 0)
        //   uv non-directional (0..4) -> 1 + uv
        //   uv directional            -> 6 + tail position, skipping the removed
        //                                co-located entry.
        // Internal chroma modes 5.. follow default_mode_list_uv order (V,H,D45,D135,
        // D67,D113,D157,D203), so tail position p = uv - 5.
        let uv_ctx = (midx != NO_MIDX) as usize;
        let uv_idx = if uv_ctx == 0 {
            enc.uv_mode
        } else {
            // Co-located luma internal mode (5..=12) -> its default_mode_list_uv position.
            let luma_p = match mode_idx {
                5 => 0,  // V
                6 => 1,  // H
                7 => 2,  // D45
                8 => 3,  // D135
                9 => 5,  // D113
                10 => 6, // D157
                11 => 7, // D203
                _ => 4,  // D67
            };
            if enc.uv_mode < 5 {
                1 + enc.uv_mode
            } else {
                let p = enc.uv_mode - 5;
                if p == luma_p {
                    0
                } else {
                    6 + p - (luma_p < p) as usize
                }
            }
        };
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
    y_ctx: usize,
) {
    encode_intra_modes_with_dpcm(
        enc,
        mode_idx,
        has_chroma,
        lossless,
        partition_cdf,
        y_ctx,
        LosslessDpcm::default(),
    );
}

fn encode_intra_modes_with_dpcm(
    enc: &mut RangeEncoder,
    mode_idx: usize,
    has_chroma: bool,
    lossless: bool,
    partition_cdf: Option<u32>,
    y_ctx: usize,
    dpcm: LosslessDpcm,
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
    enc.emit_intra_inter(); // inter tile: intra_inter=0 before mode-info
    maybe_emit_cdef(enc);
    maybe_emit_ccso(enc);
    maybe_emit_delta_q(enc);
    let use_dpcm_y = lossless && dpcm.y.is_some();
    if lossless {
        // Lossless intra reads use_dpcm_y before the normal luma mode. When set,
        // dpcm_mode_y replaces the regular mode syntax (0=vertical, 1=horizontal).
        enc.encode_bool(16384, use_dpcm_y as u32);
    }
    if let Some(mode) = dpcm.y {
        debug_assert!(lossless);
        enc.encode_bool(16384, mode.bit());
    } else {
        enc.sym_y_set(0); // intra_y mode set 0
        // y_mode_idx context = count of directional bottom-left / above-right
        // neighbors (decoder get_y_mode_idx_ctx).
        enc.sym_y_idx0(y_ctx, mode_idx, 7);
    }
    if has_chroma {
        let use_dpcm_uv = lossless && dpcm.uv.is_some();
        if lossless {
            // Chroma has an independent DPCM decision shared by U and V.
            enc.encode_bool(16384, use_dpcm_uv as u32);
        }
        if let Some(mode) = dpcm.uv {
            debug_assert!(lossless);
            enc.encode_bool(16384, mode.bit());
            return;
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
        if dpcm.y.is_some() {
            // A DPCM luma mode is directional. The reordered chroma mode list
            // places the co-located V/H mode first and DC at index 1.
            debug_assert_eq!(enc.uv_mode, 0);
            emit_uv_mode_idx(enc, 1, 1);
        } else {
            // Co-located luma is non-directional, so the internal numbering
            // (0=DC, 1=SMOOTH, ..., 4=PAETH) is also the list index.
            emit_uv_mode_idx(enc, 0, enc.uv_mode);
        }
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
        let max_base_range = if high_freq { 6 } else { 8 };
        if x == 0 && y == 0 {
            enc.bool_dc_sign(DC_SIGN_QC[enc.qc][dc_sign_ctx] as u32, sign);
        } else {
            enc.encode_bypass(sign, 1);
        }
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
    let mut levels = [0u8; PLVL_BUF];
    let mut stored = Vec::with_capacity(coeffs.len());
    let mut coeff_cursor = ReverseCoeffCursor::new(coeffs);
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = coeff_cursor.level_at(scan_pos);
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
        let sl = encode_chroma4_token(enc, mag, is_eob, base_ctx, hi_ctx, lf);
        levels[pidx_w(rc, bwl)] = sl as u8;
        if level != 0 {
            stored.push((level, !lf));
        }
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
        let max_base_range = if high_freq { 6u32 } else { 5u32 };
        enc.encode_bypass(if level < 0 { 1 } else { 0 }, 1);
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
    let mut levels = [0i32; PLVL_BUF];
    let mut stored = Vec::with_capacity(coeffs.len());
    let mut coeff_cursor = ReverseCoeffCursor::new(coeffs);
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = coeff_cursor.level_at(scan_pos);
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
        if level != 0 {
            stored.push((rc, x, y, level, high_freq));
        }
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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

/// Encode one TX16 DCT luma residual for an inter block.
fn encode_luma_tu16_inter(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    skip_cdf: u32,
    dc_sign_ctx: usize,
) -> u32 {
    let nonzero: Vec<Coeff> = coeffs.iter().copied().filter(|&(_, l)| l != 0).collect();
    if nonzero.is_empty() {
        enc.bool_txb_skip(skip_cdf, 1);
        return 0;
    }
    enc.bool_txb_skip(skip_cdf, 0);
    let eob = nonzero.iter().map(|&(s, _)| s).max().unwrap();
    encode_eob(enc, eob, EobCdf::Eob256Inter, EOB_HI_BIT_QC[enc.qc], 1, 7);
    let eoby = eob >> 4;
    let eobx = eob - (eoby << 4);
    let diag = eobx + eoby;
    let eob_ctx = if diag < 2 {
        1
    } else if diag > 28 {
        2
    } else {
        0
    };
    enc.emit_inter_tx16_dct(eob_ctx);
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
    let mut levels = [0i32; PLVL_BUF];
    let mut stored = Vec::with_capacity(coeffs.len());
    let mut coeff_cursor = ReverseCoeffCursor::new(coeffs);
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = coeff_cursor.level_at(scan_pos);
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
        if level != 0 {
            stored.push((rc, x, y, level, high_freq));
        }
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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
        && let Some((_cdf, idx, nsym)) = tx_type_cdf
    {
        // Adaptive intra ext-tx cdf for TX_8X8 (decoder adapts per block); the
        // static `_cdf` (TXTP_EXT8) is only the initial state, kept for the
        // non-adaptive fallback inside `sym_intra_ext_tx8`.
        enc.sym_intra_ext_tx8(idx, nsym);
    }
    let stored = encode_luma8_tokens_scan_w(enc, &nonzero, eob, &SCAN8X8, 64, 3);
    encode_luma_signs(enc, &nonzero, &stored, dc_sign_ctx);
    nonzero
        .iter()
        .map(|&(_, l)| l.unsigned_abs())
        .sum::<u32>()
        .min(63)
}

/// Signalling, geometry and scan state for a single rectangular 128-coefficient
/// luma leaf.
#[derive(Clone, Copy)]
pub(crate) struct LumaLeafRect128Spec {
    pub(crate) skip_cdf: u32,
    pub(crate) dc_sign_ctx: usize,
    pub(crate) mode_idx: usize,
    pub(crate) has_chroma: bool,
    pub(crate) width_mi: usize,
    pub(crate) height_mi: usize,
    pub(crate) part_cdf: u32,
    pub(crate) tx_part_cdf: u32,
    pub(crate) scan: &'static [u16],
    pub(crate) tx_type_cdf: Option<(&'static [u16], usize, usize)>,
}

pub(crate) fn encode_luma_leaf_rect128(
    enc: &mut RangeEncoder,
    tu: &[Coeff],
    spec: &LumaLeafRect128Spec,
) -> u32 {
    let LumaLeafRect128Spec {
        skip_cdf,
        dc_sign_ctx,
        mode_idx,
        has_chroma,
        width_mi: bw4,
        height_mi: bh4,
        part_cdf,
        tx_part_cdf: do_part_cdf,
        scan,
        tx_type_cdf,
    } = *spec;
    enc.cur_bw4 = bw4;
    enc.cur_bh4 = bh4;
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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
        && let Some((_cdf, idx, nsym)) = tx_type_cdf
    {
        // TX_8X16 / TX_16X8 map (txsize_sqr_map) to the square TX_8X8 ext-tx cdf
        // slot, which the decoder adapts across all three sizes. Code it through
        // the shared adaptive buffer so encoder/decoder cdfs stay in lockstep
        // (a static encode here drifts and desyncs large tiles).
        enc.sym_intra_ext_tx8(idx, nsym);
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
    let mut levels = [0i32; PLVL_BUF];
    let mut stored = Vec::with_capacity(coeffs.len());
    let mut coeff_cursor = ReverseCoeffCursor::new(coeffs);
    let mask = (1 << bwl) - 1;
    for scan_pos in (0..=eob).rev() {
        let level = coeff_cursor.level_at(scan_pos);
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
        if level != 0 {
            stored.push((rc, x, y, level, high_freq));
        }
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
    if enc.inter_txb {
        // inter tx_type (set 3): ctx = get_lp2tx_ctx (diag of last coeff). TX32: bwl=5.
        // Encoder `eob` is the 0-based last scan index (= AVM eob-1), so use it directly.
        let eoby = eob >> 5;
        let eobx = eob - (eoby << 5);
        let diag = eobx + eoby;
        let eob_ctx = if diag < 2 {
            1
        } else if diag > 60 {
            2
        } else {
            0
        };
        enc.emit_inter_tx_type(eob_ctx, 1); // DCT_DCT (idx1)
    }
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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
/// Coefficients, contexts and directional signalling for one 64x64 luma block.
#[derive(Clone, Copy)]
pub(crate) struct LumaSplitDirSpec<'a> {
    pub(crate) tus: &'a [Vec<Coeff>; 4],
    pub(crate) skip_cdfs: &'a [u32; 4],
    pub(crate) dc_sign_ctxs: &'a [usize; 4],
    pub(crate) mode_idx: usize,
    pub(crate) angle_delta: i8,
    pub(crate) has_chroma: bool,
    pub(crate) part_cdf: u32,
    pub(crate) left_midx: u8,
    pub(crate) above_midx: u8,
}

pub(crate) fn encode_luma_block_split_dir(
    enc: &mut RangeEncoder,
    spec: &LumaSplitDirSpec<'_>,
) -> ([u32; 4], u8) {
    let LumaSplitDirSpec {
        tus,
        skip_cdfs,
        dc_sign_ctxs,
        mode_idx,
        angle_delta,
        has_chroma,
        part_cdf,
        left_midx: lmidx,
        above_midx: amidx,
    } = *spec;
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 16;
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_block_vert4(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
    y_ctx: usize,
) -> [u32; 4] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 16;
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        y_ctx,
    );
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

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_luma_block_horz4(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 4],
    skip_cdfs: &[u32; 4],
    dc_sign_ctxs: &[usize; 4],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
    y_ctx: usize,
) -> [u32; 4] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 16;
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        y_ctx,
    );
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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
const TX_DO_PART_INTER_32X32: u32 = 10609;
const TX_DO_PART_INTER_16X16: u32 = 2283;

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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(TX_DO_PART_64X32, 1); // do_partition = 1 (group 6 cdf == 16816)
    enc.sym_tx_part_32x64(1, 6); // type = HORZ-1 = 1
    let mut cul = [0u32; 2];
    for i in 0..2 {
        cul[i] = encode_luma_tu32(enc, &tus[i], skip_cdfs[i], dc_sign_ctxs[i]);
    }
    cul
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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
/// 64x16 leaf with tx-partition VERT: two side-by-side TX_32X16 TUs.
/// avm read_tx_partition: do_partition=1, 4way symbol VERT-1=2 (group 13).
pub(crate) fn encode_luma_leaf_64x16_vert(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 2],
    skip_cdfs: &[u32; 2],
    dc_sign_ctxs: &[usize; 2],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 2] {
    enc.cur_bw4 = 16;
    enc.cur_bh4 = 4;
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(18958, 1);
    enc.sym_tx_part_64x16(2, 6);
    let mut cul = [0u32; 2];
    for i in 0..2 {
        cul[i] = encode_luma_tu_rect_long32(
            enc,
            &tus[i],
            &LumaRectLongSpec {
                skip_cdf: skip_cdfs[i],
                dc_sign_ctx: dc_sign_ctxs[i],
                scan: &SCAN32X16,
                eob_cdf: EobCdf::Eob512,
                eob_hi: EOB_HI_BIT_QC[enc.qc],
                area: 512,
                short_cdf: &[5853, 357, 20],
                ctx2: false,
            },
        );
    }
    cul
}

/// 16x64 leaf with tx-partition HORZ: two stacked TX_16X32 TUs (symbol HORZ-1=1).
pub(crate) fn encode_luma_leaf_16x64_horz(
    enc: &mut RangeEncoder,
    tus: &[Vec<Coeff>; 2],
    skip_cdfs: &[u32; 2],
    dc_sign_ctxs: &[usize; 2],
    mode_idx: usize,
    has_chroma: bool,
    part_cdf: u32,
) -> [u32; 2] {
    enc.cur_bw4 = 4;
    enc.cur_bh4 = 16;
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(18958, 1);
    enc.sym_tx_part_16x64(1, 6);
    let mut cul = [0u32; 2];
    for i in 0..2 {
        cul[i] = encode_luma_tu_rect_long32_w(
            enc,
            &tus[i],
            &LumaRectLongSpec {
                skip_cdf: skip_cdfs[i],
                dc_sign_ctx: dc_sign_ctxs[i],
                scan: &SCAN16X32,
                eob_cdf: EobCdf::Eob512,
                eob_hi: EOB_HI_BIT_QC[enc.qc],
                area: 512,
                short_cdf: &[5853, 357, 20],
                ctx2: false,
            },
            4,
        );
    }
    cul
}

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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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

/// Shared entropy contexts and scan metadata for long-side-32 rectangular TUs.
#[derive(Clone, Copy)]
pub(crate) struct LumaRectLongSpec<'a> {
    pub(crate) skip_cdf: u32,
    pub(crate) dc_sign_ctx: usize,
    pub(crate) scan: &'a [u16],
    pub(crate) eob_cdf: EobCdf,
    pub(crate) eob_hi: u16,
    pub(crate) area: usize,
    pub(crate) short_cdf: &'a [u16; 3],
    pub(crate) ctx2: bool,
}

pub(crate) fn encode_luma_tu_rect_long32(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    spec: &LumaRectLongSpec<'_>,
) -> u32 {
    encode_luma_tu_rect_long32_w(enc, coeffs, spec, 5)
}
pub(crate) fn encode_luma_tu_rect_long32_w(
    enc: &mut RangeEncoder,
    coeffs: &[Coeff],
    spec: &LumaRectLongSpec<'_>,
    bwl: i32,
) -> u32 {
    let LumaRectLongSpec {
        skip_cdf,
        dc_sign_ctx,
        scan,
        eob_cdf,
        eob_hi,
        area,
        short_cdf,
        ctx2,
    } = *spec;
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(19451, 0); // tx_split (szctx 4) = NONE → single TX_16X32
    encode_luma_tu_rect_long32_w(
        enc,
        tu,
        &LumaRectLongSpec {
            skip_cdf,
            dc_sign_ctx,
            scan: &SCAN16X32,
            eob_cdf: EobCdf::Eob512,
            eob_hi: EOB_HI_BIT_QC[enc.qc],
            area: 512,
            short_cdf: &[5853, 357, 20],
            ctx2: false,
        },
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(19451, 0); // tx_split (szctx 4) = NONE → single TX_32X16
    encode_luma_tu_rect_long32(
        enc,
        tu,
        &LumaRectLongSpec {
            skip_cdf,
            dc_sign_ctx,
            scan: &SCAN32X16,
            eob_cdf: EobCdf::Eob512,
            eob_hi: EOB_HI_BIT_QC[enc.qc],
            area: 512,
            short_cdf: &[5853, 357, 20],
            ctx2: false,
        },
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(18958, 0); // tx_split (szctx 8) = NONE → single TX_8X32
    encode_luma_tu_rect_long32_w(
        enc,
        tu,
        &LumaRectLongSpec {
            skip_cdf,
            dc_sign_ctx,
            scan: &SCAN8X32,
            eob_cdf: EobCdf::Eob256,
            eob_hi: EOB_HI_BIT_QC[enc.qc],
            area: 256,
            short_cdf: &[6068, 608, 20],
            ctx2: true,
        },
        3,
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
    enc.bool_txfm_part(18958, 0); // tx_split (szctx 8) = NONE → single TX_32X8
    encode_luma_tu_rect_long32(
        enc,
        tu,
        &LumaRectLongSpec {
            skip_cdf,
            dc_sign_ctx,
            scan: &SCAN32X8,
            eob_cdf: EobCdf::Eob256,
            eob_hi: EOB_HI_BIT_QC[enc.qc],
            area: 256,
            short_cdf: &[6068, 608, 20],
            ctx2: true,
        },
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
    encode_intra_modes(
        enc,
        mode_idx,
        has_chroma,
        false,
        Some(part_cdf),
        false,
        enc.y_ctx,
    );
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

/// Block-level intra-BC syntax state. `allowed` follows the frame flag plus the
/// AV2 block-size restrictions; `selected` means the block is an exact copy of
/// the first default BVP candidate (0, -64), so NEARMV needs no MVD.
#[derive(Clone, Copy, Default)]
pub(crate) struct LosslessIntrabc {
    pub(crate) allowed: bool,
    pub(crate) selected: bool,
    pub(crate) use_ctx: usize,
    pub(crate) skip_ctx: usize,
}

/// Direction selected by the AV2 lossless DPCM syntax. The coded bit follows
/// the specification directly: zero is vertical and one is horizontal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)] // Quarantined until block-wide DPCM residuals are bit-exact.
pub(crate) enum LosslessDpcmMode {
    Vertical,
    Horizontal,
}

impl LosslessDpcmMode {
    #[inline]
    const fn bit(self) -> u32 {
        match self {
            Self::Vertical => 0,
            Self::Horizontal => 1,
        }
    }
}

/// Per-plane-group DPCM choices for one shared-tree lossless block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LosslessDpcm {
    pub(crate) y: Option<LosslessDpcmMode>,
    pub(crate) uv: Option<LosslessDpcmMode>,
}

/// All syntax and residual data for one lossless luma coding block.
#[derive(Clone, Copy)]
pub(crate) struct LosslessLumaBlock<'a> {
    pub(crate) tus: &'a [Vec<Coeff>],
    pub(crate) skip_cdfs: &'a [u32],
    pub(crate) dc_sign_ctxs: &'a [usize],
    pub(crate) mode_idx: usize,
    pub(crate) has_chroma: bool,
    pub(crate) partition_cdf: Option<u32>,
    pub(crate) palette: Option<&'a crate::av2::lossless::LumaPalette>,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) bit_depth: u8,
    pub(crate) intrabc: LosslessIntrabc,
    pub(crate) dpcm: LosslessDpcm,
}

// Inverse AVM_CDF2 defaults used by the static-CDF coded-lossless path.
const USE_INTRABC_CDF: [u32; 3] = [683, 17_596, 28_265];
const SKIP_FLAG_CDF: [u32; 3] = [6_903, 18_452, 28_170];
const INTRABC_MODE_CDF: u32 = 2_775;

/// Assemble one lossless coding block. Ordinary intra blocks use TX_4X4 WHT
/// residuals. Exact intra-BC blocks signal skip + NEARMV with default BVP index
/// zero and therefore carry neither transform coefficients nor an MVD.
pub(crate) fn encode_lossless_luma_sb(enc: &mut RangeEncoder, block: &LosslessLumaBlock<'_>) {
    let LosslessLumaBlock {
        tus,
        skip_cdfs,
        dc_sign_ctxs,
        mode_idx,
        has_chroma,
        partition_cdf,
        palette,
        width: block_width,
        height: block_height,
        bit_depth,
        intrabc,
        dpcm,
    } = *block;

    if let Some(cdf) = partition_cdf {
        enc.bool_do_split(cdf, 0);
    }
    debug_assert!(!intrabc.selected || intrabc.allowed);
    debug_assert!(intrabc.use_ctx < USE_INTRABC_CDF.len());
    debug_assert!(intrabc.skip_ctx < SKIP_FLAG_CDF.len());
    if intrabc.allowed {
        enc.encode_bool(USE_INTRABC_CDF[intrabc.use_ctx], intrabc.selected as u32);
    }
    if intrabc.selected {
        // read_skip(): skip_flag=1. In coded lossless this also suppresses TX
        // size/partition and all plane residual syntax.
        enc.encode_bool(SKIP_FLAG_CDF[intrabc.skip_ctx], 1);
        maybe_emit_cdef(enc);
        maybe_emit_ccso(enc);
        maybe_emit_delta_q(enc);
        // intrabc_mode=1 => NEARMV/no MVD. max_bvp_drl_bits_minus_1 is zero,
        // so exactly one raw DRL bit follows; zero selects RefMvIdx 0, whose
        // selects RefMvIdx 1, whose normative 64x64-SB fallback is (-320, 0):
        // one SB width plus AVM's required 256-pixel IBC delay to the left.
        enc.encode_bool(INTRABC_MODE_CDF, 1);
        enc.encode_bypass(1, 1);
        return;
    }

    encode_intra_modes_with_dpcm(enc, mode_idx, has_chroma, true, None, enc.y_ctx, dpcm);
    // Palette mode info follows the luma/chroma intra modes.  It is present for
    // every palette-eligible DC block once screen-content tools are enabled.
    if dpcm.y.is_none() && mode_idx == 0 && block_width >= 8 && block_height >= 8 {
        enc.sym_palette_y_mode(usize::from(palette.is_some()));
        if let Some(p) = palette {
            enc.sym_palette_y_size(p.colors.len());
            encode_luma_palette_colors(enc, p, bit_depth);
        }
    }
    // This AV2 revision carries luma palettes only (the chroma palette syntax
    // was removed); chroma continues through its regular lossless TUs.
    if let Some(p) = palette {
        encode_luma_palette_map(enc, p);
    }
    for (i, tu) in tus.iter().enumerate() {
        encode_luma_tu4(enc, tu, skip_cdfs[i], dc_sign_ctxs[i]);
    }
}

fn ceil_log2(v: u32) -> u32 {
    if v <= 1 {
        0
    } else {
        32 - (v - 1).leading_zeros()
    }
}

fn encode_luma_palette_colors(
    enc: &mut RangeEncoder,
    palette: &crate::av2::lossless::LumaPalette,
    bit_depth: u8,
) {
    let colors = &palette.colors;
    enc.encode_bypass(colors[0] as u32, bit_depth as u32);
    let min_bits = bit_depth as u32 - 3;
    let max_delta = colors
        .windows(2)
        .map(|v| (v[1] - v[0]) as u32)
        .max()
        .unwrap();
    let mut bits = ceil_log2(max_delta).max(min_bits);
    enc.encode_bypass(bits - min_bits, 2);
    let mut range = (1u32 << bit_depth) - colors[0] as u32 - 1;
    for pair in colors.windows(2) {
        let delta = (pair[1] - pair[0]) as u32;
        enc.encode_bypass(delta - 1, bits);
        range -= delta;
        bits = bits.min(ceil_log2(range));
    }
}

fn palette_color_ctx(
    map: &[u8],
    stride: usize,
    y: usize,
    x: usize,
    size: usize,
) -> (usize, [u8; 8]) {
    let mut order = [0u8; 8];
    let mut used = [false; 8];
    let mut count = 0usize;
    macro_rules! push {
        ($color:expr) => {{
            let color = $color;
            if !used[color as usize] {
                order[count] = color;
                used[color as usize] = true;
                count += 1;
            }
        }};
    }
    let ctx = if y == 0 {
        push!(map[x - 1]);
        0
    } else if x == 0 {
        push!(map[(y - 1) * stride]);
        0
    } else {
        let left = map[y * stride + x - 1];
        let diag = map[(y - 1) * stride + x - 1];
        let top = map[(y - 1) * stride + x];
        if left == diag && left == top {
            push!(left);
            4
        } else if left == top {
            push!(left);
            push!(diag);
            3
        } else if left == diag {
            push!(left);
            push!(top);
            2
        } else if diag == top {
            push!(top);
            push!(left);
            2
        } else {
            // This AVM palette-simplification revision keeps neighbor order
            // (left, top, top-left); it does not sort left/top here.
            push!(left);
            push!(top);
            push!(diag);
            1
        }
    };
    for color in 0..size as u8 {
        push!(color);
    }
    debug_assert_eq!(count, size);
    (ctx, order)
}

fn encode_luma_palette_map(enc: &mut RangeEncoder, palette: &crate::av2::lossless::LumaPalette) {
    let (w, h) = (palette.width, palette.height);
    let size = palette.colors.len();
    for y in 0..h {
        // A zero line-copy symbol means every color index in this row is coded.
        enc.sym_palette_identity_row_off(y == 0);
        for x in 0..w {
            let color = palette.map[y * w + x];
            if y == 0 && x == 0 {
                enc.encode_uniform(color as u32, size as u32);
            } else {
                let (ctx, order) = palette_color_ctx(&palette.map, w, y, x, size);
                let symbol = order[..size].iter().position(|&v| v == color).unwrap();
                enc.sym_palette_y_color(size, ctx, symbol);
            }
        }
    }
}
