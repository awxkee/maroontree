//! Rate / cost estimation used by the mode search and trellis: calibrated
//! per-level token costs, CDF costs, and lambda helpers. Extracted from `av1real`.

use crate::intrapred::DC_PRED;
use crate::tables::{COEFF_BASE_RANGE, EOB_BITW, NUM_BASE_LEVELS};

pub(crate) fn est_block_bits(cf: &[i32], scan: &[usize]) -> u32 {
    let mut eob_idx: i32 = -1;
    for (i, &rc) in scan.iter().enumerate() {
        if cf[rc] != 0 {
            eob_idx = i as i32;
        }
    }
    if eob_idx < 0 {
        return 1;
    }
    let mag: u32 = cf.iter().map(|&c| c.unsigned_abs()).sum();
    mag + EOB_BITW * (eob_idx as u32 + 1)
}

/// Static per-level bit estimate for the AV1 coefficient token structure (base
/// token 0..3, the base-range ladder for levels >= 3, a golomb tail for levels
/// >= 15, plus one sign bit for any nonzero). Used only by the encoder's trellis
/// > quantizer to compare candidate levels — it need not be exact, since only the
/// > *relative* costs drive the decision.
pub(crate) fn coef_rate_bits(level: u32) -> f64 {
    match level {
        0 => 0.9, // a "0" base token coded in the interior run
        1 => 1.7 + 1.0,
        2 => 2.6 + 1.0,
        _ => {
            let mut b = 3.0 + 1.0; // base token == 3 (escape) + sign
            let total_br = ((level as i32) - 3).min(COEFF_BASE_RANGE); // 0..12
            let steps = (total_br / 3 + 1) as f64; // base-range symbols actually coded
            b += steps * 1.3;
            if level >= 15 {
                let v = level - 15;
                b += 2.0 * ((32 - (v + 1).leading_zeros()) as f64) - 1.0; // ~exp-golomb
            }
            b
        }
    }
}

/// `lambda0` for the trellis quantizer (R-D tradeoff, in `ac_q^2` units so the
/// behaviour is q-adaptive). Calibrated so the per-coefficient round-down and
/// EOB-trim land on the R-D frontier: meaningfully smaller streams for a
/// negligible PSNR cost, beating the naive "raise q" baseline.
pub(crate) const TRELLIS_LAMBDA0: f64 = 0.05;

/// dav1d DTT4_IDTX_1DDCT set index for ADST_ADST at TX_8X8 intra.
pub(crate) const ADST_ADST_TX8_IDX: usize = 4;
/// dav1d DTT4_IDTX set index for ADST_ADST at TX_16X16 intra (5-type set).
pub(crate) const ADST_ADST_TX16_IDX: usize = 2;

pub(crate) fn trellis_lambda() -> f64 {
    TRELLIS_LAMBDA0
}

/// Probability -> bit-cost table. `COST_Q[p]` holds `-log2(p / 32768)` in
/// Q22 fixed point (1/2^22 bit units) for every CDF partition `p` in
/// `[1, 32768]`. Built once; replaces a per-call `log2()` (a libm transcendental
/// that dominated the trellis) with a single array load. Q22 keeps the rounding
/// error ~1e-7 bits, far below anything the R-D comparison can resolve, so the
/// chosen levels are identical to the float version.
pub(crate) const COST_Q_FRAC: u32 = 22;
pub(crate) const COST_Q_SCALE_INV: f64 = 1.0 / (1u32 << COST_Q_FRAC) as f64;

pub(crate) fn cost_q_table() -> &'static [u32; 32769] {
    static TABLE: std::sync::OnceLock<Box<[u32; 32769]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u32; 32769]);
        for (p, slot) in t.iter_mut().enumerate().skip(1) {
            let bits = -((p as f64) * (1.0 / 32768.0)).log2();
            *slot = (bits * (1u32 << COST_Q_FRAC) as f64).round() as u32;
        }
        t
    })
}

/// Bits to code symbol `s` against an (inverse-form) CDF: `-log2(p)` where the
/// probability is `(cdf[s-1] - cdf[s]) / 32768` (with `cdf[-1] = 32768`). This
/// matches the MSAC's symbol partition (ignoring the negligible `EC_MIN_PROB`
/// term), so it is the same rate libaom's cost tables approximate. The `-log2`
/// is a precomputed fixed-point table lookup (see [`cost_q_table`]).
#[inline]
pub(crate) fn cdf_cost(cdf: &[u16], s: usize) -> f64 {
    let fl = if s > 0 { cdf[s - 1] as i32 } else { 32768 };
    let fh = cdf[s] as i32;
    let p = (fl - fh).max(1) as usize;
    cost_q_table()[p] as f64 * COST_Q_SCALE_INV
}

/// Bypass bits for the Exp-Golomb tail coding `v` (level ≥ 15 carries `v=L-15`).
#[inline]
pub(crate) fn golomb_cost(v: u32) -> f64 {
    let len = 32 - (v + 1).leading_zeros();
    (2 * len - 1) as f64
}

/// Accurate bit cost of the base-range (hi_tok) ladder for magnitude `m` (≥ 3)
/// against `br_cdf`, plus the Exp-Golomb tail when `m ≥ 15`.
pub(crate) fn hi_tok_cost(m: u32, br_cdf: &[u16]) -> f64 {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0i32;
    let mut bits = 0.0;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        bits += cdf_cost(br_cdf, s as usize);
        coded += s;
        if s < 3 {
            break;
        }
    }
    if m >= 15 {
        bits += golomb_cost(m - 15);
    }
    bits
}

/// Candidate non-directional luma modes evaluated by the mode search, in CDF
/// symbol order (DC first).
/// Estimated coded bits for a quantized block, for the intra mode search. Unlike
/// `est_block_bits` (a partition-time proxy whose EOB term wrongly penalises the
/// many-small-coefficient residuals that good prediction produces), this sums
/// the calibrated per-level token cost over the coded prefix, so it tracks the
/// real entropy cost and ranks predictors correctly.
pub(crate) fn block_rate_bits(cf: &[i32], scan: &[usize]) -> f64 {
    let mut eob: i32 = -1;
    for (i, &rc) in scan.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i as i32;
        }
    }
    if eob < 0 {
        return 1.0; // all-zero: just the txb_skip flag
    }
    let mut bits = 2.0; // eob_pt / skip-flag overhead
    for &rc in scan.iter().take(eob as usize + 1) {
        bits += coef_rate_bits(cf[rc].unsigned_abs());
    }
    bits
}

/// R-D weight for the intra luma mode search (cost = pixel SSE + lambda * proxy
/// bits, with `lambda = MODE_LAMBDA0 * ac_q^2` so it tracks the quantizer).
pub(crate) const MODE_LAMBDA0: f64 = 0.02;
#[inline]
pub(crate) fn mode_lambda() -> f64 {
    MODE_LAMBDA0
}
/// Rough extra bits to *signal* a non-DC luma mode (DC is the most probable
/// symbol; the others cost a little more). Keeps the search from switching modes
/// for a negligible residual gain.
/// Estimated cost (in bits) of *choosing* a non-DC luma mode: the rare y_mode
/// symbol, the shift of the uv_mode CDF context (chroma still codes DC, but
/// under a less-favourable context), and CDF-adaptation churn. DC is free. This
/// is what makes the search only leave DC for a clear net win.
#[inline]
pub(crate) fn mode_signal_bits(m: usize) -> f64 {
    if m == DC_PRED { 0.0 } else { MODE_SIGNAL_BITS }
}
pub(crate) const MODE_SIGNAL_BITS: f64 = 30.0;
