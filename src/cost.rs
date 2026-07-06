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
fn coef_rate_bits_slow(level: u32) -> f32 {
    match level {
        0 => 0.9,
        1 => 1.7 + 1.0,
        2 => 2.6 + 1.0,
        _ => {
            let mut b = 3.0 + 1.0;
            let total_br = ((level as i32) - 3).min(COEFF_BASE_RANGE);
            let steps = (total_br / 3 + 1) as f32;
            b += steps * 1.3;
            if level >= 15 {
                let v = level - 15;
                b += 2.0 * ((32 - (v + 1).leading_zeros()) as f32) - 1.0;
            }
            b
        }
    }
}

// Rate LUT for the hot low-magnitude range (bit-identical to coef_rate_bits_slow);
// ≥64 falls back to the closed form.
static COEF_RATE_LUT: [f32; 64] = [
    0.9, 2.7, 3.6, 5.3, 5.3, 5.3, 6.6, 6.6, 6.6, 7.9, 7.9, 7.9, 9.2, 9.2, 9.2, 11.5, 13.5, 13.5,
    15.5, 15.5, 15.5, 15.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 17.5, 19.5, 19.5, 19.5, 19.5,
    19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 19.5, 21.5, 21.5, 21.5, 21.5,
    21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5, 21.5,
];

#[inline]
pub(crate) fn coef_rate_bits(level: u32) -> f32 {
    if (level as usize) < 64 {
        COEF_RATE_LUT[level as usize]
    } else {
        coef_rate_bits_slow(level)
    }
}

/// `lambda0` for the trellis quantizer (R-D tradeoff, in `ac_q^2` units so the
/// behaviour is q-adaptive). Calibrated so the per-coefficient round-down and
/// EOB-trim land on the R-D frontier: meaningfully smaller streams for a
/// negligible PSNR cost, beating the naive "raise q" baseline.
pub(crate) const TRELLIS_LAMBDA0: f64 = 0.05;

/// dav1d DTT4_IDTX_1DDCT set index for ADST_ADST at TX_8X8 intra.
pub(crate) const ADST_ADST_TX8_IDX: usize = 4;
/// Asymmetric ADST set indices at TX_8X8 intra (7-type DTT4_IDTX_1DDCT set:
/// IDTX=0, DCT_DCT=1, V_DCT=2, H_DCT=3, ADST_ADST=4, ADST_DCT=5, DCT_ADST=6).
pub(crate) const ADST_DCT_TX8_IDX: usize = 5;
pub(crate) const DCT_ADST_TX8_IDX: usize = 6;
/// dav1d DTT4_IDTX set index for ADST_ADST at TX_16X16 intra (5-type set).
pub(crate) const ADST_ADST_TX16_IDX: usize = 2;
/// Asymmetric ADST set indices at TX_16X16 intra (5-type DTT4_IDTX set:
/// IDTX=0, DCT_DCT=1, ADST_ADST=2, ADST_DCT=3, DCT_ADST=4).
pub(crate) const ADST_DCT_TX16_IDX: usize = 3;
pub(crate) const DCT_ADST_TX16_IDX: usize = 4;

pub(crate) fn trellis_lambda() -> f64 {
    TRELLIS_LAMBDA0
}

// ---------------------------------------------------------------------------
// libaom rdmult port (av1/encoder/rd.c, BSD-2-Clause — same license family as
// this crate). Reimplemented in Rust from the reference algorithm, not copied
// verbatim. For an all-intra still (AVIF key frame) the relevant path is the
// KF_UPDATE branch of `av1_compute_rd_mult_based_on_qindex`.
//
//   q       = av1_dc_quant_QTX(qindex, 0, bit_depth)   // == our dc_q()
//   rdmult  = q * q
//   rdmult *= def_kf_rd_multiplier(q) = 3.3 + 0.0015*q  // KF branch
//   // SSIMULACRA2 / IQ tuning weight (good-quality, non-realtime):
//   weight  = clamp(((255 - qindex) * 3) / 4, 0, 72) + 128   // 128..200
//   rdmult *= weight / 128
//   // bit-depth normalisation: 8-bit none, 10-bit >>4, 12-bit >>8
//
// libaom's integer RDCOST is
//   RDCOST(rdmult, R, D) = ((rdmult * R + (1<<9)) >> 10) + (D << 4)
// with R in *real* entropy-coded bits (Q9, 1 bit = 512) and D = integer SSE.
// This crate keeps a *proxy* bit estimate and integer SSE, so the integer rdmult
// cannot be used as-is. We expose the libaom rdmult shape (the q-dependence and
// the SSIMULACRA2 weight — the parts that fix decision *consistency* across q)
// and fold the fixed-point/units difference into one calibration constant so
// `cost = SSE + mode_lambda_aom(...) * proxy_bits` stays on the same scale the
// existing thresholds were tuned against.

/// libaom's DC-quant-domain KF rd multiplier: `3.3 + 0.0015*q`.
#[inline]
fn def_kf_rd_multiplier(q: f64) -> f64 {
    3.3 + 0.0015 * q
}

/// libaom's SSIMULACRA2 / IQ tuning weight for good-quality (non-realtime)
/// all-intra: `clamp(((255-qindex)*3)/4, 0, 72) + 128`, range 128..=200, applied
/// as `weight/128`. Biases toward larger transforms at low/mid qindex and ramps
/// to neutral by qindex 159 — libaom tuned this directly on SSIMULACRA2.
#[inline]
pub(crate) fn aom_ssimulacra2_rdmult_weight(qindex: u8) -> f64 {
    let w = (((255i32 - qindex as i32) * 3) / 4).clamp(0, 72) + 128;
    w as f64 / 128.0
}

#[allow(dead_code)]
pub(crate) const AOM_RDMULT_CALIB: f64 = 1.0 / (1 << 4) as f64; // undo libaom's D<<4

#[allow(dead_code)]
#[inline]
pub(crate) fn mode_lambda_aom(dc_q: f64, qindex: u8, bd: u8, tune_ssimulacra2: bool) -> f64 {
    let mut rdmult = dc_q * dc_q * def_kf_rd_multiplier(dc_q);
    if tune_ssimulacra2 {
        rdmult *= aom_ssimulacra2_rdmult_weight(qindex);
    }
    // Bit-depth normalisation (libaom: 10-bit >>4, 12-bit >>8).
    rdmult *= match bd {
        10 => 1.0 / (1 << 4) as f64,
        12 => 1.0 / (1 << 8) as f64,
        _ => 1.0,
    };
    // libaom RDCOST weights rate as rdmult/1024 against D<<4 distortion; our cost
    // compares against raw SSE, so divide by 1024 and by the D<<4 factor.
    rdmult * (1.0 / 1024.0) * AOM_RDMULT_CALIB
}

#[inline]
pub(crate) fn mode_lambda_weight(qindex: u8, tune: bool) -> f64 {
    if tune {
        aom_ssimulacra2_rdmult_weight(qindex)
    } else {
        1.0
    }
}

/// Q22 fixed point (1/2^22 bit units) for every CDF partition `p` in
/// `[1, 32768]`. Built once; replaces a per-call `log2()` (a libm transcendental
/// that dominated the trellis) with a single array load. Q22 keeps the rounding
/// error ~1e-7 bits, far below anything the R-D comparison can resolve, so the
/// chosen levels are identical to the float version.
pub(crate) const COST_Q_FRAC: u32 = 22;
pub(crate) const COST_Q_SCALE_INV: f32 = 1.0 / (1u32 << COST_Q_FRAC) as f32;

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
pub(crate) fn cdf_cost(cdf: &[u16], s: usize) -> f32 {
    let fl = if s > 0 { cdf[s - 1] as i32 } else { 32768 };
    let fh = cdf[s] as i32;
    let p = (fl - fh).max(1) as usize;
    cost_q_table()[p] as f32 * COST_Q_SCALE_INV
}

/// Bypass bits for the Exp-Golomb tail coding `v` (level ≥ 15 carries `v=L-15`).
#[inline]
pub(crate) fn golomb_cost(v: u32) -> f32 {
    let len = 32 - (v + 1).leading_zeros();
    (2 * len - 1) as f32
}

/// Accurate bit cost of the base-range (hi_tok) ladder for magnitude `m` (≥ 3)
/// against `br_cdf`, plus the Exp-Golomb tail when `m ≥ 15`.
pub(crate) fn hi_tok_cost(m: u32, br_cdf: &[u16]) -> f32 {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0i32;
    let mut bits = 0.0f32;
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
pub(crate) fn block_rate_bits(cf: &[i32], scan: &[usize]) -> f32 {
    let mut eob: i32 = -1;
    for (i, &rc) in scan.iter().enumerate() {
        if cf[rc] != 0 {
            eob = i as i32;
        }
    }
    if eob < 0 {
        return 1.0; // all-zero: just the txb_skip flag
    }
    let mut bits = 2.0f32; // eob_pt / skip-flag overhead
    for &rc in scan.iter().take(eob as usize + 1) {
        bits += coef_rate_bits(cf[rc].unsigned_abs());
    }
    bits
}

/// R-D weight for the intra luma mode search (cost = pixel SSE + lambda * proxy
/// Legacy mode-search lambda scale (`ac_q^2` units); retained as the calibration
/// reference for `mode_lambda_q` and its tests.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const MODE_LAMBDA0: f64 = 0.02;

/// Calibration folding libaom's rdmult scale onto this crate's (proxy-bits, SSE)
/// axis, fixed so `mode_lambda_q` matches the legacy `MODE_LAMBDA0*ac_q^2` at the
/// q=128 reference. Only the *shape* changes, not the reference operating point.
pub(crate) const MODE_AOM_CALIB: f64 = 0.009005174719460433;

/// Mode-search lambda with libaom's key-frame rdmult *shape*
/// `dc_q^2 * (3.3 + 0.0015*dc_q)` (av1/encoder/rd.c KF path) instead of the
/// naive `ac_q^2`. The old law grew ~2x too fast at high qindex, over-dropping
/// coefficients where libaom keeps them; this tracks libaom's q-dependence so
/// equal-SSIM decisions match. `dc_q` is the bit-depth-correct DC dequant step.
#[inline]
pub(crate) fn mode_lambda_q(dc_q: f64) -> f64 {
    MODE_AOM_CALIB * dc_q * dc_q * def_kf_rd_multiplier(dc_q)
}

#[inline]
pub(crate) fn rate_cost(lambda: f64, rate: f32) -> f64 {
    lambda * rate as f64
}

#[inline]
pub(crate) fn rd_cost_i64(distortion: i64, lambda: f64, rate: f32) -> f64 {
    distortion as f64 + rate_cost(lambda, rate)
}

/// Rough extra bits to *signal* a non-DC luma mode (DC is the most probable
/// symbol; the others cost a little more). Keeps the search from switching modes
/// for a negligible residual gain.
/// Estimated cost (in bits) of *choosing* a non-DC luma mode: the rare y_mode
/// symbol, the shift of the uv_mode CDF context (chroma still codes DC, but
/// under a less-favourable context), and CDF-adaptation churn. DC is free. This
/// is what makes the search only leave DC for a clear net win.
#[inline]
pub(crate) fn mode_signal_bits(m: usize) -> f32 {
    if m == DC_PRED { 0.0 } else { MODE_SIGNAL_BITS }
}
pub(crate) const MODE_SIGNAL_BITS: f32 = 30.0;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{ac_q, dc_q};

    // libaom av1/encoder/rd.c KF path oracle: q=dc_quant, rdmult=q^2*(3.3+0.0015q).
    fn aom_kf_rdmult(dcq: f64) -> f64 {
        dcq * dcq * (3.3 + 0.0015 * dcq)
    }

    #[test]
    fn mode_lambda_matches_aom_kf_shape() {
        // The lambda must be proportional to libaom's KF rdmult across the whole
        // qindex range (single constant), unlike the old ac_q^2 law.
        let mut ratios = vec![];
        for q in (16u8..=240).step_by(8) {
            let dcq = dc_q(q, 8) as f64;
            let lam = mode_lambda_q(dcq);
            ratios.push(lam / aom_kf_rdmult(dcq));
        }
        let (lo, hi) = (
            ratios.iter().cloned().fold(f64::MAX, f64::min),
            ratios.iter().cloned().fold(0.0, f64::max),
        );
        assert!(
            (hi - lo) / hi < 1e-9,
            "lambda not proportional to aom rdmult"
        );
    }

    #[test]
    fn reference_q128_preserved() {
        // Calibration keeps the q=128 operating point equal to the legacy law.
        let acq = ac_q(128, 8) as f64;
        let dcq = dc_q(128, 8) as f64;
        let legacy = MODE_LAMBDA0 * acq * acq;
        assert!((mode_lambda_q(dcq) - legacy).abs() / legacy < 1e-6);
    }

    #[test]
    fn old_law_diverged_from_aom() {
        // Guard: the old ac_q^2 law was NOT proportional to aom (that was the bug).
        let r = |q: u8| {
            let acq = ac_q(q, 8) as f64;
            (MODE_LAMBDA0 * acq * acq) / aom_kf_rdmult(dc_q(q, 8) as f64)
        };
        assert!(
            r(224) / r(32) > 1.5,
            "expected old law to over-weight high q"
        );
    }
}
