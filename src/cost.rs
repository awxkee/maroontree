//! Rate / cost estimation used by the mode search and trellis: calibrated
//! per-level token costs, CDF costs, and lambda helpers. Extracted from `av1real`.

use crate::aq_common::f_fmlaf;
use crate::tables::{COEFF_BASE_RANGE, EOB_BITW, NUM_BASE_LEVELS};

#[allow(dead_code)]
pub(crate) fn est_block_bits(cf: &[i32], scan: &[u32]) -> u32 {
    let Some(eob) = scan.iter().rposition(|&rc| cf[rc as usize] != 0) else {
        return 1;
    };
    let mag: u32 = scan[..=eob]
        .iter()
        .map(|&rc| cf[rc as usize].unsigned_abs())
        .sum();
    mag + EOB_BITW * (eob as u32 + 1)
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
                b += f_fmlaf(2.0, (32 - (v + 1).leading_zeros()) as f32, -1.0);
            }
            b
        }
    }
}

// Rate LUT for the hot low-magnitude range (bit-identical to coef_rate_bits_slow);
// ≥64 falls back to the closed form.
pub(crate) static COEF_RATE_LUT: [f32; 64] = [
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
/// behavior is q-adaptive). Calibrated so the per-coefficient round-down and
/// EOB-trim land on the R-D frontier: meaningfully smaller streams for a
/// negligible PSNR cost, beating the naive "raise q" baseline.
pub(crate) const TRELLIS_LAMBDA0: f32 = 0.05;

/// dav1d DTT4_IDTX_1DDCT set index for ADST_ADST at TX_8X8 intra.
pub(crate) const ADST_ADST_TX8_IDX: usize = 4;
/// Asymmetric ADST set indices at TX_8X8 intra (7-type DTT4_IDTX_1DDCT set:
/// IDTX=0, DCT_DCT=1, V_DCT=2, H_DCT=3, ADST_ADST=4, ADST_DCT=5, DCT_ADST=6).
pub(crate) const ADST_DCT_TX8_IDX: usize = 5;
pub(crate) const DCT_ADST_TX8_IDX: usize = 6;
/// dav1d DTT4_IDTX set index for IDTX at TX_16X16 intra (5-type set).
pub(crate) const IDTX_TX16_IDX: usize = 0;
/// dav1d DTT4_IDTX set index for ADST_ADST at TX_16X16 intra (5-type set).
pub(crate) const ADST_ADST_TX16_IDX: usize = 2;
/// Asymmetric ADST set indices at TX_16X16 intra (5-type DTT4_IDTX set:
/// IDTX=0, DCT_DCT=1, ADST_ADST=2, ADST_DCT=3, DCT_ADST=4).
pub(crate) const ADST_DCT_TX16_IDX: usize = 3;
pub(crate) const DCT_ADST_TX16_IDX: usize = 4;

pub(crate) fn trellis_lambda() -> f32 {
    TRELLIS_LAMBDA0
}

/// libaom's DC-quant-domain KF rd multiplier: `3.3 + 0.0015*q`.
#[inline]
fn def_kf_rd_multiplier(q: f32) -> f32 {
    3.3 + 0.0015 * q
}

/// libaom's SSIMULACRA2 / IQ tuning weight for good-quality (non-realtime)
/// all-intra: `clamp(((255-qindex)*3)/4, 0, 72) + 128`, range 128..=200, applied
/// as `weight/128`. Biases toward larger transforms at low/mid qindex and ramps
/// to neutral by qindex 159 — libaom tuned this directly on SSIMULACRA2.
#[inline]
pub(crate) fn aom_ssimulacra2_rdmult_weight(qindex: u8) -> f32 {
    let w = (((255i32 - qindex as i32) * 3) / 4).clamp(0, 72) + 128;
    w as f32 * (1. / 128.0)
}

#[inline]
pub(crate) fn mode_lambda_weight(qindex: u8) -> f32 {
    aom_ssimulacra2_rdmult_weight(qindex)
}

/// Q22 fixed point (1/2^22 bit units) for every CDF partition `p` in
/// `[1, 32768]`. Built once; replaces a per-call `log2()` (a libm transcendental
/// that dominated the trellis) with a single array load. Q22 keeps the rounding
/// error ~1e-6 bits, far below anything the R-D comparison can resolve, so the
/// chosen levels are identical to the float version.
pub(crate) const COST_Q_FRAC: u32 = 22;
pub(crate) const COST_Q_SCALE_INV: f32 = 1.0 / (1u32 << COST_Q_FRAC) as f32;

pub(crate) fn cost_q_table() -> &'static [u32; 32769] {
    static TABLE: std::sync::OnceLock<Box<[u32; 32769]>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        let mut t = Box::new([0u32; 32769]);
        for (p, slot) in t.iter_mut().enumerate().skip(1) {
            let bits = -((p as f32) * (1.0 / 32768.0)).log2();
            *slot = (bits * (1u32 << COST_Q_FRAC) as f32).round() as u32;
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
    cdf_cost_with_table(cdf, s, cost_q_table())
}

/// [`cdf_cost`] with the immutable probability-cost table supplied by the
/// caller. Hot coefficient loops use this form so one `OnceLock` state check is
/// paid per block instead of once per coded symbol.
#[inline]
pub(crate) fn cdf_cost_with_table(cdf: &[u16], s: usize, table: &[u32; 32769]) -> f32 {
    let fl = if s > 0 { cdf[s - 1] as i32 } else { 32768 };
    let fh = cdf[s] as i32;
    let p = (fl - fh).max(1) as usize;
    table[p] as f32 * COST_Q_SCALE_INV
}

/// Bypass bits for the Exp-Golomb tail coding `v` (level ≥ 15 carries `v=L-15`).
#[inline]
pub(crate) fn golomb_cost(v: u32) -> f32 {
    let len = 32 - (v + 1).leading_zeros();
    (2 * len - 1) as f32
}

/// Cumulative base-range ladder costs for one br context row: entry `j`
/// is the ladder cost of `total_br == j` (j = (m-3).min(12)); the Golomb
/// tail for m >= 15 is added by the caller. Float-op order matches
/// `hi_tok_cost` term-for-term so cached and direct values are identical. The
/// caller supplies a hoisted probability-cost table.
pub(crate) fn br_cum_row_with_table(br_cdf: &[u16], table: &[u32; 32769]) -> [f32; 13] {
    let c = [
        cdf_cost_with_table(br_cdf, 0, table),
        cdf_cost_with_table(br_cdf, 1, table),
        cdf_cost_with_table(br_cdf, 2, table),
        cdf_cost_with_table(br_cdf, 3, table),
    ];
    // The ladder consumes groups of three. Spell out the four possible prefix
    // lengths to avoid 13 tiny nested loops while preserving the exact
    // left-to-right f32 addition order of `hi_tok_cost`.
    let c3x2 = c[3] + c[3];
    let c3x3 = c3x2 + c[3];
    let c3x4 = c3x3 + c[3];
    [
        c[0],
        c[1],
        c[2],
        c[3] + c[0],
        c[3] + c[1],
        c[3] + c[2],
        c3x2 + c[0],
        c3x2 + c[1],
        c3x2 + c[2],
        c3x3 + c[0],
        c3x3 + c[1],
        c3x3 + c[2],
        c3x4,
    ]
}

/// [`hi_tok_cost`] with a caller-hoisted probability-cost table.
pub(crate) fn hi_tok_cost_with_table(m: u32, br_cdf: &[u16], table: &[u32; 32769]) -> f32 {
    let total_br = (m as i32 - (NUM_BASE_LEVELS + 1)).min(COEFF_BASE_RANGE);
    let mut coded = 0i32;
    let mut bits = 0.0f32;
    for _ in 0..(COEFF_BASE_RANGE / 3) {
        let s = (total_br - coded).min(3);
        bits += cdf_cost_with_table(br_cdf, s as usize, table);
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
pub(crate) fn block_rate_bits(cf: &[i32], scan: &[u32]) -> f32 {
    let Some(eob) = scan.iter().rposition(|&rc| cf[rc as usize] != 0) else {
        return 1.0; // all-zero: just the txb_skip flag
    };

    let mut bits = 2.0f32; // eob_pt / skip-flag overhead
    for &rc32 in &scan[..=eob] {
        bits += coef_rate_bits(cf[rc32 as usize].unsigned_abs());
    }
    bits
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) const MODE_LAMBDA0: f32 = 0.02;

pub(crate) const MODE_AOM_CALIB: f32 = 0.009005174719460433;

#[inline]
pub(crate) fn mode_lambda_q(dc_q: f32) -> f32 {
    MODE_AOM_CALIB * lambda_scale() * dc_q * dc_q * def_kf_rd_multiplier(dc_q)
}

pub(crate) fn use_proxy_rate(speed: crate::avif::Speed) -> bool {
    let t = crate::tuning::get();
    match speed {
        crate::avif::Speed::Fast => t.proxy_rate_fast,
        crate::avif::Speed::Medium => t.proxy_rate_medium,
        crate::avif::Speed::Slow => t.proxy_rate_slow,
    }
}

fn lambda_scale() -> f32 {
    1.0
}

#[inline]
pub(crate) fn rate_cost(lambda: f32, rate: f32) -> f32 {
    lambda * rate
}

#[inline]
pub(crate) fn rd_cost_i64(distortion: i64, lambda: f32, rate: f32) -> f32 {
    distortion as f32 + rate_cost(lambda, rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::{ac_q, dc_q};

    // libaom av1/encoder/rd.c KF path oracle: q=dc_quant, rdmult=q^2*(3.3+0.0015q).
    fn aom_kf_rdmult(dcq: f32) -> f32 {
        dcq * dcq * (3.3 + 0.0015 * dcq)
    }

    #[test]
    fn mode_lambda_matches_aom_kf_shape() {
        // The lambda must be proportional to libaom's KF rdmult across the whole
        // qindex range (single constant), unlike the old ac_q^2 law.
        let mut ratios = vec![];
        for q in (16u8..=240).step_by(8) {
            let dcq = dc_q(q, 8) as f32;
            let lam = mode_lambda_q(dcq);
            ratios.push(lam / aom_kf_rdmult(dcq));
        }
        let (lo, hi) = (
            ratios.iter().cloned().fold(f32::MAX, f32::min),
            ratios.iter().cloned().fold(0.0, f32::max),
        );
        assert!(
            (hi - lo) / hi < 1e-5,
            "lambda not proportional to aom rdmult"
        );
    }

    #[test]
    fn reference_q128_preserved() {
        // Calibration keeps the q=128 operating point equal to the legacy law.
        let acq = ac_q(128, 8) as f32;
        let dcq = dc_q(128, 8) as f32;
        let legacy = MODE_LAMBDA0 * acq * acq;
        assert!((mode_lambda_q(dcq) - legacy).abs() / legacy < 1e-6);
    }

    #[test]
    fn old_law_diverged_from_aom() {
        // Guard: the old ac_q^2 law was NOT proportional to aom (that was the bug).
        let r = |q: u8| {
            let acq = ac_q(q, 8) as f32;
            (MODE_LAMBDA0 * acq * acq) / aom_kf_rdmult(dc_q(q, 8) as f32)
        };
        assert!(
            r(224) / r(32) > 1.5,
            "expected old law to over-weight high q"
        );
    }
}
